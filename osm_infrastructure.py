"""Sygnalizacja świetlna i przejścia dla pieszych (OpenStreetMap) jako atrybuty
odcinków przystanek -> przystanek.

Przepływ:
    infra = prepare_infrastructure()          # OSM (raz, z cache) + duplikaty + światła zamiast przejść
    day   = gtfs_data.load_day("2026-10-28")
    segs  = build_segments(day)               # geometria odcinków z GTFS
    table = count_infrastructure(segs, infra) # ile sygnalizacji / przejść na odcinku
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import union_all
from shapely.geometry import LineString
from shapely.ops import substring

# from typing import TYPE_CHECKING

# if TYPE_CHECKING:           # tylko dla podpowiedzi typów; w repo moduł nazywa się tram_data
#     from tram_data import DayData

from tram_data import DayData

CRS_METRIC = "EPSG:2180"                       # układ metryczny dla Polski (PL-1992)
PLACE = "Kraków, Poland"
TAGS = {"highway": ["traffic_signals", "crossing"]}
CACHE = Path("data/osm/infrastructure_raw.parquet")


# ---------------------------------------------------------------- OSM
def fetch_infrastructure(place: str = PLACE, cache: Path = CACHE,
                         refresh: bool = False) -> gpd.GeoDataFrame:
    """Punkty sygnalizacji i przejść z OSM. Wynik zapisuje do pliku, żeby nie
    odpytywać Overpass przy każdym uruchomieniu."""
    if cache.exists() and not refresh:
        return gpd.read_parquet(cache)
    import osmnx as ox  # import tu, żeby reszta modułu działała bez osmnx
    raw = ox.features_from_place(place, tags=TAGS).reset_index()
    pts = raw[raw.geometry.geom_type == "Point"].copy()
    keep = [c for c in ["osmid", "highway", "crossing", "geometry"] if c in pts]
    cache.parent.mkdir(parents=True, exist_ok=True)
    pts[keep].to_parquet(cache)
    return pts[keep]


def classify(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """kind: 'signal' (skrzyżowanie z sygnalizacją), 'ped_signal' (przejście
    z sygnalizacją), 'crossing' (przejście bez sygnalizacji)."""
    hw = gdf["highway"] if "highway" in gdf else pd.Series(index=gdf.index, dtype=object)
    cr = gdf["crossing"] if "crossing" in gdf else pd.Series(index=gdf.index, dtype=object)
    kind = np.select(
        [hw == "traffic_signals",
         (hw == "crossing") & (cr == "traffic_signals"),
         hw == "crossing"],
        ["signal", "ped_signal", "crossing"], default="")
    out = gdf[kind != ""].copy()
    out["kind"] = kind[kind != ""]
    return out[["kind", "geometry"]].to_crs(CRS_METRIC)


def dedupe(infra: gpd.GeoDataFrame, radius_m: float = 15.0) -> gpd.GeoDataFrame:
    """Skrzyżowanie bywa opisane kilkoma węzłami (po jednym na wlot). Punkty tego
    samego rodzaju bliższe niż radius_m scalamy w jeden.

    Pusty wynik OSM jest poprawnym przypadkiem (np. przy błędach/ograniczeniach
    zapytania lub po filtrowaniu obiektów), dlatego zwracamy pustą tabelę o tym
    samym schemacie zamiast wywoływać pd.concat([]).
    """
    if infra.empty:
        return infra[["kind", "geometry"]].copy().reset_index(drop=True)

    parts = []
    for kind, g in infra.groupby("kind"):
        merged = union_all(g.geometry.buffer(radius_m / 2).values)
        geoms = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
        parts.append(gpd.GeoDataFrame(
            {"kind": kind, "geometry": [p.centroid for p in geoms]}, crs=infra.crs))
    return pd.concat(parts, ignore_index=True)


PRIORITY = ["signal", "ped_signal", "crossing"]   # od najsilniejszego


def resolve_overlaps(infra: gpd.GeoDataFrame, radius_m: float = 30.0) -> gpd.GeoDataFrame:
    """Jeśli przejście stoi przy światłach, liczymy tylko światła (piesi i tak
    czekają na czerwonym). Punkt słabszego rodzaju leżący w promieniu radius_m od
    punktu silniejszego rodzaju jest usuwany: signal > ped_signal > crossing."""
    keep = pd.Series(True, index=infra.index)
    for i, kind in enumerate(PRIORITY[:-1]):
        strong = infra[infra["kind"] == kind]
        weak = infra[infra["kind"].isin(PRIORITY[i + 1:])]
        if strong.empty or weak.empty:
            continue
        zone = gpd.GeoDataFrame(geometry=strong.geometry.buffer(radius_m), crs=infra.crs)
        hit = gpd.sjoin(weak, zone, how="inner", predicate="intersects")
        keep[hit.index.unique()] = False
    return infra[keep].reset_index(drop=True)


def prepare_infrastructure(dedupe_m: float = 15.0, overlap_m: float = 30.0,
                           **kw) -> gpd.GeoDataFrame:
    """OSM -> klasyfikacja -> scalenie duplikatów -> światła zamiast przejść."""
    infra = dedupe(classify(fetch_infrastructure(**kw)), dedupe_m)
    return resolve_overlaps(infra, overlap_m)


# ------------------------------------------------------------ odcinki GTFS
def build_segments(day: DayData) -> gpd.GeoDataFrame:
    """Geometria odcinków przystanek->przystanek: fragment trasy (shapes) między
    kolejnymi przystankami kursu. Odcinek identyfikuje para (stop_from, stop_to),
    wspólna dla wszystkich linii jadących tym torem."""
    stops = gpd.GeoDataFrame(
        day.stops, crs="EPSG:4326",
        geometry=gpd.points_from_xy(day.stops["stop_lon"], day.stops["stop_lat"])
    ).to_crs(CRS_METRIC)
    stop_pt = dict(zip(stops["stop_id"], stops.geometry))

    shp = gpd.GeoDataFrame(
        day.shapes, crs="EPSG:4326",
        geometry=gpd.points_from_xy(day.shapes["shape_pt_lon"], day.shapes["shape_pt_lat"])
    ).to_crs(CRS_METRIC)
    lines = {sid: LineString([(p.x, p.y) for p in g.geometry])
             for sid, g in shp.sort_values("shape_pt_sequence").groupby("shape_id")}

    st = (day.stop_times.merge(day.trips[["trip_id", "shape_id"]], on="trip_id")
          .sort_values(["trip_id", "stop_sequence"]))
    patterns = (st.groupby("trip_id")
                .agg(shape_id=("shape_id", "first"), stops=("stop_id", tuple))
                .drop_duplicates())
    routes_on_shape = (day.trips.groupby("shape_id")["route_short_name"]
                       .agg(lambda s: ",".join(sorted(set(s)))).to_dict())

    segs: dict[tuple[str, str], dict] = {}
    for shape_id, stop_ids in zip(patterns["shape_id"], patterns["stops"]):
        line = lines[shape_id]
        d_prev, prev = 0.0, None
        for sid in stop_ids:
            rest = substring(line, d_prev, line.length)
            d = d_prev + (rest.project(stop_pt[sid]) if rest.length > 0 else 0.0)
            if prev is not None and d - d_prev > 1e-6:
                key = (prev, sid)
                if key not in segs:
                    segs[key] = {"stop_from": prev, "stop_to": sid,
                                 "shape_id": shape_id,
                                 "routes": routes_on_shape.get(shape_id, ""),
                                 "geometry": substring(line, d_prev, d)}
            prev, d_prev = sid, d       # rzutowanie "do przodu": omija pętle

    out = gpd.GeoDataFrame(list(segs.values()), crs=CRS_METRIC)
    out.insert(0, "segment_id", range(len(out)))
    out["length_m"] = out.geometry.length
    return out


def count_infrastructure(segments: gpd.GeoDataFrame, infra: gpd.GeoDataFrame,
                         buffer_m: float = 15.0) -> pd.DataFrame:
    """Liczba punktów każdego rodzaju w pasie buffer_m wokół odcinka. Końce pasa
    są płaskie, więc punkt przy przystanku nie liczy się podwójnie."""
    buf = segments[["segment_id"]].copy()
    buf["geometry"] = segments.geometry.buffer(buffer_m, cap_style="flat")
    buf = gpd.GeoDataFrame(buf, crs=segments.crs)
    j = gpd.sjoin(infra, buf, how="inner", predicate="within")
    cnt = (j.groupby(["segment_id", "kind"]).size().unstack(fill_value=0)
           .reindex(columns=["signal", "ped_signal", "crossing"], fill_value=0))
    out = (segments.drop(columns="geometry")
           .merge(cnt, left_on="segment_id", right_index=True, how="left")
           .fillna({"signal": 0, "ped_signal": 0, "crossing": 0}))
    for c in ["signal", "ped_signal", "crossing"]:
        out[c] = out[c].astype(int)
    return out