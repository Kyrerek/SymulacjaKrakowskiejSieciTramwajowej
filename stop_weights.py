"""Wagi przystanków (względny popyt) z danych GUS i OpenStreetMap.

Wynik: data/stop_weights.json  {stop_id: waga}, średnia waga = 1.  PassengerSystem
wczytuje ten plik automatycznie (StopWeights.from_default), więc nie trzeba nic zmieniać
w main.py. Szczegóły (ludność, punkty przyciągające, waga) trafiają do CSV.

Metoda
  1. Ludność z siatki GUS 1 km (Spis 2021) jest rozdzielana na punkty wewnątrz komórki:
     równomiernie albo, z opcją --buildings, proporcjonalnie do powierzchni zabudowy
     mieszkalnej z OSM (mapowanie dazymetryczne). Każdy punkt trafia do NAJBLIŻSZEGO
     przystanku w promieniu --pop-radius, więc ludność nie jest liczona podwójnie.
  2. Punkty przyciągające z OSM (galerie, uczelnie, szpitale, dworce, szkoły...) mają wagi
     z tabeli POI_WEIGHTS (założenia do dostrojenia) i też trafiają do najbliższego
     przystanku w promieniu --poi-radius.
  3. waga = (1-gamma) * ludność/średnia + gamma * punkty/średnia, ograniczona od dołu
     (--floor) i od góry (--cap, żeby kilka węzłów nie zdominowało sieci przy rzadkich
     danych). Waga przystanku (słupka) = waga węzła / liczba słupków węzła, potem
     normalizacja do średniej 1.

Uruchomienie:
  python stop_weights.py --gus data/gus/siatka.shp [--pop-col NAZWA] [--buildings]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely import contains_xy, box

CRS_METRIC = "EPSG:2180"
PLACE = "Kraków, Poland"

# Waga jednego obiektu OSM jako celu/źródła podróży. "klucz=*" oznacza dowolną wartość.
# To ZAŁOŻENIA, nie dane: wpływ tabeli sprawdzajcie analizą wrażliwości (--gamma, --poi-weights).
POI_WEIGHTS: dict[str, float] = {
    "shop=mall": 10.0, "railway=station": 8.0, "amenity=bus_station": 5.0,
    "amenity=university": 6.0, "amenity=college": 3.0, "amenity=hospital": 6.0,
    "amenity=clinic": 1.0, "amenity=school": 1.5, "shop=supermarket": 1.5,
    "leisure=stadium": 5.0, "amenity=theatre": 2.0, "amenity=cinema": 2.0,
    "amenity=marketplace": 2.0, "tourism=museum": 2.0, "tourism=attraction": 1.0,
    "tourism=hotel": 1.0, "office=*": 0.5,
}
RESIDENTIAL = ["residential", "apartments", "house", "detached", "semidetached_house",
               "terrace", "dormitory", "bungalow"]
_POP_COLUMNS = ("tot", "total", "ogolem", "ogółem", "liczba", "pop", "population",
                "ludnosc", "ludność", "l_ludnosc", "zaludnienie")


# --------------------------------------------------------------- przystanki
def load_stops(gtfs_dir) -> pd.DataFrame:
    """Przystanki (słupki) z GTFS z kolumnami x, y w EPSG:2180 i station_code (węzeł =
    prefiks stop_code, np. '920' z '920-01')."""
    s = pd.read_csv(Path(gtfs_dir) / "stops.txt", dtype=str, encoding="utf-8-sig")
    code = s["stop_code"] if "stop_code" in s else pd.Series(index=s.index, dtype=object)
    s["station_code"] = code.str.split("-").str[0].fillna(s["stop_name"])
    pts = gpd.GeoSeries(gpd.points_from_xy(s["stop_lon"].astype(float), s["stop_lat"].astype(float)),
                        crs="EPSG:4326").to_crs(CRS_METRIC)
    s["x"], s["y"] = pts.x.values, pts.y.values
    return s[["stop_id", "stop_name", "station_code", "x", "y"]]


def stations_from_stops(stops: pd.DataFrame) -> pd.DataFrame:
    return (stops.groupby("station_code")
            .agg(name=("stop_name", "first"), x=("x", "mean"), y=("y", "mean"),
                 n_stops=("stop_id", "size")).reset_index())


# --------------------------------------------------------------------- GUS
def load_gus_grid(path, pop_col: str | None = None) -> gpd.GeoDataFrame:
    """Siatka ludności GUS (SHP/GeoJSON/GPKG) -> kolumna 'pop' + geometria."""
    g = gpd.read_file(path)
    if g.crs is None:
        g = g.set_crs(CRS_METRIC)            # założenie: układ PL-1992
    cols = {c.lower(): c for c in g.columns if c != "geometry"}
    if pop_col:
        if pop_col not in g.columns:
            raise ValueError(f"Brak kolumny {pop_col!r}. Kolumny: {list(g.columns)}")
        col = pop_col
    else:
        col = next((cols[c] for c in _POP_COLUMNS if c in cols), None)
        if col is None:
            numeric = [c for c in g.columns if c != "geometry" and pd.api.types.is_numeric_dtype(g[c])]
            if len(numeric) != 1:
                raise ValueError(f"Nie rozpoznano kolumny z liczbą ludności. Kolumny: "
                                 f"{list(g.columns)}. Podaj --pop-col.")
            col = numeric[0]
    out = g[[col, "geometry"]].rename(columns={col: "pop"})
    out["pop"] = pd.to_numeric(out["pop"], errors="coerce").fillna(0.0).clip(lower=0)
    return out


def clip_to_area(grid: gpd.GeoDataFrame, stations: pd.DataFrame, margin_m: float) -> gpd.GeoDataFrame:
    area = box(stations.x.min() - margin_m, stations.y.min() - margin_m,
               stations.x.max() + margin_m, stations.y.max() + margin_m)
    area = gpd.GeoSeries([area], crs=CRS_METRIC).to_crs(grid.crs).iloc[0]
    return grid[grid.intersects(area)].to_crs(CRS_METRIC).reset_index(drop=True)


def uniform_points(grid_m: gpd.GeoDataFrame, step_m: float = 100.0):
    """Ludność komórki rozłożona równomiernie na regularnych punktach wewnątrz niej."""
    xs, ys, ps = [], [], []
    for geom, pop in zip(grid_m.geometry, grid_m["pop"]):
        if pop <= 0 or geom.is_empty:
            continue
        minx, miny, maxx, maxy = geom.bounds
        nx, ny = max(1, round((maxx - minx) / step_m)), max(1, round((maxy - miny) / step_m))
        gx = minx + (np.arange(nx) + 0.5) * (maxx - minx) / nx
        gy = miny + (np.arange(ny) + 0.5) * (maxy - miny) / ny
        X, Y = (a.ravel() for a in np.meshgrid(gx, gy))
        inside = contains_xy(geom, X, Y)
        if inside.any():
            xs.append(X[inside]); ys.append(Y[inside])
            ps.append(np.full(inside.sum(), pop / inside.sum()))
    if not xs:
        return np.empty((0, 2)), np.empty(0)
    return np.column_stack([np.concatenate(xs), np.concatenate(ys)]), np.concatenate(ps)


def building_points(grid_m: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame, step_m: float = 100.0):
    """Mapowanie dazymetryczne: ludność komórki rozdzielona na budynki mieszkalne
    proporcjonalnie do masy (powierzchnia x kondygnacje). Komórki bez budynków
    mieszkalnych wracają do rozkładu równomiernego."""
    j = gpd.sjoin(buildings[["mass", "geometry"]], grid_m[["pop", "geometry"]],
                  how="inner", predicate="within")
    cell_mass = j.groupby("index_right")["mass"].sum()
    j = j[j["index_right"].map(cell_mass) > 0]
    pop_b = j["pop"] * j["mass"] / j["index_right"].map(cell_mass)
    xy = np.column_stack([j.geometry.x.values, j.geometry.y.values])
    rest = grid_m.drop(index=list(set(j["index_right"])))
    uxy, up = uniform_points(rest, step_m)
    return np.vstack([xy, uxy]), np.concatenate([pop_b.values, up])


# --------------------------------------------------------------------- OSM
def osm_tags(weights: dict[str, float]) -> dict:
    tags: dict = {}
    for spec in weights:
        key, val = spec.split("=", 1)
        if val == "*":
            tags[key] = True
        elif tags.get(key) is not True:
            tags.setdefault(key, []).append(val)
    return tags


def fetch_osm(place: str, tags: dict, cache: Path, refresh: bool = False) -> gpd.GeoDataFrame:
    """Pobiera obiekty OSM (z cache w pliku Parquet, żeby nie odpytywać Overpass wielokrotnie)."""
    cache = Path(cache)
    if cache.exists() and not refresh:
        return gpd.read_parquet(cache)
    import osmnx as ox

    ox.settings.max_query_area_size = 25_000_000

    raw = ox.features_from_place(place, tags=tags).reset_index()
    keep = [c for c in list(tags) + ["building:levels", "geometry"] if c in raw.columns]
    raw = raw[keep].copy()
    for c in keep:
        if c != "geometry":
            raw[c] = raw[c].where(raw[c].isna(), raw[c].astype(str))
    cache.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(cache)
    return raw


def classify_pois(raw: gpd.GeoDataFrame, table: dict[str, float] = POI_WEIGHTS) -> gpd.GeoDataFrame:
    """Waga każdego obiektu = największa waga z pasujących tagów; geometria -> środek."""
    w = np.zeros(len(raw))
    for spec, weight in table.items():
        key, val = spec.split("=", 1)
        if key in raw.columns:
            m = raw[key].notna().values if val == "*" else (raw[key] == val).values
            w = np.where(m, np.maximum(w, weight), w)
    out = raw[w > 0].copy()
    out["weight"] = w[w > 0]
    out = out.to_crs(CRS_METRIC)
    out["geometry"] = out.geometry.centroid
    return out[["weight", "geometry"]].reset_index(drop=True)


def prepare_buildings(raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Budynki mieszkalne -> środek i masa (powierzchnia x liczba kondygnacji)."""
    b = raw[raw["building"].isin(RESIDENTIAL)].to_crs(CRS_METRIC).copy()
    levels = pd.to_numeric(b["building:levels"], errors="coerce") if "building:levels" in b else 1.0
    b["mass"] = b.geometry.area * pd.Series(levels, index=b.index).fillna(1.0).clip(lower=1.0)
    b["geometry"] = b.geometry.centroid
    return b[["mass", "geometry"]].reset_index(drop=True)


# ----------------------------------------------------------------- wagi
def assign_nearest(xy: np.ndarray, values: np.ndarray, stations: pd.DataFrame, radius_m: float):
    """Sumuje `values` punktów w najbliższych węzłach (w promieniu). Zwraca (sumy, suma_przypisana)."""
    n = len(stations)
    if len(xy) == 0:
        return np.zeros(n), 0.0
    dist, idx = cKDTree(stations[["x", "y"]].values).query(xy, distance_upper_bound=radius_m)
    ok = np.isfinite(dist)
    return np.bincount(idx[ok], weights=values[ok], minlength=n), float(values[ok].sum())


def compute_weights(stops: pd.DataFrame, grid: gpd.GeoDataFrame | None = None,
                    pois: gpd.GeoDataFrame | None = None, buildings: gpd.GeoDataFrame | None = None,
                    pop_radius: float = 500.0, poi_radius: float = 300.0, gamma: float = 0.5,
                    step_m: float = 100.0, power: float = 1.0, floor: float = 0.05,
                    cap: float = 15.0):
    """Zwraca (tabela węzłów, tabela przystanków z wagą, informacje diagnostyczne)."""
    st = stations_from_stops(stops)
    info: dict = {}
    st["pop"] = 0.0
    st["poi"] = 0.0
    if grid is not None:
        g = clip_to_area(grid, st, pop_radius + 1500)
        xy, p = building_points(g, buildings, step_m) if buildings is not None else uniform_points(g, step_m)
        st["pop"], assigned = assign_nearest(xy, p, st, pop_radius)
        info["population_in_area"] = float(g["pop"].sum())
        info["population_assigned"] = assigned
    if pois is not None and len(pois):
        xy = np.column_stack([pois.geometry.x.values, pois.geometry.y.values])
        st["poi"], assigned = assign_nearest(xy, pois["weight"].values.astype(float), st, poi_radius)
        info["poi_weight_total"] = float(pois["weight"].sum())
        info["poi_weight_assigned"] = assigned
    has_pop, has_poi = st["pop"].sum() > 0, st["poi"].sum() > 0
    if not (has_pop or has_poi):
        raise ValueError("Brak danych o ludności i punktach przyciągających w zasięgu przystanków")
    gamma = 1.0 if not has_pop else 0.0 if not has_poi else gamma
    pop_n = st["pop"] / st["pop"].mean() if has_pop else 0.0
    poi_n = st["poi"] / st["poi"].mean() if has_poi else 0.0
    raw_score = (1 - gamma) * pop_n + gamma * poi_n
    info["stations_capped"] = int((raw_score > cap).sum())
    score = np.clip(raw_score, floor, cap) ** power
    st["weight_station"] = score / score.mean()
    info["gamma_used"] = gamma

    out = stops.merge(st[["station_code", "n_stops", "weight_station"]], on="station_code")
    out["weight"] = out["weight_station"] / out["n_stops"]     # węzeł dzieli popyt na słupki
    out["weight"] /= out["weight"].mean()
    return st, out[["stop_id", "stop_name", "station_code", "weight"]], info


def main() -> None:
    ap = argparse.ArgumentParser(description="Wagi przystanków z GUS i OSM")
    ap.add_argument("--gtfs", default="data/GTFS_KRK_T")
    ap.add_argument("--gus", help="plik siatki GUS (SHP/GeoJSON/GPKG)")
    ap.add_argument("--pop-col", help="nazwa kolumny z liczbą ludności")
    ap.add_argument("--buildings", action="store_true", help="rozkład ludności wg budynków OSM")
    ap.add_argument("--no-osm", action="store_true", help="bez punktów przyciągających z OSM")
    ap.add_argument("--place", default=PLACE)
    ap.add_argument("--osm-dir", default="data/osm")
    ap.add_argument("--poi-weights", help="JSON {'klucz=wartość': waga} zastępujący POI_WEIGHTS")
    ap.add_argument("--pop-radius", type=float, default=500.0)
    ap.add_argument("--poi-radius", type=float, default=300.0)
    ap.add_argument("--gamma", type=float, default=0.5, help="udział punktów przyciągających")
    ap.add_argument("--cap", type=float, default=15.0, help="górne ograniczenie wagi węzła (średnia = 1)")
    ap.add_argument("--floor", type=float, default=0.05, help="dolne ograniczenie wagi węzła")
    ap.add_argument("--out", default="data/stop_weights.json")
    a = ap.parse_args()

    stops = load_stops(a.gtfs)
    grid = load_gus_grid(a.gus, a.pop_col) if a.gus else None
    osm_dir = Path(a.osm_dir)
    table = json.loads(Path(a.poi_weights).read_text(encoding="utf-8")) if a.poi_weights else POI_WEIGHTS
    pois = None if a.no_osm else classify_pois(fetch_osm(a.place, osm_tags(table), osm_dir / "pois_raw.parquet"), table)
    buildings = None
    if a.buildings:
        raw_b = fetch_osm(a.place, {"building": RESIDENTIAL}, osm_dir / "buildings_raw.parquet")
        buildings = prepare_buildings(raw_b)

    st, weights, info = compute_weights(stops, grid, pois, buildings, a.pop_radius, a.poi_radius,
                                        a.gamma, floor=a.floor, cap=a.cap)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(zip(weights["stop_id"], weights["weight"].round(4))),
                              ensure_ascii=False, indent=1), encoding="utf-8")
    st.to_csv(out.with_name("stop_weights_details.csv"), index=False)
    print(json.dumps(info, indent=1))
    print("najwyższe wagi węzłów:\n", st.sort_values("weight_station", ascending=False)
          .head(15)[["name", "n_stops", "pop", "poi", "weight_station"]].round(2).to_string(index=False))
    print(f"zapisano {len(weights)} wag do {out}")


if __name__ == "__main__":
    main()