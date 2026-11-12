"""Testy jednostkowe modułu osm_infrastructure.

Testy nie łączą się z OpenStreetMap/Overpass. Dane wejściowe są małymi,
syntetycznymi GeoDataFrame'ami, a pobieranie OSM jest mockowane.
"""
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

import osm_infrastructure as osm
from tram_data import DayData


@pytest.fixture
def raw_osm_gdf():
    """Przykładowe obiekty OSM w Krakowie (współrzędne geograficzne)."""
    return gpd.GeoDataFrame(
        {
            "osmid": [1, 2, 3, 4, 5],
            "highway": [
                "traffic_signals",
                "crossing",
                "crossing",
                "bus_stop",
                "traffic_signals",
            ],
            "crossing": [None, "traffic_signals", "uncontrolled", None, None],
        },
        geometry=[
            Point(19.94, 50.06),
            Point(19.941, 50.06),
            Point(19.942, 50.06),
            Point(19.943, 50.06),
            LineString([(19.944, 50.06), (19.945, 50.06)]),
        ],
        crs="EPSG:4326",
    )


def metric_gdf(rows):
    """Buduje GeoDataFrame w metrycznym CRS EPSG:2180.

    rows: lista krotek (kind, x, y), współrzędne w metrach.
    """
    return gpd.GeoDataFrame(
        {"kind": [r[0] for r in rows]},
        geometry=[Point(r[1], r[2]) for r in rows],
        crs=osm.CRS_METRIC,
    )


def test_classify_assigns_correct_kinds(raw_osm_gdf):
    # fetch_infrastructure odpowiada za odrzucenie geometrii innych niż Point;
    # classify odpowiada za przypisanie rodzaju na podstawie tagów OSM.
    points = raw_osm_gdf.loc[raw_osm_gdf.geometry.geom_type == "Point"].copy()
    result = osm.classify(points)

    assert result.crs.to_epsg() == 2180
    assert set(result["kind"]) == {"signal", "ped_signal", "crossing"}
    assert len(result) == 3
    assert result.geometry.geom_type.eq("Point").all()


def test_classify_handles_missing_optional_columns():
    source = gpd.GeoDataFrame(
        {"highway": ["traffic_signals", "crossing", "service"]},
        geometry=[Point(19.94, 50.06), Point(19.941, 50.06), Point(19.942, 50.06)],
        crs="EPSG:4326",
    )

    result = osm.classify(source)

    assert result["kind"].tolist() == ["signal", "crossing"]


def test_classify_returns_empty_geodataframe_when_nothing_matches():
    source = gpd.GeoDataFrame(
        {"highway": ["bus_stop"]},
        geometry=[Point(19.94, 50.06)],
        crs="EPSG:4326",
    )

    result = osm.classify(source)

    assert result.empty
    assert "kind" in result.columns
    assert result.crs.to_epsg() == 2180


def test_dedupe_merges_nearby_points_only_within_same_kind():
    source = metric_gdf(
        [
            ("signal", 100.0, 100.0),
            ("signal", 105.0, 100.0),  # ten metrów od poprzedniego? bufor 7.5 m łączy
            ("signal", 200.0, 100.0),  # osobny obiekt
            ("crossing", 100.0, 100.0),  # inny rodzaj nie może zniknąć
        ]
    )

    result = osm.dedupe(source, radius_m=15.0)

    assert len(result) == 3
    assert result["kind"].value_counts().to_dict() == {"signal": 2, "crossing": 1}
    assert result.crs.to_epsg() == 2180
    assert result.geometry.geom_type.eq("Point").all()


def test_dedupe_handles_empty_input():
    source = gpd.GeoDataFrame(
        {"kind": pd.Series(dtype="object")},
        geometry=gpd.GeoSeries([], crs=osm.CRS_METRIC),
        crs=osm.CRS_METRIC,
    )

    result = osm.dedupe(source)

    assert result.empty
    assert list(result.columns) == ["kind", "geometry"]
    assert result.crs.to_epsg() == 2180


def test_resolve_overlaps_applies_priority_signal_then_ped_signal():
    source = metric_gdf(
        [
            ("signal", 0.0, 0.0),
            ("ped_signal", 10.0, 0.0),  # usunięte przez signal
            ("crossing", 20.0, 0.0),  # usunięte przez signal
            ("ped_signal", 100.0, 0.0),
            ("crossing", 110.0, 0.0),  # usunięte przez ped_signal
            ("crossing", 200.0, 0.0),  # pozostaje
        ]
    )

    result = osm.resolve_overlaps(source, radius_m=30.0)

    assert len(result) == 3
    assert result["kind"].value_counts().to_dict() == {
        "signal": 1,
        "ped_signal": 1,
        "crossing": 1,
    }
    assert result.crs.to_epsg() == 2180


def test_prepare_infrastructure_runs_the_full_pipeline(monkeypatch, raw_osm_gdf):
    # classify: sygnalizacja i przejścia; dedupe: bez zmian; overlap:
    # ped_signal/crossing blisko sygnalizacji zostaną usunięte.
    points = raw_osm_gdf.loc[raw_osm_gdf.geometry.geom_type == "Point"].copy()
    monkeypatch.setattr(osm, "fetch_infrastructure", lambda **kwargs: points)

    result = osm.prepare_infrastructure(dedupe_m=1.0, overlap_m=150.0)

    assert set(result["kind"]) == {"signal"}
    assert result.crs.to_epsg() == 2180


def test_prepare_infrastructure_handles_empty_osm_result(monkeypatch):
    empty = gpd.GeoDataFrame(
        {
            "highway": pd.Series(dtype="object"),
            "crossing": pd.Series(dtype="object"),
        },
        geometry=gpd.GeoSeries([], crs="EPSG:4326"),
        crs="EPSG:4326",
    )
    monkeypatch.setattr(osm, "fetch_infrastructure", lambda **kwargs: empty)

    result = osm.prepare_infrastructure()

    assert result.empty
    assert list(result.columns) == ["kind", "geometry"]
    assert result.crs.to_epsg() == 2180


def test_fetch_infrastructure_uses_cache_when_available(tmp_path, monkeypatch):
    cache = tmp_path / "infrastructure.parquet"
    cache.touch()
    cached = metric_gdf([("signal", 1.0, 2.0)])
    calls = []

    monkeypatch.setattr(osm.gpd, "read_parquet", lambda path: calls.append(path) or cached)

    result = osm.fetch_infrastructure(cache=cache, refresh=False)

    assert result.equals(cached)
    assert calls == [cache]


def test_fetch_infrastructure_downloads_points_and_writes_cache(
    tmp_path, monkeypatch, raw_osm_gdf
):
    cache = tmp_path / "nested" / "infrastructure.parquet"
    calls = {}

    def fake_features_from_place(place, tags):
        calls["place"] = place
        calls["tags"] = tags
        return raw_osm_gdf

    fake_osmnx = SimpleNamespace(features_from_place=fake_features_from_place)
    monkeypatch.setitem(__import__("sys").modules, "osmnx", fake_osmnx)

    def fake_to_parquet(frame, path, *args, **kwargs):
        calls["written_path"] = Path(path)
        calls["written_frame"] = frame.copy()

    monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", fake_to_parquet)

    result = osm.fetch_infrastructure(place="Kraków, Poland", cache=cache, refresh=True)

    assert calls["place"] == "Kraków, Poland"
    assert calls["tags"] == osm.TAGS
    assert calls["written_path"] == cache
    assert cache.parent.exists()
    # Linie OSM są odrzucane, zostają wyłącznie obiekty punktowe.
    assert len(result) == 4
    assert result.geometry.geom_type.eq("Point").all()
    assert set(result.columns) == {"osmid", "highway", "crossing", "geometry"}


def test_build_segments_creates_segments_between_consecutive_stops():
    day = DayData(
        date=pd.Timestamp("2026-10-28"),
        trips=pd.DataFrame(
            {
                "trip_id": ["trip-1", "trip-2"],
                "shape_id": ["shape-1", "shape-1"],
                "route_short_name": ["1", "4"],
            }
        ),
        stop_times=pd.DataFrame(
            {
                "trip_id": ["trip-1", "trip-1", "trip-1", "trip-2", "trip-2", "trip-2"],
                "stop_sequence": [1, 2, 3, 1, 2, 3],
                "stop_id": ["A", "B", "C", "A", "B", "C"],
            }
        ),
        stops=pd.DataFrame(
            {
                "stop_id": ["A", "B", "C"],
                "stop_lon": [19.9400, 19.9410, 19.9420],
                "stop_lat": [50.0600, 50.0600, 50.0600],
            }
        ),
        shapes=pd.DataFrame(
            {
                "shape_id": ["shape-1"] * 3,
                "shape_pt_sequence": [1, 2, 3],
                "shape_pt_lon": [19.9400, 19.9410, 19.9420],
                "shape_pt_lat": [50.0600, 50.0600, 50.0600],
            }
        ),
    )

    result = osm.build_segments(day)

    assert len(result) == 2
    assert result[["stop_from", "stop_to"]].values.tolist() == [["A", "B"], ["B", "C"]]
    assert set(result["routes"]) == {"1,4"}
    assert (result["length_m"] > 0).all()
    assert result.crs.to_epsg() == 2180
    assert result.geometry.geom_type.eq("LineString").all()


def test_count_infrastructure_counts_points_near_segment_and_returns_zeroes():
    segments = gpd.GeoDataFrame(
        {"segment_id": [7], "stop_from": ["A"], "stop_to": ["B"]},
        geometry=[LineString([(0.0, 0.0), (100.0, 0.0)])],
        crs=osm.CRS_METRIC,
    )
    infra = metric_gdf(
        [
            ("signal", 50.0, 5.0),       # w buforze
            ("ped_signal", 50.0, 10.0),  # w buforze
            ("crossing", 50.0, 40.0),    # poza buforem
        ]
    )

    result = osm.count_infrastructure(segments, infra, buffer_m=15.0)

    row = result.iloc[0]
    assert row["signal"] == 1
    assert row["ped_signal"] == 1
    assert row["crossing"] == 0
    assert all(pd.api.types.is_integer_dtype(result[c]) for c in ["signal", "ped_signal", "crossing"])


def test_count_infrastructure_handles_empty_infrastructure():
    segments = gpd.GeoDataFrame(
        {"segment_id": [1]},
        geometry=[LineString([(0.0, 0.0), (100.0, 0.0)])],
        crs=osm.CRS_METRIC,
    )
    infra = gpd.GeoDataFrame(
        {"kind": pd.Series(dtype="object")},
        geometry=gpd.GeoSeries([], crs=osm.CRS_METRIC),
        crs=osm.CRS_METRIC,
    )

    result = osm.count_infrastructure(segments, infra)

    assert result.loc[0, ["signal", "ped_signal", "crossing"]].tolist() == [0, 0, 0]
