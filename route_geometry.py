"""Geometria linii tramwajowych z danych GTFS.

Moduł łączy kształty tras (``shapes.txt``) z uporządkowanymi przystankami
(``stop_times.txt``). Współrzędne są podane jako ``(lat, lon)``; przy
przekazywaniu do GeoJSON/Folium należy odwrócić kolejność na ``[lon, lat]``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import tram_data
from models import Point, Stop, StopRoad


@dataclass(frozen=True)
class GeoPoint:
    lat: float
    lon: float

    def as_latlon(self) -> tuple[float, float]:
        """Format używany m.in. przez folium.PolyLine: (lat, lon)."""
        return self.lat, self.lon

    def as_lonlat(self) -> list[float]:
        """Format współrzędnych GeoJSON: [lon, lat]."""
        return [self.lon, self.lat]


@dataclass(frozen=True)
class RouteStop:
    stop_id: str
    name: str
    code: str
    station_code: str
    sequence: int
    lat: float
    lon: float

    @property
    def point(self) -> GeoPoint:
        return GeoPoint(self.lat, self.lon)


@dataclass(frozen=True)
class RouteGeometry:
    tram_number: str
    direction_id: str
    direction: str
    shape_id: str
    points: tuple[GeoPoint, ...]
    stops: tuple[RouteStop, ...]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Granice w kolejności (min_lat, min_lon, max_lat, max_lon)."""
        coords = [*self.points, *(stop.point for stop in self.stops)]
        if not coords:
            raise ValueError("Trasa nie ma współrzędnych")
        return (
            min(p.lat for p in coords), min(p.lon for p in coords),
            max(p.lat for p in coords), max(p.lon for p in coords),
        )

    @property
    def center(self) -> tuple[float, float]:
        min_lat, min_lon, max_lat, max_lon = self.bounds
        return (min_lat + max_lat) / 2, (min_lon + max_lon) / 2

    def to_geojson(self) -> dict:
        """Zwraca FeatureCollection z linią trasy i punktami przystanków."""
        features = [{
            "type": "Feature",
            "geometry": {"type": "LineString",
                         "coordinates": [p.as_lonlat() for p in self.points]},
            "properties": {"tram_number": self.tram_number,
                           "direction_id": self.direction_id,
                           "direction": self.direction,
                           "shape_id": self.shape_id},
        }]
        for stop in self.stops:
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": stop.point.as_lonlat()},
                "properties": {"stop_id": stop.stop_id, "name": stop.name,
                               "code": stop.code, "station_code": stop.station_code,
                               "sequence": stop.sequence,
                               "tram_number": self.tram_number,
                               "direction_id": self.direction_id},
            })
        return {"type": "FeatureCollection", "features": features}


@dataclass
class RouteRoadNetwork:
    """Trasa przygotowana dla silnika ruchu.

    ``stops`` i ``roads`` są bezpośrednio zgodne z klasami w katalogu
    ``models``. Każdy obiekt ``StopRoad`` dostaje dodatkowo atrybut
    ``length_m`` z długością geometryczną odcinka.
    """
    geometry: RouteGeometry
    stops: list[Stop]
    roads: list[StopRoad]
    stop_shape_indexes: list[int]


def _closest_shape_index(
    stop: RouteStop,
    points: tuple[GeoPoint, ...],
    first_index: int,
) -> int:
    """Najbliższy punkt kształtu, szukany od wskazanego indeksu.

    Ograniczenie zakresu utrzymuje kolejność przystanków nawet gdy trasa
    zawraca lub przechodzi blisko własnego wcześniejszego fragmentu.
    """
    if first_index >= len(points):
        raise ValueError("Brak punktów kształtu dla kolejnego przystanku")
    stop_point = Point(stop.lat, stop.lon)
    return min(
        range(first_index, len(points)),
        key=lambda index: stop_point.distance_to(Point(points[index].lat, points[index].lon)),
    )


def build_stop_roads(
    tram_number,
    direction_id,
    dates=None,
    data_dir: Path = tram_data.DATA_DIR,
) -> RouteRoadNetwork:
    """Buduje odcinki torowiska między kolejnymi przystankami.

    Przystanek zostaje dopasowany do najbliższego punktu odpowiedniego
    ``shape`` GTFS. Dla każdej pary sąsiadujących przystanków powstaje jeden
    ``StopRoad`` zawierający punkty rzeczywistego przebiegu torowiska.

    Przykład::

        network = build_stop_roads("4", direction_id=0)
        tram = Tram(1, network.stops, loop_route_num=1, name="Linia 4")
    """
    geometry = get_route_geometry(tram_number, direction_id, dates, data_dir)
    if geometry is None:
        raise ValueError(f"Nie znaleziono linii {tram_number}, kierunek {direction_id}")
    if isinstance(geometry, list):  # Ochrona przed wywołaniem bez direction_id.
        raise ValueError("direction_id jest wymagany do budowy odcinków")
    if len(geometry.stops) < 2:
        raise ValueError("Trasa musi mieć co najmniej dwa przystanki")
    if len(geometry.points) < 2:
        raise ValueError("Geometria trasy musi mieć co najmniej dwa punkty")

    stop_shape_indexes: list[int] = []
    search_from = 0
    for stop in geometry.stops:
        index = _closest_shape_index(stop, geometry.points, search_from)
        stop_shape_indexes.append(index)
        search_from = index

    model_stops = [
        Stop(id_=stop.stop_id, lat=stop.lat, lon=stop.lon, name=stop.name)
        for stop in geometry.stops
    ]
    roads: list[StopRoad] = []
    for index, (start, end) in enumerate(zip(model_stops, model_stops[1:])):
        start_shape_index = stop_shape_indexes[index]
        end_shape_index = stop_shape_indexes[index + 1]

        # Użycie dokładnych współrzędnych przystanków zapewnia, że marker
        # tramwaju zaczyna i kończy odcinek dokładnie na przystanku.
        segment_points = [Point(start.lat, start.lon)]
        segment_points.extend(
            Point(point.lat, point.lon)
            for point in geometry.points[start_shape_index:end_shape_index + 1]
        )
        segment_points.append(Point(end.lat, end.lon))

        # Usunięcie bezpośrednio powtórzonych punktów zapobiega odcinkom
        # o długości zero podczas interpolacji w motion.py.
        unique_points = [segment_points[0]]
        for point in segment_points[1:]:
            if point.lat != unique_points[-1].lat or point.lon != unique_points[-1].lon:
                unique_points.append(point)

        road = StopRoad(start=start, end=end, points=unique_points)
        road.length_m = sum(
            first.distance_to(second)
            for first, second in zip(unique_points, unique_points[1:])
        )
        roads.append(road)

    for stop in model_stops:
        stop.add_roads([road for road in roads if road.start.id == stop.id])

    return RouteRoadNetwork(geometry, model_stops, roads, stop_shape_indexes)


def get_route_geometries(
    trams=None,
    dates=None,
    data_dir: Path = tram_data.DATA_DIR,
) -> list[RouteGeometry]:
    """Zwraca główny wariant geometrii dla każdej linii i kierunku.

    ``trams`` może być listą numerów linii, np. ``["4", " tram 8 "]``
    (prefiks ``tram`` jest opcjonalny). ``dates`` przekazywane jest dalej do
    filtra usług GTFS. Dla każdej pary linia/kierunek wybierany jest wariant
    reprezentatywny przez istniejące funkcje z ``tram_data.py``.
    """
    normalized = None
    if trams is not None:
        normalized = []
        for tram in trams:
            number = str(tram).strip()
            if number.lower().startswith("tram "):
                number = number[5:].strip()
            if number:
                normalized.append(number)
    shapes = tram_data.get_shapes_of_trips(normalized, dates, data_dir)
    stops = tram_data.get_stops_for_tram(normalized, dates, data_dir)

    if shapes.empty:
        return []

    result: list[RouteGeometry] = []
    keys = ["tram_number", "direction_id"]
    for (tram_number, direction_id), shape_rows in shapes.groupby(keys, sort=True):
        shape_rows = shape_rows.sort_values("shape_pt_sequence")
        first = shape_rows.iloc[0]
        route_stops = stops[
            (stops["tram_number"].astype(str) == str(tram_number))
            & (stops["direction_id"].astype(str) == str(direction_id))
        ].sort_values("stop_sequence")
        points = tuple(
            GeoPoint(float(row.shape_pt_lat), float(row.shape_pt_lon))
            for row in shape_rows.itertuples(index=False)
        )
        stop_items = tuple(
            RouteStop(
                stop_id=str(row.stop_id),
                name=str(row.stop_name),
                code="" if pd.isna(row.stop_number) else str(row.stop_number),
                station_code="" if pd.isna(row.station_code) else str(row.station_code),
                sequence=int(row.stop_sequence),
                lat=float(row.stop_lat),
                lon=float(row.stop_lon),
            )
            for row in route_stops.itertuples(index=False)
        )
        result.append(RouteGeometry(
            tram_number=str(tram_number),
            direction_id=str(direction_id),
            direction="" if pd.isna(first.direction) else str(first.direction),
            shape_id=str(first.shape_id),
            points=points,
            stops=stop_items,
        ))
    return result


def get_route_geometry(
    tram_number,
    direction_id=None,
    dates=None,
    data_dir: Path = tram_data.DATA_DIR,
) -> RouteGeometry | list[RouteGeometry] | None:
    """Pobiera geometrię jednej linii; bez kierunku zwraca oba kierunki."""
    geometries = get_route_geometries([tram_number], dates, data_dir)
    if direction_id is None:
        return geometries
    wanted = str(direction_id)
    return next((g for g in geometries if g.direction_id == wanted), None)


def get_available_lines(data_dir: Path = tram_data.DATA_DIR) -> list[str]:
    """Numery linii tramwajowych dostępnych w pliku routes.txt."""
    routes = tram_data._read("routes.txt", Path(data_dir))
    if routes is None:
        return []
    return sorted(routes["route_short_name"].dropna().astype(str).unique(),
                  key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))


def map_center(geometries: list[RouteGeometry]) -> tuple[float, float]:
    """Środek wspólnego obszaru geometrii, wygodny do ustawienia mapy."""
    points = [p for route in geometries for p in route.points]
    if not points:
        raise ValueError("Nie można wyznaczyć środka: brak geometrii tras")
    return ((min(p.lat for p in points) + max(p.lat for p in points)) / 2,
            (min(p.lon for p in points) + max(p.lon for p in points)) / 2)
