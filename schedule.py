"""Tramwaje tworzone z rozkładu jazdy GTFS.

Jeden tramwaj = jeden blok (``block_id``) jednego dnia rozkładowego, czyli ciąg kursów
wykonywanych przez ten sam pojazd. Dzięki temu opóźnienie z jednego kursu przenosi się
na następny (jeśli postój na pętli nie wystarczy, żeby je odrobić).

Zasady ruchu (realizowane w motion.py):
  * tramwaj nie odjeżdża z pierwszego przystanku kursu przed czasem z rozkładu;
  * na przystankach pośrednich nie czeka na rozkład - jedzie od razu po postoju;
  * czasy przejazdu odcinków pochodzą z samplerów wytrenowanych na danych realtime,
    a gdy samplera dla odcinka brak - z mediany czasu rozkładowego tego odcinka;
  * gdy kolejny kurs zaczyna się na innym przystanku niż skończył się poprzedni
    (zjazd do zajezdni), tramwaj znika z mapy i pojawia się przed kolejnym kursem.

    network = build_schedule_network(start, stop, lines=["4", "52"])
    trams = network.make_trams(start, stop)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import pandas as pd

import tram_data
from models import Point, Stop, StopRoad, Tram
from route_geometry import split_shape_by_stops

if TYPE_CHECKING:
    from trainers import StopRoadTrainer

# Ile sekund przed odjazdem pierwszego kursu (albo kursu po zjeździe) tramwaj
# pojawia się na mapie na przystanku początkowym.
APPEAR_BEFORE_S = 120.0
MIN_PLANNED_TRAVEL_S = 20.0
# Stały czas postoju na przystanku (dopóki nie podepniemy PassengerSystem).
DWELL_S = 30.0


@dataclass(slots=True)
class ScheduledStop:
    stop: Stop
    arrival: datetime
    departure: datetime


@dataclass(slots=True)
class ScheduledTrip:
    trip_id: str
    line: str
    direction_id: str
    headsign: str
    stops: list[ScheduledStop]

    @property
    def departure(self) -> datetime:
        return self.stops[0].departure

    @property
    def arrival(self) -> datetime:
        return self.stops[-1].arrival


@dataclass(slots=True)
class ScheduledBlock:
    block_id: str
    service_day: date
    trips: list[ScheduledTrip]

    @property
    def key(self) -> str:
        return f"{self.block_id}@{self.service_day:%Y%m%d}"

    @property
    def lines(self) -> list[str]:
        return list(dict.fromkeys(trip.line for trip in self.trips))


@dataclass
class ScheduleNetwork:
    stops: dict[str, Stop]
    roads: dict[tuple[str, str], StopRoad]
    blocks: list[ScheduledBlock]
    planned_travel_s: dict[tuple[str, str], float] = field(default_factory=dict)

    def bind_travel_times(self, trainer: StopRoadTrainer | None = None) -> tuple[int, int]:
        """Ustawia czas przejazdu odcinków. Zwraca (z samplera, z rozkładu)."""
        from_sampler = from_schedule = 0
        for key, road in self.roads.items():
            sampler = trainer.samplers.get(key) if trainer is not None else None
            if sampler is not None:
                road.delay_ = sampler.predict
                from_sampler += 1
            else:
                planned = self.planned_travel_s.get(key, 60.0)
                road.delay_ = lambda t, d=planned: d
                from_schedule += 1
        return from_sampler, from_schedule

    def make_trams(self, start_time: datetime, stop_time: datetime, dwell_s: float = DWELL_S) -> list[Tram]:
        """Tramwaje dla bloków, które mają jakikolwiek odjazd w oknie symulacji.

        Kursy już trwające w chwili startu są przycinane do pierwszego przystanku,
        z którego odjazd wypada nie wcześniej niż ``start_time``.
        """
        trams: list[Tram] = []
        for block in self.blocks:
            trips = _trim_block(block.trips, start_time, stop_time)
            if not trips:
                continue
            first = trips[0]
            appear = max(start_time, first.departure - timedelta(seconds=APPEAR_BEFORE_S))
            route = [s.stop for trip in trips for s in trip.stops]
            tram = Tram(
                id_=block.key,
                route=route,
                start_time=appear,
                loop_route_num=1,
                name=f"Blok {block.block_id}",
                delay=lambda time, stop, d=dwell_s: d,
                schedule=trips,
            )
            trams.append(tram)
        return trams


def _trim_block(trips: list[ScheduledTrip], start_time: datetime, stop_time: datetime) -> list[ScheduledTrip]:
    result: list[ScheduledTrip] = []
    for trip in trips:
        if trip.departure >= stop_time:
            break
        if trip.arrival < start_time:
            continue
        if trip.departure >= start_time:
            result.append(trip)
            continue
        # Kurs trwa w chwili startu: zaczynamy od pierwszego przystanku z odjazdem >= start.
        rest = [s for s in trip.stops if s.departure >= start_time]
        if len(rest) >= 2:
            result.append(ScheduledTrip(trip.trip_id, trip.line, trip.direction_id, trip.headsign, rest))
    return result


def service_days(start_time: datetime, stop_time: datetime) -> list[date]:
    """Dni rozkładowe, których kursy mogą wypaść w oknie (z dniem poprzednim - kursy po północy)."""
    first = start_time.date() - timedelta(days=1)
    return [first + timedelta(days=i) for i in range((stop_time.date() - first).days + 1)]


def build_schedule_network(
    start_time: datetime,
    stop_time: datetime,
    lines: Iterable[str] | None = None,
    data_dir: Path = tram_data.DATA_DIR,
) -> ScheduleNetwork:
    lines = None if lines is None else [str(line) for line in lines]
    if lines is not None and not lines:
        return ScheduleNetwork({}, {}, [])
    stops: dict[str, Stop] = {}
    roads: dict[tuple[str, str], StopRoad] = {}
    planned: dict[tuple[str, str], list[float]] = {}
    shapes_done: set[tuple[str, tuple[str, ...]]] = set()
    blocks: list[ScheduledBlock] = []

    for day in service_days(start_time, stop_time):
        try:
            data = tram_data.load_day(day, routes=lines, data_dir=data_dir)
        except ValueError:
            continue  # brak rozkładu na ten dzień
        if data.trips.empty:
            continue

        midnight = datetime.combine(day, datetime.min.time())
        stop_info = data.stops.set_index("stop_id")
        for row in stop_info.itertuples():
            if row.Index not in stops:
                stops[row.Index] = Stop(row.Index, row.stop_lat, row.stop_lon, name=row.stop_name)

        shape_points = {
            shape_id: [Point(r.shape_pt_lat, r.shape_pt_lon) for r in group.itertuples()]
            for shape_id, group in data.shapes.groupby("shape_id", sort=False)
        }
        trip_info = data.trips.set_index("trip_id")
        trips: dict[str, ScheduledTrip] = {}

        for trip_id, group in data.stop_times.groupby("trip_id", sort=False):
            if len(group) < 2:
                continue
            info = trip_info.loc[trip_id]
            ids = tuple(group["stop_id"])
            arr = group["arrival_s"].to_numpy()
            dep = group["departure_s"].to_numpy()

            for a, b, t in zip(ids, ids[1:], arr[1:] - dep[:-1]):
                planned.setdefault((a, b), []).append(float(t))

            shape_key = (info.shape_id, ids)
            if shape_key not in shapes_done:
                shapes_done.add(shape_key)
                _add_roads(stops, roads, ids, shape_points.get(info.shape_id))

            trips[trip_id] = ScheduledTrip(
                trip_id=trip_id,
                line=str(info.route_short_name),
                direction_id=str(info.direction_id),
                headsign="" if pd.isna(info.headsign) else str(info.headsign),
                stops=[
                    ScheduledStop(
                        stops[sid],
                        midnight + timedelta(seconds=int(a)),
                        midnight + timedelta(seconds=int(d)),
                    )
                    for sid, a, d in zip(ids, arr, dep)
                ],
            )

        for block_id, group in data.trips.groupby("block_id", sort=True):
            block_trips = sorted(
                (trips[t] for t in group["trip_id"] if t in trips),
                key=lambda trip: trip.departure,
            )
            if block_trips:
                blocks.append(ScheduledBlock(str(block_id), day, block_trips))

    for stop in stops.values():
        stop.add_roads([road for (start, _), road in roads.items() if start == stop.id])

    planned_median = {
        key: max(MIN_PLANNED_TRAVEL_S, float(pd.Series(values).median()))
        for key, values in planned.items()
    }
    return ScheduleNetwork(stops, roads, blocks, planned_median)


def _add_roads(stops: dict[str, Stop], roads: dict[tuple[str, str], StopRoad],
               ids: tuple[str, ...], shape: list[Point] | None) -> None:
    missing = [(a, b) for a, b in zip(ids, ids[1:]) if (a, b) not in roads]
    if not missing:
        return
    trip_stops = [stops[sid] for sid in ids]
    if shape and len(shape) >= 2:
        _, segments = split_shape_by_stops(shape, trip_stops)
    else:
        segments = [[Point(a.lat, a.lon), Point(b.lat, b.lon)] for a, b in zip(trip_stops, trip_stops[1:])]
    for a, b, points in zip(trip_stops, trip_stops[1:], segments):
        key = (str(a.id), str(b.id))
        if key not in roads:
            roads[key] = StopRoad(a, b, points)
