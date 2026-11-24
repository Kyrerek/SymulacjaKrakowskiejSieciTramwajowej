from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING
from models.Point import Point
from models.Stop import Stop
from models.StopRoad import StopRoad
from models.TramStatus import TramStatus

if TYPE_CHECKING:
    from fleet import TramType
    from passengers import PassengerSystem


class Tram(Point):
    def __init__(
        self,
        id_,
        route: list[Stop],
        start_time: datetime,
        loop_route_num: int,
        name: str = "",
        delay=lambda time, stop: 1000,
        tram_type: TramType | None = None,
        passengers: PassengerSystem | None = None,
    ):
        super().__init__(route[0].lat, route[0].lon)
        self.id = id_
        self.name = name
        self.route = route
        self.start_time = start_time
        self.loop_route_num = loop_route_num
        self.status = TramStatus.SLEEP
        self.delay_ = delay
        self.current_stop: Stop | None = None

        # Pasażerowie i tabor
        self.tram_type = tram_type
        self.passengers = passengers
        self.load: int = 0
        self.onboard: dict[str, int] = {}     # cel podróży (stop_id) -> liczba osób w pojeździe
        self.route_pos: int = 0               # indeks aktualnego przystanku w route

        # Wewnętrzny stan kinematyki dla samplera
        self._is_moving: bool = False
        self._active_road: StopRoad | None = None
        self._d0: float = 0.0
        self._d1: float = 0.0
        self._t0: float = 0.0
        self._t1: float = 0.0

    @property
    def capacity(self) -> int:
        if self.tram_type is not None:
            return self.tram_type.capacity
        return self.passengers.params.default_capacity if self.passengers else 0

    @property
    def doors(self) -> int:
        if self.tram_type is not None:
            return self.tram_type.doors
        return self.passengers.params.default_doors if self.passengers else 1

    @property
    def fullness(self) -> float:
        cap = self.capacity
        return self.load / cap if cap else 0.0

    def downstream_ids(self, horizon: int) -> set[str]:
        """Identyfikatory przystanków, które tramwaj odwiedzi w najbliższych `horizon` krokach."""
        n = len(self.route)
        return {str(self.route[(self.route_pos + k) % n].id) for k in range(1, min(horizon, n - 1) + 1)}

    def dwell_delay(self, time: datetime, stop: Stop) -> float:
        if self.passengers is not None:
            return self.passengers.exchange(time, self, stop).dwell_s
        return self.delay_(time, stop.enter_tram_num(time, self))

    def set_static(self, point: Point, status: TramStatus):
        self.status = status
        self._is_moving = False
        self.lat = point.lat
        self.lon = point.lon

    def set_moving(self, road: StopRoad, d0: float, d1: float, t0: float, t1: float):
        self.status = TramStatus.ARRIVAL
        self._is_moving = True
        self._active_road = road
        self._d0 = d0
        self._d1 = d1
        self._t0 = t0
        self._t1 = t1

    def sync_position(self, sim_now: float) -> Point:
        """Aktualizuje i zwraca współrzędne tramwaju w danej sekundzie symulacji."""
        if not self._is_moving or self._active_road is None:
            return Point(self.lat, self.lon)

        duration = self._t1 - self._t0
        progress = 1.0 if duration <= 0 else min(1.0, max(0.0, (sim_now - self._t0) / duration))
        dist = self._d0 + progress * (self._d1 - self._d0)
        pt = self._active_road.point_at_distance(dist)
        self.lat = pt.lat
        self.lon = pt.lon
        return pt