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
    from schedule import ScheduledTrip


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
        schedule: list[ScheduledTrip] | None = None,
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

        # Rozkład jazdy: lista kursów bloku (None = stary tryb cyklicznej trasy)
        self.schedule = schedule
        self.current_trip: ScheduledTrip | None = None
        self.delay_s: float = 0.0             # opóźnienie względem rozkładu (ostatni przystanek)
        self.trip_offset: int = 0             # indeks pierwszego przystanku bieżącego kursu w route
        self.trip_seg: int = 0                # indeks przystanku, z którego tramwaj ostatnio ruszył (w kursie)
        self.completed_trips: list[dict] = [] # podsumowania zakończonych kursów

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
        self._last_dist: float = 0.0          # pozycja na aktywnym odcinku (m), także w kolejce

        # Dziennik ruchu dla animacji: zdarzenia [t, rodzaj, a, b, c, d, status, opóźnienie, load, kurs].
        # rodzaj 0 = niewidoczny, 1 = postój w punkcie (a=lat, b=lon),
        # 2 = jazda po odcinku (a=StopRoad, b=d0, c=d1, d=t1). `clock` ustawia TramMotion.
        self.clock = None
        self.events: list[list] = []

    @property
    def line(self) -> str:
        return self.current_trip.line if self.current_trip is not None else str(self.id)

    @property
    def headsign(self) -> str:
        return self.current_trip.headsign if self.current_trip is not None else ""

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
        if self.current_trip is not None:
            # Tryb rozkładowy: tylko dalsze przystanki bieżącego kursu (bez zawracania na pętli).
            idx = self.route_pos - self.trip_offset
            return {str(s.stop.id) for s in self.current_trip.stops[idx + 1: idx + 1 + horizon]}
        n = len(self.route)
        return {str(self.route[(self.route_pos + k) % n].id) for k in range(1, min(horizon, n - 1) + 1)}

    def dwell_delay(self, time: datetime, stop: Stop) -> float:
        if self.passengers is not None:
            return self.passengers.exchange(time, self, stop).dwell_s
        return self.delay_(time, stop.enter_tram_num(time, self))

    def _log(self, kind: int, a=0.0, b=0.0, c=0.0, d=0.0):
        if self.clock is None:
            return
        t = self.clock()
        ev = [t, kind, a, b, c, d, self.status, self.delay_s, self.load, self.current_trip]
        if self.events and abs(self.events[-1][0] - t) < 1e-9:
            self.events[-1] = ev          # kilka zmian w tej samej chwili - liczy się ostatnia
        else:
            self.events.append(ev)

    def hide(self, status: TramStatus = TramStatus.SLEEP):
        """Tramwaj znika z mapy (przed pierwszym kursem, zjazd, koniec służby)."""
        self.status = status
        self._is_moving = False
        self._log(0)

    def set_static(self, point: Point, status: TramStatus,
                   road: StopRoad | None = None, dist: float | None = None):
        """Postój w miejscu. `road`/`dist` - gdy tramwaj stoi na odcinku (kolejka)."""
        self.status = status
        self._is_moving = False
        self.lat = point.lat
        self.lon = point.lon
        self._log(1, point.lat, point.lon)
        if road is not None and dist is not None:
            self._active_road = road
            self._last_dist = dist
        elif status == TramStatus.DWEL and self._active_road is not None:
            self._last_dist = self._active_road.total_length
        else:
            self._active_road = None
            self._last_dist = 0.0

    def start_segment(self, seg: int):
        """Początek przejazdu z przystanku `seg` bieżącego kursu (zeruje pozycję na odcinku)."""
        self.trip_seg = seg
        self._active_road = None
        self._last_dist = 0.0

    @property
    def trip_progress(self) -> float | None:
        """Postęp na bieżącym kursie w przystankach (np. 3.4 = 40% drogi z 4. do 5. przystanku)."""
        if self.current_trip is None:
            return None
        road = self._active_road
        frac = self._last_dist / road.total_length if road is not None and road.total_length > 0 else 0.0
        return self.trip_seg + min(1.0, max(0.0, frac))

    def set_moving(self, road: StopRoad, d0: float, d1: float, t0: float, t1: float):
        self.status = TramStatus.ARRIVAL
        self._is_moving = True
        self._active_road = road
        self._d0 = d0
        self._d1 = d1
        self._t0 = t0
        self._t1 = t1
        last = self.events[-1] if self.events else None
        if (last is not None and last[1] == 2 and last[2] is road and abs(last[4] - d0) < 1e-6
                and abs(last[5] - t0) < 1e-6 and last[6] == TramStatus.ARRIVAL
                and last[7] == self.delay_s and last[9] is self.current_trip):
            # Kolejny blok kolejki na tym samym odcinku, bez zatrzymania: wydłużamy przejazd
            # (prędkość jest ta sama, bo czas jest proporcjonalny do drogi).
            last[4], last[5] = d1, t1
        else:
            self._log(2, road, d0, d1, t1)

    def sync_position(self, sim_now: float) -> Point:
        """Aktualizuje i zwraca współrzędne tramwaju w danej sekundzie symulacji."""
        if not self._is_moving or self._active_road is None:
            return Point(self.lat, self.lon)

        duration = self._t1 - self._t0
        progress = 1.0 if duration <= 0 else min(1.0, max(0.0, (sim_now - self._t0) / duration))
        dist = self._d0 + progress * (self._d1 - self._d0)
        self._last_dist = dist
        pt = self._active_road.point_at_distance(dist)
        self.lat = pt.lat
        self.lon = pt.lon
        return pt