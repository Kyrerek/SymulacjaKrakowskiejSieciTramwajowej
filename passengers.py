"""System pasażerów: kolejki na przystankach, wsiadanie, wysiadanie i czas postoju.

Model zagregowany, bez procesów SimPy dla pojedynczych osób:
  * napływ na przystanek jest liczony leniwie - przy wizycie tramwaju losujemy, ile osób
    przyszło od ostatniej aktualizacji (niejednorodny proces Poissona), więc koszt to O(1)
    na postój, a nie O(liczba pasażerów);
  * pasażerowie są grupowani w partie (czas przybycia, liczba) osobno dla każdego celu
    podróży, więc dla każdej osoby znamy cel (stąd wysiadający) i czas oczekiwania;
  * wsiadają tylko ci, których cel leży na dalszej trasie tego tramwaju, w kolejności
    przybycia, do wyczerpania pojemności - reszta czeka na następny tramwaj.

WSZYSTKIE LICZBY W PassengerParams I DOMYŚLNE PROFILE SĄ PRZYBLIŻENIAMI: w danych
realtime nie ma napełnienia, a danych pasażerskich ZTP nie będzie. Rozkład przestrzenny
popytu pochodzi z GUS/OSM (stop_weights.py), a skala (base_rate_per_hour, demand_scale)
i szybkości wsiadania są parametrami niepewnymi - dlatego wyniki podajemy jako analizę
wrażliwości na te parametry, a nie jako skalibrowane przewidywanie.
"""
from __future__ import annotations

import heapq
import json
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from models.Stop import Stop
    from models.Tram import Tram


def _normalized(values) -> tuple[float, ...]:
    mean = sum(values) / len(values)
    return tuple(v / mean for v in values)


# Względne natężenie napływu w kolejnych godzinach doby (średnia = 1). PRZYBLIŻENIE.
WEEKDAY_PROFILE = _normalized((0.15, 0.08, 0.05, 0.05, 0.2, 0.7, 1.5, 2.2, 2.0, 1.4, 1.1, 1.1,
                               1.2, 1.3, 1.5, 1.9, 2.1, 1.9, 1.4, 1.0, 0.8, 0.6, 0.4, 0.25))
WEEKEND_PROFILE = _normalized((0.3, 0.2, 0.1, 0.05, 0.1, 0.25, 0.5, 0.8, 1.1, 1.4, 1.6, 1.7,
                               1.7, 1.7, 1.7, 1.6, 1.5, 1.4, 1.2, 1.0, 0.8, 0.6, 0.5, 0.4))


@dataclass(slots=True)
class PassengerParams:
    # popyt
    base_rate_per_hour: float = 50.0   # wsiadających / h na przystanek o wadze 1 i profilu 1
    demand_scale: float = 1.0          # mnożnik popytu (suwak w interfejsie)
    weekday_profile: tuple[float, ...] = WEEKDAY_PROFILE
    weekend_profile: tuple[float, ...] = WEEKEND_PROFILE
    horizon: int = 12                  # na ile kolejnych przystanków planuje podróż pasażer
    hop_decay: float = 4.0             # zanik preferencji celu z liczbą przystanków
    warmup_s: float = 600.0            # ruch "sprzed startu" zastany przez pierwszy tramwaj
    patience_s: float = 1800.0         # po tylu sekundach oczekiwania pasażer rezygnuje
    # postój: T = door_time + max(wsiadający/(k_b*drzwi), wysiadający/(k_a*drzwi))
    door_time_s: float = 15.0          # obserwowany p10 postoju ~16 s, mediana ~29 s (19.10)
    board_rate: float = 0.4            # osób / s / drzwi - DO KALIBRACJI
    alight_rate: float = 0.6           # osób / s / drzwi - DO KALIBRACJI
    max_dwell_s: float = 180.0
    # tabor, gdy tramwaj nie ma przypisanego typu
    default_capacity: int = 200
    default_doors: int = 5


DEFAULT_WEIGHTS_PATH = Path("data/stop_weights.json")   # tworzy go stop_weights.py


class StopWeights:
    """Waga atrakcyjności przystanku (1.0 = przeciętny). Klucz: stop_id albo nazwa."""

    def __init__(self, weights: dict[str, float] | None = None, default: float = 1.0):
        self.weights = dict(weights or {})
        self.default = default

    @classmethod
    def from_json(cls, path: str | Path, default: float = 1.0) -> "StopWeights":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), default)

    @classmethod
    def from_default(cls, path: str | Path = DEFAULT_WEIGHTS_PATH) -> "StopWeights":
        """Wagi z data/stop_weights.json, a gdy pliku nie ma - wszystkie równe 1."""
        return cls.from_json(path) if Path(path).exists() else cls()

    def weight(self, stop: "Stop") -> float:
        return float(self.weights.get(str(stop.id), self.weights.get(stop.name, self.default)))


@dataclass(slots=True)
class StopExchange:
    """Jedna wymiana pasażerów (rekord dziennika dla KPI i interfejsu)."""
    timestamp: datetime
    tram_id: str
    stop_id: str
    boarded: int
    alighted: int
    left_behind: int
    load_after: int
    capacity: int
    queue_after: int
    dwell_s: float


@dataclass(slots=True)
class _StopState:
    stop_id: str
    name: str
    lat: float
    lon: float
    weight: float
    reach: dict[str, int] = field(default_factory=dict)        # cel -> min. liczba przystanków
    dest_ids: list[str] = field(default_factory=list)
    dest_p: np.ndarray | None = None
    queue: dict[str, deque] = field(default_factory=dict)       # cel -> partie [t_epoch, n]
    queue_total: int = 0
    last_update: datetime | None = None
    arrived: int = 0
    boarded: int = 0
    abandoned: int = 0
    wait_sum_s: float = 0.0
    max_queue: int = 0


class PassengerSystem:
    def __init__(self, params: PassengerParams | None = None, weights: StopWeights | None = None,
                 seed: int | None = 0):
        self.params = params or PassengerParams()
        self.weights = weights if weights is not None else StopWeights.from_default()
        self.rng = np.random.default_rng(seed)
        self.exchanges: list[StopExchange] = []
        self._states: dict[str, _StopState] = {}
        self._dirty = True

    # ------------------------------------------------------------- konfiguracja
    def register_route(self, stops: list["Stop"], cyclic: bool = True) -> None:
        """Rejestruje trasę (listę przystanków, także cykliczną): ustala, dokąd można
        z danego przystanku dojechać w ramach `horizon` przystanków. Dla pojedynczego
        kursu z rozkładu `cyclic=False` - za ostatnim przystankiem trasa się kończy."""
        n = len(stops)
        for s in stops:
            if str(s.id) not in self._states:
                self._states[str(s.id)] = _StopState(
                    str(s.id), s.name, s.lat, s.lon, self.weights.weight(s))
        for i, origin in enumerate(stops):
            st = self._states[str(origin.id)]
            last_k = min(self.params.horizon, n - 1) if cyclic else min(self.params.horizon, n - 1 - i)
            for k in range(1, last_k + 1):
                dest = str(stops[(i + k) % n].id)
                if dest != st.stop_id and k < st.reach.get(dest, 10**9):
                    st.reach[dest] = k
        self._dirty = True

    def _finalize(self) -> None:
        p = self.params
        for st in self._states.values():
            st.dest_ids = list(st.reach)
            if st.dest_ids:
                w = np.array([self._states[d].weight * math.exp(-(st.reach[d] - 1) / p.hop_decay)
                              for d in st.dest_ids], dtype=float)
                st.dest_p = w / w.sum() if w.sum() > 0 else None
            else:
                st.dest_p = None
        self._dirty = False

    # ------------------------------------------------------------------- popyt
    def _expected_arrivals(self, weight: float, t0: datetime, t1: datetime) -> float:
        p, total, t = self.params, 0.0, t0
        while t < t1:
            nxt = min(t1, t.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
            profile = p.weekend_profile if t.weekday() >= 5 else p.weekday_profile
            total += (nxt - t).total_seconds() / 3600.0 * profile[t.hour]
            t = nxt
        return p.base_rate_per_hour * p.demand_scale * weight * total

    def _advance(self, st: _StopState, now: datetime) -> None:
        """Realizuje napływ pasażerów do chwili `now` i usuwa zniecierpliwionych."""
        if self._dirty:
            self._finalize()
        p = self.params
        if st.last_update is None:
            st.last_update = now - timedelta(seconds=p.warmup_s)
        if now > st.last_update:
            if st.dest_p is not None:
                n = int(self.rng.poisson(self._expected_arrivals(st.weight, st.last_update, now)))
                if n:
                    mid = (st.last_update + (now - st.last_update) / 2).timestamp()
                    for dest, c in zip(st.dest_ids, self.rng.multinomial(n, st.dest_p)):
                        if c:
                            st.queue.setdefault(dest, deque()).append([mid, int(c)])
                    st.queue_total += n
                    st.arrived += n
            st.last_update = now
        now_ts = now.timestamp()
        for dq in st.queue.values():
            while dq and now_ts - dq[0][0] > p.patience_s:
                lost = dq.popleft()[1]
                st.queue_total -= lost
                st.abandoned += lost
        st.max_queue = max(st.max_queue, st.queue_total)

    # ------------------------------------------------------------ wymiana na przystanku
    def dwell_time(self, tram: "Tram", boarded: int, alighted: int) -> float:
        p, doors = self.params, max(1, tram.doors)
        t = p.door_time_s + max(boarded / (p.board_rate * doors), alighted / (p.alight_rate * doors))
        return min(t, p.max_dwell_s)

    def exchange(self, time: datetime, tram: "Tram", stop: "Stop") -> StopExchange:
        """Wysiadają pasażerowie z celem = ten przystanek, potem wsiadają oczekujący
        (do pojemności). Aktualizuje stan tramwaju i zwraca rekord wymiany."""
        st = self._states.get(str(stop.id))
        if st is None:                       # trasa nie zarejestrowana: tylko wysiadanie
            self.register_route([stop])
            st = self._states[str(stop.id)]
        self._advance(st, time)

        alighted = tram.onboard.pop(st.stop_id, 0)
        tram.load -= alighted

        cap_left = max(0, tram.capacity - tram.load)
        eligible = tram.downstream_ids(self.params.horizon)
        now_ts = time.timestamp()
        heap = [(st.queue[d][0][0], d) for d in eligible if st.queue.get(d)]
        heapq.heapify(heap)
        boarded = 0
        while heap and cap_left > 0:
            _, dest = heapq.heappop(heap)
            batch = st.queue[dest][0]
            take = min(batch[1], cap_left)
            batch[1] -= take
            cap_left -= take
            boarded += take
            st.wait_sum_s += take * (now_ts - batch[0])
            tram.onboard[dest] = tram.onboard.get(dest, 0) + take
            if batch[1] == 0:
                st.queue[dest].popleft()
                if st.queue[dest]:
                    heapq.heappush(heap, (st.queue[dest][0][0], dest))
        tram.load += boarded
        st.queue_total -= boarded
        st.boarded += boarded
        left_behind = sum(b[1] for d in eligible for b in st.queue.get(d, ()))

        record = StopExchange(
            timestamp=time, tram_id=str(tram.id), stop_id=st.stop_id, boarded=boarded,
            alighted=alighted, left_behind=left_behind, load_after=tram.load,
            capacity=tram.capacity, queue_after=st.queue_total,
            dwell_s=self.dwell_time(tram, boarded, alighted))
        self.exchanges.append(record)
        return record

    # ----------------------------------------------------------- odczyt dla interfejsu
    def queue_size(self, stop_id: str, time: datetime) -> int:
        st = self._states[str(stop_id)]
        self._advance(st, time)
        return st.queue_total

    def stop_stats(self) -> list[dict]:
        """Zagregowane statystyki przystanków (po zakończonej symulacji)."""
        return [{"stop_id": s.stop_id, "name": s.name, "lat": s.lat, "lon": s.lon,
                 "arrived": s.arrived, "boarded": s.boarded, "abandoned": s.abandoned,
                 "max_queue": s.max_queue,
                 "mean_wait_s": (s.wait_sum_s / s.boarded) if s.boarded else 0.0}
                for s in self._states.values() if s.arrived or s.boarded]

    def summary(self) -> dict:
        ex = self.exchanges
        boarded = sum(e.boarded for e in ex)
        wait = sum(s.wait_sum_s for s in self._states.values())
        return {
            "exchanges": len(ex), "boarded": boarded, "alighted": sum(e.alighted for e in ex),
            "arrived": sum(s.arrived for s in self._states.values()),
            "abandoned": sum(s.abandoned for s in self._states.values()),
            "left_behind_events": sum(1 for e in ex if e.left_behind > 0),
            "mean_wait_s": wait / boarded if boarded else 0.0,
            "mean_dwell_s": sum(e.dwell_s for e in ex) / len(ex) if ex else 0.0,
            "max_fullness": max((e.load_after / e.capacity for e in ex if e.capacity), default=0.0),
        }