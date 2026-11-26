from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import simpy

from models import Stop, StopRoad, Tram, TramStatus
from schedule import APPEAR_BEFORE_S

TRAM_SPACING_M = 45.0
STOP_BUFFER_M = 45.0
BOARD_BEFORE_DEPARTURE_S = 30.0

# Parametry kontroli mijanych przystanków innych linii (bez zatrzymywania na wymianę pasażerów)
PASS_CLEAR_M = 15.0      # ile metrów za mijanym peronem zwalnia blokadę
PASS_LATERAL_M = 7.0     # maksymalna odległość w bok od osi toru (tolerancja przesunięcia słupka w GTFS)
PASS_MIN_DIR_COS = 0.6   # zgodność kierunku jazdy (ignoruje przystanki w przeciwną stronę)

_LAT_TO_M = 111_320.0
_LON_TO_M = 111_320.0 * math.cos(math.radians(50.06))


def _to_xy(lat: float, lon: float) -> tuple[float, float]:
    return (lon - 19.94) * _LON_TO_M, (lat - 50.06) * _LAT_TO_M


def _compute_stop_directions(all_stops: dict[str, Stop]) -> dict[str, tuple[float, float]]:
    """Wyznacza kierunek ruchu dla każdego przystanku na podstawie podpiętych odcinków."""
    dirs: dict[str, tuple[float, float]] = {}
    for stop_id, stop in all_stops.items():
        for road in stop.roads:
            if road.total_length > 2.0:
                p0 = road.point_at_distance(0.0)
                p1 = road.point_at_distance(min(road.total_length, 10.0))
                x0, y0 = _to_xy(p0.lat, p0.lon)
                x1, y1 = _to_xy(p1.lat, p1.lon)
                seg_len = math.hypot(x1 - x0, y1 - y0)
                if seg_len > 1e-6:
                    dirs[stop_id] = ((x1 - x0) / seg_len, (y1 - y0) / seg_len)
                    break

    for stop in all_stops.values():
        for road in stop.roads:
            end_id = str(road.end.id)
            if end_id not in dirs and road.total_length > 2.0:
                p0 = road.point_at_distance(max(0.0, road.total_length - 10.0))
                p1 = road.point_at_distance(road.total_length)
                x0, y0 = _to_xy(p0.lat, p0.lon)
                x1, y1 = _to_xy(p1.lat, p1.lon)
                seg_len = math.hypot(x1 - x0, y1 - y0)
                if seg_len > 1e-6:
                    dirs[end_id] = ((x1 - x0) / seg_len, (y1 - y0) / seg_len)
    return dirs


def _find_pass_through_stops(
    road: StopRoad,
    all_stops: dict[str, Stop],
    stop_dirs: dict[str, tuple[float, float]],
) -> list[tuple[Stop, float, float]]:
    """Wyszukuje przystanki innych linii leżące bezpośrednio na trasie odcinka (w tym samym kierunku).

    Zwraca listę (przystanek, dystans_slupka_na_odcinku, dystans_zwolnienia_za_slupkiem).
    """
    if road.total_length <= 2.0:
        return []

    start_id = str(road.start.id)
    end_id = str(road.end.id)
    candidates = {
        sid: _to_xy(s.lat, s.lon)
        for sid, s in all_stops.items()
        if sid != start_id and sid != end_id
    }
    if not candidates:
        return []

    best_match: dict[str, tuple[float, float]] = {}  # stop_id -> (dist2, road_dist)
    d = 0.0
    while d <= road.total_length:
        pm = road.point_at_distance(d)
        pa = road.point_at_distance(max(0.0, d - 2.0))
        pb = road.point_at_distance(min(road.total_length, d + 2.0))
        xm, ym = _to_xy(pm.lat, pm.lon)
        xa, ya = _to_xy(pa.lat, pa.lon)
        xb, yb = _to_xy(pb.lat, pb.lon)
        seg_len = math.hypot(xb - xa, yb - ya)
        ux, uy = ((xb - xa) / seg_len, (yb - ya) / seg_len) if seg_len > 1e-6 else (0.0, 0.0)

        for sid, (sx, sy) in candidates.items():
            vx, vy = sx - xm, sy - ym
            d2 = vx * vx + vy * vy
            if d2 > PASS_LATERAL_M * PASS_LATERAL_M:
                continue
            if ux != 0.0 or uy != 0.0:
                lateral = abs(-vx * uy + vy * ux)
                if lateral > PASS_LATERAL_M:
                    continue
                sdir = stop_dirs.get(sid)
                if sdir is not None and (ux * sdir[0] + uy * sdir[1]) < PASS_MIN_DIR_COS:
                    continue

            prev = best_match.get(sid)
            if prev is None or d2 < prev[0]:
                best_match[sid] = (d2, d)

        d += 2.0

    result: list[tuple[Stop, float, float]] = []
    for sid, (_, hit_dist) in best_match.items():
        clear_dist = min(road.total_length, hit_dist + PASS_CLEAR_M)
        result.append((all_stops[sid], hit_dist, clear_dist))

    result.sort(key=lambda item: item[1])
    return result


@dataclass(slots=True)
class SimulationSample:
    tram_id: str
    tram_name: str
    status: TramStatus
    timestamp: datetime
    lat: float
    lon: float
    line: str = ""
    headsign: str = ""
    delay_s: float = 0.0
    trip_id: str = ""
    progress: float | None = None     # postęp na kursie w przystankach (dla wykresu czas-droga)
    load: int = 0
    capacity: int = 0
    fullness: float = 0.0


class TramMotion:
    def __init__(self, start_time: datetime, stop_time: datetime, target_time_step: float = 10, env=None):
        self.env = simpy.Environment() if env is None else env
        self.start_time = start_time
        self.stop_time = stop_time
        self.target_time_step = float(target_time_step)
        self.samples: list[SimulationSample] = []
        self._shared_approach_slots: dict[tuple[str, int], simpy.Resource] = {}

    @property
    def current_time(self) -> datetime:
        return self.start_time + timedelta(seconds=self.env.now)

    def _get_approach_slot(self, stop_id: str, slot_from_end: int) -> simpy.Resource:
        key = (stop_id, slot_from_end)
        res = self._shared_approach_slots.get(key)
        if res is None:
            res = simpy.Resource(self.env, capacity=1)
            self._shared_approach_slots[key] = res
        return res

    def run_trams(self, trams: list[Tram]) -> list[SimulationSample]:
        self.samples = []
        max_time = (self.stop_time - self.start_time).total_seconds()
        for tram in trams:
            tram.clock = lambda: self.env.now
            tram.events = []
        if self.env.now >= max_time:
            return self.samples

        # Tramwaje ruszające wcześniej jako pierwsze rezerwują peron na starcie
        trams = sorted(
            trams,
            key=lambda t: t.schedule[0].departure if t.schedule else t.start_time,
        )

        # Wspólne zasoby peronów oraz slotów dojazdowych przed przystankiem końcowym
        shared_stop_locks: dict[str, simpy.Resource] = {}
        self._shared_approach_slots = {}
        all_stops: dict[str, Stop] = {}
        unique_roads: list[StopRoad] = []
        seen_road_ids: set[int] = set()

        for tram in trams:
            for stop in tram.route:
                stop_id = str(stop.id)

                if stop_id not in shared_stop_locks:
                    shared_stop_locks[stop_id] = stop.init_simpy(self.env)
                    all_stops[stop_id] = stop
                else:
                    stop.lock = shared_stop_locks[stop_id]

                stop.occupant = None

                for road in stop.roads:
                    if id(road) not in seen_road_ids:
                        seen_road_ids.add(id(road))
                        unique_roads.append(road)

                    if not road._queue_blocks or road._queue_blocks[0][0]._env is not self.env:
                        raw_blocks = road.init_queue_blocks(
                            self.env,
                            TRAM_SPACING_M,
                            STOP_BUFFER_M,
                        )
                        end_id = str(road.end.id)
                        merged_blocks: list[tuple[simpy.Resource, float]] = []
                        num_blocks = len(raw_blocks)
                        for idx, (res, dist) in enumerate(raw_blocks):
                            # slot_from_end = 0 to blok najbliżej przystanku końcowego (STOP_BUFFER_M przed peronem)
                            slot_from_end = num_blocks - 1 - idx
                            slot_key = (end_id, slot_from_end)
                            if slot_key not in self._shared_approach_slots:
                                self._shared_approach_slots[slot_key] = res
                            shared_res = self._shared_approach_slots[slot_key]

                            if merged_blocks and abs(merged_blocks[-1][1] - dist) <= 1.0:
                                merged_blocks[-1] = (shared_res, dist)
                            else:
                                merged_blocks.append((shared_res, dist))
                        road._queue_blocks = merged_blocks

            self.env.process(self._start_tram(tram))

        stop_dirs = _compute_stop_directions(all_stops)
        for road in unique_roads:
            road._pass_through_stops = _find_pass_through_stops(road, all_stops, stop_dirs)

        self.env.process(self._sampler_process(trams))
        self.env.run(until=max_time)
        return self.samples

    def _start_tram(self, tram: Tram):
        wait_time = (tram.start_time - self.current_time).total_seconds()
        if wait_time < 0:
            tram.hide(TramStatus.TERMINATED)
            return
        if wait_time > 0:
            yield self.env.timeout(wait_time)

        yield from self._run_tram(tram)
        tram.hide(TramStatus.TERMINATED)

    def _drive_segment(self, tram: Tram, road: StopRoad, d0: float, d1: float, total_travel: float):
        if road.total_length <= 0 or d1 <= d0 + 1e-6:
            return
        duration = total_travel * ((d1 - d0) / road.total_length)
        tram.set_moving(road, d0, d1, self.env.now, self.env.now + duration)
        yield self.env.timeout(duration)

    def _run_tram(self, tram: Tram):
        if tram.schedule is not None:
            yield from self._run_scheduled_tram(tram)
            return
        for _ in range(tram.loop_route_num):
            for idx, current_stop in enumerate(tram.route):
                next_stop = tram.route[(idx + 1) % len(tram.route)]
                yield from self._travel_to(tram, current_stop, next_stop, (idx + 1) % len(tram.route))

    def _wait_until(self, moment: datetime):
        remaining = (moment - self.current_time).total_seconds()
        if remaining > 0:
            yield self.env.timeout(remaining)

    def _run_scheduled_tram(self, tram: Tram):
        """Jazda według kursów bloku: odjazd z początku kursu nie wcześniej niż w rozkładzie."""
        at_stop = None
        route_offset = 0  # indeks pierwszego przystanku kursu w tram.route
        for trip in tram.schedule:
            first = trip.stops[0]
            if at_stop is None or at_stop.id != first.stop.id:
                # Pierwszy kurs albo kurs po zjeździe: tramwaj pojawia się przed odjazdem.
                tram.current_trip = None
                tram.hide(TramStatus.SLEEP)
                yield from self._wait_until(first.departure - timedelta(seconds=APPEAR_BEFORE_S))

            tram.current_trip = trip
            tram.route_pos = route_offset
            tram.trip_offset = route_offset
            tram.trip_seg = 0

            # Peron początkowy kursu zajęty przez cały postój przed odjazdem.
            first_lock = first.stop.lock or first.stop.init_simpy(self.env)
            tram.set_static(first.stop, TramStatus.WAIT if first_lock.count > 0 else TramStatus.LAYOVER)
            first_req = first_lock.request()
            yield first_req
            first.stop.reserve(tram)
            tram.set_static(first.stop, TramStatus.LAYOVER)

            if tram.passengers is not None:
                # Wsiadanie na przystanku początkowym kursu tuż przed odjazdem.
                yield from self._wait_until(first.departure - timedelta(seconds=BOARD_BEFORE_DEPARTURE_S))
                tram.set_static(first.stop, TramStatus.DWEL)
                yield self.env.timeout(tram.dwell_delay(self.current_time, first.stop))
                tram.set_static(first.stop, TramStatus.LAYOVER)

            yield from self._wait_until(first.departure)
            tram.delay_s = (self.current_time - first.departure).total_seconds()
            max_delay = tram.delay_s

            first.stop.release()
            first_lock.release(first_req)

            for pos, (prev, nxt) in enumerate(zip(trip.stops, trip.stops[1:]), start=1):
                tram.start_segment(pos - 1)
                yield from self._travel_to(tram, prev.stop, nxt.stop, route_offset + pos)
                tram.delay_s = (self.current_time - nxt.departure).total_seconds()
                max_delay = max(max_delay, tram.delay_s)
            tram.completed_trips.append({
                "trip_id": trip.trip_id,
                "line": trip.line,
                "headsign": trip.headsign,
                "end_delay_s": tram.delay_s,
                "max_delay_s": max_delay,
            })
            tram.trip_seg = len(trip.stops) - 1
            tram.set_static(trip.stops[-1].stop, TramStatus.LAYOVER)
            at_stop = trip.stops[-1].stop
            route_offset += len(trip.stops)

    def _travel_to(self, tram: Tram, current_stop: Stop, next_stop: Stop, next_route_pos: int):
        """Przejazd odcinka z rezerwacją kolejnego bloku przed wjazdem w jego odcinek."""
        road = next((r for r in current_stop.roads if r.end.id == next_stop.id), None)

        stop_lock = next_stop.lock or next_stop.init_simpy(self.env)
        total_travel = road.arrival_delay(self.current_time) if road else 30.0
        blocks = (
            road.get_queue_blocks(self.env, TRAM_SPACING_M, STOP_BUFFER_M)
            if road and road.total_length > 0
            else [(simpy.Resource(self.env, capacity=1), 0.0)]
        )
        pass_stops: list[tuple[Stop, float, float]] = getattr(road, "_pass_through_stops", []) if road else []

        # Zdarzenia wzdłuż odcinka posortowane po dystansie i priorytecie.
        # Każdy blok i slot rezerwujemy na POCZĄTKU jego przedziału (zanim tramwaj dojedzie do końca).
        # Priorytety w tym samym punkcie:
        # 0 = zwolnienie mijanego peronu, 1 = rezerwacja slotu przed mijanym peronem,
        # 2 = rezerwacja mijanego peronu (zwalnia slot 0), 3 = rezerwacja bloku kolejki przed peronem docelowym
        events: list[tuple[float, int, str, object]] = []
        for idx, (p_stop, hit_dist, clear_d) in enumerate(pass_stops):
            p_id = str(p_stop.id)
            approach_d = max(0.0, hit_dist - STOP_BUFFER_M)

            pass_slots: list[tuple[int, float]] = []
            slot_idx = 0
            slot_end = approach_d
            while True:
                slot_start = max(0.0, slot_end - TRAM_SPACING_M)
                pass_slots.append((slot_idx, slot_start))
                if slot_start <= 0.0:
                    break
                slot_end = slot_start
                slot_idx += 1
            pass_slots.reverse()

            for s_idx, s_start in pass_slots:
                events.append((s_start, 1, "acquire_pass_slot", (idx, p_id, s_idx)))
            events.append((approach_d, 2, "acquire_pass_stop", (idx, p_stop)))
            events.append((clear_d, 0, "release_pass_stop", idx))

        prev_block_end = 0.0
        for next_block, block_end in blocks:
            events.append((prev_block_end, 3, "acquire_track", next_block))
            prev_block_end = block_end
        events.sort(key=lambda e: (e[0], e[1]))

        current_dist = 0.0
        block, block_req = None, None
        active_pass_slots: dict[int, tuple[simpy.Resource, object]] = {}
        active_pass_reqs: dict[int, tuple[simpy.Resource, object]] = {}

        try:
            for ev_dist, _, ev_type, payload in events:
                if road and ev_dist > current_dist + 1e-6:
                    yield from self._drive_segment(tram, road, current_dist, ev_dist, total_travel)
                    current_dist = ev_dist

                if ev_type == "acquire_pass_slot":
                    idx, p_id, s_idx = payload
                    slot_res = self._get_approach_slot(p_id, s_idx)
                    prev_slot = active_pass_slots.get(idx)
                    if prev_slot is None or prev_slot[0] is not slot_res:
                        if slot_res.count > 0:
                            pos = road.point_at_distance(current_dist) if road else current_stop
                            tram.set_static(pos, TramStatus.WAIT, road, current_dist)
                        slot_req = slot_res.request()
                        yield slot_req
                        if prev_slot is not None:
                            prev_slot[0].release(prev_slot[1])
                        active_pass_slots[idx] = (slot_res, slot_req)

                elif ev_type == "acquire_pass_stop":
                    idx, p_stop = payload
                    p_lock = p_stop.lock or p_stop.init_simpy(self.env)
                    if p_lock.count > 0:
                        pos = road.point_at_distance(current_dist) if road else current_stop
                        tram.set_static(pos, TramStatus.WAIT, road, current_dist)
                    p_req = p_lock.request()
                    yield p_req
                    active_pass_reqs[idx] = (p_lock, p_req)

                    slot_pair = active_pass_slots.pop(idx, None)
                    if slot_pair is not None:
                        slot_pair[0].release(slot_pair[1])

                elif ev_type == "release_pass_stop":
                    pair = active_pass_reqs.pop(payload, None)
                    if pair is not None:
                        pair[0].release(pair[1])

                elif ev_type == "acquire_track":
                    next_block = payload
                    if next_block.count > 0:
                        pos = road.point_at_distance(current_dist) if road else current_stop
                        tram.set_static(pos, TramStatus.WAIT, road, current_dist)

                    next_req = next_block.request()
                    yield next_req

                    if block is not None:
                        block.release(block_req)
                    block, block_req = next_block, next_req

            # Dojazd do punktu bufora przed przystankiem docelowym (total_length - STOP_BUFFER_M)
            if road and prev_block_end > current_dist + 1e-6:
                yield from self._drive_segment(tram, road, current_dist, prev_block_end, total_travel)
                current_dist = prev_block_end

            for slot_res, slot_req in active_pass_slots.values():
                slot_res.release(slot_req)
            active_pass_slots.clear()

            for p_lock, p_req in active_pass_reqs.values():
                p_lock.release(p_req)
            active_pass_reqs.clear()

            # Oczekiwanie na wolny peron docelowy w odległości STOP_BUFFER_M (45 m) przed słupkiem
            if stop_lock.count > 0:
                pos = road.point_at_distance(current_dist) if road else current_stop
                tram.set_static(pos, TramStatus.WAIT, road, current_dist)

            stop_req = stop_lock.request()
            yield stop_req

            # Wjazd na peron docelowy i zwolnienie ostatniego bloku kolejki
            if block is not None:
                block.release(block_req)
                block = None
            next_stop.reserve(tram)
            tram.current_stop = next_stop
            tram.route_pos = next_route_pos

            if road:
                yield from self._drive_segment(tram, road, current_dist, road.total_length, total_travel)
            else:
                yield self.env.timeout(total_travel)

            # Wymiana pasażerów
            tram.set_static(next_stop, TramStatus.DWEL)
            yield self.env.timeout(tram.dwell_delay(self.current_time, next_stop))

            # Odjazd
            next_stop.release()
            tram.current_stop = None
            stop_lock.release(stop_req)
        finally:
            for slot_res, slot_req in active_pass_slots.values():
                slot_res.release(slot_req)
            for p_lock, p_req in active_pass_reqs.values():
                p_lock.release(p_req)
            if block is not None and block_req is not None:
                block.release(block_req)

    def _sampler_process(self, trams: list[Tram]):
        while True:
            now = self.env.now
            ts = self.current_time
            for tram in trams:
                if tram.status in (TramStatus.SLEEP, TramStatus.TERMINATED):
                    continue
                pt = tram.sync_position(now)
                self.samples.append(
                    SimulationSample(
                        tram_id=str(tram.id),
                        tram_name=tram.name,
                        status=tram.status,
                        timestamp=ts,
                        lat=pt.lat,
                        lon=pt.lon,
                        line=tram.line,
                        headsign=tram.headsign,
                        delay_s=tram.delay_s,
                        trip_id=tram.current_trip.trip_id if tram.current_trip is not None else "",
                        progress=tram.trip_progress,
                        load=tram.load,
                        capacity=tram.capacity,
                        fullness=tram.fullness,
                    )
                )
            yield self.env.timeout(self.target_time_step)