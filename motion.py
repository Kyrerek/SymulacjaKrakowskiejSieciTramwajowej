from datetime import timedelta, datetime
from models import Tram, Stop, Point, StopRoad, TramStatus
import simpy


TRAM_SPACING_M = 35.0  # długość jednego bloku kolejki (odstęp między tramwajami)
STOP_BUFFER_M = 20.0   # odległość pierwszego bloku kolejki od przystanku


class TramMotion:
    def __init__(self, start_time: datetime, stop_time: datetime, target_time_step=10, env=None):
        self.env = simpy.Environment() if env is None else env
        self.start_time = start_time
        self.stop_time = stop_time
        self.target_time_step = float(target_time_step)
        self.features = []
        self.stop_locks = {}    # stop.id -> Resource(1) dla samego peronu
        self.queue_blocks = {}  # (stop.id, slot_idx) -> Resource(1) dla bloków przed przystankiem
        self._tram_states = {}  # jedyne źródło pozycji dla samplera

    def _ensure_stop_lock(self, stop: Stop) -> simpy.Resource:
        if stop.id not in self.stop_locks:
            self.stop_locks[stop.id] = simpy.Resource(self.env, capacity=1)
        stop.lock = self.stop_locks[stop.id]
        return self.stop_locks[stop.id]

    def _get_queue_block(self, stop_id, slot_idx: int) -> simpy.Resource:
        key = (stop_id, slot_idx)
        if key not in self.queue_blocks:
            self.queue_blocks[key] = simpy.Resource(self.env, capacity=1)
        return self.queue_blocks[key]

    def _slot_distance(self, total_length: float, slot_idx: int) -> float:
        offset = STOP_BUFFER_M + slot_idx * TRAM_SPACING_M
        return max(0.0, total_length - offset)

    def run_trams(self, trams: list[Tram]):
        self.features = []
        self.queue_blocks = {}
        self._tram_states = {}
        max_time = (self.stop_time - self.start_time).total_seconds()
        if self.env.now >= max_time:
            return self.features

        for tram in trams:
            for stop in tram.route:
                self._ensure_stop_lock(stop)
            self.env.process(self._start_tram(tram))

        self.env.process(self._sampler_process(trams))
        self.env.run(until=max_time)
        return self.features

    # -------------------------------------------------------------------------
    # MASZYNA STANÓW TRAMWAJU (czysta logika SimPy bez emisji klatek)
    # -------------------------------------------------------------------------

    def _start_tram(self, tram: Tram):
        wait_time = tram.start_time - self.env.now
        if wait_time < 0:
            tram.status = TramStatus.TERMINATED
            return
        if wait_time > 0:
            yield self.env.timeout(wait_time)

        yield from self._run_tram(tram)
        tram.status = TramStatus.TERMINATED
        self._tram_states.pop(tram.id, None)

    def _drive_to(self, tram: Tram, points: list[Point], cum: list[float], d0: float, d1: float, total_travel: float):
        total_length = cum[-1]
        if total_length <= 0 or d1 <= d0 + 1e-6:
            return
        duration = total_travel * ((d1 - d0) / total_length)
        tram.status = TramStatus.ARRIVAL
        self._tram_states[tram.id] = {
            "mode": "moving",
            "points": points,
            "cum": cum,
            "d0": d0,
            "d1": d1,
            "t0": self.env.now,
            "t1": self.env.now + duration,
        }
        yield self.env.timeout(duration)

    def _hold_at(self, tram: Tram, point: Point, status: TramStatus):
        tram.status = status
        self._tram_states[tram.id] = {
            "mode": "static",
            "point": point,
        }

    def _run_tram(self, tram: Tram):
        for _ in range(tram.loop_route_num):
            for idx in range(len(tram.route)):
                current_stop = tram.route[idx]
                next_stop = tram.route[(idx + 1) % len(tram.route)]
                road = next((r for r in current_stop.roads if r.end.id == next_stop.id), None)

                stop_lock = self._ensure_stop_lock(next_stop)
                total_travel = road.arrival_delay(self.env.now) if road else 30.0
                cum = self._cumulative_distances(road.points) if road and len(road.points) >= 2 else [0.0]
                total_length = cum[-1]

                # Liczba bloków obejmująca całą długość odcinka aż do poprzedniego przystanku
                num_slots = max(1, int(max(0.0, total_length - STOP_BUFFER_M) // TRAM_SPACING_M) + 2)
                current_dist = 0.0
                block = None
                block_req = None

                # STAN 1 & 2: Przejazd przez kolejne bloki odcinka (od najdalszego do slot 0 = 20 m przed peronem)
                for slot in range(num_slots - 1, -1, -1):
                    next_block = self._get_queue_block(next_stop.id, slot)

                    # Jeśli blok przed nami jest zajęty, czekamy w miejscu przed nim
                    if next_block.count > 0:
                        pos = self._point_at_distance(road.points, cum, current_dist) if total_length > 0 else current_stop
                        self._hold_at(tram, pos, TramStatus.WAIT)

                    next_req = next_block.request()
                    yield next_req

                    # Zajęliśmy blok przed nami -> zwalniamy poprzedni i jedziemy do końca nowego bloku
                    if block is not None:
                        block.release(block_req)
                    block, block_req = next_block, next_req

                    next_dist = self._slot_distance(total_length, slot)
                    if total_length > 0:
                        yield from self._drive_to(tram, road.points, cum, current_dist, next_dist, total_travel)
                    current_dist = next_dist

                # STAN 3: Stoimy 20 m przed przystankiem (slot 0) i czekamy na wolny peron
                if stop_lock.count > 0:
                    pos = self._point_at_distance(road.points, cum, current_dist) if total_length > 0 else current_stop
                    self._hold_at(tram, pos, TramStatus.WAIT)

                stop_req = stop_lock.request()
                yield stop_req

                # STAN 4: Peron wolny -> zwalniamy blok 20 m przed przystankiem i wjeżdżamy na peron
                block.release(block_req)
                next_stop.reserve(tram)
                tram.current_stop = next_stop

                if total_length > 0:
                    yield from self._drive_to(tram, road.points, cum, current_dist, total_length, total_travel)

                # STAN 5: Postój na przystanku
                self._hold_at(tram, next_stop, TramStatus.DWEL)
                yield self.env.timeout(tram.dwell_delay(self.env.now, next_stop))

                # STAN 6: Odjazd i zwolnienie peronu
                next_stop.release()
                tram.current_stop = None
                stop_lock.release(stop_req)

    # -------------------------------------------------------------------------
    # SAMPLER (jedyne źródło punktów GeoJSON) i GEOMETRIA
    # -------------------------------------------------------------------------

    def _sampler_process(self, trams: list[Tram]):
        while True:
            now = self.env.now
            for tram in trams:
                if tram.status in (TramStatus.SLEEP, TramStatus.TERMINATED):
                    continue
                pt = self._current_position(tram, now)
                if pt is not None:
                    self._emit_point(tram, pt.lon, pt.lat, now)
            yield self.env.timeout(self.target_time_step)

    def _current_position(self, tram: Tram, now: float) -> Point | None:
        state = self._tram_states.get(tram.id)
        if state is None:
            return None
        if state["mode"] == "static":
            return state["point"]

        duration = state["t1"] - state["t0"]
        progress = 1.0 if duration <= 0 else min(1.0, max(0.0, (now - state["t0"]) / duration))
        dist = state["d0"] + progress * (state["d1"] - state["d0"])
        return self._point_at_distance(state["points"], state["cum"], dist)

    def _cumulative_distances(self, points: list[Point]) -> list[float]:
        cum = [0.0]
        for i in range(1, len(points)):
            cum.append(cum[-1] + points[i - 1].distance_to(points[i]))
        return cum

    def _point_at_distance(self, points: list[Point], cum: list[float], target_dist: float) -> Point:
        if target_dist <= 0.0:
            return points[0]
        if target_dist >= cum[-1]:
            return points[-1]

        for i in range(len(cum) - 1):
            if cum[i] <= target_dist <= cum[i + 1]:
                seg_len = cum[i + 1] - cum[i]
                fraction = (target_dist - cum[i]) / seg_len if seg_len > 0 else 0.0
                p1, p2 = points[i], points[i + 1]
                return Point(
                    p1.lat + fraction * (p2.lat - p1.lat),
                    p1.lon + fraction * (p2.lon - p1.lon),
                )
        return points[-1]

    def _emit_point(self, tram: Tram, lon: float, lat: float, offset_seconds: float):
        t = self.start_time + timedelta(seconds=offset_seconds)
        self.features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],
            },
            "properties": {
                "tram_id": str(tram.id),
                "times": [t.strftime("%Y-%m-%dT%H:%M:%S")],
                "icon": "circle",
                "iconoptions": {
                    "color": "#0288d1",
                    "fillColor": "#0288d1",
                    "fillOpacity": 1.0,
                    "radius": 9,
                    "weight": 2,
                },
                "popup": f"<b>{tram.name}</b><br>Status: {tram.status.name}<br>Czas: {t.strftime('%H:%M:%S')}",
            },
        })