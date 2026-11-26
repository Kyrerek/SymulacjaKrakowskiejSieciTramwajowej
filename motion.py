from dataclasses import dataclass
from datetime import datetime, timedelta
import simpy

from models import Stop, StopRoad, Tram, TramStatus
from schedule import APPEAR_BEFORE_S

TRAM_SPACING_M = 45.0
STOP_BUFFER_M = 45.0


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

    @property
    def current_time(self) -> datetime:
        return self.start_time + timedelta(seconds=self.env.now)

    def run_trams(self, trams: list[Tram]) -> list[SimulationSample]:
        self.samples = []
        max_time = (self.stop_time - self.start_time).total_seconds()
        if self.env.now >= max_time:
            return self.samples

        # Wspólne zasoby peronów oraz bloków kolejki dla wszystkich linii
        shared_stop_locks: dict[str, simpy.Resource] = {}
        shared_road_blocks: dict[tuple[str, str], list[tuple[simpy.Resource, float]]] = {}

        for tram in trams:
            for stop in tram.route:
                stop_id = str(stop.id)

                if stop_id not in shared_stop_locks:
                    shared_stop_locks[stop_id] = stop.init_simpy(self.env)
                else:
                    stop.lock = shared_stop_locks[stop_id]

                stop.occupant = None

                for road in stop.roads:
                    road_key = (str(road.start.id), str(road.end.id))
                    if road_key not in shared_road_blocks:
                        shared_road_blocks[road_key] = road.init_queue_blocks(
                            self.env,
                            TRAM_SPACING_M,
                            STOP_BUFFER_M,
                        )
                    else:
                        road._queue_blocks = shared_road_blocks[road_key]

            self.env.process(self._start_tram(tram))

        self.env.process(self._sampler_process(trams))
        self.env.run(until=max_time)
        return self.samples

    def _start_tram(self, tram: Tram):
        wait_time = (tram.start_time - self.current_time).total_seconds()
        if wait_time < 0:
            tram.status = TramStatus.TERMINATED
            return
        if wait_time > 0:
            yield self.env.timeout(wait_time)

        yield from self._run_tram(tram)
        tram.status = TramStatus.TERMINATED

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
                tram.status = TramStatus.SLEEP
                tram.current_trip = None
                yield from self._wait_until(first.departure - timedelta(seconds=APPEAR_BEFORE_S))

            tram.current_trip = trip
            tram.route_pos = route_offset
            tram.set_static(first.stop, TramStatus.WAIT)
            yield from self._wait_until(first.departure)
            tram.delay_s = (self.current_time - first.departure).total_seconds()

            for pos, (prev, nxt) in enumerate(zip(trip.stops, trip.stops[1:]), start=1):
                yield from self._travel_to(tram, prev.stop, nxt.stop, route_offset + pos)
                tram.delay_s = (self.current_time - nxt.departure).total_seconds()
            at_stop = trip.stops[-1].stop
            route_offset += len(trip.stops)

    def _travel_to(self, tram: Tram, current_stop: Stop, next_stop: Stop, next_route_pos: int):
        """Przejazd odcinka z kolejką przed przystankiem, wjazd na peron i postój."""
        road = next((r for r in current_stop.roads if r.end.id == next_stop.id), None)

        stop_lock = next_stop.lock or next_stop.init_simpy(self.env)
        total_travel = road.arrival_delay(self.current_time) if road else 30.0
        blocks = (
            road.get_queue_blocks(self.env, TRAM_SPACING_M, STOP_BUFFER_M)
            if road and road.total_length > 0
            else [(simpy.Resource(self.env, capacity=1), 0.0)]
        )

        current_dist = 0.0
        block, block_req = None, None

        # Przejazd przez kolejne bloki kolejki przed przystankiem
        for next_block, next_dist in blocks:
            if next_block.count > 0:
                pos = road.point_at_distance(current_dist) if road else current_stop
                tram.set_static(pos, TramStatus.WAIT)

            next_req = next_block.request()
            yield next_req

            if block is not None:
                block.release(block_req)
            block, block_req = next_block, next_req

            if road:
                yield from self._drive_segment(tram, road, current_dist, next_dist, total_travel)
            current_dist = next_dist

        # Oczekiwanie na wolny peron
        if stop_lock.count > 0:
            pos = road.point_at_distance(current_dist) if road else current_stop
            tram.set_static(pos, TramStatus.WAIT)

        stop_req = stop_lock.request()
        yield stop_req

        # Wjazd na peron i zwolnienie ostatniego bloku kolejki
        block.release(block_req)
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
                        load=tram.load,
                        capacity=tram.capacity,
                        fullness=tram.fullness,
                    )
                )
            yield self.env.timeout(self.target_time_step)