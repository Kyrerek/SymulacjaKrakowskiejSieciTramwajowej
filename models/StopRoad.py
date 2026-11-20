from __future__ import annotations
from typing import TYPE_CHECKING
import simpy
from .Point import Point

if TYPE_CHECKING:
    from .Stop import Stop


class StopRoad:
    def __init__(self, start: Stop, end: Stop, points: list[Point], delay=lambda t: 400):
        self.start = start
        self.end = end
        self.points = points
        self.delay_ = delay

        self.cumulative_distances: list[float] = self._compute_cumulative_distances(points)
        self.total_length: float = self.cumulative_distances[-1] if self.cumulative_distances else 0.0
        self._queue_blocks: list[tuple[simpy.Resource, float]] = []

    @staticmethod
    def _compute_cumulative_distances(points: list[Point]) -> list[float]:
        if not points:
            return [0.0]
        cum = [0.0]
        for i in range(1, len(points)):
            cum.append(cum[-1] + points[i - 1].distance_to(points[i]))
        return cum

    def arrival_delay(self, time_sec) -> float:
        return self.delay_(time_sec)

    def point_at_distance(self, target_dist: float) -> Point:
        """Zwraca współrzędne GPS punktu oddalonego o target_dist metrów od początku odcinka."""
        if not self.points:
            return Point(self.start.lat, self.start.lon)
        if target_dist <= 0.0:
            return self.points[0]
        if target_dist >= self.total_length:
            return self.points[-1]

        cum = self.cumulative_distances
        for i in range(len(cum) - 1):
            if cum[i] <= target_dist <= cum[i + 1]:
                seg_len = cum[i + 1] - cum[i]
                fraction = (target_dist - cum[i]) / seg_len if seg_len > 0 else 0.0
                p1, p2 = self.points[i], self.points[i + 1]
                return Point(
                    p1.lat + fraction * (p2.lat - p1.lat),
                    p1.lon + fraction * (p2.lon - p1.lon),
                )
        return self.points[-1]

    def init_queue_blocks(
        self, env: simpy.Environment, spacing_m: float = 35.0, buffer_m: float = 20.0
    ) -> list[tuple[simpy.Resource, float]]:
        """Tworzy sekwencję bloków od najdalszego do bufora przed peronem (slot 0)."""
        num_slots = max(1, int(max(0.0, self.total_length - buffer_m) // spacing_m) + 2)
        blocks: list[tuple[simpy.Resource, float]] = []
        for slot in range(num_slots - 1, -1, -1):
            offset = buffer_m + slot * spacing_m
            stop_dist = max(0.0, self.total_length - offset)
            blocks.append((simpy.Resource(env, capacity=1), stop_dist))
        self._queue_blocks = blocks
        return blocks

    def get_queue_blocks(
        self, env: simpy.Environment, spacing_m: float = 35.0, buffer_m: float = 20.0
    ) -> list[tuple[simpy.Resource, float]]:
        if not self._queue_blocks or self._queue_blocks[0][0]._env is not env:
            return self.init_queue_blocks(env, spacing_m, buffer_m)
        return self._queue_blocks