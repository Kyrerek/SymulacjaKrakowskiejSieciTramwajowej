from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING
import simpy
from .Point import Point

if TYPE_CHECKING:
    from .StopRoad import StopRoad
    from .Tram import Tram


class Stop(Point):
    def __init__(self, id_, lat: float, lon: float, name: str = "", enter_tram_num=lambda time, tram: 0):
        super().__init__(lat, lon)
        self.id = id_
        self.name = name
        self.roads: list[StopRoad] = []
        self.occupant: Tram | None = None
        self.lock: simpy.Resource | None = None
        self.enter_tram_num_ = enter_tram_num

    @property
    def is_occupied(self) -> bool:
        return self.occupant is not None or (self.lock is not None and self.lock.count > 0)

    def init_simpy(self, env: simpy.Environment) -> simpy.Resource:
        if self.lock is None or self.lock._env is not env:
            self.lock = simpy.Resource(env, capacity=1)
        self.occupant = None
        return self.lock

    def add_roads(self, roads: list[StopRoad]):
        self.roads = roads

    def enter_tram_num(self, time: datetime, tram: Tram):
        return self.enter_tram_num_(time, tram)

    def reserve(self, tram: Tram):
        self.occupant = tram

    def release(self):
        self.occupant = None