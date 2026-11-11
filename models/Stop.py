from __future__ import annotations
from typing import TYPE_CHECKING
from .Point import Point

if TYPE_CHECKING:
    from .StopRoad import StopRoad
    from .Tram import Tram

class Stop(Point):
    def __init__(self,id_,lat,lon,name="",enter_tram_num = lambda time,tram: 0):
        super().__init__(lat,lon)
        self.id = id_
        self.name = name
        self.is_ocupied = False
        self.enter_tram_num_ = enter_tram_num


    def add_roads(self,roads: list[StopRoad]):
        self.roads = roads


    def enter_tram_num(self,time_sec,tram: Tram):
        return self.enter_tram_num_(time_sec,tram)