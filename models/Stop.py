from __future__ import annotations
from typing import TYPE_CHECKING
from .Point import Point

if TYPE_CHECKING:
    from .StopRoad import StopRoad

class Stop(Point):
    def __init__(self,id_,lat,lon,name=""):
        super().__init__(lat,lon)
        self.id = id_
        self.name = name
        self.is_ocupied = False


    def add_roads(self,roads: list[StopRoad]):
        self.roads = roads