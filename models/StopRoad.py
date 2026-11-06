from __future__ import annotations
from typing import TYPE_CHECKING
from .Point import Point

if TYPE_CHECKING:
    from .Stop import Stop

class StopRoad:
    def __init__(self,start: Stop,end: Stop,points: list[Point], delay = lambda t: 400):
        self.start = start
        self.end = end
        self.points = points
        self.delay_ = delay

    def arrival_delay(self,time):
        return self.delay_(time)