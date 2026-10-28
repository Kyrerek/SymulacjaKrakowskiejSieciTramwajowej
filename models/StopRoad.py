from __future__ import annotations
from typing import TYPE_CHECKING
from .Point import Point

if TYPE_CHECKING:
    from .Stop import Stop

class StopRoad:
    def __init__(self,start: Stop,end: Stop,points: list[Point]):
        self.start = start
        self.end = end
        self.points = points