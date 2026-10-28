from models.Point import Point
from models.Stop import Stop
from models.TramStatus import TramStatus

class Tram(Point):
    def __init__(self,id_,route: list[Stop],name=""):
        super().__init__(route[0].lat,route[0].lon)
        self.id = id_
        self.name = name
        self.route = route
        self.curStopIdx = 0
        self.status = TramStatus.TERMINATED


    def start(self):
        self.start = TramStatus.ARRIVAL