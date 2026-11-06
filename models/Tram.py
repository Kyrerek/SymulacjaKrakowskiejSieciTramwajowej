from models.Point import Point
from models.Stop import Stop
from models.TramStatus import TramStatus

class Tram(Point):
    def __init__(self,id_,route: list[Stop],loop_route_num: int,name="", delay = lambda time,p_num: 30):
        super().__init__(route[0].lat,route[0].lon)
        self.id = id_
        self.name = name
        self.route = route
        self.loop_route_num = loop_route_num
        self.status = TramStatus.TERMINATED
        self.delay_ = delay


    def start(self):
        self.start = TramStatus.ARRIVAL


    def dwell_delay(self,time,stop: Stop):
        return self.delay_(time,stop.enter_tram_num(time,self))