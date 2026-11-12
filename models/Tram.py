from models.Point import Point
from models.Stop import Stop
from models.TramStatus import TramStatus

class Tram(Point):
    def __init__(
        self,
        id_,
        route: list[Stop],
        start_time: int,
        loop_route_num: int,
        name="",
        delay = lambda time,
        p_num: 30
    ):
        super().__init__(route[0].lat,route[0].lon)
        self.id = id_
        self.name = name
        self.route = route
        self.start_time = start_time # sekundy od startu symulacji
        self.loop_route_num = loop_route_num
        self.status = TramStatus.SLEEP
        self.delay_ = delay
        self.current_stop = None


    def check_start(self,time_sec):
        if self.start_time > time_sec and self.status == TramStatus.SLEEP:
            self.status = TramStatus.ARRIVAL


    def dwell_delay(self,time_sec,stop: Stop):
        return self.delay_(time_sec,stop.enter_tram_num(time_sec,self))