from datetime import timedelta, datetime
from models import Tram, TramStatus, Point
import simpy
import math

class TramMotion:
    def __init__(self, start_time: datetime, target_time_step = 10, env=simpy.Environment()):
        self.env = env
        self.start_time = start_time
        self.target_time_step = target_time_step
        self.features = []

    def run_trams(self, trams: list[Tram]):
        for tram in trams:
            self.env.process(self._run_tram(tram))
        self.env.run()
        return self.features

    def _run_tram(self, tram: Tram):
        curStopIdx = 0
        time = 0
        for i in range(tram.loop_route_num):
            for idx in range(len(tram.route)):
                time = yield from self._run_route_data(tram,self._generate_arrival_data(tram,idx,time))
                time = yield from self._run_route_data(tram,self._generate_dwell_data(tram.route[idx],time))


    def _run_route_data(self, tram,route_data):

        _,_,time = route_data[-1]
  
        for lon, lat, offset_seconds in route_data:
            # Wyliczamy czas na podstawie indywidualnego offsetu dla tego punktu
            t = self.start_time + timedelta(seconds=offset_seconds)
            time_str = t.strftime("%Y-%m-%dT%H:%M:%S")
            
            self.features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": {
                    "times": [time_str],
                    "icon": "circle",
                    "iconoptions": {
                        "color": "#0288d1",
                        "fillColor": "#0288d1",
                        "fillOpacity": 1.0,
                        "radius": 9,
                        "weight": 2,
                    },
                    "popup": f"<b>{tram.name}</b><br>Czas: {t.strftime('%H:%M:%S')}"
                }
            })
            
            yield self.env.timeout(1)

        return time


    def _generate_dwell_data(self, point: Point, start_time: float, target_time_step: int = 10):
        dwell_time_sec = 100
        
        if dwell_time_sec <= 0:
            return []

        steps = int(dwell_time_sec // target_time_step)
        dwell_data = []

        for step in range(1, steps + 1):
            time_sec = start_time + step * target_time_step
            dwell_data.append((point.lon, point.lat, time_sec))

        return dwell_data

    def _generate_arrival_data(self, tram: Tram,curStopIdx: int,startTime: int):
        nextStopId = curStopIdx + 1
        if (nextStopId == len(tram.route)):
            nextStopId = 0

        matching_road = list(filter(lambda r: r.end.id == tram.route[nextStopId].id,tram.route[curStopIdx].roads))[0]
        arrival_data = self._route_data_by_duration(matching_road.points,startTime,400)
        return arrival_data



    def _route_data_by_duration(self, path: list[Point], start_time: float, travel_time_sec: float, target_time_step: int = 10):
            if len(path) < 2 or travel_time_sec <= 0:
                return []
    
            cum_distances = [0.0]
            for i in range(1, len(path)):
                d = path[i-1].distance_to(path[i])
                cum_distances.append(cum_distances[-1] + d)
    
            total_length = cum_distances[-1]
    
    
            steps = int(travel_time_sec // target_time_step) + 1
            route_data = []
    
            for step in range(steps):
                elapsed_sec = step * target_time_step
                if elapsed_sec > travel_time_sec:
                    break
    
                # Postęp czasowy na odcinku (od 0.0 do 1.0)
                progress = elapsed_sec / travel_time_sec
                target_dist = progress * total_length
                current_timestamp = start_time + elapsed_sec
    
                # Znajdujemy fizyczny punkt na łuku odpowiadający przejechanemu dystansowi
                for i in range(len(cum_distances) - 1):
                    if cum_distances[i] <= target_dist <= cum_distances[i+1]:
                        seg_len = cum_distances[i+1] - cum_distances[i]
                        fraction = (target_dist - cum_distances[i]) / seg_len if seg_len > 0 else 0
    
                        p1, p2 = path[i], path[i+1]
                        lat = p1.lat + fraction * (p2.lat - p1.lat)
                        lon = p1.lon + fraction * (p2.lon - p1.lon)
    
                        route_data.append((lon, lat, current_timestamp))
                        break
    
            return route_data
