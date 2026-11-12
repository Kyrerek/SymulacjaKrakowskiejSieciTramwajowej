from datetime import timedelta, datetime
from models import Tram, Stop, Point,StopRoad
import simpy
import math

class TramMotion:
    def __init__(self, start_time: datetime, stop_time: datetime, target_time_step = 10, env=simpy.Environment()):
        self.env = env
        self.start_time = start_time
        self.stop_time = stop_time
        self.target_time_step = target_time_step
        self.features = []

    def run_trams(self, trams: list[Tram]):
        max_time = (self.stop_time - self.start_time).total_seconds()
        for tram in trams:
            self.env.process(self._run_tram(tram))
        self.env.run(until=max_time)
        return self.features

    def _run_tram(self, tram: Tram):
        for i in range(tram.loop_route_num):
            for idx in range(len(tram.route)):
                next_stop_idx = (idx + 1) % len(tram.route)
                yield from self._run_route_data(tram,self._generate_arrival_data(tram,idx,self.env.now))
                yield from self._run_route_data(tram,self._generate_dwell_data(tram,tram.route[next_stop_idx],self.env.now))


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
            wait_time = offset_seconds - self.env.now
            if wait_time > 0:
                yield self.env.timeout(wait_time)

        return time


    def _generate_dwell_data(self, tram: Tram, stop: Stop, start_time: float):
        dwell_time_sec = tram.dwell_delay(start_time,stop)
        
        if dwell_time_sec <= 0:
            return []

        steps = int(dwell_time_sec // self.target_time_step)
        dwell_data = []

        for step in range(1, steps + 1):
            time_sec = start_time + step * self.target_time_step
            dwell_data.append((stop.lon, stop.lat, time_sec))

        return dwell_data

    def _generate_arrival_data(self, tram: Tram,curStopIdx: int,startTime: float):
        nextStopId = curStopIdx + 1
        if (nextStopId == len(tram.route)):
            nextStopId = 0

        matching_road: StopRoad = list(filter(lambda r: r.end.id == tram.route[nextStopId].id,tram.route[curStopIdx].roads))[0]
        arrival_data = self._route_data_by_duration(matching_road.points,startTime,matching_road.arrival_delay(startTime))
        return arrival_data



    def _route_data_by_duration(self, path: list[Point], start_time: float, travel_time_sec: float):
            if len(path) < 2 or travel_time_sec <= 0:
                return []
    
            cum_distances = [0.0]
            for i in range(1, len(path)):
                d = path[i-1].distance_to(path[i])
                cum_distances.append(cum_distances[-1] + d)
    
            total_length = cum_distances[-1]
    
    
            steps = int(travel_time_sec // self.target_time_step) + 1
            route_data = []
    
            for step in range(steps):
                elapsed_sec = step * self.target_time_step
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
