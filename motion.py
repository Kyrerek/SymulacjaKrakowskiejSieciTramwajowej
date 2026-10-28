from datetime import timedelta, datetime
from models import Tram, TramStatus
import simpy

class TramMotion:
    def __init__(self, start_time: datetime, steps: int, travel_minutes: int, env=simpy.Environment()):
        self.env = env
        self.start_time = start_time
        self.steps = steps
        self.travel_minutes = travel_minutes
        self.features = []

    def run_trams(self, trams: list[Tram]):
        for tram in trams:
            self.env.process(self._run_tram(tram))
        self.env.run()
        return self.features

    def _run_tram(self, tram: Tram):
        """
        Symulacja z customowym czasem (offsetem w sekundach) dla każdego punktu.
        Format: (longitude, latitude, sekundy_od_startu)
        """
        route_data = [
            (19.9450, 50.0647, 0),    # Dworzec Główny (Start: 0. sekunda)
            (19.9430, 50.0652, 20),   # Punkt pośredni 1 (po 20 sekundach)
            (19.9412, 50.0655, 50),   # Zakręt przy Basztowej (po 50 sekundach)
            (19.9390, 50.0652, 70),   # Punkt pośredni 2 (po 70 sekundach)
            (19.9370, 50.0648, 100),  # Basztowa LOT (po 100 sekundach)
            (19.9350, 50.0640, 130),  # Punkt pośredni 3 (po 130 sekundach)
            (19.9332, 50.0632, 160)   # Teatr Bagatela (Koniec: po 160 sekundach)
        ]

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

        tram.status = TramStatus.TERMINATED