import math

class Point:
    def __init__(self,lat,lon):
        self.lat = lat
        self.lon = lon

    def distance_to(self, other: "Point") -> float:
        """Oblicza odległość w metrach do innego punktu (Haversine)."""
        EARTH_RADIUS_M = 6_371_000

        phi1 = math.radians(self.lat)
        phi2 = math.radians(other.lat)
        dphi = math.radians(other.lat - self.lat)
        dlam = math.radians(other.lon - self.lon)

        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))
