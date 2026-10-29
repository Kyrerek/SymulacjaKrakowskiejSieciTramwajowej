from datetime import datetime
import folium
from folium.plugins import TimestampedGeoJson
import simpy
import streamlit as st
from streamlit_folium import st_folium

from models import Stop,StopRoad,Tram,Point
from motion import TramMotion

st.set_page_config(page_title="Symulacja Tramwaju Kraków", layout="wide")
st.title("Symulacja przejazdu tramwaju: A ➔ B")

# 1. Definicja przystanków
stops = [
  Stop(id_=0,lat=50.0647,lon=19.9450,name="Dworzec Główny"),
  Stop(id_=1,lat=50.0632,lon=19.9332,name="Teatr Bagatela")
]

# 2. Punkty pośrednie reprezentujące torowisko (przez ul. Basztową)
path_a_to_b = [
    Point(lat=50.0647, lon=19.9450),  # Dworzec Główny (Start)
    Point(lat=50.0655, lon=19.9412),  # Zakręt przy Basztowej
    Point(lat=50.0648, lon=19.9370),  # Basztowa LOT
    Point(lat=50.0632, lon=19.9332),  # Teatr Bagatela (Koniec)
]

stop_roads = [
  StopRoad(start=stops[0],end=stops[1],points=path_a_to_b),
  StopRoad(start=stops[1],end=stops[0],points=list(reversed(path_a_to_b)))
]

# połączenie dróg z przystankami
for stop in stops:
  matching_roads = list(filter(lambda stop_road: stop_road.start.id == stop.id,stop_roads))
  stop.add_roads(matching_roads)


# tramwaje
trams = [
  Tram(id_=0,route=[stops[0],stops[1]],loop_route_num=3,name="Tram")
]

trams[0].start()

tram_manager = TramMotion(
    start_time=datetime(2026, 10, 27, 12, 0, 0),
)

features = tram_manager.run_trams(trams)

geojson_data = {"type": "FeatureCollection", "features": features}

# 3. Tworzenie mapy Folium
m = folium.Map(location=[50.0640, 19.9390], zoom_start=15)

# Torowisko i przystanki
for road in stop_roads:
  points_coords = [[p.lat, p.lon] for p in road.points]
  folium.PolyLine(
      points_coords,
      color="#0288d1",
      weight=4,
      opacity=0.5,
      tooltip="Trasa przejazdu",
  ).add_to(m)


for stop in stops:
    folium.Marker(
        location=[stop.lat, stop.lon],
        popup=f"Przystanek: {stop.name}",
        tooltip=stop.name,
        icon=folium.MarkerCluster() if False else folium.Icon(color="green", icon="info-sign")
    ).add_to(m)


# 4. Natywna wtyczka animacji (maksymalna prędkość 60 FPS)
TimestampedGeoJson(
    geojson_data,
    period="PT10S",
    duration="PT9S",
    add_last_point=False,
    auto_play=False,
    loop=False,
    max_speed=60,
    loop_button=True,
    time_slider_drag_update=True,
).add_to(m)

# 5. Minimalny, czysty CSS usuwający problematyczny element ::before
clean_css = """
<style>
.timecontrol-speed::before {
    display: none !important;
}
</style>
"""
m.get_root().html.add_child(folium.Element(clean_css))

# 6. Renderowanie w Streamlit
st_folium(m, width="100%", height=530)