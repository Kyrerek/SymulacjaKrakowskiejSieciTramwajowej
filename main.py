from datetime import datetime, timedelta
import folium
from folium.plugins import TimestampedGeoJson
import simpy
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="Symulacja Tramwaju Kraków", layout="wide")
st.title("Symulacja przejazdu tramwaju: A ➔ B")

# 1. Definicja przystanków
STOP_A = {"name": "Dworzec Główny", "lat": 50.0647, "lon": 19.9450}
STOP_B = {"name": "Teatr Bagatela", "lat": 50.0632, "lon": 19.9332}
TRAVEL_MINUTES = 5
KROKI = 30

# 2. Obliczenia w SimPy
env = simpy.Environment()
features = []
start_time = datetime(2026, 10, 27, 12, 0, 0)


def symulacja_tramwaju(env):
  for i in range(KROKI + 1):
    postep = i / KROKI
    lat = STOP_A["lat"] + postep * (STOP_B["lat"] - STOP_A["lat"])
    lon = STOP_A["lon"] + postep * (STOP_B["lon"] - STOP_A["lon"])

    t_current = start_time + timedelta(seconds=i * 10)
    time_str = t_current.strftime("%Y-%m-%dT%H:%M:%SZ")

    features.append({
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
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
            "popup": (
                f"<b>Tramwaj Linia 50</b><br>Postęp: {int(postep * 100)}%<br>Czas:"
                f" {t_current.strftime('%H:%M:%S')}"
            ),
        },
    })
    yield env.timeout(TRAVEL_MINUTES / KROKI)


env.process(symulacja_tramwaju(env))
env.run()

geojson_data = {"type": "FeatureCollection", "features": features}

# 3. Tworzenie mapy Folium
m = folium.Map(location=[50.0640, 19.9390], zoom_start=15)

# Torowisko i przystanki
folium.PolyLine(
    [[STOP_A["lat"], STOP_A["lon"]], [STOP_B["lat"], STOP_B["lon"]]],
    color="#0288d1",
    weight=4,
    opacity=0.5,
    tooltip="Trasa przejazdu",
).add_to(m)

folium.Marker(
    [STOP_A["lat"], STOP_A["lon"]],
    popup=f"Start: {STOP_A['name']}",
    tooltip=STOP_A["name"],
    icon=folium.Icon(color="green", icon="play", prefix="fa"),
).add_to(m)

folium.Marker(
    [STOP_B["lat"], STOP_B["lon"]],
    popup=f"Koniec: {STOP_B['name']}",
    tooltip=STOP_B["name"],
    icon=folium.Icon(color="red", icon="stop", prefix="fa"),
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
    date_options="HH:mm:ss",
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