"""Mapa i prosta animacja wszystkich linii tramwajowych z GTFS."""
from datetime import datetime

import folium
from folium.plugins import TimestampedGeoJson
import streamlit as st
from streamlit_folium import st_folium

from models import StopRoad, Tram
from motion import TramMotion
from route_geometry import build_stop_roads, get_available_lines, get_route_geometry


LINE_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9a6324",
    "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]


def build_cyclic_route(outbound, inbound):
    """Łączy kierunek tam i z powrotem w cykl wymagany przez motion.py."""
    if len(inbound.stops) < 2 or not inbound.roads:
        raise ValueError("Brak pełnej trasy powrotnej")

    stops = outbound.stops + inbound.stops[1:-1]
    first_return = inbound.roads[0]
    return_target = inbound.stops[1] if len(inbound.stops) > 2 else outbound.stops[0]
    outbound.stops[-1].roads.append(
        StopRoad(outbound.stops[-1], return_target, first_return.points)
    )

    if len(inbound.stops) > 2:
        last_return = inbound.roads[-1]
        stops[-1].roads.append(StopRoad(stops[-1], outbound.stops[0], last_return.points))
    return stops


st.set_page_config(page_title="Symulacja sieci tramwajowej Krakowa", layout="wide")
st.title("Symulacja krakowskiej sieci tramwajowej")
st.caption("Wszystkie linie: rzeczywista geometria GTFS i demonstracyjna animacja jazdy.")

with st.sidebar:
    st.header("Widok mapy")
    show_stops = st.checkbox("Pokaż przystanki", value=False)

lines = get_available_lines()
if not lines:
    st.error("Nie znaleziono linii w data/GTFS_KRK_T/routes.txt.")
    st.stop()

start_time = datetime(2026, 10, 27, 12, 0, 0)
features = []
display_roads = []
display_stops = []
loaded_lines = []
skipped_lines = []

for line_index, line in enumerate(lines):
    directions = get_route_geometry(line)
    if not directions or len(directions) < 2:
        skipped_lines.append(line)
        continue

    try:
        outbound = build_stop_roads(line, directions[0].direction_id)
        inbound = build_stop_roads(line, directions[1].direction_id)
        cyclic_stops = build_cyclic_route(outbound, inbound)
    except ValueError:
        skipped_lines.append(line)
        continue

    color = LINE_COLORS[line_index % len(LINE_COLORS)]
    tram = Tram(
        id_=line,
        route=cyclic_stops,
        loop_route_num=1,
        name=f"Linia {line}",
    )
    line_features = TramMotion(start_time=start_time, target_time_step=10).run_trams([tram])
    for feature in line_features:
        feature["properties"]["iconoptions"]["color"] = color
        feature["properties"]["iconoptions"]["fillColor"] = color
    features.extend(line_features)
    display_roads.extend((road, line, color) for road in outbound.roads + inbound.roads)
    display_stops.extend(cyclic_stops)
    loaded_lines.append(line)

if not loaded_lines:
    st.error("Nie udało się zbudować żadnej trasy do animacji.")
    st.stop()

all_points = [point for road, _, _ in display_roads for point in road.points]
min_lat = min(point.lat for point in all_points)
max_lat = max(point.lat for point in all_points)
min_lon = min(point.lon for point in all_points)
max_lon = max(point.lon for point in all_points)
m = folium.Map(location=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2], control_scale=True)
m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

for road, line, color in display_roads:
    folium.PolyLine(
        locations=[(point.lat, point.lon) for point in road.points],
        color=color,
        weight=3,
        opacity=0.7,
        tooltip=f"Linia {line}",
    ).add_to(m)

if show_stops:
    seen_stop_ids = set()
    for stop in display_stops:
        if stop.id in seen_stop_ids:
            continue
        seen_stop_ids.add(stop.id)
        folium.CircleMarker(
            location=(stop.lat, stop.lon),
            radius=3,
            color="#1f2937",
            fill=True,
            fill_color="#ffffff",
            fill_opacity=0.9,
            tooltip=stop.name,
        ).add_to(m)

TimestampedGeoJson(
    {"type": "FeatureCollection", "features": features},
    period="PT10S",
    duration="PT9S",
    add_last_point=True,
    auto_play=False,
    loop=False,
    max_speed=60,
    loop_button=True,
    time_slider_drag_update=True,
).add_to(m)

st.caption(
    f"{len(loaded_lines)} linii · {len(features)} punktów animacji · "
    "czas przejazdu i postoje są obecnie stałe w motion.py"
)
if skipped_lines:
    st.warning("Pominięte linie bez kompletnej trasy w obu kierunkach: " + ", ".join(skipped_lines))

st_folium(m, width="100%", height=700, returned_objects=[])
