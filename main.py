"""Mapa i prosta animacja wszystkich linii tramwajowych z GTFS."""
from datetime import date, datetime, time, timedelta

import folium
from folium.plugins import TimestampedGeoJson
import streamlit as st
from streamlit_folium import st_folium
import random

import pandas as pd

import tram_data

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

@st.cache_data
def feed_date_range():
    cal_dates = tram_data._read("calendar_dates.txt", tram_data.DATA_DIR)
    cal = tram_data._read("calendar.txt", tram_data.DATA_DIR)
    values = []
    if cal_dates is not None and not cal_dates.empty:
        values += list(cal_dates["date"])
    if cal is not None and not cal.empty:
        values += list(cal["start_date"]) + list(cal["end_date"])
    if not values:
        today = date.today()
        return today, today
    parsed = pd.to_datetime(pd.Series(values), format="%Y%m%d")
    return parsed.min().date(), parsed.max().date()


@st.cache_data
def lines_active_in(days):
    trips = tram_data._line_trips(None, list(days), tram_data.DATA_DIR)
    return set(trips["route_short_name"].astype(str))


feed_min, feed_max = feed_date_range()
default_day = min(max(date(2026, 10, 27), feed_min), feed_max)

with st.sidebar:
    st.header("Czas symulacji")
    period_mode = st.radio("Zakres", ["Jeden dzień", "Okres"], horizontal=True)
    if period_mode == "Jeden dzień":
        sim_day = st.date_input(
            "Data", value=default_day, min_value=feed_min, max_value=feed_max, format="DD.MM.YYYY"
        )
        col_from, col_to = st.columns(2)
        with col_from:
            from_time = st.time_input("Od", value=time(12, 0), step=timedelta(minutes=15))
        with col_to:
            to_time = st.time_input("Do", value=time(14, 0), step=timedelta(minutes=15))
        start_time = datetime.combine(sim_day, from_time)
        stop_time = datetime.combine(sim_day, to_time)
    else:
        date_range = st.date_input(
            "Daty",
            value=(default_day, min(default_day + timedelta(days=1), feed_max)),
            min_value=feed_min,
            max_value=feed_max,
            format="DD.MM.YYYY",
        )
        if not isinstance(date_range, (list, tuple)) or len(date_range) != 2:
            st.info("Wybierz datę początkową i końcową.")
            st.stop()
        start_day, end_day = date_range
        col_from, col_to = st.columns(2)
        with col_from:
            from_time = st.time_input("Godzina startu", value=time(12, 0), step=timedelta(minutes=15))
        with col_to:
            to_time = st.time_input("Godzina końca", value=time(12, 0), step=timedelta(minutes=15))
        start_time = datetime.combine(start_day, from_time)
        stop_time = datetime.combine(end_day, to_time)

    st.header("Widok mapy")
    show_stops = st.checkbox("Pokaż przystanki", value=False)

if stop_time <= start_time:
    st.error("Koniec symulacji musi być późniejszy niż początek.")
    st.stop()

sim_days = tuple(
    pd.Timestamp(start_time.date() + timedelta(days=i))
    for i in range((stop_time.date() - start_time.date()).days + 1)
)
active_lines = lines_active_in(sim_days)

lines = [line for line in get_available_lines() if str(line) in active_lines]
if not lines:
    st.error("Brak kursujących linii w wybranym okresie.")
    st.stop()

sim_duration_sec = int((stop_time - start_time).total_seconds())
features = []
display_roads = []
display_stops = []
loaded_lines = []
skipped_lines = []
trams = []
line_colors = {}

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
    line_colors[str(line)] = color


    random_start_sec = random.randint(0, min(3600, sim_duration_sec))
    random_loops = random.randint(1, 3)

    tram = Tram(
        id_=line,
        route=cyclic_stops,
        start_time=random_start_sec,
        loop_route_num=random_loops,
        name=f"Linia {line} (start +{random_start_sec // 60} min, {random_loops}x)",
    )

    trams.append(tram)
    display_roads.extend((road, line, color) for road in outbound.roads + inbound.roads)
    display_stops.extend(cyclic_stops)
    loaded_lines.append(line)

features = TramMotion(start_time=start_time, stop_time=stop_time, target_time_step=10).run_trams(trams)

for feature in features:
    props = feature.get("properties", {})
    # Dopasowanie po tram_id / id lub z treści popup/tooltip (np. "Linia 52")
    raw_id = str(props.get("tram_id") or props.get("id") or "")
    if raw_id not in line_colors:
        label = str(props.get("popup") or props.get("tooltip") or "")
        raw_id = next((ln for ln in line_colors if f"Linia {ln}" in label), "")

    tram_color = line_colors.get(raw_id, "#e6194b")
    if "iconoptions" in props:
        props["iconoptions"]["color"] = tram_color
        props["iconoptions"]["fillColor"] = tram_color

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
    f"{start_time:%d.%m.%Y %H:%M} – {stop_time:%d.%m.%Y %H:%M} · "
    f"{len(loaded_lines)} linii · {len(features)} punktów animacji · "
    "czas przejazdu i postoje są obecnie stałe w motion.py"
)
if skipped_lines:
    st.warning("Pominięte linie bez kompletnej trasy w obu kierunkach: " + ", ".join(skipped_lines))


# Minimalny, czysty CSS usuwający problematyczny element ::before
clean_css = """
<style>
.timecontrol-speed::before {
    display: none !important;
}
</style>
"""
m.get_root().html.add_child(folium.Element(clean_css))



st_folium(m, width="100%", height=700, returned_objects=[])
