"""Mapa i prosta animacja wszystkich linii tramwajowych z GTFS."""
from datetime import date, datetime, time, timedelta
from pathlib import Path
import copy
import hashlib
import random

import folium
from folium.plugins import TimestampedGeoJson
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import tram_data
from models import Tram
from motion import TramMotion
from route_geometry import (
    build_cyclic_route,
    build_stop_roads,
    get_available_lines,
    get_route_geometry,
)
from trainers import StopRoadTrainer


LINE_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9a6324",
    "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]


def samples_to_geojson_features(samples, line_colors: dict[str, str]) -> list[dict]:
    geojson_features = []
    for s in samples:
        tram_id = s["tram_id"] if isinstance(s, dict) else s.tram_id
        tram_name = s["tram_name"] if isinstance(s, dict) else s.tram_name
        status = s["status"] if isinstance(s, dict) else s.status.name
        timestamp = datetime.fromisoformat(s["timestamp"]) if isinstance(s, dict) else s.timestamp
        lat = s["lat"] if isinstance(s, dict) else s.lat
        lon = s["lon"] if isinstance(s, dict) else s.lon
        color = line_colors.get(str(tram_id), "#e6194b")
        geojson_features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],
            },
            "properties": {
                "tram_id": str(tram_id),
                "times": [timestamp.strftime("%Y-%m-%dT%H:%M:%S")],
                "icon": "circle",
                "iconoptions": {
                    "color": color,
                    "fillColor": color,
                    "fillOpacity": 1.0,
                    "radius": 9,
                    "weight": 2,
                },
                "popup": f"<b>{tram_name}</b><br>Status: {status}<br>Czas: {timestamp:%H:%M:%S}",
            },
        })
    return geojson_features


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


@st.cache_resource
def load_trainer() -> StopRoadTrainer | None:
    path = Path("data/trained_samplers.json")
    if not path.exists():
        return None
    return StopRoadTrainer.load(path)


@st.cache_data
def get_available_lines_cached():
    return tuple(get_available_lines())


@st.cache_resource
def build_route_networks_cached(lines: tuple[str, ...]):
    route_networks = {}
    skipped = []
    for line in lines:
        directions = get_route_geometry(line)
        if not directions or len(directions) < 2:
            skipped.append(line)
            continue
        try:
            outbound = build_stop_roads(line, directions[0].direction_id)
            inbound = build_stop_roads(line, directions[1].direction_id)
            cyclic_stops = build_cyclic_route(outbound, inbound)
            route_networks[str(line)] = {
                "outbound": outbound,
                "inbound": inbound,
                "stops": cyclic_stops,
            }
        except ValueError:
            skipped.append(line)
    return route_networks, tuple(skipped)


@st.cache_data
def run_cached_simulation(
    start_iso: str,
    stop_iso: str,
    lines: tuple[str, ...],
    target_time_step: int = 5,
):
    start_time = datetime.fromisoformat(start_iso)
    stop_time = datetime.fromisoformat(stop_iso)
    sim_duration_sec = int((stop_time - start_time).total_seconds())
    route_networks, network_skipped = build_route_networks_cached(lines)
    trainer = load_trainer()
    seed = int(hashlib.sha256(f"{start_iso}|{stop_iso}|{lines}".encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    trams = []

    for line in lines:
        route = route_networks.get(str(line))
        if route is None:
            continue

        cyclic_stops = copy.deepcopy(route["stops"])

        speed_factor = rng.uniform(0.1, 2)
        if trainer is not None:
            trainer.bind_to_route(cyclic_stops)
        else:
            for stop in cyclic_stops:
                for road in stop.roads:
                    est_time = max(80.0, road.total_length / 2.5) * speed_factor
                    road.delay_ = lambda t, d=est_time: d

        random_start = start_time + timedelta(seconds=rng.randint(0, min(3600, sim_duration_sec)))
        random_loops = rng.randint(1, 3)
        trams.append(Tram(
            id_=line,
            route=cyclic_stops,
            start_time=random_start,
            loop_route_num=random_loops,
            name=f"Linia {line} (start {random_start:%H:%M:%S}, {random_loops}x)",
            delay=lambda t, s: 30,
        ))

    samples = TramMotion(
        start_time=start_time,
        stop_time=stop_time,
        target_time_step=target_time_step,
    ).run_trams(trams)

    return (
        [
            {
                "tram_id": sample.tram_id,
                "tram_name": sample.tram_name,
                "status": sample.status.name,
                "timestamp": sample.timestamp.isoformat(),
                "lat": sample.lat,
                "lon": sample.lon,
            }
            for sample in samples
        ],
        tuple(line for line in lines if str(line) in route_networks),
        network_skipped,
    )


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

lines = [line for line in get_available_lines_cached() if str(line) in active_lines]
if not lines:
    st.error("Brak kursujących linii w wybranym okresie.")
    st.stop()

route_networks, route_skipped = build_route_networks_cached(tuple(lines))

display_roads = []
display_stops = []
loaded_lines = []
line_colors = {}

for line_index, line in enumerate(lines):
    route = route_networks.get(str(line))
    if route is None:
        continue

    outbound = route["outbound"]
    inbound = route["inbound"]
    cyclic_stops = route["stops"]
    color = LINE_COLORS[line_index % len(LINE_COLORS)]
    line_colors[str(line)] = color

    display_roads.extend((road, line, color) for road in outbound.roads + inbound.roads)
    display_stops.extend(cyclic_stops)
    loaded_lines.append(line)

if not loaded_lines:
    st.error("Nie udało się zbudować żadnej trasy do animacji.")
    st.stop()

sample_dicts, loaded_lines_tuple, network_skipped = run_cached_simulation(
    start_time.isoformat(),
    stop_time.isoformat(),
    tuple(lines),
    5,
)
loaded_lines = list(loaded_lines_tuple)
skipped_lines = list(dict.fromkeys([*route_skipped, *network_skipped]))
features = samples_to_geojson_features(sample_dicts, line_colors)

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
    period="PT5S",
    duration="PT0S",
    add_last_point=True,
    auto_play=False,
    loop=False,
    max_speed=60,
    loop_button=True,
    time_slider_drag_update=True,
).add_to(m)

st.caption(
    f"{start_time:%d.%m.%Y %H:%M} – {stop_time:%d.%m.%Y %H:%M} · "
    f"{len(loaded_lines)} linii · {len(features)} punktów animacji"
)
if skipped_lines:
    st.warning("Pominięte linie bez kompletnej trasy w obu kierunkach: " + ", ".join(skipped_lines))

clean_css = """
<style>
.timecontrol-speed::before {
    display: none !important;
}
</style>
"""
m.get_root().html.add_child(folium.Element(clean_css))

st_folium(m, width="100%", height=700, returned_objects=[])