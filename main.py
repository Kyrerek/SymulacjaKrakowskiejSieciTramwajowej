"""Mapa i prosta animacja wszystkich linii tramwajowych z GTFS."""
from datetime import date, datetime, time, timedelta
from pathlib import Path
import hashlib
import math

import numpy as np

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import tram_data
from motion import TramMotion
from route_geometry import (
    build_cyclic_route,
    build_stop_roads,
    get_available_lines,
    get_route_geometry,
)
from schedule import build_schedule_network
from sidebar import render_sidebar, render_sidebar_export
from tram_animation import TramAnimation, samples_to_tracks
from trainers import StopRoadTrainer


LINE_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9a6324",
    "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]


# Górny limit liczby próbek animacji - powyżej krok czasowy jest zwiększany,
# żeby strona z mapą nie rosła ponad kilka MB. Animacja i tak interpoluje
# pozycje między próbkami, więc większy krok nie powoduje skoków.
MAX_ANIMATION_SAMPLES = 350_000


st.set_page_config(page_title="Symulacja sieci tramwajowej Krakowa", layout="wide")
st.title("Symulacja krakowskiej sieci tramwajowej")
st.caption(
    "Tramwaje kursują według rozkładu GTFS (jeden pojazd na blok kursów); "
    "czasy przejazdu odcinków losowane z rozkładów wytrenowanych na danych realtime."
)


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


@st.cache_resource
def build_schedule_network_cached(start_iso: str, stop_iso: str, lines: tuple[str, ...]):
    return build_schedule_network(
        datetime.fromisoformat(start_iso),
        datetime.fromisoformat(stop_iso),
        lines,
    )


@st.cache_data
def run_cached_simulation(
    start_iso: str,
    stop_iso: str,
    lines: tuple[str, ...],
    min_time_step: int = 5,
):
    """Symulacja z rozkładu GTFS: jeden tramwaj na blok (ciąg kursów jednego pojazdu)."""
    start_time = datetime.fromisoformat(start_iso)
    stop_time = datetime.fromisoformat(stop_iso)
    sim_duration_sec = (stop_time - start_time).total_seconds()

    # Sieć jest współdzielona z cache; motion.run_trams na starcie zakłada od nowa
    # blokady peronów i kolejek, a odcinki dostają tu świeże funkcje czasu przejazdu.
    network = build_schedule_network_cached(start_iso, stop_iso, lines)
    from_sampler, from_schedule = network.bind_travel_times(load_trainer())
    trams = network.make_trams(start_time, stop_time)

    seed = int(hashlib.sha256(f"{start_iso}|{stop_iso}|{lines}".encode()).hexdigest()[:8], 16)
    np.random.seed(seed)  # samplery czasów przejazdu losują z np.random

    time_step = max(min_time_step, math.ceil(len(trams) * sim_duration_sec / MAX_ANIMATION_SAMPLES))
    time_step = int(math.ceil(time_step / 5) * 5)
    samples = TramMotion(
        start_time=start_time,
        stop_time=stop_time,
        target_time_step=time_step,
    ).run_trams(trams)

    delays = [s.delay_s for s in samples]
    stats = {
        "trams": len(trams),
        "time_step": time_step,
        "samples": len(samples),
        "roads_from_sampler": from_sampler,
        "roads_from_schedule": from_schedule,
        "median_delay_s": float(np.median(delays)) if delays else 0.0,
    }
    simulated_lines = sorted({trip.line for tram in trams for trip in tram.schedule})
    return samples_to_tracks(samples, start_time, time_step), tuple(simulated_lines), stats


feed_min, feed_max = feed_date_range()
default_day = min(max(date(2026, 10, 27), feed_min), feed_max)

cfg = render_sidebar(feed_min, feed_max, default_day)
start_time = cfg.start_time
stop_time = cfg.stop_time

if cfg.loaded_sim is not None:
    lines = [str(line) for line in cfg.loaded_sim["simulated_lines"]]
else:
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
    color = LINE_COLORS[line_index % len(LINE_COLORS)]
    line_colors[str(line)] = color

    route = route_networks.get(str(line))
    if route is None:
        continue

    outbound = route["outbound"]
    inbound = route["inbound"]
    cyclic_stops = route["stops"]

    display_roads.extend((road, line, color) for road in outbound.roads + inbound.roads)
    display_stops.extend(cyclic_stops)
    loaded_lines.append(line)

if cfg.loaded_sim is not None:
    tracks = cfg.loaded_sim["tracks"]
    simulated_lines = tuple(cfg.loaded_sim["simulated_lines"])
    sim_stats = cfg.loaded_sim["stats"]
    skipped_lines = list(cfg.loaded_sim.get("skipped_lines", []))
else:
    if not loaded_lines:
        st.error("Nie udało się zbudować żadnej trasy do animacji.")
        st.stop()

    tracks, simulated_lines, sim_stats = run_cached_simulation(
        start_time.isoformat(),
        stop_time.isoformat(),
        tuple(str(line) for line in lines),
        5,
    )
    skipped_lines = list(route_skipped)

all_coords = [(point.lat, point.lon) for road, _, _ in display_roads for point in road.points]
if not all_coords:
    all_coords = [(lat, lon) for tr in tracks for lat, lon in zip(tr["c"][0::2], tr["c"][1::2])]

min_lat = min(lat for lat, _ in all_coords)
max_lat = max(lat for lat, _ in all_coords)
min_lon = min(lon for _, lon in all_coords)
max_lon = max(lon for _, lon in all_coords)
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

if cfg.show_stops:
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

TramAnimation(
    tracks,
    line_colors,
    start_time=start_time,
    duration_s=(stop_time - start_time).total_seconds(),
).add_to(m)

st.caption(
    f"{start_time:%d.%m.%Y %H:%M} – {stop_time:%d.%m.%Y %H:%M} · "
    f"{len(simulated_lines)} linii · {sim_stats['trams']} tramwajów (bloków) · "
    f"krok próbkowania {sim_stats['time_step']} s · mediana opóźnienia {sim_stats['median_delay_s'] / 60:.1f} min · "
    f"czasy przejazdu: {sim_stats['roads_from_sampler']} odcinków z danych realtime, "
    f"{sim_stats['roads_from_schedule']} z rozkładu"
)
if skipped_lines:
    st.warning("Pominięte linie bez kompletnej trasy w obu kierunkach: " + ", ".join(skipped_lines))

st_folium(m, width="100%", height=700, returned_objects=[])

if cfg.loaded_sim is None:
    render_sidebar_export(
        start_time=start_time,
        stop_time=stop_time,
        simulated_lines=simulated_lines,
        skipped_lines=skipped_lines,
        sim_stats=sim_stats,
        tracks=tracks,
    )