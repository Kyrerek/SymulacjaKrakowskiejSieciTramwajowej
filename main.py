"""Mapa i prosta animacja wszystkich linii tramwajowych z GTFS."""
from datetime import date, datetime, time, timedelta
from pathlib import Path
import hashlib
import math

import numpy as np

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

import tram_data
from analytics import (
    PUNCTUAL_EARLY_S,
    PUNCTUAL_LATE_S,
    compute_metrics,
    delay_by_line,
    time_distance_data,
)
from fleet import Fleet
from motion import TramMotion
from passengers import PassengerParams, PassengerSystem
from route_geometry import (
    build_cyclic_route,
    build_stop_roads,
    get_available_lines,
    get_route_geometry,
)
from schedule import build_schedule_network
from sidebar import render_sidebar, render_sidebar_export
from tram_animation import (
    DELAY_CLASSES,
    TramAnimation,
    animation_points,
    ensure_animation_data,
    trams_to_animation,
)
from trainers import StopRoadTrainer


LINE_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9a6324",
    "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]


# Górny limit liczby próbek do statystyk i wykresów - powyżej krok czasowy jest zwiększany.
# Animacja nie korzysta z próbek (odtwarza dziennik ruchu tramwajów), więc krok jej nie psuje.
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
def load_fleet() -> Fleet | None:
    path = Path("data/fleet.json")
    return Fleet.load(path) if path.exists() else None


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


@st.cache_data(show_spinner="Symulacja w toku…")
def run_cached_simulation(
    start_iso: str,
    stop_iso: str,
    lines: tuple[str, ...],
    min_time_step: int = 5,
    passengers_on: bool = False,
    demand_scale: float = 1.0,
):
    """Symulacja z rozkładu GTFS: jeden tramwaj na blok (ciąg kursów jednego pojazdu)."""
    start_time = datetime.fromisoformat(start_iso)
    stop_time = datetime.fromisoformat(stop_iso)
    sim_duration_sec = (stop_time - start_time).total_seconds()

    seed = int(hashlib.sha256(f"{start_iso}|{stop_iso}|{lines}".encode()).hexdigest()[:8], 16)
    np.random.seed(seed)  # samplery czasów przejazdu losują z np.random

    # Sieć jest współdzielona z cache; motion.run_trams na starcie zakłada od nowa
    # blokady peronów i kolejek, a odcinki dostają tu świeże funkcje czasu przejazdu.
    network = build_schedule_network_cached(start_iso, stop_iso, lines)
    from_sampler, from_schedule = network.bind_travel_times(load_trainer())
    system = PassengerSystem(PassengerParams(demand_scale=demand_scale), seed=seed) if passengers_on else None
    trams = network.make_trams(
        start_time,
        stop_time,
        passengers=system,
        fleet=load_fleet() if passengers_on else None,
        rng=np.random.default_rng(seed),
    )

    time_step = max(min_time_step, math.ceil(len(trams) * sim_duration_sec / MAX_ANIMATION_SAMPLES))
    time_step = int(math.ceil(time_step / 5) * 5)
    samples = TramMotion(
        start_time=start_time,
        stop_time=stop_time,
        target_time_step=time_step,
    ).run_trams(trams)

    stats = {
        "trams": len(trams),
        "time_step": time_step,
        "samples": len(samples),
        "roads_from_sampler": from_sampler,
        "roads_from_schedule": from_schedule,
        "passengers": passengers_on,
    }
    metrics = compute_metrics(samples, trams, time_step, system.summary() if system else None)
    tracks = trams_to_animation(trams)  # dziennik ruchu dla animacji (niezależny od kroku próbek)
    simulated_lines = sorted({trip.line for tram in trams for trip in tram.schedule})
    return (
        tracks,
        tuple(simulated_lines),
        stats,
        metrics,
        delay_by_line(samples).to_dict("records"),
        time_distance_data(samples, trams, network, start_time),
    )


def line_sort_key(value: str):
    return (not value.isdigit(), int(value) if value.isdigit() else value)


def delay_color(delay_min: float) -> str:
    for limit, color, _ in DELAY_CLASSES:
        if limit is None or delay_min * 60 < limit:
            return color
    return DELAY_CLASSES[-1][1]


def render_metrics(metrics: dict) -> None:
    cols = st.columns(5)
    cols[0].metric("W ruchu (szczyt)", metrics["max_in_motion"], help="Największa liczba tramwajów jadących jednocześnie.")
    cols[1].metric("Opóźnienie (mediana)", f"{metrics['median_delay_s'] / 60:.1f} min")
    if metrics["punctuality"] is not None:
        cols[2].metric(
            "Punktualne kursy",
            f"{metrics['punctuality']:.0%}",
            help=(
                f"Kursy zakończone od {-PUNCTUAL_EARLY_S // 60} min przed do {PUNCTUAL_LATE_S // 60} min po "
                f"czasie z rozkładu. Za wcześnie: {metrics['early_share']:.0%}, "
                f"za późno: {metrics['late_share']:.0%}."
            ),
        )
    else:
        cols[2].metric("Punktualne kursy", "–")
    cols[3].metric("Kursy", metrics["completed_trips"], help="Kursy zakończone w oknie symulacji.")
    cols[4].metric(
        "W kolejkach",
        f"{metrics['queue_hours']:.1f} h",
        help="Łączny czas, który tramwaje spędziły, czekając przed zajętym przystankiem.",
    )
    pax = metrics.get("passengers")
    if pax:
        cols = st.columns(5)
        cols[0].metric("Wsiadło", f"{pax['boarded']:,}".replace(",", " "))
        cols[1].metric("Śr. oczekiwanie", f"{pax['mean_wait_s'] / 60:.1f} min")
        cols[2].metric("Śr. postój", f"{pax['mean_dwell_s']:.0f} s")
        cols[3].metric("Maks. napełnienie", f"{pax['max_fullness']:.0%}")
        cols[4].metric(
            "Niezabrani",
            pax["left_behind_events"],
            help="Liczba postojów, na których część pasażerów nie zmieściła się do tramwaju.",
        )


def delay_by_line_figure(df: pd.DataFrame, line_colors: dict[str, str]) -> go.Figure:
    fig = go.Figure()
    for row in df.itertuples():
        color = line_colors.get(str(row.line), "#6b7280")
        fig.add_trace(go.Box(
            x=[str(row.line)],
            q1=[row.q1], median=[row.median], q3=[row.q3],
            lowerfence=[row.p5], upperfence=[row.p95], mean=[row.mean],
            name=str(row.line), marker_color=color, fillcolor=color, line=dict(color="#374151"),
            opacity=0.85, showlegend=False,
            hovertemplate=(
                f"<b>Linia {row.line}</b><br>mediana {row.median:.1f} min<br>"
                f"kwartyle {row.q1:.1f} – {row.q3:.1f} min<br>p5–p95 {row.p5:.1f} – {row.p95:.1f} min"
                "<extra></extra>"
            ),
        ))
    fig.add_hline(y=0, line=dict(color="#9ca3af", dash="dot"))
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Linia", type="category"),
        yaxis=dict(title="Opóźnienie [min]"),
    )
    return fig


def time_distance_figure(entry: dict) -> go.Figure:
    fig = go.Figure()
    start = datetime.fromisoformat(entry["start"])
    for series in entry["series"]:
        max_delay = max(series["delay"]) if series["delay"] else 0.0
        fig.add_trace(go.Scattergl(
            x=[start + timedelta(seconds=t) for t in series["t"]],
            y=[None if y is None else round(y, 3) for y in series["y"]],
            mode="lines",
            line=dict(color=delay_color(max_delay), width=2),
            customdata=series["delay"],
            name=series["trip_id"], showlegend=False,
            hovertemplate="%{x|%H:%M:%S}<br>opóźnienie %{customdata} min<extra></extra>",
        ))
    stops = entry["stops"]
    fig.update_layout(
        height=max(420, 22 * len(stops)), margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Czas"),
        yaxis=dict(tickmode="array", tickvals=list(range(len(stops))), ticktext=stops,
                   range=[-0.5, len(stops) - 0.5]),
    )
    return fig


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
    tracks = ensure_animation_data(cfg.loaded_sim["tracks"])  # obsługuje też stary format pliku
    simulated_lines = tuple(cfg.loaded_sim["simulated_lines"])
    sim_stats = cfg.loaded_sim["stats"]
    skipped_lines = list(cfg.loaded_sim.get("skipped_lines", []))
    # Starsze pliki nie mają wskaźników i danych wykresów - wtedy ich nie pokazujemy.
    metrics = cfg.loaded_sim.get("metrics")
    delay_records = cfg.loaded_sim.get("delay_by_line", [])
    td_data = cfg.loaded_sim.get("time_distance", {})
else:
    if not loaded_lines:
        st.error("Nie udało się zbudować żadnej trasy do animacji.")
        st.stop()

    tracks, simulated_lines, sim_stats, metrics, delay_records, td_data = run_cached_simulation(
        start_time.isoformat(),
        stop_time.isoformat(),
        tuple(str(line) for line in lines),
        5,
        cfg.passengers_on,
        cfg.demand_scale,
    )
    skipped_lines = list(route_skipped)
delay_df = pd.DataFrame(delay_records)

all_coords = [(point.lat, point.lon) for road, _, _ in display_roads for point in road.points]
if not all_coords:
    all_coords = animation_points(tracks)

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
    with_fullness=bool(sim_stats.get("passengers")),
).add_to(m)

st.caption(
    f"{start_time:%d.%m.%Y %H:%M} – {stop_time:%d.%m.%Y %H:%M} · "
    f"{len(simulated_lines)} linii · {sim_stats['trams']} tramwajów (bloków) · "
    f"krok próbek do statystyk {sim_stats['time_step']} s · "
    f"czasy przejazdu: {sim_stats['roads_from_sampler']} odcinków z danych realtime, "
    f"{sim_stats['roads_from_schedule']} z rozkładu"
)
if metrics:
    render_metrics(metrics)
if skipped_lines:
    st.warning("Pominięte linie bez kompletnej trasy w obu kierunkach: " + ", ".join(skipped_lines))

st_folium(m, width="100%", height=700, returned_objects=[])

tab_lines, tab_td = st.tabs(["Opóźnienia według linii", "Wykres czas–droga"])
with tab_lines:
    if delay_df.empty:
        st.info("Brak danych o opóźnieniach.")
    else:
        st.plotly_chart(delay_by_line_figure(delay_df, line_colors), width="stretch")
        st.caption(
            "Pudełko: kwartyle opóźnienia (25–75%), kreska: mediana, wąsy: 5–95 percentyl. "
            "Liczone z całego czasu jazdy, bez postojów na pętli. Wartości ujemne = jazda przed czasem."
        )
with tab_td:
    if not td_data:
        st.info("Brak danych do wykresu.")
    else:
        td_lines = sorted({e["line"] for e in td_data.values()}, key=line_sort_key)
        col_line, col_dir = st.columns([1, 2])
        with col_line:
            td_line = st.selectbox("Linia", td_lines)
        directions = sorted(e["direction"] for e in td_data.values() if e["line"] == td_line)
        with col_dir:
            td_dir = st.radio(
                "Kierunek",
                directions,
                format_func=lambda d: "→ " + td_data[f"{td_line}|{d}"]["headsign"],
                horizontal=True,
            )
        entry = td_data[f"{td_line}|{td_dir}"]
        st.plotly_chart(time_distance_figure(entry), width="stretch")
        st.caption(
            "Każda linia to jeden kurs: im bardziej stroma, tym szybciej jedzie tramwaj. "
            "Linie zbliżające się do siebie oznaczają tramwaje jadące „w peletonie”. "
            "Kolor: maksymalne opóźnienie kursu (jak w trybie „Opóźnienie” na mapie)."
        )

if cfg.loaded_sim is None:
    render_sidebar_export(
        start_time=start_time,
        stop_time=stop_time,
        simulated_lines=simulated_lines,
        skipped_lines=skipped_lines,
        sim_stats=sim_stats,
        tracks=tracks,
        extras={"metrics": metrics, "delay_by_line": delay_records, "time_distance": td_data},
    )
