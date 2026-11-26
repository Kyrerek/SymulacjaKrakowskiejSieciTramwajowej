
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import json
from typing import Any

import streamlit as st


@dataclass
class SidebarConfig:
    source_mode: str
    start_time: datetime
    stop_time: datetime
    show_stops: bool
    loaded_sim: dict[str, Any] | None = None


@st.cache_data
def _parse_uploaded_sim(raw_bytes: bytes) -> dict[str, Any]:
    return json.loads(raw_bytes.decode("utf-8"))


def render_sidebar(feed_min: date, feed_max: date, default_day: date) -> SidebarConfig:
    with st.sidebar:
        st.header("Źródło danych")
        source_mode = st.radio("Tryb", ["Nowa symulacja", "Z pliku JSON"], horizontal=True)
        uploaded_file = None
        start_time = None
        stop_time = None

        if source_mode == "Nowa symulacja":
            st.header("Czas symulacji")
            period_mode = st.radio("Zakres", ["Jeden dzień", "Okres"], horizontal=True)
            if period_mode == "Jeden dzień":
                sim_day = st.date_input(
                    "Data",
                    value=default_day,
                    min_value=feed_min,
                    max_value=feed_max,
                    format="DD.MM.YYYY",
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
                    from_time = st.time_input(
                        "Godzina startu", value=time(12, 0), step=timedelta(minutes=15)
                    )
                with col_to:
                    to_time = st.time_input(
                        "Godzina końca", value=time(12, 0), step=timedelta(minutes=15)
                    )
                start_time = datetime.combine(start_day, from_time)
                stop_time = datetime.combine(end_day, to_time)
        else:
            st.header("Plik symulacji")
            uploaded_file = st.file_uploader("Wybierz plik JSON", type=["json"])

        st.header("Widok mapy")
        show_stops = st.checkbox("Pokaż przystanki", value=False)

    if source_mode == "Z pliku JSON":
        if uploaded_file is None:
            st.info("Wybierz plik JSON z zapisaną symulacją w panelu bocznym.")
            st.stop()
        try:
            loaded_sim = _parse_uploaded_sim(uploaded_file.getvalue())
            start_time = datetime.fromisoformat(loaded_sim["start_time"])
            stop_time = datetime.fromisoformat(loaded_sim["stop_time"])
        except Exception as exc:
            st.error(f"Nie udało się wczytać pliku symulacji: {exc}")
            st.stop()
        return SidebarConfig(
            source_mode=source_mode,
            start_time=start_time,
            stop_time=stop_time,
            show_stops=show_stops,
            loaded_sim=loaded_sim,
        )

    if stop_time <= start_time:
        st.error("Koniec symulacji musi być późniejszy niż początek.")
        st.stop()

    return SidebarConfig(
        source_mode=source_mode,
        start_time=start_time,
        stop_time=stop_time,
        show_stops=show_stops,
        loaded_sim=None,
    )


def render_sidebar_export(
    start_time: datetime,
    stop_time: datetime,
    simulated_lines: tuple[str, ...] | list[str],
    skipped_lines: list[str],
    sim_stats: dict[str, Any],
    tracks: list[dict[str, Any]],
) -> None:
    export_payload = {
        "start_time": start_time.isoformat(),
        "stop_time": stop_time.isoformat(),
        "duration_s": (stop_time - start_time).total_seconds(),
        "simulated_lines": list(simulated_lines),
        "skipped_lines": list(skipped_lines),
        "stats": sim_stats,
        "tracks": tracks,
    }
    export_json = json.dumps(
        export_payload,
        ensure_ascii=False,
        default=lambda o: (
            o.tolist() if hasattr(o, "tolist") else (o.__dict__ if hasattr(o, "__dict__") else str(o))
        ),
    )
    with st.sidebar:
        st.divider()
        st.download_button(
            label="Pobierz dane symulacji (JSON)",
            data=export_json,
            file_name=f"sim_{start_time:%Y%m%d_%H%M}_{stop_time:%Y%m%d_%H%M}.json",
            mime="application/json",
            use_container_width=True,
        )