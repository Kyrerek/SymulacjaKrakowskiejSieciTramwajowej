"""Obsługa danych GTFS (tramwaje ZTP Kraków) - część A projektu.

Główne wejście:  load_day("2026-10-28", routes=[1, 12])  ->  DayData
Czasy we wszystkich tabelach to sekundy od północy dnia symulacji
(kursy nocne mają > 86400, np. "25:10:00" -> 90600).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/GTFS_KRK_T")
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday",
             "friday", "saturday", "sunday"]


@dataclass
class DayData:
    date: pd.Timestamp
    trips: pd.DataFrame        # trip_id, route_short_name, direction_id, headsign, block_id, shape_id, service_id
    stop_times: pd.DataFrame   # trip_id, stop_sequence, stop_id, arrival_s, departure_s, shape_dist_traveled
    stops: pd.DataFrame        # stop_id, stop_name, stop_code, stop_lat, stop_lon
    shapes: pd.DataFrame       # shape_id, shape_pt_sequence, shape_pt_lat, shape_pt_lon


def parse_gtfs_time(s: pd.Series) -> pd.Series:
    p = s.str.strip().str.split(":", expand=True).astype(int)
    return p[0] * 3600 + p[1] * 60 + p[2]


@lru_cache(maxsize=None)
def _read(name: str, data_dir: Path) -> pd.DataFrame | None:
    """Czyta plik GTFS jako tekst (identyfikatory zostają tekstem). Wynik jest
    cache'owany - NIE modyfikuj zwróconej tabeli, rób .copy()."""
    path = Path(data_dir) / name
    if not path.exists():
        return None
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig")


def active_services(day, data_dir: Path = DATA_DIR) -> set[str]:
    """service_id obowiązujące danego dnia (calendar + calendar_dates)."""
    day = pd.Timestamp(day).normalize()
    ymd = day.strftime("%Y%m%d")
    active: set[str] = set()

    cal = _read("calendar.txt", Path(data_dir))
    if cal is not None:
        m = ((cal[_WEEKDAYS[day.weekday()]] == "1")
             & (cal["start_date"] <= ymd) & (cal["end_date"] >= ymd))
        active |= set(cal.loc[m, "service_id"])

    cd = _read("calendar_dates.txt", Path(data_dir))
    if cd is not None:
        d = cd[cd["date"] == ymd]
        active |= set(d.loc[d["exception_type"] == "1", "service_id"])
        active -= set(d.loc[d["exception_type"] == "2", "service_id"])
    return active

feed_start = "23.10.2026"
feed_end = "20.01.2027"
def load_day(day, routes: list | None = None, data_dir: Path = DATA_DIR) -> DayData:
    """Dane jednego dnia; opcjonalnie tylko wybrane linie (numery, np. [1, 12])."""
    data_dir = Path(data_dir)
    day = pd.Timestamp(day).normalize()
    services = active_services(day, data_dir)
    if not services:
        raise ValueError(f"Brak rozkładu na {day.date()} (aktualnie feed jest od {feed_start} do {feed_end})")

    r = _read("routes.txt", data_dir)[["route_id", "route_short_name"]]
    if routes is not None:
        r = r[r["route_short_name"].isin([str(x) for x in routes])]

    trips = _read("trips.txt", data_dir)
    trips = trips[trips["service_id"].isin(services)].merge(r, on="route_id")
    trips = trips.rename(columns={"trip_headsign": "headsign"})[
        ["trip_id", "route_short_name", "direction_id", "headsign",
         "block_id", "shape_id", "service_id"]].reset_index(drop=True)

    st = _read("stop_times.txt", data_dir)
    st = st[st["trip_id"].isin(trips["trip_id"])].copy()
    st["arrival_s"] = parse_gtfs_time(st["arrival_time"])
    st["departure_s"] = parse_gtfs_time(st["departure_time"])
    st["stop_sequence"] = st["stop_sequence"].astype(int)
    st["shape_dist_traveled"] = pd.to_numeric(st["shape_dist_traveled"])
    st = (st[["trip_id", "stop_sequence", "stop_id", "arrival_s",
              "departure_s", "shape_dist_traveled"]]
          .sort_values(["trip_id", "stop_sequence"]).reset_index(drop=True))

    stops = _read("stops.txt", data_dir)
    stops = stops[stops["stop_id"].isin(st["stop_id"])].copy()
    stops["stop_lat"] = stops["stop_lat"].astype(float)
    stops["stop_lon"] = stops["stop_lon"].astype(float)
    stops = stops[["stop_id", "stop_name", "stop_code", "stop_lat", "stop_lon"]
                  ].reset_index(drop=True)

    sh = _read("shapes.txt", data_dir)
    sh = sh[sh["shape_id"].isin(trips["shape_id"])].copy()
    sh["shape_pt_sequence"] = sh["shape_pt_sequence"].astype(int)
    sh["shape_pt_lat"] = sh["shape_pt_lat"].astype(float)
    sh["shape_pt_lon"] = sh["shape_pt_lon"].astype(float)
    sh = (sh[["shape_id", "shape_pt_sequence", "shape_pt_lat", "shape_pt_lon"]]
          .sort_values(["shape_id", "shape_pt_sequence"]).reset_index(drop=True))

    return DayData(day, trips, st, stops, sh)


def validate_day(day: DayData) -> list[str]:
    """Lista wykrytych problemów w danych dnia (pusta = brak problemów)."""
    p: list[str] = []
    st = day.stop_times
    n = st.groupby("trip_id").size()
    if (n < 2).any():
        p.append(f"{(n < 2).sum()} kursów ma mniej niż 2 przystanki")
    if (st["departure_s"] < st["arrival_s"]).any():
        p.append(f"{(st['departure_s'] < st['arrival_s']).sum()} wierszy z odjazdem przed przyjazdem")
    back = (st.groupby("trip_id")["arrival_s"].diff() < 0).sum()
    if back:
        p.append(f"{back} przystanków z czasem wcześniejszym niż poprzedni")
    if (st["stop_sequence"].groupby(st["trip_id"]).diff() <= 0).any():
        p.append("niemonotoniczny stop_sequence w niektórych kursach")
    if day.stops[["stop_lat", "stop_lon"]].isna().any(axis=1).any():
        p.append("przystanki bez współrzędnych")
    no_shape = ~day.trips["shape_id"].isin(day.shapes["shape_id"])
    if no_shape.any():
        p.append(f"{no_shape.sum()} kursów bez geometrii w shapes")

    tt = st.groupby("trip_id").agg(start=("departure_s", "first"),
                                   end=("arrival_s", "last"))
    b = (day.trips.merge(tt, left_on="trip_id", right_index=True)
         .sort_values(["block_id", "start"]))
    b["prev_end"] = b.groupby("block_id")["end"].shift()
    ov = (b["start"] < b["prev_end"]).sum()
    if ov:
        p.append(f"{ov} kursów zaczyna się, zanim skończył się poprzedni kurs tego samego bloku")
    return p


def segments(day: DayData) -> pd.DataFrame:
    """Odcinki między kolejnymi przystankami kursu (dla silnika i kalibracji)."""
    st = day.stop_times.sort_values(["trip_id", "stop_sequence"])
    g = st.groupby("trip_id")
    out = pd.DataFrame({
        "trip_id": st["trip_id"],
        "stop_from": g["stop_id"].shift(),
        "stop_to": st["stop_id"],
        "planned_travel_s": st["arrival_s"] - g["departure_s"].shift(),
        "dist": st["shape_dist_traveled"] - g["shape_dist_traveled"].shift(),
    })
    return out.dropna(subset=["stop_from"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dane ogólne o liniach (niezależne od jednego dnia) - dla mapy, wyboru linii,
# wspólnych odcinków. Opcjonalny filtr: trams (numery linii) i dates (daty).
# ---------------------------------------------------------------------------
def _line_trips(trams, dates, data_dir: Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    r = _read("routes.txt", data_dir)[["route_id", "route_short_name"]]
    if trams:
        r = r[r["route_short_name"].isin([str(t) for t in trams])]
    trips = _read("trips.txt", data_dir).merge(r, on="route_id")
    if dates:
        services: set[str] = set()
        for d in dates:
            services |= active_services(d, data_dir)
        trips = trips[trips["service_id"].isin(services)]
    return trips


def _representative_trips(trips: pd.DataFrame) -> pd.DataFrame:
    """Po jednym kursie na (linia, kierunek): wariant trasy (shape_id), który
    występuje najczęściej."""
    keys = ["route_short_name", "direction_id"]
    cnt = trips.groupby(keys + ["shape_id"]).size().rename("n").reset_index()
    best = (cnt.sort_values(keys + ["n", "shape_id"],
                            ascending=[True, True, False, True])
            .drop_duplicates(keys)[keys + ["shape_id"]])
    return (trips.merge(best, on=keys + ["shape_id"])
            .sort_values("trip_id").drop_duplicates(keys)
            .sort_values(keys).reset_index(drop=True))


def get_shapes_of_trips(trams=None, dates=None, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Geometria tras: dla każdej linii (i kierunku) główny wariant przebiegu."""
    rep = _representative_trips(_line_trips(trams, dates, data_dir))
    sh = _read("shapes.txt", Path(data_dir))
    out = rep.merge(sh, on="shape_id")
    out["shape_pt_sequence"] = out["shape_pt_sequence"].astype(int)
    out["shape_pt_lat"] = out["shape_pt_lat"].astype(float)
    out["shape_pt_lon"] = out["shape_pt_lon"].astype(float)
    out = out.rename(columns={"route_short_name": "tram_number",
                              "trip_headsign": "direction"})
    return (out[["tram_number", "direction_id", "direction", "shape_id",
                 "shape_pt_sequence", "shape_pt_lat", "shape_pt_lon"]]
            .sort_values(["tram_number", "direction_id", "shape_pt_sequence"])
            .reset_index(drop=True))


def get_all_stops(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Wszystkie przystanki. station_code łączy perony tego samego przystanku
    (prefiks stop_code, np. '920' z '920-01')."""
    s = _read("stops.txt", Path(data_dir)).copy()
    s["stop_lat"] = s["stop_lat"].astype(float)
    s["stop_lon"] = s["stop_lon"].astype(float)
    s["station_code"] = s["stop_code"].str.split("-").str[0]
    s = s.rename(columns={"stop_desc": "stop_number"})
    return s[["stop_id", "stop_name", "stop_number", "stop_code",
              "station_code", "stop_lat", "stop_lon"]].reset_index(drop=True)


def get_stops_for_tram(trams=None, dates=None, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Przystanki linii w kolejności przejazdu, osobno dla każdego kierunku."""
    data_dir = Path(data_dir)
    rep = _representative_trips(_line_trips(trams, dates, data_dir))
    st = _read("stop_times.txt", data_dir)
    st = st[st["trip_id"].isin(rep["trip_id"])].copy()
    st["stop_sequence"] = st["stop_sequence"].astype(int)
    out = (rep[["trip_id", "route_short_name", "direction_id", "trip_headsign"]]
           .merge(st[["trip_id", "stop_sequence", "stop_id"]], on="trip_id")
           .merge(get_all_stops(data_dir), on="stop_id")
           .rename(columns={"route_short_name": "tram_number",
                            "trip_headsign": "direction"}))
    return (out[["tram_number", "direction_id", "direction", "stop_sequence",
                 "stop_id", "stop_name", "stop_number", "station_code",
                 "stop_lat", "stop_lon"]]
            .sort_values(["tram_number", "direction_id", "stop_sequence"])
            .reset_index(drop=True))