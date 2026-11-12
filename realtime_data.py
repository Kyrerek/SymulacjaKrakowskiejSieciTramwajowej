"""Przetwarzanie danych GTFS-Realtime (ZTP Kraków) i powiązanie z rozkładem.

Surowe pliki z kolektora (`<pobranie>_<last-modified>.pb`, w katalogach lub
archiwach .tar.gz) zamieniamy na tabele:

    parse_vehicle_positions  -> pozycje pojazdów (1 wiersz = 1 raport pojazdu)
    build_stop_events        -> postoje na przystankach (przyjazd, odjazd, postój,
                                opóźnienie względem rozkładu)
    build_segment_observations -> czasy przejazdu przystanek->przystanek
                                (klucz (stop_from, stop_to) jak w StopRoad)
    sampler_data             -> dane w formacie StopRoadSampler.fit
    parse_trip_updates       -> ostatnie prognozy ZTP dla (kurs, przystanek)
    parse_alerts             -> komunikaty o zakłóceniach

Powiązanie z rozkładem: identyfikator kursu `trip_id` jest wspólny
(np. block_176_trip_10_service_1), ale NUMERY BLOKÓW ZMIENIAJĄ SIĘ między wersjami
rozkładu. Dlatego rozkład dobieramy do daty danych (pick_static_snapshot) i
sprawdzamy zgodność (check_static_match), zanim policzymy opóźnienia.
Wszystkie czasy zdarzeń to sekundy epoch (UTC); czasy rozkładowe to sekundy od
początku doby usługowej (jak w GTFS, mogą przekraczać 86400).
"""
from __future__ import annotations

import argparse
import json
import re
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd
from google.transit import gtfs_realtime_pb2 as rt

TZ = "Europe/Warsaw"
SAMPLE_S = 10          # typowy odstęp między raportami jednego pojazdu (zmierzone 19.10)
_FNAME = re.compile(r"(\d+)_(\d+)\.pb$")
_STATUS = {v.number: v.name for v in rt.VehiclePosition.VehicleStopStatus.DESCRIPTOR.values}


# ------------------------------------------------------------ czytanie plików
def iter_feed_files(source) -> Iterator[tuple[int, int, bytes]]:
    """Zwraca (pobranie_ts, last_modified_ts, bajty) dla każdego pliku .pb.
    source: archiwum .tar/.tar.gz, katalog (rekurencyjnie) albo lista takich ścieżek."""
    if isinstance(source, (list, tuple)):
        for s in source:
            yield from iter_feed_files(s)
        return
    p = Path(source)
    if p.is_dir():
        for f in sorted(p.rglob("*.pb")):
            m = _FNAME.search(f.name)
            if m:
                yield int(m[1]), int(m[2]), f.read_bytes()
    elif tarfile.is_tarfile(p):
        with tarfile.open(p, "r:*") as tar:
            for member in tar:
                m = _FNAME.search(member.name) if member.isfile() else None
                if m:
                    yield int(m[1]), int(m[2]), tar.extractfile(member).read()
    else:
        raise ValueError(f"Nieobsługiwane źródło danych: {p}")


def _parse_feed(raw: bytes):
    """FeedMessage albo None, gdy plik jest uszkodzony (zdarzają się ucięte)."""
    feed = rt.FeedMessage()
    try:
        feed.ParseFromString(raw)
    except Exception:
        return None
    return feed


def file_report(source) -> dict:
    """Jakość zebranych plików: liczba, uszkodzone, odstępy i luki w danych."""
    lm, bad, total = [], 0, 0
    for _, last_mod, raw in iter_feed_files(source):
        total += 1
        if _parse_feed(raw) is None:
            bad += 1
        else:
            lm.append(last_mod)
    s = pd.Series(sorted(set(lm)))
    d = s.diff().dropna()
    return {"files": total, "corrupt": bad,
            "interval_median_s": float(d.median()) if len(d) else None,
            "interval_p90_s": float(d.quantile(.9)) if len(d) else None,
            "gaps_over_60s": int((d > 60).sum()),
            "gap_minutes_total": round(float(d[d > 60].sum()) / 60, 1)}


# --------------------------------------------------------------- czas, doba usługowa
def service_day_start(date) -> pd.Timestamp:
    """Początek doby usługowej GTFS: południe minus 12 h (poprawne przy zmianie czasu)."""
    noon = (pd.Timestamp(date).normalize() + pd.Timedelta(hours=12)).tz_localize(TZ)
    return noon - pd.Timedelta(hours=12)


def _day_start_epoch(dates) -> dict:
    return {d: service_day_start(d).timestamp() for d in pd.unique(pd.Series(dates))}


# ------------------------------------------------------------------ pozycje pojazdów
def parse_vehicle_positions(source, max_files: int | None = None) -> pd.DataFrame:
    """Pozycje pojazdów; powtórzone raporty (ten sam pojazd i czas) są usuwane."""
    rows, bad = [], 0
    for i, (fetched, _, raw) in enumerate(iter_feed_files(source)):
        if max_files is not None and i >= max_files:
            break
        feed = _parse_feed(raw)
        if feed is None:
            bad += 1
            continue
        hdr = feed.header.timestamp
        for e in feed.entity:
            if not e.HasField("vehicle"):
                continue
            v = e.vehicle
            rows.append((fetched, hdr, v.timestamp, v.vehicle.id, v.trip.trip_id,
                         v.position.latitude, v.position.longitude, v.position.bearing,
                         v.current_stop_sequence, v.stop_id, int(v.current_status),
                         v.congestion_level == rt.VehiclePosition.CONGESTION))
    cols = ["fetched_ts", "header_ts", "vehicle_ts", "vehicle_id", "trip_id", "lat",
            "lon", "bearing", "stop_sequence", "stop_id", "status_code", "congestion"]
    df = pd.DataFrame(rows, columns=cols)
    df = (df[df["vehicle_ts"] > 0]
          .drop_duplicates(["vehicle_id", "vehicle_ts"])
          .sort_values(["vehicle_id", "vehicle_ts"]).reset_index(drop=True))
    df["status"] = df["status_code"].map(_STATUS)
    df = df.drop(columns="status_code")
    df.attrs["corrupt_files"] = bad
    return df


# ---------------------------------------------------------------- rozkład statyczny
@dataclass
class StaticSchedule:
    stop_times: pd.DataFrame   # trip_id, stop_sequence, stop_id, arrival_s, departure_s
    trips: pd.DataFrame        # trip_id, route_short_name, direction_id, block_id, shape_id
    version: str = ""


def _read_member(path, name: str, usecols=None) -> pd.DataFrame:
    p = Path(path)
    if p.is_dir():
        return pd.read_csv(p / name, dtype=str, encoding="utf-8-sig", usecols=usecols)
    with zipfile.ZipFile(p) as zf, zf.open(name) as f:
        return pd.read_csv(f, dtype=str, encoding="utf-8-sig", usecols=usecols)


def _secs(s: pd.Series) -> pd.Series:
    p = s.str.strip().str.split(":", expand=True).astype(int)
    return p[0] * 3600 + p[1] * 60 + p[2]


def load_static(path) -> StaticSchedule:
    """Rozkład z katalogu GTFS albo z pliku .zip (np. migawki z kolektora)."""
    st = _read_member(path, "stop_times.txt",
                      ["trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time"])
    st["stop_sequence"] = st["stop_sequence"].astype(int)
    st["arrival_s"] = _secs(st.pop("arrival_time"))
    st["departure_s"] = _secs(st.pop("departure_time"))
    trips = _read_member(path, "trips.txt")
    routes = _read_member(path, "routes.txt", ["route_id", "route_short_name"])
    trips = trips.merge(routes, on="route_id", how="left")
    keep = [c for c in ["trip_id", "route_short_name", "direction_id", "block_id",
                        "shape_id", "trip_headsign"] if c in trips]
    try:
        version = _read_member(path, "feed_info.txt")["feed_version"].iloc[0]
    except Exception:
        version = ""
    return StaticSchedule(st, trips[keep], str(version))


_SNAP = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{6})")


def _snapshot_valid_from(path: Path) -> pd.Timestamp:
    try:
        return pd.to_datetime(_read_member(path, "calendar.txt")["start_date"], format="%Y%m%d").min()
    except Exception:
        return pd.to_datetime(_read_member(path, "calendar_dates.txt")["date"], format="%Y%m%d").min()


def pick_static_snapshot(day, snapshot_dir) -> Path:
    """Wybiera migawkę rozkładu (pliki z kolektora: GTFS_KRK_T_<data>_<godz>.zip)
    obowiązującą w danym dniu: największy początek ważności <= dzień. Przy tym samym
    początku bierze najpóźniej zapisaną migawkę sprzed końca tego dnia."""
    day = pd.Timestamp(day).normalize()
    cands = []
    for p in sorted(Path(snapshot_dir).glob("*.zip")):
        m = _SNAP.search(p.name)
        stamp = pd.Timestamp(f"{m[1]} {m[2][:2]}:{m[2][2:4]}:{m[2][4:]}") if m else pd.Timestamp.min
        cands.append((_snapshot_valid_from(p), stamp, p))
    ok = [c for c in cands if c[0] <= day]
    if not ok:
        raise ValueError(f"Brak migawki rozkładu ważnej w dniu {day.date()}")
    top = max(c[0] for c in ok)
    same = [c for c in ok if c[0] == top]
    before = [c for c in same if c[1] < day + pd.Timedelta(days=1)]
    return max(before or same, key=lambda c: c[1] if before else -c[1].value)[2]


def assign_service_date(df: pd.DataFrame, static: StaticSchedule | None = None,
                        ts_col: str = "vehicle_ts") -> pd.Series:
    """Doba usługowa każdego wiersza. Z rozkładem: wybiera dzień, dla którego
    planowany start kursu jest najbliżej obserwacji (obsługuje kursy po północy,
    np. 25:10). Bez rozkładu: raporty przed 3:00 lokalnie należą do poprzedniej doby."""
    local = pd.to_datetime(df[ts_col], unit="s", utc=True).dt.tz_convert(TZ)
    d0 = local.dt.tz_localize(None).dt.normalize()
    d1 = d0 - pd.Timedelta(days=1)
    fallback = d1.where(local.dt.hour < 3, d0)
    if static is None:
        return fallback
    first_dep = df["trip_id"].map(static.stop_times.groupby("trip_id")["departure_s"].min())
    ep = _day_start_epoch(pd.concat([d0, d1]))
    err0 = (df[ts_col] - (d0.map(ep) + first_dep)).abs()
    err1 = (df[ts_col] - (d1.map(ep) + first_dep)).abs()
    chosen = d1.where(err1 < err0, d0)
    return chosen.where(first_dep.notna(), fallback)


def check_static_match(vp: pd.DataFrame, static: StaticSchedule) -> dict:
    """Czy rozkład pasuje do danych: (1) czy identyfikatory kursów istnieją,
    (2) czy przy tym samym (trip_id, stop_sequence) jest ten sam przystanek.
    Sam zgodny trip_id NIE wystarcza: bloki są numerowane od nowa w nowych wersjach."""
    obs = vp[(vp["trip_id"] != "") & (vp["stop_sequence"] > 0)]
    trip_cov = float(obs["trip_id"].drop_duplicates().isin(static.trips["trip_id"]).mean())
    pairs = obs[["trip_id", "stop_sequence", "stop_id"]].drop_duplicates()
    m = pairs.merge(static.stop_times[["trip_id", "stop_sequence", "stop_id"]],
                    on=["trip_id", "stop_sequence"], how="inner", suffixes=("", "_static"))
    agree = float((m["stop_id"] == m["stop_id_static"]).mean()) if len(m) else 0.0
    return {"trip_id_coverage": round(trip_cov, 3), "stop_agreement": round(agree, 3),
            "ok": bool(trip_cov >= 0.9 and agree >= 0.95), "static_version": static.version}


# ------------------------------------------------------- postoje i opóźnienia
def build_stop_events(vp: pd.DataFrame, static: StaticSchedule | None = None,
                      sample_s: float = SAMPLE_S) -> pd.DataFrame:
    """Postoje z raportów STOPPED_AT. Pojazd raportuje co ~sample_s, więc przyjazd
    szacujemy jako pierwszy raport minus sample_s/2, odjazd jako ostatni plus
    sample_s/2 (dokładność rzędu +-sample_s). Z rozkładem dodaje opóźnienia."""
    s = vp[(vp["status"] == "STOPPED_AT") & (vp["trip_id"] != "")].copy()
    s["service_date"] = assign_service_date(s, static)
    g = s.groupby(["service_date", "trip_id", "vehicle_id", "stop_sequence", "stop_id"])["vehicle_ts"]
    ev = g.agg(first_ts="min", last_ts="max", n_samples="count").reset_index()
    ev["arrival_ts"] = ev["first_ts"] - sample_s / 2
    ev["departure_ts"] = ev["last_ts"] + sample_s / 2
    ev["dwell_s"] = ev["departure_ts"] - ev["arrival_ts"]
    ev["day_start_ts"] = ev["service_date"].map(_day_start_epoch(ev["service_date"]))
    if static is not None:
        sch = static.stop_times.rename(columns={"stop_id": "sched_stop_id",
                                                "arrival_s": "sched_arrival_s",
                                                "departure_s": "sched_departure_s"})
        ev = ev.merge(sch, on=["trip_id", "stop_sequence"], how="left")
        ev["stop_match"] = ev["stop_id"] == ev["sched_stop_id"]
        ok = ev["stop_match"]
        ev["delay_arrival_s"] = (ev["arrival_ts"] - (ev["day_start_ts"] + ev["sched_arrival_s"])).where(ok)
        ev["delay_departure_s"] = (ev["departure_ts"] - (ev["day_start_ts"] + ev["sched_departure_s"])).where(ok)
        ev = ev.merge(static.trips, on="trip_id", how="left")
    return ev.sort_values(["service_date", "vehicle_id", "arrival_ts"]).reset_index(drop=True)


def build_segment_observations(events: pd.DataFrame, max_travel_s: float = 1800) -> pd.DataFrame:
    """Czasy przejazdu między kolejnymi przystankami kursu (obie sąsiednie
    sekwencje muszą mieć zaobserwowany postój). Klucz (stop_from, stop_to)
    odpowiada StopRoad i odcinkom z osm_infrastructure.build_segments."""
    ev = events.sort_values(["service_date", "trip_id", "stop_sequence"]).reset_index(drop=True)
    grp = ev.groupby(["service_date", "trip_id"])
    nxt = grp[["stop_sequence", "stop_id", "arrival_ts", "vehicle_id"]].shift(-1)
    consecutive = (nxt["stop_sequence"] == ev["stop_sequence"] + 1) & (nxt["vehicle_id"] == ev["vehicle_id"])
    out = pd.DataFrame({
        "service_date": ev["service_date"], "trip_id": ev["trip_id"],
        "vehicle_id": ev["vehicle_id"], "seq_from": ev["stop_sequence"],
        "stop_from": ev["stop_id"], "stop_to": nxt["stop_id"],
        "departure_ts": ev["departure_ts"], "arrival_ts": nxt["arrival_ts"],
        "tod_s": ev["departure_ts"] - ev["day_start_ts"],
        "dwell_from_s": ev["dwell_s"],
    })[consecutive].copy()
    out["travel_s"] = out["arrival_ts"] - out["departure_ts"]
    if "sched_arrival_s" in ev:
        nxt_sched = grp["sched_arrival_s"].shift(-1)[consecutive]
        out["planned_travel_s"] = nxt_sched - ev.loc[consecutive, "sched_departure_s"]
        out["extra_s"] = out["travel_s"] - out["planned_travel_s"]
    out["valid"] = (out["travel_s"] > 0) & (out["travel_s"] < max_travel_s)
    return out.reset_index(drop=True)


def sampler_data(seg: pd.DataFrame, value: str = "travel_s") -> dict[tuple[str, str], list[tuple[float, float]]]:
    """Dane dla StopRoadSampler.fit: {(stop_from, stop_to): [(czas_doby_s, wartość), ...]}.
    Domyślnie czas przejazdu (zawsze > 0, więc log() w samplerze jest bezpieczny);
    opóźnienie względem rozkładu bywa <= 0 i nie nadaje się do rozkładu log-normalnego."""
    v = seg[seg["valid"] & (seg[value] > 0)]
    return {k: list(zip(g["tod_s"], g[value])) for k, g in v.groupby(["stop_from", "stop_to"])}


# ------------------------------------------------------------------ trip updates
def parse_trip_updates(source, max_files: int | None = None) -> pd.DataFrame:
    """Prognozy ZTP (bezwzględne czasy przyjazdu/odjazdu dla kolejnych przystanków).
    Zwraca po 1 wierszu na (kurs, przystanek, doba): pierwszą i ostatnią prognozę.
    Ostatnia prognoza jest najbliższa rzeczywistości. Uwaga: w trip_updates nie ma
    stop_sequence, a przystanek powtórzony w kursie (pętla) zostanie scalony."""
    acc: dict[tuple, list] = {}
    for i, (fetched, _, raw) in enumerate(iter_feed_files(source)):
        if max_files is not None and i >= max_files:
            break
        feed = _parse_feed(raw)
        if feed is None:
            continue
        bucket = (pd.Timestamp(feed.header.timestamp, unit="s", tz="UTC").tz_convert(TZ)
                  - pd.Timedelta(hours=4)).date()
        for e in feed.entity:
            tu = e.trip_update
            trip, veh = tu.trip.trip_id, tu.vehicle.id
            for u in tu.stop_time_update:
                arr = u.arrival.time if u.HasField("arrival") else 0
                dep = u.departure.time if u.HasField("departure") else 0
                key = (trip, u.stop_id, bucket)
                r = acc.get(key)
                if r is None:
                    acc[key] = [trip, u.stop_id, bucket, veh, arr, arr, dep, dep, fetched, fetched, 1]
                else:
                    if arr:
                        r[5] = arr
                        r[4] = r[4] or arr
                    if dep:
                        r[7] = dep
                        r[6] = r[6] or dep
                    r[9], r[10] = fetched, r[10] + 1
    cols = ["trip_id", "stop_id", "service_date", "vehicle_id", "first_arrival_ts",
            "last_arrival_ts", "first_departure_ts", "last_departure_ts",
            "first_seen_ts", "last_seen_ts", "n_obs"]
    df = pd.DataFrame(list(acc.values()), columns=cols)
    df["service_date"] = pd.to_datetime(df["service_date"])
    return df


def trip_update_delays(tu: pd.DataFrame, static: StaticSchedule) -> pd.DataFrame:
    """Dodaje opóźnienia prognozowane przez ZTP (ostatnia prognoza vs rozkład)."""
    m = tu.merge(static.stop_times, on=["trip_id", "stop_id"], how="inner")
    m["day_start_ts"] = m["service_date"].map(_day_start_epoch(m["service_date"]))
    arr = m["last_arrival_ts"].where(m["last_arrival_ts"] > 0)
    dep = m["last_departure_ts"].where(m["last_departure_ts"] > 0)
    m["tu_delay_arrival_s"] = arr - (m["day_start_ts"] + m["arrival_s"])
    m["tu_delay_departure_s"] = dep - (m["day_start_ts"] + m["departure_s"])
    return m


# ------------------------------------------------------------------------ alerty
def parse_alerts(source) -> pd.DataFrame:
    """Unikalne komunikaty z czasem pierwszego i ostatniego pojawienia się.
    is_test oznacza wpisy testowe ZTP (19.10 były tylko takie)."""
    seen: dict[str, dict] = {}
    for fetched, _, raw in iter_feed_files(source):
        feed = _parse_feed(raw)
        if feed is None:
            continue
        for e in feed.entity:
            if not e.HasField("alert"):
                continue
            a = e.alert
            txt = lambda t: t.translation[0].text if t.translation else ""
            r = seen.setdefault(e.id, {
                "alert_id": e.id, "cause": rt.Alert.Cause.Name(a.cause),
                "effect": rt.Alert.Effect.Name(a.effect), "header": txt(a.header_text),
                "description": txt(a.description_text),
                "routes": ",".join(sorted({x.route_id for x in a.informed_entity if x.route_id})),
                "first_seen_ts": fetched, "last_seen_ts": fetched, "n_files": 0})
            r["last_seen_ts"], r["n_files"] = fetched, r["n_files"] + 1
    df = pd.DataFrame(list(seen.values()))
    if not df.empty:
        df["is_test"] = (df["header"] + " " + df["description"]).str.contains("TEST", case=False)
    return df


# ------------------------------------------------------------------- cały dzień
def build_day(day, vp_source, out_dir, static_path=None, snapshot_dir=None,
              tu_source=None, alerts_source=None) -> dict:
    """Parsuje dane jednego dnia i zapisuje tabele (Parquet) + raport do out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = pd.Timestamp(day).strftime("%Y-%m-%d")
    report: dict = {"day": tag}

    vp = parse_vehicle_positions(vp_source)
    report["vehicle_positions"] = {"rows": len(vp), "vehicles": int(vp["vehicle_id"].nunique()),
                                   "corrupt_files": vp.attrs.get("corrupt_files", 0)}
    vp.to_parquet(out / f"vehicle_positions_{tag}.parquet")

    static = None
    if static_path is None and snapshot_dir is not None:
        static_path = pick_static_snapshot(day, snapshot_dir)
    if static_path is not None:
        static = load_static(static_path)
        report["static"] = {"path": str(static_path), **check_static_match(vp, static)}
        if not report["static"]["ok"]:
            report["warning"] = ("Rozkład nie pasuje do danych (wersja?). Opóźnienia "
                                 "policzone tylko dla kursów ze zgodnym przystankiem.")

    ev = build_stop_events(vp, static)
    seg = build_segment_observations(ev)
    ev.to_parquet(out / f"stop_events_{tag}.parquet")
    seg.to_parquet(out / f"segment_observations_{tag}.parquet")
    report["stop_events"] = len(ev)
    report["segment_observations"] = {"rows": len(seg), "valid": int(seg["valid"].sum())}

    if tu_source is not None:
        tu = parse_trip_updates(tu_source)
        tu.to_parquet(out / f"trip_updates_last_{tag}.parquet")
        report["trip_updates_rows"] = len(tu)
    if alerts_source is not None:
        al = parse_alerts(alerts_source)
        al.to_parquet(out / f"alerts_{tag}.parquet")
        report["alerts"] = {"unique": len(al), "non_test": int((~al["is_test"]).sum()) if len(al) else 0}

    (out / f"report_{tag}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Przetwarzanie danych GTFS-RT jednego dnia")
    ap.add_argument("--day", required=True, help="np. 2026-10-19")
    ap.add_argument("--vp", required=True, help="archiwum lub katalog z vehicle_positions")
    ap.add_argument("--tu", help="trip_updates (opcjonalnie)")
    ap.add_argument("--alerts", help="service_alerts (opcjonalnie)")
    ap.add_argument("--static", help="rozkład: katalog GTFS lub .zip")
    ap.add_argument("--snapshots", help="katalog static_gtfs z kolektora (wybór wg daty)")
    ap.add_argument("--out", default="data/rt")
    a = ap.parse_args()
    print(json.dumps(build_day(a.day, a.vp, a.out, a.static, a.snapshots, a.tu, a.alerts),
                     indent=2, default=str))


if __name__ == "__main__":
    main()