"""Wskaźniki i dane do wykresów liczone z wyników symulacji (dla main.py).

Wszystko liczone po stronie Pythona, żeby do przeglądarki trafiały tylko gotowe,
małe zestawienia (a nie setki tysięcy próbek).
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from models import TramStatus

# Kurs punktualny: na ostatnim przystanku od 1 min przed do 3 min po czasie z rozkładu.
PUNCTUAL_EARLY_S = -60
PUNCTUAL_LATE_S = 180


def compute_metrics(samples, trams, time_step: float, passenger_summary: dict | None = None) -> dict:
    active = [s for s in samples if s.status != TramStatus.LAYOVER]
    in_motion = Counter(s.timestamp for s in samples if s.status == TramStatus.ARRIVAL)
    trips = [t for tram in trams for t in tram.completed_trips]
    end_delays = np.array([t["end_delay_s"] for t in trips]) if trips else np.array([])
    queue_samples = sum(1 for s in samples if s.status == TramStatus.WAIT)

    metrics = {
        "max_in_motion": max(in_motion.values()) if in_motion else 0,
        "median_delay_s": float(np.median([s.delay_s for s in active])) if active else 0.0,
        "completed_trips": len(trips),
        "punctuality": float(np.mean((end_delays >= PUNCTUAL_EARLY_S) & (end_delays <= PUNCTUAL_LATE_S)))
        if trips else None,
        "early_share": float(np.mean(end_delays < PUNCTUAL_EARLY_S)) if trips else None,
        "late_share": float(np.mean(end_delays > PUNCTUAL_LATE_S)) if trips else None,
        "queue_hours": queue_samples * time_step / 3600.0,
    }
    if passenger_summary:
        metrics["passengers"] = passenger_summary
    return metrics


def delay_by_line(samples) -> pd.DataFrame:
    """Rozkład opóźnienia (min) według linii: kwartyle, p5/p95 i średnia (bez postojów na pętli)."""
    by_line: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        if s.status != TramStatus.LAYOVER and s.line:
            by_line[s.line].append(s.delay_s / 60.0)
    rows = []
    for line, values in by_line.items():
        v = np.asarray(values)
        p5, q1, med, q3, p95 = np.percentile(v, [5, 25, 50, 75, 95])
        rows.append({"line": line, "p5": p5, "q1": q1, "median": med, "q3": q3, "p95": p95,
                     "mean": float(v.mean()), "n": len(v)})
    df = pd.DataFrame(rows)
    return df.sort_values("median", ascending=False).reset_index(drop=True) if not df.empty else df


def time_distance_data(samples, trams, network, start_time) -> dict[str, dict]:
    """Dane wykresu czas-droga: dla każdej linii i kierunku oś Y = przystanki głównego wariantu trasy.

    Zwraca {"linia|kierunek": {"line", "direction", "headsign", "stops": [nazwy],
    "series": [{trip_id, t (s od start_time), y, delay}]}} - same typy JSON, więc dane da się
    zapisać razem z symulacją i wczytać z pliku.
    """
    # Przystanki kursów tak, jak widzi je symulacja (kursy trwające na starcie są przycięte).
    trip_stops: dict[str, list] = {}
    trip_meta: dict[str, tuple[str, str, str]] = {}
    for tram in trams:
        for trip in tram.schedule:
            trip_stops[trip.trip_id] = [s.stop for s in trip.stops]
            trip_meta[trip.trip_id] = (trip.line, trip.direction_id, trip.headsign)

    # Główny wariant = najczęstsza pełna sekwencja przystanków danej linii i kierunku.
    variants: dict[tuple[str, str], Counter] = defaultdict(Counter)
    headsigns: dict[tuple[str, str], Counter] = defaultdict(Counter)
    full_stops: dict[tuple[str, ...], list] = {}
    for block in network.blocks:
        for trip in block.trips:
            if trip.trip_id not in trip_meta:
                continue
            key = (trip.line, trip.direction_id)
            ids = tuple(str(s.stop.id) for s in trip.stops)
            variants[key][ids] += 1
            headsigns[key][trip.headsign] += 1
            full_stops.setdefault(ids, [s.stop for s in trip.stops])

    main: dict[tuple[str, str], dict] = {}
    for key, counter in variants.items():
        ids = max(counter.items(), key=lambda kv: (kv[1], len(kv[0])))[0]
        main[key] = {
            "index": {sid: i for i, sid in enumerate(ids)},
            "stops": [s.name for s in full_stops[ids]],
            "headsign": headsigns[key].most_common(1)[0][0],
        }

    points: dict[str, dict] = {}
    for s in samples:
        if not s.trip_id or s.progress is None or s.trip_id not in trip_meta:
            continue
        line, direction, _ = trip_meta[s.trip_id]
        m = main.get((line, direction))
        if m is None:
            continue
        stops = trip_stops[s.trip_id]
        k = int(s.progress)
        frac = s.progress - k
        a = m["index"].get(str(stops[k].id)) if k < len(stops) else None
        b = m["index"].get(str(stops[k + 1].id)) if k + 1 < len(stops) else None
        if a is not None and frac < 1e-6:
            y = float(a)
        elif a is not None and b == a + 1:
            y = round(a + frac, 3)
        else:
            y = None  # odcinek spoza głównego wariantu (np. objazd, zjazd)
        p = points.setdefault(s.trip_id, {"t": [], "y": [], "delay": []})
        p["t"].append(int(round((s.timestamp - start_time).total_seconds())))
        p["y"].append(y)
        p["delay"].append(round(s.delay_s / 60.0, 1))

    result: dict[str, dict] = {}
    for trip_id, p in points.items():
        line, direction, _ = trip_meta[trip_id]
        m = main[(line, direction)]
        entry = result.setdefault(f"{line}|{direction}", {
            "line": line, "direction": direction, "start": start_time.isoformat(),
            "headsign": m["headsign"], "stops": m["stops"], "series": []})
        entry["series"].append({"trip_id": trip_id, **p})
    return result
