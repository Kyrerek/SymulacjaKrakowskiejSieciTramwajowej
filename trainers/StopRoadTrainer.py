from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

import realtime_data as rt_proc
from delay_samplers import StopRoadWeekSampler, StopRoadSampler
from models import StopRoad


class StopRoadTrainer:
    """
    Zarządza treningiem, zapisem i dopasowaniem obiektów StopRoadSampler
    dla całej sieci na podstawie danych GTFS-Realtime.
    """

    def __init__(self, dist_num: int = 5, min_samples_per_segment: int = 5):
        self.dist_num = dist_num
        self.min_samples_per_segment = min_samples_per_segment
        self.samplers: dict[tuple[str, str], StopRoadWeekSampler] = {}
        self.training_stats: dict = {}

    # --------------------------------------------------------- trening z danych
    def fit_from_observations(self, seg_df: pd.DataFrame, value_col: str = "travel_s") -> dict:
        """Trenuje samplery na podstawie tabeli z build_segment_observations()."""
        
        df = seg_df.copy()
        if "valid" in df.columns:
            df = df[df["valid"]]
        df = df[df[value_col].notna() & (df[value_col] > 0) & (df[value_col] < 1800)]

        # Wyznaczamy dzień tygodnia: 0 = poniedziałek .. 6 = niedziela
        if "day" in df.columns:
            df["week_day"] = pd.to_datetime(df["day"]).dt.dayofweek
        else:
            df["week_day"] = (
                pd.to_datetime(df["departure_ts"], unit="s", utc=True)
                .dt.tz_convert("Europe/Warsaw")
                .dt.dayofweek
            )

        trained_count = 0
        skipped_count = 0

        for (stop_from, stop_to), seg_group in df.groupby(["stop_from","stop_to"]):
            # Normalizujemy czas doby do 0..86400s (kursy nocne w GTFS mają >86400s)
            all_week_samples = [
                (float(t) % 86400.0, float(v))
                for t, v in zip(seg_group["tod_s"], seg_group[value_col])
            ]

            if len(all_week_samples) < self.min_samples_per_segment:
                skipped_count += 1
                continue

            # Dzielimy próbki na 7 dni tygodnia (0..6)
            by_day: dict[int, list[tuple[float, float]]] = {d: [] for d in range(7)}
            for wd, sub in seg_group.groupby("week_day"):
                by_day[int(wd)] = [
                    (float(t) % 86400.0, float(v))
                    for t, v in zip(sub["tod_s"], sub[value_col])
                ]

            week_data: list[list[tuple[float, float]]] = []
            dist_nums: list[int] = []

            for d in range(7):
                # Jeśli w dany dzień było za mało kursów, bierzemy próbki z całego tygodnia
                day_samples = (
                    by_day[d]
                    if len(by_day[d]) >= self.min_samples_per_segment
                    else all_week_samples
                )
                effective_buckets = min(self.dist_num, max(2, len(day_samples) // 3))
                week_data.append(day_samples)
                dist_nums.append(effective_buckets)

            sampler = StopRoadWeekSampler()
            sampler.fit(week_data, dist_nums=dist_nums)
            self.samplers[(str(stop_from), str(stop_to))] = sampler
            trained_count += 1

        self.training_stats = {
            "trained_segments": trained_count,
            "skipped_low_samples": skipped_count,
            "total_observations": int(seg_df["valid"].sum()) if "valid" in seg_df else len(seg_df),
        }
        return self.training_stats

    def fit_from_raw_rt(
        self,
        vp_source,
        static_path=None,
        snapshot_dir=None,
        day=None,
    ) -> dict:
        """Trenuje bezpośrednio z surowych plików .pb / .tar.gz."""
        vp = rt_proc.parse_vehicle_positions(vp_source)
        static = None
        if static_path is None and snapshot_dir is not None and day is not None:
            static_path = rt_proc.pick_static_snapshot(day, snapshot_dir)
        if static_path is not None:
            static = rt_proc.load_static(static_path)

        ev = rt_proc.build_stop_events(vp, static)
        seg = rt_proc.build_segment_observations(ev)
        return self.fit_from_observations(seg)

    # --------------------------------------------------------- zapis i odczyt JSON
    def save(self, path: str | Path) -> None:
        """Zapisuje wszystkie wytrenowane samplery do jednego pliku JSON."""
        payload = {
            "dist_num": self.dist_num,
            "stats": self.training_stats,
            "segments": {
                f"{s_from}->{s_to}": {
                    f"sampler_{day}": {
                        "thresholds": sampler.thresholds,
                        "dists": [list(d) for d in sampler.dists],
                        "metrics": sampler.metrics
                    } 
                    for day,sampler in enumerate(week_sampler.samplers) 
                }
                for (s_from, s_to), week_sampler in self.samplers.items()
            },
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> StopRoadTrainer:
        """Wczytuje wytrenowane samplery z pliku JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        trainer = cls(dist_num=payload.get("dist_num", 5))
        trainer.training_stats = payload.get("stats", {})

        for key, segment_data in payload.get("segments", {}).items():
            s_from, s_to = key.split("->", 1)
            sorted_days = sorted(segment_data.items(), key=lambda kv: int(kv[0].split("_")[1]))
            samplers = []
            for _, data in sorted_days:
                sampler = StopRoadSampler(
                    thresholds=data["thresholds"],
                    dists=[tuple(d) for d in data["dists"]],
                )
                sampler.metrics = data.get("metrics", {})
                samplers.append(sampler)
            trainer.samplers[(s_from, s_to)] = StopRoadWeekSampler(samplers)

        return trainer

    def get_predict_fn(self, road: StopRoad) -> Callable[[float], float]:
        """Zwraca funkcję predict(time_sec) gotową do przypisania pod road.delay_."""
        key = (str(road.start.id), str(road.end.id))
        sampler = self.samplers.get(key)
        if sampler is None:
            return lambda time: 60.0
        return sampler.predict

    def bind_to_route(self, stops: list) -> None:
        """Podpina sampler.predict pod każdy StopRoad na liście przystanków trasy."""
        for stop in stops:
            for road in stop.roads:
                road.delay_ = self.get_predict_fn(road)