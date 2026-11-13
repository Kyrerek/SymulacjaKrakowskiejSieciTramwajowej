from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

import realtime_data as rt_proc
from delay_samplers import StopRoadSampler
from models import StopRoad


class StopRoadTrainer:
    """
    Zarządza treningiem, zapisem i dopasowaniem obiektów StopRoadSampler
    dla całej sieci na podstawie danych GTFS-Realtime.
    """

    def __init__(self, dist_num: int = 5, min_samples_per_segment: int = 5):
        self.dist_num = dist_num
        self.min_samples_per_segment = min_samples_per_segment
        self.samplers: dict[tuple[str, str], StopRoadSampler] = {}
        self.training_stats: dict = {}

    # --------------------------------------------------------- trening z danych
    def fit_from_observations(self, seg_df: pd.DataFrame, value_col: str = "travel_s") -> dict:
        """Trenuje samplery na podstawie tabeli z build_segment_observations()."""
        grouped_data = rt_proc.sampler_data(seg_df, value=value_col)

        trained_count = 0
        skipped_count = 0

        for (stop_from, stop_to), samples in grouped_data.items():
            # Normalizujemy czas doby do 0..86400s (kursy nocne w GTFS mają >86400s)
            normalized_samples = [(float(t) % 86400.0, float(v)) for t, v in samples]

            if len(normalized_samples) < self.min_samples_per_segment:
                skipped_count += 1
                continue

            # Dopasowujemy liczbę liści drzewa, jeśli próbek jest mało
            effective_buckets = min(self.dist_num, max(1, len(normalized_samples) // 3))

            sampler = StopRoadSampler()
            sampler.fit(normalized_samples, dist_num=effective_buckets)
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
                    "thresholds": sampler.thresholds,
                    "dists": [list(d) for d in sampler.dists],
                    "metrics": sampler.metrics,
                }
                for (s_from, s_to), sampler in self.samplers.items()
            },
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> StopRoadTrainer:
        """Wczytuje wytrenowane samplery z pliku JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        trainer = cls(dist_num=payload.get("dist_num", 5))
        trainer.training_stats = payload.get("stats", {})

        for key, data in payload.get("segments", {}).items():
            s_from, s_to = key.split("->", 1)
            sampler = StopRoadSampler(
                thresholds=data["thresholds"],
                dists=[tuple(d) for d in data["dists"]],
            )
            sampler.metrics = data.get("metrics", {})
            trainer.samplers[(s_from, s_to)] = sampler

        return trainer

    def get_predict_fn(self, road: StopRoad) -> Callable[[float], float]:
        """Zwraca funkcję predict(time_sec) gotową do przypisania pod road.delay_."""
        key = (str(road.start.id), str(road.end.id))
        sampler = self.samplers.get(key)
        if sampler is None:
            return lambda time_sec: 60.0
        return sampler.predict

    def bind_to_route(self, stops: list) -> None:
        """Podpina sampler.predict pod każdy StopRoad na liście przystanków trasy."""
        for stop in stops:
            for road in stop.roads:
                road.delay_ = self.get_predict_fn(road)