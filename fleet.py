"""Tabor tramwajowy: typ pojazdu (pojemność, drzwi) na podstawie tablicy.

Plik JSON (data/fleet.json) opisuje grupy pojazdów. Pojazd identyfikuje prefiks
z tablicy plus numer w przedziale [id_min, id_max] (np. "HW101" -> prefiks HW,
numer 101; numer jest równy vehicle.id z GTFS-RT, ale sam numer nie wystarcza,
bo zakresy różnych prefiksów się nakładają).

    fleet = Fleet.load("data/fleet.json")
    fleet.lookup("RP603")                 # TramType albo None
    fleet.sample(rng)                     # losowy typ (np. dla symulowanego kursu)
"""
from __future__ import annotations

import json
import re
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

_PLATE = re.compile(r"^\s*([A-Za-z]+)\s*-?\s*(\d+)\s*$")


@dataclass(frozen=True, slots=True)
class TramType:
    prefix: str
    id_min: int
    id_max: int
    model: str
    capacity: int
    doors: int
    length_m: float | None = None
    seats: int | None = None
    low_floor: bool | None = None

    @property
    def key(self) -> tuple[str, int]:
        return self.prefix, self.id_min

    @property
    def n_ids(self) -> int:
        return self.id_max - self.id_min + 1


class Fleet:
    def __init__(self, types: list[TramType], sources: list[dict] | None = None):
        if not types:
            raise ValueError("Pusty wykaz taboru")
        self.types = list(types)
        self.sources = list(sources or [])
        self._by_prefix: dict[str, list[TramType]] = {}
        for t in sorted(self.types, key=lambda t: (t.prefix, t.id_min)):
            self._by_prefix.setdefault(t.prefix, []).append(t)

    @classmethod
    def load(cls, path: str | Path) -> "Fleet":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        types = []
        for i, g in enumerate(raw["groups"]):
            missing = [k for k in ("prefix", "id_min", "id_max", "model", "capacity", "doors")
                       if g.get(k) in (None, "")]
            if missing:
                raise ValueError(f"fleet.json, grupa #{i}: brak pól {missing}")
            types.append(TramType(
                prefix=str(g["prefix"]).upper(), id_min=int(g["id_min"]), id_max=int(g["id_max"]),
                model=str(g["model"]), capacity=int(g["capacity"]), doors=int(g["doors"]),
                length_m=g.get("length_m"), seats=g.get("seats"), low_floor=g.get("low_floor")))
        return cls(types, raw.get("sources"))

    def lookup(self, plate: str | None = None, *, prefix: str | None = None,
               vehicle_id: int | str | None = None) -> TramType | None:
        """Typ pojazdu z tablicy ("HW101") albo z pary (prefix, vehicle_id)."""
        if plate is not None:
            m = _PLATE.match(plate)
            if not m:
                return None
            prefix, vehicle_id = m.group(1), m.group(2)
        if prefix is None or vehicle_id is None:
            return None
        try:
            number = int(vehicle_id)
        except (TypeError, ValueError):
            return None
        group = self._by_prefix.get(prefix.upper(), [])
        i = bisect_right([t.id_min for t in group], number) - 1
        if i >= 0 and group[i].id_min <= number <= group[i].id_max:
            return group[i]
        return None

    def sample(self, rng: np.random.Generator,
               weights: Mapping[tuple[str, int], float] | None = None) -> TramType:
        """Losowy typ. Domyślnie waga = szerokość zakresu numerów (przybliżenie udziału
        w taborze); można podać własne wagi {(prefix, id_min): waga}, np. z liczby
        kursów obserwowanych w danych realtime."""
        w = np.array([(weights or {}).get(t.key, t.n_ids) for t in self.types], dtype=float)
        if w.sum() <= 0:
            raise ValueError("Wagi typów sumują się do zera")
        return self.types[int(rng.choice(len(self.types), p=w / w.sum()))]