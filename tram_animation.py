"""Płynna animacja tramwajów na mapie Folium z przełączanym kolorowaniem.

Animacja odtwarza dziennik ruchu każdego tramwaju (Tram.events), a nie próbki co X sekund:
  * jazda po odcinku = (t0, t1, odcinek, d0, d1) - pozycja w chwili t jest liczona w
    przeglądarce wzdłuż geometrii odcinka, więc tramwaj zawsze trzyma się torów,
    niezależnie od długości symulowanego okresu;
  * postój = punkt od chwili t do następnego zdarzenia;
  * niewidoczny = przed pierwszym kursem, po zjeździe, po końcu służby.

Kolor kropki zależy od wybranego trybu (przełącznik na mapie, działa bez przeliczania):
  Linia / Opóźnienie / Status / Napełnienie (to ostatnie tylko z symulacją pasażerów).

Format danych (słownik JSON, zapisywany też w pliku symulacji), zwięzły - czasy jako różnice
w pełnych sekundach od poprzedniego wpisu, współrzędne jako liczby całkowite (stopnie * 1e5):
  {"format": 3,
   "roads":  [{"c": [lat, lon, ...], "cum": [m, ...]}],          # geometria odcinków
   "tracks": [{"ev": [...],                                        # zdarzenia, zmienna długość:
                 [dt, 0]                         - niewidoczny
                 [dt, 1, lat*1e5, lon*1e5]       - postój w punkcie
                 [dt, 2, odcinek, d0, d1, czas]  - jazda po odcinku od d0 do d1 (m) przez `czas` s
                 [dt, 3, lat*1e5, lon*1e5]       - punkt ze starego formatu (interpolacja liniowa)
               "st"/"dl"/"fl"/"tr": [dt, wartość, ...],            # zmiany statusu, opóźnienia (s),
                                                                    # napełnienia (%), kursu
               "trips": [[linia, kierunek], ...], "name": str, "cap": int}]}
"""
from __future__ import annotations

import json
from datetime import datetime

from branca.element import MacroElement
from folium.template import Template

from models import TramStatus

SPEEDS = [1, 5, 10, 30, 60, 120, 300, 600]
DEFAULT_SPEED = 60
ANIMATION_FORMAT = 3

STATUS_CODES = {
    TramStatus.ARRIVAL: 0,   # jazda
    TramStatus.DWEL: 1,      # postój na przystanku
    TramStatus.WAIT: 2,      # kolejka przed przystankiem / czekanie na peron
    TramStatus.LAYOVER: 3,   # pętla / oczekiwanie na odjazd z rozkładu
}

# Progi i kolory (wspólne dla mapy i wykresów w main.py)
DELAY_CLASSES = [  # (górna granica w s, kolor, etykieta)
    (-60, "#2563eb", "przed czasem > 1 min"),
    (60, "#16a34a", "±1 min"),
    (180, "#a3e635", "1–3 min"),
    (300, "#f59e0b", "3–5 min"),
    (600, "#ef4444", "5–10 min"),
    (None, "#7f1d1d", "> 10 min"),
]
STATUS_CLASSES = [
    (0, "#2563eb", "jazda"),
    (1, "#16a34a", "postój na przystanku"),
    (2, "#dc2626", "kolejka przed przystankiem"),
    (3, "#9ca3af", "pętla / czeka na odjazd"),
]
FULLNESS_CLASSES = [  # (górna granica w %, kolor, etykieta)
    (25, "#16a34a", "< 25%"),
    (50, "#a3e635", "25–50%"),
    (85, "#f59e0b", "50–85%"),
    (None, "#dc2626", "≥ 85%"),
]


def _encode_events(events: list[tuple]) -> list:
    """[(t, rodzaj, ...)] z czasami bezwzględnymi -> zwięzła lista z różnicami czasu."""
    out: list = []
    prev = 0
    for t, kind, *rest in events:
        t = int(round(t))
        out.extend((t - prev, kind))
        prev = t
        if kind in (1, 3):
            out.extend((int(round(rest[0] * 1e5)), int(round(rest[1] * 1e5))))
        elif kind == 2:
            road, d0, d1, t1 = rest
            out.extend((road, int(round(d0)), int(round(d1)), max(0, int(round(t1)) - t)))
    return out


def _encode_series(series: list) -> list:
    """[t, wartość, ...] -> [dt, wartość, ...] (czasy zaokrąglone do sekund)."""
    out: list = []
    prev = 0
    for k in range(0, len(series), 2):
        t = int(round(series[k]))
        out.extend((t - prev, series[k + 1]))
        prev = t
    return out


def _push_change(series: list, t: float, value) -> None:
    if len(series) >= 2 and series[-1] == value:
        return
    if len(series) >= 2 and series[-2] == t:
        series[-1] = value
        return
    series.extend((t, value))


def trams_to_animation(trams) -> dict:
    """Dane animacji z dzienników ruchu tramwajów (po TramMotion.run_trams)."""
    road_index: dict[int, int] = {}
    roads: list[dict] = []

    def road_id(road) -> int:
        key = id(road)
        if key not in road_index:
            road_index[key] = len(roads)
            roads.append({
                "c": [round(v, 5) for p in road.points for v in (p.lat, p.lon)],
                "cum": [round(d, 1) for d in road.cumulative_distances],
            })
        return road_index[key]

    tracks: list[dict] = []
    for tram in trams:
        if not tram.events:
            continue
        trips = list(tram.schedule or [])
        trip_pos = {id(trip): i for i, trip in enumerate(trips)}
        trip_info = [[trip.line, trip.headsign] for trip in trips] or [[str(tram.line), ""]]
        cap = tram.capacity
        ev: list[tuple] = []
        st: list = []
        dl: list = []
        fl: list = []
        tr: list = []
        for t, kind, a, b, c, d, status, delay, load, trip in tram.events:
            t = round(t)
            if ev and ev[-1][0] == t:
                ev.pop()                      # po zaokrągleniu do sekundy liczy się ostatnie
            if kind == 2:
                ev.append((t, 2, road_id(a), b, c, d))
            elif kind == 1:
                ev.append((t, 1, a, b))
            else:
                ev.append((t, 0))
            _push_change(st, t, STATUS_CODES.get(status, 0))
            _push_change(dl, t, int(round(delay)))
            _push_change(fl, t, int(round(100 * load / cap)) if cap else 0)
            _push_change(tr, t, trip_pos.get(id(trip), 0) if trip is not None else 0)
        tracks.append({"ev": _encode_events(ev), "st": _encode_series(st), "dl": _encode_series(dl),
                       "fl": _encode_series(fl), "tr": _encode_series(tr),
                       "trips": trip_info, "name": tram.name, "cap": cap})
    return {"format": ANIMATION_FORMAT, "roads": roads, "tracks": tracks}


def legacy_to_animation(tracks: list[dict]) -> dict:
    """Stary format pliku (próbki: t, c, d, s, f) -> format zdarzeń (punkty interpolowane liniowo)."""
    out: list[dict] = []
    for track in tracks:
        times = track.get("t", [])
        coords = track.get("c", [])
        if not times:
            continue
        ev = [(t, 3, coords[2 * i], coords[2 * i + 1]) for i, t in enumerate(times)]
        ev.append((times[-1] + 1, 0))

        def by_time(changes: list) -> list:      # [indeks próbki, wartość] -> [dt, wartość]
            return _encode_series([times[changes[k]] if k % 2 == 0 else changes[k]
                                   for k in range(len(changes))])

        out.append({
            "ev": _encode_events(ev),
            "st": by_time(track.get("s", [])),
            "dl": by_time(track.get("d", [])),
            "fl": by_time(track.get("f", [])),
            "tr": [times[0], 0],
            "trips": [[track.get("line", ""), track.get("headsign", "")]],
            "name": track.get("tram_name", ""),
            "cap": track.get("capacity", 0),
        })
    return {"format": ANIMATION_FORMAT, "roads": [], "tracks": out}


def ensure_animation_data(data) -> dict:
    """Przyjmuje dane w nowym formacie albo listę śladów ze starego pliku JSON."""
    if isinstance(data, dict) and data.get("format") == ANIMATION_FORMAT:
        return data
    return legacy_to_animation(data or [])


def animation_lines(data: dict) -> list[str]:
    return sorted({str(info[0]) for tr in data["tracks"] for info in tr["trips"] if info[0] != ""},
                  key=lambda v: (not v.isdigit(), int(v) if v.isdigit() else v))


def animation_points(data: dict) -> list[tuple[float, float]]:
    """Wszystkie współrzędne (do wyznaczenia widoku mapy, gdy brak geometrii linii)."""
    pts = [(r["c"][i], r["c"][i + 1]) for r in data["roads"] for i in range(0, len(r["c"]), 2)]
    for tr in data["tracks"]:
        ev, i = tr["ev"], 0
        while i < len(ev):
            kind = ev[i + 1]
            if kind in (1, 3):
                pts.append((ev[i + 2] / 1e5, ev[i + 3] / 1e5))
            i += {0: 2, 1: 4, 2: 6, 3: 4}[kind]
    return pts


class TramAnimation(MacroElement):
    _template = Template("""
        {% macro script(this, kwargs) %}
        (function () {
            var map = {{ this._parent.get_name() }};
            var data = {{ this.data }};
            var renderer = L.canvas({ padding: 0.5 });
            var s = data.start;
            var startMs = Date.UTC(s[0], s[1] - 1, s[2], s[3], s[4], s[5]);
            var mode = data.defaultMode;
            var lineColor = {};
            data.lineColors.forEach(function (lc) { lineColor[lc[0]] = lc[1]; });

            // Rozpakowanie zwięzłego formatu do tablic: czasy bezwzględne, 6 liczb na zdarzenie.
            function decodeEvents(src) {
                var out = [], t = 0, i = 0;
                while (i < src.length) {
                    t += src[i]; var kind = src[i + 1];
                    if (kind === 0) { out.push(t, 0, 0, 0, 0, 0); i += 2; }
                    else if (kind === 2) { out.push(t, 2, src[i + 2], src[i + 3], src[i + 4], t + src[i + 5]); i += 6; }
                    else { out.push(t, kind, src[i + 2] / 1e5, src[i + 3] / 1e5, 0, 0); i += 4; }
                }
                return new Float64Array(out);
            }
            function decodeSeries(src) {
                var out = src ? src.slice() : [], t = 0;
                for (var k = 0; k < out.length; k += 2) { t += out[k]; out[k] = t; }
                return out;
            }
            data.tracks.forEach(function (tr) {
                tr.ev = decodeEvents(tr.ev);
                tr.st = decodeSeries(tr.st); tr.dl = decodeSeries(tr.dl);
                tr.fl = decodeSeries(tr.fl); tr.tr = decodeSeries(tr.tr);
            });

            var tracks = data.tracks.map(function (tr) {
                var marker = L.circleMarker([0, 0], {
                    renderer: renderer, radius: 6, color: "#ffffff", weight: 1,
                    fillColor: "#e6194b", fillOpacity: 1.0
                });
                var trk = { tr: tr, marker: marker, shown: false, i: 0, col: null };
                marker.bindPopup(function () { return popupHtml(trk); });
                return trk;
            });

            // --- pozycja ----------------------------------------------------------
            function roadPoint(ri, dist) {
                var r = data.roads[ri], cum = r.cum, c = r.c, n = cum.length;
                if (dist <= 0 || n < 2) return [c[0], c[1]];
                if (dist >= cum[n - 1]) return [c[2 * n - 2], c[2 * n - 1]];
                var lo = 0, hi = n - 1;
                while (hi - lo > 1) {
                    var mid = (lo + hi) >> 1;
                    if (cum[mid] <= dist) lo = mid; else hi = mid;
                }
                var seg = cum[hi] - cum[lo], f = seg > 0 ? (dist - cum[lo]) / seg : 0;
                return [c[2 * lo] + f * (c[2 * hi] - c[2 * lo]), c[2 * lo + 1] + f * (c[2 * hi + 1] - c[2 * lo + 1])];
            }
            function eventIndex(trk, t) {            // ostatnie zdarzenie z czasem <= t
                var ev = trk.tr.ev, n = ev.length / 6, i = trk.i;
                if (i < n && ev[6 * i] <= t && (i === n - 1 || ev[6 * (i + 1)] > t)) return i;
                var lo = 0, hi = n - 1;
                while (lo < hi) {
                    var mid = (lo + hi + 1) >> 1;
                    if (ev[6 * mid] <= t) lo = mid; else hi = mid - 1;
                }
                trk.i = lo;
                return lo;
            }
            function position(trk, t) {
                var ev = trk.tr.ev, n = ev.length / 6;
                if (!n || t < ev[0]) return null;
                var i = eventIndex(trk, t), b = 6 * i, kind = ev[b + 1];
                if (kind === 0) return null;
                if (kind === 1) return [ev[b + 2], ev[b + 3]];
                if (kind === 2) {
                    var t0 = ev[b], t1 = ev[b + 5];
                    var f = t1 > t0 ? Math.min(1, (t - t0) / (t1 - t0)) : 1;
                    return roadPoint(ev[b + 2], ev[b + 3] + f * (ev[b + 4] - ev[b + 3]));
                }
                // rodzaj 3: punkt ze starego formatu, interpolacja do następnego punktu
                if (i + 1 < n && ev[b + 7] === 3) {
                    var g = (t - ev[b]) / (ev[b + 6] - ev[b]);
                    return [ev[b + 2] + g * (ev[b + 8] - ev[b + 2]), ev[b + 3] + g * (ev[b + 9] - ev[b + 3])];
                }
                return [ev[b + 2], ev[b + 3]];
            }

            // --- wartości (listy zmian [t, wartość, ...]) ---------------------------
            function stepValue(list, t) {
                if (!list || !list.length) return null;
                if (list[0] > t) return list[1];
                var lo = 0, hi = list.length / 2 - 1;
                while (lo < hi) {
                    var mid = (lo + hi + 1) >> 1;
                    if (list[2 * mid] <= t) lo = mid; else hi = mid - 1;
                }
                return list[2 * lo + 1];
            }
            function classColor(classes, v) {
                for (var k = 0; k < classes.length; k++) {
                    if (classes[k][0] === null || v < classes[k][0]) return classes[k][1];
                }
                return classes[classes.length - 1][1];
            }
            function statusColor(v) {
                for (var k = 0; k < data.statusClasses.length; k++) {
                    if (data.statusClasses[k][0] === v) return data.statusClasses[k][1];
                }
                return "#2563eb";
            }
            function tripInfo(tr, t) {
                var idx = stepValue(tr.tr, t);
                return tr.trips[idx === null ? 0 : idx] || tr.trips[0];
            }
            function colorFor(trk, t) {
                var tr = trk.tr;
                if (mode === "delay") return classColor(data.delayClasses, stepValue(tr.dl, t) || 0);
                if (mode === "status") return statusColor(stepValue(tr.st, t));
                if (mode === "fullness") return classColor(data.fullnessClasses, stepValue(tr.fl, t) || 0);
                return lineColor[String(tripInfo(tr, t)[0])] || "#e6194b";
            }
            var statusNames = {};
            data.statusClasses.forEach(function (c) { statusNames[c[0]] = c[2]; });
            function popupHtml(trk) {
                var tr = trk.tr, info = tripInfo(tr, simT), d = stepValue(tr.dl, simT);
                var html = "<b>Linia " + info[0] + (info[1] ? " → " + info[1] : "") + "</b><br>" +
                    tr.name + "<br>Status: " + (statusNames[stepValue(tr.st, simT)] || "–");
                if (d !== null) html += "<br>Opóźnienie: " + (d / 60).toFixed(1) + " min";
                if (data.withFullness && tr.cap) {
                    html += "<br>Napełnienie: " + stepValue(tr.fl, simT) + "% (" + tr.cap + " miejsc)";
                }
                return html;
            }

            // --- sterowanie -------------------------------------------------------
            var simT = 0, playing = false, speed = data.defaultSpeed, lastFrame = null;
            function pad(x) { return (x < 10 ? "0" : "") + x; }
            function label(t) {
                var d = new Date(startMs + Math.round(t) * 1000);
                return pad(d.getUTCDate()) + "." + pad(d.getUTCMonth() + 1) + " " +
                    pad(d.getUTCHours()) + ":" + pad(d.getUTCMinutes()) + ":" + pad(d.getUTCSeconds());
            }

            var ui = L.control({ position: "bottomleft" });
            var playBtn, slider, timeLbl, speedSel;
            ui.onAdd = function () {
                var div = L.DomUtil.create("div", "tram-anim-control");
                div.innerHTML =
                    '<button type="button" class="ta-play" title="Odtwórz / pauza">▶</button>' +
                    '<input type="range" class="ta-slider" min="0" step="1">' +
                    '<span class="ta-time"></span>' +
                    '<select class="ta-speed" title="Prędkość odtwarzania"></select>';
                playBtn = div.querySelector(".ta-play");
                slider = div.querySelector(".ta-slider");
                timeLbl = div.querySelector(".ta-time");
                speedSel = div.querySelector(".ta-speed");
                slider.max = data.duration;
                data.speeds.forEach(function (v) {
                    var o = document.createElement("option");
                    o.value = v; o.textContent = v + "×";
                    if (v === speed) o.selected = true;
                    speedSel.appendChild(o);
                });
                playBtn.onclick = function () {
                    if (simT >= data.duration) simT = 0;
                    playing = !playing;
                    lastFrame = null;
                    playBtn.textContent = playing ? "⏸" : "▶";
                };
                slider.oninput = function () { simT = +slider.value; render(); };
                speedSel.onchange = function () { speed = +speedSel.value; };
                L.DomEvent.disableClickPropagation(div);
                L.DomEvent.disableScrollPropagation(div);
                return div;
            };
            ui.addTo(map);

            // --- przełącznik kolorów i legenda ------------------------------------
            var legend = L.control({ position: "topright" });
            var legendBody;
            legend.onAdd = function () {
                var div = L.DomUtil.create("div", "tram-legend");
                var opts = data.modes.map(function (m) {
                    return '<label class="tl-mode' + (m.enabled ? "" : " tl-disabled") + '"' +
                        (m.enabled ? "" : ' title="' + m.hint + '"') + '>' +
                        '<input type="radio" name="tl-mode" value="' + m.key + '"' +
                        (m.key === mode ? " checked" : "") + (m.enabled ? "" : " disabled") + '> ' +
                        m.label + '</label>';
                }).join("");
                div.innerHTML = '<div class="tl-title">Kolor tramwajów</div>' + opts +
                    '<div class="tl-body"></div>';
                legendBody = div.querySelector(".tl-body");
                div.querySelectorAll("input[name=tl-mode]").forEach(function (inp) {
                    inp.onchange = function () { mode = inp.value; drawLegend(); render(); };
                });
                L.DomEvent.disableClickPropagation(div);
                L.DomEvent.disableScrollPropagation(div);
                return div;
            };
            legend.addTo(map);

            function swatch(color, text) {
                return '<div class="tl-row"><span class="tl-dot" style="background:' + color +
                    '"></span>' + text + '</div>';
            }
            function drawLegend() {
                var rows = [];
                if (mode === "line") {
                    rows = data.lineColors.map(function (lc) { return swatch(lc[1], lc[0]); });
                    legendBody.className = "tl-body tl-grid";
                } else {
                    var classes = mode === "delay" ? data.delayClasses
                        : mode === "status" ? data.statusClasses : data.fullnessClasses;
                    rows = classes.map(function (c) { return swatch(c[1], c[2]); });
                    legendBody.className = "tl-body";
                }
                legendBody.innerHTML = rows.join("");
            }
            drawLegend();

            // --- pętla animacji ---------------------------------------------------
            function render() {
                for (var k = 0; k < tracks.length; k++) {
                    var trk = tracks[k], p = position(trk, simT);
                    if (p) {
                        trk.marker.setLatLng(p);
                        var col = colorFor(trk, simT);
                        if (col !== trk.col) { trk.marker.setStyle({ fillColor: col }); trk.col = col; }
                        if (!trk.shown) { trk.marker.addTo(map); trk.shown = true; }
                    } else if (trk.shown) {
                        map.removeLayer(trk.marker); trk.shown = false;
                    }
                }
                slider.value = simT;
                timeLbl.textContent = label(simT);
            }
            function frame(now) {
                if (playing) {
                    if (lastFrame !== null) {
                        simT = Math.min(data.duration, simT + (now - lastFrame) / 1000 * speed);
                        if (simT >= data.duration) { playing = false; playBtn.textContent = "▶"; }
                    }
                    lastFrame = now;
                    render();
                }
                requestAnimationFrame(frame);
            }
            render();
            requestAnimationFrame(frame);
        })();
        {% endmacro %}

        {% macro header(this, kwargs) %}
        <style>
            .tram-anim-control, .tram-legend {
                background: rgba(255, 255, 255, 0.94); border-radius: 6px;
                box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3); font: 13px/1.3 sans-serif; color: #111;
            }
            .tram-anim-control { display: flex; align-items: center; gap: 8px; padding: 6px 10px; }
            .tram-anim-control button {
                width: 30px; height: 26px; cursor: pointer; font-size: 14px;
                border: 1px solid #999; border-radius: 4px; background: #fff;
            }
            .tram-anim-control .ta-slider { width: 320px; }
            .tram-anim-control .ta-time { font-variant-numeric: tabular-nums; min-width: 105px; }
            .tram-legend { padding: 8px 10px; max-width: 220px; }
            .tram-legend .tl-title { font-weight: 600; margin-bottom: 4px; }
            .tram-legend .tl-mode { display: block; cursor: pointer; }
            .tram-legend .tl-disabled { color: #999; cursor: not-allowed; }
            .tram-legend .tl-body { margin-top: 6px; padding-top: 6px; border-top: 1px solid #ddd; }
            .tram-legend .tl-grid { display: grid; grid-template-columns: repeat(4, auto); column-gap: 8px; }
            .tram-legend .tl-row { display: flex; align-items: center; gap: 5px; white-space: nowrap; }
            .tram-legend .tl-dot {
                width: 10px; height: 10px; border-radius: 50%; border: 1px solid #fff;
                box-shadow: 0 0 0 1px #999; flex: none;
            }
        </style>
        {% endmacro %}
    """)

    def __init__(self, anim_data, line_colors: dict[str, str], start_time: datetime,
                 duration_s: float, with_fullness: bool = False, default_mode: str = "line",
                 speeds: list[int] = SPEEDS, default_speed: int = DEFAULT_SPEED):
        super().__init__()
        self._name = "TramAnimation"
        anim = ensure_animation_data(anim_data)
        modes = [
            {"key": "line", "label": "Linia", "enabled": True},
            {"key": "delay", "label": "Opóźnienie", "enabled": True},
            {"key": "status", "label": "Status", "enabled": True},
            {"key": "fullness", "label": "Napełnienie", "enabled": with_fullness,
             "hint": "Włącz symulację pasażerów w panelu bocznym"},
        ]
        self.data = json.dumps({
            "start": [start_time.year, start_time.month, start_time.day,
                      start_time.hour, start_time.minute, start_time.second],
            "duration": int(duration_s),
            "speeds": speeds,
            "defaultSpeed": default_speed,
            "defaultMode": default_mode if any(m["key"] == default_mode and m["enabled"] for m in modes) else "line",
            "modes": modes,
            "withFullness": with_fullness,
            "lineColors": [[line, line_colors.get(line, "#e6194b")] for line in animation_lines(anim)],
            "delayClasses": [list(c) for c in DELAY_CLASSES],
            "statusClasses": [list(c) for c in STATUS_CLASSES],
            "fullnessClasses": [list(c) for c in FULLNESS_CLASSES],
            "roads": anim["roads"],
            "tracks": anim["tracks"],
        }, separators=(",", ":"), ensure_ascii=False)
