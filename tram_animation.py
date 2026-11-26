"""Płynna animacja tramwajów na mapie Folium.

Zamiast TimestampedGeoJson (który przeskakuje co krok symulacji) pozycje są
interpolowane liniowo między próbkami w przeglądarce, w każdej klatce
(requestAnimationFrame). Dane są przekazywane w zwartej postaci:
  t - sekundy od startu symulacji, c - płaska lista [lat, lon, lat, lon, ...].
"""
from __future__ import annotations

import json
from datetime import datetime

from branca.element import MacroElement
from folium.template import Template

SPEEDS = [1, 5, 10, 30, 60, 120, 300, 600]
DEFAULT_SPEED = 60


def samples_to_tracks(samples, start_time: datetime, time_step: float) -> list[dict]:
    """Grupuje próbki w ślady: jeden ślad = tramwaj na jednym kursie/linii bez przerw."""
    tracks: list[dict] = []
    open_tracks: dict[str, dict] = {}
    for s in samples:
        t = int(round((s.timestamp - start_time).total_seconds()))
        signature = (s.line, s.headsign)
        track = open_tracks.get(s.tram_id)
        if track is None or track["signature"] != signature or t - track["t"][-1] > time_step:
            track = {
                "signature": signature,
                "tram_name": s.tram_name,
                "line": s.line,
                "headsign": s.headsign,
                "t": [],
                "c": [],
                "max_delay_s": 0.0,
            }
            open_tracks[s.tram_id] = track
            tracks.append(track)
        track["t"].append(t)
        track["c"].extend((round(s.lat, 5), round(s.lon, 5)))
        track["max_delay_s"] = max(track["max_delay_s"], s.delay_s)
    for track in tracks:
        del track["signature"]
    return tracks


class TramAnimation(MacroElement):
    _template = Template("""
        {% macro script(this, kwargs) %}
        (function () {
            var map = {{ this._parent.get_name() }};
            var data = {{ this.data }};
            var renderer = L.canvas({ padding: 0.5 });
            var s = data.start;
            var startMs = Date.UTC(s[0], s[1] - 1, s[2], s[3], s[4], s[5]);

            var tracks = data.tracks.map(function (tr) {
                var marker = L.circleMarker([tr.c[0], tr.c[1]], {
                    renderer: renderer, radius: 6, color: "#ffffff", weight: 1,
                    fillColor: tr.color, fillOpacity: 1.0
                });
                marker.bindPopup(tr.popup);
                return { t: tr.t, c: tr.c, marker: marker, shown: false, i: 0 };
            });

            function locate(tr, t) {
                var ts = tr.t, n = ts.length;
                if (t < ts[0] || t > ts[n - 1]) return null;
                var i = tr.i;
                if (i >= n - 1 || ts[i] > t || ts[i + 1] < t) {
                    var lo = 0, hi = n - 1;            // wyszukiwanie binarne
                    while (hi - lo > 1) {
                        var mid = (lo + hi) >> 1;
                        if (ts[mid] <= t) lo = mid; else hi = mid;
                    }
                    i = lo;
                }
                tr.i = i;
                if (i >= n - 1) return [tr.c[2 * i], tr.c[2 * i + 1]];
                var f = (t - ts[i]) / (ts[i + 1] - ts[i]);
                return [
                    tr.c[2 * i] + f * (tr.c[2 * i + 2] - tr.c[2 * i]),
                    tr.c[2 * i + 1] + f * (tr.c[2 * i + 3] - tr.c[2 * i + 1])
                ];
            }

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
                    '<button type="button" class="ta-play">▶</button>' +
                    '<input type="range" class="ta-slider" min="0" step="1">' +
                    '<span class="ta-time"></span>' +
                    '<select class="ta-speed"></select>';
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

            function render() {
                for (var k = 0; k < tracks.length; k++) {
                    var tr = tracks[k], p = locate(tr, simT);
                    if (p) {
                        tr.marker.setLatLng(p);
                        if (!tr.shown) { tr.marker.addTo(map); tr.shown = true; }
                    } else if (tr.shown) {
                        map.removeLayer(tr.marker); tr.shown = false;
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
            .tram-anim-control {
                display: flex; align-items: center; gap: 8px;
                background: rgba(255, 255, 255, 0.92); padding: 6px 10px;
                border-radius: 6px; box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3);
                font: 13px/1.2 sans-serif; color: #111;
            }
            .tram-anim-control button {
                width: 30px; height: 26px; cursor: pointer; font-size: 14px;
                border: 1px solid #999; border-radius: 4px; background: #fff;
            }
            .tram-anim-control .ta-slider { width: 320px; }
            .tram-anim-control .ta-time { font-variant-numeric: tabular-nums; min-width: 105px; }
        </style>
        {% endmacro %}
    """)

    def __init__(self, tracks: list[dict], line_colors: dict[str, str], start_time: datetime,
                 duration_s: float, speeds: list[int] = SPEEDS, default_speed: int = DEFAULT_SPEED):
        super().__init__()
        self._name = "TramAnimation"
        payload_tracks = []
        for track in tracks:
            headsign = f" → {track['headsign']}" if track["headsign"] else ""
            payload_tracks.append({
                "t": track["t"],
                "c": track["c"],
                "color": line_colors.get(str(track["line"]), "#e6194b"),
                "popup": (
                    f"<b>Linia {track['line']}{headsign}</b><br>{track['tram_name']}"
                    f"<br>Maks. opóźnienie: {track['max_delay_s'] / 60:.1f} min"
                ),
            })
        self.data = json.dumps({
            "start": [start_time.year, start_time.month, start_time.day,
                      start_time.hour, start_time.minute, start_time.second],
            "duration": int(duration_s),
            "speeds": speeds,
            "defaultSpeed": default_speed,
            "tracks": payload_tracks,
        }, separators=(",", ":"), ensure_ascii=False)
