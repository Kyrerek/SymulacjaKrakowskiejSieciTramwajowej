from datetime import datetime, timedelta
from models import Point, Stop, StopRoad, Tram, TramStatus
from motion import TramMotion


def _make_stop(label: str, lat: float = 50.0, lon: float = 20.0) -> Stop:
    return Stop(label, lat, lon, name=label)


def test_stop_occupation_state():
    stop = _make_stop("A")
    tram = Tram(
        id_=1,
        route=[stop],
        start_time=datetime(2026, 10, 27, 12, 0),
        loop_route_num=1,
        name="T1",
    )

    assert stop.is_occupied is False
    assert stop.occupant is None

    stop.reserve(tram)
    assert stop.is_occupied is True
    assert stop.occupant is tram

    stop.release()
    assert stop.is_occupied is False
    assert stop.occupant is None


def test_second_tram_waits_in_queue_block_when_stop_is_occupied():
    start_dt = datetime(2026, 10, 27, 12, 0, 0)
    stop_dt = datetime(2026, 10, 27, 12, 1, 0)

    # Dwa przystanki oddalone o ~111 m (0.001 stopnia szerokości geograficznej)
    stop_a = _make_stop("A", lat=50.000, lon=20.0)
    stop_b = _make_stop("B", lat=50.001, lon=20.0)

    road_ab = StopRoad(stop_a, stop_b, [Point(50.000, 20.0), Point(50.001, 20.0)], delay=lambda t: 10.0)
    road_ba = StopRoad(stop_b, stop_a, [Point(50.001, 20.0), Point(50.000, 20.0)], delay=lambda t: 10.0)
    stop_a.add_roads([road_ab])
    stop_b.add_roads([road_ba])

    # Tramwaj 1 dojeżdża na B w 10s i stoi tam przez 40s (od t=10s do t=50s)
    tram_1 = Tram(
        id_=1,
        route=[stop_a, stop_b],
        start_time=start_dt,
        loop_route_num=1,
        name="T1",
        delay=lambda time, pax: 40.0,
    )
    # Tramwaj 2 rusza 5s później -> podjeżdża pod B w t=15s, gdy T1 wciąż zajmuje peron
    tram_2 = Tram(
        id_=2,
        route=[stop_a, stop_b],
        start_time=start_dt + timedelta(seconds=5),
        loop_route_num=1,
        name="T2",
        delay=lambda time, pax: 5.0,
    )

    samples = TramMotion(start_time=start_dt, stop_time=stop_dt, target_time_step=5).run_trams([tram_1, tram_2])

    # W 25. sekundzie symulacji T1 stoi na peronie (DWEL), a T2 czeka w kolejce przed przystankiem (WAIT)
    t25 = start_dt + timedelta(seconds=25)
    t1_at_25 = next(s for s in samples if s.tram_id == "1" and s.timestamp == t25)
    t2_at_25 = next(s for s in samples if s.tram_id == "2" and s.timestamp == t25)

    assert t1_at_25.status == TramStatus.DWEL
    assert t2_at_25.status == TramStatus.WAIT
    # T2 nie może stać w tym samym miejscu co T1 (musi być zatrzymany ok. 20 m przed peronem)
    assert Point(t2_at_25.lat, t2_at_25.lon).distance_to(stop_b) >= 15.0