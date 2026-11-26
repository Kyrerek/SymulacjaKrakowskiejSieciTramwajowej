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


def test_shared_track_prevents_overtaking_across_different_end_stops():
    """Tramwaj jadący do B2 nie wyprzedzi wolniejszego tramwaju jadącego tym samym torem do B1."""
    start_dt = datetime(2026, 10, 27, 12, 0, 0)
    stop_dt = datetime(2026, 10, 27, 12, 4, 0)

    # Wspólny korytarz ~333 m na północ, następnie rozjazd do B1 (pętla) i B2 (przelot)
    stop_a = _make_stop("A", lat=50.000, lon=20.000)
    stop_b1 = _make_stop("B1", lat=50.003, lon=20.0003)
    stop_b2 = _make_stop("B2", lat=50.0035, lon=20.000)

    shared_pts = [Point(50.000, 20.000), Point(50.0015, 20.000), Point(50.0028, 20.000)]
    road_slow = StopRoad(stop_a, stop_b1, [*shared_pts, Point(stop_b1.lat, stop_b1.lon)], delay=lambda t: 180.0)
    road_fast = StopRoad(stop_a, stop_b2, [*shared_pts, Point(stop_b2.lat, stop_b2.lon)], delay=lambda t: 30.0)
    stop_a.add_roads([road_slow, road_fast])

    slow_tram = Tram(1, [stop_a, stop_b1], start_dt, 1, "Slow", delay=lambda t, s: 10.0)
    fast_tram = Tram(2, [stop_a, stop_b2], start_dt + timedelta(seconds=10), 1, "Fast", delay=lambda t, s: 10.0)

    samples = TramMotion(start_dt, stop_dt, target_time_step=5).run_trams([slow_tram, fast_tram])

    # Na wspólnym odcinku (t=40..120s) szybszy tramwaj musi czekać za wolniejszym (nie może mieć większego lat)
    for sec in range(40, 125, 5):
        ts = start_dt + timedelta(seconds=sec)
        s_slow = next(s for s in samples if s.tram_id == "1" and s.timestamp == ts)
        s_fast = next(s for s in samples if s.tram_id == "2" and s.timestamp == ts)
        if s_slow.status != TramStatus.TERMINATED and s_fast.status != TramStatus.TERMINATED:
            assert s_fast.lat <= s_slow.lat + 1e-6


def test_opposite_direction_tracks_do_not_block_each_other():
    """Sąsiednie tory w przeciwnych kierunkach (~3 m obok siebie) nie blokują się wzajemnie."""
    start_dt = datetime(2026, 10, 27, 12, 0, 0)
    stop_dt = datetime(2026, 10, 27, 12, 2, 0)

    a_n = _make_stop("AN", lat=50.000, lon=20.00000)
    b_n = _make_stop("BN", lat=50.003, lon=20.00000)
    b_s = _make_stop("BS", lat=50.003, lon=20.00004)  # ~2.8 m w bok
    a_s = _make_stop("AS", lat=50.000, lon=20.00004)

    a_n.add_roads([StopRoad(a_n, b_n, [Point(a_n.lat, a_n.lon), Point(b_n.lat, b_n.lon)], delay=lambda t: 30.0)])
    b_s.add_roads([StopRoad(b_s, a_s, [Point(b_s.lat, b_s.lon), Point(a_s.lat, a_s.lon)], delay=lambda t: 30.0)])

    t_north = Tram(1, [a_n, b_n], start_dt, 1, "N", delay=lambda t, s: 5.0)
    t_south = Tram(2, [b_s, a_s], start_dt, 1, "S", delay=lambda t, s: 5.0)

    samples = TramMotion(start_dt, stop_dt, target_time_step=5).run_trams([t_north, t_south])
    assert all(s.status != TramStatus.WAIT for s in samples)
