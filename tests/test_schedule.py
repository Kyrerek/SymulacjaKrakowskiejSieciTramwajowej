from datetime import datetime, timedelta

from models import Point, Stop, StopRoad, Tram, TramStatus
from motion import TramMotion
from schedule import ScheduledStop, ScheduledTrip, _trim_block

T0 = datetime(2026, 10, 27, 12, 0, 0)


def _line(a: Stop, b: Stop, travel_s: float) -> StopRoad:
    return StopRoad(a, b, [Point(a.lat, a.lon), Point(b.lat, b.lon)], delay=lambda t: travel_s)


def _trip(trip_id: str, stops: list[Stop], dep_offsets_s: list[int]) -> ScheduledTrip:
    return ScheduledTrip(
        trip_id, "4", "0", "Test",
        [ScheduledStop(s, T0 + timedelta(seconds=o), T0 + timedelta(seconds=o)) for s, o in zip(stops, dep_offsets_s)],
    )


def _network():
    a, b, c = Stop("A", 50.000, 20.0), Stop("B", 50.002, 20.0), Stop("C", 50.004, 20.0)
    a.add_roads([_line(a, b, 60.0)])
    b.add_roads([_line(b, c, 60.0), _line(b, a, 60.0)])
    c.add_roads([_line(c, b, 60.0)])
    return a, b, c


def test_tram_waits_for_scheduled_departure_and_chains_trips():
    a, b, c = _network()
    out = _trip("t1", [a, b, c], [60, 120, 180])
    back = _trip("t2", [c, b, a], [600, 660, 720])
    tram = Tram("blk", route=[a, b, c, c, b, a], start_time=T0, loop_route_num=1,
                delay=lambda t, s: 10.0, schedule=[out, back])

    samples = TramMotion(T0, T0 + timedelta(minutes=20), target_time_step=1).run_trams([tram])

    def status_at(sec):
        return next(s for s in samples if s.timestamp == T0 + timedelta(seconds=sec))

    # Do odjazdu (60 s) stoi na A, potem jedzie.
    assert status_at(30).status == TramStatus.WAIT and status_at(30).lat == a.lat
    assert status_at(90).status == TramStatus.ARRIVAL
    # Po dojeździe do C czeka na pętli do odjazdu drugiego kursu (600 s), nie rusza wcześniej.
    at_loop = status_at(500)
    assert at_loop.lat == c.lat and at_loop.status == TramStatus.WAIT
    assert status_at(605).status == TramStatus.ARRIVAL
    assert status_at(605).headsign == "Test"
    # Bez opóźnień wejściowych: 60 s jazdy + 10 s postoju vs 60 s w rozkładzie -> 10 s na odcinek.
    assert 0 < tram.delay_s <= 30


def test_trim_block_cuts_trip_in_progress():
    a, b, c = _network()
    trip = _trip("t1", [a, b, c], [-120, 30, 90])
    trimmed = _trim_block([trip], T0, T0 + timedelta(hours=1))
    assert [s.stop.id for s in trimmed[0].stops] == ["B", "C"]
    assert _trim_block([_trip("old", [a, b], [-300, -200])], T0, T0 + timedelta(hours=1)) == []
    assert _trim_block([_trip("late", [a, b], [4000, 4100])], T0, T0 + timedelta(hours=1)) == []
