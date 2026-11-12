"""Testy jednostkowe modułu realtime_data (syntetyczne pliki GTFS-RT, bez sieci)."""
import tarfile
import zipfile

import pandas as pd
import pytest
from google.transit import gtfs_realtime_pb2 as rt

import realtime_data as rd

DAY = "2026-10-19"
START = rd.service_day_start(DAY).timestamp()


# ------------------------------------------------------------------ pomocnicze
def vp_feed(header_ts, vehicles):
    """vehicles: lista (vehicle_id, trip_id, vehicle_ts, status, seq, stop_id)."""
    f = rt.FeedMessage()
    f.header.gtfs_realtime_version = "2.0"
    f.header.timestamp = int(header_ts)
    for i, (vid, trip, ts, status, seq, stop) in enumerate(vehicles):
        e = f.entity.add()
        e.id = str(i)
        v = e.vehicle
        v.vehicle.id = vid
        v.trip.trip_id = trip
        v.timestamp = int(ts)
        v.current_status = status
        v.current_stop_sequence = seq
        v.stop_id = stop
        v.position.latitude, v.position.longitude = 50.06, 19.94
    return f.SerializeToString()


def write_pb(folder, fetched, last_mod, raw):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{fetched}_{last_mod}.pb").write_bytes(raw)


STOPPED, TRANSIT = rt.VehiclePosition.STOPPED_AT, rt.VehiclePosition.IN_TRANSIT_TO


@pytest.fixture
def static():
    return rd.StaticSchedule(
        stop_times=pd.DataFrame({
            "trip_id": ["T1", "T1", "NIGHT", "NIGHT"],
            "stop_sequence": [1, 2, 1, 2],
            "stop_id": ["A", "B", "A", "B"],
            "arrival_s": [28800, 28920, 90000, 90120],
            "departure_s": [28800, 28920, 90000, 90120],
        }),
        trips=pd.DataFrame({"trip_id": ["T1", "NIGHT"], "route_short_name": ["1", "62"]}),
        version="test",
    )


@pytest.fixture
def vp_t1(tmp_path):
    """Kurs T1: 60 s spóźnienia; postój A (3 raporty), przejazd, postój B (2 raporty)."""
    base = START + 28800 + 60
    rows = [("V1", "T1", base + 0, STOPPED, 1, "A"), ("V1", "T1", base + 10, STOPPED, 1, "A"),
            ("V1", "T1", base + 20, STOPPED, 1, "A"), ("V1", "T1", base + 50, TRANSIT, 2, "B"),
            ("V1", "T1", base + 120, STOPPED, 2, "B"), ("V1", "T1", base + 130, STOPPED, 2, "B")]
    for i, r in enumerate(rows):
        write_pb(tmp_path, int(r[2]) + 2, int(r[2]), vp_feed(r[2], [r]))
    return rd.parse_vehicle_positions(tmp_path), base


# ---------------------------------------------------------------- czytanie plików
def test_iter_feed_files_reads_directory_and_tar(tmp_path):
    d = tmp_path / "dir"
    write_pb(d, 100, 98, b"x")
    write_pb(d, 110, 108, b"y")
    (d / "ignorowany.txt").write_text("nie pb")
    tar_path = tmp_path / "a.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(d / "100_98.pb", arcname="2026-10-19/100_98.pb")

    assert [(f, m) for f, m, _ in rd.iter_feed_files(d)] == [(100, 98), (110, 108)]
    assert [(f, m, b) for f, m, b in rd.iter_feed_files(tar_path)] == [(100, 98, b"x")]


def test_parse_vehicle_positions_skips_corrupt_and_removes_duplicates(tmp_path):
    ok = vp_feed(1000, [("V1", "T1", 995, STOPPED, 1, "A")])
    write_pb(tmp_path, 1001, 1000, ok)
    write_pb(tmp_path, 1011, 1010, ok)                 # ten sam raport pojazdu
    write_pb(tmp_path, 1021, 1020, ok[:5])            # ucięty plik

    df = rd.parse_vehicle_positions(tmp_path)

    assert len(df) == 1
    assert df.attrs["corrupt_files"] == 1
    assert df.loc[0, "status"] == "STOPPED_AT"


def test_file_report_counts_corrupt_and_gaps(tmp_path):
    ok = vp_feed(1000, [("V1", "T1", 995, STOPPED, 1, "A")])
    for lm in (1000, 1010, 1020, 1200):                # luka 180 s
        write_pb(tmp_path, lm + 1, lm, ok)
    write_pb(tmp_path, 1300, 1299, b"\xff\xff")

    r = rd.file_report(tmp_path)

    assert r["files"] == 5 and r["corrupt"] == 1
    assert r["interval_median_s"] == 10 and r["gaps_over_60s"] == 1


# ------------------------------------------------------------------ doba usługowa
def test_service_day_start_is_local_midnight_and_handles_dst():
    assert rd.service_day_start("2026-10-19").timestamp() == pd.Timestamp("2026-10-18 22:00", tz="UTC").timestamp()
    # w dniu zmiany czasu (25.10) doba GTFS zaczyna się o 23:00 UTC dnia poprzedniego
    assert rd.service_day_start("2026-10-25").timestamp() == pd.Timestamp("2026-10-24 23:00", tz="UTC").timestamp()


def test_assign_service_date_handles_trips_after_midnight(static):
    night_ts = pd.Timestamp("2026-10-20 00:30", tz="Europe/Warsaw").timestamp()
    day_ts = pd.Timestamp("2026-10-20 08:30", tz="Europe/Warsaw").timestamp()
    df = pd.DataFrame({"trip_id": ["NIGHT", "T1"], "vehicle_ts": [night_ts, day_ts]})

    result = rd.assign_service_date(df, static)

    assert result.tolist() == [pd.Timestamp("2026-10-19"), pd.Timestamp("2026-10-20")]


def test_assign_service_date_fallback_without_static():
    ts = pd.Timestamp("2026-10-20 01:30", tz="Europe/Warsaw").timestamp()
    df = pd.DataFrame({"trip_id": ["X"], "vehicle_ts": [ts]})

    assert rd.assign_service_date(df).iloc[0] == pd.Timestamp("2026-10-19")


# ------------------------------------------------------------- postoje, opóźnienia
def test_stop_events_dwell_and_delay_against_schedule(vp_t1, static):
    vp, base = vp_t1

    ev = rd.build_stop_events(vp, static)

    a, b = ev[ev["stop_id"] == "A"].iloc[0], ev[ev["stop_id"] == "B"].iloc[0]
    assert a["arrival_ts"] == base - 5 and a["departure_ts"] == base + 25
    assert a["dwell_s"] == 30 and a["n_samples"] == 3
    assert a["delay_arrival_s"] == 55 and a["delay_departure_s"] == 85
    assert b["delay_arrival_s"] == 55
    assert a["route_short_name"] == "1" and bool(a["stop_match"])


def test_stop_events_without_static_have_no_delay_columns(vp_t1):
    vp, _ = vp_t1

    ev = rd.build_stop_events(vp)

    assert "delay_arrival_s" not in ev.columns
    assert len(ev) == 2


def test_segment_observations_travel_time_and_extra_vs_plan(vp_t1, static):
    vp, base = vp_t1
    seg = rd.build_segment_observations(rd.build_stop_events(vp, static))

    assert len(seg) == 1
    row = seg.iloc[0]
    assert (row["stop_from"], row["stop_to"]) == ("A", "B")
    assert row["travel_s"] == 90 and row["planned_travel_s"] == 120 and row["extra_s"] == -30
    assert row["tod_s"] == 28800 + 60 + 25
    assert bool(row["valid"])


def test_sampler_data_keeps_only_positive_valid_values(vp_t1, static):
    vp, _ = vp_t1
    seg = rd.build_segment_observations(rd.build_stop_events(vp, static))

    assert rd.sampler_data(seg) == {("A", "B"): [(28885.0, 90.0)]}
    assert rd.sampler_data(seg, value="extra_s") == {}      # extra_s < 0: odrzucone


# ----------------------------------------------------------- zgodność z rozkładem
def test_check_static_match_detects_wrong_schedule_version(vp_t1, static):
    vp, _ = vp_t1
    assert rd.check_static_match(vp, static)["ok"]

    other = rd.StaticSchedule(static.stop_times.assign(stop_id=["X", "Y", "X", "Y"]), static.trips)
    report = rd.check_static_match(vp, other)

    assert report["trip_id_coverage"] == 1.0       # identyfikatory kursów się zgadzają...
    assert report["stop_agreement"] == 0.0         # ...ale to inny rozkład
    assert not report["ok"]


def _zip_with_calendar(path, start_date):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("calendar.txt", f"service_id,start_date,end_date\ns1,{start_date},20270101\n")


def test_pick_static_snapshot_chooses_version_valid_on_day(tmp_path):
    old = tmp_path / "GTFS_KRK_T_2026-10-09_120000.zip"
    new = tmp_path / "GTFS_KRK_T_2026-10-22_180000.zip"
    _zip_with_calendar(old, "20261010")
    _zip_with_calendar(new, "20261023")

    assert rd.pick_static_snapshot("2026-10-19", tmp_path) == old
    assert rd.pick_static_snapshot("2026-10-22", tmp_path) == old   # nowy jeszcze nie obowiązuje
    assert rd.pick_static_snapshot("2026-10-23", tmp_path) == new
    with pytest.raises(ValueError):
        rd.pick_static_snapshot("2026-10-01", tmp_path)


# --------------------------------------------------------- trip updates i alerty
def test_parse_trip_updates_keeps_first_and_last_prediction(tmp_path):
    def feed(header_ts, arr):
        f = rt.FeedMessage()
        f.header.gtfs_realtime_version, f.header.timestamp = "2.0", header_ts

        entity = f.entity.add()
        entity.id = "trip_update_1"

        tu = entity.trip_update
        tu.trip.trip_id, tu.vehicle.id = "T1", "V1"
        u = tu.stop_time_update.add()
        u.stop_id = "B"
        u.arrival.time = arr
        u.departure.time = arr + 20
        return f.SerializeToString()

    t0 = int(START + 28800)
    write_pb(tmp_path, t0, t0 - 1, feed(t0, t0 + 100))
    write_pb(tmp_path, t0 + 10, t0 + 9, feed(t0 + 10, t0 + 160))

    tu = rd.parse_trip_updates(tmp_path)

    assert len(tu) == 1
    row = tu.iloc[0]
    assert row["first_arrival_ts"] == t0 + 100 and row["last_arrival_ts"] == t0 + 160
    assert row["n_obs"] == 2 and row["service_date"] == pd.Timestamp(DAY)


def test_parse_alerts_flags_test_entries(tmp_path):
    f = rt.FeedMessage()
    f.header.gtfs_realtime_version, f.header.timestamp = "2.0", 1000
    for i, text in enumerate(["TEST_1 EBBLO", "Objazd linii 1"]):
        a = f.entity.add()
        a.id = f"alert{i}"
        a.alert.description_text.translation.add().text = text
    write_pb(tmp_path, 1001, 1000, f.SerializeToString())

    df = rd.parse_alerts(tmp_path)

    assert df.set_index("alert_id")["is_test"].to_dict() == {"alert0": True, "alert1": False}