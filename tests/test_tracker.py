import pytest

import geofilter
from tracker import AircraftStore


def _store(**kw):
    return AircraftStore(**{"ttl_s": 60, "position_ttl_s": 30, "min_messages": 1, **kw})


def test_messages_merge_into_one_aircraft():
    s = _store()
    s.update({"icao": "738A1B", "typecode": 4, "category": 3, "callsign": "ELY315__"}, 100.0)
    s.update({"icao": "738A1B", "altitude": 36000, "latitude": 32.3, "longitude": 34.9}, 101.0, -12.0)
    s.update({"icao": "738A1B", "groundspeed": 450, "track": 270.0, "vertical_rate": -64}, 102.0)
    (v,) = s.snapshot(now=103.0)
    a = v.aircraft
    assert (a.callsign, a.emitter_category, a.altitude_ft, a.speed_kt) == ("ELY315", "A3", 36000, 450)
    assert a.messages == 3 and a.signal_dbfs == -12.0
    assert v.in_range and v.distance_km < 30 and v.classification.is_commercial


def test_airspeed_used_when_no_groundspeed():
    s = _store()
    s.update({"icao": "A", "airspeed": 300, "heading": 90.0}, 0.0)
    (v,) = s.snapshot(now=1.0)
    assert v.aircraft.speed_kt == 300 and v.aircraft.track_deg == 90.0


def test_out_of_range_position_is_flagged_not_dropped():
    s = _store()
    s.update({"icao": "B", "latitude": 35.5, "longitude": 34.9}, 0.0)
    (v,) = s.snapshot(now=1.0)
    assert not v.in_range and v.distance_km > 200


def test_expiry_and_stale_position():
    s = _store()
    s.update({"icao": "C", "latitude": 32.0, "longitude": 34.8}, 0.0)
    s.update({"icao": "C", "altitude": 1000}, 40.0)
    (v,) = s.snapshot(now=41.0)
    assert v.aircraft.lat is None and not v.in_range
    assert s.snapshot(now=200.0) == [] and len(s) == 0


def test_min_messages_hides_single_frame_contacts():
    s = _store(min_messages=2)
    s.update({"icao": "D"}, 0.0)
    assert s.snapshot(now=0.5) == []
    s.update({"icao": "D"}, 0.6)
    assert len(s.snapshot(now=1.0)) == 1


def test_message_without_icao_is_ignored():
    s = _store()
    s.update({"df": 17}, 0.0)
    assert len(s) == 0


def test_fresh_position_is_live_and_unmoved():
    s = _store(estimate_after_s=3)
    s.update({"icao": "E", "latitude": 32.0, "longitude": 34.8, "groundspeed": 360, "track": 0.0}, 0.0)
    (v,) = s.snapshot(now=2.0)
    assert not v.estimated and v.position_age_s == 2.0
    assert (v.aircraft.lat, v.aircraft.lon) == (32.0, 34.8)


def test_stale_position_is_dead_reckoned_along_track():
    s = _store(estimate_after_s=3)
    s.update({"icao": "F", "latitude": 32.0, "longitude": 34.8}, 0.0)
    s.update({"icao": "F", "groundspeed": 360, "track": 0.0}, 15.0)
    (v,) = s.snapshot(now=20.0)
    assert v.estimated and v.position_age_s == 20.0
    moved = geofilter.haversine_km(32.0, 34.8, v.aircraft.lat, v.aircraft.lon)
    assert moved == pytest.approx(360 * 1.852 / 3600 * 20, rel=1e-3)
    assert v.aircraft.lat > 32.0 and v.aircraft.lon == pytest.approx(34.8, abs=1e-9)


def test_stale_position_without_true_track_is_flagged_but_not_moved():
    s = _store(estimate_after_s=3)
    s.update({"icao": "G", "latitude": 32.0, "longitude": 34.8, "airspeed": 300, "heading": 90.0}, 0.0)
    (v,) = s.snapshot(now=10.0)
    assert v.estimated and (v.aircraft.lat, v.aircraft.lon) == (32.0, 34.8)
