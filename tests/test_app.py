import app
from pipeline import Receiver
from tracker import AircraftStore


def _views():
    s = AircraftStore(min_messages=1)
    s.update({"icao": "738A1B", "callsign": "ELY315", "altitude": 36000, "groundspeed": 450,
              "latitude": 32.3, "longitude": 34.9}, 100.0)
    s.update({"icao": "4X0001", "callsign": "4XCGA", "altitude": 2500,
              "latitude": 32.1, "longitude": 34.8}, 100.0)
    s.update({"icao": "FAR001", "callsign": "THY5", "latitude": 36.0, "longitude": 34.9}, 100.0)
    s.update({"icao": "NOPOS1", "callsign": "WZZ1"}, 100.0)
    return s.snapshot(now=101.0)


def _trace(fig, prefix):
    return next(t for t in fig.data if t.name.startswith(prefix))


def test_figure_splits_commercial_and_other_and_drops_out_of_range():
    fig = app.build_figure(_views(), show_other=True)
    com = _trace(fig, "Commercial")
    assert len(com.lat) == 1 and com.customdata[0][0] == "ELY315"
    assert "36,000 ft" in com.customdata[0] and "450 kt" in com.customdata[0]
    assert len(_trace(fig, "Other").lat) == 1


def test_figure_hides_other_when_toggled_off():
    fig = app.build_figure(_views(), show_other=False)
    assert not any(t.name.startswith("Other") for t in fig.data)


def test_status_and_table_render():
    views = _views()
    status = {"sdr_state": "disconnected", "sdr_error": "OSError: no device", "messages_per_s": 0.0,
              "crc_ok": 0, "crc_fixed": 0, "crc_bad": 0}
    text = app.build_status(status, views)
    assert "no device" in text and "2 aircraft" in text and "1 commercial" in text
    table = app.build_table(views, show_other=True, now=101.0)
    assert len(table.children[1].children) == 3


def test_sdr_badge_colour_follows_state():
    up = app.build_sdr_badge({"sdr_state": "connected", "sdr_error": None})[0]
    down = app.build_sdr_badge({"sdr_state": "disconnected", "sdr_error": "OSError: no device"})[0]
    assert "connected" in up.children[1] and up.style["background"] == app.SDR_BADGE["connected"][0]
    assert down.style["background"] == app.SDR_BADGE["disconnected"][0] and down.title == "OSError: no device"


def test_user_map_view_survives_refresh():
    view = app.merge_map_view({"map.center": {"lat": 31.0, "lon": 35.0}, "map.zoom": 9,
                               "map._derived": {}}, None)
    assert view == {"center": {"lat": 31.0, "lon": 35.0}, "zoom": 9}
    assert app.merge_map_view({"autosize": True}, view) is None
    assert app.merge_map_view({"map.zoom": 11}, view)["center"] == {"lat": 31.0, "lon": 35.0}
    fig = app.build_figure(_views(), view=view)
    assert fig.layout.map.zoom == 9 and fig.layout.map.center.lat == 31.0


def test_create_app_layout():
    dash_app = app.create_app(Receiver())
    ids = {getattr(c, "id", None) for c in dash_app.layout.children}
    assert {"status", "map", "table", "tick"} <= ids


def test_estimated_positions_are_faded_and_labelled():
    s = AircraftStore(min_messages=1, estimate_after_s=3)
    s.update({"icao": "LIVE01", "callsign": "ELY1", "latitude": 32.2, "longitude": 34.9}, 100.0)
    s.update({"icao": "OLD001", "callsign": "ELY2", "latitude": 32.3, "longitude": 34.9,
              "groundspeed": 400, "track": 90.0}, 90.0)
    fig = app.build_figure(s.snapshot(now=101.0))
    com = _trace(fig, "Commercial")
    by_cs = {c[0]: (op, c[8]) for op, c in zip(com.marker.opacity, com.customdata)}
    assert by_cs["ELY1"] == (1.0, "live")
    assert by_cs["ELY2"][0] < 1.0 and by_cs["ELY2"][1] == "estimated, last fix 11 s ago"
