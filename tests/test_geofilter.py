import numpy as np
import pytest

import config
import geofilter
from geofilter import COMMERCIAL, OTHER, UNKNOWN, classify, haversine_km

BASE = (config.BASE_LAT, config.BASE_LON)


def test_haversine_basics():
    assert haversine_km(*BASE, *BASE) == 0
    assert haversine_km(0, 0, 1, 0) == pytest.approx(111.195, abs=0.01)
    assert haversine_km(0, 179.5, 0, -179.5) == pytest.approx(111.195, abs=0.01)


def test_haversine_petah_tikva_to_jerusalem():
    assert haversine_km(*BASE, 31.7683, 35.2137) == pytest.approx(46.4, abs=1.0)


def test_haversine_vectorised():
    d = haversine_km(BASE[0], BASE[1], np.array([BASE[0], BASE[0] + 1]), np.array([BASE[1], BASE[1]]))
    assert d.shape == (2,) and d[0] == 0 and d[1] == pytest.approx(111.195, abs=0.01)


def test_within_radius():
    assert geofilter.within_radius(32.5, 34.9)
    assert not geofilter.within_radius(34.0, 34.9)
    assert not geofilter.within_radius(None, None)
    assert geofilter.within_radius(33.8, 34.888, radius_km=200)
    assert not geofilter.within_radius(33.9, 34.888, radius_km=200)


def test_circle_points_lie_on_radius():
    lat, lon = geofilter.circle_points(200.0)
    assert np.allclose(haversine_km(BASE[0], BASE[1], lat, lon), 200.0, atol=0.01)


@pytest.mark.parametrize("callsign, cat, label", [
    ("ELY315", None, COMMERCIAL),
    ("WZZ4YK", None, COMMERCIAL),
    ("RYR12AB ", None, COMMERCIAL),
    ("ABC123", None, COMMERCIAL),
    ("ABC123", "A5", COMMERCIAL),
    ("ABC123", "A7", OTHER),
    ("RCH871", None, OTHER),
    ("4XCGA", None, OTHER),
    ("N172SP", None, OTHER),
    ("", None, UNKNOWN),
    (None, "A3", COMMERCIAL),
    (None, "B1", OTHER),
    ("12345", None, OTHER),
])
def test_classify(callsign, cat, label):
    assert classify(callsign, cat).label == label


def test_classification_confidence_and_reason():
    c = classify("ELY001")
    assert c.is_commercial and c.confidence == "high" and "ELY" in c.reason
    assert classify("ZZZ1").confidence == "medium"


@pytest.mark.parametrize("bearing", [0, 45, 90, 180, 270, 333])
def test_destination_point_distance_and_bearing(bearing):
    lat, lon = geofilter.destination_point(*BASE, bearing, 25.0)
    assert haversine_km(*BASE, lat, lon) == pytest.approx(25.0, rel=1e-6)
    back = geofilter.destination_point(lat, lon, (bearing + 180) % 360, 25.0)
    assert back == pytest.approx(BASE, abs=0.01)
