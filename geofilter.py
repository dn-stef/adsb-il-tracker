from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

import config

EARTH_RADIUS_KM = 6371.0088

COMMERCIAL = "commercial"
OTHER = "other"
UNKNOWN = "unknown"

_AIRLINE_RE = re.compile(r"^([A-Z]{3})(\d[0-9A-Z]{0,3})$")
_REGISTRATION_RE = re.compile(r"^(4X[A-Z]{3}|N\d[0-9A-Z]{0,4}|[A-Z]{1,2}[A-Z]{3,4})$")


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    d = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return float(d) if np.ndim(d) == 0 else d


def distance_from_base_km(lat, lon, base=(config.BASE_LAT, config.BASE_LON)):
    return haversine_km(base[0], base[1], lat, lon)


def within_radius(lat, lon, radius_km: float = config.RADIUS_KM,
                  base=(config.BASE_LAT, config.BASE_LON)) -> bool:
    if lat is None or lon is None:
        return False
    return distance_from_base_km(lat, lon, base) <= radius_km


def circle_points(radius_km: float = config.RADIUS_KM,
                  base=(config.BASE_LAT, config.BASE_LON), n: int = 181):
    lat0, lon0 = map(math.radians, base)
    d = radius_km / EARTH_RADIUS_KM
    brg = np.linspace(0, 2 * np.pi, n)
    lat = np.arcsin(np.sin(lat0) * np.cos(d) + np.cos(lat0) * np.sin(d) * np.cos(brg))
    lon = lon0 + np.arctan2(np.sin(brg) * np.sin(d) * np.cos(lat0),
                            np.cos(d) - np.sin(lat0) * np.sin(lat))
    return np.degrees(lat), (np.degrees(lon) + 540) % 360 - 180


def destination_point(lat: float, lon: float, bearing_deg: float, distance_km: float) -> tuple[float, float]:
    lat1, lon1, brg = map(math.radians, (lat, lon, bearing_deg))
    d = distance_km / EARTH_RADIUS_KM
    lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(brg))
    lon2 = lon1 + math.atan2(math.sin(brg) * math.sin(d) * math.cos(lat1),
                             math.cos(d) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), (math.degrees(lon2) + 540) % 360 - 180


@dataclass(frozen=True)
class Classification:
    label: str
    confidence: str
    reason: str

    @property
    def is_commercial(self) -> bool:
        return self.label == COMMERCIAL


def classify(callsign: str | None, emitter_category: str | None = None) -> Classification:
    cs = (callsign or "").strip().upper().replace("_", "").replace("#", "")
    cat = (emitter_category or "").upper() or None

    if cs and any(cs.startswith(p) for p in config.NON_COMMERCIAL_PREFIXES):
        return Classification(OTHER, "high", "military/government callsign prefix")

    m = _AIRLINE_RE.match(cs)
    if m:
        prefix = m.group(1)
        if cat in config.NON_COMMERCIAL_EMITTER_CATEGORIES:
            return Classification(OTHER, "medium", f"airline-style callsign but emitter category {cat}")
        if prefix in config.COMMERCIAL_AIRLINE_PREFIXES:
            return Classification(COMMERCIAL, "high", f"known airline designator {prefix}")
        if cat in config.COMMERCIAL_EMITTER_CATEGORIES:
            return Classification(COMMERCIAL, "high", f"airline-style callsign, emitter category {cat}")
        return Classification(COMMERCIAL, "medium", "airline-style callsign (designator + flight number)")

    if cs and _REGISTRATION_RE.match(cs):
        return Classification(OTHER, "medium", "registration used as callsign (GA/private)")

    if cat in config.COMMERCIAL_EMITTER_CATEGORIES:
        return Classification(COMMERCIAL, "low", f"emitter category {cat}, no airline callsign")
    if cat in config.NON_COMMERCIAL_EMITTER_CATEGORIES:
        return Classification(OTHER, "medium", f"emitter category {cat}")
    if not cs:
        return Classification(UNKNOWN, "low", "no callsign received yet")
    return Classification(OTHER, "low", "callsign does not match an airline pattern")
