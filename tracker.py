from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace

import config
import geofilter


@dataclass
class Aircraft:
    icao: str
    first_seen: float
    last_seen: float
    messages: int = 0
    callsign: str | None = None
    emitter_category: str | None = None
    altitude_ft: int | None = None
    groundspeed_kt: float | None = None
    airspeed_kt: float | None = None
    track_deg: float | None = None
    track_is_true: bool = False
    vertical_rate_fpm: int | None = None
    lat: float | None = None
    lon: float | None = None
    position_time: float | None = None
    signal_dbfs: float | None = None

    @property
    def speed_kt(self) -> float | None:
        return self.groundspeed_kt if self.groundspeed_kt is not None else self.airspeed_kt


@dataclass(frozen=True)
class TrackedView:
    aircraft: Aircraft
    distance_km: float | None
    in_range: bool
    classification: geofilter.Classification
    position_age_s: float | None = None
    estimated: bool = False


class AircraftStore:
    def __init__(self, ttl_s: float = config.AIRCRAFT_TTL_S,
                 position_ttl_s: float = config.POSITION_TTL_S,
                 min_messages: int = config.MIN_MESSAGES_TO_DISPLAY,
                 estimate_after_s: float = config.POSITION_ESTIMATE_AFTER_S):
        self.ttl_s = ttl_s
        self.position_ttl_s = position_ttl_s
        self.estimate_after_s = estimate_after_s
        self.min_messages = min_messages
        self._lock = threading.Lock()
        self._aircraft: dict[str, Aircraft] = {}

    def update(self, msg: dict, timestamp: float, signal_dbfs: float | None = None) -> None:
        icao = msg.get("icao")
        if not icao:
            return
        with self._lock:
            ac = self._aircraft.get(icao)
            if ac is None:
                ac = self._aircraft[icao] = Aircraft(icao=icao, first_seen=timestamp, last_seen=timestamp)
            ac.last_seen = max(ac.last_seen, timestamp)
            ac.messages += 1
            if signal_dbfs is not None:
                ac.signal_dbfs = signal_dbfs

            tc = msg.get("typecode")
            if msg.get("callsign"):
                ac.callsign = msg["callsign"].strip("_ ").strip() or ac.callsign
                cat = msg.get("category")
                if isinstance(tc, int) and 1 <= tc <= 4 and cat:
                    ac.emitter_category = f"{'DCBA'[tc - 1]}{cat}"
            if msg.get("altitude") is not None:
                ac.altitude_ft = msg["altitude"]
            if msg.get("groundspeed") is not None:
                ac.groundspeed_kt = msg["groundspeed"]
            if msg.get("airspeed") is not None:
                ac.airspeed_kt = msg["airspeed"]
            if msg.get("track") is not None:
                ac.track_deg, ac.track_is_true = msg["track"], True
            elif msg.get("heading") is not None and ac.track_deg is None:
                ac.track_deg = msg["heading"]
            if msg.get("vertical_rate") is not None:
                ac.vertical_rate_fpm = msg["vertical_rate"]
            if msg.get("latitude") is not None and msg.get("longitude") is not None:
                ac.lat, ac.lon = msg["latitude"], msg["longitude"]
                ac.position_time = timestamp

    def snapshot(self, now: float | None = None) -> list[TrackedView]:
        now = time.time() if now is None else now
        with self._lock:
            for icao in [k for k, a in self._aircraft.items() if now - a.last_seen > self.ttl_s]:
                del self._aircraft[icao]
            aircraft = [replace(a) for a in self._aircraft.values() if a.messages >= self.min_messages]

        views = []
        for ac in aircraft:
            age = None if ac.position_time is None else max(0.0, now - ac.position_time)
            estimated = False
            if age is None or age > self.position_ttl_s:
                ac.lat = ac.lon = None
                age = None
            elif age > self.estimate_after_s:
                estimated = True
                if ac.groundspeed_kt and ac.track_is_true and ac.track_deg is not None:
                    ac.lat, ac.lon = geofilter.destination_point(
                        ac.lat, ac.lon, ac.track_deg, ac.groundspeed_kt * 1.852 / 3600 * age)
            dist = None if ac.lat is None else geofilter.distance_from_base_km(ac.lat, ac.lon)
            views.append(TrackedView(
                aircraft=ac,
                distance_km=dist,
                in_range=dist is not None and dist <= config.RADIUS_KM,
                classification=geofilter.classify(ac.callsign, ac.emitter_category),
                position_age_s=age,
                estimated=estimated,
            ))
        return views

    def __len__(self) -> int:
        with self._lock:
            return len(self._aircraft)
