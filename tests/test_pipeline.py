import time

import numpy as np

import config
from pipeline import Receiver, _Block
from synth import make_iq
from tracker import AircraftStore
from test_decode import IDENT_KLM, POS_EVEN, POS_ODD, VEL_GS


def _wait(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


class FakeSource:
    def __init__(self, blocks):
        self.blocks = list(blocks)
        self.closed = False

    def open(self):
        pass

    def read_block(self):
        if not self.blocks:
            raise OSError("device unplugged")
        return self.blocks.pop(0)

    def close(self):
        self.closed = True


def test_receiver_decodes_into_store_and_survives_unplug(monkeypatch):
    monkeypatch.setattr(config, "SDR_RETRY_SECONDS", 0.05)
    frames = [(IDENT_KLM, 200), (VEL_GS, 900)] + [
        (POS_EVEN if i % 2 == 0 else POS_ODD, 1600 + i * 700) for i in range(8)]
    iq = make_iq(frames, 8000, noise_rms=0.01)
    blocks = [iq[i:i + 1000] for i in range(0, iq.size, 1000)] + [np.zeros(2000, np.complex64)]
    sources = [FakeSource(blocks)]
    rx = Receiver(store=AircraftStore(min_messages=1),
                  source_factory=lambda: sources.pop(0) if sources else FakeSource([]))
    rx.start()
    try:
        assert _wait(lambda: rx.status()["sdr_state"] == "disconnected")
        assert _wait(lambda: rx.status()["crc_ok"] == len(frames) and len(rx.store.snapshot()) == 3)
        views = {v.aircraft.icao: v for v in rx.store.snapshot()}
        assert views["4840D6"].aircraft.callsign == "KLM1023"
        assert views["485020"].aircraft.groundspeed_kt == 159
        assert "OSError" in rx.status()["sdr_error"]
    finally:
        rx.stop()


def test_receiver_reports_missing_dongle(monkeypatch):
    monkeypatch.setattr(config, "SDR_RETRY_SECONDS", 0.05)

    class Missing(FakeSource):
        def open(self):
            raise OSError("Could not open SDR (device index = 0)")

    rx = Receiver(source_factory=lambda: Missing([]))
    rx.start()
    try:
        assert _wait(lambda: rx.status()["sdr_state"] == "disconnected")
        assert "Could not open SDR" in rx.status()["sdr_error"]
    finally:
        rx.stop()


def test_queue_overflow_drops_oldest_and_marks_gap():
    rx = Receiver(source_factory=lambda: FakeSource([]))
    for i in range(config.IQ_QUEUE_MAX_BLOCKS + 3):
        rx._enqueue(_Block(np.zeros(10, np.complex64), float(i), False))
    assert rx.blocks_dropped == 3
    assert rx._queue.queue[0].after_gap and rx._queue.queue[0].timestamp == 3.0
