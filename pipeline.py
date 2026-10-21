from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from dataclasses import dataclass

import config
from acquire import SDRSource
from decode import Demodulator, MessageParser
from tracker import AircraftStore

log = logging.getLogger(__name__)


@dataclass
class _Block:
    samples: object
    timestamp: float
    after_gap: bool


class Receiver:
    def __init__(self, store: AircraftStore | None = None, source_factory=SDRSource):
        self.store = store if store is not None else AircraftStore()
        self._source_factory = source_factory
        self._queue: queue.Queue[_Block] = queue.Queue(maxsize=config.IQ_QUEUE_MAX_BLOCKS)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.demod = Demodulator()
        self.parser = MessageParser()

        self._status_lock = threading.Lock()
        self._sdr_state = "starting"
        self._sdr_error: str | None = None
        self.blocks_dropped = 0
        self.decoder_errors = 0
        self._rate_hist: collections.deque = collections.deque(maxlen=64)


    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._acquire_loop, name="sdr-acquire", daemon=True),
            threading.Thread(target=self._decode_loop, name="modes-decode", daemon=True),
        ]
        for t in self._threads:
            t.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout)
        self._threads = []


    def _set_sdr_state(self, state: str, error: str | None = None) -> None:
        with self._status_lock:
            self._sdr_state, self._sdr_error = state, error

    def _acquire_loop(self) -> None:
        gap = True
        while not self._stop.is_set():
            source = self._source_factory()
            try:
                source.open()
                self._set_sdr_state("connected")
                while not self._stop.is_set():
                    samples = source.read_block()
                    ts = time.time() - len(samples) / config.SAMPLE_RATE_HZ if len(samples) else time.time()
                    self._enqueue(_Block(samples, ts, gap))
                    gap = False
            except Exception as exc:
                self._set_sdr_state("disconnected", f"{type(exc).__name__}: {exc}")
                log.warning("SDR unavailable (%s); retrying in %g s", exc, config.SDR_RETRY_SECONDS)
            finally:
                source.close()
            gap = True
            self._stop.wait(config.SDR_RETRY_SECONDS)

    def _enqueue(self, block: _Block) -> None:
        try:
            self._queue.put_nowait(block)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        self.blocks_dropped += 1
        with self._queue.mutex:
            if self._queue.queue:
                self._queue.queue[0].after_gap = True
        try:
            self._queue.put_nowait(block)
        except queue.Full:
            self.blocks_dropped += 1

    def _decode_loop(self) -> None:
        while not self._stop.is_set():
            try:
                block = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if block.after_gap:
                    self.demod.reset()
                for frame in self.demod.process(block.samples, block.timestamp):
                    msg = self.parser.parse(frame)
                    if msg is not None:
                        self.store.update(msg, frame.timestamp, frame.signal_dbfs)
            except Exception:
                self.decoder_errors += 1
                self.demod.reset()
                log.exception("decoder error; block discarded")


    def status(self) -> dict:
        now = time.monotonic()
        s = self.demod.stats
        good = s.crc_ok + s.crc_fixed
        self._rate_hist.append((now, good))
        t0, g0 = next(((t, g) for t, g in self._rate_hist if now - t <= 30), (now, good))
        rate = (good - g0) / (now - t0) if now - t0 > 0.5 else 0.0
        with self._status_lock:
            state, err = self._sdr_state, self._sdr_error
        return {
            "sdr_state": state,
            "sdr_error": err,
            "messages_per_s": rate,
            "blocks_dropped": self.blocks_dropped,
            "decoder_errors": self.decoder_errors,
            "parse_errors": self.parser.errors,
            **s.as_dict(),
        }
