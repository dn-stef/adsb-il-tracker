from __future__ import annotations

import logging
import time

import numpy as np

import config

log = logging.getLogger(__name__)


def bytes_to_iq(raw) -> np.ndarray:
    u8 = raw if isinstance(raw, np.ndarray) else np.frombuffer(raw, dtype=np.uint8)
    if u8.size % 2:
        u8 = u8[:-1]
    f = (u8.astype(np.float32) - 127.5) / 127.5
    return (f[0::2] + 1j * f[1::2]).astype(np.complex64)


class SDRSource:
    def __init__(
        self,
        device_index: int = config.SDR_DEVICE_INDEX,
        center_freq_hz: float = config.CENTER_FREQ_HZ,
        sample_rate_hz: float = config.SAMPLE_RATE_HZ,
        gain_db: float | str = config.SDR_GAIN_DB,
        ppm: int = config.SDR_PPM_CORRECTION,
        bias_tee: bool = config.SDR_BIAS_TEE,
        samples_per_block: int = config.SAMPLES_PER_BLOCK,
    ):
        self.device_index = device_index
        self.center_freq_hz = center_freq_hz
        self.sample_rate_hz = sample_rate_hz
        self.gain_db = gain_db
        self.ppm = ppm
        self.bias_tee = bias_tee
        self.samples_per_block = samples_per_block
        self._sdr = None

    @property
    def is_open(self) -> bool:
        return self._sdr is not None

    def open(self) -> None:
        from rtlsdr import RtlSdr

        sdr = RtlSdr(device_index=self.device_index)
        try:
            sdr.sample_rate = self.sample_rate_hz
            sdr.center_freq = self.center_freq_hz
            if self.ppm:
                sdr.freq_correction = self.ppm
            sdr.gain = self.gain_db
            if self.bias_tee and hasattr(sdr, "set_bias_tee"):
                sdr.set_bias_tee(True)
        except Exception:
            sdr.close()
            raise
        self._sdr = sdr
        log.info("SDR open: %.3f MHz, %.1f MS/s, gain %s",
                 self.center_freq_hz / 1e6, self.sample_rate_hz / 1e6, self.gain_db)

    def read_block(self) -> np.ndarray:
        if self._sdr is None:
            raise RuntimeError("SDR is not open")
        raw = self._sdr.read_bytes(2 * self.samples_per_block)
        return bytes_to_iq(raw)

    def close(self) -> None:
        if self._sdr is not None:
            try:
                self._sdr.close()
            except Exception:
                log.debug("error closing SDR", exc_info=True)
            self._sdr = None

    def __enter__(self) -> "SDRSource":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _level_check(seconds: float = 5.0) -> None:
    with SDRSource() as src:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            mag = np.abs(src.read_block())
            clip = np.mean(mag > 0.95) * 100
            print(f"median {np.median(mag):.4f}  p99.9 {np.percentile(mag, 99.9):.3f}  "
                  f"max {mag.max():.3f}  clipped {clip:.3f}%")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _level_check()
