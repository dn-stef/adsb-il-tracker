from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
from scipy import signal

import config

log = logging.getLogger(__name__)


CRC24_POLY = 0xFFF409
LONG_BITS = 112
SHORT_BITS = 56
PREAMBLE_US = 8


def _make_crc_table() -> list[int]:
    table = []
    for byte in range(256):
        crc = byte << 16
        for _ in range(8):
            crc = ((crc << 1) ^ CRC24_POLY) if crc & 0x800000 else (crc << 1)
        table.append(crc & 0xFFFFFF)
    return table


_CRC_TABLE = _make_crc_table()


def crc24(data: bytes) -> int:
    crc = 0
    for b in data:
        crc = ((crc << 8) & 0xFFFFFF) ^ _CRC_TABLE[((crc >> 16) ^ b) & 0xFF]
    return crc


def crc_residual(msg: bytes) -> int:
    if len(msg) not in (SHORT_BITS // 8, LONG_BITS // 8):
        raise ValueError(f"Mode S message must be 7 or 14 bytes, got {len(msg)}")
    return crc24(msg[:-3]) ^ int.from_bytes(msg[-3:], "big")


def crc_ok(hex_msg: str) -> bool:
    try:
        return crc_residual(bytes.fromhex(hex_msg)) == 0
    except ValueError:
        return False


def _syndrome_table(nbits: int) -> np.ndarray:
    synd = np.zeros(nbits, dtype=np.uint32)
    for j in range(nbits):
        m = bytearray(nbits // 8)
        m[j // 8] = 0x80 >> (j % 8)
        synd[j] = crc_residual(bytes(m))
    return synd


LONG_SYNDROMES = _syndrome_table(LONG_BITS)
_LONG_FIXABLE = {int(s): j for j, s in enumerate(LONG_SYNDROMES) if j >= 5}


def residuals_from_bits(bits: np.ndarray) -> np.ndarray:
    return np.bitwise_xor.reduce(np.where(bits.astype(bool), LONG_SYNDROMES, 0), axis=1)


@dataclass(frozen=True)
class Frame:
    hex: str
    df: int
    timestamp: float
    signal_dbfs: float
    corrected_bits: int


@dataclass
class DemodStats:
    blocks: int = 0
    samples: int = 0
    preambles: int = 0
    wrong_df: int = 0
    crc_ok: int = 0
    crc_fixed: int = 0
    crc_bad: int = 0
    overlaps: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


class Demodulator:
    def __init__(
        self,
        sample_rate_hz: float = config.SAMPLE_RATE_HZ,
        threshold: float = config.PREAMBLE_THRESHOLD,
        min_ratio: float = config.PREAMBLE_MIN_RATIO,
        accepted_dfs=config.ACCEPTED_DOWNLINK_FORMATS,
        fix_single_bit: bool = config.FIX_SINGLE_BIT_ERRORS,
    ):
        sps = sample_rate_hz / 1e6
        if sps != int(sps) or int(sps) < 2 or int(sps) % 2:
            raise ValueError("sample rate must be an even whole number of MS/s (e.g. 2 MS/s)")
        self.sample_rate_hz = float(sample_rate_hz)
        self.sps = int(sps)
        self.threshold = threshold
        self.min_ratio = min_ratio
        self.accepted_dfs = frozenset(accepted_dfs)
        self.fix_single_bit = fix_single_bit
        self.stats = DemodStats()

        half = self.sps // 2
        pre_len = PREAMBLE_US * self.sps
        pulse = np.zeros(pre_len, dtype=bool)
        for start_us in (0.0, 1.0, 3.5, 4.5):
            s = int(round(start_us * self.sps))
            pulse[s:s + half] = True
        self._tpl_score = np.where(pulse, 1.0 / pulse.sum(), -1.0 / (~pulse).sum()).astype(np.float32)
        self._tpl_high = np.where(pulse, 1.0 / pulse.sum(), 0.0).astype(np.float32)
        self._pre_len = pre_len
        self.frame_samples = pre_len + LONG_BITS * self.sps

        chip = (np.arange(LONG_BITS) * self.sps)[:, None]
        self._early_idx = pre_len + chip + np.arange(half)
        self._late_idx = pre_len + chip + half + np.arange(half)

        self.reset()

    def reset(self) -> None:
        self._carry = np.zeros(0, dtype=np.float32)
        self._carry_abs = 0
        self._last_end_abs = -1

    def process(self, samples: np.ndarray, timestamp: float = 0.0) -> list[Frame]:
        samples = np.asarray(samples)
        if samples.ndim != 1:
            samples = samples.ravel()
        mag = np.abs(samples) if np.iscomplexobj(samples) else samples
        mag = np.nan_to_num(mag.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)

        self.stats.blocks += 1
        self.stats.samples += mag.size

        carry_len = self._carry.size
        buf = np.concatenate([self._carry, mag]) if carry_len else mag
        buf_abs = self._carry_abs
        t_buf0 = timestamp - carry_len / self.sample_rate_hz

        frames: list[Frame] = []
        n_starts = buf.size - self.frame_samples + 1
        if n_starts > 2:
            frames = self._detect(buf, n_starts, buf_abs, t_buf0)
            keep = self.frame_samples + 2
        else:
            keep = buf.size

        self._carry = buf[buf.size - keep:].copy()
        self._carry_abs = buf_abs + buf.size - keep
        return frames


    def _detect(self, buf: np.ndarray, n_starts: int, buf_abs: int, t_buf0: float) -> list[Frame]:
        noise = float(np.median(buf[:: max(1, buf.size // 65536)])) + 1e-9

        score = signal.correlate(buf, self._tpl_score, mode="valid")[:n_starts]
        high = signal.correlate(buf, self._tpl_high, mode="valid")[:n_starts]
        low = high - score
        good = (score > self.threshold * noise) & (high > self.min_ratio * low)
        if not good.any():
            return []
        peaks, _ = signal.find_peaks(np.where(good, score, 0.0), height=self.threshold * noise)
        if peaks.size == 0:
            return []
        self.stats.preambles += peaks.size

        variants = [self._slice_early_late(buf, peaks)]
        if self.sps == 2:
            variants.append(self._slice_phase_aware(buf, peaks, noise))

        weights = 1 << np.arange(4, -1, -1)
        checked = []
        for bits in variants:
            df = bits[:, :5] @ weights
            df_ok = np.isin(df, list(self.accepted_dfs))
            resid = np.where(df_ok, residuals_from_bits(bits).astype(np.int64), -1)
            checked.append((bits, df, resid))
        any_df_ok = np.any([r >= 0 for _, _, r in checked], axis=0)
        self.stats.wrong_df += int((~any_df_ok).sum())

        frames: list[Frame] = []
        for k in np.flatnonzero(any_df_ok):
            choice = self._pick_variant(checked, k)
            if choice is None:
                self.stats.crc_bad += 1
                continue
            bits, df, fixed = choice
            p = int(peaks[k])
            start_abs = buf_abs + p
            if start_abs < self._last_end_abs:
                self.stats.overlaps += 1
                continue
            self._last_end_abs = start_abs + self.frame_samples
            if fixed:
                self.stats.crc_fixed += 1
            else:
                self.stats.crc_ok += 1
            frames.append(Frame(
                hex=np.packbits(bits).tobytes().hex().upper(),
                df=df,
                timestamp=t_buf0 + p / self.sample_rate_hz,
                signal_dbfs=float(20 * np.log10(max(high[p], 1e-9))),
                corrected_bits=fixed,
            ))
        return frames

    def _pick_variant(self, checked, k):
        for bits, df, resid in checked:
            if resid[k] == 0:
                return bits[k], int(df[k]), 0
        if self.fix_single_bit:
            for bits, df, resid in checked:
                j = _LONG_FIXABLE.get(int(resid[k])) if resid[k] > 0 else None
                if j is not None:
                    repaired = bits[k].copy()
                    repaired[j] ^= 1
                    return repaired, int(df[k]), 1
        return None

    def _slice_early_late(self, buf: np.ndarray, peaks: np.ndarray) -> np.ndarray:
        starts = peaks[:, None, None]
        early = buf[starts + self._early_idx].sum(axis=2)
        late = buf[starts + self._late_idx].sum(axis=2)
        return (early > late).astype(np.uint8)

    def _slice_phase_aware(self, buf: np.ndarray, peaks: np.ndarray, noise: float) -> np.ndarray:
        m = buf - noise
        pk = peaks[:, None]
        on = m[pk + np.array([0, 2, 7, 9])].sum(axis=1)
        after = m[pk + np.array([1, 3, 8, 10])].sum(axis=1)
        before = m[pk + np.array([-1, 1, 6, 8])].sum(axis=1)
        eps = 1e-9
        phi_after = np.clip(after / np.maximum(on + after, eps), 0.0, 1.0)
        phi_before = np.clip(before / np.maximum(on + before, eps), 0.0, 1.0)
        shift = phi_before > phi_after
        phi = np.where(shift, 1.0 - phi_before, phi_after)
        amp = np.where(shift, on + before, on + after) / 4.0
        data0 = peaks - shift.astype(np.int64) + self._pre_len

        n = peaks.size
        head, tail = (1.0 - phi) * amp, phi * amp
        exp_a = np.stack([np.stack([tail, tail + head]), np.stack([np.zeros(n), head])])
        exp_b = np.stack([np.stack([head, tail]), np.stack([head, tail])])

        cost = np.stack([np.full(n, np.inf), np.zeros(n)])
        back = np.zeros((LONG_BITS, 2, n), dtype=np.uint8)
        for i in range(LONG_BITS):
            a = m[data0 + 2 * i]
            b = m[data0 + 2 * i + 1]
            metric = cost[:, None, :] + (a - exp_a) ** 2 + (b - exp_b) ** 2
            back[i] = np.argmin(metric, axis=0)
            cost = np.min(metric, axis=0)

        bits = np.zeros((n, LONG_BITS), dtype=np.uint8)
        state = np.argmin(cost, axis=0)
        cols = np.arange(n)
        for i in range(LONG_BITS - 1, -1, -1):
            bits[:, i] = state
            state = back[i, state, cols]
        return bits


def demodulate(samples: np.ndarray, sample_rate_hz: float = config.SAMPLE_RATE_HZ,
               timestamp: float = 0.0, **kwargs) -> list[Frame]:
    d = Demodulator(sample_rate_hz=sample_rate_hz, **kwargs)
    frames = d.process(samples, timestamp)
    pad = np.zeros(d.frame_samples + 4, dtype=np.float32)
    frames += d.process(pad, timestamp + len(samples) / d.sample_rate_hz)
    return frames


def parse_hex(hex_msg: str) -> dict | None:
    import pyModeS

    try:
        return dict(pyModeS.decode(hex_msg))
    except Exception:
        log.debug("pyModeS could not parse %s", hex_msg, exc_info=True)
        return None


class MessageParser:
    def __init__(self, pair_window_s: float = config.CPR_PAIR_WINDOW_S,
                 surface_ref: tuple[float, float] = (config.BASE_LAT, config.BASE_LON)):
        import pyModeS

        self._pipe = pyModeS.PipeDecoder(pair_window=pair_window_s, surface_ref=surface_ref)
        self.errors = 0

    def parse(self, frame: Frame) -> dict | None:
        try:
            d = self._pipe.decode(frame.hex, timestamp=frame.timestamp)
        except Exception:
            self.errors += 1
            log.debug("pyModeS rejected %s", frame.hex, exc_info=True)
            return None
        if not d or "error" in d or d.get("crc_valid") is False:
            self.errors += 1
            return None
        return dict(d)
