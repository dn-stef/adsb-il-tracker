from __future__ import annotations

import numpy as np

OVERSAMPLE = 20


def frame_envelope(hex_msg: str, sps: int, oversample: int = OVERSAMPLE) -> np.ndarray:
    rate = sps * oversample
    half = rate // 2
    bits = np.unpackbits(np.frombuffer(bytes.fromhex(hex_msg), dtype=np.uint8))
    env = np.zeros((8 + bits.size) * rate, dtype=np.float32)
    for start_us in (0.0, 1.0, 3.5, 4.5):
        s = int(round(start_us * rate))
        env[s:s + half] = 1.0
    for i, b in enumerate(bits):
        s = (8 + i) * rate + (0 if b else half)
        env[s:s + half] = 1.0
    return env


def make_iq(
    frames: list[tuple[str, float]],
    n_samples: int,
    sample_rate_hz: float = 2e6,
    amplitude: float = 0.5,
    noise_rms: float = 0.01,
    freq_offset_hz: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sps = int(sample_rate_hz / 1e6)
    hi = np.zeros(n_samples * OVERSAMPLE, dtype=np.float32)
    for hex_msg, start in frames:
        env = frame_envelope(hex_msg, sps)
        s = int(round(start * OVERSAMPLE))
        hi[s:s + env.size] += amplitude * env[: max(0, hi.size - s)]
    env = hi.reshape(n_samples, OVERSAMPLE).mean(axis=1)
    t = np.arange(n_samples) / sample_rate_hz
    carrier = np.exp(1j * (2 * np.pi * freq_offset_hz * t + rng.uniform(0, 2 * np.pi)))
    noise = noise_rms / np.sqrt(2) * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))
    return (env * carrier + noise).astype(np.complex64)


def flip_bits(hex_msg: str, *positions: int) -> str:
    v = int(hex_msg, 16)
    n = len(hex_msg) * 4
    for p in positions:
        v ^= 1 << (n - 1 - p)
    return f"{v:0{len(hex_msg)}X}"
