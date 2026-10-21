import ctypes

import numpy as np
import pytest

from acquire import bytes_to_iq

RAW = [0, 255, 128, 127, 255, 0]


@pytest.mark.parametrize("make", [
    bytes,
    bytearray,
    lambda v: np.array(v, dtype=np.uint8),
    lambda v: (ctypes.c_ubyte * len(v))(*v),
], ids=["bytes", "bytearray", "ndarray", "ctypes"])
def test_bytes_to_iq_accepts_all_buffer_types(make):
    iq = bytes_to_iq(make(RAW))
    assert iq.dtype == np.complex64 and iq.size == 3
    assert iq[0] == pytest.approx(-1 + 1j, abs=1e-6)
    assert iq[1].real == pytest.approx(0.5 / 127.5, abs=1e-6)
    assert iq[2] == pytest.approx(1 - 1j, abs=1e-6)


def test_bytes_to_iq_copies_reused_ctypes_buffer():
    buf = (ctypes.c_ubyte * 4)(255, 255, 255, 255)
    iq = bytes_to_iq(buf)
    buf[0] = 0
    assert iq[0].real == pytest.approx(1.0)


def test_bytes_to_iq_drops_odd_trailing_byte():
    assert bytes_to_iq(bytes([255, 0, 7])).size == 1
