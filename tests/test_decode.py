import numpy as np
import pytest

import decode
from decode import Demodulator, MessageParser, crc24, crc_ok, crc_residual, demodulate, parse_hex
from synth import flip_bits, make_iq

IDENT_KLM = "8D4840D6202CC371C32CE0576098"
POS_EVEN = "8D40621D58C382D690C8AC2863A7"
POS_ODD = "8D40621D58C386435CC412692AD6"
VEL_GS = "8D485020994409940838175B284F"
VEL_AS = "8DA05F219B06B6AF189400CBC33F"
KNOWN = [IDENT_KLM, POS_EVEN, POS_ODD, VEL_GS, VEL_AS]

FS = 2e6
FRAME = 240


@pytest.mark.parametrize("msg", KNOWN)
def test_known_frames_have_zero_crc_residual(msg):
    b = bytes.fromhex(msg)
    assert crc24(b[:11]) == int.from_bytes(b[11:], "big")
    assert crc_residual(b) == 0
    assert crc_ok(msg)


def test_every_single_bit_flip_is_detected():
    for bit in range(112):
        assert not crc_ok(flip_bits(IDENT_KLM, bit))


def test_vectorised_syndrome_crc_matches_bytewise_crc():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=(200, 112), dtype=np.uint8)
    expected = [crc_residual(np.packbits(row).tobytes()) for row in bits]
    assert decode.residuals_from_bits(bits).tolist() == expected


@pytest.mark.parametrize("bad", ["", "8D48", "ZZ4840D6202CC371C32CE0576098", IDENT_KLM + "00"])
def test_malformed_hex_is_not_crc_ok(bad):
    assert crc_ok(bad) is False


def test_crc_residual_rejects_wrong_length():
    with pytest.raises(ValueError):
        crc_residual(b"\x00" * 10)


def test_demodulates_clean_frames_exactly():
    placed = [(m, 1000 + i * 700) for i, m in enumerate(KNOWN)]
    frames = demodulate(make_iq(placed, 8000))
    assert [f.hex for f in frames] == KNOWN
    assert all(f.df == 17 and f.corrected_bits == 0 for f in frames)


def test_demodulates_with_noise_carrier_offset_and_fractional_timing():
    placed = [(IDENT_KLM, 500.2), (VEL_GS, 1500.9), (POS_EVEN, 2600.1)]
    iq = make_iq(placed, 4000, amplitude=0.3, noise_rms=0.03, freq_offset_hz=50e3, seed=7)
    assert sorted(f.hex for f in demodulate(iq)) == sorted(m for m, _ in placed)


@pytest.mark.parametrize("phase", np.round(np.arange(0.0, 1.0, 0.05), 2))
def test_every_subsample_phase_decodes_exactly_without_noise(phase):
    iq = make_iq([(IDENT_KLM, 500 + phase)], 1200, noise_rms=0.0)
    (f,) = demodulate(iq, fix_single_bit=False)
    assert f.hex == IDENT_KLM and f.corrected_bits == 0


@pytest.mark.parametrize("phase", [0.3, 0.6, 0.7])
def test_straddling_pulses_decode_reliably_in_noise(phase):
    hits = sum(len(demodulate(make_iq([(KNOWN[i % 5], 500 + phase)], 1200, amplitude=0.3,
                                      noise_rms=0.03, seed=i), fix_single_bit=False))
               for i in range(20))
    assert hits >= 18


def test_weak_frame_below_threshold_is_ignored_without_crashing():
    iq = make_iq([(IDENT_KLM, 500)], 2000, amplitude=0.02, noise_rms=0.05, seed=3)
    assert demodulate(iq) == []


def test_frame_timestamp_and_signal_level():
    iq = make_iq([(VEL_GS, 1234)], 3000, amplitude=0.5, noise_rms=0.001)
    (f,) = demodulate(iq, timestamp=100.0)
    assert f.timestamp == pytest.approx(100.0 + 1234 / FS, abs=1e-6)
    assert f.signal_dbfs == pytest.approx(20 * np.log10(0.5), abs=0.5)


@pytest.mark.parametrize("split", [1003, 1010, 1100, 1200, 1236, 1239, 1240, 1241, 1300])
def test_frame_straddling_block_boundary_is_recovered_once(split):
    iq = make_iq([(IDENT_KLM, 1000), (POS_ODD, 3000)], 5000)
    d = Demodulator()
    frames = d.process(iq[:split], 0.0) + d.process(iq[split:], split / FS)
    frames += d.process(np.zeros(FRAME + 4, np.complex64), len(iq) / FS)
    assert [f.hex for f in frames] == [IDENT_KLM, POS_ODD]
    assert frames[0].timestamp == pytest.approx(1000 / FS, abs=1e-6)


def test_streaming_small_blocks_matches_one_shot():
    placed = [(m, 300 + i * 450) for i, m in enumerate(KNOWN)]
    iq = make_iq(placed, 3000, noise_rms=0.02, seed=11)
    d = Demodulator()
    got = []
    for i in range(0, iq.size, 97):
        got += d.process(iq[i:i + 97], i / FS)
    got += d.process(np.zeros(FRAME + 4, np.complex64), iq.size / FS)
    assert [f.hex for f in got] == KNOWN


def test_reset_discards_partial_frame_after_gap():
    iq = make_iq([(IDENT_KLM, 1000)], 3000)
    d = Demodulator()
    d.process(iq[:1100], 0.0)
    d.reset()
    frames = d.process(iq[1100:], 1100 / FS) + d.process(np.zeros(FRAME + 4, np.complex64), 1.0)
    assert frames == []


def test_noise_only_produces_no_frames():
    rng = np.random.default_rng(42)
    iq = (0.05 * (rng.standard_normal(2_000_000) + 1j * rng.standard_normal(2_000_000))).astype(np.complex64)
    d = Demodulator()
    assert d.process(iq, 0.0) == []
    assert d.stats.crc_ok == 0 and d.stats.crc_fixed == 0


def test_corrupt_frame_is_discarded_and_counted():
    bad = flip_bits(IDENT_KLM, 40, 77)
    d = Demodulator()
    frames = d.process(make_iq([(bad, 500), (VEL_GS, 1500)], 3000), 0.0)
    assert [f.hex for f in frames] == [VEL_GS]
    assert d.stats.crc_bad >= 1


def test_single_bit_error_is_repaired_when_enabled():
    bad = flip_bits(POS_EVEN, 60)
    (f,) = demodulate(make_iq([(bad, 500)], 2000))
    assert f.hex == POS_EVEN and f.corrected_bits == 1
    assert demodulate(make_iq([(bad, 500)], 2000), fix_single_bit=False) == []


def test_df_bit_errors_are_never_repaired():
    bad = flip_bits(IDENT_KLM, 4)
    assert demodulate(make_iq([(bad, 500)], 2000)) == []


def test_non_adsb_downlink_formats_are_skipped():
    df20 = f"{(int(IDENT_KLM, 16) & ~(0x1F << 107)) | (20 << 107):028X}"
    d = Demodulator()
    d.process(make_iq([(df20, 500)], 2000), 0.0)
    assert d.stats.wrong_df >= 1 and d.stats.crc_ok == 0


@pytest.mark.parametrize("junk", [
    np.array([], dtype=np.complex64),
    np.zeros(10, dtype=np.complex64),
    np.full(5000, np.nan + 1j * np.nan, dtype=np.complex64),
    np.full(5000, np.inf, dtype=np.float32),
    np.ones((50, 100), dtype=np.complex64),
    np.ones(5000, dtype=np.complex64) * 1e6,
])
def test_garbage_input_never_raises(junk):
    d = Demodulator()
    assert d.process(junk, 0.0) == []
    assert d.process(make_iq([(VEL_GS, 300)], 1000), 1.0) is not None


def test_real_magnitude_input_is_accepted():
    iq = make_iq([(VEL_AS, 400)], 1200)
    assert [f.hex for f in demodulate(np.abs(iq))] == [VEL_AS]


@pytest.mark.parametrize("rate", [1e6, 2.4e6, 3e6])
def test_unsupported_sample_rates_are_rejected(rate):
    with pytest.raises(ValueError):
        Demodulator(sample_rate_hz=rate)


def test_four_msps_is_supported():
    iq = make_iq([(IDENT_KLM, 800)], 2000, sample_rate_hz=4e6)
    assert [f.hex for f in demodulate(iq, sample_rate_hz=4e6)] == [IDENT_KLM]


def test_parse_identification():
    d = parse_hex(IDENT_KLM)
    assert d["icao"] == "4840D6" and d["callsign"] == "KLM1023"


def test_parse_altitude_and_velocity():
    assert parse_hex(POS_EVEN)["altitude"] == 38000
    v = parse_hex(VEL_GS)
    assert v["groundspeed"] == 159 and v["track"] == pytest.approx(182.88, abs=0.01)
    assert v["vertical_rate"] == -832
    a = parse_hex(VEL_AS)
    assert a["airspeed"] == 375 and a["heading"] == pytest.approx(243.98, abs=0.01)


def test_parse_garbage_returns_none():
    assert parse_hex("not hex") is None


def test_message_parser_resolves_cpr_pair_position():
    p = MessageParser()
    last = None
    for i in range(9):
        msg = POS_EVEN if i % 2 == 0 else POS_ODD
        last = p.parse(decode.Frame(hex=msg, df=17, timestamp=float(i), signal_dbfs=-10, corrected_bits=0))
    assert last["latitude"] == pytest.approx(52.2572, abs=1e-3)
    assert last["longitude"] == pytest.approx(3.9194, abs=1e-3)


def test_message_parser_swallows_bad_frames():
    p = MessageParser()
    assert p.parse(decode.Frame(hex="XYZ", df=17, timestamp=0.0, signal_dbfs=0, corrected_bits=0)) is None
    assert p.errors == 1
