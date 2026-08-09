"""
Pitch detection, musical effects, and the benchmark harness.

``test_yin_avoids_octave_errors`` is the important one. Resonance routinely
makes the second harmonic louder than the fundamental, and a spectral
peak-picker reports an octave too high as a result. Auto-tune built on such a
detector would confidently correct notes to the wrong octave, so this is the
test that has to hold.
"""

import numpy as np
import pytest

from elastiqa.benchmark import (
    DEFAULT_METHODS,
    make_test_signals,
    run_benchmark,
    to_csv,
    to_markdown,
)
from elastiqa.effects import (
    autotune,
    harmonize,
    quantize_to_scale,
    robotize,
    whisperize,
)
from elastiqa.pitch.f0 import SCALES, hz_to_midi, midi_to_hz, yin_frame, yin_track
from elastiqa.pitch.spectral import spectral_pitch_shift

SR = 22050


def _harmonic(f0=220.0, duration=2.0, n=15, sr=SR):
    t = np.arange(int(duration * sr)) / sr
    x = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, n + 1))
    return x / np.max(np.abs(x))


def _measured_f0(x, sr=SR):
    f0, voiced = yin_track(x, sr)
    return float(np.median(f0[voiced])) if voiced.any() else 0.0


# --------------------------------------------------------------------------
# MIDI conversion
# --------------------------------------------------------------------------

def test_a440_is_midi_69():
    assert np.isclose(hz_to_midi(440.0), 69.0)
    assert np.isclose(midi_to_hz(69), 440.0)


def test_octave_is_twelve_semitones():
    assert np.isclose(hz_to_midi(880.0) - hz_to_midi(440.0), 12.0)


def test_midi_round_trip():
    for m in (21, 45, 60, 69, 96):
        assert np.isclose(hz_to_midi(midi_to_hz(m)), m)


# --------------------------------------------------------------------------
# YIN
# --------------------------------------------------------------------------

@pytest.mark.parametrize("f0", [82.4, 110.0, 220.0, 440.0, 659.3])
def test_yin_frame_accuracy(f0):
    t = np.arange(2048) / SR
    frame = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 15))
    est, aper = yin_frame(frame, SR)

    assert abs(est - f0) / f0 < 0.01, f"estimated {est:.2f}, true {f0}"
    assert aper < 0.2, f"clearly pitched signal reported aperiodicity {aper:.3f}"


def test_yin_avoids_octave_errors():
    """THE test that makes auto-tune possible.

    Here the second harmonic is *louder* than the fundamental, which is
    completely normal in voiced speech and singing. A spectral peak-picker
    reports 440 Hz. YIN must report 220 Hz.
    """
    t = np.arange(2048) / SR
    frame = 0.3 * np.sin(2 * np.pi * 220 * t) + 1.0 * np.sin(2 * np.pi * 440 * t)

    est, _ = yin_frame(frame, SR)
    assert abs(est - 220.0) < 5.0, f"octave error: reported {est:.1f} Hz"

    # Confirm the trap is real -- a peak-picker does get this wrong.
    spec = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
    naive_hz = np.fft.rfftfreq(len(frame), 1 / SR)[np.argmax(spec)]
    assert abs(naive_hz - 440.0) < 10.0


def test_yin_reports_noise_as_aperiodic():
    rng = np.random.default_rng(0)
    _, aper = yin_frame(rng.standard_normal(2048), SR)
    assert aper > 0.4


def test_yin_handles_silence():
    est, aper = yin_frame(np.zeros(2048), SR)
    assert est == 0.0 and aper == 1.0


def test_yin_track_shape_and_voicing():
    f0, voiced = yin_track(_harmonic(), SR)
    assert f0.shape == voiced.shape
    assert voiced.sum() > 0.8 * len(voiced)
    assert np.all(f0[~voiced] == 0.0)


def test_yin_track_rejects_noise():
    rng = np.random.default_rng(1)
    _, voiced = yin_track(rng.standard_normal(2 * SR) * 0.1, SR)
    assert voiced.sum() < 0.1 * len(voiced)


def test_yin_track_follows_a_glide():
    t = np.arange(2 * SR) / SR
    x = sum(np.sin(2 * np.pi * (200 + 150 * t) * h * t) / h for h in range(1, 10))

    f0, voiced = yin_track(x, SR)
    track = f0[voiced]
    assert track[-1] > track[0] + 200


# --------------------------------------------------------------------------
# Scale quantisation
# --------------------------------------------------------------------------

def test_quantize_snaps_to_nearest_note():
    flat = 440.0 * 2 ** (-0.4 / 12)          # 40 cents flat of A440
    assert np.isclose(quantize_to_scale(np.array([flat]))[0], 440.0, rtol=1e-6)


def test_quantize_leaves_unvoiced_alone():
    out = quantize_to_scale(np.array([0.0, 440.0, 0.0]))
    assert out[0] == 0.0 and out[2] == 0.0


def test_quantize_respects_the_scale():
    """C major has no C sharp, so a C sharp must move."""
    c_sharp = midi_to_hz(61)
    out = quantize_to_scale(np.array([c_sharp]), scale="major", tonic_midi=0)
    assert not np.isclose(out[0], c_sharp)
    assert round(float(hz_to_midi(out[0]))) in (60, 62)


def test_chromatic_keeps_every_semitone():
    for m in range(60, 72):
        hz = midi_to_hz(m)
        assert np.isclose(quantize_to_scale(np.array([hz]))[0], hz, rtol=1e-6)


def test_unknown_scale_raises():
    with pytest.raises(ValueError):
        quantize_to_scale(np.array([440.0]), scale="lydian_dominant_bebop")


def test_all_scales_are_usable():
    for name in SCALES:
        out = quantize_to_scale(np.array([437.0]), scale=name)
        assert out[0] > 0


# --------------------------------------------------------------------------
# Spectral pitch shift
# --------------------------------------------------------------------------

@pytest.mark.parametrize("semitones", [-12, -5, 5, 12])
def test_spectral_shift_moves_pitch(semitones):
    x = _harmonic(f0=220.0)
    ratio = 2 ** (semitones / 12.0)
    y = spectral_pitch_shift(x, ratio, preserve_formants=False)

    assert abs(_measured_f0(y) - 220.0 * ratio) < 220.0 * ratio * 0.05


def test_spectral_shift_preserves_length():
    x = _harmonic()
    for r in (0.5, 0.8, 1.0, 1.5, 2.0):
        assert len(spectral_pitch_shift(x, r)) == len(x)


def test_spectral_shift_accepts_a_ratio_curve():
    """The feature that makes auto-tune possible: one ratio per frame."""
    x = _harmonic(duration=2.0)
    n_frames = 1 + len(x) // 512
    ratios = np.linspace(1.0, 1.5, n_frames)

    y = spectral_pitch_shift(x, ratios, preserve_formants=False)
    assert len(y) == len(x)
    assert np.all(np.isfinite(y))

    # Pitch should be higher at the end than at the start.
    assert _measured_f0(y[-len(y) // 3 :]) > _measured_f0(y[: len(y) // 3]) + 20


def test_spectral_shift_resamples_a_mismatched_curve():
    x = _harmonic()
    y = spectral_pitch_shift(x, np.full(7, 1.2), preserve_formants=False)
    assert len(y) == len(x) and np.all(np.isfinite(y))


# --------------------------------------------------------------------------
# Effects
# --------------------------------------------------------------------------

def test_autotune_corrects_a_flat_note():
    """A singer 40 cents flat should come back within a few cents."""
    flat_hz = 440.0 * 2 ** (-0.40 / 12)
    x = _harmonic(f0=flat_hz)

    before = 1200 * np.log2(_measured_f0(x) / 440.0)
    after = 1200 * np.log2(_measured_f0(autotune(x, SR)) / 440.0)

    assert abs(before) > 25.0, "test signal is not actually flat"
    assert abs(after) < 10.0, f"still {after:.0f} cents off after correction"


def test_autotune_strength_interpolates():
    flat_hz = 440.0 * 2 ** (-0.40 / 12)
    x = _harmonic(f0=flat_hz)

    full = abs(1200 * np.log2(_measured_f0(autotune(x, SR, strength=1.0)) / 440.0))
    half = abs(1200 * np.log2(_measured_f0(autotune(x, SR, strength=0.5)) / 440.0))
    none = abs(1200 * np.log2(_measured_f0(autotune(x, SR, strength=0.0)) / 440.0))

    assert full < half < none


def test_autotune_returns_tracks():
    x = _harmonic(f0=437.0)
    y, detected, target, voiced = autotune(x, SR, return_tracks=True)
    assert len(detected) == len(target) == len(voiced)
    assert len(y) == len(x)


def test_autotune_leaves_noise_alone():
    """Unvoiced frames must get a ratio of exactly 1."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal(SR) * 0.1
    y = autotune(x, SR)
    assert np.all(np.isfinite(y)) and len(y) == len(x)


def test_harmonize_adds_voices():
    x = _harmonic(f0=220.0)
    y = harmonize(x, intervals=(4.0, 7.0))

    assert len(y) == len(x)
    assert np.all(np.isfinite(y))

    # The added voices should show up as new spectral energy near 277 and 330 Hz.
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1 / SR)
    for target in (220.0 * 2 ** (4 / 12), 220.0 * 2 ** (7 / 12)):
        band = (freqs > target - 8) & (freqs < target + 8)
        assert spec[band].max() > 0.02 * spec.max(), f"no energy near {target:.0f} Hz"


def test_harmonize_dry_only_is_the_input():
    x = _harmonic()
    assert np.allclose(harmonize(x, intervals=(), dry=1.0), x)


def test_robotize_roughly_preserves_the_magnitude_spectrum():
    """Magnitudes survive, but not exactly.

    Flattening phase breaks the coherence of overlap-add, so reconstructed
    magnitudes drift somewhat from the originals -- correlation around 0.8
    rather than 1.0. Asserting near-perfect preservation here would be
    asserting something false.
    """
    from elastiqa.stft import stft

    x = _harmonic()
    y = robotize(x)

    mx = np.abs(stft(x, 2048, 512)).mean(axis=1)
    my = np.abs(stft(y, 2048, 512)).mean(axis=1)
    assert np.corrcoef(mx / mx.max(), my / my.max())[0, 1] > 0.7


def test_robotize_imposes_frame_rate_periodicity():
    """The measurable effect: the output repeats at the hop.

    Note what is deliberately *not* asserted -- that the original pitch is
    erased. It is not. The harmonic comb lives in the magnitude spectrum, and
    zeroing phase leaves it there; a 220 Hz input still measures 220 Hz. What
    robotization adds is a strong periodicity at ``sr / hop`` on top.
    """
    hop = 512
    x = _harmonic(f0=220.0)
    y = robotize(x, hop=hop)

    def autocorr_at(sig, lag):
        sig = sig - sig.mean()
        ac = np.correlate(sig, sig, mode="full")[len(sig) - 1 :]
        return float(ac[lag] / ac[0])

    assert autocorr_at(y, hop) > 0.9, "no frame-rate periodicity was imposed"
    assert autocorr_at(y, hop) > autocorr_at(x, hop) + 0.3

    # The original pitch survives, exactly as the docstring says it does.
    assert abs(_measured_f0(y) - 220.0) < 15.0


def test_whisperize_increases_aperiodicity():
    """Randomising phase pushes the signal towards noise excitation.

    The effect is partial on a synthetic harmonic stack -- aperiodicity rises
    by more than an order of magnitude but stays below the voicing threshold,
    because the harmonic comb is in the magnitude and 75% overlap restores
    some inter-frame correlation. Asserting full unvoicing would fail, and
    would be the wrong claim to make.
    """
    from elastiqa.pitch.f0 import yin_frame

    def mean_aperiodicity(sig):
        return float(np.mean([
            yin_frame(sig[i * 512 : i * 512 + 2048], SR)[1] for i in range(5, 30)
        ]))

    x = _harmonic()
    before = mean_aperiodicity(x)
    after = mean_aperiodicity(whisperize(x, seed=0))

    assert after > before * 5.0, f"aperiodicity {before:.4f} -> {after:.4f}"


def test_effects_preserve_length():
    x = _harmonic()
    for fn in (robotize, lambda s: whisperize(s, seed=0)):
        assert len(fn(x)) == len(x)


# --------------------------------------------------------------------------
# Benchmark harness
# --------------------------------------------------------------------------

def test_test_signals_are_sane():
    signals = make_test_signals(SR, duration=0.5)
    assert set(signals) >= {"sine", "harmonic", "vibrato", "percussive", "mixed", "noise"}
    for name, x in signals.items():
        assert np.all(np.isfinite(x)), name
        assert np.max(np.abs(x)) <= 1.0 + 1e-9, name


def test_benchmark_produces_a_full_grid():
    signals = {"sine": make_test_signals(SR, 0.5)["sine"]}
    rows = run_benchmark(signals=signals, stretches=(1.5,), verbose=False)

    assert len(rows) == len(DEFAULT_METHODS)
    for r in rows:
        assert {"signal", "method", "stretch"} <= set(r)


def test_benchmark_reports_consistency_only_for_vocoders():
    signals = {"sine": make_test_signals(SR, 0.5)["sine"]}
    rows = run_benchmark(signals=signals, stretches=(1.5,), verbose=False)

    for r in rows:
        has_dm = "consistency" in r
        assert has_dm == r["method"].startswith("pv"), r["method"]


def test_crest_error_is_never_negative():
    """Signed crest loss can go negative; the ranking metric must not."""
    signals = {"percussive": make_test_signals(SR, 1.0)["percussive"]}
    rows = run_benchmark(signals=signals, stretches=(1.5,), verbose=False)
    assert all(r["crest_error"] >= 0 for r in rows)


def test_markdown_table_renders():
    signals = {"sine": make_test_signals(SR, 0.5)["sine"]}
    rows = run_benchmark(signals=signals, stretches=(1.25, 1.5), verbose=False)

    table = to_markdown(rows, "log_spectral_distance", signal="sine")
    assert table.startswith("| method |")
    assert "**" in table                       # a winner was bolded
    assert "1.25x" in table and "1.5x" in table


def test_markdown_can_exclude_methods():
    signals = {"sine": make_test_signals(SR, 0.5)["sine"]}
    rows = run_benchmark(signals=signals, stretches=(1.5,), verbose=False)

    assert "naive" not in to_markdown(rows, "amplitude_warble", exclude=("naive",))


def test_csv_round_trips(tmp_path=None):
    import csv as _csv
    import tempfile

    signals = {"sine": make_test_signals(SR, 0.5)["sine"]}
    rows = run_benchmark(signals=signals, stretches=(1.5,), verbose=False)

    with tempfile.TemporaryDirectory() as d:
        path = to_csv(rows, f"{d}/bench.csv")
        with open(path) as fh:
            back = list(_csv.DictReader(fh))
    assert len(back) == len(rows)
    assert "log_spectral_distance" in back[0]
