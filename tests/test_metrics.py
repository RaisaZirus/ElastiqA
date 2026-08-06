"""
Metrics.

The subtle one is ``test_consistency_of_real_signal_is_zero``. It encodes the
mistake that makes every algorithm look perfect: computing D_M on the STFT of
the *output* rather than on the modified STFT the algorithm produced. The
transform of a real signal is always self-consistent, so that version of the
measure returns ~0 no matter how badly the vocoder is behaving.
"""

import numpy as np
import pytest

from elastiqa.metrics import (
    consistency,
    evaluate,
    log_spectral_distance,
    ser,
    spectral_convergence,
)
from elastiqa.stft import stft
from elastiqa.tsm import time_stretch

SR = 22050


def _tone(f0=440.0, duration=1.5):
    t = np.arange(int(duration * SR)) / SR
    return np.sin(2 * np.pi * f0 * t)


def test_ser_identical_signals_is_infinite():
    x = _tone()
    assert ser(x, x) == float("inf")


def test_ser_decreases_with_noise():
    rng = np.random.default_rng(0)
    x = _tone()
    quiet = ser(x, x + 0.001 * rng.standard_normal(len(x)))
    loud = ser(x, x + 0.1 * rng.standard_normal(len(x)))
    assert quiet > loud


def test_consistency_of_real_signal_is_zero():
    """A real signal's own STFT is consistent by construction.

    This is why `evaluate` requires the modified STFT to be passed in
    explicitly rather than recomputing it from the output.
    """
    x = _tone()
    assert consistency(stft(x, 2048, 512), hop=512) < 1e-6


def test_consistency_detects_broken_phase():
    """Randomising phase should score far worse than leaving it alone."""
    x = _tone()
    X = stft(x, 2048, 512)

    rng = np.random.default_rng(1)
    X_broken = np.abs(X) * np.exp(1j * rng.uniform(-np.pi, np.pi, X.shape))

    assert consistency(X_broken, hop=512) > 0.1


def test_consistency_is_nonnegative():
    x = _tone()
    for mode in ("vocoder", "passthrough", "zero"):
        _, Y = time_stretch(x, 1.5, mode=mode, return_stft=True)
        assert consistency(Y, hop=512) >= 0.0


def test_consistency_of_silence_is_zero():
    """Must not divide by zero on an empty signal."""
    assert consistency(stft(np.zeros(10000), 2048, 512), hop=512) == 0.0


def test_spectral_convergence_zero_for_identical():
    x = _tone()
    assert spectral_convergence(x, x) < 1e-10


def test_spectral_convergence_handles_length_mismatch():
    """The whole point: reference and test have different durations."""
    x = _tone()
    y = time_stretch(x, 1.5)
    assert len(y) != len(x)

    score = spectral_convergence(x, y)
    assert np.isfinite(score) and score >= 0.0


def test_log_spectral_distance_zero_for_identical():
    x = _tone()
    assert log_spectral_distance(x, x) < 1e-6


def test_log_spectral_distance_grows_with_distortion():
    rng = np.random.default_rng(2)
    x = _tone()
    near = log_spectral_distance(x, x + 0.001 * rng.standard_normal(len(x)))
    far = log_spectral_distance(x, x + 0.2 * rng.standard_normal(len(x)))
    assert far > near


def test_evaluate_returns_expected_keys():
    x = _tone()
    y, Y = time_stretch(x, 1.5, return_stft=True)

    row = evaluate(x, y, stretch=1.5, modified_stft=Y)
    for key in ("stretch", "spectral_convergence", "log_spectral_distance", "consistency"):
        assert key in row
    assert "ser_db" not in row          # only meaningful at stretch = 1


def test_evaluate_includes_ser_at_unity():
    x = _tone()
    y, Y = time_stretch(x, 1.0, return_stft=True)
    row = evaluate(x, y, stretch=1.0, modified_stft=Y)
    assert row["ser_db"] > 40.0


def test_evaluate_omits_consistency_without_stft():
    x = _tone()
    y = time_stretch(x, 1.5)
    assert "consistency" not in evaluate(x, y, stretch=1.5)
