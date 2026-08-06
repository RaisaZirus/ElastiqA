"""
Phase arithmetic.

``test_instantaneous_frequency_offgrid`` is the most important test in the
project after perfect reconstruction. It is the only automated check that the
heterodyned phase-increment logic is correct, and that logic is the heart of
the phase vocoder. If it fails, the vocoder will produce output that sounds
plausible but is subtly detuned and smeared -- the hardest possible bug to
diagnose by ear.
"""

import numpy as np
import pytest

from elastiqa.phase import (
    expected_phase_advance,
    instantaneous_frequency,
    princarg,
)
from elastiqa.stft import stft

SR = 22050


def test_princarg_leaves_principal_values_alone():
    phi = np.array([-3.0, -1.0, 0.0, 0.5, 1.0, 3.0])
    assert np.allclose(princarg(phi), phi)


def test_princarg_wraps_multiples_of_two_pi():
    phi = np.array([0.5, 0.5 + 2 * np.pi, 0.5 - 2 * np.pi, 0.5 + 20 * np.pi])
    assert np.allclose(princarg(phi), 0.5)


def test_princarg_range():
    rng = np.random.default_rng(0)
    phi = rng.uniform(-100, 100, size=10000)
    wrapped = princarg(phi)
    assert np.all(wrapped >= -np.pi - 1e-12)
    assert np.all(wrapped <= np.pi + 1e-12)


def test_princarg_preserves_complex_exponential():
    """Wrapping must not change the angle modulo 2*pi -- the only thing that
    actually matters downstream."""
    rng = np.random.default_rng(1)
    phi = rng.uniform(-50, 50, size=5000)
    assert np.allclose(np.exp(1j * princarg(phi)), np.exp(1j * phi))


def test_princarg_at_pi_boundary():
    """+pi must not silently flip to -pi (a modulo implementation does)."""
    assert np.isclose(float(princarg(np.pi)), np.pi)
    assert np.isclose(float(princarg(-np.pi)), -np.pi)


def test_expected_advance_matches_bin_centres():
    n_fft, hop = 1024, 256
    omega = expected_phase_advance(n_fft, hop)

    assert len(omega) == n_fft // 2 + 1
    assert np.isclose(omega[0], 0.0)                       # DC never advances
    # Bin k advances by 2*pi*k*hop/n_fft.
    assert np.isclose(omega[4], 2 * np.pi * 4 * hop / n_fft)


@pytest.mark.parametrize("freq_hz", [440.0, 1000.0, 2500.0])
def test_instantaneous_frequency_on_grid(freq_hz):
    """A tone at any frequency should be recovered at its own peak bin."""
    n_fft, hop = 2048, 512
    t = np.arange(SR) / SR
    x = np.sin(2 * np.pi * freq_hz * t)

    X = stft(x, n_fft=n_fft, hop=hop)
    ifreq = instantaneous_frequency(X, hop=hop, n_fft=n_fft)

    mid = X.shape[1] // 2
    peak = int(np.argmax(np.abs(X[:, mid])))
    est_hz = ifreq[peak, mid] * SR / (2 * np.pi)

    assert abs(est_hz - freq_hz) < 1.0, f"estimated {est_hz:.2f} Hz"


def test_instantaneous_frequency_offgrid():
    """THE critical test.

    A tone deliberately placed between two bin centres. The raw bin-centre
    frequency would be wrong by up to half a bin (~5 Hz here); correct
    heterodyned phase unwrapping recovers the true frequency to a fraction of
    a Hz. If this passes, the phase logic is sound.
    """
    n_fft, hop = 2048, 512
    bin_width = SR / n_fft                     # ~10.8 Hz
    true_hz = 440.0 + 0.37 * bin_width         # firmly between bins

    t = np.arange(2 * SR) / SR
    x = np.sin(2 * np.pi * true_hz * t)

    X = stft(x, n_fft=n_fft, hop=hop)
    ifreq = instantaneous_frequency(X, hop=hop, n_fft=n_fft)

    mid = X.shape[1] // 2
    peak = int(np.argmax(np.abs(X[:, mid])))

    centre_hz = peak * SR / n_fft
    est_hz = ifreq[peak, mid] * SR / (2 * np.pi)

    # The naive bin-centre answer is meaningfully wrong ...
    assert abs(centre_hz - true_hz) > 1.0
    # ... and the phase estimate is not.
    assert abs(est_hz - true_hz) < 0.5, (
        f"true {true_hz:.3f} Hz, bin centre {centre_hz:.3f} Hz, "
        f"estimated {est_hz:.3f} Hz"
    )


def test_instantaneous_frequency_shape():
    n_fft, hop = 1024, 256
    x = np.random.default_rng(2).standard_normal(10000)

    X = stft(x, n_fft=n_fft, hop=hop)
    ifreq = instantaneous_frequency(X, hop=hop, n_fft=n_fft)

    assert ifreq.shape == X.shape
    assert np.all(np.isfinite(ifreq))


def test_instantaneous_frequency_tracks_a_chirp():
    """A rising tone should give a rising frequency estimate."""
    n_fft, hop = 2048, 512
    t = np.arange(2 * SR) / SR
    x = np.sin(2 * np.pi * (200 + 400 * t) * t)   # 200 Hz -> ~1800 Hz

    X = stft(x, n_fft=n_fft, hop=hop)
    ifreq = instantaneous_frequency(X, hop=hop, n_fft=n_fft)

    peaks = np.argmax(np.abs(X), axis=0)
    track = np.array(
        [ifreq[p, m] * SR / (2 * np.pi) for m, p in enumerate(peaks)]
    )

    interior = track[5:-5]
    assert interior[-1] > interior[0] + 500      # clearly rising
    assert np.all(np.diff(interior) > -50)       # monotone up to noise
