"""
Identity phase locking.

The headline assertion is ``test_locking_reduces_warble``: locking should
essentially eliminate the amplitude fluctuation that the unlocked vocoder
leaves behind. That is the audible payoff of vertical phase coherence, and it
is measurable without a listening test.
"""

import numpy as np
import pytest

from elastiqa.metrics import amplitude_warble, consistency, ser
from elastiqa.tsm import (
    find_peaks,
    regions_of_influence,
    time_stretch,
    time_stretch_locked,
)

SR = 22050


def _harmonic(f0=220.0, duration=2.0, n_harmonics=25, sr=SR):
    t = np.arange(int(duration * sr)) / sr
    x = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, n_harmonics + 1))
    return x / np.max(np.abs(x))


def _f0_hz(x, sr=SR, lo=70.0, hi=500.0):
    x = x - x.mean()
    corr = np.correlate(x, x, mode="full")[len(x) - 1 :]
    lo_lag, hi_lag = int(sr / hi), int(sr / lo)
    return float(sr / (lo_lag + int(np.argmax(corr[lo_lag : hi_lag + 1]))))


# --------------------------------------------------------------------------
# Peak picking
# --------------------------------------------------------------------------

def test_finds_a_single_peak():
    mag = np.zeros(100)
    mag[50] = 1.0
    mag[49] = mag[51] = 0.5
    mag[48] = mag[52] = 0.2
    assert list(find_peaks(mag)) == [50]


def test_finds_multiple_peaks():
    mag = np.zeros(200)
    for centre in (30, 80, 140):
        mag[centre] = 1.0
        mag[centre - 1] = mag[centre + 1] = 0.5
    assert list(find_peaks(mag)) == [30, 80, 140]


def test_rejects_peaks_below_threshold():
    mag = np.zeros(200)
    mag[50] = 1.0
    mag[49] = mag[51] = 0.5
    mag[150] = 0.001              # a tiny bump
    mag[149] = mag[151] = 0.0005

    assert list(find_peaks(mag, threshold=0.01)) == [50]
    assert 150 in find_peaks(mag, threshold=0.0)


def test_four_neighbour_test_rejects_ripple():
    """A single-neighbour test would call every wiggle a peak."""
    x = np.linspace(0, 20 * np.pi, 400)
    mag = np.abs(np.sin(x)) + 0.02 * np.sin(37 * x)   # ripple on top
    peaks = find_peaks(mag)
    assert len(peaks) < 30, f"{len(peaks)} peaks -- ripple is being promoted"


def test_no_peaks_in_flat_or_tiny_input():
    assert len(find_peaks(np.ones(100))) == 0
    assert len(find_peaks(np.zeros(3))) == 0


def test_real_spectrum_peaks_land_on_harmonics():
    """Peaks of a harmonic stack should sit at integer multiples of f0."""
    from elastiqa.stft import stft

    x = _harmonic(f0=220.0)
    X = stft(x, 2048, 512)
    mag = np.abs(X[:, X.shape[1] // 2])

    peaks = find_peaks(mag, threshold=1e-3)
    peak_hz = peaks * SR / 2048

    # Each peak should be near a multiple of 220 Hz.
    for hz in peak_hz[:8]:
        nearest = round(hz / 220.0) * 220.0
        assert abs(hz - nearest) < 15.0, f"peak at {hz:.1f} Hz is not harmonic"


# --------------------------------------------------------------------------
# Regions of influence
# --------------------------------------------------------------------------

def test_regions_assign_every_bin():
    owner = regions_of_influence(np.array([10, 50, 90]), n_bins=100)
    assert len(owner) == 100
    assert set(np.unique(owner)) == {10, 50, 90}


def test_regions_split_at_midpoints():
    owner = regions_of_influence(np.array([10, 50]), n_bins=100)
    assert owner[0] == 10 and owner[29] == 10      # left of the midpoint
    assert owner[31] == 50 and owner[99] == 50     # right of it


def test_regions_without_peaks_are_identity():
    """No peaks must degrade to the unlocked vocoder, not to a crash."""
    owner = regions_of_influence(np.array([], dtype=int), n_bins=50)
    assert np.array_equal(owner, np.arange(50))


def test_peaks_own_themselves():
    peaks = np.array([5, 40, 77])
    owner = regions_of_influence(peaks, n_bins=100)
    for p in peaks:
        assert owner[p] == p


# --------------------------------------------------------------------------
# Locked vocoder
# --------------------------------------------------------------------------

@pytest.mark.parametrize("stretch", [0.5, 0.75, 1.0, 1.5, 2.0])
def test_locked_output_length(stretch):
    x = _harmonic()
    y = time_stretch_locked(x, stretch)
    assert abs(len(y) - int(round(len(x) * stretch))) <= 1


def test_locked_identity_is_transparent():
    x = _harmonic()
    assert ser(x, time_stretch_locked(x, 1.0)) > 40.0


@pytest.mark.parametrize("stretch", [0.5, 1.5, 2.0])
def test_locked_preserves_pitch(stretch):
    x = _harmonic(f0=220.0)
    assert abs(_f0_hz(time_stretch_locked(x, stretch)) - 220.0) < 3.0


@pytest.mark.parametrize("stretch", [1.25, 1.5, 2.0])
def test_locking_improves_consistency(stretch):
    """Vertical coherence should show up in D_M, not just by ear."""
    x = _harmonic()
    _, Y_plain = time_stretch(x, stretch, return_stft=True)
    _, Y_locked = time_stretch_locked(x, stretch, return_stft=True)

    d_plain = consistency(Y_plain, hop=512)
    d_locked = consistency(Y_locked, hop=512)

    assert d_locked < d_plain, f"locked {d_locked:.5f} vs plain {d_plain:.5f}"


def test_locking_reduces_warble():
    """THE payoff.

    On a pure tone the ideal output has a perfectly flat envelope, so any
    fluctuation is the algorithm's fault. Locking should cut it by an order of
    magnitude relative to the unlocked vocoder.
    """
    t = np.arange(2 * SR) / SR
    tone = np.sin(2 * np.pi * 220 * t)

    w_plain = amplitude_warble(time_stretch(tone, 2.0))
    w_locked = amplitude_warble(time_stretch_locked(tone, 2.0))

    assert w_locked < w_plain / 5.0, (
        f"locked warble {w_locked:.5f} vs plain {w_plain:.5f} -- "
        "expected at least a 5x reduction"
    )


def test_locking_helps_most_at_large_stretch():
    """The advantage should widen as frames are pulled further apart."""
    x = _harmonic()
    ratios = []
    for s in (1.25, 2.0):
        _, Yp = time_stretch(x, s, return_stft=True)
        _, Yl = time_stretch_locked(x, s, return_stft=True)
        ratios.append(consistency(Yp, hop=512) / max(consistency(Yl, hop=512), 1e-12))

    assert ratios[1] > ratios[0], f"improvement ratios {ratios}"


def test_locked_rejects_bad_stretch():
    with pytest.raises(ValueError):
        time_stretch_locked(_harmonic(duration=0.5), 0.0)
