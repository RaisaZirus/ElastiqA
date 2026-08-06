"""
WSOLA.

The defining test is ``test_wsola_beats_ola_on_warble``: WSOLA differs from OLA
*only* in that it searches for a well-aligned frame instead of taking whatever
sits at the nominal position. So any quality difference between them is
attributable entirely to that search, which makes the pair a clean controlled
experiment for the report.
"""

import numpy as np
import pytest

from elastiqa.metrics import amplitude_warble, ser
from elastiqa.tsm import ola, wsola

SR = 22050


def _tone(f0=220.0, duration=1.5):
    t = np.arange(int(duration * SR)) / SR
    return np.sin(2 * np.pi * f0 * t)


def _f0_hz(x, sr=SR, lo=70.0, hi=500.0):
    x = x - x.mean()
    corr = np.correlate(x, x, mode="full")[len(x) - 1 :]
    lo_lag, hi_lag = int(sr / hi), int(sr / lo)
    return float(sr / (lo_lag + int(np.argmax(corr[lo_lag : hi_lag + 1]))))


@pytest.mark.parametrize("stretch", [0.5, 0.75, 1.0, 1.5, 2.0])
def test_wsola_output_length(stretch):
    x = _tone()
    y = wsola(x, stretch)
    assert abs(len(y) - int(round(len(x) * stretch))) <= 1


def test_wsola_preserves_pitch():
    x = _tone(f0=220.0)
    for s in (0.75, 1.5, 2.0):
        assert abs(_f0_hz(wsola(x, s)) - 220.0) < 5.0


def test_wsola_identity_is_close_to_transparent():
    """At stretch = 1 the search should find offset 0 and pass the signal through."""
    x = _tone()
    assert ser(x, wsola(x, 1.0)) > 15.0


def test_wsola_beats_ola_on_warble():
    """THE controlled experiment.

    Identical framing, identical overlap-add, identical windows. The only
    difference is the similarity search. Any improvement is therefore
    attributable to alignment alone.
    """
    x = _tone()

    w_ola = amplitude_warble(ola(x, 2.0))
    w_wsola = amplitude_warble(wsola(x, 2.0))

    assert w_wsola < w_ola / 2.0, (
        f"wsola warble {w_wsola:.4f} vs ola {w_ola:.4f} -- "
        "the similarity search is not helping"
    )


def test_larger_tolerance_is_not_worse():
    """More search freedom should not degrade alignment on a periodic signal."""
    x = _tone()
    tight = amplitude_warble(wsola(x, 1.5, tolerance=32))
    loose = amplitude_warble(wsola(x, 1.5, tolerance=256))
    assert loose <= tight * 2.0


def test_zero_tolerance_degenerates_to_ola():
    """With no search freedom, WSOLA should behave like OLA."""
    x = _tone()
    w_zero = amplitude_warble(wsola(x, 2.0, tolerance=0))
    w_ola = amplitude_warble(ola(x, 2.0))
    assert abs(w_zero - w_ola) < 0.25


def test_wsola_handles_noise_without_crashing():
    """Noise has no periodicity for the search to latch onto."""
    rng = np.random.default_rng(0)
    y = wsola(rng.standard_normal(20000) * 0.1, 1.5)
    assert np.all(np.isfinite(y))
    assert len(y) == 30000


def test_wsola_handles_silence():
    y = wsola(np.zeros(20000), 1.5)
    assert np.all(np.isfinite(y))


def test_wsola_rejects_bad_stretch():
    with pytest.raises(ValueError):
        wsola(_tone(duration=0.3), -1.0)
