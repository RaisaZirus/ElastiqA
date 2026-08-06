"""
The naive baseline.

These tests do not check that naive resampling sounds good -- it does not.
They check that it fails in exactly the documented way, because the whole
project is framed as a response to that failure. If the baseline stopped
coupling pitch to speed, the motivation would evaporate.
"""

import numpy as np
import pytest

from elastiqa.naive import (
    ratio_to_semitones,
    resample,
    semitones_to_ratio,
    speed_change,
)

SR = 22050


def _dominant_hz(x, sr=SR):
    """Frequency of the largest spectral peak."""
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1 / sr)[np.argmax(spec)])


@pytest.mark.parametrize("ratio", [0.5, 0.75, 1.0, 1.5, 2.0])
def test_output_length_follows_ratio(ratio):
    x = np.zeros(10000)
    assert len(resample(x, ratio)) == int(round(10000 * ratio))


def test_identity_ratio_is_lossless_enough():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(5000)
    assert np.allclose(resample(x, 1.0), x)


@pytest.mark.parametrize("speed,expected", [(2.0, 880.0), (0.5, 220.0)])
def test_speed_change_drags_pitch_along(speed, expected):
    """The defining flaw: changing duration changes pitch by the same factor."""
    t = np.arange(2 * SR) / SR
    x = np.sin(2 * np.pi * 440 * t)

    y = speed_change(x, speed)
    assert abs(_dominant_hz(y) - expected) < 5.0
    assert abs(len(y) - len(x) / speed) <= 1


def test_semitone_conversions_round_trip():
    for st in [-12, -7, -1, 0, 1, 5, 12, 24]:
        assert np.isclose(ratio_to_semitones(semitones_to_ratio(st)), st)


def test_octave_is_factor_of_two():
    assert np.isclose(semitones_to_ratio(12), 2.0)
    assert np.isclose(semitones_to_ratio(-12), 0.5)


def test_rejects_bad_ratios():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            resample(np.zeros(100), bad)
        with pytest.raises(ValueError):
            speed_change(np.zeros(100), bad)


def test_downsampling_aliases():
    """Without an anti-alias filter, a high tone folds back down.

    Kept deliberately: the aliasing demo is one of the clearest illustrations
    of the sampling theorem you can put in a presentation.
    """
    t = np.arange(SR) / SR
    x = np.sin(2 * np.pi * 8000 * t)      # well above SR/4

    y = resample(x, 0.25)                 # effective rate SR/4 = 5512.5 Hz
    # 8000 Hz sampled at 5512.5 Hz folds to |8000 - 5512.5| = 2487.5 Hz.
    assert _dominant_hz(y, sr=SR / 4) < 3000.0
