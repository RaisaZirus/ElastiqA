"""
Time-scale modification.

Two properties define a working phase vocoder, and both are tested here:

1. Duration changes by the requested factor.
2. Pitch does not change at all.

Property 2 is the whole point. ``test_pitch_is_preserved`` is the test that
distinguishes this from naive resampling, and it should be the first thing you
check after touching anything in the phase path.
"""

import numpy as np
import pytest

from elastiqa.metrics import consistency, ser
from elastiqa.pitch import pitch_shift
from elastiqa.tsm import PHASE_MODES, ola, time_stretch

SR = 22050


def _harmonic(f0=220.0, duration=2.0, n_harmonics=20, sr=SR):
    """A tonal signal. Tonal material is where phase errors are audible."""
    t = np.arange(int(duration * sr)) / sr
    x = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, n_harmonics + 1))
    return x / np.max(np.abs(x))


def _dominant_hz(x, sr=SR):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1 / sr)[np.argmax(spec)])


# --------------------------------------------------------------------------
# Length and identity
# --------------------------------------------------------------------------

@pytest.mark.parametrize("stretch", [0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
def test_output_length(stretch):
    x = _harmonic()
    y = time_stretch(x, stretch)
    assert abs(len(y) - int(round(len(x) * stretch))) <= 1


def test_identity_stretch_is_transparent():
    """At stretch = 1 the vocoder must return the input essentially unchanged.

    This is the vocoder's own version of the perfect-reconstruction gate. If
    it fails, no amount of phase-locking work in week 3 will help.
    """
    x = _harmonic()
    y = time_stretch(x, 1.0)
    assert ser(x, y) > 40.0, f"identity SER only {ser(x, y):.1f} dB"


@pytest.mark.parametrize("stretch", [0.5, 1.5, 2.0])
def test_pitch_is_preserved(stretch):
    """THE defining property. Duration changes; the fundamental does not."""
    x = _harmonic(f0=220.0)
    y = time_stretch(x, stretch)

    assert abs(_dominant_hz(y) - 220.0) < 3.0, (
        f"stretch={stretch} moved the fundamental to {_dominant_hz(y):.1f} Hz"
    )


def test_naive_baseline_does_change_pitch():
    """Sanity check on the comparison: the baseline must fail this test."""
    from elastiqa.naive import speed_change

    x = _harmonic(f0=220.0)
    y = speed_change(x, 2.0)
    assert abs(_dominant_hz(y) - 440.0) < 5.0


# --------------------------------------------------------------------------
# Phase modes
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode", list(PHASE_MODES))
def test_all_modes_run_and_are_finite(mode):
    x = _harmonic(duration=1.0)
    y = time_stretch(x, 1.5, mode=mode, seed=0)
    assert np.all(np.isfinite(y))
    assert len(y) > 0


@pytest.mark.parametrize("stretch", [1.25, 1.5, 2.0])
def test_vocoder_beats_passthrough_on_consistency(stretch):
    """The week 2 headline result.

    Correct phase propagation must produce a measurably more consistent STFT
    than reusing analysis phase. If this inequality ever flips, the phase
    accumulation has broken.
    """
    x = _harmonic()

    _, Y_good = time_stretch(x, stretch, mode="vocoder", return_stft=True)
    _, Y_bad = time_stretch(x, stretch, mode="passthrough", return_stft=True)

    d_good = consistency(Y_good, hop=512)
    d_bad = consistency(Y_bad, hop=512)

    assert d_good < d_bad, f"vocoder {d_good:.4f} not better than passthrough {d_bad:.4f}"


def test_passthrough_degrades_with_stretch():
    """Phasiness should get worse as frames are pulled further apart."""
    x = _harmonic()
    scores = []
    for s in (1.25, 1.5, 2.0):
        _, Y = time_stretch(x, s, mode="passthrough", return_stft=True)
        scores.append(consistency(Y, hop=512))
    assert scores[0] < scores[1] < scores[2], scores


def test_destroying_phase_is_worst():
    """Zeroed and randomised phase should score worse than either real mode."""
    x = _harmonic()

    scores = {}
    for mode in PHASE_MODES:
        _, Y = time_stretch(x, 1.5, mode=mode, return_stft=True, seed=0)
        scores[mode] = consistency(Y, hop=512)

    assert scores["vocoder"] < scores["passthrough"]
    assert scores["passthrough"] < scores["random"]
    assert scores["passthrough"] < scores["zero"]


def test_magnitude_spectrogram_is_roughly_preserved():
    """Phase modes change phase, not magnitude. The spectrogram should survive."""
    from elastiqa.stft import stft

    x = _harmonic()
    y = time_stretch(x, 1.5)

    mx = np.abs(stft(x, 2048, 512)).mean(axis=1)
    my = np.abs(stft(y, 2048, 512)).mean(axis=1)

    # Compare the shape of the average spectrum, not its absolute level.
    mx, my = mx / mx.max(), my / my.max()
    assert np.corrcoef(mx, my)[0, 1] > 0.95


def test_rejects_bad_arguments():
    x = _harmonic(duration=0.5)
    with pytest.raises(ValueError):
        time_stretch(x, 0.0)
    with pytest.raises(ValueError):
        time_stretch(x, -1.0)
    with pytest.raises(ValueError):
        time_stretch(x, 1.5, mode="nonsense")


# --------------------------------------------------------------------------
# OLA baseline
# --------------------------------------------------------------------------

@pytest.mark.parametrize("stretch", [0.5, 1.5, 2.0])
def test_ola_length(stretch):
    x = _harmonic()
    y = ola(x, stretch)
    assert abs(len(y) - int(round(len(x) * stretch))) <= 1


def test_ola_decouples_pitch_from_speed_but_imprecisely():
    """OLA sits between naive resampling and the vocoder, and the gap is large.

    Measured on a 220 Hz harmonic stack at stretch = 1.5:

        naive resampling  ~147 Hz   (pitch fully dragged along)
        OLA               ~233 Hz   (decoupled, but smeared by ~13 Hz)
        phase vocoder      220 Hz   (exact)

    OLA's error comes from laying frames down at arbitrary relative phase,
    which modulates the signal at the frame rate and throws sidebands around
    every harmonic. WSOLA (week 3) fixes precisely this by searching for a
    frame offset that maximises waveform similarity.
    """
    from elastiqa.naive import speed_change

    x = _harmonic(f0=220.0)

    err_naive = abs(_dominant_hz(speed_change(x, 1 / 1.5)) - 220.0)
    err_ola = abs(_dominant_hz(ola(x, 1.5)) - 220.0)
    err_pv = abs(_dominant_hz(time_stretch(x, 1.5)) - 220.0)

    assert err_pv < err_ola < err_naive, (
        f"pv {err_pv:.1f} Hz, ola {err_ola:.1f} Hz, naive {err_naive:.1f} Hz"
    )
    assert err_ola < 25.0        # in the right ballpark ...
    assert err_pv < 3.0          # ... but the vocoder is exact


def test_ola_is_worse_than_vocoder_on_tonal_material():
    """On sustained harmonics the vocoder should win on spectral distance."""
    from elastiqa.metrics import log_spectral_distance

    x = _harmonic()
    lsd_pv = log_spectral_distance(x, time_stretch(x, 1.5))
    lsd_ola = log_spectral_distance(x, ola(x, 1.5))
    assert lsd_pv < lsd_ola, f"pv {lsd_pv:.2f} dB vs ola {lsd_ola:.2f} dB"


# --------------------------------------------------------------------------
# Pitch shifting
# --------------------------------------------------------------------------

@pytest.mark.parametrize("semitones,expected", [(12, 440.0), (-12, 110.0), (7, 329.6)])
def test_pitch_shift_moves_the_fundamental(semitones, expected):
    x = _harmonic(f0=220.0)
    y = pitch_shift(x, semitones)
    assert abs(_dominant_hz(y) - expected) < expected * 0.03


def test_pitch_shift_preserves_duration():
    x = _harmonic()
    for st in (-7, -3, 0, 5, 12):
        assert len(pitch_shift(x, st)) == len(x)


def test_zero_semitones_is_near_identity():
    x = _harmonic()
    assert ser(x, pitch_shift(x, 0)) > 20.0
