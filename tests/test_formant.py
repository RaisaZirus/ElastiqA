"""
Formant preservation.

``test_formants_stay_put`` is the test that defines the feature. It uses a
synthetic vowel with formants at known frequencies, so "did the resonances
move?" has an exact answer rather than a subjective one.
"""

import numpy as np
import pytest

from elastiqa.pitch import cepstral_envelope, pitch_shift, pitch_shift_formant
from elastiqa.stft import bin_frequencies, stft
from elastiqa.tsm import find_peaks

SR = 22050
N_FFT = 2048
FORMANTS = (500.0, 1500.0, 2500.0)


def _vowel(f0=165.0, duration=2.0, formants=FORMANTS):
    """Harmonic stack shaped by fixed resonances."""
    t = np.arange(int(duration * SR)) / SR
    x = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 40))

    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    envelope = np.full_like(freqs, 0.02)
    for centre, bw, gain in zip(formants, (90, 120, 160), (1.0, 0.5, 0.3)):
        envelope += gain / (1.0 + ((freqs - centre) / bw) ** 2)

    x = np.fft.irfft(spec * envelope, n=len(x))
    return x / np.max(np.abs(x))


def _f0_hz(x, sr=SR, lo=70.0, hi=600.0):
    x = x - x.mean()
    corr = np.correlate(x, x, mode="full")[len(x) - 1 :]
    lo_lag, hi_lag = int(sr / hi), int(sr / lo)
    return float(sr / (lo_lag + int(np.argmax(corr[lo_lag : hi_lag + 1]))))


def _formant_peaks(x, n=3, fmax=3500.0):
    """The n strongest envelope peaks below fmax, in Hz, sorted ascending."""
    X = stft(x, N_FFT, 512)
    env = cepstral_envelope(np.abs(X).mean(axis=1), quefrency=40)
    freqs = bin_frequencies(N_FFT, SR)

    upto = int(np.searchsorted(freqs, fmax))
    peaks = find_peaks(env[:upto], threshold=0.0)
    if len(peaks) == 0:
        return []

    strongest = peaks[np.argsort(env[peaks])[::-1][:n]]
    return sorted(float(freqs[p]) for p in strongest)


# --------------------------------------------------------------------------
# Cepstral envelope
# --------------------------------------------------------------------------

def test_envelope_is_positive_and_right_shape():
    X = stft(_vowel(), N_FFT, 512)
    env = cepstral_envelope(np.abs(X))
    assert env.shape == X.shape
    assert np.all(env > 0)


def test_envelope_handles_single_frame():
    X = stft(_vowel(), N_FFT, 512)
    env = cepstral_envelope(np.abs(X[:, 10]))
    assert env.shape == (X.shape[0],)


def test_envelope_is_smoother_than_the_spectrum():
    """The point of liftering: keep the resonances, drop the harmonic comb."""
    X = stft(_vowel(), N_FFT, 512)
    mag = np.abs(X[:, X.shape[1] // 2])
    env = cepstral_envelope(mag, quefrency=40)

    rough_mag = np.sum(np.abs(np.diff(np.log(np.maximum(mag, 1e-10)))))
    rough_env = np.sum(np.abs(np.diff(np.log(env))))
    assert rough_env < rough_mag / 3.0


def test_envelope_finds_the_formants():
    peaks = _formant_peaks(_vowel())
    assert len(peaks) == 3
    for found, expected in zip(peaks, FORMANTS):
        assert abs(found - expected) < 120.0, f"{peaks} vs {FORMANTS}"


def test_low_quefrency_smooths_more():
    X = stft(_vowel(), N_FFT, 512)
    mag = np.abs(X[:, X.shape[1] // 2])

    def roughness(q):
        return float(np.sum(np.abs(np.diff(np.log(cepstral_envelope(mag, q))))))

    assert roughness(15) < roughness(80)


# --------------------------------------------------------------------------
# Formant-preserving pitch shift
# --------------------------------------------------------------------------

@pytest.mark.parametrize("semitones", [-7, -3, 5, 7])
def test_pitch_still_moves(semitones):
    """Correction must not undo the pitch shift itself."""
    x = _vowel(f0=165.0)
    y = pitch_shift_formant(x, semitones)
    expected = 165.0 * 2 ** (semitones / 12.0)
    assert abs(_f0_hz(y) - expected) < expected * 0.05


@pytest.mark.parametrize("semitones", [-7, 7])
def test_formants_stay_put(semitones):
    """THE defining test.

    Plain shifting scales the formants by the pitch ratio. Correction must
    leave them where they started, to within a fraction of a bandwidth.
    """
    x = _vowel()
    original = _formant_peaks(x)

    plain = _formant_peaks(pitch_shift(x, semitones))
    corrected = _formant_peaks(pitch_shift_formant(x, semitones))

    err_plain = sum(abs(a - b) for a, b in zip(plain, original))
    err_corrected = sum(abs(a - b) for a, b in zip(corrected, original))

    assert err_corrected < err_plain / 3.0, (
        f"original {original}, plain {plain}, corrected {corrected}"
    )


def test_plain_shift_does_move_formants():
    """Sanity check on the comparison: the baseline must fail this."""
    x = _vowel()
    original = _formant_peaks(x)
    shifted = _formant_peaks(pitch_shift(x, -7))

    ratio = 2 ** (-7 / 12.0)
    for orig, shift in zip(original, shifted):
        assert abs(shift - orig * ratio) < orig * 0.2


def test_length_is_preserved():
    x = _vowel()
    for st in (-12, -5, 0, 5, 12):
        assert len(pitch_shift_formant(x, st)) == len(x)


def test_output_is_finite_and_bounded():
    """The gain clamp must stop the correction exploding in empty bins."""
    x = _vowel()
    for st in (-12, -7, 7, 12):
        y = pitch_shift_formant(x, st)
        assert np.all(np.isfinite(y))
        assert np.max(np.abs(y)) < 50.0


def test_strength_zero_is_close_to_uncorrected():
    x = _vowel()
    weak = _formant_peaks(pitch_shift_formant(x, -7, strength=0.0))
    plain = _formant_peaks(pitch_shift(x, -7))
    err = sum(abs(a - b) for a, b in zip(weak, plain))
    assert err < 200.0


def test_works_on_noise_without_blowing_up():
    rng = np.random.default_rng(0)
    y = pitch_shift_formant(rng.standard_normal(20000) * 0.1, 5)
    assert np.all(np.isfinite(y))
