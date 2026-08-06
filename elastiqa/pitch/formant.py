"""
Formant preservation: the source-filter model, and why chipmunks happen.

A voiced sound is well described as an excitation passed through a filter. The
vocal folds produce a buzzy, harmonically rich source at the fundamental
frequency; the vocal tract shapes it with resonances -- *formants* -- whose
frequencies are set by the geometry of throat, tongue, and lips. Formants are
what distinguish one vowel from another, and their absolute positions are a
strong cue to the size of the speaker.

The problem
-----------
Resampling scales the *entire* spectrum, so it moves the formants along with
the pitch. A voice shifted up an octave sounds like a small creature, not like
the same person singing higher, because the listener infers a shorter vocal
tract from the raised resonances. Real singers move pitch while holding
formants roughly fixed; naive pitch shifting cannot.

Separating the two
------------------
In the log-magnitude domain the model is additive::

    log |X(f)| = log |E(f)| + log |H(f)|

where ``E`` is the excitation (fast variation: individual harmonics) and ``H``
is the envelope (slow variation: formants). Taking the inverse Fourier
transform of the log spectrum gives the *cepstrum*, in which those two
components separate by rate of variation -- the envelope occupies the low
quefrencies, the harmonic comb sits at the quefrency of the pitch period.

Truncating the cepstrum ("liftering") therefore isolates the envelope.

The correction
--------------
Rather than shifting excitation and envelope separately, this module shifts
normally and then *repairs* the envelope: divide out the shifted signal's
envelope and multiply in the original's. Same result, far fewer places to go
wrong, and it composes with any pitch shifter.

    corrected = shifted * (envelope_original / envelope_shifted)
"""

from __future__ import annotations

import numpy as np

from ..stft import istft, stft
from .shift import pitch_shift

__all__ = ["cepstral_envelope", "formant_correct", "pitch_shift_formant"]

_EPS = 1e-10


def cepstral_envelope(
    magnitude: np.ndarray, quefrency: int = 40, n_fft: int | None = None
) -> np.ndarray:
    """Spectral envelope by cepstral liftering.

    Parameters
    ----------
    magnitude
        Magnitude spectrogram, shape ``(n_bins, n_frames)`` or ``(n_bins,)``.
    quefrency
        Number of cepstral coefficients to keep. This is the one parameter
        that matters, and it must sit well below the pitch period measured in
        samples: at 22.05 kHz a 165 Hz voice has a period of ~134 samples, so
        a quefrency of 40 separates the envelope cleanly. Too high and you
        start tracking individual harmonics (the "envelope" acquires a comb
        and the correction cancels itself out); too low and real formants get
        smoothed away.
    n_fft
        FFT size; inferred from the bin count when omitted.

    Returns
    -------
    ndarray
        Smooth positive envelope, same shape as ``magnitude``.
    """
    magnitude = np.asarray(magnitude, dtype=np.float64)
    squeeze = magnitude.ndim == 1
    if squeeze:
        magnitude = magnitude[:, None]

    n_bins = magnitude.shape[0]
    if n_fft is None:
        n_fft = 2 * (n_bins - 1)

    log_mag = np.log(np.maximum(magnitude, _EPS))

    # Real cepstrum: the log spectrum is real and even, so irfft returns a
    # real sequence of length n_fft.
    cep = np.fft.irfft(log_mag, n=n_fft, axis=0)

    # Lifter: keep low quefrencies, and their mirror at the top of the buffer.
    lifter = np.zeros(n_fft)
    q = min(quefrency, n_fft // 2)
    lifter[:q] = 1.0
    if q > 1:
        lifter[-(q - 1) :] = 1.0

    envelope = np.exp(np.fft.rfft(cep * lifter[:, None], axis=0).real)
    envelope = np.maximum(envelope[:n_bins], _EPS)

    return envelope[:, 0] if squeeze else envelope


def formant_correct(
    original: np.ndarray,
    shifted: np.ndarray,
    n_fft: int = 2048,
    hop: int | None = None,
    window: str = "hann",
    quefrency: int = 40,
    strength: float = 1.0,
) -> np.ndarray:
    """Restore ``original``'s spectral envelope onto ``shifted``.

    Parameters
    ----------
    original, shifted
        Signals of the same length. ``shifted`` should be a pitch-shifted
        version of ``original`` with duration preserved.
    quefrency
        Envelope smoothness; see `cepstral_envelope`.
    strength
        Interpolates between no correction (0.0) and full correction (1.0).
        Values above 1.0 over-correct, which is itself a usable effect -- it
        exaggerates the vocal-tract size cue in the opposite direction.

    Notes
    -----
    Correction gains are clipped to a 20 dB range. Without a clamp, bins where
    the shifted envelope is near zero produce enormous gains and the output
    explodes into noise -- most often at the top of the spectrum after a
    downward shift, where the shifted signal simply has no energy left.
    """
    hop = n_fft // 4 if hop is None else hop

    n = min(len(original), len(shifted))
    original, shifted = original[:n], shifted[:n]

    X = stft(original, n_fft=n_fft, hop=hop, window=window)
    Y = stft(shifted, n_fft=n_fft, hop=hop, window=window)

    frames = min(X.shape[1], Y.shape[1])
    X, Y = X[:, :frames], Y[:, :frames]

    env_x = cepstral_envelope(np.abs(X), quefrency=quefrency, n_fft=n_fft)
    env_y = cepstral_envelope(np.abs(Y), quefrency=quefrency, n_fft=n_fft)

    gain = env_x / np.maximum(env_y, _EPS)
    if strength != 1.0:
        gain = gain ** strength
    gain = np.clip(gain, 0.1, 10.0)

    return istft(Y * gain, hop=hop, window=window, length=n)


def pitch_shift_formant(
    x: np.ndarray,
    semitones: float,
    n_fft: int = 2048,
    hop: int | None = None,
    window: str = "hann",
    quefrency: int = 40,
    strength: float = 1.0,
    locked: bool = True,
) -> np.ndarray:
    """Pitch shift with formants held in place.

    The demo that sells the project: compare against `pitch_shift` at the same
    interval and the difference is immediate. A downward shift keeps sounding
    like a person rather than turning into a growl; an upward shift keeps
    sounding like a person rather than a chipmunk.

    Parameters
    ----------
    semitones
        Interval in 12-tone equal temperament.
    locked
        Use the phase-locked vocoder for the underlying stretch.
    strength
        Correction amount; see `formant_correct`.

    Examples
    --------
        plain     = eq.pitch_shift(x, -7)           # formants move too
        preserved = eq.pitch_shift_formant(x, -7)   # formants stay put
    """
    hop = n_fft // 4 if hop is None else hop

    if locked:
        from ..naive import resample, semitones_to_ratio
        from ..tsm.phase_lock import time_stretch_locked

        ratio = semitones_to_ratio(semitones)
        stretched = time_stretch_locked(
            x, stretch=ratio, n_fft=n_fft, hop=hop, window=window
        )
        shifted = resample(stretched, 1.0 / ratio)
        if len(shifted) < len(x):
            shifted = np.pad(shifted, (0, len(x) - len(shifted)))
        shifted = shifted[: len(x)]
    else:
        shifted = pitch_shift(x, semitones, n_fft=n_fft, hop=hop, window=window)

    return formant_correct(
        x, shifted, n_fft=n_fft, hop=hop, window=window,
        quefrency=quefrency, strength=strength,
    )
