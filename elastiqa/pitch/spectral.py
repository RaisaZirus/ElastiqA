"""
Spectral-domain pitch shifting with a time-varying ratio.

Weeks 1-4 shifted pitch by composing two operations: stretch by ``r``, then
resample by ``1/r``. That is correct and easy to reason about, but it can only
apply one ratio to the whole signal. Auto-tune needs a *different* ratio on
every frame, because the singer's error changes note by note.

Doing it in one pass
--------------------
Work directly on the STFT and move the spectrum along the frequency axis::

    mag_shifted[k]  = mag[k / r]                    (interpolated)
    freq_shifted[k] = r * instantaneous_freq[k / r]

Magnitude is resampled along frequency, and the instantaneous frequency
carried by each bin is both relocated *and* scaled -- a partial that was at
300 Hz must now advance in phase as if it were at 300r Hz. Accumulating that
scaled frequency gives synthesis phase.

Duration is untouched because the number of frames never changes, so no
resampling step is needed and ``r`` can vary per frame at no extra cost.

Trade-off against stretch-and-resample
--------------------------------------
Cheaper (one STFT pass), and it supports time-varying ratios. Slightly softer
at the top of the spectrum, because linear interpolation of the magnitude
spectrum is a cruder reconstruction than resampling in the time domain. For a
fixed ratio on critical material, `pitch_shift_formant` remains the better
choice; for anything time-varying, this is the only option.
"""

from __future__ import annotations

import numpy as np

from ..phase import instantaneous_frequency
from ..stft import istft, stft
from .formant import cepstral_envelope

__all__ = ["spectral_pitch_shift"]

_EPS = 1e-10


def spectral_pitch_shift(
    x: np.ndarray,
    ratio: float | np.ndarray,
    n_fft: int = 2048,
    hop: int | None = None,
    window: str = "hann",
    preserve_formants: bool = True,
    quefrency: int = 40,
) -> np.ndarray:
    """Shift pitch in the STFT domain, preserving duration.

    Parameters
    ----------
    x
        Real 1-D signal.
    ratio
        Frequency multiplier. Either a scalar, or an array with one value per
        STFT frame for a time-varying shift. Arrays of a different length are
        resampled onto the frame grid, so an f0 track computed with a different
        hop still works.
    preserve_formants
        Restore the original spectral envelope after shifting. On by default,
        because without it every shift sounds like a chipmunk.
    quefrency
        Envelope smoothness; see `cepstral_envelope`.

    Examples
    --------
        up = spectral_pitch_shift(x, 1.5)                 # constant

        ratios = target_hz / detected_hz                  # per frame
        tuned = spectral_pitch_shift(x, ratios)
    """
    hop = n_fft // 4 if hop is None else hop

    x = np.asarray(x, dtype=np.float64)
    X = stft(x, n_fft=n_fft, hop=hop, window=window)
    n_bins, n_frames = X.shape

    ratios = np.atleast_1d(np.asarray(ratio, dtype=np.float64))
    if ratios.size == 1:
        ratios = np.full(n_frames, float(ratios[0]))
    elif ratios.size != n_frames:
        # Resample the ratio curve onto the frame grid.
        src = np.linspace(0.0, 1.0, ratios.size)
        dst = np.linspace(0.0, 1.0, n_frames)
        ratios = np.interp(dst, src, ratios)

    ratios = np.maximum(ratios, _EPS)

    mag = np.abs(X)
    ifreq = instantaneous_frequency(X, hop=hop, n_fft=n_fft)

    bins = np.arange(n_bins, dtype=np.float64)
    Y = np.zeros_like(X)
    phase = np.angle(X[:, 0]).copy()

    for i in range(n_frames):
        r = ratios[i]
        source = bins / r                       # where each output bin reads from

        # Bins whose source lies beyond Nyquist have no content to carry.
        valid = source <= n_bins - 1

        m = np.zeros(n_bins)
        f = np.zeros(n_bins)
        m[valid] = np.interp(source[valid], bins, mag[:, i])
        f[valid] = r * np.interp(source[valid], bins, ifreq[:, i])

        phase = phase + f * hop
        Y[:, i] = m * np.exp(1j * phase)

    if preserve_formants:
        env_original = cepstral_envelope(mag, quefrency=quefrency, n_fft=n_fft)
        env_shifted = cepstral_envelope(np.abs(Y), quefrency=quefrency, n_fft=n_fft)
        gain = np.clip(env_original / np.maximum(env_shifted, _EPS), 0.1, 10.0)
        Y = Y * gain

    return istft(Y, hop=hop, window=window, length=len(x))
