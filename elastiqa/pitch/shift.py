"""
Pitch shifting.

Once time stretching works, pitch shifting is a composition of two operations
you already have::

    stretch by s, then resample by 1/s

The stretch changes duration but not pitch. The resample changes both, by the
inverse factor. Duration returns to where it started; pitch does not.

What this does *not* fix
------------------------
Resampling scales the entire spectrum, formants included. Shift a voice up and
the vocal-tract resonances move up with it, which is why the result still
sounds cartoonish even though the algorithm is "correct". Separating the
spectral envelope from the excitation is week 4's job; until then, expect
chipmunks with better phase coherence.
"""

from __future__ import annotations

import numpy as np

from ..naive import resample, semitones_to_ratio
from ..tsm.vocoder import time_stretch

__all__ = ["pitch_shift"]


def pitch_shift(
    x: np.ndarray,
    semitones: float,
    n_fft: int = 2048,
    hop: int | None = None,
    window: str = "hann",
    mode: str = "vocoder",
) -> np.ndarray:
    """Shift pitch by ``semitones`` while preserving duration.

    Parameters
    ----------
    x
        Real 1-D signal.
    semitones
        Interval in 12-tone equal temperament. ``+12`` is one octave up,
        ``-7`` is a perfect fifth down.
    mode
        Passed to the underlying vocoder, so ``mode="passthrough"`` gives the
        broken-phase comparison for the same shift.

    Examples
    --------
        low  = pitch_shift(x, -7)      # a fifth down, same length
        high = pitch_shift(x, +5)
    """
    ratio = semitones_to_ratio(semitones)

    # Stretch first (longer for an upward shift), then resample back down.
    stretched = time_stretch(
        x, stretch=ratio, n_fft=n_fft, hop=hop, window=window, mode=mode
    )
    shifted = resample(stretched, 1.0 / ratio)

    n = len(x)
    if len(shifted) < n:
        shifted = np.pad(shifted, (0, n - len(shifted)))
    return shifted[:n]
