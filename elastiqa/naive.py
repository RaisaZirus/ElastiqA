"""
Naive resampling: the baseline the whole project exists to beat.

Playing a recording back at a different sample rate changes its duration and
its pitch together, locked to the same ratio. This is what a tape machine does
when you spin it faster, and it is why a sped-up voice sounds like a chipmunk:
every frequency component -- including the vocal-tract resonances (formants)
that identify the speaker -- is scaled by the same factor.

Every subsequent module in ElastiqA is an attempt to break that coupling.
Keep this baseline in every comparison; it is the clearest possible statement
of the problem.
"""

from __future__ import annotations

import numpy as np

__all__ = ["resample", "speed_change", "semitones_to_ratio", "ratio_to_semitones"]


def resample(x: np.ndarray, ratio: float) -> np.ndarray:
    """Resample by linear interpolation.

    Parameters
    ----------
    x
        Input signal.
    ratio
        Output length multiplier. ``ratio = 2.0`` produces twice as many
        samples, so playback at the original sample rate is twice as long and
        one octave lower.

    Notes
    -----
    Linear interpolation is a crude reconstruction filter. Downsampling
    (``ratio < 1``) without first low-pass filtering will alias; that is
    deliberate here, because demonstrating the aliasing is pedagogically
    useful. Use ``scipy.signal.resample_poly`` if you want it done properly.
    """
    x = np.asarray(x, dtype=np.float64)
    if ratio <= 0:
        raise ValueError(f"ratio must be positive; got {ratio}")

    n_out = int(round(len(x) * ratio))
    if n_out < 2 or len(x) < 2:
        return np.zeros(max(n_out, 0), dtype=np.float64)

    # Map output sample positions back onto the input grid.
    positions = np.linspace(0.0, len(x) - 1.0, n_out)
    return np.interp(positions, np.arange(len(x)), x)


def speed_change(x: np.ndarray, speed: float) -> np.ndarray:
    """Play back ``speed`` times faster, with pitch dragged along for the ride.

    ``speed = 2.0`` gives half the duration and one octave up.
    """
    if speed <= 0:
        raise ValueError(f"speed must be positive; got {speed}")
    return resample(x, 1.0 / speed)


def semitones_to_ratio(semitones: float) -> float:
    """Convert a musical interval to a frequency ratio (12-TET)."""
    return float(2.0 ** (semitones / 12.0))


def ratio_to_semitones(ratio: float) -> float:
    """Convert a frequency ratio to semitones (12-TET)."""
    if ratio <= 0:
        raise ValueError(f"ratio must be positive; got {ratio}")
    return float(12.0 * np.log2(ratio))
