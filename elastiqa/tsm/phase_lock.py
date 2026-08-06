"""
Phase locking: enforcing vertical coherence.

The standard phase vocoder treats every frequency bin as an independent
sinusoid and propagates its phase separately. That preserves *horizontal*
coherence -- each bin's phase evolves correctly along time -- but says nothing
about the relationship *between* bins at a given instant.

Why that matters
----------------
A single sinusoid does not occupy one bin. Windowing spreads it across a main
lobe several bins wide, and the bins of that lobe have a fixed phase
relationship determined by the window. Propagating them independently lets
that relationship drift. The lobe stops describing one coherent sinusoid and
starts describing several slightly disagreeing ones, which is heard as
phasiness, reverberance, or "loss of presence".

The fix (Laroche & Dolson 1999)
-------------------------------
Pick the spectral peaks. Assign every bin to the nearest peak, forming that
peak's *region of influence*. Accumulate phase only at the peak, then set every
other bin in the region so that its phase relative to the peak matches the
analysis frame::

    phi_s[k] = phi_s[p] + (phi_a[k] - phi_a[p])      for k in region(p)

The lobe now moves as a rigid body. Relative phase within it is exactly what
the analysis said it should be, so the lobe still describes one sinusoid.

Identity vs. scaled locking
---------------------------
This module implements *identity* phase locking, the simpler of the two schemes
in the paper: the offset from peak to bin is copied verbatim. *Scaled* phase
locking additionally tracks peaks from frame to frame and scales the offset,
which handles peaks that move between bins. Identity locking captures most of
the benefit for a fraction of the complexity, and published evaluation finds it
the best-scoring classical method on music signals -- so it is a sound place to
stop.
"""

from __future__ import annotations

import numpy as np

from ..phase import expected_phase_advance, princarg
from ..stft import istft, stft

__all__ = [
    "find_peaks",
    "regions_of_influence",
    "locked_phase_vocoder_stft",
    "time_stretch_locked",
]


def find_peaks(magnitude: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    """Indices of spectral peaks in a single magnitude frame.

    A bin is a peak when it exceeds its two neighbours on each side. The
    four-neighbour test (rather than two) is what Laroche and Dolson use; it
    rejects the small ripples that a single-neighbour test would promote into
    spurious peaks, each of which would then claim a region and lock bins to a
    meaningless reference.

    Parameters
    ----------
    magnitude
        Non-negative magnitudes for one frame, shape ``(n_bins,)``.
    threshold
        Peaks below ``threshold * magnitude.max()`` are discarded. Without
        this, near-silent frames produce dozens of noise peaks.

    Returns
    -------
    ndarray
        Sorted peak bin indices. Possibly empty.
    """
    magnitude = np.asarray(magnitude)
    n = len(magnitude)
    if n < 5:
        return np.array([], dtype=int)

    m = magnitude
    interior = np.arange(2, n - 2)
    is_peak = (
        (m[interior] > m[interior - 1])
        & (m[interior] > m[interior + 1])
        & (m[interior] > m[interior - 2])
        & (m[interior] > m[interior + 2])
    )
    peaks = interior[is_peak]

    if peaks.size and threshold > 0:
        peaks = peaks[m[peaks] > threshold * m.max()]

    return peaks


def regions_of_influence(peaks: np.ndarray, n_bins: int) -> np.ndarray:
    """Map every bin to the peak that governs it.

    Boundaries sit at the midpoint between consecutive peaks, so each bin is
    assigned to whichever peak is nearer.

    Returns
    -------
    ndarray, shape ``(n_bins,)``
        ``owner[k]`` is the bin index of the peak governing bin ``k``. When
        there are no peaks, every bin owns itself, which degrades gracefully
        to the unlocked vocoder.
    """
    owner = np.arange(n_bins)
    if len(peaks) == 0:
        return owner

    # Midpoints between adjacent peaks are the region boundaries.
    bounds = ((peaks[:-1] + peaks[1:]) // 2) + 1
    edges = np.concatenate([[0], bounds, [n_bins]])

    for i, p in enumerate(peaks):
        owner[edges[i] : edges[i + 1]] = p

    return owner


def locked_phase_vocoder_stft(
    X: np.ndarray,
    stretch: float,
    hop: int,
    n_fft: int | None = None,
    threshold: float = 1e-4,
) -> np.ndarray:
    """Time-scale an STFT with identity phase locking.

    Same structure as the unlocked vocoder: fractional read positions,
    interpolated magnitudes, accumulated phase. The difference is the final
    step, where non-peak bins are slaved to their region's peak instead of
    using their own accumulated phase.

    Parameters
    ----------
    threshold
        Relative magnitude floor for peak detection. Raise it if quiet frames
        are producing noisy peak sets.
    """
    if stretch <= 0:
        raise ValueError(f"stretch must be positive; got {stretch}")

    X = np.asarray(X)
    n_bins, n_frames = X.shape
    if n_fft is None:
        n_fft = 2 * (n_bins - 1)
    if n_frames < 2:
        return X.copy()

    omega = expected_phase_advance(n_fft, hop)
    steps = np.arange(0, n_frames - 1, 1.0 / stretch)

    mag = np.abs(X)
    ang = np.angle(X)

    Y = np.zeros((n_bins, len(steps)), dtype=np.complex128)
    phase = ang[:, 0].copy()

    for i, t in enumerate(steps):
        left = int(np.floor(t))
        frac = t - left
        right = min(left + 1, n_frames - 1)

        m = (1.0 - frac) * mag[:, left] + frac * mag[:, right]

        # Lock every bin to the phase of the peak that governs it, keeping the
        # analysis frame's peak-to-bin offset. This is the whole technique.
        peaks = find_peaks(m, threshold=threshold)
        owner = regions_of_influence(peaks, n_bins)

        locked = phase[owner] + (ang[:, left] - ang[owner, left])
        Y[:, i] = m * np.exp(1j * locked)

        dphi = ang[:, right] - ang[:, left]
        deviation = princarg(dphi - omega)
        phase = phase + omega + deviation

    return Y


def time_stretch_locked(
    x: np.ndarray,
    stretch: float,
    n_fft: int = 2048,
    hop: int | None = None,
    window: str = "hann",
    threshold: float = 1e-4,
    return_stft: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Time stretching with identity phase locking.

    Drop-in replacement for `elastiqa.time_stretch` with the same signature,
    so the two can be swapped in a benchmark loop.

    Examples
    --------
        plain  = eq.time_stretch(x, 1.5)
        locked = eq.time_stretch_locked(x, 1.5)
    """
    hop = n_fft // 4 if hop is None else hop

    X = stft(x, n_fft=n_fft, hop=hop, window=window)
    Y = locked_phase_vocoder_stft(
        X, stretch, hop=hop, n_fft=n_fft, threshold=threshold
    )

    target = int(round(len(x) * stretch))
    y = istft(Y, hop=hop, window=window, length=target)

    return (y, Y) if return_stft else y
