"""
Transient detection and phase reset.

The phase vocoder models every bin as a sinusoid of slowly varying frequency.
A drum hit is the opposite of that: broadband, and over in a few milliseconds.
Accumulating phase across an onset averages the attack over the whole analysis
window, so the hit arrives softened and spread -- audible as smearing, and as
"pre-echo" when energy leaks backwards in time ahead of the strike.

The fix
-------
Detect onsets, and at those frames throw away the accumulated phase and adopt
the analysis phase instead. The frames that describe the attack are then
reproduced with their original phase relationships intact, so the transient
stays sharp. The cost is a discontinuity in the phase trajectory, which is
inaudible precisely because it happens where the signal is already
discontinuous.

An honest caveat
----------------
Driedger and Müller argue that explicit transient detection is the wrong
approach on two grounds: detection errors propagate directly into the output,
and perceptually transparent transient preservation is hard even when detection
is perfect. Their alternative separates the signal into harmonic and percussive
components and applies the vocoder only to the harmonic part.

Both positions are defensible, and the tension between them is worth reporting
rather than resolving. This module implements the explicit approach because it
is the one that teaches you what a transient does to a phase trajectory.
"""

from __future__ import annotations

import numpy as np

from ..phase import expected_phase_advance, princarg
from ..stft import istft, stft
from .phase_lock import find_peaks, regions_of_influence

__all__ = ["spectral_flux", "detect_onsets", "time_stretch_transient"]


def spectral_flux(X: np.ndarray) -> np.ndarray:
    """Half-wave rectified spectral flux, one value per frame.

    Sums how much each bin *grew* between consecutive frames, ignoring decay.
    Rectification is what makes this an onset detector rather than a change
    detector: a note ending is a large spectral change but not an onset, and
    including negative differences would flag it.

    Returns
    -------
    ndarray, shape ``(n_frames,)``
        Normalised to a maximum of 1. Frame 0 is zero by construction.
    """
    mag = np.abs(np.asarray(X))
    diff = np.diff(mag, axis=1)
    flux = np.sum(np.maximum(diff, 0.0), axis=0)
    flux = np.concatenate([[0.0], flux])

    peak = float(flux.max())
    return flux / peak if peak > 1e-12 else flux


def detect_onsets(
    X: np.ndarray,
    threshold: float = 1.5,
    median_span: int = 9,
    min_separation: int = 3,
) -> np.ndarray:
    """Frame indices where an onset occurs.

    Three conditions must hold together:

    1. **Adaptive threshold** -- flux exceeds ``threshold`` times the local
       median. A fixed global threshold fails badly on material with varying
       loudness: quiet passages get no onsets at all and loud ones get
       everything.
    2. **Local maximum** -- flux peaks rather than merely being high, so a
       single attack yields one detection instead of a smear.
    3. **Rising energy** -- total frame magnitude is not falling.

    Condition 3 is the non-obvious one. Rectified flux is often described as
    ignoring note endings, but that is only true of *gradual* ones. A fade-out
    amplitude-modulates the signal, which creates sidebands, which means some
    bins genuinely gain energy while the note dies; and an abrupt cutoff is a
    discontinuity that splatters energy across every bin at once, producing a
    flux spike often larger than the original attack. Both would be reported
    as onsets by flux alone. Requiring energy to be rising removes them, which
    is what we want here: resetting phase where a sound is *ending* costs
    quality and buys nothing.

    Parameters
    ----------
    threshold
        Multiplier on the local median. Higher is more conservative.
    median_span
        Width of the local median window, in frames.
    min_separation
        Minimum gap between reported onsets, in frames.

    Returns
    -------
    ndarray
        Sorted frame indices. Possibly empty.
    """
    X = np.asarray(X)
    flux = spectral_flux(X)
    n = len(flux)
    if n < 3:
        return np.array([], dtype=int)

    half = max(median_span // 2, 1)
    padded = np.pad(flux, (half, half), mode="edge")
    local_median = np.array(
        [np.median(padded[i : i + 2 * half + 1]) for i in range(n)]
    )

    # A floor keeps near-silent passages from producing onsets purely from noise.
    floor = 0.05 * float(flux.max()) if flux.max() > 0 else 0.0
    candidate = (flux > threshold * local_median) & (flux > floor)

    is_local_max = np.zeros(n, dtype=bool)
    is_local_max[1:-1] = (flux[1:-1] >= flux[:-2]) & (flux[1:-1] >= flux[2:])

    # Condition 3: the sound must be growing, judged over a short horizon
    # rather than between adjacent frames. An instantaneous comparison is not
    # enough: at an abrupt cutoff the broadband splatter briefly *raises* the
    # summed magnitude even though the sound is ending. Comparing the few
    # frames after against the few before captures the trend instead.
    energy = np.abs(X).sum(axis=0)
    span = 3
    rising = np.zeros(n, dtype=bool)
    for i in range(n):
        before = energy[max(i - span, 0) : i]
        after = energy[i : min(i + span, n)]
        if before.size == 0 or after.size == 0:
            continue
        rising[i] = after.mean() > before.mean()

    hits = np.flatnonzero(candidate & is_local_max & rising)

    # Enforce minimum separation, keeping the first of each cluster.
    kept: list[int] = []
    for h in hits:
        if not kept or h - kept[-1] >= min_separation:
            kept.append(int(h))

    return np.array(kept, dtype=int)


def time_stretch_transient(
    x: np.ndarray,
    stretch: float,
    n_fft: int = 2048,
    hop: int | None = None,
    window: str = "hann",
    threshold: float = 1.5,
    lock: bool = True,
    peak_threshold: float = 1e-4,
    return_stft: bool = False,
    return_onsets: bool = False,
):
    """Time stretching with phase locking plus phase reset at onsets.

    Combines the week 3 locked vocoder with transient handling, which is the
    configuration that should perform best on mixed material.

    Parameters
    ----------
    threshold
        Onset detection sensitivity; see `detect_onsets`.
    lock
        Whether to apply identity phase locking between onsets. Turning it off
        isolates the contribution of transient handling alone, which is what
        you want for the ablation table.
    return_onsets
        Also return the detected onset frame indices, for plotting.
    """
    if stretch <= 0:
        raise ValueError(f"stretch must be positive; got {stretch}")

    hop = n_fft // 4 if hop is None else hop
    X = stft(x, n_fft=n_fft, hop=hop, window=window)

    n_bins, n_frames = X.shape
    onsets = set(detect_onsets(X, threshold=threshold).tolist())

    omega = expected_phase_advance(n_fft, hop)
    steps = np.arange(0, max(n_frames - 1, 1), 1.0 / stretch)

    mag = np.abs(X)
    ang = np.angle(X)

    Y = np.zeros((n_bins, len(steps)), dtype=np.complex128)
    phase = ang[:, 0].copy()

    for i, t in enumerate(steps):
        left = int(np.floor(t))
        frac = t - left
        right = min(left + 1, n_frames - 1)

        m = (1.0 - frac) * mag[:, left] + frac * mag[:, right]

        if left in onsets:
            # Discard the accumulated trajectory; the attack's own phase
            # relationships are what make it sound like an attack.
            phase = ang[:, left].copy()
            Y[:, i] = m * np.exp(1j * phase)
        elif lock:
            peaks = find_peaks(m, threshold=peak_threshold)
            owner = regions_of_influence(peaks, n_bins)
            locked = phase[owner] + (ang[:, left] - ang[owner, left])
            Y[:, i] = m * np.exp(1j * locked)
        else:
            Y[:, i] = m * np.exp(1j * phase)

        dphi = ang[:, right] - ang[:, left]
        phase = phase + omega + princarg(dphi - omega)

    target = int(round(len(x) * stretch))
    y = istft(Y, hop=hop, window=window, length=target)

    result: tuple = (y,)
    if return_stft:
        result += (Y,)
    if return_onsets:
        result += (np.array(sorted(onsets), dtype=int),)
    return result[0] if len(result) == 1 else result
