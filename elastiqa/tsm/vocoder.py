"""
The phase vocoder.

Time-scale modification in the STFT domain. The idea is simple: read analysis
frames at one rate, write synthesis frames at another. The difficulty is
entirely in what happens to phase along the way.

Formulation
-----------
Rather than using different analysis and synthesis hops (which restricts the
stretch factor to ratios of integers), we keep a single hop and step through
the analysis frames at *fractional* positions::

    t = 0, 1/s, 2/s, 3/s, ...        for stretch factor s

Magnitudes are linearly interpolated between neighbouring frames. Phase is not
interpolated -- it is *accumulated* from instantaneous frequency estimates,
which is the entire point of the algorithm.

For each output frame we advance the running phase by the true phase increment
measured between the two bracketing analysis frames::

    dphi     = angle(X[:, i+1]) - angle(X[:, i])
    deviation = princarg(dphi - omega)
    phase    += omega + deviation

With ``s = 1`` this reduces to ordinary phase accumulation and the output is
perceptually identical to the input.

Phase modes
-----------
``mode="vocoder"``
    The correct algorithm described above.

``mode="passthrough"``
    Deliberately broken: reuse the analysis phase verbatim. Successive frames
    no longer line up at the waveform level, so overlap-add produces partial
    cancellation. This is the classic "phasiness" artefact, and having it as a
    switchable mode means you can demonstrate the problem rather than merely
    describe it. Keep it in the repository; it is a deliverable.

``mode="zero"`` / ``mode="random"``
    Set phase to zero or randomise it every frame. Both destroy the waveform
    while preserving the magnitude spectrogram, which is a compact way of
    showing exactly what information phase carries. They also happen to be
    robot and whisper effects.
"""

from __future__ import annotations

import numpy as np

from ..phase import expected_phase_advance, princarg
from ..stft import istft, stft

__all__ = ["phase_vocoder_stft", "time_stretch", "PHASE_MODES"]

PHASE_MODES = ("vocoder", "passthrough", "zero", "random")


def phase_vocoder_stft(
    X: np.ndarray,
    stretch: float,
    hop: int,
    n_fft: int | None = None,
    mode: str = "vocoder",
    seed: int | None = None,
) -> np.ndarray:
    """Time-scale a complex STFT by ``stretch``, operating on phase only.

    Parameters
    ----------
    X
        Complex STFT, shape ``(n_bins, n_frames)``.
    stretch
        Output duration multiplier. ``2.0`` is twice as long (slower);
        ``0.5`` is half as long (faster). Pitch is unchanged.
    hop
        The hop used for both analysis and synthesis.
    n_fft
        FFT size; inferred from ``X`` when omitted.
    mode
        One of ``PHASE_MODES``. See the module docstring.
    seed
        Seed for ``mode="random"``, so figures are reproducible.

    Returns
    -------
    ndarray
        Modified STFT with approximately ``stretch * n_frames`` columns.
    """
    if mode not in PHASE_MODES:
        raise ValueError(f"mode must be one of {PHASE_MODES}; got {mode!r}")
    if stretch <= 0:
        raise ValueError(f"stretch must be positive; got {stretch}")

    X = np.asarray(X)
    n_bins, n_frames = X.shape
    if n_fft is None:
        n_fft = 2 * (n_bins - 1)
    if n_frames < 2:
        return X.copy()

    omega = expected_phase_advance(n_fft, hop)

    # Fractional read positions into the analysis frames.
    steps = np.arange(0, n_frames - 1, 1.0 / stretch)
    Y = np.zeros((n_bins, len(steps)), dtype=np.complex128)

    mag = np.abs(X)
    ang = np.angle(X)

    rng = np.random.default_rng(seed)
    phase = ang[:, 0].copy()

    for i, t in enumerate(steps):
        left = int(np.floor(t))
        frac = t - left
        right = min(left + 1, n_frames - 1)

        # Magnitude: linear interpolation between the bracketing frames.
        m = (1.0 - frac) * mag[:, left] + frac * mag[:, right]

        if mode == "vocoder":
            Y[:, i] = m * np.exp(1j * phase)
        elif mode == "passthrough":
            # The bug, on purpose: analysis phase used as-is.
            Y[:, i] = m * np.exp(1j * ang[:, left])
        elif mode == "zero":
            Y[:, i] = m
        else:  # random
            Y[:, i] = m * np.exp(1j * rng.uniform(-np.pi, np.pi, n_bins))

        # Advance the running phase by the measured true increment. Done
        # unconditionally so that switching modes changes only the output,
        # never the trajectory being compared against.
        dphi = ang[:, right] - ang[:, left]
        deviation = princarg(dphi - omega)
        phase = phase + omega + deviation

    return Y


def time_stretch(
    x: np.ndarray,
    stretch: float,
    n_fft: int = 2048,
    hop: int | None = None,
    window: str = "hann",
    mode: str = "vocoder",
    seed: int | None = None,
    return_stft: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Change the duration of a signal without changing its pitch.

    Parameters
    ----------
    x
        Real 1-D signal.
    stretch
        Duration multiplier. ``1.5`` makes it 50% longer.
    n_fft, hop, window
        STFT parameters. The default hop of ``n_fft // 4`` (75% overlap) is
        the standard choice; see ``tests/test_cola.py`` for why 50% is not
        good enough once phase is modified.
    mode
        Phase handling; see `phase_vocoder_stft`.
    return_stft
        Also return the modified STFT. Required for the consistency measure,
        which must be computed on the STFT the algorithm *asked for*, not on
        the STFT of the signal that came out -- the latter is trivially
        consistent, because it is the transform of a real signal.

    Examples
    --------
        slow = time_stretch(x, 1.5)                      # correct
        bad  = time_stretch(x, 1.5, mode="passthrough")  # audibly phasey

        y, Y = time_stretch(x, 1.5, return_stft=True)
        d_m = consistency(Y, hop=512)
    """
    hop = n_fft // 4 if hop is None else hop

    X = stft(x, n_fft=n_fft, hop=hop, window=window)
    Y = phase_vocoder_stft(X, stretch, hop=hop, n_fft=n_fft, mode=mode, seed=seed)

    target = int(round(len(x) * stretch))
    y = istft(Y, hop=hop, window=window, length=target)

    return (y, Y) if return_stft else y
