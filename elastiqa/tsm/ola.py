"""
Overlap-add: the simplest possible time-scale modification.

Cut the signal into overlapping windowed frames, then lay them back down at a
different spacing. No frequency analysis, no phase arithmetic -- just moving
chunks of waveform around.

It works better than it has any right to on noisy or percussive material,
because those signals have no long-term phase structure to destroy. On tonal
material it fails audibly: adjacent frames are laid down at arbitrary relative
phase, so harmonics partially cancel and the result warbles at the frame rate.

This is the honest baseline for the vocoder to beat. It is also the ancestor
of WSOLA (week 3), which fixes the phase mismatch by *searching* for a frame
offset that maximises waveform similarity rather than by reasoning about
frequency.
"""

from __future__ import annotations

import numpy as np

from ..stft import get_window, window_envelope

__all__ = ["ola"]

_EPS = 1e-12


def ola(
    x: np.ndarray,
    stretch: float,
    n_fft: int = 2048,
    hop: int | None = None,
    window: str = "hann",
) -> np.ndarray:
    """Time-scale by overlap-add with no phase correction.

    Parameters
    ----------
    x
        Real 1-D signal.
    stretch
        Duration multiplier.
    n_fft
        Frame length in samples.
    hop
        Synthesis hop. The analysis hop is derived as ``hop / stretch``.
    window
        Window applied on analysis and synthesis.

    Notes
    -----
    The analysis hop is generally fractional, so frames are read at rounded
    positions. That rounding is itself a small source of jitter, which is part
    of why OLA sounds the way it does.
    """
    if stretch <= 0:
        raise ValueError(f"stretch must be positive; got {stretch}")

    x = np.asarray(x, dtype=np.float64)
    hop_s = n_fft // 4 if hop is None else hop
    hop_a = hop_s / stretch

    w = get_window(window, n_fft)

    n_frames = max(int(np.floor((len(x) - n_fft) / hop_a)) + 1, 1)
    out_len = (n_frames - 1) * hop_s + n_fft
    y = np.zeros(out_len, dtype=np.float64)

    for m in range(n_frames):
        start = int(round(m * hop_a))
        frame = x[start : start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        y[m * hop_s : m * hop_s + n_fft] += frame * w * w

    env = window_envelope(w, hop_s, n_frames, out_len)
    nonzero = env > _EPS
    y[nonzero] /= env[nonzero]

    target = int(round(len(x) * stretch))
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    return y[:target]
