"""
Short-Time Fourier Transform with exact (perfect-reconstruction) inversion.

This module is the foundation of ElastiqA. Everything else -- time stretching,
pitch shifting, phase locking -- assumes that ``istft(stft(x)) == x`` to
numerical precision. Do not build on top of this until the tests pass.

Design notes
------------
We use *weighted overlap-add* (WOLA): the analysis window ``w`` is applied
before the forward FFT, and the same window is applied again after the inverse
FFT during synthesis. The overlapped frames are then divided by the summed
squared window envelope::

    env[n] = sum_m  w[n - m*H]^2

This guarantees perfect reconstruction for *any* window/hop combination where
``env[n] > 0`` for all n, rather than only for hops that satisfy the strict
COLA condition. It also matches what the phase vocoder needs in later weeks,
because synthesis windowing suppresses the frame-boundary discontinuities that
appear once we start modifying phase.

Frames are *centred*: the signal is zero-padded by ``n_fft // 2`` on each side
so that frame ``m`` is centred on sample ``m * hop`` of the original signal.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "get_window",
    "window_envelope",
    "check_cola",
    "stft",
    "istft",
    "frame_times",
    "bin_frequencies",
]

# Envelope values below this are treated as zero to avoid dividing by ~0 in
# regions covered by too few frames (only ever the padded edges).
_EPS = 1e-12


def get_window(name: str, n_fft: int, periodic: bool = True) -> np.ndarray:
    """Return an analysis window of length ``n_fft``.

    Parameters
    ----------
    name
        One of 
        ``"hann"``, 
        ``"hamming"``, 
        ``"blackman"``, 
        ``"rect"``, or
        ``"sqrt_hann"``.
    n_fft
        Window length in samples.
    periodic
        If True (default) use the periodic (DFT-even) form, which is the
        correct choice for spectral analysis. If False use the symmetric
        form, which is the correct choice for filter design.

    Notes
    -----
    The periodic form of a length-N window is the first N samples of the
    length-(N+1) symmetric window. Using the symmetric form for STFT analysis
    introduces a small but real bias in the overlap-add envelope.
    """
    m = n_fft + 1 if periodic else n_fft
    name = name.lower()

    if name == "hann":
        w = np.hanning(m)
    elif name == "hamming":
        w = np.hamming(m)
    elif name == "blackman":
        w = np.blackman(m)
    elif name == "rect":
        w = np.ones(m)
    elif name == "sqrt_hann":
        w = np.sqrt(np.hanning(m))
    else:
        raise ValueError(f"unknown window: {name!r}")

    return np.asarray(w[:n_fft], dtype=np.float64)


def window_envelope(
    window: np.ndarray, hop: int, n_frames: int, out_len: int
) -> np.ndarray:
    """Sum of squared windows laid down at each frame position.

    This is the normalisation term for weighted overlap-add. For a Hann window
    at 75% overlap it is constant (= 1.5) across the interior of the signal and
    tapers only at the edges.
    """
    n_fft = len(window)
    env = np.zeros(out_len, dtype=np.float64)
    w2 = window ** 2
    for m in range(n_frames):
        start = m * hop
        end = min(start + n_fft, out_len)
        if start >= out_len:
            break
        env[start:end] += w2[: end - start]
    return env


def check_cola(
    window: np.ndarray, hop: int, tol: float = 1e-10
) -> tuple[bool, float, float]:
    """Test whether ``window**2`` satisfies constant overlap-add at ``hop``.

    Returns
    -------
    (ok, value, ripple)
        ``ok`` is True when the interior envelope is constant to within
        ``tol``; ``value`` is the mean interior envelope level; ``ripple`` is
        its peak-to-peak deviation.

    Examples
    --------
    Hann at 75% overlap is the standard phase-vocoder configuration::

        >>> w = get_window("hann", 1024)
        >>> ok, val, ripple = check_cola(w, 256)
        >>> ok
        True
    """
    n_fft = len(window)
    n_frames = 8 * (n_fft // hop) + 8
    total = n_frames * hop + n_fft
    env = window_envelope(window, hop, n_frames, total)

    # Only the interior is meaningful; the first and last n_fft samples are
    # ramping up and down as frames enter and leave.
    interior = env[n_fft : total - n_fft]
    if interior.size == 0:
        raise ValueError("hop too large relative to n_fft to assess COLA")

    ripple = float(interior.max() - interior.min())
    value = float(interior.mean())
    return bool(ripple < tol), value, ripple


def stft(
    x: np.ndarray,
    n_fft: int = 2048,
    hop: int | None = None,
    window: str | np.ndarray = "hann",
    center: bool = True,
) -> np.ndarray:
    """Forward STFT.

    Parameters
    ----------
    x = Real-valued 1-D input signal.
    n_fft = FFT size and window length.
    hop = Analysis hop in samples. Defaults to ``n_fft // 4`` (75% overlap).
    window = Window name or a precomputed array of length ``n_fft``.
    center
        Pad by ``n_fft // 2`` on each side so frame ``m`` is centred on
        sample ``m * hop``.

    Returns
    -------
    ndarray, shape ``(n_fft // 2 + 1, n_frames)``, complex
        One column per frame, one row per positive-frequency bin.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("stft expects a 1-D signal; mix or loop over channels")

    hop = n_fft // 4 if hop is None else hop
    if hop <= 0 or hop > n_fft:
        raise ValueError(f"hop must be in (0, n_fft]; got {hop}")

    w = get_window(window, n_fft) if isinstance(window, str) else np.asarray(window)
    if len(w) != n_fft:
        raise ValueError(f"window length {len(w)} != n_fft {n_fft}")

    pad = n_fft // 2 if center else 0
    x_pad = np.pad(x, (pad, pad), mode="constant")

    # Extend so the final frame is fully in bounds and the tail is covered by
    # as many frames as the interior. Without this the last few samples are
    # normalised by a partial envelope and reconstruction is inexact there.
    n_frames = int(np.ceil(max(len(x_pad) - n_fft, 0) / hop)) + 1
    needed = (n_frames - 1) * hop + n_fft
    if needed > len(x_pad):
        x_pad = np.pad(x_pad, (0, needed - len(x_pad)), mode="constant")

    # Vectorised framing via a strided view -- no data copied until the window
    # multiply below.
    frames = np.lib.stride_tricks.sliding_window_view(x_pad, n_fft)[:: hop]
    frames = frames[:n_frames] * w

    return np.fft.rfft(frames, n=n_fft, axis=-1).T.copy()


def istft(
    X: np.ndarray,
    hop: int | None = None,
    window: str | np.ndarray = "hann",
    center: bool = True,
    length: int | None = None,
) -> np.ndarray:
    """Inverse STFT via weighted overlap-add.

    Parameters
    ----------
    X = Complex STFT of shape ``(n_bins, n_frames)`` as produced by `stft`.
    hop = Synthesis hop. Must match the analysis hop for exact inversion; the
        phase vocoder deliberately makes it differ.
    window = Synthesis window. Must match the analysis window for exact inversion.
    center
        Whether the forward transform was centred (trims the padding).
    length
        Trim or zero-pad the result to exactly this many samples. Pass the
        original signal length to guarantee a shape match.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("istft expects a 2-D array (n_bins, n_frames)")

    n_bins, n_frames = X.shape
    n_fft = 2 * (n_bins - 1)
    hop = n_fft // 4 if hop is None else hop

    w = get_window(window, n_fft) if isinstance(window, str) else np.asarray(window)
    if len(w) != n_fft:
        raise ValueError(f"window length {len(w)} != n_fft {n_fft}")

    out_len = (n_frames - 1) * hop + n_fft
    y = np.zeros(out_len, dtype=np.float64)

    frames = np.fft.irfft(X.T, n=n_fft, axis=-1) * w
    for m in range(n_frames):
        start = m * hop
        y[start : start + n_fft] += frames[m]

    env = window_envelope(w, hop, n_frames, out_len)
    nonzero = env > _EPS
    y[nonzero] /= env[nonzero]
    y[~nonzero] = 0.0

    if center:
        pad = n_fft // 2
        y = y[pad:]

    if length is not None:
        if len(y) < length:
            y = np.pad(y, (0, length - len(y)), mode="constant")
        y = y[:length]

    return y


def frame_times(n_frames: int, hop: int, sr: int) -> np.ndarray:
    """Centre time in seconds of each STFT frame (assumes ``center=True``)."""
    return np.arange(n_frames) * hop / sr


def bin_frequencies(n_fft: int, sr: int) -> np.ndarray:
    """Centre frequency in Hz of each rFFT bin."""
    return np.fft.rfftfreq(n_fft, d=1.0 / sr)
