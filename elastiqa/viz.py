"""
Plotting helpers.

These exist so that every notebook and every figure in the report is generated
by the same code path. If a spectrogram in the paper looks different from a
spectrogram in the app, that is a bug, not a styling choice.
"""

from __future__ import annotations

import numpy as np

from .stft import bin_frequencies, stft

__all__ = ["plot_waveform", "plot_spectrogram", "compare_spectrograms", "plot_envelope"]

_DB_FLOOR = -80.0


def _db(mag: np.ndarray, floor: float = _DB_FLOOR) -> np.ndarray:
    """Magnitude to dB, normalised so the loudest point is 0 dB."""
    ref = float(mag.max()) if mag.size else 1.0
    ref = max(ref, 1e-12)
    with np.errstate(divide="ignore"):
        out = 20.0 * np.log10(np.maximum(mag, 1e-12) / ref)
    return np.maximum(out, floor)


def plot_waveform(x: np.ndarray, sr: int, ax=None, title: str = "", **kwargs):
    """Plot a time-domain waveform."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2.5))

    t = np.arange(len(x)) / sr
    ax.plot(t, x, linewidth=0.6, **kwargs)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    ax.set_xlim(0, t[-1] if len(t) else 1)
    if title:
        ax.set_title(title)
    return ax


def plot_spectrogram(
    x: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop: int | None = None,
    window: str = "hann",
    ax=None,
    title: str = "",
    fmax: float | None = None,
    cmap: str = "magma",
):
    """Plot a log-magnitude spectrogram in dB.

    Returns the axis so callers can add annotations.
    """
    import matplotlib.pyplot as plt

    hop = n_fft // 4 if hop is None else hop
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    X = stft(x, n_fft=n_fft, hop=hop, window=window)
    db = _db(np.abs(X))

    freqs = bin_frequencies(n_fft, sr)
    duration = len(x) / sr
    extent = (0.0, duration, freqs[0], freqs[-1])

    im = ax.imshow(
        db, origin="lower", aspect="auto", extent=extent, cmap=cmap, vmin=_DB_FLOOR
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    if fmax is not None:
        ax.set_ylim(0, fmax)
    if title:
        ax.set_title(title)

    cbar = ax.figure.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("dB")
    return ax


def compare_spectrograms(
    signals: dict[str, np.ndarray],
    sr: int,
    n_fft: int = 2048,
    hop: int | None = None,
    fmax: float | None = 5000.0,
    figsize: tuple[float, float] | None = None,
):
    """Stack several spectrograms vertically for A/B comparison.

    Parameters
    ----------
    signals
        Mapping of label -> signal. Order is preserved.

    Examples
    --------
        compare_spectrograms(
            {"original": x, "naive": y_naive, "vocoder": y_pv}, sr
        )
    """
    import matplotlib.pyplot as plt

    n = len(signals)
    if n == 0:
        raise ValueError("nothing to plot")

    figsize = figsize or (10, 3.0 * n)
    fig, axes = plt.subplots(n, 1, figsize=figsize, squeeze=False)

    for ax, (label, sig) in zip(axes[:, 0], signals.items()):
        plot_spectrogram(sig, sr, n_fft=n_fft, hop=hop, ax=ax, title=label, fmax=fmax)

    fig.tight_layout()
    return fig, axes[:, 0]


def plot_envelope(window: np.ndarray, hop: int, n_frames: int = 12, ax=None):
    """Show individual squared windows and their overlap-add sum.

    This is the figure that makes the COLA condition obvious: the individual
    humps are not constant, but their sum is (in the interior).
    """
    import matplotlib.pyplot as plt

    from .stft import window_envelope

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))

    n_fft = len(window)
    total = (n_frames - 1) * hop + n_fft
    w2 = window ** 2

    for m in range(n_frames):
        idx = np.arange(n_fft) + m * hop
        ax.plot(idx, w2, color="0.75", linewidth=0.8)

    env = window_envelope(window, hop, n_frames, total)
    ax.plot(env, color="crimson", linewidth=1.8, label="sum of $w^2$")
    ax.axvspan(0, n_fft, color="0.9", zorder=0)
    ax.axvspan(total - n_fft, total, color="0.9", zorder=0)
    ax.set_xlabel("sample")
    ax.set_ylabel("$w^2$")
    ax.set_title(f"overlap-add envelope (n_fft={n_fft}, hop={hop})")
    ax.legend()
    return ax
