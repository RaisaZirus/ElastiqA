"""
Audio loading and saving.

Uses ``soundfile`` when available (handles FLAC, OGG, and 24-bit WAV), and
falls back to ``scipy.io.wavfile`` otherwise so the package works in a bare
NumPy/SciPy environment.

All signals are handled internally as float64 in the range [-1, 1], mono.
Stereo files are downmixed on load unless ``mono=False``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

try:  # pragma: no cover - depends on environment
    import soundfile as _sf

    _HAVE_SOUNDFILE = True
except ImportError:  # pragma: no cover
    _sf = None
    _HAVE_SOUNDFILE = False

from scipy.io import wavfile as _wavfile

__all__ = ["load", "save", "to_mono", "normalise", "have_soundfile"]


def have_soundfile() -> bool:
    """Whether the optional ``soundfile`` backend is installed."""
    return _HAVE_SOUNDFILE


def to_mono(x: np.ndarray) -> np.ndarray:
    """Downmix to mono by averaging channels. Accepts (n,) or (n, ch)."""
    x = np.asarray(x, dtype=np.float64)
    return x if x.ndim == 1 else x.mean(axis=1)


def normalise(x: np.ndarray, peak: float = 0.98) -> np.ndarray:
    """Scale so the largest absolute sample equals ``peak``.

    Silent input is returned unchanged rather than amplified into noise.
    """
    x = np.asarray(x, dtype=np.float64)
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return x if m < 1e-12 else x * (peak / m)


def load(path: str | Path, mono: bool = True) -> tuple[np.ndarray, int]:
    """Load an audio file as float64 in [-1, 1].

    Returns
    -------
    (samples, sample_rate)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if _HAVE_SOUNDFILE:
        x, sr = _sf.read(str(path), dtype="float64", always_2d=False)
    else:
        sr, raw = _wavfile.read(str(path))
        x = _int_to_float(raw)

    if mono:
        x = to_mono(x)
    return np.asarray(x, dtype=np.float64), int(sr)


def save(path: str | Path, x: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    """Write a signal to disk, warning if it would clip."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=np.float64)

    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.0:
        warnings.warn(
            f"signal peaks at {peak:.3f} and will clip; "
            "call normalise() before saving",
            RuntimeWarning,
            stacklevel=2,
        )

    if _HAVE_SOUNDFILE:
        _sf.write(str(path), x, sr, subtype=subtype)
    else:
        clipped = np.clip(x, -1.0, 1.0)
        _wavfile.write(str(path), int(sr), (clipped * 32767.0).astype(np.int16))


def _int_to_float(raw: np.ndarray) -> np.ndarray:
    """Convert an integer PCM array from scipy to float64 in [-1, 1]."""
    if raw.dtype.kind == "f":
        return raw.astype(np.float64)
    if raw.dtype == np.int16:
        return raw.astype(np.float64) / 32768.0
    if raw.dtype == np.int32:
        return raw.astype(np.float64) / 2147483648.0
    if raw.dtype == np.uint8:  # WAV stores 8-bit as unsigned
        return (raw.astype(np.float64) - 128.0) / 128.0
    raise ValueError(f"unsupported PCM dtype: {raw.dtype}")
