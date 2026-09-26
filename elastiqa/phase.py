"""
Phase arithmetic: principal-argument wrapping and instantaneous frequency.

This is the smallest module in ElastiqA and the one most likely to sink the
project if it is wrong. Every phase-vocoder artefact that sounds like a bug in
the synthesis stage is, in practice, usually a bug here.

The problem
-----------
``np.angle`` returns values in ``(-pi, pi]``. A sinusoid whose frequency sits
between two bin centres advances by more than ``pi`` radians per hop, so the
raw frame-to-frame phase difference is ambiguous: it has been wrapped, and we
cannot tell how many full turns were discarded.

The fix (heterodyned phase increment)
-------------------------------------
We know roughly how much phase a bin *should* advance if its content sat
exactly at the bin centre::

    expected[k] = 2*pi * k * hop / n_fft

The true advance differs from this by less than ``pi`` in magnitude for any
sinusoid within half a bin of centre -- which, with a Hann window, is where
essentially all of a peak's energy lives. So we subtract the expectation,
wrap the small remainder, and add it back. That recovers the unwrapped
advance, and dividing by the hop gives instantaneous frequency in radians
per sample.
"""

from __future__ import annotations

import numpy as np

__all__ = ["princarg", "expected_phase_advance", "instantaneous_frequency"]


def princarg(phi: np.ndarray | float) -> np.ndarray | float:
    """Wrap angles to the principal interval ``[-pi, pi]``.

    Implemented as ``phi - 2*pi*round(phi / (2*pi))`` rather than with a
    modulo, because the modulo form maps exactly ``+pi`` to ``-pi`` and that
    asymmetry shows up as a discontinuity in phase-tracking plots. Both forms
    are equivalent under ``exp(1j * .)``.

    Examples
    --------
    >>> float(princarg(3 * np.pi))
    3.14159...
    >>> float(princarg(0.5))
    0.5
    """
    phi = np.asarray(phi, dtype=np.float64)
    return phi - 2.0 * np.pi * np.round(phi / (2.0 * np.pi))


def expected_phase_advance(n_fft: int, hop: int) -> np.ndarray:
    """Phase advance per hop for a sinusoid sitting at each bin centre.

    Returns an array of length ``n_fft // 2 + 1`` in radians.
    """
    k = np.arange(n_fft // 2 + 1)
    return 2.0 * np.pi * k * hop / n_fft


def instantaneous_frequency(
    X: np.ndarray, hop: int, n_fft: int | None = None
) -> np.ndarray:
    """Estimate per-bin instantaneous frequency from a complex STFT.

    Parameters
    ----------
    X = Complex STFT, shape ``(n_bins, n_frames)``.
    hop = The *analysis* hop used to produce ``X``.
    n_fft = FFT size. Inferred from ``X.shape[0]`` when omitted.

    Returns
    -------
    ndarray, shape ``(n_bins, n_frames)``
        Instantaneous frequency in radians per sample. Column 0 is filled with
        the bin-centre frequencies, since no previous frame exists to
        difference against.

    Notes
    -----
    Multiply by ``sr / (2*pi)`` to convert to Hz. The estimate is only
    meaningful at and around spectral peaks; in low-energy bins the phase is
    essentially noise and the "frequency" it reports is meaningless. Later
    modules exploit this by tracking peaks rather than trusting every bin.
    """
    X = np.asarray(X)
    n_bins = X.shape[0]
    if n_fft is None:
        n_fft = 2 * (n_bins - 1)

    omega = expected_phase_advance(n_fft, hop)[:, None]

    phi = np.angle(X)
    dphi = np.diff(phi, axis=1)
    deviation = princarg(dphi - omega)
    freq = (omega + deviation) / hop

    first = np.broadcast_to(omega / hop, (n_bins, 1))
    return np.concatenate([first, freq], axis=1)
