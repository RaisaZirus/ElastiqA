"""
Objective quality metrics for time-scale modification.

The central difficulty of evaluating TSM is that input and output have
different lengths, so ordinary reference-based audio quality measures cannot
be applied at all. Only a handful of measures work in this setting, and the
literature is candid that they are high-level indicators of phasiness rather
than true perceptual predictors.

That is not a reason to skip them. Having *any* number turns "it sounds worse"
into something you can plot, bisect, and regression-test. Pair these with a
listening test rather than trusting either alone.

Measures implemented
--------------------
``consistency`` (D_M, Laroche & Dolson 1999)
    The important one, and the only one here that needs no length matching.

``ser``
    Signal-to-error ratio; only meaningful at stretch = 1.

``spectral_convergence`` and ``log_spectral_distance``
    Standard magnitude-spectrogram distances, applied after interpolating the
    reference to the test length.
"""

from __future__ import annotations

import numpy as np

from .stft import stft

__all__ = [
    "consistency",
    "ser",
    "spectral_convergence",
    "log_spectral_distance",
    "amplitude_warble",
    "crest_factor",
    "evaluate",
]

_EPS = 1e-12


def amplitude_warble(x: np.ndarray, trim: int = 2000) -> float:
    """Relative fluctuation of the amplitude envelope. Lower is better.

    Computed as ``std(envelope) / mean(envelope)`` using the analytic-signal
    envelope from the Hilbert transform.

    Why this metric earns its place
    -------------------------------
    It measures the artefact directly. When overlapping frames disagree about
    phase, they partially cancel where they overlap, and the output amplitude
    dips and revives at the frame rate. That warble is what listeners describe
    as phasiness or warbling, so a number that tracks it corresponds to
    something you can actually hear -- unlike a spectral distance, which
    confounds this artefact with the (unavoidable) fact that a stretched
    signal has a different spectrogram from the original.

    On a steady sine the ideal value is 0, which makes it a clean discriminator
    between algorithms: any deviation is entirely the algorithm's doing.

    IMPORTANT: only valid for narrowband signals
    --------------------------------------------
    This measure is meaningful on a **pure or near-pure tone only**. A signal
    with many harmonics has a peaky waveform and therefore an intrinsically
    fluctuating analytic envelope: measured on a 25-harmonic stack, every
    algorithm scores around 0.5 regardless of quality, because the metric is
    dominated by the signal's own crest factor rather than by any artefact.

    Use a single sinusoid as the probe signal. Reporting warble on broadband
    material is not conservative -- it is meaningless, and it will hide real
    differences between methods behind a constant floor.

    Parameters
    ----------
    x
        Signal to measure. Must be narrowband for the result to mean anything.
    trim
        Samples discarded from each end, to exclude fade-in and fade-out.
    """
    from scipy.signal import hilbert

    x = np.asarray(x, dtype=np.float64)
    if len(x) <= 2 * trim + 16:
        trim = 0

    envelope = np.abs(hilbert(x))
    if trim:
        envelope = envelope[trim:-trim]

    mean = float(np.mean(envelope))
    if mean < _EPS:
        return 0.0
    return float(np.std(envelope) / mean)


def consistency(
    Y: np.ndarray, hop: int, window: str = "hann", n_fft: int | None = None
) -> float:
    """Laroche & Dolson's consistency measure D_M. Lower is better.

    A phase vocoder produces a *modified* STFT that in general is not the STFT
    of any real signal -- the frames disagree about what the waveform should be
    in their overlapping regions. Overlap-add resolves the disagreement by
    averaging, and the averaging is what you hear as phasiness.

    D_M quantifies the disagreement directly: resynthesise, re-analyse, and
    measure how far the resulting magnitudes drifted from the ones you asked
    for::

        D_M = sum |  |STFT(istft(Y))| - |Y|  |^2  /  sum |Y|^2

    Zero means the modified STFT was perfectly consistent. Values grow as the
    algorithm's phase handling degrades, which makes this the natural metric
    for comparing vocoder variants.

    Notes
    -----
    Requires no reference signal and no length alignment, which is exactly why
    it survives in a field where most quality measures do not apply.
    """
    from .stft import istft  # local import to avoid a cycle at module load

    Y = np.asarray(Y)
    if n_fft is None:
        n_fft = 2 * (Y.shape[0] - 1)

    y = istft(Y, hop=hop, window=window)
    Z = stft(y, n_fft=n_fft, hop=hop, window=window)

    n = min(Y.shape[1], Z.shape[1])
    diff = np.abs(Z[:, :n]) - np.abs(Y[:, :n])

    denom = float(np.sum(np.abs(Y[:, :n]) ** 2))
    if denom < _EPS:
        return 0.0
    return float(np.sum(diff ** 2) / denom)


def ser(reference: np.ndarray, test: np.ndarray) -> float:
    """Signal-to-error ratio in dB. Higher is better.

    Only meaningful when the two signals should be sample-aligned, i.e. at
    stretch = 1. Use it as an identity check, not as a quality measure.
    """
    reference = np.asarray(reference, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)

    n = min(len(reference), len(test))
    ref, tst = reference[:n], test[:n]

    signal = float(np.sum(ref ** 2))
    error = float(np.sum((ref - tst) ** 2))

    if error < _EPS:
        return float("inf")
    if signal < _EPS:
        return float("-inf")
    return float(10.0 * np.log10(signal / error))


def _aligned_magnitudes(
    reference: np.ndarray, test: np.ndarray, n_fft: int, hop: int
) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude spectrograms with the reference resampled to the test length.

    Interpolating the reference magnitude spectrum to the length of the test
    spectrum is the alignment strategy reported to work best in the TSM
    evaluation literature.
    """
    R = np.abs(stft(reference, n_fft=n_fft, hop=hop))
    T = np.abs(stft(test, n_fft=n_fft, hop=hop))

    if R.shape[1] != T.shape[1]:
        src = np.linspace(0.0, 1.0, R.shape[1])
        dst = np.linspace(0.0, 1.0, T.shape[1])
        R = np.stack([np.interp(dst, src, row) for row in R])

    return R, T


def spectral_convergence(
    reference: np.ndarray, test: np.ndarray, n_fft: int = 2048, hop: int = 512
) -> float:
    """Relative Frobenius distance between magnitude spectrograms. Lower is better."""
    R, T = _aligned_magnitudes(reference, test, n_fft, hop)
    denom = float(np.linalg.norm(R))
    if denom < _EPS:
        return 0.0
    return float(np.linalg.norm(T - R) / denom)


def log_spectral_distance(
    reference: np.ndarray, test: np.ndarray, n_fft: int = 2048, hop: int = 512
) -> float:
    """RMS difference of log-magnitude spectra, in dB. Lower is better.

    Closer to perception than a linear distance, because loudness is roughly
    logarithmic in amplitude.
    """
    R, T = _aligned_magnitudes(reference, test, n_fft, hop)
    log_r = 20.0 * np.log10(np.maximum(R, _EPS))
    log_t = 20.0 * np.log10(np.maximum(T, _EPS))
    return float(np.sqrt(np.mean((log_t - log_r) ** 2)))


def crest_factor(x: np.ndarray) -> float:
    """Peak-to-RMS ratio of the amplitude envelope. Higher means sharper attacks.

    Smearing a transient spreads its energy over a wider span of time. Total
    energy is roughly conserved, so the peak comes down while the RMS stays
    put -- which is exactly what this ratio detects.

    Read it as a *relative* measure against the unprocessed original: a value
    of 11.0 means nothing on its own, but 11.0 where the original scored 11.4
    means the attacks survived, and 9.4 means they did not.

    Complements `amplitude_warble`, which is only valid on narrowband signals.
    This one is designed for the opposite case: broadband, percussive material.
    """
    from scipy.signal import hilbert

    x = np.asarray(x, dtype=np.float64)
    if x.size < 16:
        return 0.0

    envelope = np.abs(hilbert(x))
    rms = float(np.sqrt(np.mean(envelope ** 2)))
    if rms < _EPS:
        return 0.0
    return float(envelope.max() / rms)


def evaluate(
    reference: np.ndarray,
    test: np.ndarray,
    stretch: float,
    modified_stft: np.ndarray | None = None,
    n_fft: int = 2048,
    hop: int = 512,
) -> dict[str, float]:
    """Run the full metric suite and return a dict ready for a results table.

    Parameters
    ----------
    reference
        The original signal.
    test
        The time-scaled output.
    stretch
        The factor that was applied, recorded in the output row.
    modified_stft
        The STFT the algorithm produced *before* resynthesis, as returned by
        ``time_stretch(..., return_stft=True)``. Required for the consistency
        measure. Passing ``stft(test)`` instead would always give ~0, because
        the transform of a real signal is consistent by construction -- an easy
        mistake that silently makes every algorithm look perfect.

    Notes
    -----
    ``ser_db`` is included only at stretch = 1, where sample alignment makes
    it meaningful.
    """
    out: dict[str, float] = {
        "stretch": float(stretch),
        "spectral_convergence": spectral_convergence(reference, test, n_fft, hop),
        "log_spectral_distance": log_spectral_distance(reference, test, n_fft, hop),
        "amplitude_warble": amplitude_warble(test),
        "crest_factor": crest_factor(test),
    }
    if modified_stft is not None:
        out["consistency"] = consistency(modified_stft, hop=hop, n_fft=n_fft)
    if np.isclose(stretch, 1.0):
        out["ser_db"] = ser(reference, test)
    return out
