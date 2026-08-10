"""
Fundamental frequency estimation (YIN).

Auto-tune needs to know what note is being sung before it can correct it, and
that turns out to be the hard part. Picking the largest spectral peak fails
constantly: resonance routinely makes the second harmonic louder than the
fundamental, so a naive detector reports an octave too high. This is the
classic *octave error*, and it is why a real algorithm is needed here.

YIN (de Cheveigne & Kawahara 2002) works in the time domain and asks a
different question: for what lag does the signal most resemble a shifted copy
of itself? Four refinements over plain autocorrelation, each fixing a specific
failure:

1. **Squared difference instead of correlation.** Autocorrelation is biased
   towards short lags because it is maximised by loud regions rather than
   similar ones; a difference function is not.
2. **Cumulative mean normalisation.** The difference function is trivially zero
   at lag 0 and small at short lags. Dividing by the running mean removes that
   bias and makes a fixed threshold meaningful.
3. **Absolute threshold.** Take the *first* dip below the threshold rather than
   the deepest. The deepest dip is often at twice the true period, because a
   signal also resembles itself two periods later -- this single choice is what
   removes most octave errors.
4. **Parabolic interpolation.** The true period is rarely an integer number of
   samples; fitting a parabola to the dip recovers sub-sample precision.
"""

from __future__ import annotations

import numpy as np

__all__ = ["yin_frame", "yin_track", "hz_to_midi", "midi_to_hz", "SCALES"]

_EPS = 1e-12

#: Scale degrees as semitone offsets from the tonic.
SCALES: dict[str, tuple[int, ...]] = {
    "chromatic": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "pentatonic": (0, 2, 4, 7, 9),
    "minor_pentatonic": (0, 3, 5, 7, 10),
    "blues": (0, 3, 5, 6, 7, 10),
    "whole_tone": (0, 2, 4, 6, 8, 10),
}


def hz_to_midi(hz: np.ndarray | float) -> np.ndarray | float:
    """Frequency to MIDI note number. A4 = 440 Hz = note 69."""
    hz = np.asarray(hz, dtype=np.float64)
    return 69.0 + 12.0 * np.log2(np.maximum(hz, _EPS) / 440.0)


def midi_to_hz(midi: np.ndarray | float) -> np.ndarray | float:
    """MIDI note number to frequency."""
    return 440.0 * 2.0 ** ((np.asarray(midi, dtype=np.float64) - 69.0) / 12.0)


def yin_frame(
    frame: np.ndarray,
    sr: int,
    fmin: float = 70.0,
    fmax: float = 1000.0,
    threshold: float = 0.15,
) -> tuple[float, float]:
    """Estimate f0 for one frame.

    Returns
    -------
    (f0_hz, aperiodicity)
        ``f0_hz`` is 0.0 when no periodicity is found. ``aperiodicity`` is the
        normalised difference at the chosen lag: near 0 means strongly
        periodic (clearly pitched), near 1 means noise-like. Use it as a
        voicing gate -- correcting the pitch of an unvoiced consonant produces
        artefacts, so auto-tune should leave those frames alone.
    """
    frame = np.asarray(frame, dtype=np.float64)
    frame = frame - frame.mean()
    n = len(frame)

    tau_min = max(int(sr / fmax), 1)
    tau_max = min(int(sr / fmin), n // 2)
    if tau_max <= tau_min:
        return 0.0, 1.0

    # Step 1: squared difference function, computed via autocorrelation so the
    # cost is one FFT rather than a double loop.
    power = float(np.dot(frame, frame))
    if power < _EPS:
        return 0.0, 1.0

    size = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(frame, size)
    acf = np.fft.irfft(spec * np.conj(spec), size)[: tau_max + 1]

    cumsum = np.concatenate([[0.0], np.cumsum(frame ** 2)])
    lags = np.arange(tau_max + 1)
    energy_head = cumsum[n - lags] - cumsum[0]
    energy_tail = cumsum[n] - cumsum[lags]
    diff = energy_head + energy_tail - 2.0 * acf
    diff = np.maximum(diff, 0.0)

    # Step 2: cumulative mean normalisation.
    cmnd = np.ones_like(diff)
    running = np.cumsum(diff[1:])
    nonzero = running > _EPS
    cmnd[1:][nonzero] = (
        diff[1:][nonzero] * np.arange(1, tau_max + 1)[nonzero] / running[nonzero]
    )

    # Step 3: first dip below threshold, not the deepest.
    search = cmnd[tau_min : tau_max + 1]
    below = np.flatnonzero(search < threshold)
    if below.size:
        tau = int(below[0]) + tau_min
        # Walk down to the local minimum of this dip.
        while tau + 1 <= tau_max and cmnd[tau + 1] < cmnd[tau]:
            tau += 1
    else:
        tau = int(np.argmin(search)) + tau_min

    aperiodicity = float(cmnd[tau])

    # Step 4: parabolic interpolation around the dip.
    if 0 < tau < tau_max:
        a, b, c = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
        denom = a - 2.0 * b + c
        if abs(denom) > _EPS:
            tau = tau + 0.5 * (a - c) / denom

    if tau <= 0:
        return 0.0, 1.0
    return float(sr / tau), aperiodicity


def yin_track(
    x: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop: int = 512,
    fmin: float = 70.0,
    fmax: float = 1000.0,
    threshold: float = 0.15,
    voiced_max_aperiodicity: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate f0 over time.

    Parameters
    ----------
    frame_length
        Analysis window in samples. Must span at least two periods of the
        lowest frequency you expect: at 22.05 kHz, tracking down to 70 Hz needs
        at least 630 samples, so 2048 is comfortable.
    voiced_max_aperiodicity
        Frames above this are marked unvoiced and their f0 set to 0.

    Returns
    -------
    (f0, voiced)
        Both of length ``n_frames``, frame-aligned with a centred STFT using
        the same hop.
    """
    x = np.asarray(x, dtype=np.float64)
    pad = frame_length // 2
    xp = np.pad(x, (pad, pad), mode="constant")

    n_frames = 1 + max(len(xp) - frame_length, 0) // hop
    f0 = np.zeros(n_frames)
    aper = np.ones(n_frames)

    for i in range(n_frames):
        frame = xp[i * hop : i * hop + frame_length]
        if len(frame) < frame_length:
            break
        f0[i], aper[i] = yin_frame(
            frame, sr, fmin=fmin, fmax=fmax, threshold=threshold
        )

    voiced = (aper <= voiced_max_aperiodicity) & (f0 > 0)
    f0 = np.where(voiced, f0, 0.0)
    return f0, voiced
