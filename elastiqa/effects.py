from __future__ import annotations

import numpy as np

from .pitch.f0 import SCALES, hz_to_midi, midi_to_hz, yin_track
from .pitch.spectral import spectral_pitch_shift
from .stft import istft, stft

__all__ = ["autotune", "harmonize", "robotize", "whisperize", "quantize_to_scale"]

_EPS = 1e-10


def quantize_to_scale(
    f0_hz: np.ndarray,
    scale: str = "chromatic",
    tonic_midi: int = 0,
) -> np.ndarray:
    """Snap each frequency to the nearest note of a scale.

    Parameters
    ----------
    f0_hz
        Detected frequencies. Zeros (unvoiced) pass through unchanged.
    scale
        Key from `SCALES`.
    tonic_midi
        MIDI number of the tonic, modulo 12. C = 0, D = 2, and so on.

    Returns
    -------
    ndarray
        Target frequencies, same shape.
    """
    if scale not in SCALES:
        raise ValueError(f"unknown scale {scale!r}; choose from {sorted(SCALES)}")

    degrees = np.array(SCALES[scale])
    f0_hz = np.asarray(f0_hz, dtype=np.float64)

    voiced = f0_hz > 0
    out = f0_hz.copy()
    if not np.any(voiced):
        return out

    midi = hz_to_midi(f0_hz[voiced])

    # Allowed pitch classes across every octave in range.
    octaves = np.arange(0, 11) * 12
    allowed = np.sort((degrees[None, :] + octaves[:, None]).ravel() + (tonic_midi % 12))

    idx = np.abs(midi[:, None] - allowed[None, :]).argmin(axis=1)
    out[voiced] = midi_to_hz(allowed[idx])
    return out


def autotune(
    x: np.ndarray,
    sr: int,
    scale: str = "chromatic",
    tonic_midi: int = 0,
    strength: float = 1.0,
    n_fft: int = 2048,
    hop: int = 512,
    fmin: float = 70.0,
    fmax: float = 1000.0,
    preserve_formants: bool = True,
    return_tracks: bool = False,
):
    """Snap the sung pitch to the nearest note of a scale.

    Parameters
    ----------
    strength
        0.0 leaves the performance alone; 1.0 snaps hard to the grid. The
        famous effect is ``strength=1.0`` on a chromatic or pentatonic scale,
        where the pitch jumps between notes instead of gliding. Intermediate
        values are what a producer would actually use.
    preserve_formants
        Keep the singer's vocal tract fixed while the pitch moves. Without it,
        correcting a flat note also makes the singer sound briefly smaller.
    return_tracks
        Also return ``(detected_hz, target_hz, voiced)`` for plotting.

    Notes
    -----
    Unvoiced frames get a ratio of exactly 1.0. Pitch-correcting a consonant
    is meaningless and sounds bad, so the voicing gate matters as much as the
    detector.
    """
    x = np.asarray(x, dtype=np.float64)

    detected, voiced = yin_track(
        x, sr, frame_length=n_fft, hop=hop, fmin=fmin, fmax=fmax
    )
    target = quantize_to_scale(detected, scale=scale, tonic_midi=tonic_midi)

    ratios = np.ones_like(detected)
    ok = voiced & (detected > 0)
    ratios[ok] = target[ok] / detected[ok]

    # Interpolate between no correction and full correction.
    ratios = ratios ** float(strength)

    y = spectral_pitch_shift(
        x, ratios, n_fft=n_fft, hop=hop, preserve_formants=preserve_formants
    )

    if return_tracks:
        return y, detected, target, voiced
    return y


def harmonize(
    x: np.ndarray,
    intervals: tuple[float, ...] = (4.0, 7.0),
    dry: float = 1.0,
    wet: float = 0.7,
    n_fft: int = 2048,
    hop: int = 512,
    preserve_formants: bool = True,
) -> np.ndarray:
    """Add pitch-shifted copies at musical intervals.

    Parameters
    ----------
    intervals
        Semitone offsets. The default ``(4, 7)`` is a major triad above the
        input: root, major third, perfect fifth.
    dry, wet
        Level of the original and of each harmony voice.

    Notes
    -----
    Every voice is a real pitch shift of the same performance, so they share
    its timing and phrasing exactly. That is what makes the result sound like
    one singer multitracked rather than like a chorus effect.
    """
    x = np.asarray(x, dtype=np.float64)
    out = dry * x.copy()

    for semitones in intervals:
        ratio = 2.0 ** (semitones / 12.0)
        voice = spectral_pitch_shift(
            x, ratio, n_fft=n_fft, hop=hop, preserve_formants=preserve_formants
        )
        out[: len(voice)] += wet * voice[: len(out)]

    peak = float(np.max(np.abs(out))) if out.size else 0.0
    return out / peak * 0.95 if peak > 1.0 else out


def robotize(
    x: np.ndarray,
    n_fft: int = 2048,
    hop: int = 512,
    window: str = "hann",
) -> np.ndarray:
    """Zero the phase of every frame.

    Each frame becomes a zero-phase pulse centred on its own position, so the
    output acquires a strong periodicity at the frame rate ``sr / hop``.
    Measured on a harmonic test signal, autocorrelation at lag ``hop`` rises
    from 0.43 to 0.99.

    What this does *not* do
    -----------------------
    It does not replace the original pitch with the frame rate. That claim is
    common and, for a harmonic input, wrong: the harmonic comb lives in the
    *magnitude* spectrum, which zeroing phase leaves in place. A 220 Hz input
    still measures 220 Hz afterwards. What you hear is the original pitch with
    a frame-rate buzz laid over it -- and the buzz dominates perceptually for
    speech, which is why the effect reads as "robotic".

    Choose ``hop`` deliberately, since it sets the buzz frequency: 512 samples
    at 22.05 kHz gives 43 Hz, while 256 gives 86 Hz.

    Overlap-add is no longer coherent once phase is flattened, so the output's
    magnitude spectrum is *close to* but not identical to the input's --
    correlation around 0.8 rather than 1.0.
    """
    X = stft(np.asarray(x, dtype=np.float64), n_fft=n_fft, hop=hop, window=window)
    return istft(np.abs(X), hop=hop, window=window, length=len(x))


def whisperize(
    x: np.ndarray,
    n_fft: int = 2048,
    hop: int = 512,
    window: str = "hann",
    seed: int | None = None,
) -> np.ndarray:
    """Randomise the phase of every frame.

    The complement of `robotize`. Destroying phase coherence between
    overlapping frames breaks up the periodic waveform structure, pushing the
    signal towards noise excitation -- which is what whispering is: the vocal
    tract filter driven by turbulence instead of by the vocal folds.

    Speech stays intelligible because intelligibility lives mostly in the
    formants, and the formants are in the magnitude, which is untouched.

    How complete is the effect?
    ---------------------------
    Partial, and worth being precise about. On a strongly harmonic synthetic
    signal, mean YIN aperiodicity rises from 0.001 to 0.034 -- a large relative
    change, but still well under the 0.5 voicing threshold, because the
    harmonic comb survives in the magnitude spectrum and 75% overlap
    reintroduces some correlation between neighbouring frames.

    On real speech the effect is far more convincing than that number suggests,
    since natural signals are less relentlessly periodic than a synthetic
    harmonic stack. Reducing the overlap strengthens the effect at the cost of
    intelligibility.
    """
    x = np.asarray(x, dtype=np.float64)
    X = stft(x, n_fft=n_fft, hop=hop, window=window)

    rng = np.random.default_rng(seed)
    phase = rng.uniform(-np.pi, np.pi, X.shape)

    return istft(np.abs(X) * np.exp(1j * phase), hop=hop, window=window, length=len(x))

"""
Musical effects built on the vocoder.

Four effects, chosen because each isolates a different piece of the machinery:

``autotune``
    Pitch detection plus a time-varying shift. The full pipeline end to end.

``harmonize``
    Several constant shifts mixed together. Shows that the shifter composes.

``robotize`` / ``whisperize``
    Two lines each, and the clearest possible demonstration of what phase
    carries: keep the magnitude spectrogram exactly and destroy the phase, and
    the words survive while the pitch and voice quality do not.
"""

