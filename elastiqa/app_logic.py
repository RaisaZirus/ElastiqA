"""
Processing logic for the web interface.

Deliberately free of any Gradio import. Everything the app *does* lives here
and is covered by the test suite; ``app.py`` only wires widgets to these
functions. That split means a broken button is a UI bug and a wrong number is
a logic bug, and the two never get confused.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np

from . import (
    amplitude_warble,
    autotune,
    consistency,
    crest_factor,
    harmonize,
    log_spectral_distance,
    normalise,
    ola,
    pitch_shift,
    pitch_shift_formant,
    robotize,
    speed_change,
    stft,
    time_stretch,
    time_stretch_locked,
    time_stretch_transient,
    whisperize,
    wsola,
    yin_track,
)

__all__ = [
    "MAX_SECONDS",
    "load_path",
    "STRETCH_METHODS",
    "PITCH_METHODS",
    "EFFECTS",
    "prepare_audio",
    "to_gradio_audio",
    "run_time_stretch",
    "run_pitch_shift",
    "run_effect",
    "demo_signals",
    "format_metrics",
    "spectrogram_image",
    "spectrogram_png",
]

#: Uploads longer than this are truncated. A six-minute track would otherwise
#: stall the app mid-demonstration, which is the worst possible moment.
MAX_SECONDS = 15.0

STRETCH_METHODS: dict[str, str] = {
    "Phase vocoder + locking (best)": "pv_locked",
    "Phase vocoder + locking + transient reset": "pv_transient",
    "Phase vocoder (plain)": "pv",
    "Phase vocoder with broken phase": "pv_broken",
    "WSOLA (time domain)": "wsola",
    "Overlap-add (baseline)": "ola",
    "Naive resampling (pitch changes too)": "naive",
}

PITCH_METHODS: dict[str, str] = {
    "Formant-preserving (natural)": "formant",
    "Plain shift (chipmunk / growl)": "plain",
    "Naive resampling (length changes too)": "naive",
}

EFFECTS = ("Auto-tune", "Harmoniser", "Robot", "Whisper")


def load_path(path: str) -> tuple[int, np.ndarray]:
    """Decode an audio file without going through ffmpeg where possible.

    ``soundfile`` handles WAV, FLAC, and OGG natively; ``scipy`` handles WAV.
    Compressed formats -- MP3, M4A, AAC -- genuinely require ffmpeg, so if we
    get here without it, say so plainly rather than letting a ``FileNotFoundError``
    from a subprocess call surface as the user-facing error.
    """
    from pathlib import Path

    suffix = Path(path).suffix.lower()

    try:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float64", always_2d=False)
        return int(sr), np.asarray(data)
    except ImportError:
        pass
    except Exception as exc:
        if suffix not in (".wav", ".wave"):
            raise ValueError(_FFMPEG_HINT.format(suffix=suffix or "that format")) from exc
        raise ValueError(f"Could not read that WAV file: {exc}") from exc

    if suffix in (".wav", ".wave"):
        from scipy.io import wavfile

        sr, raw = wavfile.read(path)
        if raw.dtype.kind in "iu":
            info = np.iinfo(raw.dtype)
            raw = raw.astype(np.float64) / max(abs(info.min), info.max)
        return int(sr), np.asarray(raw, dtype=np.float64)

    raise ValueError(_FFMPEG_HINT.format(suffix=suffix or "that format"))


_FFMPEG_HINT = (
    "Cannot read {suffix} files without ffmpeg installed. "
    "Either convert the clip to WAV and upload that, or install ffmpeg "
    "(Windows: winget install Gyan.FFmpeg, then reopen your terminal; "
    "macOS: brew install ffmpeg; Linux: apt install ffmpeg)."
)


def prepare_audio(
    audio: tuple[int, np.ndarray] | str | None, max_seconds: float = MAX_SECONDS
) -> tuple[np.ndarray, int, str]:
    """Normalise Gradio audio input into mono float64 in [-1, 1].

    Accepts either the ``(sample_rate, samples)`` tuple Gradio produces with
    ``type="numpy"``, or a path string from ``type="filepath"``. Samples may
    arrive as int16, stereo, or both.

    Returns
    -------
    (x, sr, note)
        ``note`` is a message for the user -- empty when nothing needed saying.
    """
    if audio is None:
        raise ValueError("No audio yet. Upload a file or record from the microphone.")

    if isinstance(audio, (str, bytes)) or hasattr(audio, "__fspath__"):
        audio = load_path(str(audio))

    sr, data = audio
    x = np.asarray(data)

    if x.dtype.kind in "iu":
        info = np.iinfo(x.dtype)
        x = x.astype(np.float64) / max(abs(info.min), info.max)
    else:
        x = x.astype(np.float64)

    if x.ndim > 1:
        x = x.mean(axis=1)

    note = ""
    limit = int(max_seconds * sr)
    if len(x) > limit:
        x = x[:limit]
        note = f"Trimmed to the first {max_seconds:.0f} seconds."

    if len(x) < 1024:
        raise ValueError("That clip is too short to analyse. Try at least half a second.")

    peak = float(np.max(np.abs(x)))
    if peak < 1e-6:
        raise ValueError("That clip is silent.")

    return normalise(x, 0.95), int(sr), note


def to_gradio_audio(x: np.ndarray, sr: int) -> tuple[int, np.ndarray]:
    """Package a signal for a Gradio audio component, as 16-bit PCM."""
    y = np.clip(normalise(np.asarray(x, dtype=np.float64), 0.95), -1.0, 1.0)
    return int(sr), (y * 32767.0).astype(np.int16)


def run_time_stretch(
    x: np.ndarray,
    sr: int,
    stretch: float,
    method: str,
    n_fft: int = 2048,
    hop: int = 512,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one time-scaling method and measure the result.

    Returns
    -------
    (y, metrics)
        ``metrics`` always carries duration and spectral distance; the
        consistency measure appears only for methods that expose a modified
        STFT.
    """
    key = STRETCH_METHODS.get(method, method)
    modified: np.ndarray | None = None

    if key == "naive":
        y = speed_change(x, 1.0 / stretch)
    elif key == "ola":
        y = ola(x, stretch, n_fft=n_fft, hop=hop)
    elif key == "wsola":
        y = wsola(x, stretch, n_fft=n_fft, hop=hop)
    elif key == "pv":
        y, modified = time_stretch(x, stretch, n_fft=n_fft, hop=hop, return_stft=True)
    elif key == "pv_broken":
        y, modified = time_stretch(
            x, stretch, n_fft=n_fft, hop=hop, mode="passthrough", return_stft=True
        )
    elif key == "pv_locked":
        y, modified = time_stretch_locked(
            x, stretch, n_fft=n_fft, hop=hop, return_stft=True
        )
    elif key == "pv_transient":
        y, modified = time_stretch_transient(
            x, stretch, n_fft=n_fft, hop=hop, return_stft=True
        )
    else:
        raise ValueError(f"Unknown method: {method!r}")

    metrics: dict[str, Any] = {
        "input length": f"{len(x) / sr:.2f} s",
        "output length": f"{len(y) / sr:.2f} s",
        "spectral distance": f"{log_spectral_distance(x, y, n_fft, hop):.2f} dB",
    }
    if modified is not None:
        metrics["consistency (D_M)"] = f"{consistency(modified, hop=hop, n_fft=n_fft):.5f}"
    if key in ("pv_transient", "wsola", "ola"):
        metrics["crest factor"] = (
            f"{crest_factor(y):.2f}  (input {crest_factor(x):.2f})"
        )

    return y, metrics


def run_pitch_shift(
    x: np.ndarray,
    sr: int,
    semitones: float,
    method: str,
    n_fft: int = 2048,
    hop: int = 512,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one pitch-shifting method and measure the result."""
    key = PITCH_METHODS.get(method, method)

    if key == "naive":
        y = speed_change(x, 2.0 ** (semitones / 12.0))
    elif key == "plain":
        y = pitch_shift(x, semitones, n_fft=n_fft, hop=hop)
    elif key == "formant":
        y = pitch_shift_formant(x, semitones, n_fft=n_fft, hop=hop)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    metrics: dict[str, Any] = {
        "requested shift": f"{semitones:+.1f} semitones",
        "output length": f"{len(y) / sr:.2f} s  (input {len(x) / sr:.2f} s)",
    }

    before = _median_f0(x, sr, n_fft, hop)
    after = _median_f0(y, sr, n_fft, hop)
    if before > 0 and after > 0:
        measured = 12.0 * np.log2(after / before)
        metrics["detected pitch"] = f"{before:.1f} Hz -> {after:.1f} Hz"
        metrics["measured shift"] = f"{measured:+.2f} semitones"

    return y, metrics


def _median_f0(x: np.ndarray, sr: int, n_fft: int, hop: int) -> float:
    f0, voiced = yin_track(x, sr, frame_length=n_fft, hop=hop)
    return float(np.median(f0[voiced])) if voiced.any() else 0.0


def run_effect(
    x: np.ndarray,
    sr: int,
    effect: str,
    scale: str = "chromatic",
    strength: float = 1.0,
    intervals: tuple[float, ...] = (4.0, 7.0),
    n_fft: int = 2048,
    hop: int = 512,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one musical effect and measure the result."""
    metrics: dict[str, Any] = {}

    if effect == "Auto-tune":
        y = autotune(
            x, sr, scale=scale, strength=strength, n_fft=n_fft, hop=hop
        )
        metrics["scale"] = scale
        metrics["strength"] = f"{strength:.2f}"
        metrics["tuning error"] = (
            f"{_tuning_error(x, sr, n_fft, hop):.1f} -> "
            f"{_tuning_error(y, sr, n_fft, hop):.1f} cents"
        )
    elif effect == "Harmoniser":
        y = harmonize(x, intervals=tuple(intervals), n_fft=n_fft, hop=hop)
        metrics["intervals"] = ", ".join(f"{i:+.0f}" for i in intervals) + " semitones"
    elif effect == "Robot":
        y = robotize(x, n_fft=n_fft, hop=hop)
        metrics["buzz frequency"] = f"{sr / hop:.0f} Hz  (= sample rate / hop)"
        metrics["note"] = "The original pitch survives; the buzz is laid over it."
    elif effect == "Whisper":
        y = whisperize(x, n_fft=n_fft, hop=hop, seed=0)
        metrics["note"] = "Phase randomised. Magnitudes, and so the words, remain."
    else:
        raise ValueError(f"Unknown effect: {effect!r}")

    return y, metrics


def _tuning_error(x: np.ndarray, sr: int, n_fft: int, hop: int) -> float:
    """Mean absolute distance to the nearest semitone, in cents."""
    from .pitch.f0 import hz_to_midi, midi_to_hz

    f0, voiced = yin_track(x, sr, frame_length=n_fft, hop=hop)
    if not voiced.any():
        return float("nan")
    live = f0[voiced]
    nearest = midi_to_hz(np.round(hz_to_midi(live)))
    return float(np.mean(np.abs(1200.0 * np.log2(live / nearest))))


def demo_signals(sr: int = 22050, duration: float = 2.0) -> dict[str, np.ndarray]:
    """Fixed examples for the landing tab.

    Pre-rendered rather than computed on demand: a pause while someone waits
    for the very first sound is the one that loses the room.
    """
    t = np.arange(int(duration * sr)) / sr

    f0 = 165.0 * (1.0 + 0.02 * np.sin(2 * np.pi * 5.0 * t))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    voice = sum(np.sin(h * phase) / h for h in range(1, 40))

    spec = np.fft.rfft(voice)
    freqs = np.fft.rfftfreq(len(voice), 1 / sr)
    env = np.full_like(freqs, 0.02)
    for centre, bw, gain in [(500, 90, 1.0), (1500, 120, 0.5), (2500, 160, 0.3)]:
        env += gain / (1.0 + ((freqs - centre) / bw) ** 2)
    voice = np.fft.irfft(spec * env, n=len(voice))

    fade = int(0.02 * sr)
    ramp = np.linspace(0, 1, fade)
    voice[:fade] *= ramp
    voice[-fade:] *= ramp[::-1]
    voice = normalise(voice, 0.9)

    return {
        "original": voice,
        "naive": speed_change(voice, 1 / 1.5),
        "vocoder": time_stretch_locked(voice, 1.5),
        "broken": time_stretch(voice, 1.5, mode="passthrough"),
        "pitch_plain": pitch_shift(voice, -7),
        "pitch_formant": pitch_shift_formant(voice, -7),
    }


def format_metrics(metrics: dict[str, Any]) -> str:
    """Render a metrics dict as a markdown table."""
    if not metrics:
        return ""
    rows = "\n".join(f"| {k} | `{v}` |" for k, v in metrics.items())
    return f"| measurement | value |\n|---|---|\n{rows}"


def _spectrogram_figure(
    signals: dict[str, np.ndarray],
    sr: int,
    fmax: float = 4000.0,
    n_fft: int = 2048,
    hop: int = 512,
):
    """Build the stacked-spectrogram figure. Callers must close it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(signals)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.6 * n), squeeze=False)

    for ax, (label, sig) in zip(axes[:, 0], signals.items()):
        X = np.abs(stft(sig, n_fft=n_fft, hop=hop))
        ref = max(float(X.max()), 1e-12)
        db = np.maximum(20.0 * np.log10(np.maximum(X, 1e-12) / ref), -80.0)

        ax.imshow(
            db, origin="lower", aspect="auto", cmap="magma", vmin=-80.0,
            extent=(0.0, len(sig) / sr, 0.0, sr / 2.0),
        )
        ax.set_ylim(0, fmax)
        ax.set_ylabel("Hz")
        ax.set_title(label, fontsize=9, loc="left")

    axes[-1, 0].set_xlabel("time (s)")
    fig.tight_layout()
    return fig


def spectrogram_image(
    signals: dict[str, np.ndarray],
    sr: int,
    fmax: float = 4000.0,
    n_fft: int = 2048,
    hop: int = 512,
) -> np.ndarray:
    """Render stacked spectrograms as an RGB array.

    This is what the Gradio image component wants -- it accepts an ndarray, a
    PIL image, or a path, and rejects raw PNG bytes. Rendering straight from
    the Agg canvas avoids a round trip through an encoder and keeps the figure
    lifecycle in one place.
    """
    import matplotlib.pyplot as plt

    fig = _spectrogram_figure(signals, sr, fmax=fmax, n_fft=n_fft, hop=hop)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[..., :3].copy()
    plt.close(fig)
    return rgb


def spectrogram_png(
    signals: dict[str, np.ndarray],
    sr: int,
    fmax: float = 4000.0,
    n_fft: int = 2048,
    hop: int = 512,
) -> bytes:
    """Render stacked spectrograms as PNG bytes, for saving to disk.

    Returned as bytes rather than a figure so the caller never has to remember
    to close it -- an accumulating pile of open figures is a slow memory leak
    in a long-running app.
    """
    import matplotlib.pyplot as plt

    fig = _spectrogram_figure(signals, sr, fmax=fmax, n_fft=n_fft, hop=hop)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()
