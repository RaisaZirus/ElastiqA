#!/usr/bin/env python3
"""
Week 1 demo: prove the foundation works, and show the problem to be solved.

Generates
---------
figures/01_cola_envelope.png     individual windows and their overlap-add sum
figures/02_reconstruction.png    input, output, and error of the STFT round trip
figures/03_naive_chipmunk.png    spectrograms showing pitch dragged along by speed
audio/original.wav               a synthetic voice-like test signal
audio/naive_fast.wav             1.5x faster -- and a perfect fifth sharp
audio/naive_slow.wav             0.67x slower -- and correspondingly flat

Run from the repository root::

    python scripts/week1_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import elastiqa as eq  # noqa: E402

SR = 22050
FIG_DIR = Path("figures")
AUD_DIR = Path("audio/outputs")


def synth_voice(duration: float = 2.5, f0: float = 165.0, sr: int = SR) -> np.ndarray:
    """A crude vowel-like signal: harmonic stack shaped by fixed formants.

    Not a real voice, but it has the two properties that matter for
    demonstrating the pitch/formant problem: a clear fundamental with
    harmonics, and resonant peaks at fixed frequencies. When you resample it,
    you will see the formants move -- which is exactly what should not happen.
    """
    t = np.arange(int(duration * sr)) / sr

    # Slight vibrato so the signal is not perfectly stationary.
    f0_t = f0 * (1.0 + 0.02 * np.sin(2 * np.pi * 5.0 * t))
    phase = 2 * np.pi * np.cumsum(f0_t) / sr

    x = np.zeros_like(t)
    for h in range(1, 40):
        x += (1.0 / h) * np.sin(h * phase)

    # Formants for a schwa-ish vowel: 500, 1500, 2500 Hz.
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    envelope = np.zeros_like(freqs)
    for centre, bw, gain in [(500, 90, 1.0), (1500, 120, 0.5), (2500, 160, 0.3)]:
        envelope += gain / (1.0 + ((freqs - centre) / bw) ** 2)
    envelope += 0.02

    x = np.fft.irfft(spec * envelope, n=len(x))

    # Fade in and out so the edges do not click.
    fade = int(0.02 * sr)
    ramp = np.linspace(0, 1, fade)
    x[:fade] *= ramp
    x[-fade:] *= ramp[::-1]

    return eq.normalise(x, 0.9)


def figure_cola() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))

    w = eq.get_window("hann", 512)
    eq.viz.plot_envelope(w, hop=128, n_frames=12, ax=axes[0])
    ok, val, ripple = eq.check_cola(w, 128)
    axes[0].set_title(
        f"75% overlap (hop = N/4): COLA satisfied, sum = {val:.3f}, "
        f"ripple = {ripple:.1e}"
    )

    eq.viz.plot_envelope(w, hop=256, n_frames=12, ax=axes[1])
    ok2, val2, ripple2 = eq.check_cola(w, 256)
    axes[1].set_title(
        f"50% overlap (hop = N/2): COLA VIOLATED for $w^2$, "
        f"ripple = {ripple2:.3f}  <-- audible buzz once phase is modified"
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_cola_envelope.png", dpi=150)
    plt.close(fig)
    print(f"  hop=N/4 -> COLA {ok}, ripple {ripple:.2e}")
    print(f"  hop=N/2 -> COLA {ok2}, ripple {ripple2:.2e}")


def figure_reconstruction(x: np.ndarray) -> None:
    n_fft, hop = 2048, 512
    X = eq.stft(x, n_fft=n_fft, hop=hop)
    y = eq.istft(X, hop=hop, length=len(x))

    err = np.abs(y - x)
    max_err = float(err.max())
    print(f"  round-trip max error: {max_err:.3e}")

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    eq.viz.plot_waveform(x, SR, ax=axes[0], title="input")
    eq.viz.plot_waveform(y, SR, ax=axes[1], title="istft(stft(x))")
    axes[2].semilogy(np.arange(len(err)) / SR, np.maximum(err, 1e-20), linewidth=0.5)
    axes[2].axhline(1e-10, color="crimson", linestyle="--", label="tolerance 1e-10")
    axes[2].set_ylim(1e-20, 1e-6)
    axes[2].set_ylabel("|error|")
    axes[2].set_xlabel("time (s)")
    axes[2].set_title(f"reconstruction error (max = {max_err:.2e})")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_reconstruction.png", dpi=150)
    plt.close(fig)


def figure_naive(x: np.ndarray) -> None:
    fast = eq.speed_change(x, 1.5)
    slow = eq.speed_change(x, 1 / 1.5)

    eq.save(AUD_DIR / "original.wav", x, SR)
    eq.save(AUD_DIR / "naive_fast.wav", fast, SR)
    eq.save(AUD_DIR / "naive_slow.wav", slow, SR)

    fig, axes = eq.viz.compare_spectrograms(
        {
            "original": x,
            "naive 1.5x faster  (pitch up ~7 semitones, formants moved)": fast,
            "naive 1.5x slower  (pitch down ~7 semitones, formants moved)": slow,
        },
        SR,
        fmax=4000,
    )
    for ax in axes:
        for f in (500, 1500, 2500):
            ax.axhline(f, color="cyan", linestyle=":", linewidth=0.8, alpha=0.7)

    fig.suptitle(
        "Dotted lines mark the original formants. They should stay put -- "
        "they do not.",
        y=1.005,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_naive_chipmunk.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  wrote 3 wav files to {AUD_DIR}/")


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    AUD_DIR.mkdir(parents=True, exist_ok=True)

    print("ElastiqA week 1 demo")
    print(f"  soundfile backend: {eq.have_soundfile()}")

    x = synth_voice()

    print("\n[1/3] overlap-add envelope")
    figure_cola()

    print("\n[2/3] perfect reconstruction")
    figure_reconstruction(x)

    print("\n[3/3] naive resampling baseline")
    figure_naive(x)

    print(f"\nDone. Figures in {FIG_DIR}/, audio in {AUD_DIR}/")


if __name__ == "__main__":
    main()
