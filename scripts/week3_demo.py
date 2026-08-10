#!/usr/bin/env python3
"""
Week 3 demo: vertical phase coherence, and a five-way method comparison.

Generates
---------
figures/08_peaks_and_regions.png   peak picking and regions of influence
figures/09_warble.png              amplitude envelopes -- the audible artefact
figures/10_method_comparison.png   D_M and warble across five methods
figures/11_locked_vs_plain.png     spectrograms, plain vs locked
results/week3_methods.csv          full results table

audio/outputs/
    m_ola_2x.wav / m_wsola_2x.wav / m_pv_2x.wav / m_locked_2x.wav
    voice_locked_slow.wav

Listen in this order at 2x: OLA (warbling), WSOLA (better), plain vocoder
(phasey), locked vocoder (clean). That progression is the project's argument.

Run from the repository root::

    python scripts/week3_demo.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from week1_demo import synth_voice  # noqa: E402

import elastiqa as eq  # noqa: E402

SR = 22050
N_FFT, HOP = 2048, 512
FIG_DIR = Path("figures")
AUD_DIR = Path("audio/outputs")
RES_DIR = Path("results")

STRETCHES = [0.5, 0.75, 1.25, 1.5, 2.0]

METHODS = {
    "ola": lambda x, s: eq.ola(x, s, n_fft=N_FFT, hop=HOP),
    "wsola": lambda x, s: eq.wsola(x, s, n_fft=N_FFT, hop=HOP),
    "pv": lambda x, s: eq.time_stretch(x, s, n_fft=N_FFT, hop=HOP),
    "pv_locked": lambda x, s: eq.time_stretch_locked(x, s, n_fft=N_FFT, hop=HOP),
}


def harmonic_tone(f0=220.0, duration=2.5, n=25):
    t = np.arange(int(duration * SR)) / SR
    partials = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, n + 1))
    return eq.normalise(partials)


def figure_peaks_and_regions(x: np.ndarray) -> None:
    """Show what the locking algorithm actually sees."""
    X = eq.stft(x, N_FFT, HOP)
    mag = np.abs(X[:, X.shape[1] // 2])

    peaks = eq.find_peaks(mag, threshold=1e-3)
    owner = eq.regions_of_influence(peaks, len(mag))

    freqs = eq.bin_frequencies(N_FFT, SR)
    upto = np.searchsorted(freqs, 3000)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.semilogy(freqs[:upto], np.maximum(mag[:upto], 1e-8), linewidth=1.0,
                color="0.35", label="magnitude spectrum")

    shown = peaks[peaks < upto]
    ax.plot(freqs[shown], mag[shown], "v", color="crimson", markersize=7,
            label=f"detected peaks ({len(shown)} shown)")

    # Region boundaries: where ownership changes.
    changes = np.flatnonzero(np.diff(owner[:upto])) + 1
    for c in changes:
        ax.axvline(freqs[c], color="steelblue", alpha=0.3, linewidth=0.8)

    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("magnitude")
    ax.set_title(
        "Peak picking and regions of influence — every bin between two blue "
        "lines is locked to the red peak inside it"
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_peaks_and_regions.png", dpi=150)
    plt.close(fig)

    print(f"  {len(peaks)} peaks detected, {len(np.unique(owner))} regions")


def figure_warble() -> None:
    """The artefact, drawn. A pure tone should come out with a flat envelope."""
    t = np.arange(2 * SR) / SR
    tone = np.sin(2 * np.pi * 220 * t)

    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True, sharey=True)

    for ax, (name, fn) in zip(axes, METHODS.items()):
        y = fn(tone, 2.0)
        env = np.abs(hilbert(y))
        score = eq.amplitude_warble(y)

        tt = np.arange(len(y)) / SR
        ax.plot(tt, y, linewidth=0.3, color="0.75")
        ax.plot(tt, env, linewidth=1.4, color="crimson")
        ax.plot(tt, -env, linewidth=1.4, color="crimson")
        ax.set_ylabel("amp")
        ax.set_title(f"{name}  —  warble = {score:.4f}", loc="left")
        ax.set_ylim(-1.6, 1.6)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(
        "Pure 220 Hz tone stretched 2x. The ideal envelope is a flat line.",
        y=1.0,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_warble.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    for name, fn in METHODS.items():
        y = fn(tone, 2.0)
        eq.save(AUD_DIR / f"m_{name}_2x.wav", eq.normalise(y), SR)
        print(f"  {name:10s} warble = {eq.amplitude_warble(y):.4f}")


def sweep(x: np.ndarray, label: str) -> list[dict]:
    rows = []
    for name, fn in METHODS.items():
        for s in STRETCHES:
            y = fn(x, s)
            row = eq.evaluate(x, y, stretch=s, n_fft=N_FFT, hop=HOP)

            # Only the two vocoders expose a modified STFT for D_M.
            if name == "pv":
                _, Y = eq.time_stretch(x, s, n_fft=N_FFT, hop=HOP, return_stft=True)
                row["consistency"] = eq.consistency(Y, hop=HOP, n_fft=N_FFT)
            elif name == "pv_locked":
                _, Y = eq.time_stretch_locked(
                    x, s, n_fft=N_FFT, hop=HOP, return_stft=True
                )
                row["consistency"] = eq.consistency(Y, hop=HOP, n_fft=N_FFT)

            row["method"] = name
            row["signal"] = label
            rows.append(row)
    return rows


def figure_method_comparison(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    markers = {"ola": "^", "wsola": "v", "pv": "o", "pv_locked": "s"}

    for name in METHODS:
        sel = [r for r in rows if r["method"] == name and r["signal"] == "sine"]
        axes[0].plot(
            [r["stretch"] for r in sel],
            [r["amplitude_warble"] for r in sel],
            marker=markers[name], label=name,
        )

    axes[0].set_yscale("log")
    axes[0].set_xlabel("stretch factor")
    axes[0].set_ylabel("amplitude warble (log scale)")
    axes[0].set_title("Envelope fluctuation on a pure 220 Hz tone")
    axes[0].legend()
    axes[0].grid(alpha=0.3, which="both")

    for name in ("pv", "pv_locked"):
        sel = [r for r in rows if r["method"] == name and r["signal"] == "tone"]
        axes[1].plot(
            [r["stretch"] for r in sel],
            [r["consistency"] for r in sel],
            marker=markers[name], label=name,
        )

    axes[1].set_yscale("log")
    axes[1].set_xlabel("stretch factor")
    axes[1].set_ylabel("$D_M$ (log scale)")
    axes[1].set_title("Consistency — phase locking vs. plain propagation")
    axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_method_comparison.png", dpi=150)
    plt.close(fig)


def figure_locked_vs_plain(voice: np.ndarray) -> None:
    plain = eq.time_stretch(voice, 2.0, n_fft=N_FFT, hop=HOP)
    locked = eq.time_stretch_locked(voice, 2.0, n_fft=N_FFT, hop=HOP)

    fig, _ = eq.viz.compare_spectrograms(
        {
            "original": voice,
            "plain phase vocoder, 2x": plain,
            "identity phase locking, 2x": locked,
        },
        SR,
        fmax=3000,
    )
    fig.savefig(FIG_DIR / "11_locked_vs_plain.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    eq.save(AUD_DIR / "voice_locked_slow.wav", eq.normalise(locked), SR)


def write_csv(rows: list[dict]) -> None:
    RES_DIR.mkdir(exist_ok=True)
    path = RES_DIR / "week3_methods.csv"
    fields = ["signal", "method", "stretch", "amplitude_warble", "consistency",
              "spectral_convergence", "log_spectral_distance", "ser_db"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    AUD_DIR.mkdir(parents=True, exist_ok=True)

    print("ElastiqA week 3 demo\n")
    tone = harmonic_tone()
    voice = synth_voice(duration=2.0)

    print("[1/5] peaks and regions")
    figure_peaks_and_regions(tone)

    print("\n[2/5] amplitude warble at 2x")
    figure_warble()

    print("\n[3/5] method sweep")
    sine = np.sin(2 * np.pi * 220 * np.arange(int(2.5 * SR)) / SR)
    rows = sweep(sine, "sine") + sweep(tone, "tone") + sweep(voice, "voice")

    print("\n[4/5] comparison figure")
    figure_method_comparison(rows)

    print("\n[5/5] locked vs plain on voice")
    figure_locked_vs_plain(voice)

    print()
    write_csv(rows)
    print("\nDone. Listen in order: m_ola_2x, m_wsola_2x, m_pv_2x, m_pv_locked_2x.")


if __name__ == "__main__":
    main()
