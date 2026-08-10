#!/usr/bin/env python3
"""
Week 5 demo: musical effects and the full benchmark.

Generates
---------
figures/16_pitch_tracking.png    YIN f0 track with voicing
figures/17_autotune.png          detected vs. quantised pitch, plus strength
figures/18_harmonizer.png        spectrum of the harmonised signal
figures/19_phase_effects.png     robot and whisper, spectrograms and waveforms
figures/20_benchmark.png         six methods x five stretches, four content classes
results/benchmark.csv            180 rows -- the full sweep
results/benchmark_tables.md      markdown tables ready to paste into the README

audio/outputs/
    sung_flat.wav / sung_autotuned.wav / sung_autotune_half.wav
    harmonized.wav / robot_voice.wav / whisper_voice.wav

Run from the repository root::

    python scripts/week5_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from week1_demo import synth_voice  # noqa: E402

import elastiqa as eq  # noqa: E402

SR = 22050
N_FFT, HOP = 2048, 512
FIG_DIR = Path("figures")
AUD_DIR = Path("audio/outputs")
RES_DIR = Path("results")


def sung_melody(duration_per_note=0.6, detune_cents=(-45, +38, -30, +25)):
    """A four-note melody, each note deliberately out of tune.

    Real singers drift; this makes the drift explicit and known, so the
    correction can be measured rather than merely heard.
    """
    notes_midi = [69, 71, 72, 74]              # A4 B4 C5 D5
    pieces = []

    for midi, cents in zip(notes_midi, detune_cents):
        hz = eq.midi_to_hz(midi) * 2 ** (cents / 1200.0)
        n = int(duration_per_note * SR)
        t = np.arange(n) / SR
        note = sum(np.sin(2 * np.pi * hz * h * t) / h for h in range(1, 16))

        fade = int(0.02 * SR)
        ramp = np.linspace(0, 1, fade)
        note[:fade] *= ramp
        note[-fade:] *= ramp[::-1]
        pieces.append(note)

    return eq.normalise(np.concatenate(pieces)), notes_midi, detune_cents


def figure_pitch_tracking(x):
    f0, voiced = eq.yin_track(x, SR, frame_length=N_FFT, hop=HOP)
    t = np.arange(len(f0)) * HOP / SR

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t[voiced], f0[voiced], "o", markersize=2.5, color="steelblue",
            label="YIN f0 (voiced frames)")
    ax.plot(t[~voiced], np.full((~voiced).sum(), 0), "x", markersize=3,
            color="0.7", label="unvoiced")

    for midi in (69, 71, 72, 74):
        hz = eq.midi_to_hz(midi)
        ax.axhline(hz, color="crimson", linestyle=":", linewidth=0.9)
        ax.text(t[-1] * 1.005, hz, f" {hz:.0f} Hz", fontsize=7, va="center")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_ylim(380, 620)
    ax.set_title("YIN pitch track — dotted lines are the in-tune targets")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "16_pitch_tracking.png", dpi=150)
    plt.close(fig)

    print(f"  {voiced.sum()}/{len(voiced)} frames voiced")


def figure_autotune(x, notes_midi, detune_cents):
    y, detected, target, voiced = eq.autotune(
        x, SR, scale="chromatic", n_fft=N_FFT, hop=HOP, return_tracks=True
    )
    half = eq.autotune(x, SR, strength=0.5, n_fft=N_FFT, hop=HOP)

    t = np.arange(len(detected)) * HOP / SR

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    axes[0].plot(t[voiced], detected[voiced], "o", markersize=2.5,
                 color="crimson", label="detected (out of tune)")
    axes[0].plot(t[voiced], target[voiced], "-", linewidth=2.0,
                 color="steelblue", label="quantised target")
    axes[0].set_ylabel("frequency (Hz)")
    axes[0].set_ylim(380, 620)
    axes[0].set_title("Detection and quantisation")
    axes[0].legend(fontsize=8)

    for sig, label, colour in [
        (x, "input", "crimson"),
        (half, "strength = 0.5", "goldenrod"),
        (y, "strength = 1.0", "steelblue"),
    ]:
        f0, v = eq.yin_track(sig, SR, frame_length=N_FFT, hop=HOP)
        cents = np.full(len(f0), np.nan)
        nearest = eq.midi_to_hz(np.round(eq.hz_to_midi(np.maximum(f0, 1e-9))))
        cents[v] = 1200 * np.log2(f0[v] / nearest[v])
        axes[1].plot(np.arange(len(cents)) * HOP / SR, cents, linewidth=1.4,
                     color=colour, label=label)

    axes[1].axhline(0, color="0.5", linewidth=0.8)
    axes[1].set_ylabel("cents from nearest note")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylim(-60, 60)
    axes[1].set_title("Tuning error before and after")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "17_autotune.png", dpi=150)
    plt.close(fig)

    eq.save(AUD_DIR / "sung_flat.wav", eq.normalise(x), SR)
    eq.save(AUD_DIR / "sung_autotuned.wav", eq.normalise(y), SR)
    eq.save(AUD_DIR / "sung_autotune_half.wav", eq.normalise(half), SR)

    def mean_error(sig):
        f0, v = eq.yin_track(sig, SR, frame_length=N_FFT, hop=HOP)
        if not v.any():
            return float("nan")
        nearest = eq.midi_to_hz(np.round(eq.hz_to_midi(f0[v])))
        return float(np.mean(np.abs(1200 * np.log2(f0[v] / nearest))))

    print(f"  planted detune: {detune_cents} cents")
    print(f"  mean |error|  input {mean_error(x):5.1f} c  "
          f"strength 0.5 {mean_error(half):5.1f} c  "
          f"strength 1.0 {mean_error(y):5.1f} c")


def figure_harmonizer(x):
    y = eq.harmonize(x, intervals=(4.0, 7.0), n_fft=N_FFT, hop=HOP)

    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    upto = int(np.searchsorted(freqs, 1200))

    def spectrum(sig):
        s = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
        return s / s.max()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.semilogy(freqs[:upto], np.maximum(spectrum(x)[:upto], 1e-6),
                linewidth=1.0, color="0.55", label="input")
    ax.semilogy(freqs[:upto], np.maximum(spectrum(y)[:upto], 1e-6),
                linewidth=1.2, color="steelblue", label="harmonised (root + 3rd + 5th)")

    root = eq.midi_to_hz(69)
    for semis, name in [(0, "root"), (4, "+4"), (7, "+7")]:
        ax.axvline(root * 2 ** (semis / 12), color="crimson", linestyle=":",
                   linewidth=0.9)
        ax.text(root * 2 ** (semis / 12), 1.3, name, fontsize=7, ha="center")

    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("normalised magnitude")
    ax.set_title("Harmoniser — new partials appear at the added intervals")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "18_harmonizer.png", dpi=150)
    plt.close(fig)

    eq.save(AUD_DIR / "harmonized.wav", eq.normalise(y), SR)


def figure_phase_effects(voice):
    robot = eq.robotize(voice, n_fft=N_FFT, hop=HOP)
    whisper = eq.whisperize(voice, n_fft=N_FFT, hop=HOP, seed=0)

    fig, _ = eq.viz.compare_spectrograms(
        {
            "original": voice,
            "robotize — phase zeroed": robot,
            "whisperize — phase randomised": whisper,
        },
        SR, fmax=3500,
    )
    fig.suptitle(
        "Magnitudes are (nearly) untouched in all three. Only phase differs.",
        y=1.003, fontsize=10,
    )
    fig.savefig(FIG_DIR / "19_phase_effects.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    eq.save(AUD_DIR / "robot_voice.wav", eq.normalise(robot), SR)
    eq.save(AUD_DIR / "whisper_voice.wav", eq.normalise(whisper), SR)

    def autocorr_at(sig, lag):
        sig = sig - sig.mean()
        ac = np.correlate(sig, sig, mode="full")[len(sig) - 1 :]
        return float(ac[lag] / ac[0])

    print(f"  autocorr at hop: original {autocorr_at(voice, HOP):.3f}  "
          f"robot {autocorr_at(robot, HOP):.3f}")


def figure_benchmark(rows):
    classes = ["sine", "harmonic", "percussive", "mixed"]
    metrics = {
        "sine": ("amplitude_warble", "amplitude warble"),
        "harmonic": ("log_spectral_distance", "log-spectral distance (dB)"),
        "percussive": ("crest_error", "crest error"),
        "mixed": ("log_spectral_distance", "log-spectral distance (dB)"),
    }
    markers = {"ola": "^", "wsola": "v", "pv": "o",
               "pv_locked": "s", "pv_transient": "D"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    for ax, cls in zip(axes.ravel(), classes):
        metric, ylabel = metrics[cls]
        for method, marker in markers.items():
            sel = sorted(
                (r for r in rows if r["signal"] == cls and r["method"] == method),
                key=lambda r: r["stretch"],
            )
            if not sel:
                continue
            ax.plot([r["stretch"] for r in sel], [r[metric] for r in sel],
                    marker=marker, markersize=5, label=method)

        if cls == "sine":
            ax.set_yscale("log")
        ax.set_title(f"{cls} — {ylabel}")
        ax.set_xlabel("stretch factor")
        ax.grid(alpha=0.3)
        ax.axvline(1.0, color="0.85", zorder=0)

    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "No method wins everywhere — which is the point of benchmarking more "
        "than one signal (naive baseline excluded).",
        y=1.0, fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "20_benchmark.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_tables(rows):
    RES_DIR.mkdir(exist_ok=True)
    eq.to_csv(rows, RES_DIR / "benchmark.csv")

    sections = [
        ("Amplitude warble — pure sine (lower is better)",
         "amplitude_warble", "sine"),
        ("Log-spectral distance — stationary harmonic (dB, lower is better)",
         "log_spectral_distance", "harmonic"),
        ("Log-spectral distance — vibrato (dB, lower is better)",
         "log_spectral_distance", "vibrato"),
        ("Crest error — percussive (lower is better)",
         "crest_error", "percussive"),
        ("Log-spectral distance — mixed content (dB, lower is better)",
         "log_spectral_distance", "mixed"),
        ("Consistency $D_M$ — vibrato, vocoders only (lower is better)",
         "consistency", "vibrato"),
    ]

    lines = ["# ElastiqA benchmark", "",
             "Six methods, five stretch factors, six content classes.",
             "The naive baseline is excluded from these tables: it does not",
             "preserve pitch, and none of these metrics measure pitch, so it",
             "wins several of them for the wrong reason.", ""]

    for title, metric, signal in sections:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(eq.to_markdown(rows, metric, signal=signal, exclude=("naive",)))
        lines.append("")

    path = RES_DIR / "benchmark_tables.md"
    path.write_text("\n".join(lines))
    print(f"  wrote {RES_DIR}/benchmark.csv and {path.name}")


def main():
    FIG_DIR.mkdir(exist_ok=True)
    AUD_DIR.mkdir(parents=True, exist_ok=True)

    print("ElastiqA week 5 demo\n")
    melody, notes, cents = sung_melody()
    voice = synth_voice(duration=2.0)

    print("[1/5] pitch tracking")
    figure_pitch_tracking(melody)

    print("\n[2/5] auto-tune")
    figure_autotune(melody, notes, cents)

    print("\n[3/5] harmoniser")
    figure_harmonizer(melody)

    print("\n[4/5] phase effects")
    figure_phase_effects(voice)

    print("\n[5/5] full benchmark (this takes a minute)")
    rows = eq.run_benchmark(verbose=False)
    print(f"  {len(rows)} rows")
    figure_benchmark(rows)
    write_tables(rows)

    print("\nDone. Play sung_flat.wav then sung_autotuned.wav.")


if __name__ == "__main__":
    main()
