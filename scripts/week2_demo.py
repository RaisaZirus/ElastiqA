#!/usr/bin/env python3
"""
Week 2 demo: the phase vocoder, and proof that phase handling is what matters.

Generates
---------
figures/04_pitch_preserved.png    naive vs vocoder spectrograms at 1.5x slower
figures/05_phase_modes.png        the four phase modes, side by side
figures/06_consistency.png        D_M against stretch factor for every method
figures/07_waveform_detail.png    zoomed waveforms showing frame cancellation
results/week2_metrics.csv         the full results table

audio/outputs/
    pv_slow.wav / pv_fast.wav             correct vocoder
    broken_slow.wav                       passthrough phase -- listen for phasiness
    ola_slow.wav                          time-domain baseline
    robot.wav / whisper.wav               zeroed and randomised phase
    pitch_down_7.wav / pitch_up_5.wav     pitch shifting

The A/B that matters is pv_slow.wav against broken_slow.wav. Same magnitudes,
same stretch factor, same everything except how phase was propagated.

Run from the repository root::

    python scripts/week2_demo.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import elastiqa as eq  # noqa: E402
from week1_demo import synth_voice  # noqa: E402

SR = 22050
N_FFT, HOP = 2048, 512
FIG_DIR = Path("figures")
AUD_DIR = Path("audio/outputs")
RES_DIR = Path("results")

STRETCHES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


def _f0_hz(x: np.ndarray, sr: int = SR, lo: float = 70.0, hi: float = 500.0) -> float:
    """Fundamental frequency by autocorrelation.

    A spectral peak-pick is unreliable here: resonance routinely makes the
    second harmonic louder than the fundamental, so the peak lands an octave
    high. Autocorrelation looks for the period instead of the loudest partial,
    which is exactly the distinction that matters. (A proper YIN implementation
    arrives in week 5 for auto-tune; this is the two-line version.)
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()

    corr = np.correlate(x, x, mode="full")[len(x) - 1 :]
    lo_lag, hi_lag = int(sr / hi), int(sr / lo)
    lo_lag = max(lo_lag, 1)

    segment = corr[lo_lag : hi_lag + 1]
    if segment.size == 0:
        return 0.0
    return float(sr / (lo_lag + int(np.argmax(segment))))


def figure_pitch_preserved(x: np.ndarray) -> None:
    """The headline: same duration change, opposite outcome for pitch."""
    naive = eq.speed_change(x, 1 / 1.5)
    pv = eq.time_stretch(x, 1.5, n_fft=N_FFT, hop=HOP)

    fig, axes = eq.viz.compare_spectrograms(
        {
            "original": x,
            f"naive resampling, 1.5x longer  -> {_f0_hz(naive):.0f} Hz "
            "(pitch dragged down)": naive,
            f"phase vocoder, 1.5x longer  -> {_f0_hz(pv):.0f} Hz "
            "(pitch unchanged)": pv,
        },
        SR,
        fmax=3000,
    )
    for ax in axes:
        ax.axhline(500, color="cyan", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.axhline(1500, color="cyan", linestyle=":", linewidth=0.8, alpha=0.7)

    fig.savefig(FIG_DIR / "04_pitch_preserved.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    eq.save(AUD_DIR / "pv_slow.wav", eq.normalise(pv), SR)
    eq.save(AUD_DIR / "pv_fast.wav", eq.normalise(eq.time_stretch(x, 0.6)), SR)
    print(f"  original {_f0_hz(x):.0f} Hz -> "
          f"naive {_f0_hz(naive):.0f} Hz, vocoder {_f0_hz(pv):.0f} Hz")


def figure_phase_modes(x: np.ndarray) -> None:
    """All four phase modes at the same stretch, so magnitude is held constant."""
    outputs, scores = {}, {}
    for mode in eq.PHASE_MODES:
        y, Y = eq.time_stretch(
            x, 1.5, n_fft=N_FFT, hop=HOP, mode=mode, return_stft=True, seed=0
        )
        outputs[mode] = y
        scores[mode] = eq.consistency(Y, hop=HOP, n_fft=N_FFT)

    labelled = {
        f"{mode}   (D_M = {scores[mode]:.4f})": y for mode, y in outputs.items()
    }
    fig, _ = eq.viz.compare_spectrograms(labelled, SR, fmax=3000)
    fig.suptitle(
        "Identical magnitudes, identical stretch. Only the phase handling differs.",
        y=1.003,
        fontsize=10,
    )
    fig.savefig(FIG_DIR / "05_phase_modes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    eq.save(AUD_DIR / "broken_slow.wav", eq.normalise(outputs["passthrough"]), SR)
    eq.save(AUD_DIR / "robot.wav", eq.normalise(outputs["zero"]), SR)
    eq.save(AUD_DIR / "whisper.wav", eq.normalise(outputs["random"]), SR)

    for mode, d in scores.items():
        print(f"  {mode:12s} D_M = {d:.4f}")


def harmonic_tone(f0: float = 220.0, duration: float = 2.5, n: int = 25) -> np.ndarray:
    """A stationary harmonic stack -- no vibrato, no onsets.

    The phase vocoder models each bin as a sinusoid of slowly varying
    frequency. A stationary tone satisfies that assumption exactly, so this is
    the signal on which the algorithm's advantage should be clearest. Comparing
    against the vibrato voice shows how the advantage narrows once the
    assumption is strained.
    """
    t = np.arange(int(duration * SR)) / SR
    x = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, n + 1))
    return eq.normalise(x)


def _sweep(x: np.ndarray, label: str) -> tuple[list[dict], dict[str, list[float]]]:
    rows: list[dict] = []
    curves: dict[str, list[float]] = {}

    for mode in ("vocoder", "passthrough"):
        vals = []
        for st in STRETCHES:
            y, Y = eq.time_stretch(
                x, st, n_fft=N_FFT, hop=HOP, mode=mode, return_stft=True, seed=0
            )
            row = eq.evaluate(x, y, stretch=st, modified_stft=Y, n_fft=N_FFT, hop=HOP)
            row["method"] = mode
            row["signal"] = label
            rows.append(row)
            vals.append(row["consistency"])
        curves[mode] = vals

    ola_lsd = []
    for st in STRETCHES:
        y = eq.ola(x, st, n_fft=N_FFT, hop=HOP)
        row = eq.evaluate(x, y, stretch=st, n_fft=N_FFT, hop=HOP)
        row["method"] = "ola"
        row["signal"] = label
        rows.append(row)
        ola_lsd.append(row["log_spectral_distance"])
    curves["ola_lsd"] = ola_lsd

    return rows, curves


def figure_consistency_curve(voice: np.ndarray) -> list[dict]:
    """D_M against stretch factor, on stationary and non-stationary material."""
    tone = harmonic_tone()

    rows_tone, curves_tone = _sweep(tone, "stationary_tone")
    rows_voice, curves_voice = _sweep(voice, "vibrato_voice")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, curves, title in [
        (axes[0], curves_tone, "Stationary harmonic tone"),
        (axes[1], curves_voice, "Vibrato voice"),
    ]:
        ax.plot(STRETCHES, curves["vocoder"], "o-", label="phase vocoder")
        ax.plot(STRETCHES, curves["passthrough"], "s--", label="passthrough (broken)")
        ax.set_xlabel("stretch factor")
        ax.set_ylabel("$D_M$  (lower is better)")
        ax.set_title(title)
        ax.axvline(1.0, color="0.85", zorder=0)
        ax.legend()
        ax.grid(alpha=0.3)

    pv_lsd = [
        r["log_spectral_distance"] for r in rows_tone if r["method"] == "vocoder"
    ]
    axes[2].plot(STRETCHES, pv_lsd, "o-", label="phase vocoder")
    axes[2].plot(STRETCHES, curves_tone["ola_lsd"], "^--", label="OLA")
    axes[2].set_xlabel("stretch factor")
    axes[2].set_ylabel("log-spectral distance (dB)")
    axes[2].set_title("Spectral distortion (stationary tone)")
    axes[2].axvline(1.0, color="0.85", zorder=0)
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.suptitle(
        "The vocoder's advantage is clean on stationary content and narrows "
        "on vibrato -- the algorithm assumes slowly varying sinusoids.",
        y=1.02,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_consistency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    eq.save(AUD_DIR / "ola_slow.wav", eq.normalise(eq.ola(voice, 1.5)), SR)
    eq.save(AUD_DIR / "tone_pv_slow.wav", eq.normalise(eq.time_stretch(tone, 1.5)), SR)
    eq.save(
        AUD_DIR / "tone_broken_slow.wav",
        eq.normalise(eq.time_stretch(tone, 1.5, mode="passthrough")),
        SR,
    )

    for st, a, b in zip(STRETCHES, curves_tone["vocoder"], curves_tone["passthrough"]):
        flag = "ok " if a <= b else "!! "
        print(f"  {flag}stretch={st:<5} vocoder D_M={a:.4f}  passthrough D_M={b:.4f}")

    return rows_tone + rows_voice


def figure_waveform_detail(x: np.ndarray) -> None:
    """Zoom in far enough to see the cancellation that D_M is measuring."""
    good = eq.time_stretch(x, 1.5, n_fft=N_FFT, hop=HOP, mode="vocoder")
    bad = eq.time_stretch(x, 1.5, n_fft=N_FFT, hop=HOP, mode="passthrough")

    start = len(good) // 2
    n = 900
    t = np.arange(n) / SR * 1000  # ms

    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True, sharey=True)
    axes[0].plot(t, good[start : start + n], linewidth=0.9, color="teal")
    axes[0].set_title("vocoder: phase accumulated -- periodicity intact")
    axes[1].plot(t, bad[start : start + n], linewidth=0.9, color="crimson")
    axes[1].set_title(
        "passthrough: analysis phase reused -- frames fight each other, "
        "amplitude collapses and revives"
    )
    axes[1].set_xlabel("time (ms)")
    for ax in axes:
        ax.set_ylabel("amplitude")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_waveform_detail.png", dpi=150)
    plt.close(fig)


def do_pitch_shifting(x: np.ndarray) -> None:
    down = eq.pitch_shift(x, -7, n_fft=N_FFT, hop=HOP)
    up = eq.pitch_shift(x, +5, n_fft=N_FFT, hop=HOP)

    eq.save(AUD_DIR / "pitch_down_7.wav", eq.normalise(down), SR)
    eq.save(AUD_DIR / "pitch_up_5.wav", eq.normalise(up), SR)

    print(f"  {_f0_hz(x):.0f} Hz -> -7 st: {_f0_hz(down):.0f} Hz, "
          f"+5 st: {_f0_hz(up):.0f} Hz  (lengths preserved: "
          f"{len(down) == len(up) == len(x)})")
    print("  NOTE: formants still move with the pitch. That is week 4's problem.")


def write_csv(rows: list[dict]) -> None:
    RES_DIR.mkdir(exist_ok=True)
    path = RES_DIR / "week2_metrics.csv"
    fields = ["signal", "method", "stretch", "consistency", "spectral_convergence",
              "log_spectral_distance", "ser_db"]

    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"  wrote {path} ({len(rows)} rows)")


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    AUD_DIR.mkdir(parents=True, exist_ok=True)

    print("ElastiqA week 2 demo\n")
    x = synth_voice(duration=2.5)

    print("[1/5] pitch preservation")
    figure_pitch_preserved(x)

    print("\n[2/5] phase modes")
    figure_phase_modes(x)

    print("\n[3/5] consistency curves")
    rows = figure_consistency_curve(x)

    print("\n[4/5] waveform detail")
    figure_waveform_detail(x)

    print("\n[5/5] pitch shifting")
    do_pitch_shifting(x)

    print()
    write_csv(rows)
    print(f"\nDone. Compare {AUD_DIR}/pv_slow.wav against {AUD_DIR}/broken_slow.wav.")


if __name__ == "__main__":
    main()
