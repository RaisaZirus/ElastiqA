#!/usr/bin/env python3
"""
Week 4 demo: transient preservation and formant-preserving pitch shift.

Generates
---------
figures/12_onset_detection.png     waveform, spectral flux, detected onsets
figures/13_transient_smearing.png  attack detail: plain vs. phase reset
figures/14_formant_envelope.png    spectrum and cepstral envelope
figures/15_formants_move.png       the payoff figure -- envelopes overlaid
results/week4_ablation.csv         ablation table

audio/outputs/
    drums_plain_15x.wav / drums_transient_15x.wav
    voice_down7_plain.wav / voice_down7_formant.wav
    voice_up7_plain.wav   / voice_up7_formant.wav

The pair to play for anyone who asks what the project does:
voice_down7_plain.wav (growly, wrong) against voice_down7_formant.wav
(same person, lower).

Run from the repository root::

    python scripts/week4_demo.py
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

import elastiqa as eq  # noqa: E402
from week1_demo import synth_voice  # noqa: E402

SR = 22050
N_FFT, HOP = 2048, 512
FIG_DIR = Path("figures")
AUD_DIR = Path("audio/outputs")
RES_DIR = Path("results")

FORMANTS = (500.0, 1500.0, 2500.0)


def synth_drums(n_hits=8, duration=3.0, spacing=0.35, seed=0):
    """Synthetic percussion: a decaying low body plus a noise burst."""
    x = np.zeros(int(duration * SR))
    rng = np.random.default_rng(seed)

    for i in range(n_hits):
        start = int((0.15 + i * spacing) * SR)
        length = int(0.12 * SR)
        if start + length > len(x):
            break
        t = np.arange(length) / SR
        body = np.sin(2 * np.pi * 90 * t) * np.exp(-t * 28)
        noise = rng.standard_normal(length) * np.exp(-t * 70) * 0.6
        x[start : start + length] += body + noise

    return eq.normalise(x)


def figure_onset_detection(drums: np.ndarray) -> None:
    X = eq.stft(drums, N_FFT, HOP)
    flux = eq.spectral_flux(X)
    onsets = eq.detect_onsets(X)

    frame_t = np.arange(len(flux)) * HOP / SR
    wave_t = np.arange(len(drums)) / SR

    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)

    axes[0].plot(wave_t, drums, linewidth=0.4, color="0.4")
    axes[0].set_ylabel("amplitude")
    axes[0].set_title("waveform")

    axes[1].plot(frame_t, flux, linewidth=1.2, color="steelblue", label="spectral flux")
    for o in onsets:
        for ax in axes:
            ax.axvline(o * HOP / SR, color="crimson", linestyle="--",
                       linewidth=1.0, alpha=0.8)
    axes[1].set_ylabel("flux")
    axes[1].set_xlabel("time (s)")
    axes[1].set_title(f"half-wave rectified spectral flux — {len(onsets)} onsets")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "12_onset_detection.png", dpi=150)
    plt.close(fig)
    print(f"  {len(onsets)} onsets detected")


def figure_transient_smearing(drums: np.ndarray) -> list[dict]:
    """Zoom in on one attack and measure what each variant does to it."""
    variants = {
        "locked, no transient handling": eq.time_stretch_locked(
            drums, 1.5, n_fft=N_FFT, hop=HOP
        ),
        "locked + phase reset at onsets": eq.time_stretch_transient(
            drums, 1.5, n_fft=N_FFT, hop=HOP
        ),
        "wsola": eq.wsola(drums, 1.5, n_fft=N_FFT, hop=HOP),
    }

    c_orig = eq.crest_factor(drums)

    fig, axes = plt.subplots(len(variants) + 1, 1, figsize=(11, 8), sharex=True)
    window = slice(int(0.6 * SR), int(0.95 * SR))

    seg = drums[int(0.4 * SR) : int(0.75 * SR)]
    tt = np.arange(len(seg)) / SR * 1000
    axes[0].plot(tt, seg, linewidth=0.5, color="0.4")
    axes[0].plot(tt, np.abs(hilbert(seg)), linewidth=1.5, color="black")
    axes[0].set_title(f"original — crest factor {c_orig:.2f}", loc="left")

    rows = []
    for ax, (name, y) in zip(axes[1:], variants.items()):
        c = eq.crest_factor(y)
        seg = y[window]
        tt = np.arange(len(seg)) / SR * 1000
        ax.plot(tt, seg, linewidth=0.5, color="0.6")
        ax.plot(tt, np.abs(hilbert(seg)), linewidth=1.5, color="crimson")
        ax.set_title(f"{name} — crest factor {c:.2f}", loc="left")
        rows.append({"variant": name, "crest_factor": c,
                     "crest_loss": c_orig - c})
        print(f"  {name:32s} crest {c:.2f}  (lost {c_orig - c:.2f})")

    axes[-1].set_xlabel("time (ms)")
    fig.suptitle("A single drum attack, stretched 1.5x", y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "13_transient_smearing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    eq.save(AUD_DIR / "drums_plain_15x.wav",
            eq.normalise(variants["locked, no transient handling"]), SR)
    eq.save(AUD_DIR / "drums_transient_15x.wav",
            eq.normalise(variants["locked + phase reset at onsets"]), SR)

    rows.insert(0, {"variant": "original", "crest_factor": c_orig, "crest_loss": 0.0})
    return rows


def figure_formant_envelope(voice: np.ndarray) -> None:
    """Show the cepstrum separating envelope from harmonics."""
    X = eq.stft(voice, N_FFT, HOP)
    mag = np.abs(X[:, X.shape[1] // 2])
    freqs = eq.bin_frequencies(N_FFT, SR)
    upto = int(np.searchsorted(freqs, 4000))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].semilogy(freqs[:upto], np.maximum(mag[:upto], 1e-8),
                     linewidth=0.8, color="0.6", label="magnitude spectrum")
    for q, colour in [(20, "steelblue"), (40, "crimson"), (120, "seagreen")]:
        env = eq.cepstral_envelope(mag, quefrency=q)
        axes[0].semilogy(freqs[:upto], env[:upto], linewidth=1.6, color=colour,
                         label=f"envelope, quefrency={q}")
    for f in FORMANTS:
        axes[0].axvline(f, color="0.8", linestyle=":", zorder=0)

    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("magnitude")
    axes[0].set_title("Cepstral liftering: quefrency sets the smoothing")
    axes[0].legend(fontsize=8)

    cep = np.fft.irfft(np.log(np.maximum(mag, 1e-10)), n=N_FFT)
    axes[1].plot(np.arange(400), cep[:400], linewidth=0.9, color="0.3")
    axes[1].axvline(40, color="crimson", linestyle="--", label="lifter cutoff (q=40)")
    pitch_period = SR / 165.0
    axes[1].axvline(pitch_period, color="seagreen", linestyle="--",
                    label=f"pitch period ({pitch_period:.0f} samples)")
    axes[1].set_xlabel("quefrency (samples)")
    axes[1].set_ylabel("cepstrum")
    axes[1].set_title("The two components separate by rate of variation")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "14_formant_envelope.png", dpi=150)
    plt.close(fig)


def _envelope_of(x: np.ndarray) -> np.ndarray:
    X = eq.stft(x, N_FFT, HOP)
    return eq.cepstral_envelope(np.abs(X).mean(axis=1), quefrency=40)


def _formant_peaks(x, n=3, fmax=3500.0):
    env = _envelope_of(x)
    freqs = eq.bin_frequencies(N_FFT, SR)
    upto = int(np.searchsorted(freqs, fmax))
    peaks = eq.find_peaks(env[:upto], threshold=0.0)
    if len(peaks) == 0:
        return []
    strongest = peaks[np.argsort(env[peaks])[::-1][:n]]
    return sorted(float(freqs[p]) for p in strongest)


def _f0_hz(x, lo=70.0, hi=600.0):
    x = x - x.mean()
    corr = np.correlate(x, x, mode="full")[len(x) - 1 :]
    lo_lag, hi_lag = int(SR / hi), int(SR / lo)
    return float(SR / (lo_lag + int(np.argmax(corr[lo_lag : hi_lag + 1]))))


def figure_formants_move(voice: np.ndarray) -> list[dict]:
    """The payoff figure: envelopes overlaid, with and without correction."""
    freqs = eq.bin_frequencies(N_FFT, SR)
    upto = int(np.searchsorted(freqs, 4000))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    rows = []

    for ax, st in zip(axes, (-7, +7)):
        plain = eq.pitch_shift(voice, st, n_fft=N_FFT, hop=HOP)
        fixed = eq.pitch_shift_formant(voice, st, n_fft=N_FFT, hop=HOP)

        for sig, label, colour, style in [
            (voice, "original", "0.3", "-"),
            (plain, f"plain shift {st:+d} st", "crimson", "--"),
            (fixed, f"formant-preserved {st:+d} st", "steelblue", "-"),
        ]:
            ax.semilogy(freqs[:upto], _envelope_of(sig)[:upto],
                        linewidth=1.6, color=colour, linestyle=style, label=label)

        for f in FORMANTS:
            ax.axvline(f, color="0.85", linestyle=":", zorder=0)

        ax.set_xlabel("frequency (Hz)")
        ax.set_title(f"{st:+d} semitones")
        ax.legend(fontsize=8)

        for name, sig in [("plain", plain), ("formant", fixed)]:
            rows.append({
                "shift_semitones": st,
                "method": name,
                "f0_hz": round(_f0_hz(sig), 1),
                "formants_hz": ";".join(f"{p:.0f}" for p in _formant_peaks(sig)),
            })

        tag = "down7" if st < 0 else "up7"
        eq.save(AUD_DIR / f"voice_{tag}_plain.wav", eq.normalise(plain), SR)
        eq.save(AUD_DIR / f"voice_{tag}_formant.wav", eq.normalise(fixed), SR)

    axes[0].set_ylabel("envelope magnitude")
    fig.suptitle(
        "Dotted lines are the original formants. Red drifts off them; blue stays.",
        y=1.0, fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "15_formants_move.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    rows.insert(0, {"shift_semitones": 0, "method": "original",
                    "f0_hz": round(_f0_hz(voice), 1),
                    "formants_hz": ";".join(f"{p:.0f}" for p in _formant_peaks(voice))})
    for r in rows:
        print(f"  {r['method']:9s} {r['shift_semitones']:+3d} st  "
              f"f0={r['f0_hz']:6.1f} Hz  formants=[{r['formants_hz']}]")
    return rows


def write_csv(transient_rows: list[dict], formant_rows: list[dict]) -> None:
    RES_DIR.mkdir(exist_ok=True)

    path = RES_DIR / "week4_ablation.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "a", "b", "c", "d"])
        w.writerow(["transients", "variant", "crest_factor", "crest_loss", ""])
        for r in transient_rows:
            w.writerow(["transients", r["variant"], f"{r['crest_factor']:.3f}",
                        f"{r['crest_loss']:.3f}", ""])
        w.writerow(["formants", "method", "shift_semitones", "f0_hz", "formants_hz"])
        for r in formant_rows:
            w.writerow(["formants", r["method"], r["shift_semitones"],
                        r["f0_hz"], r["formants_hz"]])
    print(f"  wrote {path}")


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    AUD_DIR.mkdir(parents=True, exist_ok=True)

    print("ElastiqA week 4 demo\n")
    drums = synth_drums()
    voice = synth_voice(duration=2.5)

    print("[1/4] onset detection")
    figure_onset_detection(drums)

    print("\n[2/4] transient smearing at 1.5x")
    transient_rows = figure_transient_smearing(drums)

    print("\n[3/4] cepstral envelope")
    figure_formant_envelope(voice)

    print("\n[4/4] formant preservation")
    formant_rows = figure_formants_move(voice)

    print()
    write_csv(transient_rows, formant_rows)
    print("\nDone. Play voice_down7_plain.wav, then voice_down7_formant.wav.")


if __name__ == "__main__":
    main()
