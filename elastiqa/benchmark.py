"""
Benchmark harness.

The point of the project is not that it implements a phase vocoder -- plenty of
repositories do that. The point is that it *measures* one, against alternatives,
on controlled material, reproducibly. This module is where that happens.

Design
------
A benchmark is the cross product of three axes:

* **methods** -- the six TSM algorithms
* **stretch factors** -- from heavy compression to heavy expansion
* **signals** -- content classes chosen so that the known strengths and
  weaknesses of each algorithm have a chance to show

Every cell yields a row of metrics. Rows go to CSV for the report and to a
markdown table for the README.

On choosing signals
-------------------
Using one signal would be worse than useless, because no TSM method wins
everywhere and a single-signal benchmark would hide that. Driedger and Müller
make the point directly: music comprises harmonic, percussive, and transient
components, and no single method copes with all of them equally well. The
default set deliberately includes a case each method is expected to lose.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np

from .metrics import (
    amplitude_warble,
    consistency,
    crest_factor,
    log_spectral_distance,
    spectral_convergence,
)
from .tsm import (
    ola,
    time_stretch,
    time_stretch_locked,
    time_stretch_transient,
    wsola,
)

__all__ = [
    "DEFAULT_METHODS",
    "DEFAULT_STRETCHES",
    "make_test_signals",
    "run_benchmark",
    "to_csv",
    "to_markdown",
]

#: Methods under test. ``stft_of`` returns the modified STFT when the method
#: has one, so the consistency measure can be computed; time-domain methods
#: return None and simply have no D_M value.
DEFAULT_METHODS: dict[str, dict] = {
    "naive": {
        "fn": lambda x, s, **kw: _naive(x, s),
        "stft_of": None,
        "label": "naive resampling",
    },
    "ola": {"fn": ola, "stft_of": None, "label": "overlap-add"},
    "wsola": {"fn": wsola, "stft_of": None, "label": "WSOLA"},
    "pv": {
        "fn": time_stretch,
        "stft_of": lambda x, s, **kw: time_stretch(x, s, return_stft=True, **kw)[1],
        "label": "phase vocoder",
    },
    "pv_locked": {
        "fn": time_stretch_locked,
        "stft_of": lambda x, s, **kw: time_stretch_locked(
            x, s, return_stft=True, **kw
        )[1],
        "label": "PV + identity phase locking",
    },
    "pv_transient": {
        "fn": time_stretch_transient,
        "stft_of": lambda x, s, **kw: time_stretch_transient(
            x, s, return_stft=True, **kw
        )[1],
        "label": "PV + locking + transient reset",
    },
}

DEFAULT_STRETCHES = (0.5, 0.75, 1.25, 1.5, 2.0)


def _naive(x: np.ndarray, stretch: float) -> np.ndarray:
    from .naive import resample

    y = resample(x, stretch)
    target = int(round(len(x) * stretch))
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    return y[:target]


def make_test_signals(sr: int = 22050, duration: float = 2.5) -> dict[str, np.ndarray]:
    """Content classes covering the failure modes of each method.

    ``sine``
        A single sinusoid. The only signal on which `amplitude_warble` is
        valid, and the cleanest discriminator of phase handling.
    ``harmonic``
        A stationary harmonic stack -- the phase vocoder's best case, because
        it satisfies the slowly-varying-sinusoid assumption exactly.
    ``vibrato``
        The same stack with frequency modulation. Strains that assumption and
        is where phase locking earns its keep.
    ``percussive``
        Sharp attacks. The vocoder's worst case and WSOLA's traditional
        strength. Rank this class with ``crest_error`` rather than
        ``crest_loss``: a *negative* loss means the method raised the crest
        factor above the original, which is damage (artificial discontinuities)
        rather than fidelity, and signed ranking would reward it.
    ``mixed``
        Tone plus drums. No single method is best here, which is the point.
    ``noise``
        Broadband and aperiodic. A control: nothing should do well, and any
        method that appears to is being flattered by the metric.
    """
    n = int(duration * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(0)

    def norm(x):
        peak = float(np.max(np.abs(x)))
        return x / peak if peak > 1e-12 else x

    sine = np.sin(2 * np.pi * 220 * t)

    harmonic = norm(sum(np.sin(2 * np.pi * 220 * h * t) / h for h in range(1, 26)))

    f_inst = 220.0 * (1.0 + 0.03 * np.sin(2 * np.pi * 5.5 * t))
    phase = 2 * np.pi * np.cumsum(f_inst) / sr
    vibrato = norm(sum(np.sin(h * phase) / h for h in range(1, 26)))

    percussive = np.zeros(n)
    for i in range(int(duration / 0.35)):
        start = int((0.15 + i * 0.35) * sr)
        length = int(0.12 * sr)
        if start + length > n:
            break
        tt = np.arange(length) / sr
        percussive[start : start + length] += (
            np.sin(2 * np.pi * 90 * tt) * np.exp(-tt * 28)
            + rng.standard_normal(length) * np.exp(-tt * 70) * 0.6
        )
    percussive = norm(percussive)

    mixed = norm(0.6 * harmonic + 0.7 * percussive)
    noise = norm(rng.standard_normal(n) * 0.3)

    return {
        "sine": sine,
        "harmonic": harmonic,
        "vibrato": vibrato,
        "percussive": percussive,
        "mixed": mixed,
        "noise": noise,
    }


def run_benchmark(
    signals: dict[str, np.ndarray] | None = None,
    methods: dict[str, dict] | None = None,
    stretches: tuple[float, ...] = DEFAULT_STRETCHES,
    sr: int = 22050,
    n_fft: int = 2048,
    hop: int = 512,
    verbose: bool = True,
    progress: Callable[[str], None] | None = None,
) -> list[dict]:
    """Run the full sweep and return one row per (signal, method, stretch).

    Returns
    -------
    list of dict
        Keys: signal, method, stretch, and the metric columns. ``consistency``
        is absent for time-domain methods, and ``amplitude_warble`` should only
        be read on the ``sine`` signal (see `elastiqa.metrics.amplitude_warble`).
    """
    signals = make_test_signals(sr) if signals is None else signals
    methods = DEFAULT_METHODS if methods is None else methods
    emit = progress or (print if verbose else (lambda _: None))

    rows: list[dict] = []
    total = len(signals) * len(methods) * len(stretches)
    done = 0

    for sig_name, x in signals.items():
        reference_crest = crest_factor(x)

        for method_name, spec in methods.items():
            fn = spec["fn"]
            stft_of = spec.get("stft_of")

            for s in stretches:
                try:
                    y = fn(x, s, n_fft=n_fft, hop=hop)
                except TypeError:
                    y = fn(x, s)          # naive takes no STFT parameters

                row = {
                    "signal": sig_name,
                    "method": method_name,
                    "stretch": float(s),
                    "amplitude_warble": amplitude_warble(y),
                    "crest_factor": crest_factor(y),
                    "crest_loss": reference_crest - crest_factor(y),
                    "crest_error": abs(reference_crest - crest_factor(y)),
                    "spectral_convergence": spectral_convergence(x, y, n_fft, hop),
                    "log_spectral_distance": log_spectral_distance(x, y, n_fft, hop),
                }

                if stft_of is not None:
                    Y = stft_of(x, s, n_fft=n_fft, hop=hop)
                    row["consistency"] = consistency(Y, hop=hop, n_fft=n_fft)

                rows.append(row)
                done += 1
                if done % 10 == 0 or done == total:
                    emit(f"  {done}/{total}")

    return rows


_FIELDS = [
    "signal", "method", "stretch", "consistency", "amplitude_warble",
    "crest_factor", "crest_loss", "crest_error", "spectral_convergence",
    "log_spectral_distance",
]


def to_csv(rows: list[dict], path: str | Path) -> Path:
    """Write results to CSV, one row per benchmark cell."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _fmt(r.get(k, "")) for k in _FIELDS})
    return path


def _fmt(v):
    return f"{v:.5f}" if isinstance(v, float) else v


def to_markdown(
    rows: list[dict],
    metric: str = "log_spectral_distance",
    signal: str | None = None,
    lower_is_better: bool = True,
    exclude: tuple[str, ...] = (),
) -> str:
    """Render one metric as a markdown table: methods down, stretches across.

    The best value in each column is bolded, which is what makes a results
    table readable at a glance.

    Parameters
    ----------
    exclude
        Method names to omit. Usually ``("naive",)``.

    A warning about the naive baseline
    ----------------------------------
    Naive resampling does not preserve pitch, and none of these metrics measure
    pitch. It therefore *wins* several of them outright -- resampling a sine
    yields a flawless sine, so its amplitude warble is near zero even though
    the note has moved by an octave.

    That is not a defect in the metrics; it is a reminder that they assume the
    pitch-preservation constraint is already satisfied and only rank quality
    *within* that class. Keep naive in the CSV as the baseline it is, and
    exclude it from quality rankings.
    """
    sel = [
        r for r in rows
        if (signal is None or r["signal"] == signal) and r["method"] not in exclude
    ]
    if not sel:
        return "_no rows_"

    methods = list(dict.fromkeys(r["method"] for r in sel))
    stretches = sorted({r["stretch"] for r in sel})

    grid: dict[tuple[str, float], float] = {}
    for r in sel:
        if metric in r and r[metric] is not None:
            grid[(r["method"], r["stretch"])] = float(r[metric])

    best: dict[float, float] = {}
    for s in stretches:
        vals = [grid[(m, s)] for m in methods if (m, s) in grid]
        if vals:
            best[s] = min(vals) if lower_is_better else max(vals)

    header = "| method | " + " | ".join(f"{s}x" for s in stretches) + " |"
    divider = "|---|" + "---:|" * len(stretches)
    lines = [header, divider]

    for m in methods:
        cells = []
        for s in stretches:
            if (m, s) not in grid:
                cells.append("—")
                continue
            v = grid[(m, s)]
            text = f"{v:.4f}"
            if s in best and np.isclose(v, best[s]):
                text = f"**{text}**"
            cells.append(text)
        lines.append(f"| {m} | " + " | ".join(cells) + " |")

    return "\n".join(lines)
