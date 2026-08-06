"""
WSOLA: Waveform Similarity Overlap-Add (Verhelst & Roelands 1993).

The frequency-domain answer to OLA's phase problem is to reason about
instantaneous frequency. The time-domain answer is simpler and, on some
material, better: *search* for a frame that happens to line up.

The idea
--------
After writing a frame taken from input position ``p``, the waveform that would
naturally follow is ``x[p + Hs : p + Hs + N]``. Call that the *natural
continuation*. The next frame ideally starts at analysis position ``a``, but
there is no reason ``x[a:a+N]`` should line up in phase with the natural
continuation.

So do not insist on ``a``. Search a small window ``a ± tolerance`` for the
offset whose waveform is most similar to the natural continuation, measured by
normalised cross-correlation, and take that frame instead. The timing error is
at most a few milliseconds -- inaudible -- while the phase alignment is
dramatically better.

Trade-offs against the phase vocoder
------------------------------------
WSOLA has no notion of frequency at all, which is both its weakness and its
strength. On a single voice or a monophonic instrument it is excellent and
often preferred, because it never smears transients. On polyphonic music it
struggles: there is no single offset that aligns several independent sources at
once, so it picks a compromise and introduces stuttering or transient
duplication.

Having both algorithms in the benchmark is what makes the comparison
interesting -- neither wins everywhere.
"""

from __future__ import annotations

import numpy as np

from ..stft import get_window, window_envelope

__all__ = ["wsola"]

_EPS = 1e-12


def _best_offset(
    x: np.ndarray, centre: int, target: np.ndarray, tolerance: int, n_fft: int
) -> int:
    """Offset within +/- tolerance whose frame best matches ``target``.

    Uses normalised cross-correlation so that loud regions do not automatically
    win over quiet ones -- we want similarity of *shape*, not of energy.
    """
    lo = max(centre - tolerance, 0)
    hi = min(centre + tolerance, len(x) - n_fft)
    if hi <= lo:
        return int(np.clip(centre, 0, max(len(x) - n_fft, 0)))

    target_norm = float(np.linalg.norm(target))
    if target_norm < _EPS:
        return centre

    best_score, best = -np.inf, centre
    for pos in range(lo, hi + 1):
        candidate = x[pos : pos + n_fft]
        denom = float(np.linalg.norm(candidate)) * target_norm
        if denom < _EPS:
            continue
        score = float(np.dot(candidate, target)) / denom
        if score > best_score:
            best_score, best = score, pos

    return best


def wsola(
    x: np.ndarray,
    stretch: float,
    n_fft: int = 2048,
    hop: int | None = None,
    window: str = "hann",
    tolerance: int | None = None,
) -> np.ndarray:
    """Time-scale by waveform-similarity overlap-add.

    Parameters
    ----------
    x
        Real 1-D signal.
    stretch
        Duration multiplier.
    n_fft
        Frame length in samples.
    hop
        Synthesis hop. Analysis hop is ``hop / stretch``.
    tolerance
        Half-width of the similarity search, in samples. Defaults to
        ``hop // 2``. Larger values align better but distort timing more;
        the search is also the algorithm's entire computational cost, so this
        parameter trades quality against speed directly.

    Notes
    -----
    The search is a plain loop over candidate offsets. It is O(tolerance x
    n_fft) per frame, which is slow in pure Python but transparent. An FFT-based
    cross-correlation would be far faster and is a reasonable optimisation to
    mention in the report.
    """
    if stretch <= 0:
        raise ValueError(f"stretch must be positive; got {stretch}")

    x = np.asarray(x, dtype=np.float64)
    hop_s = n_fft // 4 if hop is None else hop
    hop_a = hop_s / stretch
    tolerance = hop_s // 2 if tolerance is None else int(tolerance)

    w = get_window(window, n_fft)

    # Pad so the similarity search never runs off either end.
    pad = n_fft + tolerance + hop_s
    xp = np.pad(x, (tolerance, pad), mode="constant")

    n_frames = max(int(np.floor((len(x) - n_fft) / hop_a)) + 1, 1)
    out_len = (n_frames - 1) * hop_s + n_fft
    y = np.zeros(out_len, dtype=np.float64)

    prev = 0  # input position the previous frame was taken from
    for m in range(n_frames):
        ideal = int(round(m * hop_a)) + tolerance

        if m == 0:
            pos = ideal
        else:
            # What would naturally have come next after the previous frame.
            natural = xp[prev + hop_s : prev + hop_s + n_fft]
            if len(natural) < n_fft:
                natural = np.pad(natural, (0, n_fft - len(natural)))
            pos = _best_offset(xp, ideal, natural, tolerance, n_fft)

        frame = xp[pos : pos + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))

        y[m * hop_s : m * hop_s + n_fft] += frame * w * w
        prev = pos

    env = window_envelope(w, hop_s, n_frames, out_len)
    nonzero = env > _EPS
    y[nonzero] /= env[nonzero]

    target = int(round(len(x) * stretch))
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    return y[:target]
