"""
ElastiqA -- phase-coherent time stretching and pitch shifting.

An educational phase vocoder implementation and time-scale modification
benchmark. Unaffiliated with zplane's elastique.

Week 1 provides the foundation: an exactly invertible STFT, correct phase
arithmetic, and the naive resampling baseline. Later weeks build the vocoder,
phase locking, transient handling, and formant preservation on top.

Quick start
-----------
    import elastiqa as eq

    x, sr = eq.load("voice.wav")
    X = eq.stft(x, n_fft=2048, hop=512)
    y = eq.istft(X, hop=512, length=len(x))   # y == x to ~1e-15
"""

from .audio_io import have_soundfile, load, normalise, save, to_mono
from .naive import ratio_to_semitones, resample, semitones_to_ratio, speed_change
from .phase import expected_phase_advance, instantaneous_frequency, princarg
from .stft import (
    bin_frequencies,
    check_cola,
    frame_times,
    get_window,
    istft,
    stft,
    window_envelope,
)

__version__ = "0.1.0"

__all__ = [
    # stft
    "stft",
    "istft",
    "get_window",
    "check_cola",
    "window_envelope",
    "frame_times",
    "bin_frequencies",
    # phase
    "princarg",
    "expected_phase_advance",
    "instantaneous_frequency",
    # naive baseline
    "resample",
    "speed_change",
    "semitones_to_ratio",
    "ratio_to_semitones",
    # io
    "load",
    "save",
    "to_mono",
    "normalise",
    "have_soundfile",
    "__version__",
]
from . import viz
