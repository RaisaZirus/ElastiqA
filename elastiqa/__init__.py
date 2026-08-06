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
from .metrics import (
    amplitude_warble,
    consistency,
    crest_factor,
    evaluate,
    log_spectral_distance,
    ser,
    spectral_convergence,
)
from .pitch import (
    cepstral_envelope,
    formant_correct,
    pitch_shift,
    pitch_shift_formant,
)
from .stft import (
    bin_frequencies,
    check_cola,
    frame_times,
    get_window,
    istft,
    stft,
    window_envelope,
)

from .tsm import (
    PHASE_MODES,
    find_peaks,
    locked_phase_vocoder_stft,
    ola,
    phase_vocoder_stft,
    regions_of_influence,
    time_stretch,
    time_stretch_locked,
    time_stretch_transient,
    detect_onsets,
    spectral_flux,
    wsola,
)
from . import viz

__version__ = "0.4.0"

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
    # tsm
    "time_stretch",
    "phase_vocoder_stft",
    "ola",
    "wsola",
    "time_stretch_locked",
    "locked_phase_vocoder_stft",
    "find_peaks",
    "regions_of_influence",
    "time_stretch_transient",
    "detect_onsets",
    "spectral_flux",
    "pitch_shift_formant",
    "formant_correct",
    "cepstral_envelope",
    "PHASE_MODES",
    "pitch_shift",
    # metrics
    "consistency",
    "ser",
    "spectral_convergence",
    "log_spectral_distance",
    "amplitude_warble",
    "crest_factor",
    "evaluate",
    # io
    "load",
    "save",
    "to_mono",
    "normalise",
    "have_soundfile",
    "__version__",
]
