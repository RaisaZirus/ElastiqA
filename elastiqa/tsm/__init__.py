"""Time-scale modification algorithms."""

from .ola import ola
from .phase_lock import (
    find_peaks,
    locked_phase_vocoder_stft,
    regions_of_influence,
    time_stretch_locked,
)
from .transient import detect_onsets, spectral_flux, time_stretch_transient
from .vocoder import PHASE_MODES, phase_vocoder_stft, time_stretch
from .wsola import wsola

__all__ = [
    "ola", "wsola", "time_stretch", "time_stretch_locked",
    "time_stretch_transient", "phase_vocoder_stft",
    "locked_phase_vocoder_stft", "find_peaks", "regions_of_influence",
    "spectral_flux", "detect_onsets", "PHASE_MODES",
]
