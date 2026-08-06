"""
Transient detection and phase reset.

The headline is ``test_phase_reset_preserves_attacks``: transient handling
should recover most of the crest factor that plain vocoding loses on
percussive material. Note that the metric is only meaningful relative to the
unprocessed original, which is why every assertion here is a comparison rather
than an absolute threshold.
"""

import numpy as np
import pytest

from elastiqa.metrics import crest_factor, ser
from elastiqa.stft import stft
from elastiqa.tsm import (
    detect_onsets,
    spectral_flux,
    time_stretch_locked,
    time_stretch_transient,
)

SR = 22050


def _drums(n_hits=8, duration=3.0, spacing=0.35, seed=0):
    """Synthetic percussion: decaying low body plus a noise burst."""
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

    return x / np.max(np.abs(x))


def _tone(duration=2.0, f0=220.0, fade=0.05):
    """A steady tone with fades.

    The fades are not cosmetic. A sinusoid that starts or stops abruptly is
    genuinely discontinuous, and any honest onset detector will fire on it --
    see ``test_abrupt_cutoff_does_register``. Without fades this would not be
    a test of "steady tone", it would be a test of two clicks.
    """
    t = np.arange(int(duration * SR)) / SR
    x = np.sin(2 * np.pi * f0 * t)

    n = int(fade * SR)
    ramp = np.linspace(0.0, 1.0, n)
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x


# --------------------------------------------------------------------------
# Spectral flux
# --------------------------------------------------------------------------

def test_flux_shape_and_range():
    flux = spectral_flux(stft(_drums(), 2048, 512))
    assert flux[0] == 0.0
    assert np.all(flux >= 0.0)
    assert np.isclose(flux.max(), 1.0)


def test_flux_is_near_zero_for_steady_tone():
    """A sustained tone has no onsets after its own attack."""
    flux = spectral_flux(stft(_tone(), 2048, 512))
    assert np.median(flux) < 0.05


def test_flux_spikes_at_hits():
    flux = spectral_flux(stft(_drums(n_hits=4, spacing=0.5), 2048, 512))
    assert flux.max() > 10 * np.median(flux)


def test_flux_ignores_gradual_decay():
    """Rectification means a note *fading out* must not register as an onset."""
    t = np.arange(2 * SR) / SR
    x = np.sin(2 * np.pi * 300 * t) * np.exp(-t * 2.0)   # smooth decay

    n = int(0.05 * SR)
    x[:n] *= np.linspace(0.0, 1.0, n)

    flux = spectral_flux(stft(x, 2048, 512))
    frames = len(flux)

    attack = flux[: frames // 8].max()
    decay = flux[frames // 4 :].max()
    assert decay < attack / 2.0, f"decay {decay:.3f} vs attack {attack:.3f}"


def test_abrupt_cutoff_does_register():
    """Documenting a real property rather than pretending it away.

    Rectified flux is often described as ignoring note endings. That is only
    true of *gradual* endings. Truncating a sinusoid mid-cycle is a genuine
    discontinuity, and a discontinuity splatters energy across every bin at
    once -- so almost every bin registers an increase and the flux spikes,
    often higher than at the original attack.

    This is not a defect. An abrupt cutoff is a transient, and resetting phase
    there is the right thing to do. But it does mean flux measures
    *discontinuity*, not *energy increase*, and a report that claims otherwise
    is overstating the method.
    """
    x = np.zeros(2 * SR)
    t = np.arange(SR) / SR
    x[:SR] = np.sin(2 * np.pi * 300 * t)      # stops dead at 1 s

    flux = spectral_flux(stft(x, 2048, 512))
    cutoff_frame = SR // 512

    near_cutoff = flux[cutoff_frame - 3 : cutoff_frame + 3].max()
    assert near_cutoff > 0.5, "flux should spike at the discontinuity"

    # ... but the detector's energy-rise gate filters it out, because a sound
    # that is ending is not an attack and resetting phase there buys nothing.
    onsets = detect_onsets(stft(x, 2048, 512))
    assert not any(abs(o - cutoff_frame) <= 2 for o in onsets), (
        f"cutoff at frame {cutoff_frame} leaked through as an onset: {onsets}"
    )


# --------------------------------------------------------------------------
# Onset detection
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_hits", [4, 6, 8])
def test_detects_the_right_number_of_onsets(n_hits):
    onsets = detect_onsets(stft(_drums(n_hits=n_hits), 2048, 512))
    assert abs(len(onsets) - n_hits) <= 1, f"found {len(onsets)}, planted {n_hits}"


def test_onsets_land_near_the_hits():
    """Detected frames should map back to the planted times within ~50 ms."""
    hop = 512
    onsets = detect_onsets(stft(_drums(n_hits=6, spacing=0.5), 2048, hop))
    times = onsets * hop / SR
    expected = [0.15 + i * 0.5 for i in range(6)]

    for want in expected:
        assert min(abs(times - want)) < 0.05, f"no onset near {want:.2f}s"


def test_no_onsets_in_steady_tone():
    """A faded-in, faded-out tone has at most its own attack."""
    assert len(detect_onsets(stft(_tone(), 2048, 512))) <= 1


def test_no_onsets_in_silence():
    assert len(detect_onsets(stft(np.zeros(20000), 2048, 512))) == 0


def test_higher_threshold_detects_fewer():
    X = stft(_drums(), 2048, 512)
    assert len(detect_onsets(X, threshold=3.0)) <= len(detect_onsets(X, threshold=1.2))


def test_min_separation_suppresses_clusters():
    X = stft(_drums(), 2048, 512)
    assert len(detect_onsets(X, min_separation=20)) <= len(
        detect_onsets(X, min_separation=1)
    )


# --------------------------------------------------------------------------
# Phase reset
# --------------------------------------------------------------------------

@pytest.mark.parametrize("stretch", [0.75, 1.5, 2.0])
def test_transient_output_length(stretch):
    x = _drums()
    y = time_stretch_transient(x, stretch)
    assert abs(len(y) - int(round(len(x) * stretch))) <= 1


def test_transient_identity_is_transparent():
    x = _drums()
    assert ser(x, time_stretch_transient(x, 1.0)) > 30.0


def test_phase_reset_preserves_attacks():
    """THE payoff.

    Plain vocoding smears percussive attacks, dropping the crest factor.
    Resetting phase at onsets should recover most of that loss.
    """
    x = _drums()

    c_orig = crest_factor(x)
    c_plain = crest_factor(time_stretch_locked(x, 1.5))
    c_transient = crest_factor(time_stretch_transient(x, 1.5))

    lost_plain = c_orig - c_plain
    lost_transient = c_orig - c_transient

    assert lost_transient < lost_plain / 2.0, (
        f"original {c_orig:.2f}, plain {c_plain:.2f}, "
        f"transient {c_transient:.2f} -- reset recovered too little"
    )


def test_transient_handling_does_not_hurt_tonal_material():
    """Transient handling must not degrade material that has no transients.

    Note what is *not* asserted here: sample-level similarity. A single phase
    reset at the tone's own attack offsets the whole downstream trajectory by
    a constant per bin, which drops SER to around 8 dB even though the two
    outputs are perceptually indistinguishable.

    That is a general warning about SER in this project. It is a valid identity
    check against an unprocessed original, and close to useless for comparing
    two vocoder variants against each other, because it punishes phase offsets
    that no listener can hear. Compare perceptual quantities instead.
    """
    from elastiqa.metrics import amplitude_warble, log_spectral_distance

    x = _tone()
    a = time_stretch_locked(x, 1.5)
    b = time_stretch_transient(x, 1.5)

    assert abs(amplitude_warble(a) - amplitude_warble(b)) < 0.005
    assert abs(log_spectral_distance(x, a) - log_spectral_distance(x, b)) < 1.0


def test_returns_onsets_when_asked():
    x = _drums()
    y, onsets = time_stretch_transient(x, 1.5, return_onsets=True)
    assert len(y) > 0
    assert len(onsets) >= 4


def test_lock_flag_isolates_the_two_effects():
    """The ablation: transient handling with and without phase locking."""
    x = _drums()
    with_lock = time_stretch_transient(x, 1.5, lock=True)
    without_lock = time_stretch_transient(x, 1.5, lock=False)

    assert len(with_lock) == len(without_lock)
    assert not np.allclose(with_lock, without_lock)


def test_rejects_bad_stretch():
    with pytest.raises(ValueError):
        time_stretch_transient(_drums(duration=0.5), 0.0)
