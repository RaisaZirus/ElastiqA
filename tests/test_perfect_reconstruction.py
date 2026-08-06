"""
The gate. If anything in this file fails, stop and fix it before writing a
single line of vocoder code. Every phase-vocoder artefact you will later chase
by ear can also be caused by a broken STFT, and you will not be able to tell
the two apart.

Tolerance is 1e-10, which is loose for float64 round-tripping through an FFT
(actual error is usually ~1e-15). If you find yourself wanting to relax it,
something is wrong.
"""

import numpy as np
import pytest

from elastiqa.stft import get_window, istft, stft

TOL = 1e-10
SR = 22050


def _signals():
    """A spread of inputs: noise, tones, impulses, silence, non-round lengths."""
    rng = np.random.default_rng(0)
    n = 20000
    t = np.arange(n) / SR
    return {
        "white_noise": rng.standard_normal(n) * 0.1,
        "sine_440": 0.5 * np.sin(2 * np.pi * 440 * t),
        # Deliberately between bin centres -- this is where naive phase
        # handling breaks and where the window must be doing real work.
        "sine_offgrid": 0.5 * np.sin(2 * np.pi * 443.7 * t),
        "chirp": 0.5 * np.sin(2 * np.pi * (100 + 2000 * t) * t),
        "impulse_train": _impulse_train(n, period=1000),
        "silence": np.zeros(n),
        "odd_length": rng.standard_normal(19997) * 0.1,
    }


def _impulse_train(n, period):
    x = np.zeros(n)
    x[::period] = 1.0
    return x


@pytest.mark.parametrize("name,x", list(_signals().items()))
@pytest.mark.parametrize("n_fft,hop", [(1024, 256), (2048, 512), (512, 128)])
def test_round_trip_hann(name, x, n_fft, hop):
    """istft(stft(x)) == x for the standard Hann / 75% overlap configuration."""
    X = stft(x, n_fft=n_fft, hop=hop, window="hann")
    y = istft(X, hop=hop, window="hann", length=len(x))

    assert y.shape == x.shape, f"{name}: shape {y.shape} != {x.shape}"
    err = np.max(np.abs(y - x))
    assert err < TOL, f"{name} @ n_fft={n_fft}, hop={hop}: max error {err:.3e}"


@pytest.mark.parametrize("window", ["hann", "hamming", "blackman", "sqrt_hann"])
def test_round_trip_windows(window):
    """Weighted overlap-add should invert exactly for any reasonable window."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(12000) * 0.1

    X = stft(x, n_fft=1024, hop=256, window=window)
    y = istft(X, hop=256, window=window, length=len(x))

    err = np.max(np.abs(y - x))
    assert err < TOL, f"window={window}: max error {err:.3e}"


@pytest.mark.parametrize("hop", [64, 128, 256, 341, 512])
def test_round_trip_various_hops(hop):
    """Including a hop that does not divide n_fft evenly (341)."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal(9000) * 0.1

    X = stft(x, n_fft=1024, hop=hop, window="hann")
    y = istft(X, hop=hop, window="hann", length=len(x))

    err = np.max(np.abs(y - x))
    assert err < TOL, f"hop={hop}: max error {err:.3e}"


def test_edges_are_reconstructed():
    """Reconstruction must hold at the very first and last samples.

    Edge handling is the most common silent failure: interior samples come
    back perfectly while the first and last few thousand are attenuated,
    producing fades that are easy to mistake for a windowing choice.
    """
    rng = np.random.default_rng(3)
    x = rng.standard_normal(8000) * 0.1

    X = stft(x, n_fft=1024, hop=256)
    y = istft(X, hop=256, length=len(x))

    head_err = np.max(np.abs(y[:512] - x[:512]))
    tail_err = np.max(np.abs(y[-512:] - x[-512:]))
    assert head_err < TOL, f"head error {head_err:.3e}"
    assert tail_err < TOL, f"tail error {tail_err:.3e}"


def test_short_signal():
    """Signals shorter than one frame must not crash or silently truncate."""
    x = np.sin(2 * np.pi * 440 * np.arange(300) / SR)
    X = stft(x, n_fft=1024, hop=256)
    y = istft(X, hop=256, length=len(x))
    assert y.shape == x.shape
    assert np.max(np.abs(y - x)) < TOL


def test_stft_shape():
    """Bin count is n_fft//2 + 1; frames advance by hop."""
    x = np.zeros(10000)
    X = stft(x, n_fft=2048, hop=512)
    assert X.shape[0] == 2048 // 2 + 1
    assert X.dtype == np.complex128


def test_linearity():
    """The STFT is linear -- a property later modules quietly rely on."""
    rng = np.random.default_rng(4)
    a, b = rng.standard_normal(5000), rng.standard_normal(5000)

    Xa, Xb = stft(a, 1024, 256), stft(b, 1024, 256)
    Xsum = stft(3.0 * a - 2.0 * b, 1024, 256)

    assert np.max(np.abs(Xsum - (3.0 * Xa - 2.0 * Xb))) < TOL


def test_parseval_energy():
    """Frame energy should match between time and frequency domains."""
    rng = np.random.default_rng(5)
    x = rng.standard_normal(4096)
    w = get_window("hann", 1024)

    frame = x[:1024] * w
    spec = np.fft.rfft(frame)

    time_energy = np.sum(frame ** 2)
    # rFFT drops the negative frequencies; DC and Nyquist are not mirrored.
    weights = np.ones(len(spec)) * 2.0
    weights[0] = weights[-1] = 1.0
    freq_energy = np.sum(weights * np.abs(spec) ** 2) / 1024

    assert np.isclose(time_energy, freq_energy, rtol=1e-9)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        stft(np.zeros((100, 2)), 1024, 256)          # stereo
    with pytest.raises(ValueError):
        stft(np.zeros(1000), 1024, hop=0)            # zero hop
    with pytest.raises(ValueError):
        stft(np.zeros(1000), 1024, hop=2048)         # hop > n_fft
    with pytest.raises(ValueError):
        istft(np.zeros(100), hop=256)                # 1-D input
