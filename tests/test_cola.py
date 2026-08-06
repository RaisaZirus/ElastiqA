"""
Constant overlap-add.

Weighted overlap-add divides by the summed squared window, so it inverts
exactly even when COLA is violated. But COLA still matters: once the vocoder
starts modifying phase, a non-constant envelope turns into audible amplitude
modulation at the frame rate -- a buzz that is very easy to misdiagnose as a
phase problem.
"""

import numpy as np
import pytest

from elastiqa.stft import check_cola, get_window, window_envelope


def test_hann_75_percent_overlap():
    """The canonical phase-vocoder setting. Hann^2 at hop=N/4 sums to 1.5."""
    w = get_window("hann", 1024)
    ok, value, ripple = check_cola(w, 256)
    assert ok, f"ripple {ripple:.3e}"
    assert np.isclose(value, 1.5, rtol=1e-9)


@pytest.mark.parametrize("n_fft", [256, 512, 1024, 2048, 4096])
def test_hann_cola_holds_at_quarter_hop(n_fft):
    ok, _, ripple = check_cola(get_window("hann", n_fft), n_fft // 4)
    assert ok, f"n_fft={n_fft}: ripple {ripple:.3e}"


@pytest.mark.parametrize("divisor", [4, 8, 16])
def test_hann_cola_at_power_of_two_hops(divisor):
    """Hann^2 satisfies COLA at hops of N/2^k for k >= 2."""
    n_fft = 2048
    ok, _, ripple = check_cola(get_window("hann", n_fft), n_fft // divisor)
    assert ok, f"hop=N/{divisor}: ripple {ripple:.3e}"


def test_hann_squared_fails_cola_at_half_hop():
    """50% overlap is NOT enough for weighted overlap-add. This is why.

    Plain Hann sums to a constant at hop=N/2, which is why 50% overlap is fine
    when you window only on analysis. But WOLA windows *twice*, so the relevant
    quantity is Hann^2 -- and Hann^2 has a large ripple at N/2. That ripple
    becomes audible amplitude modulation at the frame rate once the vocoder
    starts modifying phase.

    This is the concrete reason 75% overlap is the standard phase-vocoder
    setting, and it is worth stating explicitly in the report.
    """
    n_fft = 2048
    w = get_window("hann", n_fft)

    ok_squared, _, ripple_squared = check_cola(w, n_fft // 2)
    assert not ok_squared
    assert ripple_squared > 0.1

    # Plain Hann (not squared) does satisfy COLA here, summing to 1.0.
    ok_plain, value_plain, _ = check_cola(np.sqrt(w), n_fft // 2)
    assert ok_plain
    assert np.isclose(value_plain, 1.0, rtol=1e-9)


def test_periodic_vs_symmetric_matters():
    """The symmetric window breaks COLA. This is the classic off-by-one."""
    n_fft, hop = 1024, 256

    _, _, ripple_periodic = check_cola(get_window("hann", n_fft, periodic=True), hop)
    _, _, ripple_symmetric = check_cola(get_window("hann", n_fft, periodic=False), hop)

    assert ripple_periodic < 1e-12
    assert ripple_symmetric > ripple_periodic


def test_envelope_is_nonnegative_and_tapers():
    w = get_window("hann", 512)
    env = window_envelope(w, 128, n_frames=20, out_len=20 * 128 + 512)

    assert np.all(env >= 0)
    assert env[0] < env[len(env) // 2]      # ramps up at the head
    assert env[-1] < env[len(env) // 2]     # ramps down at the tail


def test_rect_window_cola_at_full_hop():
    """A rectangular window with no overlap trivially satisfies COLA."""
    ok, value, _ = check_cola(get_window("rect", 512), 512)
    assert ok
    assert np.isclose(value, 1.0)
