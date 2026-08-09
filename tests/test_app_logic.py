"""
Web interface logic.

These cover everything the app does apart from drawing widgets, which is why
``app_logic`` deliberately imports no Gradio. A failing test here means a wrong
number; a bug that survives all of these is a layout problem.

The input-handling tests matter more than they look. Gradio hands over whatever
the browser produced -- int16 or float, mono or stereo, six seconds or six
minutes -- and every one of those variants has to arrive as mono float64 in
[-1, 1] or the DSP downstream produces garbage.
"""

import numpy as np
import pytest

from elastiqa import app_logic as al

SR = 22050


def _tone(duration=1.5, f0=220.0, n=12, sr=SR):
    t = np.arange(int(duration * sr)) / sr
    x = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, n + 1))
    return x / np.max(np.abs(x))


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------

def test_accepts_int16_mono():
    raw = (_tone() * 20000).astype(np.int16)
    x, sr, note = al.prepare_audio((SR, raw))

    assert x.dtype == np.float64 and x.ndim == 1
    assert sr == SR and note == ""
    assert np.max(np.abs(x)) <= 1.0


def test_downmixes_stereo():
    raw = (np.stack([_tone(), _tone(f0=330)], axis=1) * 20000).astype(np.int16)
    x, _, _ = al.prepare_audio((SR, raw))
    assert x.ndim == 1


def test_accepts_float_input():
    x, _, _ = al.prepare_audio((SR, _tone()))
    assert x.dtype == np.float64


def test_truncates_long_clips_and_says_so():
    x, sr, note = al.prepare_audio((SR, _tone(duration=40.0)), max_seconds=15.0)
    assert abs(len(x) / sr - 15.0) < 0.01
    assert "15" in note


def test_short_clip_passes_without_a_note():
    _, _, note = al.prepare_audio((SR, _tone(duration=2.0)))
    assert note == ""


@pytest.mark.parametrize("bad,fragment", [
    (None, "Upload"),
    ((SR, np.zeros(SR)), "silent"),
    ((SR, np.zeros(100)), "too short"),
])
def test_rejects_unusable_input_with_a_useful_message(bad, fragment):
    with pytest.raises(ValueError) as info:
        al.prepare_audio(bad)
    assert fragment.lower() in str(info.value).lower()


def test_output_packaging_is_int16_and_unclipped():
    sr, pcm = al.to_gradio_audio(_tone() * 8.0, SR)   # deliberately too loud
    assert sr == SR and pcm.dtype == np.int16
    assert np.abs(pcm).max() <= 32767


# --------------------------------------------------------------------------
# Time stretching
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", list(al.STRETCH_METHODS))
def test_every_stretch_method_runs(method):
    x = _tone()
    y, metrics = al.run_time_stretch(x, SR, 1.5, method)

    assert np.all(np.isfinite(y))
    assert abs(len(y) - len(x) * 1.5) / len(x) < 0.02
    assert "output length" in metrics


def test_only_vocoder_methods_report_consistency():
    x = _tone()
    for label, key in al.STRETCH_METHODS.items():
        _, metrics = al.run_time_stretch(x, SR, 1.5, label)
        assert ("consistency (D_M)" in metrics) == key.startswith("pv")


def test_locking_scores_better_than_broken_phase_in_the_app():
    """The app must surface the same ordering the benchmark does."""
    x = _tone()
    _, good = al.run_time_stretch(x, SR, 1.5, "Phase vocoder + locking (best)")
    _, bad = al.run_time_stretch(x, SR, 1.5, "Phase vocoder with broken phase")

    assert float(good["consistency (D_M)"]) < float(bad["consistency (D_M)"])


def test_unknown_stretch_method_raises():
    with pytest.raises(ValueError):
        al.run_time_stretch(_tone(), SR, 1.5, "Telepathy")


# --------------------------------------------------------------------------
# Pitch shifting
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", list(al.PITCH_METHODS))
def test_every_pitch_method_runs(method):
    y, metrics = al.run_pitch_shift(_tone(), SR, -7, method)
    assert np.all(np.isfinite(y))
    assert "requested shift" in metrics


def test_measured_shift_matches_the_request():
    """The readout must not lie about what happened."""
    for st in (-7, -3, 5, 12):
        _, metrics = al.run_pitch_shift(
            _tone(), SR, st, "Formant-preserving (natural)"
        )
        measured = float(metrics["measured shift"].split()[0])
        assert abs(measured - st) < 0.5, f"asked {st}, measured {measured}"


def test_pitch_methods_preserve_length_except_naive():
    x = _tone()
    for label, key in al.PITCH_METHODS.items():
        y, _ = al.run_pitch_shift(x, SR, -7, label)
        if key == "naive":
            assert len(y) != len(x)      # the documented flaw
        else:
            assert len(y) == len(x)


def test_unknown_pitch_method_raises():
    with pytest.raises(ValueError):
        al.run_pitch_shift(_tone(), SR, 1.0, "Vibes")


# --------------------------------------------------------------------------
# Effects
# --------------------------------------------------------------------------

@pytest.mark.parametrize("effect", list(al.EFFECTS))
def test_every_effect_runs(effect):
    x = _tone()
    y, metrics = al.run_effect(x, SR, effect)
    assert np.all(np.isfinite(y))
    assert len(y) == len(x)
    assert metrics


def test_autotune_readout_shows_improvement():
    flat = _tone(f0=440.0 * 2 ** (-0.40 / 12))
    _, metrics = al.run_effect(flat, SR, "Auto-tune")

    before, after = (
        float(v) for v in metrics["tuning error"].replace("cents", "").split("->")
    )
    assert after < before


def test_unknown_effect_raises():
    with pytest.raises(ValueError):
        al.run_effect(_tone(), SR, "Reverb")


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------

def test_demo_signals_are_ready_to_play():
    demos = al.demo_signals(SR, duration=1.0)
    assert set(demos) >= {
        "original", "naive", "vocoder", "broken", "pitch_plain", "pitch_formant"
    }
    for name, sig in demos.items():
        assert np.all(np.isfinite(sig)), name
        assert len(sig) > 0, name


def test_demo_pitch_pair_has_matching_lengths():
    """The A/B on the landing tab only reads as a fair comparison if the two
    clips are the same length."""
    demos = al.demo_signals(SR, duration=1.0)
    assert len(demos["pitch_plain"]) == len(demos["pitch_formant"])


def test_metrics_render_as_a_table():
    out = al.format_metrics({"pitch": "220 Hz", "length": "1.5 s"})
    assert out.startswith("| measurement | value |")
    assert "`220 Hz`" in out


def test_empty_metrics_render_as_nothing():
    assert al.format_metrics({}) == ""


def test_spectrogram_image_is_an_rgb_array():
    """Gradio's image component takes an ndarray, a PIL image, or a path.

    It rejects raw PNG bytes, and it does so during output postprocessing --
    after the DSP has already run, so the failure surfaces as a broken result
    panel rather than as an obvious error. Assert the exact shape and dtype.
    """
    x = _tone(duration=0.6)
    img = al.spectrogram_image({"a": x, "b": x[::-1]}, SR)

    assert isinstance(img, np.ndarray)
    assert img.dtype == np.uint8
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.shape[0] > 100 and img.shape[1] > 100


def test_spectrogram_png_is_still_available_for_saving():
    x = _tone(duration=0.6)
    data = al.spectrogram_png({"a": x}, SR)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(data) > 1000


def test_audio_output_is_a_sample_rate_and_int16_pair():
    """The shape Gradio's audio component expects on output."""
    out = al.to_gradio_audio(_tone(duration=0.5), SR)

    assert isinstance(out, tuple) and len(out) == 2
    sr, data = out
    assert isinstance(sr, int)
    assert isinstance(data, np.ndarray) and data.dtype == np.int16
    assert data.ndim == 1


def test_metrics_output_is_a_string():
    """Markdown components take str, not dict."""
    _, metrics = al.run_time_stretch(_tone(), SR, 1.5, list(al.STRETCH_METHODS)[0])
    assert isinstance(al.format_metrics(metrics), str)


# --------------------------------------------------------------------------
# File decoding
# --------------------------------------------------------------------------

def test_reads_a_wav_path_without_ffmpeg():
    """WAV must work on a machine with no ffmpeg installed.

    Gradio decodes non-WAV uploads by shelling out to ffprobe, and that failure
    happens during input preprocessing -- before any callback runs, so it
    cannot be caught downstream. Decoding paths ourselves keeps the common case
    working regardless.
    """
    import os
    import tempfile

    from scipy.io import wavfile

    raw = (_tone(duration=1.0) * 20000).astype(np.int16)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clip.wav")
        wavfile.write(path, SR, raw)

        x, sr, note = al.prepare_audio(path)

    assert sr == SR
    assert x.dtype == np.float64 and x.ndim == 1
    assert 0.9 < np.max(np.abs(x)) <= 1.0


def test_compressed_formats_explain_the_ffmpeg_requirement():
    """An unreadable M4A should say what to do, not raise a subprocess error."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clip.m4a")
        with open(path, "wb") as fh:
            fh.write(b"not really audio")

        with pytest.raises(ValueError) as info:
            al.prepare_audio(path)

    message = str(info.value).lower()
    assert "ffmpeg" in message
    assert "wav" in message


def test_stereo_wav_path_is_downmixed():
    import os
    import tempfile

    from scipy.io import wavfile

    stereo = np.stack([_tone(1.0), _tone(1.0, f0=330.0)], axis=1)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "stereo.wav")
        wavfile.write(path, SR, (stereo * 20000).astype(np.int16))
        x, _, _ = al.prepare_audio(path)

    assert x.ndim == 1
