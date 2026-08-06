# ElastiqA

**Phase-coherent time stretching and pitch shifting in Python — with a reproducible benchmark of the algorithms that do it.**

[![tests](https://github.com/RaisaZirus/ElastiqA/actions/workflows/tests.yml/badge.svg)](https://github.com/RaisaZirus/ElastiqA/elastiqa/actions)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

Speeding audio up makes it higher. Slowing it down makes it lower. Breaking that
coupling — changing duration without changing pitch, or pitch without changing
duration — requires working in the short-time Fourier domain and being very
careful with phase.

ElastiqA implements the classical algorithms that do this, from the naive
baseline through to phase-locked vocoding with formant preservation, and
measures all of them against each other on the same material.

> Unaffiliated with zplane's élastique. The name is a nod, not a claim.

---

## Status

Week 1 of 6 — foundation complete.

| Module | Status |
|---|---|
| Exactly invertible STFT (weighted overlap-add) | ✅ |
| Phase arithmetic + instantaneous frequency | ✅ |
| Naive resampling baseline | ✅ |
| Standard phase vocoder | 🔜 week 2 |
| Identity phase locking (Laroche & Dolson) | 🔜 week 3 |
| WSOLA time-domain baseline | 🔜 week 3 |
| Transient detection + phase reset | 🔜 week 4 |
| Formant-preserving pitch shift | 🔜 week 4 |
| Auto-tune / harmonizer / robot / whisper | 🔜 week 5 |
| Benchmark harness + metrics | 🔜 week 5 |
| Web demo | 🔜 week 6 |

---

## Install

```bash
git clone https://github.com/RaisaZirus/ElastiqA
cd elastiqa
pip install -e ".[dev]"
```

Core requires only NumPy and SciPy. `soundfile` (better format support),
`matplotlib` (plots), and `gradio` (web demo) are optional extras.

## Use

```python
import elastiqa as eq

x, sr = eq.load("voice.wav")

X = eq.stft(x, n_fft=2048, hop=512)     # (1025, n_frames) complex
y = eq.istft(X, hop=512, length=len(x))

print(abs(y - x).max())                  # ~2e-16
```

Instantaneous frequency, the quantity the phase vocoder is built on:

```python
ifreq = eq.instantaneous_frequency(X, hop=512)   # radians/sample
hz = ifreq * sr / (2 * np.pi)
```

## Reproduce the week 1 figures

```bash
python scripts/week1_demo.py
```

Writes overlap-add envelopes, a reconstruction error plot, and spectrograms of
the naive baseline showing formants sliding out of place.

---

## Why perfect reconstruction comes first

Every artefact a phase vocoder can produce — metallic phasiness, smearing,
frame-rate buzz — can *also* be produced by a subtly broken STFT. If the
identity path is not exact, you cannot tell which of the two you are hearing,
and you will spend days debugging the wrong module.

So the test suite gates on it:

```bash
pytest tests/test_perfect_reconstruction.py
```

`istft(stft(x)) == x` to within **1e-10** across noise, tones, chirps, impulse
trains, silence, odd-length signals, four window types, and hops that do not
divide the FFT size evenly. Measured error is ~2e-16.

Two design choices behind that:

**Weighted overlap-add.** The window is applied on analysis *and* synthesis,
then the result is divided by the summed squared window. This inverts exactly
for any window/hop combination rather than only for strict-COLA hops, and the
synthesis windowing suppresses the frame-boundary discontinuities that appear
once phase is modified.

**Hann at 75% overlap.** Plain Hann satisfies COLA at 50%, but weighted
overlap-add squares the window, and Hann² at 50% has a ripple of 0.5 — audible
amplitude modulation at the frame rate. `tests/test_cola.py` asserts both
facts, since the failure is the more instructive one.

## Why phase arithmetic is separated out

`np.angle` returns values in (−π, π]. A sinusoid between two bin centres
advances more than π radians per hop, so the raw phase difference is ambiguous.
The fix is to subtract the phase advance a bin-centred sinusoid *would* have,
wrap the small remainder, and add it back:

```
expected[k] = 2πk·H/N
deviation   = princarg(Δφ − expected)
ω_true[k]   = (expected + deviation) / H
```

`tests/test_phase.py::test_instantaneous_frequency_offgrid` places a tone
deliberately 0.37 bins off centre and asserts the estimate lands within 0.5 Hz
while the bin-centre answer is off by several. That single test is the
difference between a working vocoder and one that is quietly detuned.

---

## Layout

```
elastiqa/
  stft.py        analysis/synthesis, COLA, window envelopes
  phase.py       princarg, instantaneous frequency
  naive.py       resampling baseline
  audio_io.py    load/save with soundfile or scipy fallback
  viz.py         spectrograms, waveforms, envelope plots
tests/           73 tests, all green
scripts/         reproducible figure generation
```

## References

- Flanagan & Golden (1966), *Phase Vocoder*, Bell System Technical Journal
- Portnoff (1976), *Implementation of the digital phase vocoder using the FFT*
- Puckette (1995), *Phase-locked vocoder*, IEEE ASSP Workshop
- Laroche & Dolson (1997), *Phase-vocoder: about this phasiness business*
- Laroche & Dolson (1999), *Improved phase vocoder time-scale modification of audio*, IEEE Trans. Speech and Audio Processing 7(3)
- Driedger & Müller (2016), *A review of time-scale modification of music signals*, Applied Sciences 6(2)
- Zölzer (2011), *DAFX: Digital Audio Effects*, Wiley

## License

MIT
