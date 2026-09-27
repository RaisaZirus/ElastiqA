# ElastiqA

**Phase-coherent time stretching and pitch shifting in Python — with a reproducible benchmark of the algorithms that do it.**

[![tests](https://github.com/RaisaZirus/ElastiqA/actions/workflows/tests.yml/badge.svg)](https://github.com/RaisaZirus/ElastiqA/actions)
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

**Complete.** Six weeks, 274 tests, 180 benchmark measurements, live web demo.

| Module | Status |
|---|---|
| Exactly invertible STFT (weighted overlap-add) | ✅ |
| Phase arithmetic + instantaneous frequency | ✅ |
| Naive resampling baseline | ✅ |
| Standard phase vocoder | ✅ |
| OLA time-domain baseline | ✅ |
| Consistency measure $D_M$ + spectral metrics | ✅ |
| Pitch shifting (stretch + resample) | ✅ |
| Identity phase locking (Laroche & Dolson) | ✅ |
| WSOLA (similarity search) | ✅ |
| Transient detection + phase reset | ✅ |
| Formant-preserving pitch shift | ✅ |
| YIN pitch detection | ✅ |
| Spectral pitch shift (time-varying ratio) | ✅ |
| Auto-tune / harmoniser / robot / whisper | ✅ |
| Full benchmark harness | ✅ |
| Web demo (Gradio + Hugging Face Spaces) | ✅ |

---

## Try it

**[Live demo on Hugging Face Spaces](https://huggingface.co/spaces/RaisaZirus/ElastiqA)** — upload a
clip or record from the microphone, and hear the difference between a working
phase vocoder and a broken one.

Or run it locally:

```bash
pip install -e ".[app]"
python app.py
```

## Install

```bash
git clone https://github.com/RaisaZirus/ElastiqA
cd ElastiqA
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

Time stretching and pitch shifting:

```python
slow = eq.time_stretch(x, 1.5)      # 50% longer, same pitch
down = eq.pitch_shift(x, -7)        # a fifth lower, same length

bad  = eq.time_stretch(x, 1.5, mode="passthrough")   # phase reused: phasey
robot = eq.time_stretch(x, 1.0, mode="zero")         # phase zeroed
```

## Reproduce the figures

```bash
python scripts/week1_demo.py    # foundation: COLA, reconstruction, the problem
python scripts/week2_demo.py    # the vocoder, phase modes, metrics
```

## Results so far

### The benchmark

Six methods × five stretch factors × six content classes = 180 rows, in
`results/benchmark.csv`. Log-spectral distance on mixed content (tone plus
drums), lower is better:

| method | 0.5× | 0.75× | 1.25× | 1.5× | 2.0× |
|---|---:|---:|---:|---:|---:|
| ola | 36.35 | 17.69 | 28.10 | 32.99 | 41.41 |
| wsola | **36.11** | **16.60** | 25.74 | 31.64 | 39.03 |
| pv | 46.03 | 34.82 | 25.70 | 23.45 | 22.29 |
| pv_locked | 40.80 | 28.03 | **19.93** | **18.25** | **15.66** |
| pv_transient | 39.71 | 26.70 | 20.12 | 18.51 | 15.71 |

**There is a crossover.** WSOLA wins under compression; the phase-locked
vocoder wins under expansion, by an increasing margin. Neither method is
simply better, which is exactly why the benchmark spans several content
classes rather than one — a single-signal comparison would have hidden this.

On percussive material, transient reset wins at three of five stretch factors.

### Auto-tune

A four-note melody sung with a planted detune of −45, +38, −30, +25 cents:

| | mean absolute error |
|---|---:|
| input | 34.6 cents |
| strength = 0.5 | 18.5 cents |
| strength = 1.0 | **2.1 cents** |

Built on YIN, which is accurate to under 0.1% and — critically — does not make
octave errors when the second harmonic is louder than the fundamental, as it
usually is in voiced singing.

### Earlier results

### Formants stay put

A synthetic vowel with resonances at 500 / 1500 / 2500 Hz, shifted seven
semitones in each direction. Plain shifting scales the formants along with the
pitch; correction leaves them where they were.

| signal | f0 (Hz) | detected formants (Hz) |
|---|---:|---|
| original | 164.6 | 431, 1507, 2519 |
| plain shift −7 st | 110.2 | 323, 991, 1658 |
| **formant-preserved −7 st** | **110.2** | **431, 1507, 2498** |
| plain shift +7 st | 247.8 | 807, 1529, 2218 |
| **formant-preserved +7 st** | **247.8** | **431, 1518, 2530** |

Pitch moves by exactly the requested interval in both cases. Only the
resonances differ — and audibly, that is the difference between a chipmunk and
the same person singing higher.

### Transients survive

Crest factor on synthetic percussion stretched 1.5×. Higher means sharper
attacks; read it against the original rather than in absolute terms.

| | crest factor | lost |
|---|---:|---:|
| original | 11.42 | — |
| locked vocoder | 9.39 | 2.04 |
| **+ phase reset at onsets** | **11.02** | **0.40** |
| WSOLA | 8.69 | 2.73 |

Phase reset recovers 80% of the attack sharpness that plain vocoding loses.
Onset detection found 8 of 8 planted hits.

### Earlier results

### Four methods, one pure 220 Hz tone

Amplitude warble — relative fluctuation of the analytic envelope. The ideal
output is a flat envelope, so every deviation is the algorithm's doing.

| stretch | OLA | WSOLA | phase vocoder | + phase locking |
|---:|---:|---:|---:|---:|
| 1.5× | 0.0428 | 0.00025 | 0.0080 | **0.00040** |
| 2.0× | 0.3857 | 0.0559 | 0.0158 | **0.00062** |

At 2× the ordering spans three orders of magnitude, and it matches what you
hear: OLA warbles badly, WSOLA is much better, the plain vocoder is smoother
still but phasey, and locking is clean.

Note the crossover — WSOLA tracks the locked vocoder closely up to 1.5× and
then degrades sharply, which is the known limitation of a similarity search
that has to pick one compromise offset.

### Phase locking vs. plain propagation

Consistency measure $D_M$ on a 25-harmonic stack:

| stretch | plain | locked | improvement |
|---:|---:|---:|---:|
| 1.25 | 0.0097 | 0.00084 | 11× |
| 1.50 | 0.0099 | 0.00063 | 16× |
| 2.00 | 0.0188 | 0.00077 | **24×** |

The advantage widens with stretch factor, which is the signature of a fix that
addresses the actual mechanism rather than masking a symptom.

### Earlier results

Correct phase propagation vs. reusing analysis phase, on the same harmonic
stack. Identical magnitudes; only phase handling differs.

| stretch | vocoder | passthrough (broken) |
|---:|---:|---:|
| 0.50 | **0.0006** | 0.3006 |
| 0.75 | **0.0001** | 0.0486 |
| 1.00 | 0.0000 | 0.0000 |
| 1.25 | **0.0089** | 0.0295 |
| 1.50 | **0.0096** | 0.0505 |
| 2.00 | **0.0186** | 0.0968 |

Two things the table shows. Correct phase propagation is up to two orders of
magnitude more consistent. And the broken mode degrades steadily as frames are
pulled further apart, which is exactly the audible behaviour — phasiness gets
worse the harder you stretch.

The advantage narrows on vibrato material (see `figures/06_consistency.png`),
because the vocoder models each bin as a *slowly varying* sinusoid and vibrato
strains that assumption. This is the gap that phase locking addresses in week 3.

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
