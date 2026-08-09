# Contributing

## Setup

```bash
git clone https://github.com/USERNAME/elastiqa
cd elastiqa
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

274 tests should pass in about two minutes.

## Before you open a pull request

```bash
ruff check elastiqa tests
pytest
```

## The one rule

**Do not weaken a test to make it pass.** Several tests in this repository
exist because a plausible-looking implementation was wrong in a way that was
inaudible until it was measured:

- `test_perfect_reconstruction.py` — if the identity path is not exact, every
  artefact you chase afterwards may be this instead.
- `test_consistency_of_real_signal_is_zero` — computing $D_M$ on the output's
  STFT instead of the modified STFT makes every method look perfect.
- `test_instantaneous_frequency_offgrid` — a vocoder with broken phase
  unwrapping still produces sound, just quietly detuned.
- `test_abrupt_cutoff_does_register` — spectral flux fires on discontinuities,
  not only on attacks.

If one of these fails, the code is wrong.

## Adding a method

New time-scaling methods go in `elastiqa/tsm/`, take `(x, stretch, n_fft, hop)`,
and return a real signal of length `round(len(x) * stretch)`. Register it in
`benchmark.DEFAULT_METHODS` so it appears in the results tables, and give it a
`stft_of` entry if it produces a modified STFT.
