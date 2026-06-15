# Tech House Production Skills — Complete Pipeline Reference

## How to Use This Document

Every skill in a tech house producer's workflow is broken into atomic steps.
Each step has:
1. **What the producer does** (real-world action)
2. **Why** (the sonic goal)
3. **How to do it programmatically** (Python code / libraries)
4. **Objective assessment** (measurable criteria — NOT arbitrary numbers)
5. **Iterative refinement loop** (how to know when to stop adjusting)

## Production Phases

| Phase | Skills | Files |
|-------|--------|-------|
| **1. Sound Selection** | Kick, clap, hat, percussion audition | `01_sound_selection.md` |
| **2. Drum Programming** | Pattern writing, velocity, swing | `02_drum_programming.md` |
| **3. Bass Design & Programming** | Oscillator, filter, sidechain, patterns | `03_bass_design.md` |
| **4. Synth & Melodic Elements** | Stabs, acid lines, pads | `04_synth_elements.md` |
| **5. Arrangement** | Section structure, energy, automation | `05_arrangement.md` |
| **6. Mixing** | Gain staging, EQ, compression, spatial | `06_mixing.md` |
| **7. Mastering** | Tonal balance, limiting, loudness | `07_mastering.md` |
| **8. Reference Comparison** | A/B analysis, spectral matching | `08_reference_comparison.md` |
| **Assessment** | Objective quality metrics | `assessment.py` |

## The Iterative Loop (applies to EVERY skill)

```
┌─────────────┐
│  GENERATE   │  Create the element / apply the processing
└──────┬──────┘
       ▼
┌─────────────┐
│   MEASURE   │  Run objective metrics (see assessment.py)
└──────┬──────┘
       ▼
┌─────────────┐     ┌──────────────┐
│  COMPARE    │────►│  THRESHOLD   │  Does it pass the objective test?
└──────┬──────┘     └──────┬───────┘
       │                   │
       │              YES  │  NO
       │                   │
       ▼                   ▼
┌─────────────┐     ┌─────────────┐
│   ACCEPT    │     │   ADJUST    │  Apply corrective action
└─────────────┘     └──────┬──────┘
                           │
                           └──────► (back to GENERATE)

Max iterations: 3 per skill. If not converged after 3, log and move on.
```

## Library Stack

| Library | Role | Install |
|---------|------|---------|
| `mido` | MIDI generation | `pip install mido` |
| `numpy` | Signal math | `pip install numpy` |
| `scipy` | Filters, FFT, peaks | `pip install scipy` |
| `librosa` | Audio analysis | `pip install librosa` |
| `pedalboard` | Effects chain | `pip install pedalboard` |
| `soundfile` | WAV I/O | `pip install soundfile` |
| `pyloudnorm` | LUFS metering | `pip install pyloudnorm` |
| `matplotlib` | Visualization | `pip install matplotlib` |
| `pydub` | Quick arrangement | `pip install pydub` |
