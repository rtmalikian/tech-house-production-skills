# Tech House Production Pipeline

A complete, automated tech house production pipeline — from MIDI generation to mastered MP3, driven by a Roland Fantom-6 synthesizer.

## What This Does

```
MIDI Generation → Roland Fantom Recording → Stem QC → Mixing → Mastering → MP3
```

Every step is automated. Run one command, get a mastered tech house track.

## Quick Start

```bash
# Generate MIDI
python3 midi_orchestrator.py

# Record stems from Fantom
python3 record_stems.py output/<song>.mid --bpm 128

# Master with QC, professional chain, and A/B comparison
python3 run_pipeline.py --stems output/<song>/recordings/ --song-name <song> --bpm 128 --output-dir output/<song>/mastered/
```

## Architecture

### Core Pipeline Files

| File | Purpose |
|------|---------|
| `midi_orchestrator.py` | Main MIDI generator — creates all tracks (bass, acid, stabs, pads, drums, FX) |
| `midi_config.py` | Configuration — BPM, keys, swing, humanization, register ranges |
| `midi_song_structure.py` | Arrangement — 128-bar DJ-friendly structure |
| `midi_drum_sequences.py` | 8 drum pattern families (909-style) |
| `midi_composition.py` | Bass and melody generation |
| `midi_engine.py` | MIDI utilities — swing, humanization, spatial processing |
| `record_stems.py` | Multi-pass stem recording with Fantom SysEx control |
| `run_pipeline.py` | Master orchestrator — QC → mix → master → EQ → A/B → MP3 |
| `professional_post.py` | Professional post-production chain |
| `stem_qc_v2.py` | Transient-aware stem QC system |
| `stem_qc.py` | RMS-based stem balance QC |

### Skills Documentation (`skills/`)

| Skill | Topic |
|-------|-------|
| `01_sound_selection.md` | 909 kick/clap/hat audition with FFT scoring |
| `02_drum_programming.md` | 4-on-floor, 16th hats, swing, velocity, section layering |
| `03_bass_design.md` | Oscillator→filter→saturation, sidechain, patterns |
| `04_synth_elements.md` | Chord stabs, 303 acid lines, pads |
| `05_arrangement.md` | 8-bar rule, energy curve, filter automation |
| `06_mixing.md` | Gain staging, EQ chains, compression, stereo width |
| `07_mastering.md` | Multiband, limiting, LUFS targeting |
| `08_reference_comparison.md` | A/B methodology, spectral overlay |
| `09_fantom_sysex.md` | Fantom SysEx — parameter ranges, dual LFO matrix, step LFO |
| `assessment.py` | CLI for LUFS, crest factor, spectral balance, stereo correlation |
| `tech_house_fantom_patches.json` | 56 bass, 10 drum kits, 9 stabs, 5 acid leads, 10 pads |

## Version History

### v18 — Transient-Aware QC (Latest)

**Problem:** Open hi-hat at 31 seconds was drowning out the entire mix. Previous QC system (RMS-based) missed per-transient spikes.

**Solution:** Built `stem_qc_v2.py` — analyzes each stem in 2-second windows, finds clipping moments (> -1 dBFS) and transient spikes (> 8x peak/RMS ratio), applies soft clipping (tanh) to prevent hard clipping.

**Results:**
- 10 stems had clipping peaks (0.0 dBFS)
- Acid Line: 3,376 clipping moments fixed
- Chord Stab: 9,969 clipping moments fixed
- Open Hat: 456 clipping moments fixed
- All stems now peak at -2 dBFS maximum

**Files changed:** `stem_qc_v2.py` (new), `run_pipeline.py` (integrated QC v2)

---

### v17 — Fixed Drum Gain

**Problem:** +6 dB drum gain was pushing hats and crashes into clipping.

**Solution:** Changed drum calibration gain from +6 dB to 0 dB (no boost).

**Files changed:** `record_stems.py` (calibrate_part_gain)

---

### v16 — Stem-Analyzed Reference Patterns

**Problem:** Arrangement didn't match professional tech house patterns.

**Solution:** Ran 4-stem demucs separation on 3 reference tracks (John Summit, Dennis Ferrer, AuRa), analyzed how each element builds and fades.

**Key findings:**
- Bass drops out during breakdown (AuRa: bars 160-176)
- Pads increase 50% during breakdown (fill the space)
- Atmospheric "other" elements enter just before drop (John Summit: bar 47)
- Breakdowns are 32 bars, not 16

**Files changed:** `midi_orchestrator.py` (staggered entry, bass drops out, pads louder, atmospheric elements)

---

### v15 — Tight Quantization

**Problem:** Melodies had too much velocity variation, sounding "loose."

**Solution:** Fixed all velocities to consistent values — no randomization.

**Changes:**
- Hats: fixed 120/110/100 (was random 75-127)
- Open hat: fixed 110/105 (was random 95-120)
- Arpeggios: fixed 80/70 (was variable)
- Acid: fixed 100×energy (was random 90-120)
- Stabs: fixed 100×energy (was 110×energy)
- Bass: fixed 50×energy (was 45-53)

**Files changed:** `midi_orchestrator.py`, `midi_drum_sequences.py`

---

### v14 — Commercial Length (128 bars)

**Problem:** Tracks were 2.5 minutes — too short for commercial release.

**Solution:** Extended to 128 bars (~4:00 at 128 BPM). Added filter sweeps in breakdown.

**Arrangement:**
- Intro (16 bars): Kick + hats only
- Drop 1 (32 bars): Staggered entry
- Breakdown (32 bars): Filter sweeps, riser, bass preview
- Drop 2 (32 bars): Staggered entry
- Outro (16 bars): Drums fading

**Files changed:** `midi_config.py` (TOTAL_BARS=128), `midi_song_structure.py`, `midi_orchestrator.py`

---

### v13 — Arpeggios, LFO, Reverb

**Problem:** Melodies were static, no psychedelic flavor, no depth.

**Solution:**
- Arpeggios: 8th, 16th, syncopated patterns cycling chord tones
- LFO filter modulation: S&H waveform synced to tempo, aggressive cutoff modulation
- Long reverb: 2-second tails on stabs, acid, pad (30% wet)

**Files changed:** `midi_orchestrator.py` (arpeggios), `record_stems.py` (LFO), `professional_post.py` (reverb)

---

### v12 — Reference-Optimized Mastering

**Problem:** Spectral balance didn't match professional tech house.

**Solution:** Analyzed 4 reference tracks (John Summit, MK, Dennis Ferrer, AuRa). Discovered our skills documentation had wrong targets.

**Key findings from references:**
- LUFS: -9.2 (not -14)
- Sub bass: 38% (not 15-20%)
- Low-mid: 12.6% (not 15-25%)
- Presence: 3.9% (not 10-20%)

**Files changed:** `run_pipeline.py` (targets, EQ), `professional_post.py` (mastering chain)

---

### v8 — Professional Post-Production Chain

**Problem:** Track sounded amateur compared to references.

**Solution:** Built comprehensive post-production chain:
1. Parallel compression on drums (10:1, 40% blend)
2. Reverb on claps (120ms decay)
3. Delay on acid (1/8 note, 30% feedback)
4. 4-band multiband compression
5. Stereo imaging (mono bass <120Hz, wide highs)
6. Harmonic excitation (tape saturation)
7. EQ: mud cut, presence boost, air boost
8. Hard clip + limiter
9. Final loudness match

**Files changed:** `professional_post.py` (new), `run_pipeline.py` (integrated)

---

### v5 — Kick Velocity Fix

**Problem:** Kick sounded "galloping" — inconsistent timing between hits.

**Root cause:** Kick velocity ranged from 56-127 (std=17). Ghost kicks at velocity 56 were 12 dB quieter than main kicks at 127.

**Solution:** Fixed kick velocity to 122-127 (std=2). Removed all ghost kicks, syncopated kicks, push kicks.

**Analysis:** Measured inter-kick intervals — 99% within 460-500ms (expected 480ms at 125 BPM). Standard deviation: 0.92ms.

**Files changed:** `midi_drum_sequences.py` (all 8 drum patterns), `midi_config.py` (swing=0, humanization=0)

---

### v1 — Initial Pipeline

Copied from `crate-dig/final_pipeline_june2026/` and adapted for tech house:
- BPM: 124-128 (was 80-95)
- Swing: 0% (was 5-10%)
- Arrangement: 88-bar tech house structure
- Drum patterns: 909-style (was lofi)
- Bass: syncopated tech house patterns

## Key Design Decisions

1. **Sidechain in post-production only** — Clean signal for mixing flexibility
2. **Parameter ranges over fixed values** — Randomized within tech house sweet spot
3. **Dual LFO with S&H** — Creates evolving, non-repeating filter textures
4. **Short file names + JSON sidecar** — `TH_MMDD_HHMM_BPM_Key.mid` + `.json`
5. **Multi-pass stem recording** — 15 parts per pass (Fantom has 16 USB pairs)
6. **Level calibration** — Target -6 dBFS per part via zone EQ
7. **Sync click detection** — Sample-accurate trim via USB 31/32

## Hardware Requirements

- **Roland Fantom-6** (or Fantom 7/8) — Synthesizer with USB audio
- **Mac** — For Python pipeline
- **External drive** — For output storage

## Software Requirements

```bash
pip install mido python-rtmidi soundfile pyloudnorm numpy scipy librosa pedalboard matplotlib
```

## File Naming Convention

```
TH_<MMDD>_<HHMM>_<bpm>_<key><type>.mid
TH_0613_1929_128_Fmin.mid
```

JSON sidecar with full metadata:
```json
{
  "song_name": "TH_0613_1929_128_Fmin",
  "bpm": 128,
  "key": "F# Minor",
  "scale": "harmonic_minor",
  "total_bars": 88,
  "duration_estimate_sec": 165
}
```

## GitHub Topics

`tech-house` `music-production` `midi` `audio-engineering` `mixing` `mastering` `909` `roland-fantom` `sysex` `python` `automation` `dsp` `audio-analysis`

## Reference Tracks

The pipeline was optimized against these reference tracks (not included — copyright protected):

| Track | Artist | BPM | Duration | Key Insight |
|-------|--------|-----|----------|-------------|
| Deep End | John Summit | 126 | 2:30 | 32-bar breakdown, bass drops out, "other" enters before drop |
| Hey Hey (Original Mix) | Dennis Ferrer | 124 | 5:50 | Bass enters late (bar 42!), consistent energy, vocal-driven |
| Panic Room (Jonas Rathsman Remix) | AuRa | 122 | 7:40 | 56-bar breakdown, pads increase during breakdown, sub 66%→26% |
| 17 (Remixes) | MK | — | — | Spectral balance reference |

**Spectral profile (average from stem separation analysis):**
- LUFS: -9.2 (much louder than streaming standard)
- Sub 20-60 Hz: 38%
- Bass 60-250 Hz: 42%
- Low-Mid 250-2k Hz: 13%
- Presence 2k-6k Hz: 4%
- Air 6k-20k Hz: 4%

**Arrangement patterns discovered:**
- Bass drops out during breakdown (creates tension)
- Pads increase 50% during breakdown (fill the space)
- Atmospheric elements enter just before drop (primes listener)
- Breakdowns are 16-32 bars, not 8
- Energy drops of 6-12 dB during breakdowns

## License

MIT

## Author

Raphael T. Malikian
- GitHub: [rtmalikian](https://github.com/rtmalikian)
- Email: rtmalikian@gmail.com
