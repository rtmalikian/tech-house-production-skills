# Tech House Production Pipeline

A complete, automated tech house production pipeline — from MIDI generation to mastered MP3, driven by a Roland Fantom-6 synthesizer.

## What This Does

```
MIDI Generation → Roland Fantom Recording → Stem EQ → QC → Professional Chain → Mastering → MP3
```

Every step is automated. Run one command, get a mastered tech house track.

## Quick Start

```bash
cd "/Volumes/Raphael/Tech House" && source venv/bin/activate

# Generate MIDI
python3 midi_orchestrator.py

# Get song name
SONG=$(python3 -c "import os; mids=sorted([f.replace('.mid','') for f in os.listdir('output') if f.endswith('.mid')],key=lambda f:os.path.getmtime(f'output/{f}.mid'),reverse=True);print(mids[0])")

# Record stems from Fantom (auto-detects device index)
python3 record_stems.py "output/${SONG}.mid" --bpm 128

# Master with EQ, QC, professional chain, and A/B comparison
python3 run_pipeline.py --stems "output/${SONG}/recordings/" --song-name "$SONG" --bpm 128 --output-dir "output/${SONG}/mastered"
```

## Architecture

### Core Pipeline Files

| File | Purpose |
|------|---------|
| `midi_orchestrator.py` | Main MIDI generator — bass, acid, stabs, pads, drums, FX, arpeggios |
| `midi_config.py` | Configuration — BPM, keys, zero swing/humanization, register ranges |
| `midi_song_structure.py` | 80-bar DJ-friendly arrangement (all 16-bar sections) |
| `midi_drum_sequences.py` | 8 drum pattern families (909-style, kick vel 122-127 only) |
| `midi_composition.py` | Bass and melody generation |
| `midi_engine.py` | MIDI utilities — swing, humanization, spatial processing |
| `sysex_automation.py` | Real-time SysEx automation (filter sweeps, LFO spikes on B bars) |
| `record_stems.py` | Multi-pass stem recording with Fantom SysEx control + auto-detect |
| `run_pipeline.py` | Master orchestrator — EQ → QC → mix → master → EQ → A/B → MP3 |
| `professional_post.py` | Professional post-production chain |
| `stem_eq.py` | Tech house specific EQ curves for each stem category |
| `stem_qc_v3.py` | Full balance correction (kick reference, ±20 dB) |
| `stem_qc_v2.py` | Transient-aware stem QC (catches per-transient spikes) |
| `stem_qc.py` | RMS-based stem balance QC (legacy) |

### Skills Documentation (`skills/`)

| Skill | Topic |
|-------|-------|
| `01_sound_selection.md` | 909 kick/clap/hat audition with FFT scoring |
| `02_drum_programming.md` | 4-on-floor, 16th hats, swing, velocity, section layering |
| `03_bass_design.md` | Oscillator→filter→saturation, sidechain, patterns |
| `04_synth_elements.md` | Chord stabs, 303 acid lines, pads |
| `05_arrangement.md` | 16-bar DJ-friendly sections, energy curve, filter automation |
| `06_mixing.md` | Gain staging, EQ chains, compression, stereo width |
| `07_mastering.md` | Multiband, limiting, LUFS targeting |
| `08_reference_comparison.md` | A/B methodology, spectral overlay |
| `09_fantom_sysex.md` | Fantom SysEx — parameter ranges, dual LFO matrix, step LFO |
| `assessment.py` | CLI for LUFS, crest factor, spectral balance, stereo correlation |
| `tech_house_fantom_patches.json` | 56 bass, 10 drum kits, 9 stabs, 5 acid leads, 10 pads |

## Key Design Decisions

### MIDI Playback Timing
- **Pure busy-wait** — `while time.perf_counter() < target: pass`
- NO `time.sleep()` — introduces ±120ms jitter on macOS
- Uses 100% CPU but gives ±3ms accuracy
- This is what made v21 and v28 sound tight

### Kick Velocity
- Range: **122-127 only** (no ghost kicks, no syncopated kicks)
- Ghost kicks at velocity 56-70 cause galloping — removed
- Groove comes from hat velocity variation (120/110/100)

### A-A-A-B Melody Structure
- Bars 0-2: standard pattern (A)
- Bar 3: switch-up with chromatic notes, higher octave (B)
- SysEx automation: filter cutoff + LFO depth spikes on B bars
- 6 rhythm patterns: dotted, long-short, sparse, triplet, reverse, offbeat

### Chord Progressions
- **i-VII-VI-VII** — THE classic tech house loop
- Minor 7th chords as default
- 90% minor keys (A minor, D minor, E minor)

### DJ-Friendly Arrangement (80 bars)
- Intro (16 bars): Kick + hats only (bars 0-7), + clap (bars 8-15)
- Drop 1 (16 bars): Staggered entry (kick+bass → stabs → acid)
- Breakdown (16 bars): No kick, no bass, pad only, filter sweeps
- Drop 2 (16 bars): Full energy
- Outro (16 bars): Drums fading

### Stem EQ (tech house curves)
- Kick: HPF 30Hz, +3dB@60Hz, -3dB@300Hz, +2.5dB@3.5kHz
- Open hat: HPF 350Hz, **-4dB@5.5kHz**, -2dB@7kHz (tame harshness)
- Bass: HPF 25Hz, +2dB@50Hz, +2.5dB@100Hz, -3dB@250Hz
- Pad: HPF 250Hz, -4dB@350Hz

### QC v3 (balance correction)
- Kick = reference (0 dB)
- Bass: -2 dB, Clap: -4 dB, Hats: -6/-8 dB, Pad: -8 dB
- Max correction: ±20 dB

### Professional Chain
1. Parallel compression (10:1, 60% blend)
2. Reverb on claps (300ms, 40% wet)
3. Reverb on melodic (1.5s, 45% wet)
4. Delay on acid (1/8 note, 30% feedback)
5. 4-band multiband compression
6. Stereo imaging (mono bass <120Hz)
7. Harmonic excitation (tape saturation 5%)
8. Presence boost (+4dB@3kHz)
9. Mud cut (-4dB@400Hz)
10. Hard clip at -9 dBFS
11. Brick-wall limiter
12. Final loudness match to -9 LUFS

## Version History

### v28 — Auto-Detect Device + Busy-Wait (Latest, confirmed good)

**Fixes:**
- Auto-detect Fantom audio device index (was hardcoded, broke on reconnect)
- Reverted to v21-style pure busy-wait (no sleep jitter)
- Kick timing: ±3ms accuracy (was ±120ms with sleep-based timing)

**Files changed:** `record_stems.py` (auto-detect, busy-wait)

---

### v25 — High-Precision Timing + MIDI Duration Fix

**Fixes:**
- MIDI duration bug: 301 minutes → 4 minutes (filter_build_automation was creating huge deltas)
- Bass style default for intro/breakdown sections
- Drum gain +6dB → 0dB (prevents clipping)

**Files changed:** `midi_orchestrator.py`, `record_stems.py`

---

### v20 — Stem EQ + QC v3

**Problem:** Open hat drowning mix, pad 13dB too loud, no EQ on stems.

**Solution:**
- Built `stem_eq.py` — tech house specific EQ curves for each stem category
- Built `stem_qc_v3.py` — full balance correction against kick reference
- Integrated into `run_pipeline.py` (runs automatically before mastering)

**Key EQ curve:** Open hat gets -4dB at 5.5kHz and -2dB at 7kHz to tame harshness.

**Files changed:** `stem_eq.py` (new), `stem_qc_v3.py` (new), `run_pipeline.py` (integrated)

---

### v19 — Full Balance QC

**Problem:** Pad was 13dB too loud relative to kick.

**Solution:** Built `stem_qc_v3.py` that measures active RMS of each stem and corrects to target levels (kick = 0dB reference).

**Results:**
- Pad: -3.0 → -15.1 dBFS (-12.0 dB correction)
- Bass: -32.9 → -14.7 dBFS (+20.0 dB boost)
- 19 stems corrected total

**Files changed:** `stem_qc_v3.py` (new), `run_pipeline.py` (integrated)

---

### v18 — Transient-Aware QC

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

**Problem:** Mastering was generic — not optimized for tech house.

**Solution:** Analyzed reference tracks (John Summit, MK, Dennis Ferrer, AuRa) and adjusted:
- Target LUFS: -9.2 (club loudness, not streaming -14)
- Sub boost: +4 dB at 60 Hz
- Mud cut: -6 dB at 400 Hz
- Stereo widening: 2x

**Files changed:** `professional_post.py`, `run_pipeline.py`

---

### v8 — Professional Post-Production Chain

**Problem:** Mix sounded amateur — no depth, no punch, no width.

**Solution:** Built complete professional chain:
- Parallel compression (NY-style, 40% blend)
- Reverb on claps (120ms decay, plate)
- Delay on acid (1/8 note, 30% feedback)
- 4-band multiband compression
- Stereo imaging (mono bass <120Hz)
- Harmonic excitation (tape saturation)
- Presence boost (+4dB at 3kHz)
- Hard clip + brick-wall limiter

**Files changed:** `professional_post.py` (new), `run_pipeline.py` (integrated)

---

### v5 — Kick Velocity Fix

**Problem:** Kicks sounded like galloping — inconsistent rhythm.

**Root cause:** Velocity range was 56-127. Ghost kicks at velocity 56 were much quieter than main kicks at 127, creating a loud-soft-loud-soft pattern.

**Solution:** Fixed all kick velocity to 122-127. Removed ghost kicks, syncopated kicks, push kicks. Groove now comes from hat velocity variation only.

**Files changed:** `midi_drum_sequences.py` (all 8 pattern families)

---

### v1 — Initial Pipeline

Adapted the lofi hip hop pipeline (crate-dig/june2026) for tech house:
- BPM: 80 → 128
- Keys: 90% minor
- Swing: 0% (fully quantized)
- Humanization: 0 (zero timing jitter)

**Files changed:** All files adapted from crate-dig pipeline.

---

## Pitfalls Discovered

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Galloping kicks | Velocity 56-70 ghost kicks | Velocity 122-127 only |
| Open hat drowning mix | 5-7kHz harshness | -4dB@5.5kHz EQ cut |
| Pad 13dB too loud | No balance QC | stem_qc_v3 auto-correction |
| ±120ms timing jitter | `time.sleep()` on macOS | Pure busy-wait (no sleep) |
| 301-minute MIDI files | filter_build_automation huge deltas | Removed (SysEx handles it) |
| Fantom device not found | Hardcoded index changes on reconnect | Auto-detect by name |
| Silent stems | Wrong audio device index | Auto-detect `_find_fantom_device()` |
| Bash SONG extraction fails | `basename .mid` produces empty | Python-based extraction |
| Sidechain pumping | 15ms release too fast | 150ms release |
| Master compressor pumping | `/4` divisor on release kernel | Full release time |

## Reference Tracks

The pipeline was optimized against these reference tracks (not included — copyright protected):

| Track | Artist | BPM | Key | Insight |
|-------|--------|-----|-----|---------|
| Deep End | John Summit | 126 | C# Minor | Bass drops out in breakdown, "other" enters before drop |
| Hey Hey | Dennis Ferrer | 126 | Eb Minor | Bass enters LATE (bar 42!), vocal-driven |
| Panic Room (Jonas Rathsman Remix) | AuRa | 124 | — | 56-bar breakdown, drums drop 20-50 dB |
| 17 (Remixes) | MK | 124 | — | Spectral reference, club loudness |

## License

MIT

## Author

Raphael T. Malikian — [github.com/rtmalikian](https://github.com/rtmalikian)
