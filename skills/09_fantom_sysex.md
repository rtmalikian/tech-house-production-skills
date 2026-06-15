# Skill 09: Roland Fantom SysEx Integration

## What This Covers
Every sound design parameter from Skills 01-04 can be realized on the
Roland Fantom hardware via SysEx DT1 commands. This skill maps each
production concept to the exact SysEx addresses, parameter ranges, and
patch selections needed.

## Design Philosophy: Ranges, Not Fixed Values

Every parameter uses a **range** (min-max) that gets randomized on each
song. This prevents every track from sounding the same. The Fantom's
dual LFOs + matrix routing create **evolving, living patches** — not
static snapshots.

**Sidechain pumping is done in post-production** (see Skill 06), not
via Fantom LFO. This keeps the recorded signal clean for maximum
mixing flexibility.

---

## Architecture Overview

```
Python Script
    │
    ▼
FantomController (fantom_midi_control.py)
    │
    ├── select_patch(channel, msb, lsb, pc)       → Bank Select + Program Change
    ├── _send_dt1(address, data)                   → SysEx DT1 write
    ├── _set_mfx_type_and_params(base, type, params) → MFX effect chain
    ├── apply_zcore_lfo_matrix(part_idx, name)     → Dual LFO + matrix routing
    └── enable_drum_note_fxm(part_idx, note)       → Per-drum-note processing
    │
    ▼
Roland Fantom (Z-Core synth engine)
```

### SysEx Message Format
```
F0  41  10  00 00 00 5B  12  [addr:4]  [data:N]  [checksum]  F7
 │   │   │       │        │      │          │          │
 │   │   │       │        │      │          │          └─ Roland checksum
 │   │   │       │        │      │          └─ Parameter data
 │   │   │       │        │      └─ DT1 command (0x12)
 │   │   │       │      └─ Model ID (Fantom)
 │   │   │       └─ Command
 │   │   └─ Device ID (0x10)
 └─ └─ Roland ID (0x41)
```

---

## 9.1 — Patch Selection for Tech House

### Channel Assignments

| Channel | Role | Category | Patch Examples |
|---------|------|----------|----------------|
| Ch 0 | Acid Line / Lead | bass/lead | SL-TB Saw 4, TB Bass 1, TB Dist Bs 1 |
| Ch 1 | Chord Stab | poly/lead | Gabba Stabber, Brassy VA Lead, Power Sync |
| Ch 2 | Pad (breakdowns) | pad | Soft Saw Pad, Sawchestra, JP8 Strings |
| Ch 3 | Sub Bass | bass | Dirt Sub 1, Sub Zero, Spread Sub |
| Ch 4 | Main Bass | bass | S.C House Bass, House Bs, 106 Bass |
| Ch 9 | Drums | drums | TR-909, House Kit, Techno Kit |

### Curated Tech House Patch List

```python
TECH_HOUSE_PATCHES = {
    'acid_bass': [
        {'name': 'SL-TB Saw 4',   'msb': 87, 'lsb': 68, 'pc': 89},
        {'name': 'SL-TB Saw 5',   'msb': 87, 'lsb': 68, 'pc': 91},
        {'name': 'SL-TB Sqr 3',   'msb': 87, 'lsb': 68, 'pc': 99},
        {'name': 'TB Bass 1',     'msb': 87, 'lsb': 74, 'pc': 46},
        {'name': 'TB Dist Bs 1',  'msb': 87, 'lsb': 74, 'pc': 48},
    ],
    'sub_bass': [
        {'name': 'Dirt Sub 1',    'msb': 87, 'lsb': 92, 'pc': 104},
        {'name': 'Sub Zero',      'msb': 87, 'lsb': 83, 'pc': 66},
        {'name': 'Spread Sub 1',  'msb': 87, 'lsb': 93, 'pc': 97},
        {'name': 'SL-Jn60sub3',   'msb': 87, 'lsb': 68, 'pc': 66},
    ],
    'house_bass': [
        {'name': 'S.C House Bass', 'msb': 87, 'lsb': 92, 'pc': 122},
        {'name': 'House Bs',       'msb': 87, 'lsb': 74, 'pc': 41},
        {'name': '106 Bass 2',     'msb': 87, 'lsb': 73, 'pc': 122},
        {'name': 'Reso Pumper Bass','msb': 87, 'lsb': 92, 'pc': 119},
        {'name': '24Db Mini Ladder','msb': 87, 'lsb': 92, 'pc': 112},
    ],
    'wobble_bass': [
        {'name': 'Future Wobble',  'msb': 87, 'lsb': 92, 'pc': 123},
        {'name': 'Madness Wobble', 'msb': 87, 'lsb': 92, 'pc': 124},
        {'name': 'LFO Lazor Bass', 'msb': 87, 'lsb': 92, 'pc': 121},
    ],
    'chord_stab': [
        {'name': 'Gabba Stabber',  'msb': 87, 'lsb': 92, 'pc': 85},
        {'name': 'Brassy VA Lead', 'msb': 87, 'lsb': 92, 'pc': 82},
        {'name': 'Power Sync',     'msb': 87, 'lsb': 92, 'pc': 87},
        {'name': 'Saw Super Sync', 'msb': 87, 'lsb': 92, 'pc': 83},
    ],
    'acid_lead': [
        {'name': 'SL-TB Saw 6',   'msb': 87, 'lsb': 68, 'pc': 93},
        {'name': 'SL-SH101 9',     'msb': 87, 'lsb': 68, 'pc': 115},
        {'name': 'Future Sync Lead','msb': 87, 'lsb': 92, 'pc': 69},
    ],
    'dark_pad': [
        {'name': 'Soft Saw Pad',   'msb': 87, 'lsb': 65, 'pc': 26},
        {'name': 'Sawchestra',     'msb': 87, 'lsb': 92, 'pc': 29},
        {'name': 'JP8 Strings1',   'msb': 87, 'lsb': 69, 'pc': 1},
        {'name': 'Ambien',         'msb': 87, 'lsb': 92, 'pc': 26},
    ],
    'drum_kit': [
        {'name': 'TR-909',         'msb': 86, 'lsb': 65, 'pc': 45},
        {'name': 'TR-909 comp',    'msb': 86, 'lsb': 65, 'pc': 46},
        {'name': 'House Kit',      'msb': 86, 'lsb': 65, 'pc': 38},
        {'name': 'Techno Kit',     'msb': 86, 'lsb': 65, 'pc': 36},
        {'name': 'TR-808',         'msb': 86, 'lsb': 65, 'pc': 48},
    ],
}
```

---

## 9.2 — Tone Parameter Editing via SysEx (Range-Based)

### Z-Core Part Base Address
```
Part N base: [0x02, 0x10 + N, 0x00, 0x00]
```

### Key Tone Parameters

| Parameter | Address Offset | Range | Description |
|-----------|---------------|-------|-------------|
| **Filter Type** | [0x00, 0x20, 0x00] | 0-7 | LPF1, LPF2, LPF3, HPF, BPF, PKG, LPF+HPF, FORMANT |
| **Filter Cutoff** | [0x00, 0x20, 0x01] | 0-127 | 0=dark, 127=bright |
| **Filter Resonance** | [0x00, 0x20, 0x02] | 0-127 | 0=flat, 127=self-osc |
| **TVF Env Attack** | [0x00, 0x20, 0x05] | 0-127 | Filter envelope attack |
| **TVF Env Decay** | [0x00, 0x20, 0x06] | 0-127 | Filter envelope decay |
| **TVF Env Sustain** | [0x00, 0x20, 0x07] | 0-127 | Filter envelope sustain |
| **TVF Env Release** | [0x00, 0x20, 0x08] | 0-127 | Filter envelope release |
| **TVA Env Attack** | [0x00, 0x24, 0x00] | 0-127 | Amp envelope attack |
| **TVA Env Decay** | [0x00, 0x24, 0x01] | 0-127 | Amp envelope decay |
| **TVA Env Sustain** | [0x00, 0x24, 0x02] | 0-127 | Amp envelope sustain |
| **TVA Env Release** | [0x00, 0x24, 0x03] | 0-127 | Amp envelope release |

### Parameter Ranges for Tech House

Every parameter is a **range**, not a fixed value. Randomized per-song.

| Parameter | Acid Bass | House Bass | Chord Stab | Pad | Why a Range? |
|-----------|-----------|------------|------------|-----|-------------|
| Filter Cutoff | 55–90 | 65–95 | 70–100 | 40–70 | Each song gets a different darkness |
| Filter Resonance | 25–50 | 10–30 | 15–35 | 5–20 | More resonance = more acid, less = smoother |
| TVF Env Attack | 0–5 | 0–3 | 0–2 | 10–40 | Stabs always instant, pads breathe |
| TVF Env Decay | 30–60 | 35–55 | 30–50 | 50–90 | Shorter = punchier, longer = evolving |
| TVF Env Sustain | 40–70 | 50–80 | 0–15 | 60–100 | Stabs: near zero, pads: full sustain |
| TVF Env Release | 15–40 | 20–40 | 10–25 | 40–80 | Quick for stabs, longer for pads |
| TVA Env Attack | 0–2 | 0–3 | 0–1 | 20–60 | Everything except pads: instant |
| TVA Env Decay | 30–50 | 35–55 | 40–60 | 60–100 | Controls note body length |
| TVA Env Sustain | 50–80 | 60–90 | 0–10 | 70–100 | Stabs: zero (one-shot) |
| TVA Env Release | 15–35 | 20–40 | 10–20 | 40–80 | Quick for rhythmic, slow for pads |

### Programmatic: Range-Based Parameter Setting

```python
import random

# Parameter ranges per role
BASS_FILTER_RANGES = {
    'acid': {
        'cutoff': (55, 90), 'resonance': (25, 50),
        'tvf_attack': (0, 5), 'tvf_decay': (30, 60),
        'tvf_sustain': (40, 70), 'tvf_release': (15, 40),
        'tva_attack': (0, 2), 'tva_decay': (30, 50),
        'tva_sustain': (50, 80), 'tva_release': (15, 35),
    },
    'house': {
        'cutoff': (65, 95), 'resonance': (10, 30),
        'tvf_attack': (0, 3), 'tvf_decay': (35, 55),
        'tvf_sustain': (50, 80), 'tvf_release': (20, 40),
        'tva_attack': (0, 3), 'tva_decay': (35, 55),
        'tva_sustain': (60, 90), 'tva_release': (20, 40),
    },
}

STAB_RANGES = {
    'cutoff': (70, 100), 'resonance': (15, 35),
    'tvf_attack': (0, 2), 'tvf_decay': (30, 50),
    'tvf_sustain': (0, 15), 'tvf_release': (10, 25),
    'tva_attack': (0, 1), 'tva_decay': (40, 60),
    'tva_sustain': (0, 10), 'tva_release': (10, 20),
}

PAD_RANGES = {
    'cutoff': (40, 70), 'resonance': (5, 20),
    'tvf_attack': (10, 40), 'tvf_decay': (50, 90),
    'tvf_sustain': (60, 100), 'tvf_release': (40, 80),
    'tva_attack': (20, 60), 'tva_decay': (60, 100),
    'tva_sustain': (70, 100), 'tva_release': (40, 80),
}

def _pick(range_tuple):
    """Random value within a range."""
    return random.randint(range_tuple[0], range_tuple[1])

def apply_bass_tone(fantom, part_idx, style='acid'):
    """Apply bass tone parameters with randomized ranges."""
    ranges = BASS_FILTER_RANGES.get(style, BASS_FILTER_RANGES['house'])
    base = fantom._zcore_base(part_idx)

    cutoff = _pick(ranges['cutoff'])
    resonance = _pick(ranges['resonance'])

    # Filter type: LPF24 (type 1)
    fantom._send_dt1(_addr_add(base, [0x00, 0x20, 0x00]), [1])
    # Cutoff and resonance from range
    fantom._send_dt1(_addr_add(base, [0x00, 0x20, 0x01]), [cutoff])
    fantom._send_dt1(_addr_add(base, [0x00, 0x20, 0x02]), [resonance])
    # Filter envelope from range
    fantom._send_dt1(_addr_add(base, [0x00, 0x20, 0x05]), [_pick(ranges['tvf_attack'])])
    fantom._send_dt1(_addr_add(base, [0x00, 0x20, 0x06]), [_pick(ranges['tvf_decay'])])
    fantom._send_dt1(_addr_add(base, [0x00, 0x20, 0x07]), [_pick(ranges['tvf_sustain'])])
    fantom._send_dt1(_addr_add(base, [0x00, 0x20, 0x08]), [_pick(ranges['tvf_release'])])
    # Amp envelope from range
    fantom._send_dt1(_addr_add(base, [0x00, 0x24, 0x00]), [_pick(ranges['tva_attack'])])
    fantom._send_dt1(_addr_add(base, [0x00, 0x24, 0x01]), [_pick(ranges['tva_decay'])])
    fantom._send_dt1(_addr_add(base, [0x00, 0x24, 0x02]), [_pick(ranges['tva_sustain'])])
    fantom._send_dt1(_addr_add(base, [0x00, 0x24, 0x03]), [_pick(ranges['tva_release'])])

    return f"LPF24 cutoff={cutoff} res={resonance} tvf_decay={_pick(ranges['tvf_decay'])}"

def apply_stab_tone(fantom, part_idx):
    """Apply stab tone parameters with randomized ranges."""
    base = fantom._zcore_base(part_idx)

    cutoff = _pick(STAB_RANGES['cutoff'])
    resonance = _pick(STAB_RANGES['resonance'])

    fantom._send_dt1(_addr_add(base, [0x00, 0x20, 0x00]), [1])
    fantom._send_dt1(_addr_add(base, [0x00, 0x20, 0x01]), [cutoff])
    fantom._send_dt1(_addr_add(base, [0x00, 0x20, 0x02]), [resonance])
    fantom._send_dt1(_addr_add(base, [0x00, 0x20, 0x05]), [_pick(STAB_RANGES['tvf_attack'])])
    fantom._send_dt1(_addr_add(base, [0x00, 0x20, 0x06]), [_pick(STAB_RANGES['tvf_decay'])])
    fantom._send_dt1(_addr_add(base, [0x00, 0x20, 0x07]), [_pick(STAB_RANGES['tvf_sustain'])])
    fantom._send_dt1(_addr_add(base, [0x00, 0x20, 0x08]), [_pick(STAB_RANGES['tvf_release'])])
    fantom._send_dt1(_addr_add(base, [0x00, 0x24, 0x00]), [_pick(STAB_RANGES['tva_attack'])])
    fantom._send_dt1(_addr_add(base, [0x00, 0x24, 0x01]), [_pick(STAB_RANGES['tva_decay'])])
    fantom._send_dt1(_addr_add(base, [0x00, 0x24, 0x02]), [_pick(STAB_RANGES['tva_sustain'])])
    fantom._send_dt1(_addr_add(base, [0x00, 0x24, 0x03]), [_pick(STAB_RANGES['tva_release'])])

    return f"Stab cutoff={cutoff} res={resonance}"
```

---

## 9.3 — Dual LFO Matrix for Evolving Sounds

The Fantom has **two independent LFOs per partial** (4 partials per tone).
Each LFO has its own wave, rate, and destinations. The **matrix** routes
LFO1 and LFO2 to two destinations each with independent sensitivity.
This creates complex, evolving modulation — the key to keeping tech house
sounds interesting over a 6-minute track.

### LFO Wave Options

| Index | Wave | Character | Best For |
|-------|------|-----------|----------|
| 0 | SIN | Smooth, predictable | Filter sweeps, pads |
| 1 | TRI | Linear ramp | Tremolo, rhythmic filter |
| 2 | SAW | Rising/falling | Sawtooth filter sweeps |
| 3 | SQR | On/off | Rhythmic gating, stutter |
| 4 | TRP | Trapezoid | Stepped modulation |
| 6 | S&H | Random steps | Random filter jumps, glitchy |
| 7 | CHS | Chorus-like | Organic movement |
| 8 | VSIN | Variable sine | Subtle organic variation |

### Matrix Destinations (Safe for Tech House)

| Code | Name | What It Does | Good Pairing |
|------|------|-------------|-------------|
| 2 | CUT | Filter cutoff | LFO → cutoff = acid squelch |
| 3 | RES | Resonance | LFO → resonance = evolving bite |
| 4 | LEV | Output level | LFO → level = tremolo/gate |
| 5 | PAN | Stereo position | LFO → pan = movement |
| 10 | TVF-LFO1 | Filter env depth | Subtle filter variation |
| 11 | TVF-LFO2 | Filter env depth | Cross-mod filter |
| 12 | TVA-LFO1 | Amp env depth | Rhythmic volume |
| 13 | TVA-LFO2 | Amp env depth | Cross-mod volume |
| 16 | LFO1-RATE | LFO1 speed | Self-modulating speed |
| 17 | LFO2-RATE | LFO2 speed | Evolving modulation rate |
| 34 | PW | Pulse width | Oscillator character |
| 35 | PWM | Pulse width mod | Breathing oscillator |
| 36 | FAT | Detune/fatness | Width variation |
| 37 | XMOD | Cross-modulation | Harmonic complexity |

### Parameter Ranges for LFOs

| Parameter | Bass | Stab | Pad | Acid Lead |
|-----------|------|------|-----|-----------|
| LFO1 Wave | TRI, SAW, SQR | TRI, SQR | SIN, TRI, S&H | SAW, S&H |
| LFO1 Rate | 1/8 – 1/4 | 1/4 – 1/2 | 1 – 4 bars | 1/8 – 1/4 |
| LFO1 TVF Depth | 8–18 | 10–20 | 12–22 | 15–25 |
| LFO1 TVA Depth | 3–8 | 2–6 | 4–10 | 2–6 |
| LFO2 Wave | S&H, SIN, TRI | SIN, CHS | S&H, VSIN | SQR, S&H |
| LFO2 Rate | 1/4 – 1 | 1/2 – 2 bars | 2–8 bars | 1/4 – 1/2 |
| LFO2 TVF Depth | 6–14 | 8–16 | 10–20 | 10–18 |
| LFO2 TVA Depth | 2–6 | 1–5 | 3–8 | 1–4 |

### Matrix Routing Philosophy

**LFO1** = primary movement (faster, more obvious)
**LFO2** = secondary texture (slower, more subtle, cross-modulating)

Each LFO routes to **two destinations** with independent sensitivity.
This gives 4 modulation paths per partial.

**Good combinations for tech house:**
- LFO1 → CUT + LFO1 → LFO2-RATE (filter sweep that speeds up/slows down)
- LFO2 → RES + LFO2 → FAT (resonance that evolves with detune)
- LFO1 → CUT + LFO2 → PAN (filter movement with stereo drift)
- LFO1 → TVA + LFO2 → CUT (rhythmic gate with filter evolution)
- LFO1 → PW + LFO2 → XMOD (pulse width with cross-mod complexity)

### Programmatic: Dual LFO Matrix Setup

```python
# LFO wave and rate options
LFO_WAVES = [0, 1, 2, 3, 4, 6, 7, 8]  # SIN, TRI, SAW, SQR, TRP, S&H, CHS, VSIN
LFO_RATES = [9, 12, 15, 18, 21]         # 1/8, 1/4, 1/2, 1, 2 bars in Fantom ordering

# Safe matrix destinations for tech house
SAFE_DESTINATIONS = [
    (2, "CUT"), (3, "RES"), (4, "LEV"), (5, "PAN"),
    (10, "TVF-LFO1"), (11, "TVF-LFO2"), (12, "TVA-LFO1"), (13, "TVA-LFO2"),
    (16, "LFO1-RATE"), (17, "LFO2-RATE"),
    (34, "PW"), (35, "PWM"), (36, "FAT"), (37, "XMOD"),
]

def apply_tech_house_lfo_matrix(fantom, part_idx, role='bass'):
    """
    Set up dual LFOs with matrix routing for evolving tech house sounds.
    Uses randomized ranges for all parameters.
    """
    # Role-specific parameter ranges
    role_configs = {
        'bass': {
            'lfo1_waves': [1, 2, 3],        # TRI, SAW, SQR
            'lfo1_rates': [9, 12],           # 1/8, 1/4
            'lfo1_tvf': (8, 18), 'lfo1_tva': (3, 8),
            'lfo2_waves': [0, 1, 6],         # SIN, TRI, S&H
            'lfo2_rates': [12, 15, 18],      # 1/4, 1/2, 1
            'lfo2_tvf': (6, 14), 'lfo2_tva': (2, 6),
            'matrix_primary': [2, 36],       # CUT, FAT
            'matrix_secondary': [3, 37],     # RES, XMOD
            'sens_range': (6, 16),
        },
        'stab': {
            'lfo1_waves': [1, 3],            # TRI, SQR
            'lfo1_rates': [12, 15],          # 1/4, 1/2
            'lfo1_tvf': (10, 20), 'lfo1_tva': (2, 6),
            'lfo2_waves': [0, 7],            # SIN, CHS
            'lfo2_rates': [15, 18],          # 1/2, 1
            'lfo2_tvf': (8, 16), 'lfo2_tva': (1, 5),
            'matrix_primary': [2, 5],        # CUT, PAN
            'matrix_secondary': [3, 35],     # RES, PWM
            'sens_range': (5, 14),
        },
        'acid': {
            'lfo1_waves': [2, 6],            # SAW, S&H
            'lfo1_rates': [9, 12],           # 1/8, 1/4
            'lfo1_tvf': (15, 25), 'lfo1_tva': (2, 6),
            'lfo2_waves': [0, 3, 6],         # SIN, SQR, S&H
            'lfo2_rates': [12, 15],          # 1/4, 1/2
            'lfo2_tvf': (10, 18), 'lfo2_tva': (1, 4),
            'matrix_primary': [2, 17],       # CUT, LFO2-RATE
            'matrix_secondary': [3, 37],     # RES, XMOD
            'sens_range': (8, 18),
        },
        'pad': {
            'lfo1_waves': [0, 1, 6],         # SIN, TRI, S&H
            'lfo1_rates': [15, 18, 21],      # 1/2, 1, 2
            'lfo1_tvf': (12, 22), 'lfo1_tva': (4, 10),
            'lfo2_waves': [6, 8],            # S&H, VSIN
            'lfo2_rates': [18, 21],          # 1, 2
            'lfo2_tvf': (10, 20), 'lfo2_tva': (3, 8),
            'matrix_primary': [2, 36],       # CUT, FAT
            'matrix_secondary': [5, 35],     # PAN, PWM
            'sens_range': (5, 14),
        },
    }

    cfg = role_configs.get(role, role_configs['bass'])
    wave_names = {0: "SIN", 1: "TRI", 2: "SAW", 3: "SQR", 4: "TRP", 6: "S&H", 7: "CHS", 8: "VSIN"}
    note_names = {9: "1/8", 12: "1/4", 15: "1/2", 18: "1", 21: "2"}

    modulation_routes = []
    partials_raw = []

    for partial in range(4):
        block = _addr_add(fantom._zcore_base(part_idx), [0x00, 0x30 + (partial * 2), 0x00])

        # Pick LFO parameters from ranges
        lfo1_wave = random.choice(cfg['lfo1_waves'])
        lfo2_wave = random.choice([w for w in cfg['lfo2_waves'] if w != lfo1_wave] or cfg['lfo2_waves'])
        lfo1_note = random.choice(cfg['lfo1_rates'])
        lfo2_note = random.choice(cfg['lfo2_rates'])
        lfo1_tvf = random.randint(*cfg['lfo1_tvf'])
        lfo1_tva = random.randint(*cfg['lfo1_tva'])
        lfo2_tvf = random.randint(*cfg['lfo2_tvf'])
        lfo2_tva = random.randint(*cfg['lfo2_tva'])

        # === LFO1 ===
        fantom._send_dt1(_addr_add(block, [0x00, 0x00]), [lfo1_wave])
        fantom._send_dt1(_addr_add(block, [0x00, 0x01]), [1])  # Sync ON
        fantom._send_dt1(_addr_add(block, [0x00, 0x02]), [lfo1_note])
        fantom._send_dt1(_addr_add(block, [0x00, 0x19]), _nibbles(_signed_100(lfo1_tvf), 2))
        fantom._send_dt1(_addr_add(block, [0x00, 0x1B]), _nibbles(_signed_100(lfo1_tva), 2))

        # === LFO2 ===
        fantom._send_dt1(_addr_add(block, [0x00, 0x4F]), [lfo2_wave])
        fantom._send_dt1(_addr_add(block, [0x00, 0x50]), [1])  # Sync ON
        fantom._send_dt1(_addr_add(block, [0x00, 0x51]), [lfo2_note])
        fantom._send_dt1(_addr_add(block, [0x00, 0x68]), _nibbles(_signed_100(lfo2_tvf), 2))
        fantom._send_dt1(_addr_add(block, [0x00, 0x6A]), _nibbles(_signed_100(lfo2_tva), 2))

        # === Matrix 1: LFO1 → primary destination + LFO2-RATE ===
        primary_dest = random.choice(cfg['matrix_primary'])
        sens1 = random.randint(*cfg['sens_range'])
        sens1b = random.randint(4, 10)

        fantom._send_dt1(_addr_add(block, [0x00, 0x56]), [104])  # Source: LFO1
        fantom._send_dt1(_addr_add(block, [0x00, 0x57]), [primary_dest])
        fantom._send_dt1(_addr_add(block, [0x00, 0x58]), [_signed_63(sens1)])
        fantom._send_dt1(_addr_add(block, [0x00, 0x59]), [17])   # Dest2: LFO2-RATE
        fantom._send_dt1(_addr_add(block, [0x00, 0x5A]), [_signed_63(sens1b)])

        # === Matrix 2: LFO2 → two secondary destinations ===
        dest2a = random.choice(cfg['matrix_secondary'])
        dest2b = random.choice([d for d in SAFE_DESTINATIONS if d[0] not in {primary_dest, 17}])
        sens2a = random.randint(*cfg['sens_range'])
        sens2b = random.randint(3, 10)

        fantom._send_dt1(_addr_add(block, [0x00, 0x5F]), [105])  # Source: LFO2
        fantom._send_dt1(_addr_add(block, [0x00, 0x60]), [dest2a])
        fantom._send_dt1(_addr_add(block, [0x00, 0x61]), [_signed_63(sens2a)])
        fantom._send_dt1(_addr_add(block, [0x00, 0x62]), [dest2b[0]])
        fantom._send_dt1(_addr_add(block, [0x00, 0x63]), [_signed_63(sens2b)])

        # Store partial data
        partial_raw = {
            'lfo1_wave': lfo1_wave, 'lfo1_note': lfo1_note,
            'lfo1_tvf': lfo1_tvf, 'lfo1_tva': lfo1_tva,
            'lfo2_wave': lfo2_wave, 'lfo2_note': lfo2_note,
            'lfo2_tvf': lfo2_tvf, 'lfo2_tva': lfo2_tva,
            'matrix': {
                'm1_dest1': primary_dest, 'm1_sens1': sens1,
                'm1_dest2': 17, 'm1_sens2': sens1b,
                'm2_dest1': dest2a, 'm2_sens1': sens2a,
                'm2_dest2': dest2b[0], 'm2_sens2': sens2b,
            },
        }
        partials_raw.append(partial_raw)

        # Log routes for partial 0 only
        if partial == 0:
            modulation_routes = [
                f"LFO1 {wave_names.get(lfo1_wave)}@{note_names.get(lfo1_note)} TVF={lfo1_tvf} TVA={lfo1_tva}",
                f"LFO2 {wave_names.get(lfo2_wave)}@{note_names.get(lfo2_note)} TVF={lfo2_tvf} TVA={lfo2_tva}",
                f"LFO1→{[d[1] for d in SAFE_DESTINATIONS if d[0]==primary_dest][0] if any(d[0]==primary_dest for d in SAFE_DESTINATIONS) else primary_dest} sens=+{sens1}",
                f"LFO1→LFO2-RATE sens=+{sens1b}",
                f"LFO2→{[d[1] for d in SAFE_DESTINATIONS if d[0]==dest2a][0] if any(d[0]==dest2a for d in SAFE_DESTINATIONS) else dest2a} sens=+{sens2a}",
                f"LFO2→{dest2b[1]} sens=+{sens2b}",
            ]

    label = f"LFO matrix ({role}): " + "; ".join(modulation_routes)
    return label, {"partials": partials_raw}
```

### Objective Assessment — LFO Matrix

| Metric | How to Measure | Target |
|--------|---------------|--------|
| LFO1 and LFO2 different waves | Check wave indices | Must differ |
| Rate difference | LFO1 vs LFO2 note values | ≥ 1 step apart |
| Matrix has 4 unique paths | Count unique destinations | 4 (2 per LFO) |
| No pitch modulation | Check destinations | No dest = 0 (PITCH) |
| Sensitivity spread | Range of sens values | 4–18 across all paths |

---

## 9.3b — Step LFO (Sample & Hold Patterns)

### What Is a Step LFO?

Instead of a smooth continuous wave (sine, triangle), a step LFO generates
**discrete stepped values** at each cycle. On the Fantom's Z-Core engine,
this is the **S&H (Sample & Hold)** waveform — wave type `6`.

When synced to tempo, S&H creates **rhythmic random modulation** that
changes the filter cutoff, resonance, level, or pan in discrete jumps
on every beat subdivision. This is the secret weapon for:
- 303-style step filter patterns (random cutoff jumps)
- Rhythmic resonance biting
- Stepped volume gating (trance-gate style)
- Evolving texture that never repeats

### Why Step LFOs Matter for Tech House

Traditional LFOs (sine, triangle) create predictable, smooth modulation.
After 30 seconds, the listener's brain adapts and stops noticing it.

Step LFOs (S&H) create **unpredictable, discrete changes** that keep
the ear engaged. Each bar sounds slightly different from the last.
This is how professional tech house tracks stay interesting for 6 minutes
without adding new elements.

### S&H LFO Rate Combinations

The rate determines how often a new random value is generated:

| Rate Index | Sync Value | Musical Effect | Best For |
|-----------|-----------|----------------|----------|
| 9 | 1/8 note | Fast stepped filter (8 changes/bar) | Acid bass, aggressive leads |
| 12 | 1/4 note | Medium stepped filter (4 changes/bar) | House bass, chord stabs |
| 15 | 1/2 note | Slow stepped movement (2 changes/bar) | Pads, atmospheric textures |
| 18 | 1 bar | Very slow evolution (1 change/bar) | Background movement |
| 21 | 2 bars | Glacial change (1 change/2 bars) | Long-form pad evolution |

### Polyrhythmic Step LFOs

Use LFO1 and LFO2 at **different S&H rates** to create polyrhythmic
step patterns. Example:
- LFO1: S&H at 1/4 note → modulates filter cutoff (4 changes/bar)
- LFO2: S&H at 1/8 note → modulates resonance (8 changes/bar)

The result: filter and resonance change independently at different rates,
creating a complex evolving texture that sounds like a step sequencer
but is actually modulating the synth parameters in real time.

### Step LFO Patterns for Tech House

**Pattern 1: Acid Step Filter**
```
LFO1: S&H @ 1/8 → CUT (sens 12-18)
LFO2: S&H @ 1/4 → RES (sens 8-14)
Result: Fast random filter jumps with medium resonance movement
Use: Acid bass lines, 303-style patterns
```

**Pattern 2: Rhythmic Gate**
```
LFO1: S&H @ 1/4 → LEV (sens 10-16)
LFO2: TRI @ 1/8 → TVA (sens 4-8)
Result: Stepped volume changes with smooth tremolo underneath
Use: Chord stabs, percussive elements
```

**Pattern 3: Evolving Pad**
```
LFO1: S&H @ 1 bar → CUT (sens 8-14)
LFO2: S&H @ 2 bars → PAN (sens 6-12)
Result: Filter opens/closes once per bar, pan drifts every 2 bars
Use: Breakdown pads, atmospheric textures
```

**Pattern 4: Stepped Stereo**
```
LFO1: S&H @ 1/4 → PAN (sens 8-14)
LFO2: S&H @ 1/2 → FAT (sens 5-10)
Result: Rhythmic stereo jumps with slow detune changes
Use: Percussion layers, hi-hat textures
```

**Pattern 5: Cross-Mod Step Chaos**
```
LFO1: S&H @ 1/8 → CUT (sens 14-20)
LFO1: S&H @ 1/8 → LFO2-RATE (sens 8-14)
LFO2: S&H @ 1/4 → RES (sens 10-16)
LFO2: S&H @ 1/4 → XMOD (sens 5-10)
Result: Filter jumps drive the speed of resonance changes,
        creating self-modulating chaos that evolves constantly
Use: Acid leads, experimental bass
```

**Pattern 6: 303 Step Sequence Simulation**
```
LFO1: S&H @ 1/8 → CUT (sens 16-22)    ← High sensitivity = dramatic filter jumps
LFO2: SAW @ 1/4 → CUT (sens 4-8)       ← Smooth sweep underneath the steps
Result: Stepped filter pattern riding on a smooth sweep,
        mimicking a 303's step-sequenced filter cutoff
Use: Acid bass, lead lines
```

### Programmatic: Step LFO Setup

```python
def apply_step_lfo_matrix(fantom, part_idx, pattern='acid_step'):
    """
    Set up S&H (step) LFO patterns for tech house.
    Each pattern defines LFO waves, rates, and matrix destinations.
    """
    patterns = {
        'acid_step': {
            'desc': 'Fast stepped filter + medium resonance',
            'lfo1': {'wave': 6, 'rate': 9,  'tvf': (14, 22), 'tva': (2, 6)},
            'lfo2': {'wave': 6, 'rate': 12, 'tvf': (10, 18), 'tva': (2, 5)},
            'matrix1': [(2, 'CUT', 12, 20), (17, 'LFO2-RATE', 6, 12)],
            'matrix2': [(3, 'RES', 8, 16), (37, 'XMOD', 4, 10)],
        },
        'rhythmic_gate': {
            'desc': 'Stepped volume with smooth tremolo',
            'lfo1': {'wave': 6, 'rate': 12, 'tvf': (4, 10), 'tva': (10, 18)},
            'lfo2': {'wave': 1, 'rate': 9,  'tvf': (6, 12), 'tva': (4, 10)},
            'matrix1': [(4, 'LEV', 10, 18), (5, 'PAN', 4, 10)],
            'matrix2': [(12, 'TVA-LFO1', 6, 12), (2, 'CUT', 4, 10)],
        },
        'evolving_pad': {
            'desc': 'Slow filter + pan evolution',
            'lfo1': {'wave': 6, 'rate': 18, 'tvf': (10, 18), 'tva': (4, 10)},
            'lfo2': {'wave': 6, 'rate': 21, 'tvf': (8, 14),  'tva': (3, 8)},
            'matrix1': [(2, 'CUT', 8, 16), (36, 'FAT', 4, 10)],
            'matrix2': [(5, 'PAN', 6, 14), (35, 'PWM', 4, 10)],
        },
        'stepped_stereo': {
            'desc': 'Rhythmic pan jumps with detune drift',
            'lfo1': {'wave': 6, 'rate': 12, 'tvf': (4, 10), 'tva': (4, 10)},
            'lfo2': {'wave': 6, 'rate': 15, 'tvf': (6, 12), 'tva': (3, 8)},
            'matrix1': [(5, 'PAN', 8, 16), (4, 'LEV', 4, 10)],
            'matrix2': [(36, 'FAT', 5, 12), (3, 'RES', 3, 8)],
        },
        'crossmod_chaos': {
            'desc': 'Self-modulating stepped chaos',
            'lfo1': {'wave': 6, 'rate': 9,  'tvf': (16, 24), 'tva': (2, 6)},
            'lfo2': {'wave': 6, 'rate': 12, 'tvf': (12, 20), 'tva': (2, 5)},
            'matrix1': [(2, 'CUT', 14, 22), (17, 'LFO2-RATE', 8, 16)],
            'matrix2': [(3, 'RES', 10, 18), (37, 'XMOD', 6, 12)],
        },
        'acid_303_step': {
            'desc': '303-style step filter on smooth sweep',
            'lfo1': {'wave': 6, 'rate': 9,  'tvf': (18, 26), 'tva': (2, 6)},
            'lfo2': {'wave': 2, 'rate': 12, 'tvf': (4, 10),  'tva': (2, 5)},
            'matrix1': [(2, 'CUT', 16, 24), (17, 'LFO2-RATE', 4, 10)],
            'matrix2': [(3, 'RES', 8, 16), (36, 'FAT', 4, 8)],
        },
    }

    cfg = patterns.get(pattern, patterns['acid_step'])
    modulation_routes = []
    partials_raw = []

    for partial in range(4):
        block = _addr_add(fantom._zcore_base(part_idx), [0x00, 0x30 + (partial * 2), 0x00])

        # LFO1: S&H (step) configuration
        lfo1 = cfg['lfo1']
        lfo1_tvf = random.randint(*lfo1['tvf'])
        lfo1_tva = random.randint(*lfo1['tva'])
        fantom._send_dt1(_addr_add(block, [0x00, 0x00]), [lfo1['wave']])
        fantom._send_dt1(_addr_add(block, [0x00, 0x01]), [1])  # Sync ON
        fantom._send_dt1(_addr_add(block, [0x00, 0x02]), [lfo1['rate']])
        fantom._send_dt1(_addr_add(block, [0x00, 0x19]), _nibbles(_signed_100(lfo1_tvf), 2))
        fantom._send_dt1(_addr_add(block, [0x00, 0x1B]), _nibbles(_signed_100(lfo1_tva), 2))

        # LFO2: step or smooth configuration
        lfo2 = cfg['lfo2']
        lfo2_tvf = random.randint(*lfo2['tvf'])
        lfo2_tva = random.randint(*lfo2['tva'])
        fantom._send_dt1(_addr_add(block, [0x00, 0x4F]), [lfo2['wave']])
        fantom._send_dt1(_addr_add(block, [0x00, 0x50]), [1])  # Sync ON
        fantom._send_dt1(_addr_add(block, [0x00, 0x51]), [lfo2['rate']])
        fantom._send_dt1(_addr_add(block, [0x00, 0x68]), _nibbles(_signed_100(lfo2_tvf), 2))
        fantom._send_dt1(_addr_add(block, [0x00, 0x6A]), _nibbles(_signed_100(lfo2_tva), 2))

        # Matrix 1: LFO1 → two destinations
        m1 = cfg['matrix1']
        m1_dest1, m1_name1, m1_lo1, m1_hi1 = m1[0]
        m1_dest2, m1_name2, m1_lo2, m1_hi2 = m1[1]
        m1_sens1 = random.randint(m1_lo1, m1_hi1)
        m1_sens2 = random.randint(m1_lo2, m1_hi2)

        fantom._send_dt1(_addr_add(block, [0x00, 0x56]), [104])  # Source: LFO1
        fantom._send_dt1(_addr_add(block, [0x00, 0x57]), [m1_dest1])
        fantom._send_dt1(_addr_add(block, [0x00, 0x58]), [_signed_63(m1_sens1)])
        fantom._send_dt1(_addr_add(block, [0x00, 0x59]), [m1_dest2])
        fantom._send_dt1(_addr_add(block, [0x00, 0x5A]), [_signed_63(m1_sens2)])

        # Matrix 2: LFO2 → two destinations
        m2 = cfg['matrix2']
        m2_dest1, m2_name1, m2_lo1, m2_hi1 = m2[0]
        m2_dest2, m2_name2, m2_lo2, m2_hi2 = m2[1]
        m2_sens1 = random.randint(m2_lo1, m2_hi1)
        m2_sens2 = random.randint(m2_lo2, m2_hi2)

        fantom._send_dt1(_addr_add(block, [0x00, 0x5F]), [105])  # Source: LFO2
        fantom._send_dt1(_addr_add(block, [0x00, 0x60]), [m2_dest1])
        fantom._send_dt1(_addr_add(block, [0x00, 0x61]), [_signed_63(m2_sens1)])
        fantom._send_dt1(_addr_add(block, [0x00, 0x62]), [m2_dest2])
        fantom._send_dt1(_addr_add(block, [0x00, 0x63]), [_signed_63(m2_sens2)])

        # Store
        partials_raw.append({
            'lfo1_wave': lfo1['wave'], 'lfo1_rate': lfo1['rate'],
            'lfo1_tvf': lfo1_tvf, 'lfo1_tva': lfo1_tva,
            'lfo2_wave': lfo2['wave'], 'lfo2_rate': lfo2['rate'],
            'lfo2_tvf': lfo2_tvf, 'lfo2_tva': lfo2_tva,
            'matrix': {
                'm1_dest1': m1_dest1, 'm1_name': m1_name1, 'm1_sens': m1_sens1,
                'm1_dest2': m1_dest2, 'm1_name2': m1_name2, 'm1_sens2': m1_sens2,
                'm2_dest1': m2_dest1, 'm2_name': m2_name1, 'm2_sens': m2_sens1,
                'm2_dest2': m2_dest2, 'm2_name2': m2_name2, 'm2_sens2': m2_sens2,
            },
        })

        if partial == 0:
            wave_names = {0: "SIN", 1: "TRI", 2: "SAW", 3: "SQR", 4: "TRP", 6: "S&H", 7: "CHS", 8: "VSIN"}
            rate_names = {9: "1/8", 12: "1/4", 15: "1/2", 18: "1bar", 21: "2bar"}
            modulation_routes = [
                f"LFO1 {wave_names.get(lfo1['wave'])}@{rate_names.get(lfo1['rate'])} TVF={lfo1_tvf} TVA={lfo1_tva}",
                f"LFO2 {wave_names.get(lfo2['wave'])}@{rate_names.get(lfo2['rate'])} TVF={lfo2_tvf} TVA={lfo2_tva}",
                f"LFO1→{m1_name1} sens=+{m1_sens1}; LFO1→{m1_name2} sens=+{m1_sens2}",
                f"LFO2→{m2_name1} sens=+{m2_sens1}; LFO2→{m2_name2} sens=+{m2_sens2}",
            ]

    label = f"Step LFO ({pattern}): " + "; ".join(modulation_routes)
    return label, {"pattern": pattern, "partials": partials_raw}


# Step LFO patterns by role
STEP_LFO_ROLES = {
    'acid_bass':    ['acid_step', 'crossmod_chaos', 'acid_303_step'],
    'house_bass':   ['acid_step', 'stepped_stereo'],
    'chord_stab':   ['rhythmic_gate', 'stepped_stereo'],
    'acid_lead':    ['crossmod_chaos', 'acid_303_step', 'acid_step'],
    'pad':          ['evolving_pad', 'stepped_stereo'],
    'percussion':   ['stepped_stereo', 'rhythmic_gate'],
}
```

### Combining Step LFO with Smooth LFO

The most interesting textures come from mixing step and smooth modulation
on the same partial:

```python
def apply_hybrid_lfo_matrix(fantom, part_idx, role='acid_bass'):
    """
    Hybrid: LFO1 = S&H (step), LFO2 = smooth wave.
    Step provides rhythmic novelty, smooth provides organic movement.
    """
    # Step LFO choices
    step_waves = [6, 6, 6]  # S&H for step
    step_rates = [9, 12]     # 1/8 or 1/4 note

    # Smooth LFO choices
    smooth_waves = [0, 1, 2, 8]  # SIN, TRI, SAW, VSIN
    smooth_rates = [12, 15, 18]   # 1/4, 1/2, 1 bar

    # Destination pairs
    dest_pairs = {
        'acid_bass': {
            'step': [(2, 'CUT', 14, 22), (17, 'LFO2-RATE', 6, 12)],
            'smooth': [(3, 'RES', 6, 14), (36, 'FAT', 4, 10)],
        },
        'house_bass': {
            'step': [(2, 'CUT', 8, 16), (4, 'LEV', 4, 10)],
            'smooth': [(36, 'FAT', 5, 12), (5, 'PAN', 4, 10)],
        },
        'stab': {
            'step': [(4, 'LEV', 8, 16), (5, 'PAN', 6, 12)],
            'smooth': [(2, 'CUT', 6, 14), (35, 'PWM', 4, 10)],
        },
        'pad': {
            'step': [(2, 'CUT', 6, 14), (3, 'RES', 4, 10)],
            'smooth': [(5, 'PAN', 6, 14), (36, 'FAT', 4, 10)],
        },
    }

    cfg = dest_pairs.get(role, dest_pairs['acid_bass'])

    for partial in range(4):
        block = _addr_add(fantom._zcore_base(part_idx), [0x00, 0x30 + (partial * 2), 0x00])

        # LFO1: Step (S&H)
        lfo1_wave = 6  # S&H
        lfo1_rate = random.choice(step_rates)
        lfo1_tvf = random.randint(12, 22)
        lfo1_tva = random.randint(2, 8)

        fantom._send_dt1(_addr_add(block, [0x00, 0x00]), [lfo1_wave])
        fantom._send_dt1(_addr_add(block, [0x00, 0x01]), [1])
        fantom._send_dt1(_addr_add(block, [0x00, 0x02]), [lfo1_rate])
        fantom._send_dt1(_addr_add(block, [0x00, 0x19]), _nibbles(_signed_100(lfo1_tvf), 2))
        fantom._send_dt1(_addr_add(block, [0x00, 0x1B]), _nibbles(_signed_100(lfo1_tva), 2))

        # LFO2: Smooth
        lfo2_wave = random.choice(smooth_waves)
        lfo2_rate = random.choice(smooth_rates)
        lfo2_tvf = random.randint(6, 16)
        lfo2_tva = random.randint(2, 8)

        fantom._send_dt1(_addr_add(block, [0x00, 0x4F]), [lfo2_wave])
        fantom._send_dt1(_addr_add(block, [0x00, 0x50]), [1])
        fantom._send_dt1(_addr_add(block, [0x00, 0x51]), [lfo2_rate])
        fantom._send_dt1(_addr_add(block, [0x00, 0x68]), _nibbles(_signed_100(lfo2_tvf), 2))
        fantom._send_dt1(_addr_add(block, [0x00, 0x6A]), _nibbles(_signed_100(lfo2_tva), 2))

        # Matrix: LFO1 (step) → step destinations
        step_dests = cfg['step']
        d1, n1, lo1, hi1 = step_dests[0]
        d2, n2, lo2, hi2 = step_dests[1]
        fantom._send_dt1(_addr_add(block, [0x00, 0x56]), [104])
        fantom._send_dt1(_addr_add(block, [0x00, 0x57]), [d1])
        fantom._send_dt1(_addr_add(block, [0x00, 0x58]), [_signed_63(random.randint(lo1, hi1))])
        fantom._send_dt1(_addr_add(block, [0x00, 0x59]), [d2])
        fantom._send_dt1(_addr_add(block, [0x00, 0x5A]), [_signed_63(random.randint(lo2, hi2))])

        # Matrix: LFO2 (smooth) → smooth destinations
        smooth_dests = cfg['smooth']
        d3, n3, lo3, hi3 = smooth_dests[0]
        d4, n4, lo4, hi4 = smooth_dests[1]
        fantom._send_dt1(_addr_add(block, [0x00, 0x5F]), [105])
        fantom._send_dt1(_addr_add(block, [0x00, 0x60]), [d3])
        fantom._send_dt1(_addr_add(block, [0x00, 0x61]), [_signed_63(random.randint(lo3, hi3))])
        fantom._send_dt1(_addr_add(block, [0x00, 0x62]), [d4])
        fantom._send_dt1(_addr_add(block, [0x00, 0x63]), [_signed_63(random.randint(lo4, hi4))])

    wave_names = {0: "SIN", 1: "TRI", 2: "SAW", 6: "S&H", 8: "VSIN"}
    rate_names = {9: "1/8", 12: "1/4", 15: "1/2", 18: "1bar"}
    return (f"Hybrid LFO ({role}): "
            f"LFO1 S&H@{rate_names.get(lfo1_rate)} → {n1},{n2}; "
            f"LFO2 {wave_names.get(lfo2_wave)}@{rate_names.get(lfo2_rate)} → {n3},{n4}")
```

### Objective Assessment — Step LFO

| Metric | How to Measure | Target |
|--------|---------------|--------|
| LFO1 wave is S&H | Check wave parameter == 6 | Must be 6 |
| LFO1 and LFO2 different rates | Compare rate values | ≥ 1 step apart |
| Step destinations include CUT or LEV | Check matrix dest codes | 2 (CUT) or 4 (LEV) |
| Sensitivity creates audible change | Record audio, compare bars | Each bar should sound different |
| No pitch modulation | Check matrix | No dest = 0 (PITCH) |

---

## 9.4 — MFX Selection for Tech House (Range-Based)

### MFX Parameter Ranges

| Role | MFX | Parameter Ranges |
|------|-----|-----------------|
| **Acid Bass** | 5 Super Filter | cutoff: 60–90, resonance: 20–40, depth: 15–25 |
| **House Bass** | 45 LOFI Compress | pre_filter: 2–5, balance: 60–80 |
| **Sub Bass** | 7 Enhancer | sensitivity: 18–30, mix: 15–25 |
| **Chord Stab** | 23 Chorus | depth: 18–30, wet: 20–30% |
| **Acid Lead** | 8 Auto Wah | manual: 40–60, peak: 20–30, depth: 12–22 |
| **Pad** | 24 Flanger | depth: 10–20, feedback: 40–55, wet: 15–25% |
| **Drums** | 46 Bit Crusher | sr: 55–65%, bit: 8–12, filter: 45–55% |

### Programmatic

```python
def apply_acid_bass_mfx(fantom, part_idx):
    """Super Filter MFX with randomized ranges."""
    params = {
        1: 0,                                           # LPF
        2: 0,                                           # -12dB slope
        3: random.randint(60, 90),                      # Cutoff range
        4: random.randint(20, 40),                      # Resonance range
        5: 0,                                           # Gain
        6: 1,                                           # Mod ON
        7: random.choice([0, 2, 6]),                    # Wave: SIN, SAW, or S&H
        8: 1,                                           # Sync ON
        10: random.choice([9, 12, 15]),                 # Rate: 1/8, 1/4, or 1/2
        11: random.randint(15, 25),                     # Depth range
        12: random.randint(0, 10),                      # Attack
        13: 100,                                        # Level
    }
    fantom._set_mfx_type_and_params(fantom._tone_mfx_base(part_idx), 5, params)
    return f"MFX05 Super Filter cutoff={params[3]} res={params[4]} depth={params[11]}"

def apply_stab_chorus_mfx(fantom, part_idx):
    """Chorus MFX with randomized ranges."""
    depth = random.randint(18, 30)
    wet = random.randint(20, 30)
    params = {
        1: random.choice([0, 1]),                       # Filter OFF/LPF
        2: random.randint(8, 13),                       # Cutoff
        3: random.randint(10, 20),                      # Pre-delay
        4: 1,                                           # Sync ON
        6: random.choice([9, 12]),                      # Rate: 1/8 or 1/4
        7: depth,                                       # Depth
        8: random.choice([90, 120, 150]),               # Phase
        9: 15,                                          # Low gain center
        10: random.randint(15, 17),                     # High gain
        11: wet,                                        # Balance
        12: 100,                                        # Level
    }
    fantom._set_mfx_type_and_params(fantom._tone_mfx_base(part_idx), 23, params)
    return f"MFX23 Chorus depth={depth} wet={wet}%"

def apply_drum_crush_mfx(fantom, part_idx):
    """Bit Crusher MFX with randomized ranges."""
    sr = random.randint(55, 65)
    bit = random.randint(8, 12)
    filt = random.randint(45, 55)
    params = {
        1: _percent_to_127(sr),
        2: bit,
        3: _percent_to_127(filt),
        4: random.randint(15, 17),
        5: random.randint(14, 16),
        6: 100,
    }
    fantom._set_mfx_type_and_params(fantom._drum_mfx_base(part_idx), 46, params)
    return f"MFX46 Bit Crusher sr={sr}% bit={bit} filter={filt}%"
```

---

## 9.5 — Drum Kit Editing via SysEx

### Drum Kit Base Address
```
Part N kit base: [0x02, 0x30 + (N * 2), 0x00, 0x00]
Drum instrument set: [0x03, N * 4, 0x00, 0x00]
```

### Per-Note Drum Parameters

```
Key offset: (note - 21) * 5
Parameters at key_base:
  [0x00, 0x00] = Level (0-127)
  [0x00, 0x01] = Pan (0-127, 64=center)
  [0x00, 0x02] = Tune (-64 to +63, encoded as 0-127)
  [0x00, 0x03] = Decay (0-127)
  [0x00, 0x04] = Attack (0-127)
```

### Drum Parameter Ranges

| Drum | Level Range | Pan Range | Decay Range | Tune Range |
|------|------------|-----------|-------------|------------|
| Kick (36) | 120–127 | 64 (center) | 70–90 | key ± 50 cents |
| Clap (39) | 105–118 | 60–68 | 50–70 | 0 (no tune) |
| Closed Hat (42) | 90–108 | 55–73 | 25–40 | 0 |
| Open Hat (46) | 85–100 | 50–78 | 70–95 | 0 |
| Ride (51) | 70–90 | 60–68 | 80–100 | 0 |
| Crash (49) | 90–110 | 62–66 | 90–110 | 0 |
| Rimshot (37) | 50–70 | 58–70 | 30–50 | 0 |

### Programmatic

```python
def setup_909_kit_for_tech_house(fantom, part_idx, key='G'):
    """Configure 909 kit with randomized parameter ranges."""
    kick_cents = _key_to_cents(key)

    # Kick: tune to key, full level, center, medium decay
    tune_drum_note(fantom, part_idx, 36,
                   tune_cents=kick_cents + random.randint(-20, 20),
                   level=random.randint(120, 127),
                   pan=64,
                   decay=random.randint(70, 90))

    # Clap: center-ish, strong, medium decay
    tune_drum_note(fantom, part_idx, 39,
                   level=random.randint(105, 118),
                   pan=random.randint(60, 68),
                   decay=random.randint(50, 70))

    # Closed hat: slightly off-center, short decay
    tune_drum_note(fantom, part_idx, 42,
                   level=random.randint(90, 108),
                   pan=random.randint(55, 73),
                   decay=random.randint(25, 40))

    # Open hat: wider pan, longer decay
    tune_drum_note(fantom, part_idx, 46,
                   level=random.randint(85, 100),
                   pan=random.randint(50, 78),
                   decay=random.randint(70, 95))

    # Ride: lower level, center-ish
    tune_drum_note(fantom, part_idx, 51,
                   level=random.randint(70, 90),
                   pan=random.randint(60, 68),
                   decay=random.randint(80, 100))

    # Crash: accent level
    tune_drum_note(fantom, part_idx, 49,
                   level=random.randint(90, 110),
                   pan=random.randint(62, 66),
                   decay=random.randint(90, 110))

    # Rimshot: quiet syncopation
    tune_drum_note(fantom, part_idx, 37,
                   level=random.randint(50, 70),
                   pan=random.randint(58, 70),
                   decay=random.randint(30, 50))

def _key_to_cents(key_name):
    note_map = {'C': 0, 'C#': 100, 'D': 200, 'D#': 300, 'E': 400,
                'F': 500, 'F#': 600, 'G': 700, 'G#': 800, 'A': 900,
                'A#': 1000, 'B': 1100}
    return note_map.get(key_name, 0) - 600
```

---

## 9.6 — Complete Tech House Fantom Setup

### One-Call Setup (Range-Based, No Sidechain)

```python
def setup_fantom_for_tech_house(fantom, key='G', bpm=126):
    """
    Complete Fantom setup for tech house.
    All parameters use randomized ranges. Sidechain is post-production.
    """
    results = {}

    # Select patches (random from curated list)
    bass_patch = random.choice(TECH_HOUSE_PATCHES['house_bass'])
    fantom.select_patch(channel=4, msb=bass_patch['msb'],
                        lsb=bass_patch['lsb'], pc=bass_patch['pc'])
    results['bass'] = bass_patch['name']

    acid_patch = random.choice(TECH_HOUSE_PATCHES['acid_bass'])
    fantom.select_patch(channel=0, msb=acid_patch['msb'],
                        lsb=acid_patch['lsb'], pc=acid_patch['pc'])
    results['acid'] = acid_patch['name']

    stab_patch = random.choice(TECH_HOUSE_PATCHES['chord_stab'])
    fantom.select_patch(channel=1, msb=stab_patch['msb'],
                        lsb=stab_patch['lsb'], pc=stab_patch['pc'])
    results['stab'] = stab_patch['name']

    pad_patch = random.choice(TECH_HOUSE_PATCHES['dark_pad'])
    fantom.select_patch(channel=2, msb=pad_patch['msb'],
                        lsb=pad_patch['lsb'], pc=pad_patch['pc'])
    results['pad'] = pad_patch['name']

    drum_patch = random.choice(TECH_HOUSE_PATCHES['drum_kit'])
    fantom.select_patch(channel=9, msb=drum_patch['msb'],
                        lsb=drum_patch['lsb'], pc=drum_patch['pc'])
    results['drums'] = drum_patch['name']

    # Apply tone parameters (all randomized within ranges)
    results['bass_tone'] = apply_bass_tone(fantom, 4, style='house')
    results['acid_tone'] = apply_bass_tone(fantom, 0, style='acid')
    results['stab_tone'] = apply_stab_tone(fantom, 1)

    # Apply dual LFO matrix (role-specific ranges)
    results['bass_lfo'] = apply_tech_house_lfo_matrix(fantom, 4, role='bass')
    results['acid_lfo'] = apply_tech_house_lfo_matrix(fantom, 0, role='acid')
    results['stab_lfo'] = apply_tech_house_lfo_matrix(fantom, 1, role='stab')
    results['pad_lfo'] = apply_tech_house_lfo_matrix(fantom, 2, role='pad')

    # Apply MFX (randomized within ranges)
    results['acid_mfx'] = apply_acid_bass_mfx(fantom, 0)
    results['stab_mfx'] = apply_stab_chorus_mfx(fantom, 1)
    results['drum_mfx'] = apply_drum_crush_mfx(fantom, 0)

    # Setup drum kit (randomized within ranges)
    setup_909_kit_for_tech_house(fantom, 0, key=key)
    results['drum_tuning'] = f'Kick tuned to {key} ± random cents'

    # NOTE: Sidechain pumping is applied in post-production (Skill 06)
    # NOT via Fantom LFO — keeps recorded signal clean for mixing

    return results
```

---

## 9.7 — Mapping Skills to Fantom SysEx

| Skill | Parameter | Fantom SysEx Route | Fixed or Range? |
|-------|-----------|-------------------|-----------------|
| **Bass oscillator** | waveform | Select SL-TB Saw/Sqr patch | Random from list |
| **Bass filter cutoff** | cutoff_hz | TVF Cutoff [0x00, 0x20, 0x01] | Range 55–95 |
| **Bass filter resonance** | resonance % | TVF Resonance [0x00, 0x20, 0x02] | Range 10–50 |
| **Bass filter env** | decay_ms | TVF Env Decay [0x00, 0x20, 0x06] | Range 30–60 |
| **Bass amp env** | attack_ms | TVA Env Attack [0x00, 0x24, 0x00] | Range 0–3 |
| **Filter sweep** | LFO→CUT | Matrix dest=2, sens range 6–18 | Both LFOs |
| **Resonance movement** | LFO→RES | Matrix dest=3, sens range 4–14 | LFO2 |
| **Stereo movement** | LFO→PAN | Matrix dest=5, sens range 5–12 | LFO2 |
| **Tone fatness** | LFO→FAT | Matrix dest=36, sens range 3–10 | LFO2 |
| **Cross-mod** | LFO→XMOD | Matrix dest=37, sens range 3–10 | LFO2 |
| **Step filter (acid)** | S&H LFO→CUT | LFO1 wave=6 @ 1/8, dest=2, sens 12–22 | Step pattern |
| **Step resonance** | S&H LFO→RES | LFO2 wave=6 @ 1/4, dest=3, sens 8–16 | Step pattern |
| **Step gate** | S&H LFO→LEV | LFO1 wave=6 @ 1/4, dest=4, sens 10–18 | Step pattern |
| **Step stereo** | S&H LFO→PAN | LFO1 wave=6 @ 1/4, dest=5, sens 8–16 | Step pattern |
| **Step chaos** | S&H→CUT + S&H→LFO2-RATE | LFO1→CUT + LFO1→LFO2-RATE, self-modulating | Step + cross-mod |
| **Hybrid step+smooth** | S&H LFO1 + smooth LFO2 | LFO1=6 (step) → CUT; LFO2=0/1 (smooth) → RES,FAT | Best of both |
| **Acid squelch** | MFX Super Filter | cutoff 60–90, res 20–40 | Range |
| **Stab chorus** | MFX Chorus | depth 18–30, wet 20–30% | Range |
| **Drum crush** | MFX Bit Crusher | sr 55–65%, bit 8–12 | Range |
| **Kick tuning** | Drum key tune | cents = key ± 50 | Range |
| **Hat pan** | Drum pan | 55–73 (slightly off center) | Range |
| **Sidechain pump** | **Post-production** | **NOT on Fantom** | Skill 06 |
