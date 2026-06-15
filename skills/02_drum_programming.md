# Skill 02: Drum Programming

## What the Producer Does
Write rhythmic patterns for kick, clap, hi-hats, and percussion. Build from
a 1-bar loop into a full arrangement. Apply velocity variation and swing
for human feel. Layer elements for different sections (intro = sparse,
drop = full, breakdown = stripped).

---

## 2.1 — Kick Pattern (4-on-the-floor)

### Real-World Action
Place kick on every quarter note. Add ghost kicks on 16th-note offbeats
for groove variation (every 4th or 8th bar).

### Programmatic

```python
import mido

def write_kick_pattern(tpb=480, bars=1, ghost_kick_probability=0.2):
    """4-on-the-floor with optional ghost kicks."""
    events = []
    sixteenth = tpb // 4
    for bar in range(bars):
        bs = bar * tpb * 4
        # Four on the floor
        for beat in range(4):
            events.append({'time': bs + beat * tpb, 'note': 36, 'velocity': 127})
        # Ghost kick on "and" of beat 4 (10-20% of bars)
        if random.random() < ghost_kick_probability:
            events.append({'time': bs + 3 * tpb + 2 * sixteenth, 'note': 36, 'velocity': 60})
    return events
```

### Objective Assessment

| Metric | How to Measure | Target | Why |
|--------|---------------|--------|-----|
| Kick density | Kicks per bar | 4.0–4.5 | Four-on-floor + occasional ghost |
| Ghost velocity | Velocity of ghost kicks | 40–65 (vs 127 main) | Should be felt, not heard |
| Spacing regularity | Std dev of inter-kick intervals | < 5 ticks | Tight timing essential for groove |

---

## 2.2 — Clap Placement

### Real-World Action
Clap on beats 2 and 4. Velocity 100-120. Optional: offset 1-2 ticks late
for a laid-back feel, or layer with snare on chorus sections.

### Programmatic

```python
def write_clap_pattern(tpb=480, bars=1, offset_ticks=0):
    events = []
    for bar in range(bars):
        bs = bar * tpb * 4
        for beat_pos in [1, 3]:  # beats 2 and 4
            events.append({
                'time': bs + beat_pos * tpb + offset_ticks,
                'note': 39,  # GM Clap
                'velocity': random.randint(105, 120)
            })
    return events
```

### Objective Assessment

| Metric | Target | Why |
|--------|--------|-----|
| Claps per bar | 2 (beats 2 & 4) | Standard backbeat |
| Velocity consistency | Std dev < 8 | Human but consistent |
| Timing offset | 0–5 ticks late max | More than 5 = sloppy |

---

## 2.3 — Hi-Hat Programming

### Real-World Action
16th-note closed hats with velocity pattern:
- Downbeats (1, 5, 9, 13): 90-110
- Upbeats (3, 7, 11, 15): 70-90
- Ghost 16ths (2, 4, 6, 8...): 30-60
Apply 5-10% swing to offbeat 16ths.

### Programmatic

```python
def write_hat_pattern(tpb=480, bars=1, swing_pct=0.08):
    """16th-note hats with velocity groove and swing."""
    events = []
    sixteenth = tpb // 4
    swing_offset = int(sixteenth * swing_pct)  # swing on odd 16ths

    for bar in range(bars):
        bs = bar * tpb * 4
        for step in range(16):
            pos = step * sixteenth
            # Apply swing to odd-numbered 16ths
            if step % 2 == 1:
                pos += swing_offset

            # Velocity pattern
            if step % 4 == 0:      # downbeats
                vel = random.randint(95, 110)
            elif step % 4 == 2:    # upbeats
                vel = random.randint(75, 90)
            else:                   # ghost 16ths
                vel = random.randint(35, 55)

            events.append({'time': bs + pos, 'note': 42, 'velocity': vel})

    return events
```

### Objective Assessment

| Metric | How to Measure | Target | Why |
|--------|---------------|--------|-----|
| Hat density | Hats per bar | 16 | Full 16th-note pattern |
| Velocity range | max_vel - min_vel | 40–70 | Creates groove feel |
| Swing amount | Offset of odd 16ths / sixteenth | 5–12% | Tech house sweet spot |
| Downbeat emphasis | Avg downbeat vel / avg ghost vel | 1.8–2.5x | Groove dynamics |

---

## 2.4 — Open Hat Placement

### Real-World Action
One open hat per bar on the "and" of beat 2 (step 6) or beat 4 (step 14).
Velocity 70-90. Chorus: add second open hat.

### Programmatic

```python
def write_open_hat_pattern(tpb=480, bars=1, positions='and_of_2'):
    events = []
    sixteenth = tpb // 4
    for bar in range(bars):
        bs = bar * tpb * 4
        if positions == 'and_of_2':
            pos = 1 * tpb + 2 * sixteenth  # step 6
        elif positions == 'and_of_4':
            pos = 3 * tpb + 2 * sixteenth  # step 14
        else:
            pos = random.choice([
                1 * tpb + 2 * sixteenth,
                3 * tpb + 2 * sixteenth,
            ])
        events.append({'time': bs + pos, 'note': 46, 'velocity': random.randint(75, 90)})
    return events
```

---

## 2.5 — Percussion Layers

### Real-World Action
Add shaker (steady 8ths), rimshot (syncopated), and/or tambourine (offbeats).
Each at lower velocity than kick/clap. Mixed -12 to -18 dB below kick.

### Programmatic

```python
def write_percussion_pattern(tpb=480, bars=1):
    events = []
    sixteenth = tpb // 4
    for bar in range(bars):
        bs = bar * tpb * 4
        # Shaker: 8th notes, low velocity
        for i in range(8):
            events.append({
                'time': bs + i * 2 * sixteenth,
                'note': 70,  # GM Maracas/Shaker
                'velocity': random.randint(40, 60)
            })
        # Rimshot: syncopated accents on steps 4, 8, 12, 16
        for step in [3, 7, 11, 15]:
            if random.random() < 0.6:
                events.append({
                    'time': bs + step * sixteenth,
                    'note': 37,  # GM Rimshot
                    'velocity': random.randint(45, 65)
                })
    return events
```

---

## 2.6 — Section Layering

### The Build-Up Process

The producer builds the arrangement by adding/removing layers per section.
This is NOT just duplicating the full loop — it's sculpting energy.

| Section | Kick | Clap | Closed Hat | Open Hat | Percussion |
|---------|------|------|------------|----------|------------|
| Intro (bars 1-16) | ✓ (HP filtered) | ✗ | ✓ (low vel) | ✗ | ✗ |
| Build 1 (17-24) | ✓ | ✓ (bar 17) | ✓ | ✓ (bar 21) | ✗ |
| Drop 1 (25-40) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Breakdown (41-48) | ✗ | ✗ | ✓ (very low) | ✗ | ✗ |
| Build 2 (49-56) | ✓ (bar 49) | ✓ (bar 53) | ✓ | ✓ | ✗ |
| Drop 2 (57-72) | ✓ | ✓ | ✓ | ✓ | ✓ + extra |
| Outro (73-88) | ✓ | ✗ (bar 79) | ✓ (fading) | ✗ (bar 77) | ✗ (bar 75) |

### Programmatic Section Generation

```python
def generate_section_drums(bar_start, bar_count, section_type, tpb=480):
    """Generate drum events for a specific arrangement section."""
    all_events = []

    for bar in range(bar_start, bar_start + bar_count):
        if section_type == 'intro':
            all_events.extend(write_kick_pattern(tpb, 1))
            all_events.extend(write_hat_pattern(tpb, 1))  # full hats, low vel
            # Reduce all velocities by 30%
            for e in all_events[-16:]:
                e['velocity'] = int(e['velocity'] * 0.7)

        elif section_type == 'drop':
            all_events.extend(write_kick_pattern(tpb, 1, ghost_kick_probability=0.25))
            all_events.extend(write_clap_pattern(tpb, 1))
            all_events.extend(write_hat_pattern(tpb, 1))
            all_events.extend(write_open_hat_pattern(tpb, 1))
            all_events.extend(write_percussion_pattern(tpb, 1))

        elif section_type == 'breakdown':
            all_events.extend(write_hat_pattern(tpb, 1))
            for e in all_events[-16:]:
                e['velocity'] = int(e['velocity'] * 0.5)

        # ... other sections

    return all_events
```

### Objective Assessment — Full Arrangement

| Metric | How to Measure | Target |
|--------|---------------|--------|
| Element count per section | Count unique notes active | Intro: 1-2, Drop: 5-7 |
| Energy curve | RMS of all events per 8-bar block | Smooth rise → peak → dip → peak → fade |
| Arrangement length | Total bars * bar_duration | 5-7 minutes at 126 BPM |
| 8-bar rule compliance | Any change every 8 bars | 100% — something must change |

---

## Iterative Refinement Loop

1. **Generate** full arrangement drums
2. **Measure** energy curve (RMS per 8-bar block)
3. **Compare** to ideal energy shape: ramp → peak → valley → peak → fade
4. **If** energy doesn't dip at breakdown: remove elements
5. **If** energy doesn't peak at drops: add ghost kicks, velocity +10%
6. **If** intro energy > 30% of drop energy: reduce intro element count
7. **Max 3 iterations**, then move on
