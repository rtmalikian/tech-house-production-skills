# Skill 04: Synth & Melodic Elements

## What the Producer Does
Add chord stabs (short, punchy minor chords), acid lines (303-style),
and atmospheric pads. Tech house is rhythm-focused — melodic elements
are minimal, repetitive, and serve the groove.

---

## 4.1 — Chord Stabs

### Sound Design
Saw wave → low-pass filter → fast attack, short decay, zero sustain.
Filter envelope: fast attack, 100-200ms decay, slight resonance.

### Programmatic

```python
def generate_chord_stab(chord_notes, sr=44100, duration_s=0.2):
    """Generate a minor chord stab from scratch."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    signal = np.zeros_like(t)

    for note_hz in chord_notes:
        # Saw wave per note
        for k in range(1, 10):
            signal += ((-1)**(k+1)) * np.sin(2 * np.pi * k * note_hz * t) / k

    # Amplitude envelope (fast attack, quick decay)
    env = np.exp(-t * 8)  # Exponential decay, ~125ms to -20 dB
    signal *= env

    # Low-pass filter
    sos = butter(3, 2000, btype='low', fs=sr, output='sos')
    signal = sosfiltfilt(sos, signal)

    signal /= np.max(np.abs(signal)) * 0.9
    return signal
```

### Placement Rules
- Stabs on beat 1, sometimes beat 3
- Duration: 1/8 to 1/4 note (120-250ms at 126 BPM)
- Velocity: 70-90 (not overpowering)
- Only play in drop sections

---

## 4.2 — Acid Line (303-style)

### Sound Design
Saw or square wave → resonant low-pass filter → accent and slide.
Filter cutoff modulated by envelope, high resonance for "squelch."

### Programmatic

```python
def generate_acid_phrase(root_hz, sr=44100, bpm=126, steps=16):
    """Generate a 1-bar 303-style acid pattern."""
    sixteenth_dur = 60.0 / bpm / 4  # Duration of one 16th note
    total_dur = sixteenth_dur * steps
    t = np.linspace(0, total_dur, int(sr * total_dur), endpoint=False)
    signal = np.zeros_like(t)

    # Note sequence (root, fifth, octave, chromatic approaches)
    note_multipliers = [1, 1.5, 2, 1.67, 1.33, 1, 1.5, 2,
                        1, 1.25, 1.5, 1.33, 1, 1.5, 1, 2]
    accent_pattern = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0]

    samples_per_step = int(sr * sixteenth_dur)
    for step in range(steps):
        start = step * samples_per_step
        end = min(start + samples_per_step, len(t))
        step_t = t[start:end] - t[start]

        freq = root_hz * note_multipliers[step % len(note_multipliers)]

        # Square wave (303 character)
        note_signal = np.sign(np.sin(2 * np.pi * freq * step_t))

        # Resonant filter envelope (opens on accents)
        cutoff = 300 + accent_pattern[step % len(accent_pattern)] * 1500
        cutoff_env = cutoff * np.exp(-step_t * 15) + 300

        # Simplified: apply filter at average cutoff
        avg_cutoff = np.mean(cutoff_env)
        if avg_cutoff < sr / 2 - 100:
            sos = butter(2, avg_cutoff, btype='low', fs=sr, output='sos')
            note_signal = sosfiltfilt(sos, note_signal) if len(note_signal) > 12 else note_signal

        # Note envelope
        note_env = np.exp(-step_t * 10)
        signal[start:end] += note_signal * note_env * (1.0 if accent_pattern[step % len(accent_pattern)] else 0.5)

    signal = np.tanh(signal * 1.5) / 1.5  # Saturation
    signal /= np.max(np.abs(signal)) * 0.9
    return signal
```

---

## 4.3 — Atmospheric Pads

### Real-World Action
Detuned saw waves with slow filter modulation. Used in breakdowns only.
Not lush — filtered, dark, evolving.

### Programmatic

```python
def generate_pad(chord_notes, sr=44100, duration_s=8.0):
    """Generate a filtered pad for breakdown sections."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    signal = np.zeros_like(t)

    for note_hz in chord_notes:
        # Detuned saw waves (±5 cents)
        for detune in [-5, 0, 5]:
            freq = note_hz * (2 ** (detune / 1200))
            for k in range(1, 8):
                signal += ((-1)**(k+1)) * np.sin(2 * np.pi * k * freq * t) / k * 0.3

    # Slow filter sweep (LFO on cutoff)
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz LFO
    # Apply as amplitude modulation (simplified filter sweep)
    cutoff_base = 500
    cutoff_range = 1500
    sos_lo = butter(2, cutoff_base, btype='low', fs=sr, output='sos')
    signal = sosfiltfilt(sos_lo, signal)
    signal *= lfo  # LFO modulates brightness

    # Soft attack envelope
    attack = int(0.5 * sr)
    env = np.ones_like(t)
    env[:attack] = np.linspace(0, 1, attack)

    signal *= env
    signal /= np.max(np.abs(signal)) * 0.6
    return signal
```

---

## 4.4 — Frequency Slot Rules for Melodic Elements

| Element | Frequency Range | Volume Relative to Kick | Section |
|---------|----------------|------------------------|---------|
| Chord stab | 200–4000 Hz | -12 to -15 dB | Drops only |
| Acid line | 100–2000 Hz | -15 to -18 dB | Drops (60% chance) |
| Pad | 200–3000 Hz | -18 to -24 dB | Breakdowns, builds |
| Vocal chop | 500–6000 Hz | -12 to -15 dB | Drops (optional) |

### Objective Assessment

| Metric | Target | Why |
|--------|--------|-----|
| Element count simultaneous | ≤ 3 melodic + drums | Tech house is minimal |
| Melodic RMS vs kick RMS | -12 to -24 dB below | Drums dominate |
| Spectral overlap | < 30% between any two melodic elements | Clarity |
