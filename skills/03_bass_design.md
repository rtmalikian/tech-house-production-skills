# Skill 03: Bass Design & Programming

## What the Producer Does
Design a bass sound (oscillator → filter → saturation → envelope) and write
bass patterns that interlock with the kick. Apply sidechain compression so
bass ducks on every kick hit. Tune bass to the track's root note.

---

## 3.1 — Bass Sound Design

### Signal Chain
```
Oscillator (saw/square) → Filter (LP 24dB) → Saturation → Amp Envelope → Output
                         ↕
                    Filter Envelope (modulates cutoff)
```

### Programmatic (scipy + numpy synthesis)

```python
import numpy as np
from scipy.signal import butter, sosfiltfilt

def synthesize_bass(fundamental_hz, duration_s, sr=44100, waveform='saw'):
    """Generate a tech house bass note from scratch."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)

    # 1. Oscillator
    if waveform == 'saw':
        # Band-limited sawtooth (anti-aliased)
        signal = np.zeros_like(t)
        for k in range(1, 20):  # 20 harmonics
            signal += ((-1)**(k+1)) * np.sin(2 * np.pi * k * fundamental_hz * t) / k
        signal *= 2 / np.pi
    elif waveform == 'square':
        signal = np.sign(np.sin(2 * np.pi * fundamental_hz * t))
    else:  # sine (sub)
        signal = np.sin(2 * np.pi * fundamental_hz * t)

    # 2. Filter envelope (simulates cutoff modulation)
    # Attack: instant, Decay: 150ms, Sustain: 40%
    envelope = np.ones_like(t)
    attack_samples = int(0.002 * sr)   # 2ms
    decay_samples = int(0.15 * sr)     # 150ms
    sustain_level = 0.4

    for i in range(len(envelope)):
        if i < attack_samples:
            envelope[i] = i / attack_samples
        elif i < attack_samples + decay_samples:
            progress = (i - attack_samples) / decay_samples
            envelope[i] = 1.0 - progress * (1.0 - sustain_level)
        else:
            envelope[i] = sustain_level

    # Apply filter (cutoff modulated by envelope)
    # Average cutoff ~ 400 Hz, modulated up to 2000 Hz on attack
    cutoff_hz = 400 + envelope * 1600  # Hz over time
    # Simplified: apply a static lowpass at ~800 Hz (average)
    sos = butter(4, 800, btype='low', fs=sr, output='sos')
    signal = sosfiltfilt(sos, signal)

    # 3. Amplifier envelope (ADSR)
    amp_env = np.ones_like(t)
    amp_attack = int(0.002 * sr)    # 2ms
    amp_decay = int(0.3 * sr)       # 300ms
    amp_sustain = 0.7
    amp_release = int(0.2 * sr)     # 200ms

    for i in range(len(amp_env)):
        if i < amp_attack:
            amp_env[i] = i / amp_attack
        elif i < amp_attack + amp_decay:
            progress = (i - amp_attack) / amp_decay
            amp_env[i] = 1.0 - progress * (1.0 - amp_sustain)
        elif i > len(amp_env) - amp_release:
            progress = (len(amp_env) - i) / amp_release
            amp_env[i] = amp_sustain * progress
        else:
            amp_env[i] = amp_sustain

    signal *= amp_env

    # 4. Soft saturation
    drive = 1.5
    signal = np.tanh(signal * drive) / np.tanh(drive)

    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.9
    return signal
```

### Objective Assessment — Bass Sound

| Metric | How to Measure | Target | Why |
|--------|---------------|--------|-----|
| Fundamental clarity | Peak FFT bin at `fundamental_hz` ± 5 Hz | > -6 dB relative | Must be in tune |
| Harmonic richness | Count of harmonics > -40 dB | 5-15 | Saw = rich, sine = pure |
| Filter rolloff | Energy above cutoff / energy below | < 0.1 | Filter is working |
| Crest factor | Peak / RMS | 8-14 dB | Punchy but not spiky |

---

## 3.2 — Bass Pattern Programming

### The Rule: Kick owns the downbeat, bass owns the offbeat

Kick hits on 1, 2, 3, 4. Bass should NOT hit simultaneously on beat 1.
Instead, bass enters on the "and" of beat 1, or beat 3.

### Common Tech House Bass Patterns

```
Pattern "Offbeat":     . X . . . . X . . X . . . . X .
Pattern "Driving":     X . X . . X . . X . X . . X . .
Pattern "Minimal":     X . . . . . . . . . . . X . . .
Pattern "Rolling":     X . X . X . X . X . X . X . X .
Pattern "Chop":        X . X . X . X . . X . X . . X .
```

### Programmatic

```python
def write_bass_pattern(root_note, tpb=480, bars=1, pattern='offbeat'):
    events = []
    sixteenth = tpb // 4

    patterns = {
        'offbeat':  [0,1,0,0, 0,0,1,0, 0,1,0,0, 0,0,1,0],
        'driving':  [1,0,1,0, 0,1,0,0, 1,0,1,0, 0,1,0,0],
        'minimal':  [1,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,0,0],
        'rolling':  [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
        'chop':     [1,0,1,0, 1,0,1,0, 0,1,0,1, 0,0,1,0],
    }

    grid = patterns.get(pattern, patterns['offbeat'])

    for bar in range(bars):
        bs = bar * tpb * 4
        for step, active in enumerate(grid):
            if active:
                vel = random.randint(90, 110) if step % 4 == 0 else random.randint(70, 90)
                note_dur = sixteenth  # Short, staccato
                events.append({'time': bs + step * sixteenth, 'note': root_note, 'velocity': vel,
                               'duration': note_dur})

    return events
```

---

## 3.3 — Sidechain Compression

### Real-World Action
Route kick to sidechain input of bass compressor. Set ratio 4:1–8:1,
attack 0.1ms, release 150-200ms. Bass ducks 6-10 dB on each kick.

### Programmatic (signal-level sidechain)

```python
def apply_sidechain(bass_signal, kick_signal, sr=44100,
                    ratio=6.0, threshold_db=-20, attack_ms=0.1,
                    release_ms=150, gain_reduction_db=8):
    """Apply sidechain compression: bass ducks when kick hits."""
    # Detect kick transients
    kick_env = np.abs(kick_signal)
    kick_threshold = np.max(kick_env) * 0.1
    kick_active = kick_env > kick_threshold

    # Smooth the kick envelope (attack/release)
    attack_coeff = np.exp(-1.0 / (sr * attack_ms / 1000))
    release_coeff = np.exp(-1.0 / (sr * release_ms / 1000))

    sc_envelope = np.zeros_like(kick_env)
    for i in range(len(kick_active)):
        target = 1.0 if kick_active[i] else 0.0
        coeff = attack_coeff if target > sc_envelope[i-1] else release_coeff
        sc_envelope[i] = target + coeff * (sc_envelope[i-1] - target)

    # Apply gain reduction to bass
    gain = 1.0 - sc_envelope * (1.0 - 10**(-gain_reduction_db / 20))
    return bass_signal * gain
```

### Objective Assessment — Sidechain

| Metric | How to Measure | Target | Why |
|--------|---------------|--------|-----|
| Ducking depth | Max GR on each kick | 6-12 dB | Enough to make room |
| Recovery time | Time from max GR to 0 GR | 100-200 ms | Bass returns before next kick |
| Pump feeling | Amplitude modulation of bass | Audible "breathing" | The signature tech house groove |

---

## 3.4 — Kick-Bass Frequency Relationship

### The Cardinal Rule
Kick owns 40-80 Hz. Bass owns 80-200 Hz. They must NOT compete in the
40-80 Hz range.

### Programmatic Check

```python
def check_kick_bass_conflict(kick_path, bass_rendered_path, sr=44100):
    """Verify kick and bass don't mask each other in the low end."""
    import librosa
    kick, _ = librosa.load(kick_path, sr=sr)
    bass, _ = librosa.load(bass_rendered_path, sr=sr)

    S_kick = np.abs(librosa.stft(kick))
    S_bass = np.abs(librosa.stft(bass))
    freqs = librosa.fft_frequencies(sr=sr)

    # Energy in critical band (40-80 Hz)
    crit = np.where((freqs >= 40) & (freqs <= 80))[0]
    kick_low = np.mean(S_kick[crit, :])
    bass_low = np.mean(S_bass[crit, :])

    # Energy in bass band (80-200 Hz)
    bass_band = np.where((freqs >= 80) & (freqs <= 200))[0]
    kick_mid = np.mean(S_kick[bass_band, :])
    bass_mid = np.mean(S_bass[bass_band, :])

    conflict = bass_low > kick_low * 0.5  # Bass too loud in kick's territory

    return {
        'kick_40_80_db': round(20*np.log10(kick_low+1e-10), 1),
        'bass_40_80_db': round(20*np.log10(bass_low+1e-10), 1),
        'kick_80_200_db': round(20*np.log10(kick_mid+1e-10), 1),
        'bass_80_200_db': round(20*np.log10(bass_mid+1e-10), 1),
        'conflict': conflict,
        'recommendation': 'Cut bass below 80 Hz' if conflict else 'OK',
    }
```
