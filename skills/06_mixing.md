# Skill 06: Mixing

## What the Producer Does
Balance all elements into a cohesive stereo mix. Process each channel
(EQ, compression), set up sends (reverb, delay), bus processing (drum bus,
master bus), and manage stereo width. The goal: every element audible in
its own frequency/rhythmic slot, with the kick and bass dominating.

---

## 6.1 — Gain Staging (FIRST STEP, ALWAYS)

### Real-World Action
Set all faders to unity. Adjust clip gains so each element peaks at
-12 to -18 dBFS. Master bus peaks at -6 dBFS with everything playing.

### Programmatic

```python
def gain_stage_stems(stems: dict, kick_peak_target_db=-12):
    """
    Gain stage: set kick to target, balance everything relative.
    stems: {'kick': np.array, 'bass': np.array, 'clap': np.array, ...}
    """
    staged = {}

    # 1. Set kick to target
    kick = stems['kick']
    kick_peak = np.max(np.abs(kick))
    kick_gain = 10**((kick_peak_target_db - 20*np.log10(kick_peak + 1e-10)) / 20)
    staged['kick'] = kick * kick_gain

    # 2. Set other elements relative to kick
    relative_levels = {
        'bass': -2,       # -14 dBFS (2 dB below kick)
        'sub_bass': -3,   # -15 dBFS
        'clap': -3,       # -15 dBFS
        'hat_closed': -6, # -18 dBFS
        'hat_open': -5,   # -17 dBFS
        'percussion': -8, # -20 dBFS
        'stab': -6,       # -18 dBFS
        'acid': -6,       # -18 dBFS
        'pad': -10,       # -22 dBFS
    }

    for name, signal in stems.items():
        if name == 'kick':
            continue
        target_db = kick_peak_target_db + relative_levels.get(name, -6)
        peak = np.max(np.abs(signal))
        gain = 10**((target_db - 20*np.log10(peak + 1e-10)) / 20)
        staged[name] = signal * gain

    return staged
```

### Objective Assessment — Gain Staging

| Metric | How to Measure | Target |
|--------|---------------|--------|
| Kick peak | `20*log10(max(abs(kick)))` | -12 dBFS ± 1 |
| Bass peak | Same | -14 dBFS ± 1 |
| Master peak | Sum of all stems peak | -6 to -3 dBFS |
| Headroom | `0 - master_peak_db` | ≥ 3 dB |

---

## 6.2 — Channel EQ

### Processing Order: Subtractive → Additive

### Kick EQ Chain
```
1. HPF 30 Hz (12 dB/oct)  — remove sub-rumble
2. Cut -3 dB at 200-300 Hz, Q=1.5 — remove mud
3. Cut -2 dB at 800 Hz, Q=1 — remove boxiness
4. Boost +2 dB at 3-5 kHz, Q=2 — add beater click
```

### Bass EQ Chain
```
1. HPF 25 Hz (12 dB/oct)
2. Boost +2 dB at 50-80 Hz, Q=1 — weight
3. Cut -3 dB at 200-400 Hz, Q=1.5 — clean mud
4. Boost +2 dB at 1-2 kHz, Q=2 — growl/presence
5. LPF 8-10 kHz — remove noise
```

### Clap EQ Chain
```
1. HPF 200 Hz (18 dB/oct) — remove ALL low end
2. Cut -2 dB at 400-600 Hz, Q=1 — remove honk
3. Boost +2 dB at 2-4 kHz, Q=1.5 — snap
4. Boost +1 dB at 8-10 kHz, Q=1 — air
```

### Hi-Hat EQ Chain
```
1. HPF 300-500 Hz (18 dB/oct)
2. Cut -2 dB at 1-2 kHz if harsh
3. Boost +1 dB at 8-12 kHz — brightness
```

### Programmatic (using pedalboard)

```python
from pedalboard import Pedalboard, HighpassFilter, LowpassFilter, Compressor, Gain

def eq_kick(kick_audio, sr=44100):
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=30),
        # Parametric EQ cuts would use a plugin or manual scipy
    ])
    return board(kick_audio, sr)

def eq_bass(bass_audio, sr=44100):
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=25),
        LowpassFilter(cutoff_frequency_hz=8000),
    ])
    return board(bass_audio, sr)

def eq_clap(clap_audio, sr=44100):
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=200),
    ])
    return board(clap_audio, sr)
```

### For surgical parametric EQ, use scipy:

```python
from scipy.signal import iirpeak, lfilter

def parametric_eq(signal, freq_hz, gain_db, q, sr=44100):
    """Apply a parametric EQ band (bell curve)."""
    w0 = freq_hz / sr * 2 * np.pi
    A = 10**(gain_db / 40)
    alpha = np.sin(w0) / (2 * q)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    b = np.array([b0/a0, b1/a0, b2/a0])
    a = np.array([1, a1/a0, a2/a0])
    return lfilter(b, a, signal)
```

### Objective Assessment — EQ

| Metric | How to Measure | Target |
|--------|---------------|--------|
| Mud reduction | Energy in 200-400 Hz before/after | -2 to -6 dB reduction |
| Kick click | Energy in 3-5 kHz after | +1 to +3 dB above pre-EQ |
| Bass clarity | Spectral centroid shift | Should increase (brighter) |
| No new resonances | Peak FFT analysis | No new peaks > +6 dB |

---

## 6.3 — Compression

### Per-Element Settings

| Element | Ratio | Attack | Release | GR Target |
|---------|-------|--------|---------|-----------|
| Kick | 3:1 | 10-30ms | 50-100ms | 4-6 dB |
| Bass | 4:1 | 5-10ms | 50-100ms | 3-6 dB |
| Clap | 2-3:1 | 5-15ms | 100-200ms | 3-4 dB |
| Hats | 2:1 | 5ms | 50ms | 1-2 dB |
| Drum bus | 2:1 | 10ms | 100ms | 2-4 dB |

### Programmatic

```python
from pedalboard import Compressor

def compress_element(audio, sr, element_type='kick'):
    settings = {
        'kick':  {'threshold_db': -20, 'ratio': 3, 'attack_ms': 20, 'release_ms': 80},
        'bass':  {'threshold_db': -18, 'ratio': 4, 'attack_ms': 8,  'release_ms': 80},
        'clap':  {'threshold_db': -15, 'ratio': 2.5, 'attack_ms': 10, 'release_ms': 150},
        'hat':   {'threshold_db': -12, 'ratio': 2, 'attack_ms': 5,  'release_ms': 50},
        'drum_bus': {'threshold_db': -18, 'ratio': 2, 'attack_ms': 10, 'release_ms': 100},
    }
    s = settings.get(element_type, settings['kick'])
    board = Pedalboard([Compressor(**s)])
    return board(audio, sr)
```

### Objective Assessment — Compression

| Metric | How to Measure | Target |
|--------|---------------|--------|
| Crest factor reduction | CF_before - CF_after | 2-6 dB |
| Transient preservation | Peak level before/after | < 1 dB reduction |
| RMS increase | RMS_after - RMS_before | +1 to +3 dB |
| Pumping audible? | Amplitude modulation > 2 dB at beat rate | Should NOT pump on individual channels |

---

## 6.4 — Bus Processing

### Drum Bus
```
SSL-style bus compressor: 2:1, attack 10ms, release 100ms, 2-4 dB GR
Tape saturation: 5-10%
Air EQ shelf: +1 dB at 10 kHz
```

### Master Bus (during mixing)
```
SSL bus compressor: 2:1, attack 10-30ms, release auto, 1-3 dB GR
HPF 25 Hz
LPF 18 kHz
```

---

## 6.5 — Stereo Width Management

### Frequency-Dependent Width Rules

| Frequency | Width | Elements |
|-----------|-------|----------|
| Below 150 Hz | MONO | Kick, sub bass |
| 150-500 Hz | Narrow (±10%) | Bass harmonics |
| 500-5000 Hz | Moderate (±20%) | Stabs, claps |
| Above 5 kHz | Wide (±30%) | Hats, reverb, air |

### Programmatic

```python
def mono_below_frequency(stereo_signal, sr, cutoff_hz=150):
    """Collapse everything below cutoff to mono."""
    left, right = stereo_signal[:, 0], stereo_signal[:, 1]
    mid = (left + right) / 2
    side = (left - right) / 2

    # Highpass the side channel
    sos = butter(4, cutoff_hz, btype='high', fs=sr, output='sos')
    side = sosfiltfilt(sos, side)

    # Reconstruct
    new_left = mid + side
    new_right = mid - side
    return np.column_stack([new_left, new_right])

def measure_stereo_width(stereo_signal, sr):
    """Measure stereo correlation and width."""
    left, right = stereo_signal[:, 0], stereo_signal[:, 1]
    mid = (left + right) / 2
    side = (left - right) / 2

    correlation = np.corrcoef(left, right)[0, 1]
    side_to_mid = np.sqrt(np.mean(side**2)) / (np.sqrt(np.mean(mid**2)) + 1e-10)

    return {
        'correlation': round(correlation, 3),  # +1=mono, 0=wide, -1=phase
        'side_to_mid_ratio': round(side_to_mid, 3),  # 0=mono, 0.5=moderate, 1=wide
    }
```

### Objective Assessment — Stereo

| Metric | Target | Why |
|--------|--------|-----|
| Sub correlation (< 150 Hz) | > 0.95 | Mono bass for club |
| Mid correlation (500-5k Hz) | 0.5–0.8 | Moderate width |
| High correlation (> 5k Hz) | 0.2–0.6 | Wide hats/air |
| Side-to-mid ratio | 0.1–0.4 overall | Not too wide, not too narrow |

---

## 6.6 — Reverb & Delay Sends

### Reverb Rules for Tech House
- **Clap**: Plate, decay 1.5-2.5s, pre-delay 20-40ms, HP return at 200 Hz, LP return at 6 kHz
- **Percussion**: Room, decay 0.5-1s
- **Hats**: Very short room, decay 0.3-0.5s
- **Bass**: NO REVERB
- **Kick**: NO REVERB

### Delay
- **1/8 note** at 126 BPM = 476ms
- **Feedback**: 20-40%
- **HP on return**: 500 Hz
- **LP on return**: 5 kHz

### Objective Assessment — Reverb/Delay

| Metric | Target | Why |
|--------|--------|-----|
| Reverb tail in low end | < -40 dB below dry signal | No mud |
| Delay feedback level | Decays to -30 dB within 4 beats | Not cluttered |
| Reverb/dry ratio | 10-20% wet | Subtle, supportive |
