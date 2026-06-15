# Skill 01: Sound Selection

## What the Producer Does
Audition and select individual samples (kick, clap, hat, percussion) that
work together as a cohesive kit. The producer listens for tonal balance,
transient shape, decay length, and how each element sits in its frequency
slot relative to the others.

---

## 1.1 — Kick Drum Selection

### Real-World Action
Load 20-50 kick samples into a DAW. Audition each on beats 1-2-3-4 at
track tempo (126 BPM). Narrow to 3-5 candidates. Final selection based on:
- Fundamental pitch (tuned to track key)
- Transient click (2-5 kHz presence)
- Sub weight (40-60 Hz body)
- Tail length (150-300ms — must clear before next hit)

### Programmatic Equivalent

```python
import librosa
import numpy as np
import soundfile as sf

def analyze_kick(path, sr=44100):
    """Score a kick sample for tech house suitability."""
    y, sr = librosa.load(path, sr=sr)

    # 1. Fundamental frequency
    f0 = librosa.yin(y, fmin=30, fmax=100, sr=sr)
    f0_clean = f0[f0 > 0]
    fundamental_hz = np.median(f0_clean) if len(f0_clean) > 0 else 0

    # 2. Transient strength (attack click in 2-5 kHz)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    click_band = np.where((freqs >= 2000) & (freqs <= 5000))[0]
    click_energy = np.mean(S[click_band, :5])  # First 5 frames = attack
    total_energy = np.mean(S[:, :5]) + 1e-10
    click_ratio = click_energy / total_energy

    # 3. Sub weight (40-60 Hz)
    sub_band = np.where((freqs >= 40) & (freqs <= 60))[0]
    sub_energy = np.mean(S[sub_band, :])
    sub_ratio = sub_energy / (np.mean(S) + 1e-10)

    # 4. Tail length (time for amplitude to drop -20 dB from peak)
    envelope = np.abs(y)
    peak = np.max(envelope)
    threshold = peak * 0.1  # -20 dB
    above_threshold = np.where(envelope > threshold)[0]
    tail_samples = above_threshold[-1] - above_threshold[0] if len(above_threshold) > 0 else 0
    tail_ms = (tail_samples / sr) * 1000

    # 5. Crest factor (punch indicator)
    rms = np.sqrt(np.mean(y**2))
    crest_db = 20 * np.log10(peak / (rms + 1e-10))

    return {
        'path': path,
        'fundamental_hz': round(fundamental_hz, 1),
        'click_ratio': round(click_ratio, 3),
        'sub_ratio': round(sub_ratio, 3),
        'tail_ms': round(tail_ms, 1),
        'crest_db': round(crest_db, 1),
    }
```

### Objective Assessment (NOT arbitrary)

| Metric | How to Measure | Good Range | Why |
|--------|---------------|------------|-----|
| **Fundamental** | `librosa.yin()` median | Track key ± 2 semitones | Kick must harmonize with bass |
| **Click ratio** | Energy in 2-5 kHz / total in attack window | 0.15–0.40 | Enough cut-through on club PA |
| **Sub ratio** | Energy in 40-60 Hz / total energy | 0.3–0.8 | Felt on the dancefloor |
| **Tail length** | Time above -20 dB of peak | 120–350 ms | Must clear at 126 BPM (476 ms/beat) |
| **Crest factor** | Peak / RMS in dB | 10–18 dB | Higher = more punchy, lower = compressed |

**Decision rule:** The kick whose fundamental is closest to the track root note,
with crest factor > 12 dB and tail < 400 ms, wins. If multiple candidates pass,
pick the one with highest click_ratio.

### Iterative Refinement
1. Score all samples → rank by composite score
2. If no sample scores well on sub_ratio AND click_ratio, the kit needs a
   different sample source (not an EQ fix)
3. Re-test after any processing — EQ/saturation changes the metrics

---

## 1.2 — Clap/Snare Selection

### Real-World Action
Select claps with sharp transient (1-3 ms rise), energy spread 1-8 kHz,
minimal low-mid content (200-400 Hz). Test on beats 2 & 4 against kick.

### Programmatic Equivalent

```python
def analyze_clap(path, sr=44100):
    y, sr = librosa.load(path, sr=sr)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    # Transient rise time
    envelope = np.abs(y)
    peak_idx = np.argmax(envelope)
    rise_time_ms = (peak_idx / sr) * 1000

    # Low-mid contamination (200-400 Hz — BAD for claps)
    lomids = np.where((freqs >= 200) & (freqs <= 400))[0]
    highs = np.where((freqs >= 1000) & (freqs <= 8000))[0]
    contamination = np.mean(S[lomids, :]) / (np.mean(S[highs, :]) + 1e-10)

    # Snap (1-4 kHz energy in first 10ms)
    attack_frames = max(1, int(0.01 * sr / 512))  # ~10ms in STFT frames
    snap_band = np.where((freqs >= 1000) & (freqs <= 4000))[0]
    snap = np.mean(S[snap_band, :attack_frames])

    return {
        'rise_time_ms': round(rise_time_ms, 2),
        'low_mid_contamination': round(contamination, 3),  # lower is better
        'snap_score': round(snap, 1),
    }
```

### Objective Assessment

| Metric | Good Range | Why |
|--------|------------|-----|
| Rise time | 0.5–5 ms | Sharp enough to cut through |
| Low-mid contamination | < 0.3 | Won't fight the kick |
| Snap score | Relative — highest among candidates | The "crack" on club speakers |

---

## 1.3 — Hi-Hat Selection

### Real-World Action
Select 2-3 closed hats for velocity layering + 1 open hat. Closed hats
need crisp metallic character (5-12 kHz). Open hat needs 200-500 ms tail.

### Programmatic Equivalent

```python
def analyze_hat(path, sr=44100, is_open=False):
    y, sr = librosa.load(path, sr=sr)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    # Dominant frequency band
    mag_mean = np.mean(S, axis=1)
    dominant_idx = np.argmax(mag_mean[freqs > 2000])  # Ignore lows
    dominant_hz = freqs[2000:][dominant_idx] if len(freqs) > 2000 + dominant_idx else 0

    # Tail length
    envelope = np.abs(y)
    peak = np.max(envelope)
    threshold = peak * 0.1
    above = np.where(envelope > threshold)[0]
    tail_ms = ((above[-1] - above[0]) / sr * 1000) if len(above) > 0 else 0

    # Spectral centroid (brightness)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    return {
        'dominant_hz': round(dominant_hz, 0),
        'tail_ms': round(tail_ms, 1),
        'brightness_hz': round(centroid, 0),
        'is_open': is_open,
    }
```

### Objective Assessment

| Metric | Closed Hat | Open Hat |
|--------|-----------|----------|
| Dominant freq | 5–12 kHz | 4–10 kHz |
| Tail | 30–100 ms | 200–500 ms |
| Brightness centroid | > 6000 Hz | > 4000 Hz |

---

## 1.4 — Percussion Selection

### Frequency Slot Allocation
Each percussion element must occupy its own frequency slot to avoid masking:

| Element | Frequency Slot | Role |
|---------|---------------|------|
| Kick | 40–100 Hz + 2–5 kHz | Low end + click |
| Clap | 1–8 kHz | Snap/backbeat |
| Closed hat | 5–12 kHz | Rhythmic drive |
| Open hat | 4–10 kHz | Accent |
| Shaker | 8–16 kHz | Texture/fill |
| Rimshot | 2–6 kHz | Syncopated accent |
| Conga/tom | 80–200 Hz | Tuned percussion |
| Cowbell | 400–800 Hz | Rhythmic marker |
| Ride | 6–15 kHz | Sparkle |

### Programmatic Check
```python
def check_frequency_overlap(sample_a_path, sample_b_path, sr=44100):
    """Check if two samples will mask each other."""
    ya, _ = librosa.load(sample_a_path, sr=sr)
    yb, _ = librosa.load(sample_b_path, sr=sr)

    Sa = np.mean(np.abs(librosa.stft(ya)), axis=1)
    Sb = np.mean(np.abs(librosa.stft(yb)), axis=1)
    freqs = librosa.fft_frequencies(sr=sr)

    # Find peak frequency of each
    peak_a = freqs[np.argmax(Sa)]
    peak_b = freqs[np.argmax(Sb)]

    # If peaks are within 500 Hz, they'll mask each other
    conflict = abs(peak_a - peak_b) < 500
    return {
        'peak_a_hz': round(peak_a, 0),
        'peak_b_hz': round(peak_b, 0),
        'conflict': conflict,
        'separation_hz': round(abs(peak_a - peak_b), 0),
    }
```

---

## Composite Kit Scoring

```python
def score_kit(kick_path, clap_path, hat_path, open_hat_path, key_hz):
    """Score an entire drum kit for tech house cohesion."""
    kick = analyze_kick(kick_path)
    clap = analyze_clap(clap_path)
    hat = analyze_hat(hat_path, is_open=False)
    ohat = analyze_hat(open_hat_path, is_open=True)

    scores = []

    # Kick: fundamental close to key?
    key_error = abs(kick['fundamental_hz'] - key_hz)
    scores.append(max(0, 1 - key_error / 20))  # 0 if >20 Hz off

    # Kick: punchy enough?
    scores.append(1 if kick['crest_db'] > 12 else kick['crest_db'] / 12)

    # Clap: clean enough?
    scores.append(max(0, 1 - clap['low_mid_contamination']))

    # Hat: bright enough?
    scores.append(1 if hat['brightness_hz'] > 6000 else hat['brightness_hz'] / 6000)

    # Open hat: longer tail?
    scores.append(1 if ohat['tail_ms'] > 200 else ohat['tail_ms'] / 200)

    # No frequency conflicts?
    kick_hat = check_frequency_overlap(kick_path, hat_path)
    scores.append(0 if kick_hat['conflict'] else 1)

    return {
        'total_score': round(np.mean(scores), 3),
        'breakdown': scores,
        'kick': kick, 'clap': clap, 'hat': hat, 'open_hat': ohat,
    }
```

**Selection rule:** Kit with highest total_score that also has no frequency
conflicts between any pair of elements.
