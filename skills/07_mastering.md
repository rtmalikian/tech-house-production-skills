# Skill 07: Mastering

## What the Producer Does
Apply final processing to the stereo mix bus: tonal balance correction,
multiband compression, stereo width management, soft clipping, and
limiting. The goal: competitive loudness with preserved dynamics and
punch, ready for club play and streaming platforms.

---

## 7.1 — Mastering Signal Chain (IN ORDER)

```
Input Mix
    │
    ▼
1. Cleanup EQ (HPF 25 Hz, LPF 18 kHz, surgical cuts)
    │
    ▼
2. Mid/Side EQ (mono lows, widen highs)
    │
    ▼
3. Multiband Compression (4 bands)
    │
    ▼
4. Soft Clipping (catch peaks before limiter)
    │
    ▼
5. Limiter (ceiling -0.3 dBTP, 4-6 dB GR)
    │
    ▼
Output Master
```

---

## 7.2 — Step 1: Cleanup EQ

### Programmatic

```python
from scipy.signal import butter, sosfiltfilt

def mastering_cleanup_eq(audio, sr=44100):
    """Step 1: Remove sub-rumble and harsh air."""
    # HPF at 25 Hz (24 dB/oct = order 4)
    sos_hp = butter(4, 25, btype='high', fs=sr, output='sos')
    audio = sosfiltfilt(sos_hp, audio)

    # LPF at 18 kHz (12 dB/oct = order 2)
    sos_lp = butter(2, 18000, btype='low', fs=sr, output='sos')
    audio = sosfiltfilt(sos_lp, audio)

    return audio
```

### When to Apply Additional EQ
| Problem | Frequency | Cut/Boost | Q |
|---------|-----------|-----------|---|
| Muddy | 200-300 Hz | -1 to -2 dB | 0.7-1.0 |
| Thin low end | 50-60 Hz | +1 dB | 0.7 |
| Dull | 10-12 kHz | +1 dB | 0.7 |
| Harsh | 3-5 kHz | -1 dB | 1.5 |

### Objective Assessment — Cleanup EQ

| Metric | How to Measure | Target |
|--------|---------------|--------|
| Sub energy below 25 Hz | FFT sum < 25 Hz | < -60 dB relative |
| Air above 18 kHz | FFT sum > 18 kHz | < -50 dB relative |
| No resonant peaks | Peak detection in FFT | No peak > +6 dB above neighbors |

---

## 7.3 — Step 2: Mid/Side EQ

### Programmatic

```python
def mid_side_eq(audio_stereo, sr=44100):
    """Mid/Side processing for mastering."""
    left, right = audio_stereo[:, 0], audio_stereo[:, 1]
    mid = (left + right) / 2
    side = (left - right) / 2

    # Mono the lows in side channel
    sos_hp = butter(4, 150, btype='high', fs=sr, output='sos')
    side = sosfiltfilt(sos_hp, side)  # Side has no content below 150 Hz

    # Air boost on side (simplified: gentle shelf boost above 8 kHz)
    # In practice, use a high shelf filter
    # Here we boost side high-frequency energy by 1 dB
    fft_side = np.fft.rfft(side)
    freqs = np.fft.rfftfreq(len(side), 1/sr)
    boost_mask = freqs > 8000
    fft_side[boost_mask] *= 10**(1.0/20)  # +1 dB
    side = np.fft.irfft(fft_side, len(side))

    # Reconstruct
    left_out = mid + side
    right_out = mid - side
    return np.column_stack([left_out, right_out])
```

### Objective Assessment — Mid/Side

| Metric | Target | Why |
|--------|--------|-----|
| Side energy below 150 Hz | < -60 dB | Mono bass for club |
| Side energy above 8 kHz | +0.5 to +1.5 dB above mid | Width in highs |
| Mid correlation below 200 Hz | > 0.98 | Tight center |

---

## 7.4 — Step 3: Multiband Compression

### 4-Band Configuration

| Band | Range | Ratio | Attack | Release | GR Target |
|------|-------|-------|--------|---------|-----------|
| 1 (Sub) | 20-120 Hz | 3:1 | 20ms | 100ms | 2-3 dB |
| 2 (Low-Mid) | 120-1000 Hz | 2:1 | 10ms | 80ms | 1-2 dB |
| 3 (Presence) | 1k-8k Hz | 2:1 | 5ms | 60ms | 1-2 dB |
| 4 (Air) | 8k-20k Hz | 2:1 | 1ms | 40ms | 1-2 dB |

### Programmatic (scipy crossover + pedalboard compression)

```python
from scipy.signal import butter, sosfiltfilt
from pedalboard import Compressor, Pedalboard

def multiband_compress(audio, sr=44100):
    """4-band multiband compression for mastering."""
    # Crossover frequencies
    xovers = [120, 1000, 8000]

    # Split into bands using Linkwitz-Riley crossovers (simplified with Butterworth)
    bands = []
    remaining = audio.copy()

    for i, xover in enumerate(xovers):
        sos_lp = butter(4, xover, btype='low', fs=sr, output='sos')
        low = sosfiltfilt(sos_lp, remaining)
        sos_hp = butter(4, xover, btype='high', fs=sr, output='sos')
        high = sosfiltfilt(sos_hp, remaining)
        bands.append(low)
        remaining = high

    bands.append(remaining)  # Highest band

    # Compress each band
    settings = [
        {'threshold_db': -25, 'ratio': 3, 'attack_ms': 20, 'release_ms': 100},  # Sub
        {'threshold_db': -22, 'ratio': 2, 'attack_ms': 10, 'release_ms': 80},   # Low-mid
        {'threshold_db': -20, 'ratio': 2, 'attack_ms': 5,  'release_ms': 60},   # Presence
        {'threshold_db': -18, 'ratio': 2, 'attack_ms': 1,  'release_ms': 40},   # Air
    ]

    compressed = []
    for band, setting in zip(bands, settings):
        board = Pedalboard([Compressor(**setting)])
        compressed.append(board(band, sr))

    # Sum bands
    output = sum(compressed)
    return output
```

### Objective Assessment — Multiband

| Metric | How to Measure | Target |
|--------|---------------|--------|
| Band 1 GR | Measure RMS before/after | 2-3 dB |
| Band 4 GR | Same | 1-2 dB |
| Spectral balance change | Compare band ratios before/after | < 2 dB shift |
| No pumping | Amplitude modulation at beat rate | < 1 dB in any band |

---

## 7.5 — Step 4: Soft Clipping

```python
def soft_clip(audio, threshold_db=-1.0):
    """Soft clip to catch peaks before the limiter."""
    threshold = 10**(threshold_db / 20)
    # Tanh soft clipping
    return np.tanh(audio / threshold) * threshold
```

---

## 7.6 — Step 5: Limiting

### Programmatic

```python
from pedalboard import Limiter, Pedalboard

def master_limit(audio, sr=44100, ceiling_db=-0.3):
    """Final limiter for tech house mastering."""
    board = Pedalboard([
        Limiter(threshold_db=ceiling_db, release_ms=80)
    ])
    return board(audio, sr)
```

### With pyloudnorm for loudness targeting:

```python
import pyloudnorm as pyln

def master_with_loudness_target(audio, sr=44100, target_lufs=-8):
    """Master and verify loudness target."""
    meter = pyln.Meter(sr)

    # Apply limiter
    limited = master_limit(audio, sr)

    # Measure
    current_lufs = meter.integrated_loudness(limited)

    # If too quiet, add gain and re-limit
    if current_lufs < target_lufs - 1:
        gain_db = target_lufs - current_lufs
        gain = 10**(gain_db / 20)
        limited = master_limit(limited * gain, sr)
        current_lufs = meter.integrated_loudness(limited)

    return limited, {
        'integrated_lufs': round(current_lufs, 1),
        'peak_dbfs': round(20*np.log10(np.max(np.abs(limited)) + 1e-10), 1),
    }
```

---

## 7.7 — Final Mastering Assessment

| Metric | How to Measure | Club Target | Streaming Target |
|--------|---------------|-------------|-----------------|
| Integrated LUFS | `pyloudnorm` | -8 to -6 | -14 |
| True Peak | `np.max(abs())` in dB | < -0.3 dBTP | < -1.0 dBTP |
| Crest Factor | Peak/RMS dB | 4-8 dB | 6-10 dB |
| Dynamic Range | 95th - 5th percentile RMS | 4-8 dB | 8-12 dB |
| Stereo Correlation | `np.corrcoef(L,R)` | 0.3-0.7 | 0.4-0.8 |
| Spectral Balance | Energy ratio per band | See below | See below |

### Spectral Balance Targets (tech house)
| Band | Energy % of Total |
|------|------------------|
| Sub (20-60 Hz) | 15-20% |
| Bass (60-250 Hz) | 25-35% |
| Low-Mid (250-2k Hz) | 15-25% |
| Presence (2k-6k Hz) | 10-20% |
| Air (6k-20k Hz) | 5-15% |
