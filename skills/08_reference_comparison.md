# Skill 08: Reference Track Comparison

## What the Producer Does
Import 2-3 professional reference tracks. Level-match. A/B compare
spectral balance, low-end weight, mid-range clarity, high-frequency
brightness, stereo width, and loudness. Adjust own mix to match.

---

## 8.1 — Setup

### Programmatic

```python
import soundfile as sf
import pyloudnorm as pyln
import numpy as np
import librosa

def load_and_level_match(reference_path, mix_path, target_lufs=-14):
    """Load both tracks and level-match to same LUFS."""
    ref_data, sr = sf.read(reference_path)
    mix_data, sr2 = sf.read(mix_path)

    meter = pyln.Meter(sr)

    ref_lufs = meter.integrated_loudness(ref_data)
    mix_lufs = meter.integrated_loudness(mix_data)

    # Normalize both to target
    ref_normalized = pyln.normalize.loudness(ref_data, ref_lufs, target_lufs)
    mix_normalized = pyln.normalize.loudness(mix_data, mix_lufs, target_lufs)

    return ref_normalized, mix_normalized, sr, {
        'ref_original_lufs': round(ref_lufs, 1),
        'mix_original_lufs': round(mix_lufs, 1),
    }
```

---

## 8.2 — Spectral Comparison

### The #1 Tool: Overlaid Spectrum Analyzer

```python
def compare_spectrums(mix_path, ref_path, sr=44100):
    """Overlay frequency spectrums of mix vs reference."""
    mix, _ = librosa.load(mix_path, sr=sr, mono=True)
    ref, _ = librosa.load(ref_path, sr=sr, mono=True)

    # FFT
    mix_fft = np.abs(np.fft.rfft(mix))
    ref_fft = np.abs(np.fft.rfft(ref))
    freqs = np.fft.rfftfreq(len(mix), 1/sr)

    # Smooth with 1/3 octave bands
    def third_octave_smooth(magnitude, freqs):
        bands = []
        center_freqs = []
        f = 20
        while f < 20000:
            f_low = f / (2**(1/6))
            f_high = f * (2**(1/6))
            mask = (freqs >= f_low) & (freqs < f_high)
            if np.any(mask):
                bands.append(np.mean(magnitude[mask]))
                center_freqs.append(f)
            f *= 2**(1/3)
        return np.array(bands), np.array(center_freqs)

    mix_bands, centers = third_octave_smooth(mix_fft, freqs)
    ref_bands, _ = third_octave_smooth(ref_fft, freqs)

    # Convert to dB
    mix_db = 20 * np.log10(mix_bands + 1e-10)
    ref_db = 20 * np.log10(ref_bands + 1e-10)

    # Difference per band
    diff_db = mix_db - ref_db

    return {
        'center_frequencies': centers,
        'mix_db': mix_db,
        'ref_db': ref_db,
        'difference_db': diff_db,
    }
```

---

## 8.3 — What to Compare (in priority order)

### Priority 1: Low End (40-120 Hz)
```
Problem: Mix has less energy than reference → boost bass/sub or reduce kick HPF
Problem: Mix has more energy → cut bass below 60 Hz or tighten sidechain
Threshold: ±2 dB difference = OK, ±4 dB = needs fix
```

### Priority 2: Kick-Bass Clarity
```python
def check_kick_bass_clarity(mix_path, ref_path, sr=44100):
    """Compare kick+bass clarity in the 40-200 Hz range."""
    mix, _ = librosa.load(mix_path, sr=sr, mono=True)
    ref, _ = librosa.load(ref_path, sr=sr, mono=True)

    S_mix = np.abs(librosa.stft(mix))
    S_ref = np.abs(librosa.stft(ref))
    freqs = librosa.fft_frequencies(sr=sr)

    # Compare energy in critical bands
    kick_band = np.where((freqs >= 40) & (freqs <= 80))[0]
    bass_band = np.where((freqs >= 80) & (freqs <= 200))[0]

    kick_mix = np.mean(S_mix[kick_band, :])
    bass_mix = np.mean(S_mix[bass_band, :])
    kick_ref = np.mean(S_ref[kick_band, :])
    bass_ref = np.mean(S_ref[bass_band, :])

    return {
        'mix_kick_bass_ratio': round(20*np.log10(kick_mix/(bass_mix+1e-10)), 1),
        'ref_kick_bass_ratio': round(20*np.log10(kick_ref/(bass_ref+1e-10)), 1),
        'match': abs(20*np.log10(kick_mix/(bass_mix+1e-10)) -
                     20*np.log10(kick_ref/(bass_ref+1e-10))) < 2,
    }
```

### Priority 3: Mid-Range Clarity (500 Hz - 2 kHz)
- Tech house should be clean here
- Compare spectral centroid
- If mix sounds muddier: cut 200-400 Hz on multiple elements

### Priority 4: High-Frequency Brightness (8-16 kHz)
```
Check: Hats and highs should be within ±2 dB of reference
Too dull: Boost hats or add air EQ
Too harsh: Cut 10-12 kHz
```

### Priority 5: Stereo Width
```python
def compare_stereo_width(mix_path, ref_path, sr=44100):
    """Compare stereo width between mix and reference."""
    mix, _ = sf.read(mix_path)
    ref, _ = sf.read(ref_path)

    def get_width(signal):
        if signal.ndim == 1:
            return {'correlation': 1.0, 'side_to_mid': 0.0}
        L, R = signal[:, 0], signal[:, 1]
        mid = (L + R) / 2
        side = (L - R) / 2
        return {
            'correlation': round(np.corrcoef(L, R)[0, 1], 3),
            'side_to_mid': round(np.sqrt(np.mean(side**2)) / (np.sqrt(np.mean(mid**2)) + 1e-10), 3),
        }

    return {
        'mix': get_width(mix),
        'reference': get_width(ref),
    }
```

### Priority 6: Loudness Match
```python
def compare_loudness(mix_path, ref_path, sr=44100):
    """Compare integrated LUFS."""
    mix, _ = sf.read(mix_path)
    ref, _ = sf.read(ref_path)
    meter = pyln.Meter(sr)

    return {
        'mix_lufs': round(meter.integrated_loudness(mix), 1),
        'ref_lufs': round(meter.integrated_loudness(ref), 1),
        'difference': round(meter.integrated_loudness(mix) - meter.integrated_loudness(ref), 1),
    }
```

---

## 8.4 — Automated A/B Report

```python
def generate_ab_report(mix_path, ref_path, sr=44100):
    """Generate a comprehensive A/B comparison report."""
    report = {}

    # Loudness
    report['loudness'] = compare_loudness(mix_path, ref_path, sr)

    # Spectral
    report['spectral'] = compare_spectrums(mix_path, ref_path, sr)

    # Low end
    report['low_end'] = check_kick_bass_clarity(mix_path, ref_path, sr)

    # Stereo
    report['stereo'] = compare_stereo_width(mix_path, ref_path, sr)

    # Recommendations
    issues = []
    spectral_diff = report['spectral']['difference_db']
    centers = report['spectral']['center_frequencies']

    for freq, diff in zip(centers, spectral_diff):
        if freq < 80 and diff < -3:
            issues.append(f"LOW END: {diff:.1f} dB below reference at {freq:.0f} Hz — boost sub/bass")
        elif freq < 80 and diff > 3:
            issues.append(f"LOW END: {diff:.1f} dB above reference at {freq:.0f} Hz — tighten sidechain or cut bass")
        elif 200 <= freq <= 400 and diff > 3:
            issues.append(f"MUD: {diff:.1f} dB above reference at {freq:.0f} Hz — cut 200-400 Hz on elements")
        elif 2000 <= freq <= 5000 and diff < -3:
            issues.append(f"PRESENCE: {diff:.1f} dB below reference at {freq:.0f} Hz — boost 2-5 kHz")
        elif freq > 8000 and diff < -3:
            issues.append(f"AIR: {diff:.1f} dB below reference at {freq:.0f} Hz — add air shelf or brighten hats")

    if abs(report['loudness']['difference']) > 2:
        issues.append(f"LOUDNESS: {report['loudness']['difference']:.1f} dB off reference")

    report['issues'] = issues
    report['passes'] = len(issues) == 0

    return report
```

---

## 8.5 — Reference Tracks for Tech House

| Track | Artist | Why It's a Good Reference |
|-------|--------|--------------------------|
| Losing It | Fisher | Clean low end, punchy kick, bright hats |
| You Little Beauty | Fisher | Excellent sidechain groove |
| Opus | Eric Prydz | Sub-bass mastery, arrangement |
| Deceiver | Green Velvet | Minimal clarity, perfect balance |
| Bodies | Chris Lake | Modern tech house tonal balance |
| Cola | CamelPhat & Elderbrook | Vocal tech house reference |

### Where to Get Reference Tracks
1. Buy WAV/FLAC from Beatport (NOT Spotify MP3)
2. Match BPM to your track (use Ableton warp or rubberband CLI)
3. Level-match before comparing (never compare at different loudness)
