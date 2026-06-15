# Skill 05: Arrangement

## What the Producer Does
Transform a 1-bar loop into a 5-7 minute track. Add/remove elements every
8 bars. Automate filters, volumes, and effects. Create tension and release.
Ensure DJ-friendliness (16-bar intro/outro).

---

## 5.1 — The 8-Bar Rule

**Every 8 bars, something MUST change.** Options:
- Add or remove an element
- Open or close a filter
- Automate reverb send up/down
- Change percussion pattern
- Add/remove ghost kicks

### Programmatic Check

```python
def check_8bar_rule(events_by_bar, tpb=480):
    """Verify that every 8-bar block has at least one change."""
    violations = []
    total_bars = max(events_by_bar.keys()) + 1

    for block_start in range(0, total_bars, 8):
        block_bars = list(range(block_start, min(block_start + 8, total_bars)))
        if len(block_bars) < 2:
            continue

        # Compare first bar to each subsequent bar in the block
        first_bar_notes = set((e['note'], e['velocity'] // 20)  # Quantize vel
                              for e in events_by_bar.get(block_bars[0], []))

        has_change = False
        for bar in block_bars[1:]:
            bar_notes = set((e['note'], e['velocity'] // 20)
                            for e in events_by_bar.get(bar, []))
            if bar_notes != first_bar_notes:
                has_change = True
                break

        if not has_change:
            violations.append(block_start)

    return {
        'passes': len(violations) == 0,
        'violations': violations,
        'total_blocks': total_bars // 8,
    }
```

---

## 5.2 — Energy Curve

### Ideal Tech House Energy Shape
```
Energy
1.0 │          ████████                      ████████████
    │        ██        ██                  ██            ██
0.8 │      ██            ██              ██                ██
    │    ██                ██          ██                    ██
0.6 │  ██                    ██      ██                        ██
    │██                        ██  ██                            ██
0.4 │                            ██                                ██
    │                                                                    ██
0.2 │                                                                      ██
    └──────────────────────────────────────────────────────────────────────
      Intro    Build1    Drop1    Break    Build2    Drop2      Outro
      0-16     16-24     24-40    40-48    48-56     56-72      72-88
```

### Programmatic Energy Measurement

```python
def measure_energy_curve(audio_path, sr=44100, bars_total=88, bpm=126):
    """Measure RMS energy per bar to verify arrangement shape."""
    import soundfile as sf
    data, sr = sf.read(audio_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    bar_duration_s = (60.0 / bpm) * 4  # 4 beats per bar
    samples_per_bar = int(sr * bar_duration_s)

    energy_per_bar = []
    for bar in range(bars_total):
        start = bar * samples_per_bar
        end = min(start + samples_per_bar, len(data))
        if start >= len(data):
            break
        rms = np.sqrt(np.mean(data[start:end]**2))
        energy_per_bar.append(rms)

    # Normalize to 0-1
    max_rms = max(energy_per_bar) if energy_per_bar else 1
    energy_per_bar = [e / max_rms for e in energy_per_bar]

    return energy_per_bar
```

### Objective Assessment — Energy Curve

| Metric | How to Measure | Target |
|--------|---------------|--------|
| Intro energy | Avg RMS of bars 0-15 | 20-40% of peak |
| Drop 1 peak | Max RMS bars 24-39 | 85-100% |
| Breakdown dip | Min RMS bars 40-47 | < 40% of peak |
| Drop 2 peak | Max RMS bars 56-71 | ≥ Drop 1 peak |
| Outro decay | Avg RMS bars 80-87 | < 30% of peak |
| Smooth transitions | No jumps > 30% between adjacent bars | 100% compliance |

---

## 5.3 — Filter Automation

### Real-World Moves
| Section | Filter Move | Range |
|---------|------------|-------|
| Intro | HPF on master opens | 200 Hz → 20 Hz over 16 bars |
| Build | LPF on bass opens | 300 Hz → 1200 Hz over 8 bars |
| Breakdown | LPF on everything closes | Open → 400 Hz over 4 bars |
| Build 2 | LPF opens + resonance rises | 400 Hz → 2000 Hz, res 10% → 30% |

### Programmatic Filter Automation

```python
def apply_filter_automation(signal, sr, filter_type='lowpass',
                            start_hz=400, end_hz=2000, bars=8, bpm=126):
    """Automate a filter cutoff over N bars."""
    bar_samples = int(sr * (60.0 / bpm) * 4)
    total_samples = bar_samples * bars
    signal = signal[:total_samples]

    # Create time-varying cutoff
    cutoffs = np.linspace(start_hz, end_hz, len(signal))
    # Smooth with moving average
    window = int(sr * 0.05)  # 50ms smoothing
    cutoffs = np.convolve(cutoffs, np.ones(window)/window, mode='same')

    # Apply blockwise (512-sample blocks)
    block_size = 512
    output = np.zeros_like(signal)
    for i in range(0, len(signal) - block_size, block_size):
        cutoff = np.mean(cutoffs[i:i+block_size])
        cutoff = np.clip(cutoff, 30, sr/2 - 100)
        sos = butter(2, cutoff, btype=filter_type, fs=sr, output='sos')
        output[i:i+block_size] = sosfiltfilt(sos, signal[i:i+block_size])

    return output
```

---

## 5.4 — DJ-Friendliness Checklist

| Criterion | How to Check | Target |
|-----------|-------------|--------|
| Intro length | Bars of intro with drums-only | ≥ 16 bars |
| Outro length | Bars of outro with drums-only | ≥ 16 bars |
| No sudden starts | First bar RMS < 20% of peak | ✓ |
| No sudden endings | Last bar RMS < 10% of peak | ✓ |
| Kick present throughout | Kick hits in every bar | 100% |
| Consistent key | No key changes | ✓ |
| Total length | (bars × 4 × 60) / bpm | 5-7 minutes |
