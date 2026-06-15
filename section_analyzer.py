"""
Section Analyzer — detects busy/loud sections and maps them to song structure.

Splits audio into beat-synchronized windows, computes energy and onset density,
and returns the busiest sections for the optimizer to evaluate on.

Also provides per-section analysis (verse vs chorus energy levels) so the
optimizer can set different processing targets per section type.
"""

import numpy as np

try:
    import librosa
except ImportError:
    librosa = None


def analyze_sections(y: np.ndarray, sr: int, bpm: float,
                     percentile: int = 80, min_bars: int = 2) -> list:
    """
    Detect busy/loud sections in audio.

    Args:
        y: audio signal (mono or stereo)
        sr: sample rate
        bpm: beats per minute
        percentile: top N% busiest sections to return (default 80 = top 20%)
        min_bars: minimum number of bars to include

    Returns:
        List of section dicts sorted by busyness (descending):
        [{'start_sample': int, 'end_sample': int, 'bar_start': int, 'bar_end': int,
          'energy_db': float, 'onset_density': float, 'busyness': float}, ...]
    """
    if y.ndim == 2:
        mono = np.mean(y, axis=1)
    else:
        mono = y.copy()

    # Beat timing
    samples_per_beat = int(sr * 60.0 / bpm)
    samples_per_bar = samples_per_beat * 4  # 4/4 time
    total_bars = max(1, len(mono) // samples_per_bar)

    # Minimum section length
    min_bars = max(min_bars, 1)

    # Compute per-bar metrics
    bar_metrics = []
    for bar_idx in range(total_bars):
        start = bar_idx * samples_per_bar
        end = min(start + samples_per_bar, len(mono))
        if end - start < samples_per_beat:
            break

        segment = mono[start:end]

        # RMS energy
        rms = np.sqrt(np.mean(segment.astype(np.float64) ** 2))
        energy_db = 20.0 * np.log10(max(rms, 1e-10))

        # Onset density (onsets per bar)
        onset_density = _count_onsets(segment, sr) if librosa else 0.0

        bar_metrics.append({
            'bar_idx': bar_idx,
            'start_sample': start,
            'end_sample': end,
            'energy_db': energy_db,
            'onset_density': onset_density,
        })

    if not bar_metrics:
        return []

    # Compute busyness = normalized_energy * normalized_onset_density
    energies = np.array([m['energy_db'] for m in bar_metrics])
    onsets = np.array([m['onset_density'] for m in bar_metrics])

    # Normalize to 0-1
    e_min, e_max = energies.min(), energies.max()
    o_min, o_max = onsets.min(), onsets.max()

    e_range = max(e_max - e_min, 1e-10)
    o_range = max(o_max - o_min, 1e-10)

    for m in bar_metrics:
        e_norm = (m['energy_db'] - e_min) / e_range
        o_norm = (m['onset_density'] - o_min) / o_range
        m['busyness'] = e_norm * 0.6 + o_norm * 0.4  # weight energy slightly more

    # Sort by busyness descending
    bar_metrics.sort(key=lambda m: m['busyness'], reverse=True)

    # Select top percentile
    cutoff_idx = max(1, int(len(bar_metrics) * (100 - percentile) / 100))
    selected = bar_metrics[:cutoff_idx]

    # Group consecutive bars into sections
    selected.sort(key=lambda m: m['bar_idx'])  # re-sort by time
    sections = _group_consecutive(selected, min_bars)

    return sections


def analyze_per_section_energy(y: np.ndarray, sr: int, bpm: float) -> dict:
    """
    Analyze energy per song section (verse, chorus, etc.) from config.SONG_SECTIONS.

    Returns dict mapping section type to energy metrics:
    {
        'verse': {'bars': (8, 24), 'energy_db': -18.5, 'lufs': -16.2, 'n_bars': 16},
        'chorus': {'bars': (24, 32), 'energy_db': -14.1, 'lufs': -12.8, 'n_bars': 8},
        ...
    }
    """
    import config
    import pyloudnorm as pyln

    if y.ndim == 2:
        mono = np.mean(y, axis=1)
    else:
        mono = y.copy()

    samples_per_beat = int(sr * 60.0 / bpm)
    samples_per_bar = samples_per_beat * 4

    meter = pyln.Meter(sr)
    section_results = {}

    for section_name, (bar_start, bar_end) in config.SONG_SECTIONS.items():
        sample_start = bar_start * samples_per_bar
        sample_end = min(bar_end * samples_per_bar, len(mono))

        if sample_start >= len(mono) or sample_end <= sample_start:
            continue

        segment = y[sample_start:sample_end] if y.ndim == 2 else y[sample_start:sample_end]

        # RMS energy
        rms = np.sqrt(np.mean(segment.astype(np.float64) ** 2))
        energy_db = 20.0 * np.log10(max(rms, 1e-10))

        # LUFS for this section
        try:
            lufs = float(meter.integrated_loudness(segment))
        except Exception:
            lufs = -70.0

        section_results[section_name] = {
            'bars': (bar_start, bar_end),
            'n_bars': bar_end - bar_start,
            'energy_db': energy_db,
            'lufs': lufs,
        }

    return section_results


def get_section_type(bar_idx: int) -> str:
    """Map a bar index to its section type using config.SONG_SECTIONS."""
    import config
    for section_name, (bar_start, bar_end) in config.SONG_SECTIONS.items():
        if bar_start <= bar_idx < bar_end:
            # Simplify: pre_chorus1_build -> chorus, fill1 -> fill, etc.
            if 'pre_chorus' in section_name or 'build' in section_name:
                return 'build'
            if 'fill' in section_name:
                return 'fill'
            if 'intro' in section_name:
                return 'intro'
            if 'outro' in section_name:
                return 'outro'
            if 'chorus' in section_name:
                return 'chorus'
            if 'verse' in section_name:
                return 'verse'
    return 'unknown'


def compute_section_lufs_targets(section_energy: dict) -> dict:
    """
    Compute per-section LUFS adjustment based on measured energy differences.

    If chorus is 3 dB hotter than verse, the optimizer should know to expect
    that and not try to equalize it — just process within the section's range.

    Returns dict: {'verse': target_offset, 'chorus': target_offset, ...}
    """
    if not section_energy:
        return {}

    # Find the average energy of verse sections as baseline
    verse_lufs = []
    chorus_lufs = []
    for name, data in section_energy.items():
        if 'verse' in name and data['lufs'] > -70.0:
            verse_lufs.append(data['lufs'])
        elif 'chorus' in name and data['lufs'] > -70.0:
            chorus_lufs.append(data['lufs'])

    if not verse_lufs:
        return {}

    avg_verse = np.mean(verse_lufs)
    avg_chorus = np.mean(chorus_lufs) if chorus_lufs else avg_verse
    delta = avg_chorus - avg_verse

    # Return offsets relative to the base LUFS target
    # Positive means the section is naturally hotter
    targets = {}
    for name, data in section_energy.items():
        if data['lufs'] <= -70.0:
            targets[name] = 0.0
            continue
        # How much hotter/cooler than verse average
        targets[name] = data['lufs'] - avg_verse

    return targets


def _count_onsets(segment: np.ndarray, sr: int) -> float:
    """Count onsets in a segment using librosa."""
    if librosa is None:
        return 0.0
    try:
        onset_frames = librosa.onset.onset_detect(
            y=segment.astype(np.float32), sr=sr, units='frames',
            hop_length=512, backtrack=False
        )
        return float(len(onset_frames))
    except Exception:
        return 0.0


def _group_consecutive(bars: list, min_bars: int) -> list:
    """Group consecutive bars into sections."""
    if not bars:
        return []

    sections = []
    current = {
        'start_sample': bars[0]['start_sample'],
        'end_sample': bars[0]['end_sample'],
        'bar_start': bars[0]['bar_idx'],
        'bar_end': bars[0]['bar_idx'],
        'energy_db': bars[0]['energy_db'],
        'onset_density': bars[0]['onset_density'],
        'busyness': bars[0]['busyness'],
        'n_bars': 1,
    }

    for bar in bars[1:]:
        if bar['bar_idx'] == current['bar_end'] + 1:
            # Extend current section
            current['end_sample'] = bar['end_sample']
            current['bar_end'] = bar['bar_idx']
            current['energy_db'] = max(current['energy_db'], bar['energy_db'])
            current['onset_density'] = (current['onset_density'] * current['n_bars'] +
                                        bar['onset_density']) / (current['n_bars'] + 1)
            current['busyness'] = max(current['busyness'], bar['busyness'])
            current['n_bars'] += 1
        else:
            # Finalize current, start new
            if current['n_bars'] >= min_bars:
                sections.append(current)
            current = {
                'start_sample': bar['start_sample'],
                'end_sample': bar['end_sample'],
                'bar_start': bar['bar_idx'],
                'bar_end': bar['bar_idx'],
                'energy_db': bar['energy_db'],
                'onset_density': bar['onset_density'],
                'busyness': bar['busyness'],
                'n_bars': 1,
            }

    # Don't forget last section
    if current['n_bars'] >= min_bars:
        sections.append(current)

    # Sort by busyness descending
    sections.sort(key=lambda s: s['busyness'], reverse=True)

    return sections
