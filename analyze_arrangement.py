"""
Analyze arrangement patterns in separated reference tracks.
Detects element entry/exit, filter sweeps, builds, and fades.
"""
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import os
from scipy.signal import butter, sosfiltfilt

SEP_DIR = "/Volumes/Raphael/Tech House/references/separated/htdemucs"

def analyze_energy_curve(signal, sr, window_sec=2.0):
    """Analyze energy over time in windows."""
    window_samples = int(window_sec * sr)
    n_windows = len(signal) // window_samples
    
    energies = []
    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples
        chunk = signal[start:end]
        rms = np.sqrt(np.mean(chunk**2))
        energies.append(20 * np.log10(rms + 1e-10))
    
    return np.array(energies)

def analyze_frequency_bands_over_time(signal, sr, window_sec=2.0):
    """Analyze frequency band energy over time."""
    window_samples = int(window_sec * sr)
    n_windows = len(signal) // window_samples
    
    bands = {
        'sub': (20, 60),
        'bass': (60, 250),
        'mid': (250, 2000),
        'high': (2000, 8000),
        'air': (8000, 20000),
    }
    
    result = {name: [] for name in bands}
    
    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples
        chunk = signal[start:end]
        
        fft = np.fft.rfft(chunk)
        mag = np.abs(fft)**2
        freqs = np.fft.rfftfreq(len(chunk), 1/sr)
        total = np.sum(mag) + 1e-10
        
        for name, (lo, hi) in bands.items():
            pct = np.sum(mag[(freqs >= lo) & (freqs < hi)]) / total * 100
            result[name].append(pct)
    
    return {k: np.array(v) for k, v in result.items()}

def detect_arrangement_sections(energy, threshold_ratio=0.3):
    """Detect sections based on energy levels."""
    threshold = np.min(energy) + (np.max(energy) - np.min(energy)) * threshold_ratio
    
    sections = []
    in_section = False
    section_start = 0
    
    for i, e in enumerate(energy):
        if e > threshold and not in_section:
            in_section = True
            section_start = i
        elif e <= threshold and in_section:
            in_section = False
            sections.append((section_start, i, 'high'))
        elif e <= threshold * 0.8 and not in_section:
            sections.append((i, i+1, 'low'))
    
    if in_section:
        sections.append((section_start, len(energy), 'high'))
    
    return sections

def analyze_reference(track_name):
    """Analyze a single reference track's arrangement."""
    no_vocals_path = os.path.join(SEP_DIR, track_name, "no_vocals.wav")
    vocals_path = os.path.join(SEP_DIR, track_name, "vocals.wav")
    
    if not os.path.exists(no_vocals_path):
        print(f"  Skipping {track_name} — no_vocals.wav not found")
        return None
    
    print(f'\n{"="*60}')
    print(f'ANALYZING: {track_name}')
    print(f'{"="*60}')
    
    # Load stems
    no_vocals, sr = sf.read(no_vocals_path)
    mono_nv = no_vocals.mean(axis=1) if no_vocals.ndim > 1 else no_vocals
    
    has_vocals = os.path.exists(vocals_path)
    if has_vocals:
        vocals, _ = sf.read(vocals_path)
        mono_v = vocals.mean(axis=1) if vocals.ndim > 1 else vocals
    
    duration = len(mono_nv) / sr
    window = 2.0  # 2-second windows
    
    print(f'  Duration: {duration:.0f}s ({duration/60:.1f} min)')
    print(f'  Sample rate: {sr} Hz')
    
    # Energy curves
    nv_energy = analyze_energy_curve(mono_nv, sr, window)
    if has_vocals:
        v_energy = analyze_energy_curve(mono_v, sr, window)
    
    # Frequency band analysis
    nv_bands = analyze_frequency_bands_over_time(mono_nv, sr, window)
    
    # Detect sections
    sections = detect_arrangement_sections(nv_energy)
    
    print(f'\n  ARRANGEMENT SECTIONS:')
    for i, (start, end, level) in enumerate(sections):
        start_sec = start * window
        end_sec = end * window
        duration_sec = end_sec - start_sec
        if duration_sec > 4:  # Only show sections > 4 seconds
            print(f'    Section {i+1}: {start_sec:.0f}s - {end_sec:.0f}s ({duration_sec:.0f}s) [{level}]')
    
    # Energy profile over time (every 8 bars ≈ 15s at 128 BPM)
    print(f'\n  ENERGY PROFILE (8-bar windows):')
    bar_8 = 8 * 4 * 60 / 128  # 8 bars at 128 BPM = ~15s
    windows_per_8bar = int(bar_8 / window)
    
    for i in range(0, len(nv_energy), windows_per_8bar):
        end = min(i + windows_per_8bar, len(nv_energy))
        chunk = nv_energy[i:end]
        if len(chunk) > 0:
            avg = np.mean(chunk)
            time_sec = i * window
            bar_num = int(time_sec / (4 * 60 / 128))
            print(f'    Bar {bar_num:3d} ({time_sec:5.0f}s): {avg:5.1f} dB  {"█" * int((avg + 40) / 2)}')
    
    # Frequency band evolution
    print(f'\n  FREQUENCY EVOLUTION:')
    for band_name in ['sub', 'bass', 'mid', 'high']:
        band_data = nv_bands[band_name]
        for i in range(0, len(band_data), windows_per_8bar):
            end = min(i + windows_per_8bar, len(band_data))
            chunk = band_data[i:end]
            if len(chunk) > 0:
                avg = np.mean(chunk)
                time_sec = i * window
                bar_num = int(time_sec / (4 * 60 / 128))
                if bar_num % 16 == 0:  # Show every 16 bars
                    print(f'    {band_name:5s} bar {bar_num:3d}: {avg:5.1f}%')
    
    # Vocal entry/exit analysis
    if has_vocals:
        print(f'\n  VOCAL ENTRY/EXIT:')
        v_sections = detect_arrangement_sections(v_energy, threshold_ratio=0.2)
        for i, (start, end, level) in enumerate(v_sections):
            start_sec = start * window
            end_sec = end * window
            duration_sec = end_sec - start_sec
            if duration_sec > 4 and level == 'high':
                print(f'    Vocal active: {start_sec:.0f}s - {end_sec:.0f}s ({duration_sec:.0f}s)')
    
    return {
        'duration': duration,
        'energy': nv_energy,
        'bands': nv_bands,
        'sections': sections,
    }

# Analyze all separated tracks
print('='*60)
print('REFERENCE TRACK ARRANGEMENT ANALYSIS')
print('='*60)

results = {}
for track_name in sorted(os.listdir(SEP_DIR)):
    track_path = os.path.join(SEP_DIR, track_name)
    if os.path.isdir(track_path):
        result = analyze_reference(track_name)
        if result:
            results[track_name] = result

# Summary
print(f'\n{"="*60}')
print('SUMMARY — ARRANGEMENT PATTERNS')
print(f'{"="*60}')

for name, data in results.items():
    duration = data['duration']
    energy = data['energy']
    
    # Find energy peaks (drops)
    peak_threshold = np.mean(energy) + np.std(energy) * 0.5
    peaks = np.where(energy > peak_threshold)[0]
    
    # Find energy valleys (breakdowns)
    valley_threshold = np.mean(energy) - np.std(energy) * 0.5
    valleys = np.where(energy < valley_threshold)[0]
    
    print(f'\n  {name[:50]}:')
    print(f'    Duration: {duration:.0f}s ({duration/60:.1f} min)')
    print(f'    Energy range: {np.min(energy):.1f} to {np.max(energy):.1f} dB')
    print(f'    Drop windows: {len(peaks)} ({len(peaks)*2:.0f}s)')
    print(f'    Breakdown windows: {len(valleys)} ({len(valleys)*2:.0f}s)')
    
    # Average duration of sections
    if len(peaks) > 0:
        # Find contiguous peak regions
        peak_diffs = np.diff(peaks)
        drop_starts = [peaks[0]]
        for i, d in enumerate(peak_diffs):
            if d > 2:  # Gap > 4 seconds
                drop_starts.append(peaks[i+1])
        print(f'    Number of drops: {len(drop_starts)}')
