"""
Deep stem analysis — study how drums, bass, vocals, and other elements
build and fade throughout reference tech house tracks.
"""
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import os
from scipy.signal import butter, sosfiltfilt

SEP_DIR = "/Volumes/Raphael/Tech House/references/separated_4stem/htdemucs"
BPM = 128
BAR_SEC = 4 * 60.0 / BPM  # Duration of one bar in seconds

def analyze_stem_energy(signal, sr, window_sec=BAR_SEC):
    """Analyze energy over time in bar-sized windows."""
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

def analyze_stem_spectral(signal, sr, window_sec=BAR_SEC):
    """Analyze frequency content over time."""
    window_samples = int(window_sec * sr)
    n_windows = len(signal) // window_samples
    
    bands = {
        'sub': (20, 60),
        'bass': (60, 250),
        'mid': (250, 2000),
        'high': (2000, 8000),
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

def detect_entry_exit(energy, threshold_db=-30):
    """Detect when a stem enters and exits."""
    active = energy > threshold_db
    
    entries = []
    exits = []
    in_section = False
    
    for i, a in enumerate(active):
        if a and not in_section:
            entries.append(i)
            in_section = True
        elif not a and in_section:
            exits.append(i)
            in_section = False
    
    if in_section:
        exits.append(len(energy))
    
    return entries, exits

def analyze_reference(track_name):
    """Deep analysis of a single reference track's stems."""
    track_dir = os.path.join(SEP_DIR, track_name)
    
    if not os.path.isdir(track_dir):
        print(f"  Skipping {track_name} — not found")
        return None
    
    stems = {}
    for stem_name in ['drums', 'bass', 'vocals', 'other']:
        path = os.path.join(track_dir, f"{stem_name}.wav")
        if os.path.exists(path):
            data, sr = sf.read(path)
            mono = data.mean(axis=1) if data.ndim > 1 else data
            stems[stem_name] = mono
    
    if not stems:
        return None
    
    sr = sf.read(os.path.join(track_dir, "drums.wav"))[1]
    duration = len(stems['drums']) / sr
    
    print(f'\n{"="*70}')
    print(f'TRACK: {track_name}')
    print(f'Duration: {duration:.0f}s ({duration/60:.1f} min), {duration/BAR_SEC:.0f} bars')
    print(f'{"="*70}')
    
    # Analyze each stem
    stem_data = {}
    for name, signal in stems.items():
        energy = analyze_stem_energy(signal, sr)
        spectral = analyze_stem_spectral(signal, sr)
        entries, exits = detect_entry_exit(energy, threshold_db=-35)
        stem_data[name] = {
            'energy': energy,
            'spectral': spectral,
            'entries': entries,
            'exits': exits,
        }
    
    # Print energy profiles
    print(f'\n  ENERGY PROFILE (per bar):')
    n_bars = min(len(stem_data['drums']['energy']), 
                 len(stem_data['bass']['energy']),
                 len(stem_data['vocals']['energy']),
                 len(stem_data['other']['energy']))
    
    # Show every 4 bars
    for bar in range(0, n_bars, 4):
        d_e = stem_data['drums']['energy'][bar] if bar < len(stem_data['drums']['energy']) else -60
        b_e = stem_data['bass']['energy'][bar] if bar < len(stem_data['bass']['energy']) else -60
        v_e = stem_data['vocals']['energy'][bar] if bar < len(stem_data['vocals']['energy']) else -60
        o_e = stem_data['other']['energy'][bar] if bar < len(stem_data['other']['energy']) else -60
        
        # Normalize to 0-1 scale for display
        def norm(e): return max(0, min(20, (e + 40) / 2))
        
        d_bar = '█' * int(norm(d_e))
        b_bar = '█' * int(norm(b_e))
        v_bar = '█' * int(norm(v_e))
        o_bar = '█' * int(norm(o_e))
        
        time_sec = bar * BAR_SEC
        print(f'    Bar {bar:3d} ({time_sec:5.0f}s): D:{d_bar:<20s} B:{b_bar:<20s} V:{v_bar:<20s} O:{o_bar:<20s}')
    
    # Entry/exit analysis
    print(f'\n  ELEMENT ENTRY/EXIT:')
    for name in ['drums', 'bass', 'vocals', 'other']:
        entries = stem_data[name]['entries']
        exits = stem_data[name]['exits']
        if entries:
            for i, (entry, exit) in enumerate(zip(entries, exits)):
                entry_sec = entry * BAR_SEC
                exit_sec = exit * BAR_SEC
                duration_sec = exit_sec - entry_sec
                if duration_sec > 2:
                    print(f'    {name:8s}: {entry_sec:6.0f}s - {exit_sec:6.0f}s ({duration_sec:5.0f}s) [bar {entry}-{exit}]')
    
    # Frequency evolution
    print(f'\n  FREQUENCY EVOLUTION (sub% per 8 bars):')
    for bar in range(0, n_bars, 8):
        if bar < len(stem_data['bass']['spectral']['sub']):
            sub_pct = stem_data['bass']['spectral']['sub'][bar]
            bass_pct = stem_data['bass']['spectral']['bass'][bar]
            time_sec = bar * BAR_SEC
            print(f'    Bar {bar:3d} ({time_sec:5.0f}s): sub={sub_pct:5.1f}% bass={bass_pct:5.1f}%')
    
    # Build/drop patterns
    print(f'\n  BUILD/DROP PATTERNS:')
    drums_energy = stem_data['drums']['energy']
    bass_energy = stem_data['bass']['energy']
    
    # Find energy peaks and valleys
    if len(drums_energy) > 8:
        # Smooth with 4-bar window
        smoothed = np.convolve(drums_energy, np.ones(4)/4, mode='same')
        
        # Find peaks (drops)
        peak_threshold = np.mean(smoothed) + np.std(smoothed) * 0.5
        valley_threshold = np.mean(smoothed) - np.std(smoothed) * 0.5
        
        for bar in range(0, len(smoothed), 4):
            if smoothed[bar] > peak_threshold:
                time_sec = bar * BAR_SEC
                print(f'    DROP at bar {bar} ({time_sec:.0f}s): drums={drums_energy[bar]:.1f} dB')
            elif smoothed[bar] < valley_threshold:
                time_sec = bar * BAR_SEC
                print(f'    BREAKDOWN at bar {bar} ({time_sec:.0f}s): drums={drums_energy[bar]:.1f} dB')
    
    return stem_data

# Run analysis
print('='*70)
print('DEEP STEM ANALYSIS — Reference Tracks')
print('='*70)

all_results = {}
for track_name in sorted(os.listdir(SEP_DIR)):
    track_path = os.path.join(SEP_DIR, track_name)
    if os.path.isdir(track_path):
        result = analyze_reference(track_name)
        if result:
            all_results[track_name] = result

# Summary
print(f'\n{"="*70}')
print('SUMMARY — PATTERNS TO INCORPORATE')
print(f'{"="*70}')

for name, data in all_results.items():
    print(f'\n  {name[:50]}:')
    
    # Find when each element typically enters
    for stem in ['drums', 'bass', 'vocals', 'other']:
        entries = data[stem]['entries']
        if entries:
            first_entry = entries[0] * BAR_SEC
            print(f'    {stem:8s} first enters at: {first_entry:.0f}s (bar {entries[0]})')
    
    # Find typical breakdown/drop pattern
    drums_e = data['drums']['energy']
    if len(drums_e) > 16:
        # Find the biggest energy drop
        max_drop = 0
        drop_bar = 0
        for i in range(4, len(drums_e)):
            drop = drums_e[i-4] - drums_e[i]
            if drop > max_drop:
                max_drop = drop
                drop_bar = i
        if max_drop > 3:
            print(f'    Biggest breakdown at bar {drop_bar} ({drop_bar*BAR_SEC:.0f}s): {max_drop:.1f} dB drop')
