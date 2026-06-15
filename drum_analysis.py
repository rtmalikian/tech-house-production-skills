#!/usr/bin/env python3
"""
Drum transient analysis — diagnose galloping rhythm.
Checks timing accuracy, compression artifacts, and amplitude modulation.
"""
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from scipy.signal import find_peaks, butter, sosfiltfilt
import os

SONG = "TH_0613_1903_127_Amin"
BPM = 127
SR = 44100
BEAT_SAMPLES = int(SR * 60.0 / BPM)  # Samples per beat
SIXTEENTH_SAMPLES = BEAT_SAMPLES // 4

def load_stem(path):
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr

def detect_transients(signal, sr, threshold_ratio=0.3, min_distance_ms=30):
    """Detect transients using onset envelope."""
    envelope = np.abs(signal)
    # Smooth with 2ms window
    window = int(sr * 0.002)
    envelope = np.convolve(envelope, np.ones(window)/window, mode='same')
    
    threshold = np.max(envelope) * threshold_ratio
    min_distance = int(sr * min_distance_ms / 1000)
    
    peaks, props = find_peaks(envelope, height=threshold, distance=min_distance)
    return peaks, envelope

def analyze_kick_timing(kick_path, bpm, sr):
    """Analyze kick transient timing against the grid."""
    data, _ = load_stem(kick_path)
    peaks, envelope = detect_transients(data, sr, threshold_ratio=0.3)
    
    if len(peaks) < 2:
        return None
    
    beat_samples = int(sr * 60.0 / bpm)
    
    # Expected kick positions (every beat)
    expected_beats = np.arange(0, len(data), beat_samples)
    
    # For each detected transient, find nearest expected beat
    timing_errors = []
    for peak in peaks:
        nearest_beat = expected_beats[np.argmin(np.abs(expected_beats - peak))]
        error_samples = peak - nearest_beat
        error_ms = (error_samples / sr) * 1000
        timing_errors.append(error_ms)
    
    timing_errors = np.array(timing_errors)
    
    return {
        'num_transients': len(peaks),
        'expected_transients': len(expected_beats),
        'timing_error_mean_ms': np.mean(timing_errors),
        'timing_error_std_ms': np.std(timing_errors),
        'timing_error_max_ms': np.max(np.abs(timing_errors)),
        'timing_errors': timing_errors,
        'peak_positions': peaks,
    }

def analyze_amplitude_modulation(signal, sr, bpm):
    """Detect amplitude modulation at beat/sub-beat rate (compression pumping)."""
    envelope = np.abs(signal)
    # Smooth with 10ms window
    window = int(sr * 0.01)
    envelope = np.convolve(envelope, np.ones(window)/window, mode='same')
    
    # FFT of envelope to find modulation frequencies
    fft = np.fft.rfft(envelope)
    mag = np.abs(fft)
    freqs = np.fft.rfftfreq(len(envelope), 1/sr)
    
    # Check modulation at beat rate and sub-beat rates
    beat_hz = bpm / 60.0
    results = {}
    
    for label, target_hz in [
        ('beat_rate', beat_hz),
        ('half_beat', beat_hz * 2),
        ('quarter_beat', beat_hz * 4),
        ('eighth_beat', beat_hz * 8),
    ]:
        # Find energy near target frequency (±0.2 Hz tolerance)
        mask = (freqs >= target_hz - 0.2) & (freqs <= target_hz + 0.2)
        energy = np.sum(mag[mask])
        # Compare to average energy
        avg_energy = np.mean(mag[(freqs > 0.5) & (freqs < 20)])
        ratio = energy / (avg_energy + 1e-10)
        results[label] = {
            'target_hz': target_hz,
            'energy_ratio': float(ratio),
            'significant': ratio > 3.0,  # 3x average = significant modulation
        }
    
    return results

def analyze_gain_reduction(master_path, sr):
    """Analyze if the master shows compression pumping."""
    data, _ = load_stem(master_path)
    
    # Measure RMS in 10ms windows
    window = int(sr * 0.01)
    rms_windows = []
    for i in range(0, len(data) - window, window):
        rms_windows.append(np.sqrt(np.mean(data[i:i+window]**2)))
    
    rms_windows = np.array(rms_windows)
    rms_db = 20 * np.log10(rms_windows + 1e-10)
    
    # Look for rhythmic dips in RMS (compression pumping)
    # The gain reduction pattern would show as periodic dips
    peaks_rms, _ = find_peaks(-rms_db, distance=3, prominence=1)  # Dips = peaks of inverted
    
    # Calculate variation in RMS
    rms_variation = np.std(rms_db)
    rms_range = np.percentile(rms_db, 95) - np.percentile(rms_db, 5)
    
    return {
        'rms_mean_db': float(np.mean(rms_db)),
        'rms_std_db': float(rms_variation),
        'rms_range_db': float(rms_range),
        'num_rhythm_dips': len(peaks_rms),
        'dip_rate_hz': len(peaks_rms) / (len(data) / sr) if len(data) > 0 else 0,
    }

def analyze_stem_level_consistency(signal, sr, bpm):
    """Check if individual hit levels are consistent (or if compression is pumping)."""
    beat_samples = int(sr * 60.0 / bpm)
    peaks, _ = detect_transients(signal, sr, threshold_ratio=0.3)
    
    if len(peaks) < 4:
        return None
    
    # Measure peak level of each transient
    peak_levels = []
    for peak in peaks:
        start = max(0, peak - 100)
        end = min(len(signal), peak + 100)
        peak_levels.append(np.max(np.abs(signal[start:end])))
    
    peak_levels = np.array(peak_levels)
    peak_db = 20 * np.log10(peak_levels + 1e-10)
    
    return {
        'mean_level_db': float(np.mean(peak_db)),
        'std_level_db': float(np.std(peak_db)),
        'range_level_db': float(np.max(peak_db) - np.min(peak_db)),
        'consistent': np.std(peak_db) < 2.0,  # < 2 dB variation = consistent
    }

# ============================================================================
# RUN ANALYSIS
# ============================================================================

print("=" * 70)
print(f"DRUM TRANSIENT ANALYSIS — {SONG}")
print(f"BPM: {BPM}, Beat: {BEAT_SAMPLES} samples ({1000*BEAT_SAMPLES/SR:.1f} ms)")
print("=" * 70)

recordings_dir = f"output/{SONG}/recordings"
master_path = f"output/{SONG}/mastered/corrected_master.wav"

# 1. KICK TIMING ANALYSIS
print("\n1. KICK TIMING vs GRID")
print("-" * 40)
for kick_file in ['drum1_Kick_n36.wav', 'drum2_Kick_n36.wav']:
    path = os.path.join(recordings_dir, kick_file)
    if os.path.exists(path):
        result = analyze_kick_timing(path, BPM, SR)
        if result:
            print(f"\n  {kick_file}:")
            print(f"    Transients: {result['num_transients']} (expected ~{result['expected_transients']})")
            print(f"    Timing error mean: {result['timing_error_mean_ms']:+.2f} ms")
            print(f"    Timing error std:  {result['timing_error_std_ms']:.2f} ms")
            print(f"    Timing error max:  {result['timing_error_max_ms']:.2f} ms")
            
            if result['timing_error_std_ms'] > 2.0:
                print(f"    ⚠ TIMING IS LOOSE (std > 2ms)")
            elif result['timing_error_std_ms'] > 1.0:
                print(f"    ⚡ TIMING SLIGHTLY LOOSE (std > 1ms)")
            else:
                print(f"    ✓ TIMING TIGHT (std < 1ms)")
            
            # Show distribution of errors
            errors = result['timing_errors']
            print(f"    Distribution: {np.percentile(errors, 10):.1f} / {np.percentile(errors, 50):.1f} / {np.percentile(errors, 90):.1f} ms (10th/50th/90th)")

# 2. AMPLITUDE MODULATION ANALYSIS
print("\n2. AMPLITUDE MODULATION (Compression Pumping)")
print("-" * 40)
for kick_file in ['drum1_Kick_n36.wav', 'drum2_Kick_n36.wav']:
    path = os.path.join(recordings_dir, kick_file)
    if os.path.exists(path):
        data, _ = load_stem(path)
        mod = analyze_amplitude_modulation(data, SR, BPM)
        print(f"\n  {kick_file}:")
        for label, info in mod.items():
            status = "⚠ SIGNIFICANT" if info['significant'] else "✓ OK"
            print(f"    {label}: {info['target_hz']:.1f} Hz, ratio={info['energy_ratio']:.1f} {status}")

# 3. KICK LEVEL CONSISTENCY
print("\n3. KICK LEVEL CONSISTENCY")
print("-" * 40)
for kick_file in ['drum1_Kick_n36.wav', 'drum2_Kick_n36.wav']:
    path = os.path.join(recordings_dir, kick_file)
    if os.path.exists(path):
        data, _ = load_stem(path)
        result = analyze_stem_level_consistency(data, SR, BPM)
        if result:
            print(f"\n  {kick_file}:")
            print(f"    Mean level: {result['mean_level_db']:.1f} dBFS")
            print(f"    Std dev:    {result['std_level_db']:.1f} dB")
            print(f"    Range:      {result['range_level_db']:.1f} dB")
            status = "✓ CONSISTENT" if result['consistent'] else "⚠ VARIABLE"
            print(f"    {status}")

# 4. MASTER COMPRESSION ANALYSIS
print("\n4. MASTER COMPRESSION ANALYSIS")
print("-" * 40)
if os.path.exists(master_path):
    data, _ = load_stem(master_path)
    gain = analyze_gain_reduction(master_path, SR)
    print(f"  RMS mean: {gain['rms_mean_db']:.1f} dBFS")
    print(f"  RMS std:  {gain['rms_std_db']:.1f} dB")
    print(f"  RMS range: {gain['rms_range_db']:.1f} dB")
    print(f"  Rhythm dips: {gain['num_rhythm_dips']}")
    print(f"  Dip rate: {gain['dip_rate_hz']:.1f} Hz")
    
    beat_hz = BPM / 60.0
    if abs(gain['dip_rate_hz'] - beat_hz) < 1.0:
        print(f"  ⚠ COMPRESSION PUMPING AT BEAT RATE ({beat_hz:.1f} Hz)")
    elif gain['dip_rate_hz'] > beat_hz * 1.5:
        print(f"  ⚠ COMPRESSION PUMPING AT SUB-BEAT RATE")
    else:
        print(f"  ✓ No rhythmic compression pumping detected")

# 5. CLAP/SNARE TIMING
print("\n5. CLAP/SNARE TIMING")
print("-" * 40)
for clap_file in ['drum1_Clap_n39.wav', 'drum2_Clap_n39.wav']:
    path = os.path.join(recordings_dir, clap_file)
    if os.path.exists(path):
        data, _ = load_stem(path)
        peaks, _ = detect_transients(data, SR, threshold_ratio=0.3)
        if len(peaks) > 2:
            # Claps should be on beats 2 and 4
            beat_samples = int(SR * 60.0 / BPM)
            expected_claps = []
            for beat in range(1, int(len(data) / beat_samples)):
                if beat % 2 == 1:  # Beats 2, 4, 6, 8...
                    expected_claps.append(beat * beat_samples)
            
            errors = []
            for peak in peaks:
                nearest = expected_claps[np.argmin(np.abs(np.array(expected_claps) - peak))] if expected_claps else peak
                error_ms = ((peak - nearest) / SR) * 1000
                errors.append(error_ms)
            
            errors = np.array(errors)
            print(f"\n  {clap_file}:")
            print(f"    Transients: {len(peaks)}")
            print(f"    Timing error mean: {np.mean(errors):+.2f} ms")
            print(f"    Timing error std:  {np.std(errors):.2f} ms")
            status = "✓ TIGHT" if np.std(errors) < 1.5 else "⚠ LOOSE"
            print(f"    {status}")

# 6. HAT TIMING (are hats driving the gallop?)
print("\n6. HI-HAT TIMING ANALYSIS")
print("-" * 40)
for hat_file in ['drum1_ClosedHat_n42.wav', 'drum2_ClosedHat_n42.wav']:
    path = os.path.join(recordings_dir, hat_file)
    if os.path.exists(path):
        data, _ = load_stem(path)
        peaks, _ = detect_transients(data, SR, threshold_ratio=0.15, min_distance_ms=10)
        if len(peaks) > 4:
            # Calculate inter-onset intervals
            intervals = np.diff(peaks) / SR * 1000  # ms
            
            sixteenth_ms = (60.0 / BPM / 4) * 1000
            
            print(f"\n  {hat_file}:")
            print(f"    Transients: {len(peaks)}")
            print(f"    Interval mean: {np.mean(intervals):.1f} ms (expected ~{sixteenth_ms:.1f} ms for 16ths)")
            print(f"    Interval std:  {np.std(intervals):.1f} ms")
            print(f"    Interval CV:   {np.std(intervals)/np.mean(intervals)*100:.1f}% (coefficient of variation)")
            
            if np.std(intervals) > 3.0:
                print(f"    ⚠ HAT TIMING IS LOOSE (std > 3ms)")
            else:
                print(f"    ✓ HAT TIMING OK")

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)
