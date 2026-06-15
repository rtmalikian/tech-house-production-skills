"""
Precise kick timing — uses sync click as reference, not first kick.
"""
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, find_peaks

BPM = 128
SR = 44100
BEAT_MS = 60000.0 / BPM  # 468.75 ms at 128 BPM
BEAT_SAMPLES = int(SR * 60.0 / BPM)
SONG = "TH_0613_1929_128_Emin"

def find_sync_click(multi_ch_path):
    """Find the sync click on USB 31/32 (channels 30-31)."""
    data, sr = sf.read(multi_ch_path)
    if data.shape[1] <= 30:
        return None, None
    sync = data[:, 30]
    peak = np.max(np.abs(sync))
    if peak < 0.01:
        return None, None
    env = np.abs(sync)
    threshold = peak * 0.5
    above = np.where(env > threshold)[0]
    if len(above) == 0:
        return None, None
    return above[0], sr

def analyze_kick_with_reference(kick_path, song_start_sample, sr):
    """Analyze kick timing using song start as reference."""
    data, _ = sf.read(kick_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    
    # Band-pass filter around kick fundamental (30-100 Hz)
    sos = butter(4, [30, 100], btype='band', fs=sr, output='sos')
    bp = sosfiltfilt(sos, data)
    
    # Envelope
    env = np.abs(bp)
    w = int(sr * 0.002)  # 2ms smoothing
    env = np.convolve(env, np.ones(w)/w, mode='same')
    
    # Find peaks after song start
    min_dist = BEAT_SAMPLES - int(sr * 0.05)  # Allow 50ms tolerance
    threshold = np.max(env[song_start_sample:]) * 0.2
    peaks, _ = find_peaks(env[song_start_sample:], height=threshold, distance=min_dist)
    peaks = peaks + song_start_sample  # Offset to absolute position
    
    if len(peaks) < 10:
        return None
    
    # Expected grid: song_start + n * BEAT_SAMPLES
    total_samples = len(data)
    expected = np.arange(song_start_sample, total_samples, BEAT_SAMPLES)
    
    # For each detected kick, find nearest expected beat
    errors_ms = []
    matched_kicks = []
    for peak in peaks:
        if peak < song_start_sample:
            continue
        nearest_idx = np.argmin(np.abs(expected - peak))
        nearest = expected[nearest_idx]
        error_samples = peak - nearest
        error_ms = (error_samples / sr) * 1000
        errors_ms.append(error_ms)
        matched_kicks.append(peak)
    
    if not errors_ms:
        return None
    
    errors_ms = np.array(errors_ms)
    matched_kicks = np.array(matched_kicks)
    
    # Inter-kick intervals
    intervals = np.diff(matched_kicks) / sr * 1000
    
    return {
        'num_kicks': len(peaks),
        'num_expected': len(expected),
        'errors_ms': errors_ms,
        'mean_error': np.mean(errors_ms),
        'median_error': np.median(errors_ms),
        'std_error': np.std(errors_ms),
        'max_error': np.max(np.abs(errors_ms)),
        'intervals_ms': intervals,
        'interval_mean': np.mean(intervals),
        'interval_std': np.std(intervals),
        'song_start_sample': song_start_sample,
    }

# Find sync click
pass1_all = f"output/{SONG}/recordings/pass01_all.wav"
click_sample, sr = find_sync_click(pass1_all)

print("=" * 60)
print(f"KICK TIMING ANALYSIS — {SONG}")
print(f"BPM: {BPM}, Beat: {BEAT_MS:.2f} ms, {BEAT_SAMPLES} samples")
print("=" * 60)

if click_sample:
    # Song starts 4 beats after click
    beats_per_second = BPM / 60.0
    song_start = click_sample + int(4 * sr / beats_per_second)
    print(f"\nSync click at sample {click_sample} ({click_sample/sr:.3f}s)")
    print(f"Song start at sample {song_start} ({song_start/sr:.3f}s)")
    
    for kick_file in ['drum1_Kick_n36.wav', 'drum2_Kick_n36.wav']:
        path = f"output/{SONG}/recordings/{kick_file}"
        result = analyze_kick_with_reference(path, song_start, sr)
        
        if result:
            print(f"\n  {kick_file}:")
            print(f"    Kicks detected: {result['num_kicks']} (expected ~{result['num_expected']})")
            print(f"    Timing error mean:   {result['mean_error']:+.3f} ms")
            print(f"    Timing error median: {result['median_error']:+.3f} ms")
            print(f"    Timing error std:    {result['std_error']:.3f} ms")
            print(f"    Timing error max:    {result['max_error']:.3f} ms")
            print(f"")
            print(f"    Inter-kick intervals:")
            print(f"      Mean: {result['interval_mean']:.2f} ms (expected: {BEAT_MS:.2f} ms)")
            print(f"      Std:  {result['interval_std']:.3f} ms")
            
            # Verdict
            if result['std_error'] < 1.0:
                print(f"    ✓ KICKS ARE ON THE GRID (std < 1ms)")
            elif result['std_error'] < 2.0:
                print(f"    ⚡ KICKS SLIGHTLY LOOSE (std 1-2ms)")
            elif result['std_error'] < 5.0:
                print(f"    ⚠ KICKS LOOSE (std 2-5ms) — audible")
            else:
                print(f"    🔴 KICKS ARE OFF GRID (std > 5ms)")
            
            # Show first 30 errors
            print(f"")
            print(f"    First 30 beat errors (ms):")
            for i in range(min(30, len(result['errors_ms']))):
                e = result['errors_ms'][i]
                marker = "🔴" if abs(e) > 5 else "⚡" if abs(e) > 2 else "✓"
                print(f"      Beat {i+1:3d}: {e:+8.3f} ms  {marker}")
else:
    print("\n  No sync click found — cannot do reference-based analysis")
    
    # Fallback: analyze intervals only
    for kick_file in ['drum1_Kick_n36.wav', 'drum2_Kick_n36.wav']:
        path = f"output/{SONG}/recordings/{kick_file}"
        data, sr = sf.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        
        # Band-pass filter
        sos = butter(4, [30, 100], btype='band', fs=sr, output='sos')
        bp = sosfiltfilt(sos, data)
        env = np.abs(bp)
        w = int(sr * 0.002)
        env = np.convolve(env, np.ones(w)/w, mode='same')
        
        min_dist = BEAT_SAMPLES - int(sr * 0.05)
        threshold = np.max(env) * 0.2
        peaks, _ = find_peaks(env, height=threshold, distance=min_dist)
        
        if len(peaks) > 2:
            intervals = np.diff(peaks) / sr * 1000
            print(f"\n  {kick_file} (interval-only analysis):")
            print(f"    Peaks: {len(peaks)}")
            print(f"    Interval mean: {np.mean(intervals):.2f} ms (expected: {BEAT_MS:.2f} ms)")
            print(f"    Interval std:  {np.std(intervals):.3f} ms")
            print(f"    Interval CV:   {np.std(intervals)/np.mean(intervals)*100:.3f}%")
