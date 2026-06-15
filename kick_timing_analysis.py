"""
Precise kick timing analysis — sample-level grid alignment check.
"""
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

BPM = 128
SR = 44100
BEAT_SAMPLES = int(SR * 60.0 / BPM)
SIXTEENTH = BEAT_SAMPLES // 4

SONG = "TH_0613_1929_128_Emin"

def analyze_kick_timing(path, label):
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    
    # High-pass filter to isolate kick transient from body
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, 80, btype='high', fs=sr, output='sos')
    hp = sosfiltfilt(sos, data)
    
    # Detect transients on the HP filtered signal (cleaner attack detection)
    env = np.abs(hp)
    # Very short smoothing (0.5ms)
    w = int(sr * 0.0005)
    env = np.convolve(env, np.ones(w)/w, mode='same')
    
    # Find peaks: minimum distance = half a beat, minimum height = 30% of max
    min_dist = BEAT_SAMPLES // 2
    threshold = np.max(env) * 0.15
    peaks, props = find_peaks(env, height=threshold, distance=min_dist)
    
    if len(peaks) < 10:
        print(f"  {label}: Only {len(peaks)} transients found — too few")
        return
    
    # Calculate expected beat positions (assume first kick is near beat 1)
    # Find the first strong peak as reference
    first_peak = peaks[0]
    
    # Expected positions: every BEAT_SAMPLES from first peak
    expected = np.arange(first_peak, len(data), BEAT_SAMPLES)
    
    # For each detected peak, find nearest expected beat
    errors_samples = []
    errors_ms = []
    for peak in peaks:
        nearest_idx = np.argmin(np.abs(expected - peak))
        nearest = expected[nearest_idx]
        error = peak - nearest
        errors_samples.append(error)
        errors_ms.append((error / sr) * 1000)
    
    errors_samples = np.array(errors_samples)
    errors_ms = np.array(errors_ms)
    
    print(f"\n  {label}:")
    print(f"    Peaks detected: {len(peaks)}")
    print(f"    Expected beats: {len(expected)}")
    print(f"    First peak at sample {first_peak} ({first_peak/sr:.4f}s)")
    print(f"    Beat interval: {BEAT_SAMPLES} samples ({1000*BEAT_SAMPLES/sr:.2f} ms)")
    print(f"")
    print(f"    Timing errors (ms):")
    print(f"      Mean:   {np.mean(errors_ms):+.3f} ms")
    print(f"      Median: {np.median(errors_ms):+.3f} ms")
    print(f"      Std:    {np.std(errors_ms):.3f} ms")
    print(f"      Min:    {np.min(errors_ms):+.3f} ms")
    print(f"      Max:    {np.max(errors_ms):+.3f} ms")
    print(f"      P10:    {np.percentile(errors_ms, 10):+.3f} ms")
    print(f"      P90:    {np.percentile(errors_ms, 90):+.3f} ms")
    print(f"")
    
    # Check for systematic drift
    # Linear fit to errors over time
    peak_times = peaks / sr
    if len(peak_times) > 2:
        coeffs = np.polyfit(peak_times, errors_ms, 1)
        drift_rate = coeffs[0]  # ms per second
        print(f"    Drift analysis:")
        print(f"      Drift rate: {drift_rate:+.3f} ms/sec")
        if abs(drift_rate) > 0.5:
            print(f"      ⚠ SYSTEMATIC DRIFT DETECTED ({drift_rate:+.3f} ms/sec)")
            print(f"        Over 3 minutes: {drift_rate * 180:+.1f} ms total drift")
        else:
            print(f"      ✓ No significant systematic drift")
    
    # Check for inter-onset interval consistency
    intervals = np.diff(peaks) / sr * 1000  # ms
    print(f"")
    print(f"    Inter-onset intervals:")
    print(f"      Mean:   {np.mean(intervals):.2f} ms (expected: {1000*BEAT_SAMPLES/sr:.2f} ms)")
    print(f"      Std:    {np.std(intervals):.3f} ms")
    print(f"      CV:     {np.std(intervals)/np.mean(intervals)*100:.3f}%")
    
    if np.std(intervals) > 2.0:
        print(f"      ⚠ INTERVAL VARIATION > 2ms — audible timing inconsistency")
    elif np.std(intervals) > 1.0:
        print(f"      ⚡ Interval variation > 1ms — borderline")
    else:
        print(f"      ✓ Interval variation < 1ms — tight")
    
    # Show first 20 errors for inspection
    print(f"")
    print(f"    First 20 beat errors (ms):")
    for i in range(min(20, len(errors_ms))):
        marker = "⚠" if abs(errors_ms[i]) > 2.0 else " "
        print(f"      Beat {i+1:3d}: {errors_ms[i]:+7.3f} ms  {marker}")

print("=" * 60)
print(f"KICK TIMING ANALYSIS — {SONG}")
print(f"BPM: {BPM}, Beat: {BEAT_SAMPLES} samples ({1000*BEAT_SAMPLES/SR:.2f} ms)")
print(f"1 ms = {SR/1000:.1f} samples")
print("=" * 60)

rec_dir = f"output/{SONG}/recordings"

for kick_file in ['drum1_Kick_n36.wav', 'drum2_Kick_n36.wav']:
    analyze_kick_timing(f"{rec_dir}/{kick_file}", kick_file)

# Also check the MIDI file for timing
print("\n" + "=" * 60)
print("MIDI KICK TIMING")
print("=" * 60)

import mido
mid = mido.MidiFile(f"output/{SONG}.mid")
tpb = mid.ticks_per_beat

for track in mid.tracks:
    if 'kick' in track.name.lower() and 'n36' in track.name.lower():
        # Get absolute tick positions of all note_on events
        abs_tick = 0
        kick_ticks = []
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                kick_ticks.append(abs_tick)
        
        if not kick_ticks:
            continue
        
        # Expected: every tpb ticks
        expected_interval = tpb
        
        # Check intervals
        intervals = np.diff(kick_ticks)
        interval_errors = intervals - expected_interval
        
        print(f"\n  {track.name}:")
        print(f"    Ticks per beat: {tpb}")
        print(f"    Note count: {len(kick_ticks)}")
        print(f"    Interval mean: {np.mean(intervals):.1f} ticks (expected: {expected_interval})")
        print(f"    Interval std: {np.std(intervals):.2f} ticks")
        print(f"    Interval error mean: {np.mean(interval_errors):+.2f} ticks")
        print(f"    Interval error std: {np.std(interval_errors):.2f} ticks")
        
        if np.std(intervals) < 1.0:
            print(f"    ✓ MIDI intervals are perfectly quantized")
        else:
            print(f"    ⚠ MIDI intervals have variation")
        
        # Check absolute positions vs grid
        grid_errors = []
        for tick in kick_ticks:
            expected_nearest = round(tick / tpb) * tpb
            grid_errors.append(tick - expected_nearest)
        
        grid_errors = np.array(grid_errors)
        print(f"    Grid alignment:")
        print(f"      Mean: {np.mean(grid_errors):+.2f} ticks")
        print(f"      Std: {np.std(grid_errors):.2f} ticks")
        print(f"      Max: {np.max(np.abs(grid_errors)):.2f} ticks")
        
        if np.max(np.abs(grid_errors)) <= 1:
            print(f"      ✓ All kicks within 1 tick of grid")
        elif np.max(np.abs(grid_errors)) <= 5:
            print(f"      ⚡ Max {np.max(np.abs(grid_errors)):.0f} ticks off grid")
        else:
            print(f"      ⚠ Some kicks > 5 ticks off grid")
