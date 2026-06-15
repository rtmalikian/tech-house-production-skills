"""
STEM QC v2 — Transient-aware level checking.
Catches both overall balance issues AND per-transient spikes.
"""
import numpy as np
import soundfile as sf
import os
import json
import sys
from scipy.signal import find_peaks

# ============================================================================
# TRANSIENT-AWARE QC THRESHOLDS
# ============================================================================
THRESHOLDS = {
    'max_peak_dbfs': -3.0,           # No stem should peak above -3 dBFS
    'max_transient_ratio': 8.0,      # Peak/RMS ratio above 8x = problematic
    'max_transient_peak_dbfs': -1.0, # Individual transient max -1 dBFS
    'active_threshold_dbfs': -40.0,  # Consider stem "playing" above this
}

STEM_CATEGORIES = {
    'kick': ['kick_n36'], 'bass': ['bass'], 'sub_bass': ['sub_bass'],
    'clap': ['clap_n39'], 'snare': ['snare_n38'],
    'closed_hat': ['closedhat_n42'], 'open_hat': ['openhat_n46'],
    'ride': ['ride_n51'], 'crash': ['crash_n49'],
    'tambourine': ['tambourine_n54'], 'shaker': ['shaker', 'maracas'],
    'sidestick': ['sidestick_n37'], 'cowbell': ['cowbell_n56'],
    'stab': ['chord_stab'], 'acid': ['acid_line'],
    'pad': ['pad'], 'fx': ['fx'],
}

def classify_stem(filename):
    fname_lower = filename.lower()
    for category, patterns in STEM_CATEGORIES.items():
        for pattern in patterns:
            if pattern in fname_lower:
                return category
    return 'other'

def analyze_stem_transients(path, sr=None):
    """Analyze a stem for transient spikes."""
    data, sr = sf.read(path)
    mono = data.mean(axis=1) if data.ndim > 1 else data
    
    # Overall stats
    overall_peak = 20 * np.log10(np.max(np.abs(mono)) + 1e-10)
    overall_rms = 20 * np.log10(np.sqrt(np.mean(mono**2)) + 1e-10)
    
    # Find transients using onset detection
    # Use a short window (5ms) to find attack peaks
    window = int(sr * 0.005)  # 5ms
    hop = int(sr * 0.01)      # 10ms hop
    
    transients = []
    for i in range(0, len(mono) - window, hop):
        chunk = mono[i:i+window]
        peak = np.max(np.abs(chunk))
        if peak > 10 ** (THRESHOLDS['active_threshold_dbfs'] / 20):
            peak_db = 20 * np.log10(peak + 1e-10)
            time_sec = i / sr
            transients.append({
                'time_sec': time_sec,
                'peak_db': peak_db,
                'peak_val': peak,
            })
    
    # Find the loudest transients
    if transients:
        peaks_db = [t['peak_db'] for t in transients]
        max_transient = max(peaks_db)
        mean_transient = np.mean(peaks_db)
        std_transient = np.std(peaks_db)
        
        # Find transients that are spikes (much louder than average)
        spike_threshold = mean_transient + std_transient * 2
        spikes = [t for t in transients if t['peak_db'] > spike_threshold]
        
        # Find transients that are clipping
        clipping = [t for t in transients if t['peak_db'] > THRESHOLDS['max_peak_dbfs']]
    else:
        max_transient = -100
        mean_transient = -100
        std_transient = 0
        spikes = []
        clipping = []
    
    return {
        'overall_peak': overall_peak,
        'overall_rms': overall_rms,
        'transient_ratio': 10 ** ((overall_peak - overall_rms) / 20) if overall_rms > -60 else 0,
        'max_transient_db': max_transient,
        'mean_transient_db': mean_transient,
        'std_transient_db': std_transient,
        'num_transients': len(transients),
        'num_spikes': len(spikes),
        'num_clipping': len(clipping),
        'spikes': spikes[:10],  # Top 10 spikes
        'clipping': clipping[:10],  # Top 10 clipping moments
    }

def find_spiking_stems(stems_dir, window_sec=2.0):
    """Find stems with transient spikes in any 2-second window."""
    issues = []
    
    for fname in sorted(os.listdir(stems_dir)):
        if not fname.endswith('.wav') or fname.startswith('pass'):
            continue
        
        path = os.path.join(stems_dir, fname)
        try:
            data, sr = sf.read(path)
            mono = data.mean(axis=1) if data.ndim > 1 else data
            category = classify_stem(fname)
            
            # Analyze in 2-second windows
            window_samples = int(window_sec * sr)
            n_windows = len(mono) // window_samples
            
            for i in range(n_windows):
                start = i * window_samples
                end = start + window_samples
                chunk = mono[start:end]
                
                peak = np.max(np.abs(chunk))
                peak_db = 20 * np.log10(peak + 1e-10)
                rms = np.sqrt(np.mean(chunk**2))
                rms_db = 20 * np.log10(rms + 1e-10) if rms > 1e-10 else -100
                
                # Transient ratio
                if rms > 1e-10:
                    ratio = peak / rms
                else:
                    ratio = 0
                
                # Flag issues
                if peak_db > THRESHOLDS['max_peak_dbfs']:
                    issues.append({
                        'stem': fname,
                        'category': category,
                        'time_sec': i * window_sec,
                        'window': i,
                        'peak_db': peak_db,
                        'rms_db': rms_db,
                        'ratio': ratio,
                        'issue': 'CLIPPING' if peak_db > -1.0 else 'PEAK_TOO_HIGH',
                        'severity': 'HIGH' if peak_db > -1.0 else 'MEDIUM',
                    })
                elif ratio > THRESHOLDS['max_transient_ratio']:
                    issues.append({
                        'stem': fname,
                        'category': category,
                        'time_sec': i * window_sec,
                        'window': i,
                        'peak_db': peak_db,
                        'rms_db': rms_db,
                        'ratio': ratio,
                        'issue': 'TRANSIENT_SPIKE',
                        'severity': 'HIGH' if ratio > 15 else 'MEDIUM',
                    })
        except Exception as e:
            pass
    
    return issues

def fix_spiking_stems(stems_dir, output_dir, issues, max_peak_db=-3.0):
    """Apply limiting to stems with transient spikes."""
    os.makedirs(output_dir, exist_ok=True)
    
    fixes = []
    
    for fname in sorted(os.listdir(stems_dir)):
        if not fname.endswith('.wav') or fname.startswith('pass'):
            continue
        
        path = os.path.join(stems_dir, fname)
        output_path = os.path.join(output_dir, fname)
        
        # Check if this stem has issues
        stem_issues = [i for i in issues if i['stem'] == fname]
        
        if stem_issues:
            try:
                data, sr = sf.read(path)
                
                # Find the max peak
                max_peak = np.max(np.abs(data))
                max_peak_db = 20 * np.log10(max_peak + 1e-10)
                
                if max_peak_db > THRESHOLDS['max_peak_dbfs']:
                    # Calculate how much to reduce
                    target_peak = 10 ** (max_peak_db / 20)
                    gain = target_peak / max_peak
                    
                    # Apply gain reduction
                    corrected = data * gain
                    
                    # Soft clip to prevent hard clipping
                    corrected = np.tanh(corrected * 0.9) / 0.9
                    
                    sf.write(output_path, corrected, sr, subtype='PCM_24')
                    
                    new_peak = 20 * np.log10(np.max(np.abs(corrected)) + 1e-10)
                    
                    fixes.append({
                        'stem': fname,
                        'original_peak': max_peak_db,
                        'new_peak': new_peak,
                        'gain_db': 20 * np.log10(gain),
                        'num_issues': len(stem_issues),
                    })
                else:
                    # Copy unchanged
                    sf.write(output_path, data, sr, subtype='PCM_24')
            except:
                pass
        else:
            # Copy unchanged
            try:
                data, sr = sf.read(path)
                sf.write(output_path, data, sr, subtype='PCM_24')
            except:
                pass
    
    return fixes

def run_full_qc(stems_dir, output_dir=None, max_peak_db=-3.0):
    """Run full transient-aware QC analysis."""
    if output_dir is None:
        output_dir = stems_dir.rstrip('/') + '_qc'
    
    print("=" * 60)
    print("STEM QC v2 — Transient-Aware Analysis")
    print("=" * 60)
    
    # Step 1: Analyze transients
    print("\n1. Analyzing transients...")
    for fname in sorted(os.listdir(stems_dir)):
        if not fname.endswith('.wav') or fname.startswith('pass'):
            continue
        path = os.path.join(stems_dir, fname)
        try:
            result = analyze_stem_transients(path)
            category = classify_stem(fname)
            
            status = ''
            if result['num_clipping'] > 0:
                status = f' ← CLIPPING ({result["num_clipping"]} moments)'
            elif result['num_spikes'] > 0:
                status = f' ← SPIKES ({result["num_spikes"]})'
            elif result['max_transient_db'] > -3.0:
                status = f' ← HOT'
            
            print(f"   {fname[:35]:35s} peak={result['overall_peak']:6.1f}dB "
                  f"max_t={result['max_transient_db']:6.1f}dB "
                  f"ratio={result['transient_ratio']:5.1f}x{status}")
        except:
            pass
    
    # Step 2: Find spiking stems
    print("\n2. Finding spiking stems in 2-second windows...")
    issues = find_spiking_stems(stems_dir)
    
    if issues:
        print(f"\n   {'STEM':<35} {'TIME':>6} {'PEAK':>8} {'RMS':>8} {'RATIO':>8} {'ISSUE':<15}")
        print(f"   {'-'*35} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*15}")
        for issue in sorted(issues, key=lambda x: x['peak_db'], reverse=True)[:20]:
            stem = issue['stem'][:34]
            time = f"{issue['time_sec']:.0f}s"
            peak = f"{issue['peak_db']:.1f}"
            rms = f"{issue['rms_db']:.1f}"
            ratio = f"{issue['ratio']:.1f}x"
            print(f"   {stem:<35} {time:>6} {peak:>8} {rms:>8} {ratio:>8} {issue['issue']:<15}")
    else:
        print("   No issues found")
    
    # Step 3: Fix spiking stems
    if issues:
        print(f"\n3. Fixing {len(set(i['stem'] for i in issues))} stems with spikes...")
        fixes = fix_spiking_stems(stems_dir, output_dir, issues, max_peak_db)
        
        if fixes:
            print(f"\n   {'STEM':<35} {'BEFORE':>8} {'AFTER':>8} {'GAIN':>8}")
            print(f"   {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
            for fix in fixes:
                stem = fix['stem'][:34]
                before = f"{fix['original_peak']:.1f}"
                after = f"{fix['new_peak']:.1f}"
                gain = f"{fix['gain_db']:+.1f}"
                print(f"   {stem:<35} {before:>8} {after:>8} {gain:>8}")
    else:
        fixes = []
        # Copy all stems to output dir
        os.makedirs(output_dir, exist_ok=True)
        for fname in os.listdir(stems_dir):
            if fname.endswith('.wav') and not fname.startswith('pass'):
                src = os.path.join(stems_dir, fname)
                dst = os.path.join(output_dir, fname)
                data, sr = sf.read(src)
                sf.write(dst, data, sr, subtype='PCM_24')
    
    # Summary
    print(f"\n{'='*60}")
    print(f"QC SUMMARY")
    print(f"{'='*60}")
    print(f"  Stems analyzed:   {len([f for f in os.listdir(stems_dir) if f.endswith('.wav') and not f.startswith('pass')])}")
    print(f"  Issues found:     {len(issues)}")
    print(f"  Stems fixed:      {len(fixes)}")
    print(f"  Output:           {output_dir}")
    
    return output_dir, issues, fixes

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python stem_qc_v2.py <stems_dir> [output_dir]")
        sys.exit(1)
    
    stems_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    run_full_qc(stems_dir, output_dir)
