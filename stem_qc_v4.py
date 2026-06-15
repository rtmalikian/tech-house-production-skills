"""
STEM QC v4 — Transient-aware balance for drums, RMS for melodic stems.
Fixes the fundamental flaw: RMS is meaningless for drums (mostly silence between hits).
"""
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks
import os
import sys

# ============================================================================
# BALANCE TARGETS
# For drums: target is PEAK level of loudest transient
# For melodic: target is RMS level
# ============================================================================
DRUM_PEAK_TARGETS = {
    'kick':         -6.0,   # Kick peak should be -6 dBFS
    'snare':        -8.0,   # Snare peak
    'clap':         -8.0,   # Clap peak
    'closed_hat':  -12.0,   # Closed hat peak
    'open_hat':    -10.0,   # Open hat peak (slightly louder than closed)
    'crash':        -8.0,   # Crash peak
    'ride':        -14.0,   # Ride peak
    'tambourine':  -14.0,   # Tambourine peak
    'shaker':      -16.0,   # Shaker peak
    'sidestick':   -12.0,   # Sidestick peak
    'cowbell':     -14.0,   # Cowbell peak
}

MELODIC_RMS_TARGETS = {
    'bass':        -14.0,   # Bass RMS
    'sub_bass':    -16.0,   # Sub bass RMS
    'stab':        -14.0,   # Stab RMS
    'acid':        -14.0,   # Acid RMS
    'pad':         -16.0,   # Pad RMS
    'fx':          -18.0,   # FX RMS
}

DRUM_CATEGORIES = {'kick', 'snare', 'clap', 'closed_hat', 'open_hat', 'crash', 'ride', 'tambourine', 'shaker', 'sidestick', 'cowbell'}

CATEGORIES = {
    'kick': ['kick_n36'], 'bass': ['bass'], 'sub_bass': ['sub_bass'],
    'clap': ['clap_n39'], 'snare': ['snare_n38'],
    'closed_hat': ['closedhat_n42'], 'open_hat': ['openhat_n46'],
    'ride': ['ride_n51'], 'crash': ['crash_n49'],
    'tambourine': ['tambourine_n54'], 'shaker': ['shaker', 'maracas'],
    'sidestick': ['sidestick_n37'], 'cowbell': ['cowbell_n56'],
    'stab': ['chord_stab'], 'acid': ['acid_line'],
    'pad': ['pad'], 'fx': ['fx'],
}

def classify(fname):
    for cat, patterns in CATEGORIES.items():
        for p in patterns:
            if p in fname.lower():
                return cat
    return 'other'

def measure_drum_transients(path):
    """Measure drum stem using peak transient analysis, not RMS."""
    data, sr = sf.read(path)
    mono = data.mean(axis=1) if data.ndim > 1 else data
    
    # Find all transients using onset detection
    # Use 5ms window for envelope
    window = int(sr * 0.005)
    envelope = np.convolve(np.abs(mono), np.ones(window)/window, mode='same')
    
    # Find peaks (transients) with minimum distance of 50ms
    min_dist = int(sr * 0.05)
    peaks, props = find_peaks(envelope, height=np.max(envelope) * 0.1, distance=min_dist)
    
    if len(peaks) == 0:
        return {
            'data': data, 'sr': sr, 'category': classify(path),
            'peak_level': -60.0, 'num_transients': 0,
            'transient_mean': -60.0, 'transient_std': 0.0,
            'transient_max': -60.0, 'transient_min': -60.0,
            'is_drum': True,
        }
    
    # Measure peak level of each transient
    peak_levels = []
    for peak in peaks:
        start = max(0, peak - 200)
        end = min(len(mono), peak + 200)
        peak_val = np.max(np.abs(mono[start:end]))
        peak_levels.append(20 * np.log10(peak_val + 1e-10))
    
    peak_levels = np.array(peak_levels)
    
    return {
        'data': data, 'sr': sr, 'category': classify(path),
        'peak_level': float(20 * np.log10(np.max(np.abs(mono)) + 1e-10)),
        'num_transients': len(peaks),
        'transient_mean': float(np.mean(peak_levels)),
        'transient_std': float(np.std(peak_levels)),
        'transient_max': float(np.max(peak_levels)),
        'transient_min': float(np.min(peak_levels)),
        'is_drum': True,
    }

def measure_melodic_rms(path):
    """Measure melodic stem using RMS (appropriate for sustained sounds)."""
    data, sr = sf.read(path)
    mono = data.mean(axis=1) if data.ndim > 1 else data
    
    # Active RMS (ignoring silence)
    window = int(sr * 0.5)
    rms_vals = []
    for i in range(0, len(mono) - window, window):
        chunk = mono[i:i+window]
        rms = np.sqrt(np.mean(chunk**2))
        if rms > 0.001:
            rms_vals.append(20 * np.log10(rms + 1e-10))
    
    return {
        'data': data, 'sr': sr, 'category': classify(path),
        'rms_level': float(np.mean(rms_vals)) if rms_vals else -60.0,
        'peak_level': float(20 * np.log10(np.max(np.abs(mono)) + 1e-10)),
        'is_drum': False,
    }

def fix_stems(stems_dir, output_dir):
    """Fix stem levels using appropriate measurement for each type."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("STEM QC v4 — Transient-Aware Drum Balance")
    print("=" * 60)
    
    # Step 1: Measure all stems
    print("\n1. Measuring stems (drums=transient, melodic=RMS)...")
    stem_info = {}
    
    for fname in sorted(os.listdir(stems_dir)):
        if not fname.endswith('.wav') or fname.startswith('pass'):
            continue
        path = os.path.join(stems_dir, fname)
        try:
            category = classify(fname)
            is_drum = category in DRUM_CATEGORIES
            
            if is_drum:
                info = measure_drum_transients(path)
                target = DRUM_PEAK_TARGETS.get(category, -10.0)
                current = info['transient_mean']
                print(f"   {fname:<35} cat={category:<12} TRANSIENT mean={current:6.1f} peak={info['transient_max']:6.1f} target={target:6.1f}")
            else:
                info = measure_melodic_rms(path)
                target = MELODIC_RMS_TARGETS.get(category, -14.0)
                current = info['rms_level']
                print(f"   {fname:<35} cat={category:<12} RMS={current:6.1f} target={target:6.1f}")
            
            info['target'] = target
            info['current'] = current
            stem_info[fname] = info
        except Exception as e:
            print(f"   {fname}: ERROR {e}")
    
    # Step 2: Calculate corrections
    print("\n2. Calculating corrections...")
    corrections = {}
    
    for fname, info in stem_info.items():
        deviation = info['current'] - info['target']
        
        # Only correct if deviation is significant (> 3 dB)
        if abs(deviation) > 3:
            correction_db = -deviation  # Bring to target
            correction_db = max(-20, min(20, correction_db))  # Limit ±20 dB
            corrections[fname] = correction_db
    
    # Step 3: Apply corrections
    print("\n3. Applying corrections...")
    fixes = []
    
    for fname, info in stem_info.items():
        path = os.path.join(stems_dir, fname)
        output_path = os.path.join(output_dir, fname)
        
        data = info['data']
        sr = info['sr']
        
        if fname in corrections:
            correction_db = corrections[fname]
            gain = 10 ** (correction_db / 20)
            data = data * gain
            fixes.append({
                'stem': fname,
                'category': info['category'],
                'is_drum': info['is_drum'],
                'current': info['current'],
                'target': info['target'],
                'correction_db': correction_db,
            })
        
        # Soft clip to prevent hard clipping
        peak = np.max(np.abs(data))
        if peak > 0.95:
            data = np.tanh(data * 0.9) / 0.9
        
        # Final clip at -1 dBFS
        max_val = 10 ** (-1.0 / 20)
        data = np.clip(data, -max_val, max_val)
        
        sf.write(output_path, data, sr, subtype='PCM_24')
    
    # Step 4: Verify
    print("\n4. Verifying corrections...")
    for fname in sorted(os.listdir(output_dir)):
        if not fname.endswith('.wav') or fname.startswith('pass'):
            continue
        path = os.path.join(output_dir, fname)
        try:
            category = classify(fname)
            is_drum = category in DRUM_CATEGORIES
            
            if is_drum:
                info = measure_drum_transients(path)
                current = info['transient_mean']
                target = DRUM_PEAK_TARGETS.get(category, -10.0)
                deviation = current - target
                status = '✓' if abs(deviation) < 4 else '⚠'
                print(f"   {fname:<35} transient={current:6.1f} target={target:6.1f} dev={deviation:+6.1f} {status}")
            else:
                info = measure_melodic_rms(path)
                current = info['rms_level']
                target = MELODIC_RMS_TARGETS.get(category, -14.0)
                deviation = current - target
                status = '✓' if abs(deviation) < 4 else '⚠'
                print(f"   {fname:<35} rms={current:6.1f} target={target:6.1f} dev={deviation:+6.1f} {status}")
        except:
            pass
    
    # Summary
    print(f"\n{'='*60}")
    print(f"QC SUMMARY")
    print(f"{'='*60}")
    print(f"  Stems analyzed: {len(stem_info)}")
    print(f"  Corrections:    {len(fixes)}")
    print(f"  Output:         {output_dir}")
    
    if fixes:
        print(f"\n  FIXES APPLIED:")
        for fix in sorted(fixes, key=lambda x: abs(x['correction_db']), reverse=True):
            stem = fix['stem'][:30]
            cat = fix['category']
            method = 'TRANSIENT' if fix['is_drum'] else 'RMS'
            orig = fix['current']
            corr = fix['correction_db']
            print(f"    {stem:<30} {cat:<12} {method:<10} {orig:>6.1f} → {corr:>+6.1f} dB")
    
    return output_dir, fixes

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python stem_qc_v4.py <stems_dir> [output_dir]")
        sys.exit(1)
    
    stems_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else stems_dir.rstrip('/') + '_balanced'
    fix_stems(stems_dir, output_dir)
