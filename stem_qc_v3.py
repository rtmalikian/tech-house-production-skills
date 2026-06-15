"""
STEM QC v3 — Full balance correction + transient limiting.
Fixes both relative levels AND transient spikes.
"""
import numpy as np
import soundfile as sf
import os
import sys

# ============================================================================
# STEM BALANCE TARGETS (relative to kick at 0 dB)
# ============================================================================
TARGETS = {
    'kick':         0.0,
    'bass':        -2.0,
    'sub_bass':    -4.0,
    'clap':        -4.0,
    'snare':       -4.0,
    'closed_hat':  -8.0,
    'open_hat':    -6.0,
    'crash':       -6.0,
    'ride':       -10.0,
    'tambourine': -10.0,
    'shaker':     -12.0,
    'sidestick':   -8.0,
    'cowbell':    -10.0,
    'stab':        -6.0,
    'acid':        -6.0,
    'pad':         -8.0,
    'fx':         -10.0,
    'tom':         -8.0,
}

CATEGORIES = {
    'kick': ['kick_n36'], 'bass': ['bass'], 'sub_bass': ['sub_bass'],
    'clap': ['clap_n39'], 'snare': ['snare_n38'],
    'closed_hat': ['closedhat_n42'], 'open_hat': ['openhat_n46'],
    'ride': ['ride_n51'], 'crash': ['crash_n49'],
    'tambourine': ['tambourine_n54'], 'shaker': ['shaker', 'maracas'],
    'sidestick': ['sidestick_n37'], 'cowbell': ['cowbell_n56'],
    'stab': ['chord_stab'], 'acid': ['acid_line'],
    'pad': ['pad'], 'fx': ['fx'],
    'tom': ['lowtom', 'midtom', 'hightom'],
}

def classify(fname):
    for cat, patterns in CATEGORIES.items():
        for p in patterns:
            if p in fname.lower():
                return cat
    return 'other'

def measure_active_rms(path, window_sec=0.5):
    data, sr = sf.read(path)
    mono = data.mean(axis=1) if data.ndim > 1 else data
    window = int(window_sec * sr)
    rms_vals = []
    for i in range(0, len(mono) - window, window):
        chunk = mono[i:i+window]
        rms = np.sqrt(np.mean(chunk**2))
        if rms > 0.001:
            rms_vals.append(20 * np.log10(rms + 1e-10))
    return {
        'data': data,
        'sr': sr,
        'active_rms': float(np.mean(rms_vals)) if rms_vals else -60,
        'peak': float(20 * np.log10(np.max(np.abs(data)) + 1e-10)),
        'category': classify(path),
    }

def fix_stems(stems_dir, output_dir):
    """Fix stem levels and transients."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("STEM QC v3 — Full Balance + Transient Fix")
    print("=" * 60)
    
    # Step 1: Measure all stems
    print("\n1. Measuring stems...")
    stem_info = {}
    for fname in sorted(os.listdir(stems_dir)):
        if not fname.endswith('.wav') or fname.startswith('pass'):
            continue
        path = os.path.join(stems_dir, fname)
        try:
            info = measure_active_rms(path)
            stem_info[fname] = info
            print(f"   {fname:<35} cat={info['category']:<12} rms={info['active_rms']:>6.1f} peak={info['peak']:>6.1f}")
        except Exception as e:
            print(f"   {fname}: ERROR {e}")
    
    # Step 2: Find kick reference
    print("\n2. Finding kick reference...")
    kick_rms = -20
    for fname, info in stem_info.items():
        if info['category'] == 'kick' and info['active_rms'] > kick_rms:
            kick_rms = info['active_rms']
    print(f"   Kick reference: {kick_rms:.1f} dBFS")
    
    # Step 3: Calculate corrections
    print("\n3. Calculating corrections...")
    corrections = {}
    
    for fname, info in stem_info.items():
        category = info['category']
        target_rel = TARGETS.get(category, -8.0)
        target_abs = kick_rms + target_rel
        
        current = info['active_rms']
        deviation = current - target_abs
        
        # Only correct if deviation is significant (> 3 dB)
        if abs(deviation) > 3:
            correction_db = -deviation  # Bring to target
            # Limit correction to ±20 dB
            correction_db = max(-20, min(20, correction_db))
            corrections[fname] = correction_db
    
    # Step 4: Apply corrections + soft clipping
    print("\n4. Applying corrections...")
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
                'original_rms': info['active_rms'],
                'correction_db': correction_db,
            })
        
        # Soft clip to prevent hard clipping
        peak = np.max(np.abs(data))
        if peak > 0.95:  # If close to clipping
            data = np.tanh(data * 0.9) / 0.9
        
        # Final clip at -1 dBFS
        max_val = 10 ** (-1.0 / 20)
        data = np.clip(data, -max_val, max_val)
        
        sf.write(output_path, data, sr, subtype='PCM_24')
    
    # Step 5: Verify
    print("\n5. Verifying corrections...")
    for fname in sorted(os.listdir(output_dir)):
        if not fname.endswith('.wav') or fname.startswith('pass'):
            continue
        path = os.path.join(output_dir, fname)
        try:
            info = measure_active_rms(path)
            category = info['category']
            target_rel = TARGETS.get(category, -8.0)
            target_abs = kick_rms + target_rel
            deviation = info['active_rms'] - target_abs
            status = '✓' if abs(deviation) < 5 else '⚠'
            print(f"   {fname:<35} rms={info['active_rms']:>6.1f} target={target_abs:>6.1f} dev={deviation:>+6.1f} {status}")
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
            orig = fix['original_rms']
            corr = fix['correction_db']
            print(f"    {stem:<30} {cat:<12} {orig:>6.1f} dB → {corr:>+6.1f} dB correction")
    
    return output_dir, fixes

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python stem_qc_v3.py <stems_dir> [output_dir]")
        sys.exit(1)
    
    stems_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else stems_dir.rstrip('/') + '_balanced'
    fix_stems(stems_dir, output_dir)
