"""
STEM QC MIXDOWN — Integrates QC analysis into the pipeline.
Runs after recording, before mastering. Adjusts stem levels for balanced mix.
"""
import numpy as np
import soundfile as sf
import os
import json
import sys

# ============================================================================
# TECH HOUSE STEM BALANCE TARGETS (relative to kick at 0 dB)
# Based on reference track analysis
# ============================================================================
STEM_TARGETS = {
    # (target_db_relative_to_kick, priority)
    # Priority: 1=must be correct, 2=important, 3=nice to have
    'kick':         (0.0, 1),     # Kick is reference
    'bass':         (-2.0, 1),    # Bass just below kick
    'sub_bass':     (-4.0, 1),    # Sub below bass
    'clap':         (-4.0, 1),    # Clap prominent
    'snare':        (-4.0, 2),    # Snare layer
    'closed_hat':   (-8.0, 2),    # Hats quieter than kick
    'open_hat':     (-6.0, 2),    # Open hat slightly louder than closed
    'crash':        (-6.0, 3),    # Crash occasional
    'ride':         (-10.0, 3),   # Ride subtle
    'tambourine':   (-10.0, 3),   # Tambourine subtle
    'shaker':       (-12.0, 3),   # Shaker very subtle
    'sidestick':    (-8.0, 3),    # Sidestick
    'cowbell':      (-10.0, 3),   # Cowbell
    'tom':          (-8.0, 3),    # Toms
    'stab':         (-6.0, 1),    # Stabs prominent
    'acid':         (-6.0, 1),    # Acid line
    'pad':          (-8.0, 2),    # Pad atmospheric
    'fx':           (-10.0, 3),   # FX subtle
}

# Categories that should NEVER be louder than the kick
CATEGORIES_BELOW_KICK = [
    'closed_hat', 'open_hat', 'crash', 'ride', 'tambourine', 
    'shaker', 'sidestick', 'cowbell', 'tom', 'pad', 'fx'
]

# Categories that can be as loud as the kick
CATEGORIES_AT_KICK_LEVEL = ['bass', 'sub_bass', 'clap', 'snare', 'stab', 'acid']

# ============================================================================
# STEM CLASSIFICATION
# ============================================================================
STEM_CATEGORIES = {
    'kick': ['kick_n36'],
    'bass': ['bass'],
    'sub_bass': ['sub_bass'],
    'clap': ['clap_n39'],
    'snare': ['snare_n38'],
    'closed_hat': ['closedhat_n42'],
    'open_hat': ['openhat_n46'],
    'ride': ['ride_n51'],
    'crash': ['crash_n49'],
    'tambourine': ['tambourine_n54'],
    'shaker': ['shaker', 'maracas'],
    'sidestick': ['sidestick_n37'],
    'cowbell': ['cowbell_n56'],
    'stab': ['chord_stab'],
    'acid': ['acid_line'],
    'pad': ['pad'],
    'fx': ['fx'],
    'tom': ['lowtom', 'midtom', 'hightom'],
    'rimshot': ['rimshot'],
    'other': ['instr_n50'],
}

def classify_stem(filename):
    fname_lower = filename.lower()
    for category, patterns in STEM_CATEGORIES.items():
        for pattern in patterns:
            if pattern in fname_lower:
                return category
    return 'other'

# ============================================================================
# ANALYSIS
# ============================================================================
def measure_stem_rms(path, window_sec=0.5):
    """Measure RMS of a stem in windows."""
    data, sr = sf.read(path)
    mono = data.mean(axis=1) if data.ndim > 1 else data
    
    window_samples = int(window_sec * sr)
    n_windows = len(mono) // window_samples
    
    rms_windows = []
    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples
        chunk = mono[start:end]
        rms = np.sqrt(np.mean(chunk**2))
        rms_windows.append(20 * np.log10(rms + 1e-10))
    
    rms_array = np.array(rms_windows)
    
    # Active sections only (above -40 dBFS)
    active_mask = rms_array > -40.0
    active_rms = rms_array[active_mask] if np.any(active_mask) else rms_array
    
    return {
        'overall_rms': float(20 * np.log10(np.sqrt(np.mean(mono**2)) + 1e-10)),
        'peak': float(20 * np.log10(np.max(np.abs(mono)) + 1e-10)),
        'active_rms_mean': float(np.mean(active_rms)),
        'active_rms_std': float(np.std(active_rms)),
        'active_rms_max': float(np.max(active_rms)),
        'active_rms_min': float(np.min(active_rms)),
        'rms_windows': rms_windows,
        'sr': sr,
        'data': data,
    }

def analyze_and_fix_stems(stems_dir, output_dir=None):
    """Analyze all stems and generate corrected versions."""
    if output_dir is None:
        output_dir = stems_dir.rstrip('/') + '_balanced'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("STEM QC MIXDOWN — Balance Correction")
    print("=" * 60)
    
    # Step 1: Measure all stems
    print("\n1. Measuring stem levels...")
    stem_data = {}
    for fname in sorted(os.listdir(stems_dir)):
        if not fname.endswith('.wav') or fname.startswith('pass'):
            continue
        path = os.path.join(stems_dir, fname)
        try:
            info = measure_stem_rms(path)
            info['category'] = classify_stem(fname)
            stem_data[fname] = info
            print(f"   {fname:<35} cat={info['category']:<12} rms={info['active_rms_mean']:>6.1f} dBFS")
        except Exception as e:
            print(f"   {fname}: ERROR - {e}")
    
    # Step 2: Find kick reference level
    print("\n2. Finding kick reference level...")
    kick_rms = None
    for fname, data in stem_data.items():
        if data['category'] == 'kick':
            if kick_rms is None or data['active_rms_mean'] > kick_rms:
                kick_rms = data['active_rms_mean']
                kick_file = fname
    
    if kick_rms is None:
        print("   WARNING: No kick found, using -12 dBFS as reference")
        kick_rms = -12.0
    else:
        print(f"   Kick reference: {kick_file} at {kick_rms:.1f} dBFS")
    
    # Step 3: Calculate corrections
    print("\n3. Calculating corrections...")
    corrections = {}
    issues = []
    
    for fname, data in stem_data.items():
        category = data['category']
        target_rel, priority = STEM_TARGETS.get(category, (-8.0, 3))
        target_abs = kick_rms + target_rel
        
        current_rms = data['active_rms_mean']
        deviation = current_rms - target_abs
        
        correction_db = 0.0
        
        # Check if stem is too loud relative to target
        if deviation > 6.0:
            correction_db = -deviation + 2  # Bring to target + 2 dB
            issues.append({
                'stem': fname,
                'category': category,
                'issue': 'TOO_LOUD',
                'current': current_rms,
                'target': target_abs,
                'deviation': deviation,
                'correction': correction_db,
                'priority': priority,
            })
        
        # Check if stem is louder than kick (for categories that shouldn't be)
        if category in CATEGORIES_BELOW_KICK and current_rms > kick_rms + 3:
            correction_db = min(correction_db, kick_rms - current_rms - 3)
            issues.append({
                'stem': fname,
                'category': category,
                'issue': 'LOUDER_THAN_KICK',
                'current': current_rms,
                'kick': kick_rms,
                'correction': correction_db,
                'priority': 1,
            })
        
        if correction_db != 0:
            corrections[fname] = {
                'correction_db': correction_db,
                'current_rms': current_rms,
                'target_rms': target_abs,
            }
    
    # Step 4: Apply corrections
    print("\n4. Applying corrections...")
    corrected_count = 0
    
    for fname, data in stem_data.items():
        path = os.path.join(stems_dir, fname)
        output_path = os.path.join(output_dir, fname)
        
        if fname in corrections:
            correction = corrections[fname]
            correction_db = correction['correction_db']
            gain = 10 ** (correction_db / 20)
            
            corrected = data['data'] * gain
            corrected = np.clip(corrected, -1.0, 1.0)
            
            sf.write(output_path, corrected, data['sr'], subtype='PCM_24')
            corrected_count += 1
            
            print(f"   {fname:<35} {correction_db:>+6.1f} dB  (was {correction['current_rms']:.1f}, target {correction['target_rms']:.1f})")
        else:
            # Copy unchanged
            sf.write(output_path, data['data'], data['sr'], subtype='PCM_24')
    
    # Step 5: Verify
    print("\n5. Verifying corrections...")
    verification_issues = 0
    for fname in sorted(os.listdir(output_dir)):
        if not fname.endswith('.wav') or fname.startswith('pass'):
            continue
        path = os.path.join(output_dir, fname)
        try:
            info = measure_stem_rms(path)
            category = classify_stem(fname)
            target_rel, _ = STEM_TARGETS.get(category, (-8.0, 3))
            target_abs = kick_rms + target_rel
            deviation = info['active_rms_mean'] - target_abs
            
            status = '✓' if abs(deviation) < 8 else '⚠'
            if abs(deviation) >= 8:
                verification_issues += 1
            
            print(f"   {fname:<35} rms={info['active_rms_mean']:>6.1f} target={target_abs:>6.1f} dev={deviation:>+6.1f} {status}")
        except:
            pass
    
    # Summary
    print(f"\n{'='*60}")
    print(f"QC SUMMARY")
    print(f"{'='*60}")
    print(f"  Stems analyzed:  {len(stem_data)}")
    print(f"  Issues found:    {len(issues)}")
    print(f"  Corrections:     {corrected_count}")
    print(f"  Remaining issues:{verification_issues}")
    print(f"  Output:          {output_dir}")
    
    # Print top issues
    if issues:
        print(f"\n  TOP ISSUES FIXED:")
        for issue in sorted(issues, key=lambda x: abs(x.get('deviation', x.get('correction', 0))), reverse=True)[:5]:
            stem = issue['stem'][:30]
            if issue['issue'] == 'TOO_LOUD':
                print(f"    {stem}: {issue['deviation']:+.1f} dB too loud → corrected")
            elif issue['issue'] == 'LOUDER_THAN_KICK':
                print(f"    {stem}: louder than kick → corrected")
    
    return output_dir, corrections, issues

# ============================================================================
# CLI
# ============================================================================
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python stem_qc_mixdown.py <stems_dir> [output_dir]")
        sys.exit(1)
    
    stems_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    analyze_and_fix_stems(stems_dir, output_dir)
