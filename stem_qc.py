"""
STEM QC SYSTEM — Automatic level balance checking and correction.
Analyzes each stem, detects dominant elements, and adjusts levels before mixing.

Usage:
    python stem_qc.py <stems_dir> [--fix] [--report]
    
    --fix:    Automatically adjust stem levels to target balance
    --report: Generate detailed QC report
"""
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import os
import json
import sys
from scipy.signal import butter, sosfiltfilt

# ============================================================================
# QC THRESHOLDS (from tech house reference analysis)
# ============================================================================
QC_THRESHOLDS = {
    # Maximum RMS difference between any two stems (dB)
    'max_stem_rms_spread': 12.0,
    
    # Target RMS levels for each stem category (relative to mix)
    'target_rms': {
        'kick':      -12.0,  # Kick should be loudest
        'bass':      -14.0,  # Bass close to kick
        'clap':      -16.0,  # Clap on beats 2&4
        'snare':     -16.0,  # Snare layer
        'closed_hat': -20.0, # Hats should be quieter
        'open_hat':  -18.0,  # Open hat slightly louder than closed
        'ride':      -22.0,  # Ride subtle
        'crash':     -20.0,  # Crash occasional
        'tambourine': -22.0, # Tambourine subtle
        'shaker':    -24.0,  # Shaker very subtle
        'stab':      -16.0,  # Stabs prominent
        'acid':      -18.0,  # Acid line
        'pad':       -20.0,  # Pad atmospheric
        'fx':        -22.0,  # FX subtle
        'sub_bass':  -16.0,  # Sub bass
    },
    
    # Maximum deviation from target (dB) before correction
    'max_deviation': 6.0,
    
    # Per-bar analysis window (seconds)
    'analysis_window': 0.5,
    
    # Minimum active level (dBFS) to consider a stem "playing"
    'active_threshold': -40.0,
}

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
    'ride': ['ride_n51'],
    'other': ['instr_n50'],
}

def classify_stem(filename):
    """Classify a stem file into a category."""
    fname_lower = filename.lower()
    for category, patterns in STEM_CATEGORIES.items():
        for pattern in patterns:
            if pattern in fname_lower:
                return category
    return 'other'

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================
def analyze_stem_levels(stems_dir, window_sec=0.5):
    """Analyze RMS levels of each stem over time."""
    results = {}
    
    for fname in sorted(os.listdir(stems_dir)):
        if not fname.endswith('.wav') or fname.startswith('pass'):
            continue
        
        path = os.path.join(stems_dir, fname)
        try:
            data, sr = sf.read(path)
            mono = data.mean(axis=1) if data.ndim > 1 else data
            
            # Calculate RMS in windows
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
            
            # Overall stats
            overall_rms = 20 * np.log10(np.sqrt(np.mean(mono**2)) + 1e-10)
            peak = 20 * np.log10(np.max(np.abs(mono)) + 1e-10)
            
            # Detect active sections (above threshold)
            active_mask = rms_array > QC_THRESHOLDS['active_threshold']
            active_rms = rms_array[active_mask] if np.any(active_mask) else rms_array
            
            category = classify_stem(fname)
            
            results[fname] = {
                'category': category,
                'overall_rms': float(overall_rms),
                'peak': float(peak),
                'active_rms_mean': float(np.mean(active_rms)),
                'active_rms_std': float(np.std(active_rms)),
                'active_rms_max': float(np.max(active_rms)),
                'active_rms_min': float(np.min(active_rms)),
                'rms_windows': rms_array.tolist(),
                'duration': len(mono) / sr,
            }
        except Exception as e:
            results[fname] = {'error': str(e)}
    
    return results

def detect_dominant_stems(stem_analysis):
    """Detect stems that are too loud relative to others."""
    issues = []
    
    # Get active RMS for each stem
    active_rms = {}
    for fname, data in stem_analysis.items():
        if 'error' in data:
            continue
        active_rms[fname] = data['active_rms_mean']
    
    if not active_rms:
        return issues
    
    # Calculate the median RMS across all stems
    all_rms = list(active_rms.values())
    median_rms = np.median(all_rms)
    
    # Check each stem against the median
    for fname, rms in active_rms.items():
        deviation = rms - median_rms
        category = stem_analysis[fname]['category']
        target = QC_THRESHOLDS['target_rms'].get(category, -18.0)
        target_deviation = rms - target
        
        if deviation > QC_THRESHOLDS['max_deviation']:
            issues.append({
                'stem': fname,
                'category': category,
                'issue': 'TOO_LOUD',
                'rms': rms,
                'median': median_rms,
                'deviation': deviation,
                'target': target,
                'target_deviation': target_deviation,
                'severity': 'HIGH' if deviation > 10 else 'MEDIUM',
            })
        elif target_deviation > QC_THRESHOLDS['max_deviation']:
            issues.append({
                'stem': fname,
                'category': category,
                'issue': 'ABOVE_TARGET',
                'rms': rms,
                'target': target,
                'target_deviation': target_deviation,
                'severity': 'MEDIUM',
            })
    
    return issues

def detect_per_bar_issues(stem_analysis, window_sec=0.5, bpm=128):
    """Detect stems that are too loud in specific sections."""
    issues = []
    bar_duration = 4 * 60.0 / bpm  # Duration of one bar in seconds
    windows_per_bar = int(bar_duration / window_sec)
    
    for fname, data in stem_analysis.items():
        if 'error' in data or 'rms_windows' not in data:
            continue
        
        rms_windows = np.array(data['rms_windows'])
        category = data['category']
        target = QC_THRESHOLDS['target_rms'].get(category, -18.0)
        
        # Analyze each bar
        n_bars = len(rms_windows) // windows_per_bar
        for bar in range(n_bars):
            start = bar * windows_per_bar
            end = start + windows_per_bar
            bar_rms = rms_windows[start:end]
            bar_mean = np.mean(bar_rms)
            
            if bar_mean > target + QC_THRESHOLDS['max_deviation']:
                issues.append({
                    'stem': fname,
                    'category': category,
                    'bar': bar,
                    'time_sec': bar * bar_duration,
                    'rms': bar_mean,
                    'target': target,
                    'deviation': bar_mean - target,
                    'issue': 'BAR_TOO_LOUD',
                })
    
    return issues

def generate_fix(stem_analysis, issues, stems_dir, output_dir):
    """Generate corrected stems with adjusted levels."""
    os.makedirs(output_dir, exist_ok=True)
    
    fixes = []
    
    for issue in issues:
        fname = issue['stem']
        path = os.path.join(stems_dir, fname)
        
        if not os.path.exists(path):
            continue
        
        try:
            data, sr = sf.read(path)
            
            # Calculate correction gain
            target = issue.get('target', -18.0)
            current = issue['rms']
            correction_db = target - current
            
            # Limit correction to ±12 dB
            correction_db = max(-12.0, min(12.0, correction_db))
            
            # Apply gain
            gain = 10 ** (correction_db / 20)
            corrected = data * gain
            
            # Clip to prevent clipping
            corrected = np.clip(corrected, -1.0, 1.0)
            
            # Save corrected stem
            output_path = os.path.join(output_dir, fname)
            sf.write(output_path, corrected, sr, subtype='PCM_24')
            
            fixes.append({
                'stem': fname,
                'original_rms': current,
                'target_rms': target,
                'correction_db': correction_db,
                'output_path': output_path,
            })
        except Exception as e:
            fixes.append({
                'stem': fname,
                'error': str(e),
            })
    
    return fixes

def generate_report(stem_analysis, issues, fixes=None):
    """Generate a detailed QC report."""
    report = {
        'summary': {
            'total_stems': len(stem_analysis),
            'issues_found': len(issues),
            'high_severity': len([i for i in issues if i.get('severity') == 'HIGH']),
            'medium_severity': len([i for i in issues if i.get('severity') == 'MEDIUM']),
        },
        'stems': {},
        'issues': issues,
        'fixes': fixes or [],
    }
    
    for fname, data in stem_analysis.items():
        if 'error' in data:
            report['stems'][fname] = {'error': data['error']}
            continue
        
        report['stems'][fname] = {
            'category': data['category'],
            'overall_rms': data['overall_rms'],
            'peak': data['peak'],
            'active_rms_mean': data['active_rms_mean'],
            'active_rms_std': data['active_rms_std'],
        }
    
    return report

# ============================================================================
# MAIN
# ============================================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python stem_qc.py <stems_dir> [--fix] [--report]")
        sys.exit(1)
    
    stems_dir = sys.argv[1]
    do_fix = '--fix' in sys.argv
    do_report = '--report' in sys.argv
    
    if not os.path.isdir(stems_dir):
        print(f"Error: {stems_dir} is not a directory")
        sys.exit(1)
    
    print("=" * 60)
    print("STEM QC SYSTEM")
    print("=" * 60)
    
    # Analyze stems
    print(f"\n1. Analyzing stems in {stems_dir}...")
    stem_analysis = analyze_stem_levels(stems_dir)
    print(f"   Found {len(stem_analysis)} stems")
    
    # Detect dominant stems
    print(f"\n2. Detecting dominant stems...")
    issues = detect_dominant_stems(stem_analysis)
    
    # Detect per-bar issues
    print(f"\n3. Analyzing per-bar balance...")
    bar_issues = detect_per_bar_issues(stem_analysis)
    issues.extend(bar_issues)
    
    # Print issues
    print(f"\n4. QC Issues Found: {len(issues)}")
    if issues:
        print(f"\n   {'STEM':<35} {'CATEGORY':<12} {'ISSUE':<15} {'RMS':>8} {'TARGET':>8} {'DEV':>8}")
        print(f"   {'-'*35} {'-'*12} {'-'*15} {'-'*8} {'-'*8} {'-'*8}")
        for issue in issues[:20]:  # Show first 20
            stem = issue['stem'][:34]
            cat = issue.get('category', '?')
            iss = issue.get('issue', '?')
            rms = issue.get('rms', 0)
            target = issue.get('target', 0)
            dev = issue.get('deviation', issue.get('target_deviation', 0))
            print(f"   {stem:<35} {cat:<12} {iss:<15} {rms:>7.1f} {target:>7.1f} {dev:>+7.1f}")
    
    # Generate fixes
    if do_fix and issues:
        print(f"\n5. Generating corrected stems...")
        output_dir = stems_dir.rstrip('/') + '_qc_fixed'
        fixes = generate_fix(stem_analysis, issues, stems_dir, output_dir)
        print(f"   Saved {len(fixes)} corrected stems to {output_dir}")
        
        # Re-analyze to verify
        print(f"\n6. Verifying corrections...")
        fixed_analysis = analyze_stem_levels(output_dir)
        fixed_issues = detect_dominant_stems(fixed_analysis)
        print(f"   Issues before: {len(issues)}")
        print(f"   Issues after:  {len(fixed_issues)}")
    else:
        fixes = []
    
    # Generate report
    if do_report:
        report = generate_report(stem_analysis, issues, fixes)
        report_path = os.path.join(stems_dir, 'qc_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n7. Report saved: {report_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"QC SUMMARY")
    print(f"{'='*60}")
    print(f"  Stems analyzed: {len(stem_analysis)}")
    print(f"  Issues found:   {len(issues)}")
    high = len([i for i in issues if i.get('severity') == 'HIGH'])
    medium = len([i for i in issues if i.get('severity') == 'MEDIUM'])
    print(f"  High severity:  {high}")
    print(f"  Medium severity: {medium}")
    
    if issues:
        print(f"\n  TOP ISSUES:")
        for issue in sorted(issues, key=lambda x: abs(x.get('deviation', x.get('target_deviation', 0))), reverse=True)[:5]:
            stem = issue['stem'][:30]
            dev = issue.get('deviation', issue.get('target_deviation', 0))
            print(f"    {stem}: {dev:+.1f} dB {'(TOO LOUD)' if dev > 0 else '(too quiet)'}")
    
    return len(issues)

if __name__ == '__main__':
    sys.exit(main())
