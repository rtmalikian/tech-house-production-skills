#!/usr/bin/env python3
"""
Compatibility CLI entry point for the May25 audio pipeline.

Usage:
    python musicgen_compiled/run_pipeline.py \
        --stems output/<song>/recordings/ \
        --song-name <song> \
        --bpm 90

This wrapper delegates to orchestrator.py so the same May25-only post-recording
mix/master, cleanup, and publishing behavior is used from both entry points.
"""

import argparse
import os
import sys
import glob
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("MAY19_MUSICGEN_PROJECT_ROOT", str(SCRIPT_DIR))).resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from orchestrator import run_production_only


def find_stems(stems_dir: str) -> dict:
    """Find WAV stems in directory."""
    patterns = ["*.wav", "*.WAV"]
    stems = {}
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(stems_dir, pattern))):
            name = os.path.basename(path)
            stems[name] = path
    return stems


def main():
    parser = argparse.ArgumentParser(description="May25 Audio Pipeline")
    parser.add_argument("--stems", required=True,
                        help="Directory containing stem WAV files")
    parser.add_argument("--song-name", required=True,
                        help="Song name for output files")
    parser.add_argument("--bpm", type=float, default=126.0,
                        help="Beats per minute (default: 126 for tech house)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: output/<song>/mastered)")
    parser.add_argument("--skip-stem-processing", action="store_true",
                        help="Compatibility flag; ignored by the portable orchestrator")
    parser.add_argument("--workers", type=int, default=3,
                        help="Parallel stem workers (default: 3)")
    parser.add_argument("--arrangement-seed", type=int, default=None,
                        help="Optional fixed arrangement seed")
    parser.add_argument("--professional-controller", action="store_true",
                        help="Enable the legacy May19 professional controller stage")
    parser.add_argument("--no-professional-controller", action="store_true",
                        help="Compatibility no-op; the legacy May19 professional controller is disabled by default")
    parser.add_argument("--professional-max-passes", type=int, default=5,
                        help="Maximum May19 professional controller candidate passes (default: 5)")
    parser.add_argument("--professional-fast", action="store_true",
                        help="Run one May19 controller pass for quick technical checks")
    parser.add_argument("--keep-mastering-workdir", action="store_true",
                        help="Preserve the full improved mastering workdir for A/B debugging")
    parser.add_argument("--legacy-production-engine", action="store_true",
                        help="Explicit rollback path: run the old ProductionEngine before May25")
    parser.add_argument("--no-external-output-copy", action="store_true",
                        help="Skip copying the completed song folder to the external RFLXN output root")
    parser.add_argument("--external-output-root", default=None,
                        help="Destination root for completed song folder copies")
    args = parser.parse_args()

    # Validate stems directory
    if not os.path.isdir(args.stems):
        print(f"Error: Stems directory not found: {args.stems}")
        sys.exit(1)

    # === STEM EQ: Tech house specific curves before QC ===
    print(f"\n{'='*60}")
    print(f"STEM EQ — Tech House Curves")
    print(f"{'='*60}")
    try:
        from stem_eq import process_stems
        eq_output_dir = args.stems.rstrip('/') + '_eq'
        eq_output_dir, eq_count = process_stems(args.stems, eq_output_dir)
        if eq_count > 0:
            print(f"  Using EQ'd stems: {eq_output_dir}")
            args.stems = eq_output_dir
    except Exception as e:
        print(f"  Stem EQ skipped: {e}")

    # === STEM QC v4: Transient-aware drum balance + RMS for melodic ===
    print(f"\n{'='*60}")
    print(f"STEM QC v4 — Transient-Aware Drum Balance")
    print(f"{'='*60}")
    try:
        from stem_qc_v4 import fix_stems
        qc_output_dir = args.stems.rstrip('/') + '_balanced'
        qc_output_dir, qc_fixes = fix_stems(args.stems, qc_output_dir)
        if qc_fixes:
            print(f"  Using balanced stems: {qc_output_dir}")
            args.stems = qc_output_dir
        else:
            print(f"  No fixes needed, using original stems")
    except Exception as e:
        print(f"  QC v4 skipped: {e}")

    # Find stems
    stems = find_stems(args.stems)
    if not stems:
        print(f"Error: No WAV files found in {args.stems}")
        sys.exit(1)

    print(f"Found {len(stems)} stems in {args.stems}")

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join("/Volumes/Raphael/Tech House/output", args.song_name, "mastered")

    os.makedirs(output_dir, exist_ok=True)

    run_production_only(
        args.stems,
        args.song_name,
        args.bpm,
        output_dir=output_dir,
        workers=args.workers,
        seed=getattr(args, 'arrangement_seed', None),
        external_output_copy=not getattr(args, 'no_external_output_copy', False),
        external_output_root=getattr(args, 'external_output_root', None),
    )

    # === CORRECTIVE POST-EQ (Skill 07) ===
    # Apply spectral balance correction after mastering
    master_wav = os.path.join(output_dir, "corrected_master.wav")
    
    # First apply professional post-production chain
    if os.path.exists(master_wav):
        try:
            from professional_post import apply_professional_chain
            professional_path = apply_professional_chain(
                master_wav, args.stems, output_dir, args.song_name, args.bpm
            )
            # Use professional master for corrective EQ
            if professional_path and os.path.exists(professional_path):
                master_wav = professional_path
        except Exception as e:
            print(f"  Professional post-production skipped: {e}")
    
    if os.path.exists(master_wav):
        try:
            import numpy as np
            import soundfile as sf
            from scipy.signal import butter, sosfiltfilt, lfilter
            import pyloudnorm as pyln

            data, sr = sf.read(master_wav)
            meter = pyln.Meter(sr)

            def peaking_eq(signal, freq, gain_db, q, sr):
                w0 = 2 * np.pi * freq / sr
                A = 10 ** (gain_db / 40)
                alpha = np.sin(w0) / (2 * q)
                b0 = 1 + alpha * A
                b1 = -2 * np.cos(w0)
                b2 = 1 - alpha * A
                a0 = 1 + alpha / A
                a1 = -2 * np.cos(w0)
                a2 = 1 - alpha / A
                b = np.array([b0 / a0, b1 / a0, b2 / a0])
                a = np.array([1, a1 / a0, a2 / a0])
                return lfilter(b, a, signal)

            # HPF at 35 Hz to tame sub rumble
            sos_hp = butter(4, 35, btype='high', fs=sr, output='sos')
            if data.ndim == 2:
                corrected = np.column_stack([
                    sosfiltfilt(sos_hp, data[:, 0]),
                    sosfiltfilt(sos_hp, data[:, 1])
                ])
                # Boost sub bass at 60Hz (references have 38% sub)
                corrected[:, 0] = peaking_eq(corrected[:, 0], 60, +4, 0.7, sr)
                corrected[:, 1] = peaking_eq(corrected[:, 1], 60, +4, 0.7, sr)
                # Cut mud at 400Hz (references have 12.6% low-mid vs our 22.3%)
                corrected[:, 0] = peaking_eq(corrected[:, 0], 400, -6, 1.0, sr)
                corrected[:, 1] = peaking_eq(corrected[:, 1], 400, -6, 1.0, sr)
            else:
                corrected = sosfiltfilt(sos_hp, data)
                corrected = peaking_eq(corrected, 60, +4, 0.7, sr)
                corrected = peaking_eq(corrected, 400, -6, 1.0, sr)

            # Match LUFS
            lufs_before = meter.integrated_loudness(data)
            lufs_after = meter.integrated_loudness(corrected)
            corrected = corrected * 10 ** ((lufs_before - lufs_after) / 20)

            # Save EQ-corrected master
            eq_path = os.path.join(output_dir, f"{args.song_name}_master_eq.wav")
            sf.write(eq_path, corrected, sr, subtype='PCM_24')
            print(f"  Corrective post-EQ applied: {eq_path}")

            # Also export MP3
            try:
                import subprocess
                mp3_path = os.path.join(output_dir, f"{args.song_name}_master.mp3")
                subprocess.run([
                    'ffmpeg', '-i', eq_path,
                    '-codec:a', 'libmp3lame', '-b:a', '320k',
                    mp3_path, '-y'
                ], capture_output=True, timeout=30)
                if os.path.exists(mp3_path):
                    print(f"  MP3 exported: {mp3_path}")
            except Exception as e:
                print(f"  MP3 export skipped: {e}")

        except Exception as e:
            print(f"  Corrective post-EQ skipped: {e}")

    # === REFERENCE A/B COMPARISON (Skill 08) ===
    # Compare mastered track against tech house spectral reference
    eq_wav = os.path.join(output_dir, f"{args.song_name}_master_eq.wav")
    ref_path = os.path.join(eq_wav) if os.path.exists(eq_wav) else master_wav
    if os.path.exists(ref_path):
        try:
            import numpy as np
            import soundfile as sf
            import pyloudnorm as pyln

            data, sr = sf.read(ref_path)
            meter = pyln.Meter(sr)
            mono = data.mean(axis=1) if data.ndim > 1 else data

            # Spectral analysis
            fft = np.fft.rfft(mono)
            mag = np.abs(fft) ** 2
            freqs = np.fft.rfftfreq(len(mono), 1 / sr)

            bands = {
                'Sub 20-60': (20, 60),
                'Bass 60-250': (60, 250),
                'Low-Mid 250-2k': (250, 2000),
                'Presence 2k-6k': (2000, 6000),
                'Air 6k-20k': (6000, 20000),
            }

            # Tech house reference profile (from actual reference track analysis)
            reference = {
                'Sub 20-60': (30, 45),      # MK/John Summit/Dennis Ferrer average
                'Bass 60-250': (35, 50),     # Heavy bass presence
                'Low-Mid 250-2k': (8, 15),   # Clean mids
                'Presence 2k-6k': (2, 5),    # Not overly bright
                'Air 6k-20k': (2, 5),        # Not overly airy
            }

            total = sum(np.sum(mag[(freqs >= lo) & (freqs < hi)]) for lo, hi in bands.values())

            # LUFS
            lufs = meter.integrated_loudness(data)
            peak_db = 20 * np.log10(np.max(np.abs(data)) + 1e-10)
            rms = np.sqrt(np.mean(mono ** 2))
            crest = peak_db - 20 * np.log10(rms + 1e-10)

            # Stereo correlation
            if data.ndim == 2:
                corr = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
            else:
                corr = 1.0

            print(f"\n  === REFERENCE A/B COMPARISON (Skill 08) ===")
            print(f"  Target: Tech House spectral profile")
            print(f"  LUFS: {lufs:.1f} (club: -8 to -6, streaming: -14)")
            print(f"  Crest Factor: {crest:.1f} dB (target: 4-8)")
            print(f"  Stereo Correlation: {corr:.3f} (target: 0.3-0.7)")

            issues = []
            print(f"\n  Spectral Balance:")
            for name, (lo, hi) in bands.items():
                pct = np.sum(mag[(freqs >= lo) & (freqs < hi)]) / total * 100
                target_lo, target_hi = reference[name]
                status = '✓' if target_lo <= pct <= target_hi else '✗'
                if status == '✗':
                    if pct > target_hi:
                        issues.append(f"Too much {name}: {pct:.0f}% (target {target_lo}-{target_hi}%)")
                    else:
                        issues.append(f"Too little {name}: {pct:.0f}% (target {target_lo}-{target_hi}%)")
                print(f"    {name}: {pct:5.1f}%  target {target_lo}-{target_hi}%  {status}")

            if issues:
                print(f"\n  ⚠ Spectral Issues Found:")
                for issue in issues:
                    print(f"    - {issue}")
            else:
                print(f"\n  ✓ All spectral bands within tech house reference range")

            # Save analysis report
            report = {
                'lufs': float(lufs),
                'crest_factor_db': float(crest),
                'peak_dbfs': float(peak_db),
                'stereo_correlation': float(corr),
                'spectral_balance': {},
                'issues': issues,
                'reference_profile': {k: list(v) for k, v in reference.items()},
            }
            for name, (lo, hi) in bands.items():
                pct = float(np.sum(mag[(freqs >= lo) & (freqs < hi)]) / total * 100)
                report['spectral_balance'][name] = {
                    'percent': pct,
                    'target': list(reference[name]),
                    'in_range': reference[name][0] <= pct <= reference[name][1],
                }

            import json
            report_path = os.path.join(output_dir, f"{args.song_name}_ab_report.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"  Report saved: {report_path}")

        except Exception as e:
            print(f"  Reference A/B comparison skipped: {e}")


if __name__ == "__main__":
    main()
