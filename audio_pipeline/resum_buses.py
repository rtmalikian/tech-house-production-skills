"""
Re-sum pre-optimised stems into buses using the new numpy additive mix,
then continue the full downstream pipeline:
  1. Group opt_proc stems into buses
  2. Numpy sum each bus (no FFmpeg amix averaging)
  3. Intelligent EQ carving between buses
  4. Professional bus EQ shaping
  5. Final mix (sum buses)
  6. Mastering chain

Usage:
    python scripts/audio_pipeline/resum_buses.py
"""

import os
import sys
import glob
import gc
import numpy as np
import soundfile as sf

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.dirname(os.path.dirname(PIPELINE_DIR))
sys.path.insert(0, PIPELINE_DIR)

from post_production import ProductionEngine as PostProduction

STEM_GLOB   = os.path.join(
    REPO_ROOT,
    "output/mastered/buses",
    "opt_proc_05032026_200043_A_A_Rast_v10_refactored*.wav"
)
OUTPUT_DIR  = os.path.join(REPO_ROOT, "output/mastered")
BUS_DIR     = os.path.join(OUTPUT_DIR, "buses")
SONG_NAME   = "05032026_200043_A_A_Rast_v10_refactored"


def pad_and_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """True additive mix of two arrays with potentially different lengths."""
    if len(a) < len(b):
        res = np.zeros_like(b)
        res[:len(a)] = a
        res += b
    else:
        res = np.zeros_like(a)
        res[:len(b)] = b
        res += a
    return res


def sum_group(paths: list) -> tuple:
    """Sum a list of wav paths into one float64 array. Returns (mix, sr)."""
    mix, sr = sf.read(paths[0], always_2d=True)
    mix = mix.astype(np.float64)
    for p in paths[1:]:
        y, _ = sf.read(p, always_2d=True)
        mix = pad_and_add(mix, y.astype(np.float64))
    peak = np.max(np.abs(mix))
    if peak > 1.0:
        mix /= peak
    return mix.astype(np.float32), sr


def group_stems(paths: list) -> dict:
    buses = {"drums": [], "bass": [], "melody": [], "fx": []}
    for p in paths:
        name = os.path.basename(p).lower()
        if any(x in name for x in ['kick','snare','hat','clap','drum','bongo',
                                    'conga','tambourine','maracas','perc',
                                    'instr','side_stick']):
            buses["drums"].append(p)
        elif 'bass' in name:
            buses["bass"].append(p)
        elif any(x in name for x in ['melody','chorus','counter','pad','chord']):
            buses["melody"].append(p)
        else:
            buses["fx"].append(p)
    return buses


def main():
    stems = sorted(glob.glob(STEM_GLOB))
    if not stems:
        print(f"ERROR: No stems found matching:\n  {STEM_GLOB}")
        sys.exit(1)

    print(f"Found {len(stems)} opt_proc stems.")
    groups = group_stems(stems)
    for bus, paths in groups.items():
        print(f"  {bus:8s}: {len(paths)} stems")

    os.makedirs(BUS_DIR, exist_ok=True)

    # ── 1. Numpy sum into buses ──────────────────────────────────────────────
    print("\n[1/4] Summing stems into buses (numpy additive mix)...")
    bus_paths = {}
    for bus_name, paths in groups.items():
        if not paths:
            print(f"  Skipping empty bus: {bus_name}")
            continue
        out_path = os.path.join(BUS_DIR, f"bus_{bus_name}.wav")
        print(f"  Summing {bus_name} ({len(paths)} stems) → {os.path.basename(out_path)}")
        mix, sr = sum_group(paths)
        peak = np.max(np.abs(mix))
        print(f"    Peak after sum: {peak:.4f} ({20*np.log10(max(peak,1e-9)):.1f} dBFS)")
        sf.write(out_path, mix, sr, subtype='FLOAT')
        bus_paths[bus_name] = out_path
        del mix
        gc.collect()

    # ── 2–4. Downstream pipeline via PostProduction ──────────────────────────
    print("\nInitialising PostProduction for downstream stages...")
    pp = PostProduction(output_dir=OUTPUT_DIR)

    # ── 2. Intelligent EQ carving ────────────────────────────────────────────
    print("\n[2/4] Intelligent EQ carving between buses...")

    if "melody" in bus_paths and "drums" in bus_paths:
        m_y, sr = sf.read(bus_paths["melody"])
        d_y, _  = sf.read(bus_paths["drums"])
        m_y = pp.eq.intelligent_carve(m_y, d_y, sr, (2000, 3000), depth=-2.5)
        m_y = pp.eq.intelligent_carve(m_y, d_y, sr, (200,  450),  depth=-3.0)
        sf.write(bus_paths["melody"], m_y, sr, subtype='FLOAT')
        print("  Carved melody vs drums (2kHz snare crack + 300Hz mud)")

    if "melody" in bus_paths and "bass" in bus_paths:
        m_y, sr = sf.read(bus_paths["melody"])
        b_y, _  = sf.read(bus_paths["bass"])
        m_y = pp.eq.intelligent_carve(m_y, b_y, sr, (60, 250), depth=-4.0)
        sf.write(bus_paths["melody"], m_y, sr, subtype='FLOAT')
        print("  Carved melody vs bass (60-250Hz mud)")

    if "bass" in bus_paths and "drums" in bus_paths:
        b_y, sr = sf.read(bus_paths["bass"])
        d_y, _  = sf.read(bus_paths["drums"])
        b_y = pp.eq.apply_frequency_slotting(b_y, d_y, sr)
        sf.write(bus_paths["bass"], b_y, sr, subtype='FLOAT')
        print("  Applied kick/bass frequency slotting")

    # ── 3. Professional bus EQ shaping ──────────────────────────────────────
    print("\n[3/4] Professional bus EQ shaping...")
    for bus_name, path in bus_paths.items():
        print(f"  Optimising {bus_name} bus spectral profile...")
        y, sr = sf.read(path)
        y = pp._ensure_stereo(np.asarray(y, dtype=np.float32))
        y_opt = pp.eq.optimize_to_target(y, sr, bus_name, max_passes=8)
        sf.write(path, y_opt, sr, subtype='FLOAT')
        del y, y_opt
        gc.collect()

    # ── 4. Sum buses → final mix ─────────────────────────────────────────────
    print("\n[4/4] Summing buses into final mix...")
    bus_list = list(bus_paths.values())
    mix, sr = sf.read(bus_list[0], always_2d=True)
    mix = mix.astype(np.float64)
    for p in bus_list[1:]:
        y, _ = sf.read(p, always_2d=True)
        mix = pad_and_add(mix, y.astype(np.float64))
    peak = np.max(np.abs(mix))
    if peak > 1.0:
        mix /= peak

    premix_path = os.path.join(OUTPUT_DIR, f"{SONG_NAME}_premix.wav")
    sf.write(premix_path, mix.astype(np.float32), sr, subtype='FLOAT')
    print(f"  Pre-master mix → {os.path.basename(premix_path)}")
    print(f"  Peak: {np.max(np.abs(mix)):.4f} ({20*np.log10(max(np.max(np.abs(mix)),1e-9)):.1f} dBFS)")
    del mix
    gc.collect()

    # ── 5. Mastering ─────────────────────────────────────────────────────────
    print("\nMastering chain (M/S, EQ, limiter, loudnorm -14 LUFS)...")
    master_path = pp.apply_mastering(premix_path, SONG_NAME)
    print(f"\nDone. Master → {master_path}")


if __name__ == "__main__":
    main()
