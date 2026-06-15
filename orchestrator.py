#!/usr/bin/env python3
"""
May19 Improved Full Pipeline Orchestrator — MIDI → Recording → Production.

Integrates MIDI generation, Fantom recording, and the Python Revamp
production engine into a single unified pipeline, then runs the May19
algorithmic professional mix/master controller as a first-class stage.

Usage:
    # Full pipeline (MIDI → record → produce):
    python May19_improved_audio_pipeline/orchestrator.py --full

    # Production only (existing stems):
    python May19_improved_audio_pipeline/orchestrator.py \
        --stems output/<song>/recordings/ \
        --song-name <song> \
        --bpm 79

    # MIDI generation only:
    python May19_improved_audio_pipeline/orchestrator.py --midi-only
"""

import argparse
import os
import sys
import glob
import hashlib
import json
import random
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("MAY19_MUSICGEN_PROJECT_ROOT", str(SCRIPT_DIR))).resolve()

# Add this package's directory first (highest priority)
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# NOTE: Do NOT add scripts/audio_pipeline to sys.path here — it would shadow
# our new gain_staging.py, dynamic_eq.py, etc. The Fantom imports are handled
# lazily inside step_record() where they're actually needed.

import math
import numpy as np
import mido
import sounddevice as sd
from midi_orchestrator import main as generate_midi

SKILLBASED_DIR = str(SCRIPT_DIR.parent / "skillbased_pipeline")
_june1_orig_path = sys.path.copy()
if SKILLBASED_DIR not in sys.path:
    sys.path.append(SKILLBASED_DIR)
from june1_corrected_pipeline import (
    assign_pans, render_stems_parallel, apply_fx_automation,
    apply_master_chain, pink_reference_rms_by_category, stage_buses,
    amp_to_db, db_to_amp, write_audio,
    PREMASTER_HEADROOM_DB, SR_TARGET, BUS_NAMES,
    TARGET_LUFS, TRUE_PEAK_CEILING_DBTP,
)
sys.path = _june1_orig_path


PIPELINE_NAME = "June1 Skill-Based Pipeline"
DEFAULT_OUTPUT_ROOT = Path("/Volumes/Raphael/Tech House/output")
DEFAULT_EXTERNAL_OUTPUT_ROOT = Path("/Volumes/Raphael/Tech House/output")
MAY25_MANIFEST_FILE = "may25_artifact_manifest.json"
MAY19_MANIFEST_FILE = "may19_artifact_manifest.json"
JUNE1_ANALYSIS_FILE = "corrected_analysis.json"


def atomic_write_json(path: str, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def project_dir_for_song(song_name: str) -> str:
    return os.path.join(str(DEFAULT_OUTPUT_ROOT), song_name)


def mastered_dir_for_song(song_name: str) -> str:
    return os.path.join(project_dir_for_song(song_name), "mastered")


def project_dir_for_completed_output(song_name: str, output_dir: str = None) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve().parent
    return Path(project_dir_for_song(song_name)).resolve()


def external_output_root_available(destination_root: Path) -> bool:
    destination_root = Path(destination_root).expanduser()
    parts = destination_root.parts
    if len(parts) >= 3 and parts[1] == "Volumes":
        return Path("/", parts[1], parts[2]).is_dir()
    return destination_root.parent.is_dir()


def export_completed_song_folder(song_name: str,
                                 output_dir: str = None,
                                 destination_root: str = None,
                                 enabled: bool = True) -> str:
    """Copy the completed song project folder to the external RFLXN output root."""
    if not enabled:
        print("  External output copy disabled.")
        return None

    source_dir = project_dir_for_completed_output(song_name, output_dir)
    destination_root_path = Path(destination_root or DEFAULT_EXTERNAL_OUTPUT_ROOT).expanduser()
    if not source_dir.is_dir():
        print(f"  External output copy skipped: source folder not found: {source_dir}")
        return None
    if not external_output_root_available(destination_root_path):
        print(f"  External output copy skipped: destination volume not available: {destination_root_path}")
        return None

    destination_root_path.mkdir(parents=True, exist_ok=True)
    destination_dir = destination_root_path / source_dir.name
    try:
        if source_dir.resolve() == destination_dir.resolve():
            print(f"  External output copy skipped: source already at destination: {destination_dir}")
            return str(destination_dir)
        try:
            destination_dir.resolve().relative_to(source_dir.resolve())
            print(f"  External output copy skipped: destination is inside source: {destination_dir}")
            return None
        except ValueError:
            pass
        shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)
    except Exception as exc:
        print(f"  External output copy failed: {exc}")
        return None

    print(f"  External output copy: {source_dir} -> {destination_dir}")
    return str(destination_dir)


def find_fantom_device() -> int:
    """Probe audio devices and return the index of the Roland Fantom."""
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        name = dev.get('name', '').lower()
        if 'fantom' in name:
            print(f"  Found Roland Fantom: device {i} — {dev['name']} "
                  f"({dev['max_input_channels']} in, {dev['max_output_channels']} out)")
            return i
    print("  WARNING: Roland Fantom not found among audio devices:")
    for i, dev in enumerate(devices):
        ch_in = dev['max_input_channels']
        ch_out = dev['max_output_channels']
        if ch_in > 0:
            print(f"    {i}: {dev['name']} ({ch_in} in, {ch_out} out)")
    raise RuntimeError("Roland Fantom audio device not connected")


def get_bpm_from_midi(midi_path: str) -> float:
    """Extract BPM from a MIDI file."""
    mid = mido.MidiFile(midi_path)
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                return round(60_000_000 / msg.tempo, 2)
    return 90.0


def find_stems(stems_dir: str, exclude: list = None) -> dict:
    """Find WAV stems in directory, optionally excluding by filename substring."""
    stems = {}
    exclude = [x.lower() for x in (exclude or [])]
    for path in sorted(glob.glob(os.path.join(stems_dir, "*.wav"))):
        name = os.path.basename(path)
        if not name.startswith('.'):
            if not any(ex in name.lower() for ex in exclude):
                stems[name] = path
    return stems


def find_midi_file(song_name: str) -> str:
    """Search for MIDI file in common locations."""
    search_dirs = [
        os.path.join(str(SCRIPT_DIR), "output"),
        str(DEFAULT_OUTPUT_ROOT),
        os.path.join(str(DEFAULT_OUTPUT_ROOT), song_name, "midi"),
        os.path.join(str(PROJECT_ROOT), "midi"),
    ]
    for dir_path in search_dirs:
        matches = sorted(glob.glob(os.path.join(dir_path, f"{song_name}*.mid*")))
        if matches:
            return matches[-1]
    return None


def output_song_name(song_name: str, bpm: float = None) -> str:
    """Match ProductionEngine output naming without changing project folder names."""
    base = str(song_name or "song")
    if re.search(r'(?i)(^|[_-])\d+(?:\.\d+)?bpm($|[_-])', base):
        return base
    if bpm is None:
        return base
    return f"{base}_{int(round(float(bpm)))}bpm"


def audit_expected_variants(output_dir: str, song_name: str, bpm: float):
    """Warn when retained review variants were not copied into final/."""
    output_name = output_song_name(song_name, bpm)
    expected = [
        f"{output_name}_drums-bass_streaming_master.wav",
        f"{output_name}_pads-drums-bass_streaming_master.wav",
    ]
    final_dir = os.path.join(output_dir, "final")
    missing_final = [f for f in expected if not os.path.exists(os.path.join(final_dir, f))]

    if not missing_final:
        print("  Verified retained variants: drums-bass, pads-drums-bass")
        return

    print("  WARNING: Missing expected utility variant output(s).")
    if missing_final:
        print(f"    Missing in mastered/final/: {', '.join(missing_final)}")
    print("    Likely causes: Step 5 did not run, final copy failed,")
    print("    selected drum/bass/pad stems were absent, or variant mastering failed.")


def expected_streaming_final_names(song_name: str, bpm: float) -> set:
    output_name = output_song_name(song_name, bpm)
    return {
        f"{output_name}_drums-bass_streaming_master.wav",
        f"{output_name}_pads-drums-bass_streaming_master.wav",
        f"{output_name}_streaming_master.wav",
    }


def prune_streaming_final_surface(mastered_dir: str, song_name: str, bpm: float,
                                  require_professional: bool = True) -> dict:
    """Keep only the generic May25 streaming release surface in final/."""
    output_name = output_song_name(song_name, bpm)
    allowed_final = {f"{output_name}_streaming_master.wav"}

    removed = []
    final_dir = os.path.join(mastered_dir, "final")
    if os.path.isdir(final_dir):
        for name in sorted(os.listdir(final_dir)):
            path = os.path.join(final_dir, name)
            if name.lower().endswith(".wav") and name not in allowed_final:
                os.remove(path)
                removed.append(os.path.relpath(path, mastered_dir))

    stale_parent_names = {
        f"{output_name}_master.wav",
        f"{output_name}_streaming_master.wav",
        f"{output_name}_premaster.wav",
        f"{output_name}_drums-bass_master.wav",
        f"{output_name}_pads-drums-bass_master.wav",
        f"{output_name}_may19_professional_master.wav",
        f"{output_name}_may19_professional_premaster.wav",
        f"{output_name}_may19_professional_streaming_master.wav",
    }
    for name in sorted(stale_parent_names):
        path = os.path.join(mastered_dir, name)
        if os.path.isfile(path):
            os.remove(path)
            removed.append(name)

    if removed:
        print("  Pruned stale non-streaming final artifacts:")
        for name in removed:
            print(f"    {name}")
    return {
        "allowed_final": sorted(allowed_final),
        "removed": removed,
    }


def step_generate_midi() -> tuple:
    """Step 1: Generate MIDI sequence."""
    print("\n" + "=" * 70)
    print("[STEP 1] MIDI GENERATION")
    print("=" * 70)
    result = generate_midi()
    if isinstance(result, tuple):
        midi_path, metadata = result
    else:
        midi_path = result
        metadata = {}
    song_name = os.path.basename(midi_path).replace(".mid", "")
    bpm = metadata.get('bpm', get_bpm_from_midi(midi_path))
    print(f"  MIDI: {midi_path}")
    print(f"  Song: {song_name}")
    print(f"  BPM: {bpm}")

    # Copy MIDI to song folder
    project_dir = project_dir_for_song(song_name)
    midi_dir = os.path.join(project_dir, "midi")
    os.makedirs(midi_dir, exist_ok=True)
    dest = os.path.join(midi_dir, os.path.basename(midi_path))
    shutil.copy2(midi_path, dest)
    print(f"  Copied to: {dest}")

    return midi_path, song_name, bpm, metadata


def step_record(midi_path: str, song_name: str, metadata: dict = None) -> dict:
    """Step 2: Record stems from Roland Fantom."""
    print("\n" + "=" * 70)
    print("[STEP 2] RECORDING FROM FANTOM")
    print("=" * 70)

    try:
        # Temporarily add audio_pipeline to path for Fantom imports
        audio_pipeline_dir = str(SCRIPT_DIR / "audio_pipeline")
        if audio_pipeline_dir not in sys.path:
            sys.path.insert(0, audio_pipeline_dir)
        from audio_recorder import AudioRecorder, MultiPassOrchestrator
        from fantom_midi_control import FantomController
    except ImportError as e:
        print(f"  Error: Could not import recording modules: {e}")
        print("  Make sure Full_Pipeline_05102026/audio_pipeline is in the path.")
        return {}

    project_dir = project_dir_for_song(song_name)
    recordings_dir = os.path.join(project_dir, "recordings")
    os.makedirs(recordings_dir, exist_ok=True)

    controller = FantomController()
    if not controller.output:
        print("  Error: Fantom hardware not connected. Skipping recording.")
        return {}

    device_index = find_fantom_device()
    recorder = AudioRecorder(device_index=device_index, output_dir=recordings_dir)
    orch = MultiPassOrchestrator(recorder, controller)
    stems = orch.run_multi_pass(midi_path, song_name, metadata=metadata)
    print(f"  Recorded {len(stems)} stems")
    return stems


def step_june1_mix_master(recordings_dir: str, song_name: str, bpm: float,
                         output_dir: str = None, seed: int = None,
                         workers: int = 8, sr: int = SR_TARGET) -> str:
    """Step 3: June 1 skill-based mix/master pipeline.

    Loads recorded stems, classifies by role, applies bounded-random panning,
    pink-noise-informed gain staging, bus routing, FX automation, dual-bus
    sidechain, spatial reverb, and a streaming master chain.
    """
    print("\n" + "=" * 70)
    print("[STEP 3] JUNE 1 SKILL-BASED MIX/MASTER")
    print("=" * 70)

    if seed is None:
        seed = random.SystemRandom().randint(1, 2_147_483_647)

    if output_dir is None:
        output_dir = mastered_dir_for_song(song_name)
    os.makedirs(output_dir, exist_ok=True)

    recordings = Path(recordings_dir)
    if not recordings.is_dir():
        print(f"  Error: recordings dir not found: {recordings_dir}")
        return None
    wavs = sorted(recordings.glob("*.wav"))
    if not wavs:
        print(f"  Error: no WAV files in {recordings_dir}")
        return None

    stems = assign_pans(wavs, seed)
    length = max(
        int(math.ceil(s.frames * (sr / s.source_sr))) if s.source_sr != sr else s.frames
        for s in stems
    )
    pink_refs, pink_signal = pink_reference_rms_by_category(length, sr, seed + 10_000)

    print(f"  Rendering {len(stems)} stems with {workers} workers")
    buses, stem_rows = render_stems_parallel(stems, length, pink_refs, pink_signal, sr, bpm, workers)
    buses, bus_stage_report = stage_buses(buses, pink_signal, sr, bpm)
    automated_buses, fx_report = apply_fx_automation(buses, bpm, sr, seed)
    mix = automated_buses["premaster"]

    pre_peak_dbfs = amp_to_db(float(np.max(np.abs(mix))))
    headroom_gain_db = min(0.0, PREMASTER_HEADROOM_DB - pre_peak_dbfs)
    mix *= np.float32(db_to_amp(headroom_gain_db))
    post_peak_dbfs = amp_to_db(float(np.max(np.abs(mix))))
    print(f"  Premaster peak: {pre_peak_dbfs:.2f} dBFS -> {post_peak_dbfs:.2f} dBFS (headroom {headroom_gain_db:+.2f} dB)")

    master, master_report = apply_master_chain(mix, sr)

    mix_path = os.path.join(output_dir, "corrected_mix.wav")
    master_path = os.path.join(output_dir, "corrected_master.wav")
    analysis_path = os.path.join(output_dir, JUNE1_ANALYSIS_FILE)
    write_audio(Path(mix_path), mix, sr)
    write_audio(Path(master_path), master, sr)

    final_lufs = master_report["limiter"]["integrated_lufs"]
    final_true_peak = master_report["limiter"]["final_true_peak_dbtp"]
    print(f"  Master: {master_path}")
    print(f"  LUFS: {final_lufs:.2f} (target {TARGET_LUFS}) | True peak: {final_true_peak:.2f} dBTP (ceiling {TRUE_PEAK_CEILING_DBTP})")

    analysis = {
        "pipeline": PIPELINE_NAME,
        "song_name": song_name,
        "recordings_dir": str(recordings_dir),
        "output_dir": str(output_dir),
        "sample_rate": sr,
        "bpm": bpm,
        "seed": seed,
        "workers": workers,
        "stem_count": len(stem_rows),
        "pan_positions": {row["file"]: row["pan"] for row in stem_rows},
        "stem_rms_values": {
            row["file"]: {
                "category": row["category"],
                "bus": row["bus"],
                "active_band_rms_dbfs": row["active_band_rms_dbfs"],
                "post_gain_fullband_rms_dbfs": row["post_gain_fullband_rms_dbfs"],
                "applied_gain_db": row["applied_gain_db"],
            }
            for row in stem_rows
        },
        "mix_metrics": {
            "pre_headroom_peak_dbfs": pre_peak_dbfs,
            "headroom_gain_db": headroom_gain_db,
            "corrected_mix_peak_dbfs": post_peak_dbfs,
        },
        "fx_automation_report": fx_report,
        "master_chain": master_report,
        "final_lufs_metrics": {
            "integrated_lufs": final_lufs,
            "target_lufs": TARGET_LUFS,
            "loudness_error_db": final_lufs - TARGET_LUFS,
            "true_peak_dbtp": final_true_peak,
            "true_peak_ceiling_dbtp": TRUE_PEAK_CEILING_DBTP,
        },
        "deliverables": {
            "corrected_mix": mix_path,
            "corrected_master": master_path,
            "corrected_analysis": analysis_path,
        },
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"  Analysis: {analysis_path}")

    return master_path


def step_sample_pack(midi_path: str, song_name: str, recordings_dir: str = None) -> str:
    """Step 5: Generate companion sample pack from MIDI."""
    print("\n" + "=" * 70)
    print("[STEP 5] SAMPLE PACK GENERATION")
    print("=" * 70)

    try:
        # Temporarily add audio_pipeline to path for sample generation
        audio_pipeline_dir = str(SCRIPT_DIR / "audio_pipeline")
        if audio_pipeline_dir not in sys.path:
            sys.path.insert(0, audio_pipeline_dir)
        from generate_sample_stems import generate_sample_pack
    except ImportError as e:
        print(f"  Error: Could not import sample generation module: {e}")
        return None

    project_dir = project_dir_for_song(song_name)
    sample_output = os.path.join(project_dir, "sample_pack_build")

    try:
        device_index = find_fantom_device()
        sample_dir = generate_sample_pack(
            midi_path,
            output_root=sample_output,
            device_index=device_index,
            recordings_dir=recordings_dir,
        )
        print(f"  Sample pack: {sample_dir}")
        return sample_dir
    except Exception as e:
        print(f"  Sample generation failed: {e}")
        return None


def step_consolidate(song_name: str, midi_path: str, master_path: str,
                     stems: dict, metadata: dict):
    """Step 5: Project consolidation — generate comprehensive documentation."""
    print("\n" + "=" * 70)
    print("[STEP 6] PROJECT CONSOLIDATION")
    print("=" * 70)

    project_dir = project_dir_for_song(song_name)
    mastered_dir = os.path.join(project_dir, "mastered")
    recordings_dir = os.path.join(project_dir, "recordings")
    log_dir = os.path.join(project_dir, "log")

    # Copy MIDI to project
    midi_dir = os.path.join(project_dir, "midi")
    os.makedirs(midi_dir, exist_ok=True)
    if midi_path and os.path.exists(midi_path):
        dest = os.path.join(midi_dir, os.path.basename(midi_path))
        if not os.path.exists(dest):
            shutil.copy2(midi_path, dest)
        print(f"  MIDI: {dest}")

    # Gather data for documentation
    bpm = metadata.get('bpm', 'N/A')
    output_name = output_song_name(song_name, bpm if bpm != 'N/A' else None)
    key = metadata.get('key', 'N/A')
    scale = metadata.get('scale', 'N/A')
    time_sig = metadata.get('time_signature', '4-4')
    main_family = metadata.get('main_drum_family', 'N/A')
    chorus_family = metadata.get('chorus_drum_family', 'N/A')
    inverted = metadata.get('inverted', False)
    is_armenian = metadata.get('is_armenian', False)
    sections_meta = metadata.get('sections', {})

    # Read recording manifests
    manifests = []
    if os.path.isdir(recordings_dir):
        for f in sorted(os.listdir(recordings_dir)):
            if f.endswith('_manifest.json'):
                try:
                    with open(os.path.join(recordings_dir, f)) as mf:
                        manifests.append(json.load(mf))
                except Exception:
                    pass

    # Collect all parts from all manifests
    all_parts = []
    for m in manifests:
        all_parts.extend(m.get('parts', []))

    # Read automation plan
    auto_plan_path = os.path.join(log_dir, "automation_plan.json")
    auto_plan = None
    if os.path.exists(auto_plan_path):
        try:
            with open(auto_plan_path) as af:
                auto_plan = json.load(af)
        except Exception:
            pass

    # Find master/mix variant files
    master_files = {}
    if os.path.isdir(mastered_dir):
        master_candidates = [(mastered_dir, f) for f in sorted(os.listdir(mastered_dir))]
        final_dir = os.path.join(mastered_dir, "final")
        if os.path.isdir(final_dir):
            master_candidates.extend((final_dir, f) for f in sorted(os.listdir(final_dir)))
        for base_dir, f in master_candidates:
            if f.endswith('_master.wav'):
                rel_file = os.path.relpath(os.path.join(base_dir, f), mastered_dir)
                if (
                    f == f"{output_name}_streaming_master.wav"
                    or (
                        f.endswith("_streaming_master.wav")
                        and "may19" not in f
                        and "drums-bass" not in f
                        and "pads-drums-bass" not in f
                    )
                ):
                    master_files['Streaming Final Master'] = rel_file
                elif 'may19_professional' in f:
                    master_files['Legacy May19 Professional Reference'] = rel_file
                elif 'pads-drums-bass' in f:
                    master_files['Pads/Chords + Drums + Bass'] = rel_file
                elif 'drums-bass' in f:
                    master_files['Drums + Bass'] = rel_file
                elif 'bass1' in f:
                    master_files['Bass 1 (No Harmonic Bass)'] = rel_file
                elif 'bass2' in f:
                    master_files['Bass 2 (No Bass)'] = rel_file
                elif 'pristine' in f:
                    master_files['Pristine Mix'] = rel_file
                elif f.replace('_master.wav', '') in {song_name, output_name} or '_inv' in f:
                    master_files['Final Master'] = rel_file

    professional_report = None
    professional_report_path = os.path.join(mastered_dir, "may19_professional_passes", "producer_iteration_report.json")
    if os.path.exists(professional_report_path):
        try:
            with open(professional_report_path) as pf:
                professional_report = json.load(pf)
        except Exception:
            professional_report = None

    # Build send config lookup
    try:
        sys.path.insert(0, str(SCRIPT_DIR))
        import config as cfg
        stem_send_map = getattr(cfg, 'STEM_SEND_MAP', [])
        layer_presets = getattr(cfg, 'LAYER_PRESETS', {})
        reverb_cats = getattr(cfg, 'REVERB_CATEGORIES', {})
        delay_presets = getattr(cfg, 'DELAY_PRESETS', {})
        drum_parallel = getattr(cfg, 'DRUM_PARALLEL_COMP', {})
        drum_clip = getattr(cfg, 'DRUM_DYNAMIC_SOFT_CLIP', {})
        kick_sc = getattr(cfg, 'KICK_BASS_SIDECHAIN', {})
        effect_depth = getattr(cfg, 'EFFECT_DEPTH', {})
        lufs_targets = getattr(cfg, 'STEM_LUFS_TARGETS', {})
    except Exception:
        stem_send_map = []
        layer_presets = {}
        reverb_cats = {}
        delay_presets = {}
        drum_parallel = {}
        drum_clip = {}
        kick_sc = {}
        effect_depth = {}
        lufs_targets = {}

    def get_send_for_track(track_name):
        n = track_name.lower()
        for match, category, rev_send, dly_send in stem_send_map:
            if match in n:
                return category, rev_send, dly_send
        return None, 0.0, 0.0

    # Generate documentation
    doc_path = os.path.join(project_dir, "DOCUMENTATION.md")
    with open(doc_path, "w") as f:
        f.write(f"# Project: {song_name}\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"---\n\n")

        # === Musical Parameters ===
        f.write(f"## Musical Parameters\n\n")
        f.write(f"| Parameter | Value |\n")
        f.write(f"|-----------|-------|\n")
        f.write(f"| MIDI File | `{os.path.basename(midi_path)}` |\n")
        f.write(f"| BPM | {bpm} |\n")
        f.write(f"| Key | {key} |\n")
        f.write(f"| Scale/Mode | {scale} |\n")
        f.write(f"| Time Signature | {time_sig.replace('-', '/')} |\n")
        f.write(f"| Main Drum Family | {main_family} |\n")
        f.write(f"| Chorus Drum Family | {chorus_family} |\n")
        f.write(f"| Inversion | {'Yes' if inverted else 'No'} |\n")
        f.write(f"| Armenian/Maqam | {'Yes' if is_armenian else 'No'} |\n\n")

        # === Output Files ===
        f.write(f"## Output Files\n\n")
        f.write(f"| File | Description |\n")
        f.write(f"|------|-------------|\n")
        for desc, fname in master_files.items():
            f.write(f"| `{fname}` | {desc} |\n")
        f.write(f"\n")

        if professional_report:
            f.write("## May19 Professional Controller\n\n")
            f.write(f"| Parameter | Value |\n")
            f.write(f"|-----------|-------|\n")
            f.write(f"| Selected Pass | {professional_report.get('selected_pass', 'N/A')} |\n")
            f.write(f"| Selected Score | {professional_report.get('selected_score', 0):.2f} |\n")
            f.write(f"| Selected Premix | `{os.path.relpath(professional_report.get('selected_mix_path', ''), mastered_dir)}` |\n")
            f.write(f"| Report | `may19_professional_passes/producer_iteration_report.md` |\n\n")
            f.write("| Pass | Score | Peak dB | Return/Dry dB | Drums/Melody dB | Bass/Melody dB | Pads/Melody dB |\n")
            f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
            for item in professional_report.get('passes', []):
                metrics = item.get('metrics', {})
                selected = " selected" if item.get('selected') else ""
                f.write(
                    f"| {item.get('pass_index')}{selected} | {item.get('score', 0):.2f} | "
                    f"{metrics.get('peak_db', 0):+.2f} | {metrics.get('return_to_dry_db', 0):+.2f} | "
                    f"{metrics.get('drums_to_melody_db', 0):+.2f} | {metrics.get('bass_to_melody_db', 0):+.2f} | "
                    f"{metrics.get('pads_to_melody_db', 0):+.2f} |\n"
                )
            f.write("\n")

        # === Roland Fantom Patches ===
        f.write(f"## Roland Fantom Patches\n\n")
        if all_parts:
            f.write(f"| Part | Track | Patch Name | MSB | LSB | PC |\n")
            f.write(f"|------|-------|------------|-----|-----|-----|\n")
            for part in all_parts:
                p_idx = part.get('part', '?')
                t_name = part.get('source_track_name', 'Unknown')
                patch = part.get('patch', {})
                f.write(f"| {p_idx} | {t_name} | {patch.get('name', 'Unknown')} | {patch.get('msb', '?')} | {patch.get('lsb', '?')} | {patch.get('pc', '?')} |\n")
            f.write(f"\n")
        else:
            f.write(f"No recording manifests found.\n\n")

        # === Sound Design ===
        f.write(f"## Sound Design\n\n")

        # LFO Modulations
        f.write(f"### LFO Modulations\n\n")
        lfo_parts = [p for p in all_parts if p.get('sound_design', {}).get('lfo_matrix')]
        if lfo_parts:
            f.write(f"| Part | Track | LFO Details |\n")
            f.write(f"|------|-------|-------------|\n")
            for part in lfo_parts:
                p_idx = part.get('part', '?')
                t_name = part.get('source_track_name', 'Unknown')
                lfo = part['sound_design']['lfo_matrix']
                f.write(f"| {p_idx} | {t_name} | {lfo} |\n")
            f.write(f"\n")
        else:
            f.write(f"No LFO modulations applied.\n\n")

        # MFX Applied
        f.write(f"### MFX Applied\n\n")
        mfx_parts = [p for p in all_parts if p.get('sound_design', {}).get('mfx')]
        if mfx_parts:
            f.write(f"| Part | Track | MFX |\n")
            f.write(f"|------|-------|-----|\n")
            for part in mfx_parts:
                p_idx = part.get('part', '?')
                t_name = part.get('source_track_name', 'Unknown')
                mfx = part['sound_design']['mfx']
                f.write(f"| {p_idx} | {t_name} | {mfx} |\n")
            f.write(f"\n")
        else:
            f.write(f"No MFX applied.\n\n")

        # Drum FXM
        f.write(f"### Drum FXM\n\n")
        fxm_parts = [p for p in all_parts if p.get('sound_design', {}).get('drum_fxm')]
        if fxm_parts:
            f.write(f"| Part | Track | FXM |\n")
            f.write(f"|------|-------|-----|\n")
            for part in fxm_parts:
                p_idx = part.get('part', '?')
                t_name = part.get('source_track_name', 'Unknown')
                fxm = part['sound_design']['drum_fxm']
                f.write(f"| {p_idx} | {t_name} | {fxm} |\n")
            f.write(f"\n")
        else:
            f.write(f"No Drum FXM applied.\n\n")

        # === Reverb & Delay Sends ===
        f.write(f"## Reverb & Delay Sends\n\n")
        f.write(f"| Track Pattern | Reverb Send | Reverb Category | Delay Send |\n")
        f.write(f"|---------------|-------------|-----------------|------------|\n")
        seen = set()
        for match, category, rev_send, dly_send in stem_send_map:
            if match not in seen:
                seen.add(match)
                cat_str = category or '—'
                f.write(f"| {match} | {rev_send:.2f} | {cat_str} | {dly_send:.2f} |\n")
        f.write(f"\n")

        # === Layer Processing ===
        f.write(f"## Layer Processing\n\n")
        if layer_presets:
            f.write(f"| Layer | LUFS Target | Comp Ratio | Trim dB | EQ Bands |\n")
            f.write(f"|-------|-------------|------------|---------|----------|\n")
            for name, preset in layer_presets.items():
                lufs = preset.get('lufs_target', '—')
                ratio = preset.get('comp_ratio', '—')
                trim = preset.get('trim_db', 0.0)
                eq_bands = preset.get('eq', [])
                eq_str = ', '.join([
                    f"{'HP' if b['type']=='highpass' else 'LP' if b['type']=='lowpass' else '+' if b.get('gain_db',0)>0 else ''}{abs(b.get('gain_db',0))}@{b['freq']}" if b['type'] == 'bell' else f"{b['type'].upper()}{b['freq']}"
                    for b in eq_bands
                ]) if eq_bands else '—'
                f.write(f"| {name} | {lufs} | {ratio} | {trim:+.1f} | {eq_str} |\n")
            f.write(f"\n")

        # === Post-Processing Chain ===
        f.write(f"## Post-Processing Chain\n\n")

        f.write(f"### Per-Stem Processing\n\n")
        f.write(f"- **Gain staging**: LUFS targets per role")
        if lufs_targets:
            parts = [f"{k} {v}" for k, v in lufs_targets.items() if k != 'default']
            f.write(f" ({', '.join(parts)})")
        f.write(f"\n")
        f.write(f"- **Optimizer**: Nelder-Mead over soft_clip_ceiling + limiter_ceiling (5 evals for drums, 15 for melodic)\n")
        if drum_clip:
            f.write(f"- **Dynamic soft clip** (drum stems): {drum_clip.get('stem_headroom_db', 5)} dB headroom, {drum_clip.get('block_ms', 25)}ms blocks\n")
        f.write(f"\n")

        f.write(f"### Bus Processing\n\n")
        if drum_parallel:
            f.write(f"- **Drum bus parallel compression**: {drum_parallel.get('ratio', 10):.0f}:1 ratio, {drum_parallel.get('attack_ms', 2):.0f}ms attack, {drum_parallel.get('release_ms', 40):.0f}ms release, {drum_parallel.get('blend', 0.5)*100:.0f}% blend\n")
        f.write(f"- **Drum bus transient shaping**: +4 dB boost\n")
        f.write(f"- **Drum bus presence EQ**: 3.5 kHz +2.5 dB (Q=1.0), 70 Hz +1.5 dB (Q=1.0)\n")
        if drum_clip:
            f.write(f"- **Drum bus dynamic soft clip**: {drum_clip.get('bus_headroom_db', 4)} dB headroom, {drum_clip.get('block_ms', 25)}ms blocks\n")
        f.write(f"- **Bass-drums unmasking**: -2 dB cut at 50-120 Hz\n")
        if kick_sc:
            f.write(f"- **Kick-bass sidechain**: {kick_sc.get('depth_db', 3)} dB depth, {kick_sc.get('release_ms', 20)}ms release, threshold {kick_sc.get('threshold_db', -30)} dBFS, {kick_sc.get('freq_range', (40,120))[0]}-{kick_sc.get('freq_range', (40,120))[1]} Hz\n")
        f.write(f"- **Melody-drums unmasking**: -1.5 dB cut at 2200-3200 Hz\n")
        f.write(f"- **Melody-bass unmasking**: -1 dB cut at 120-240 Hz\n")
        f.write(f"- **Bus peak protection**: -0.1 dBFS ceiling\n")
        f.write(f"\n")

        # === Automation Effects ===
        f.write(f"## Automation Effects\n\n")
        if auto_plan and auto_plan.get('events'):
            f.write(f"| Section | Bars | Bus | Effect | Onset |\n")
            f.write(f"|---------|------|-----|--------|-------|\n")
            for event in auto_plan['events']:
                name = event.get('name', '?')
                start = event.get('start_bar', 0)
                end = event.get('end_bar', 0)
                onset = event.get('onset', 'instant')
                effects = event.get('effects', {})
                for bus, effect in effects.items():
                    f.write(f"| {name} | {start}-{end} | {bus} | {effect} | {onset} |\n")
            f.write(f"\n")

            f.write(f"### Onset Types\n\n")
            f.write(f"| Onset | Behavior |\n")
            f.write(f"|-------|----------|\n")
            f.write(f"| `instant` | Full intensity from beat 1 |\n")
            f.write(f"| `quarter_note` | Builds over 1 beat |\n")
            f.write(f"| `rapid` | Builds in first 20% of segment |\n")
            f.write(f"| `two_note` | Builds over 2 beats (pre-chorus) |\n")
            f.write(f"\n")
        else:
            f.write(f"No automation plan logged.\n\n")

        # === Effect Depths ===
        f.write(f"## Effect Depth Settings\n\n")
        if effect_depth:
            f.write(f"| Parameter | Value |\n")
            f.write(f"|-----------|-------|\n")
            for k, v in effect_depth.items():
                f.write(f"| {k} | {v} |\n")
            f.write(f"\n")

    print(f"  Documentation: {doc_path}")
    print(f"\n  PROJECT COMPLETE: {project_dir}")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remove_duplicate_final_masters(mastered_dir: str) -> tuple:
    """Remove parent mastered WAVs only when final/ has an identical copy."""
    final_dir = os.path.join(mastered_dir, "final")
    if not os.path.isdir(mastered_dir) or not os.path.isdir(final_dir):
        return [], []

    removed = []
    skipped = []
    for name in sorted(os.listdir(final_dir)):
        if not name.lower().endswith(".wav"):
            continue
        parent_path = os.path.join(mastered_dir, name)
        final_path = os.path.join(final_dir, name)
        if not os.path.isfile(parent_path) or not os.path.isfile(final_path):
            continue
        parent_size = os.path.getsize(parent_path)
        final_size = os.path.getsize(final_path)
        if parent_size != final_size:
            skipped.append((name, "size differs"))
            continue
        if _sha256_file(parent_path) != _sha256_file(final_path):
            skipped.append((name, "hash differs"))
            continue
        os.remove(parent_path)
        removed.append((name, parent_size))
    return removed, skipped


_CLEANUP_AUDIO_EXTENSIONS = (".wav", ".aif", ".aiff", ".flac")


def _is_audio_file(path: str) -> bool:
    return path.lower().endswith(_CLEANUP_AUDIO_EXTENSIONS)


def _safe_remove_empty_dirs(start_dir: str, stop_dir: str):
    """Remove empty directories from start_dir up to, but not including, stop_dir."""
    current = start_dir
    while current and os.path.abspath(current) != os.path.abspath(stop_dir):
        try:
            os.rmdir(current)
        except OSError:
            break
        current = os.path.dirname(current)


def _collect_cleanup_audio_candidates(mastered_dir: str, recordings_dir: str) -> list:
    """Return audio files that are intermediate under the current retention policy."""
    candidates = []

    def add(path: str, category: str, reason: str):
        if os.path.isfile(path) and _is_audio_file(path):
            candidates.append({
                "path": path,
                "category": category,
                "reason": reason,
                "size_bytes": os.path.getsize(path),
            })

    if os.path.isdir(mastered_dir):
        for subdir, category, reason in [
            ("processed", "processed_stem", "per-stem processed intermediate"),
            ("pristine_processed", "pristine_processed_stem", "duplicate pristine processing intermediate"),
            ("automated", "pre_section_automated_bus", "superseded by section_automated_buses"),
            ("sidechain_keys", "detector_key_bus", "detector-only sidechain key"),
            ("golden_post", "golden_post_audio", "candidate/intermediate golden-post render"),
        ]:
            root_dir = os.path.join(mastered_dir, subdir)
            if os.path.isdir(root_dir):
                for root, _, files in os.walk(root_dir):
                    for name in files:
                        add(os.path.join(root, name), category, reason)

        for name in os.listdir(mastered_dir):
            path = os.path.join(mastered_dir, name)
            if not os.path.isfile(path) or not _is_audio_file(path):
                continue
            if name.endswith("_mix.wav"):
                add(path, "temporary_mix", "temporary mix sum")
            elif name == "raw_summed_returns.wav":
                add(path, "raw_return_sum", "uncalibrated wet-return sum superseded by section_calibrated_returns.wav")

    if os.path.isdir(recordings_dir):
        for name in os.listdir(recordings_dir):
            path = os.path.join(recordings_dir, name)
            if name.endswith("_pass.wav") and "_pass0" in name:
                add(path, "raw_multichannel_pass", "raw multichannel pass capture superseded by split stem recordings")

    return candidates


def cleanup_intermediates(song_name: str, output_dir: str = None, recordings_dir: str = None,
                          dry_run: bool = False):
    """Delete intermediate WAV files after pipeline completion.
    
    PRESERVED (never deleted):
    - recordings/*.wav — individual stem WAVs (e.g. songname_pass01_part01_usb01-02_Bass.wav)
    - recordings/*.json — manifest files with patch/sound design data
    - midi/ — MIDI files
    - mastered/final/*.wav — final streaming master variants
    - mastered/buses/ — dry bus sums for review
    - mastered/section_automated_buses/ — final-stage section-aware bus renders
    - mastered/reverb_returns/ — reverb return audio for review
    - mastered/delay_returns/ — delay return audio for review
    - mastered/section_calibrated_returns.wav — calibrated combined wet return
    - mastered/*.json / mastered/*.jsonl / mastered/*.md — production diagnostics
    - sample_pack_build/ — sample pack files
    
    DELETED:
    - mastered/processed/ — per-stem processed intermediates
    - mastered/pristine_processed/ — duplicate processing
    - mastered/automated/ — pre-section automated buses superseded by section_automated_buses/
    - mastered/sidechain_keys/ — detector-only sidechain keys
    - mastered/raw_summed_returns.wav — uncalibrated return sum
    - mastered/*_mix.wav — temporary mix sums
    - mastered/*.wav duplicated exactly in mastered/final/
    - mastered/golden_post/**/*.wav — post-audio intermediate renders
    - recordings/*_pass*_pass.wav — raw multi-pass recordings only
    """
    project_dir = project_dir_for_song(song_name)
    mastered_dir = output_dir or os.path.join(project_dir, "mastered")
    recordings_dir = recordings_dir or os.path.join(project_dir, "recordings")

    deleted = []
    freed_bytes = 0
    skipped_duplicate_masters = []

    candidates = _collect_cleanup_audio_candidates(mastered_dir, recordings_dir)
    for item in candidates:
        path = item["path"]
        freed_bytes += item["size_bytes"]
        rel_base = recordings_dir if path.startswith(recordings_dir) else mastered_dir
        rel_path = os.path.relpath(path, rel_base)
        deleted.append({
            "path": f"recordings/{rel_path}" if rel_base == recordings_dir else rel_path,
            "category": item["category"],
            "reason": item["reason"],
            "size_bytes": item["size_bytes"],
        })
        if not dry_run:
            os.remove(path)
            _safe_remove_empty_dirs(os.path.dirname(path), mastered_dir if rel_base == mastered_dir else recordings_dir)

    duplicate_masters = []
    if not dry_run:
        duplicate_masters, skipped_duplicate_masters = remove_duplicate_final_masters(mastered_dir)
        for name, size_bytes in duplicate_masters:
            freed_bytes += size_bytes
            deleted.append({
                "path": f"duplicate_master/{name}",
                "category": "duplicate_parent_master",
                "reason": "identical copy exists in mastered/final",
                "size_bytes": size_bytes,
            })
        for name, reason in skipped_duplicate_masters:
            print(f"  Preserved parent master {name}: final copy {reason}")

    cleanup_manifest = {
        "song_name": song_name,
        "dry_run": dry_run,
        "freed_bytes": 0 if dry_run else freed_bytes,
        "candidate_bytes": freed_bytes,
        "deleted": deleted,
        "skipped_duplicate_masters": [
            {"name": name, "reason": reason}
            for name, reason in skipped_duplicate_masters
        ],
        "preserved_policy": [
            "recordings/*_part*_usb*.wav original split stems",
            "mastered/final/*.wav streaming final masters",
            "mastered/buses/*.wav dry buses",
            "mastered/section_automated_buses/*.wav final-stage section-aware buses",
            "mastered/reverb_returns/*.wav and mastered/delay_returns/*.wav wet returns",
            "mastered/section_calibrated_returns.wav calibrated combined wet return",
            "mastered/final/*_streaming_master.wav generic final streaming master",
            f"mastered/{JUNE1_ANALYSIS_FILE} June 1 pipeline analysis",
            "mastered/corrected_mix.wav June 1 corrected mix",
            "mastered/corrected_master.wav June 1 corrected master",
        ],
    }
    manifest_path = os.path.join(mastered_dir, "cleanup_manifest.json")
    if os.path.isdir(mastered_dir) and not dry_run:
        with open(manifest_path, "w") as f:
            json.dump(cleanup_manifest, f, indent=2)

    if deleted:
        action = "Would clean up" if dry_run else "Cleaned up"
        bytes_label = "candidate" if dry_run else "freed"
        print(f"  {action} {len(deleted)} audio items, {bytes_label} {freed_bytes / 1024 / 1024:.1f} MB")
        if not dry_run:
            print(f"  Cleanup manifest: {manifest_path}")
    else:
        print("  No intermediate files to clean up")

    return cleanup_manifest


def write_artifact_manifest(song_name: str, midi_path: str = None, master_path: str = None,
                            stems: dict = None, metadata: dict = None,
                            output_dir: str = None) -> dict:
    """Write a compact manifest for production verification and handoff."""
    project_dir = project_dir_for_song(song_name)
    mastered_dir = output_dir or os.path.join(project_dir, "mastered")
    recordings_dir = os.path.join(project_dir, "recordings")
    june1_analysis_path = os.path.join(mastered_dir, JUNE1_ANALYSIS_FILE)
    june1_analysis = {}
    if os.path.exists(june1_analysis_path):
        try:
            with open(june1_analysis_path) as ifp:
                june1_analysis = json.load(ifp)
        except Exception:
            june1_analysis = {}

    def rel(path):
        if not path:
            return None
        try:
            return os.path.relpath(path, project_dir)
        except Exception:
            return str(path)

    final_dir = os.path.join(mastered_dir, "final")
    final_masters = []
    if os.path.isdir(final_dir):
        final_masters = sorted(
            os.path.join("mastered", "final", name)
            for name in os.listdir(final_dir)
            if name.lower().endswith(".wav")
        )
    manifest = {
        "pipeline": PIPELINE_NAME,
        "song_name": song_name,
        "bpm": float(metadata.get("bpm", 0) if metadata and metadata.get("bpm") else 0),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_dir": project_dir,
        "midi_path": rel(midi_path),
        "master_path": rel(master_path),
        "stems": sorted(rel(path) for path in (stems or {}).values()),
        "final_masters": final_masters,
        "preferred_mastering": {
            "analysis": rel(june1_analysis_path),
            "corrected_master": rel(os.path.join(mastered_dir, "corrected_master.wav")),
            "corrected_mix": rel(os.path.join(mastered_dir, "corrected_mix.wav")),
        },
        "technical_reports": {
            "june1_analysis": rel(june1_analysis_path),
        },
        "hi_hat_patch_policy": {
            "blocked_patch_programs": [63, 64, 65, 66, 67, 68, 71, 72, 73, 74],
            "special_articulation_patch_programs": [69, 70],
            "special_articulation_note_offset_range": [0, 0],
            "other_allowed_drum_kits_transpose": False,
        },
        "expected_dirs": {
            "recordings": os.path.isdir(recordings_dir),
            "mastered": os.path.isdir(mastered_dir),
            "final": os.path.isdir(final_dir),
        },
    }
    manifest_path = os.path.join(mastered_dir, MAY25_MANIFEST_FILE)
    legacy_manifest_path = os.path.join(mastered_dir, MAY19_MANIFEST_FILE)
    if os.path.isdir(mastered_dir):
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(legacy_manifest_path, manifest)
        print(f"  Artifact manifest: {manifest_path}")
    return manifest


def verify_artifacts(song_name: str = None, output_dir: str = None) -> bool:
    """Verify the expected June 1 production output surface without modifying audio."""
    mastered_dir = output_dir
    if mastered_dir is None:
        if not song_name:
            raise ValueError("--song-name or --output-dir is required for --verify-only")
        mastered_dir = mastered_dir_for_song(song_name)
    if os.path.basename(os.path.normpath(mastered_dir)) != "mastered" and os.path.isdir(os.path.join(mastered_dir, "mastered")):
        mastered_dir = os.path.join(mastered_dir, "mastered")

    final_dir = os.path.join(mastered_dir, "final")
    checks = [
        ("mastered_dir", mastered_dir, os.path.isdir(mastered_dir)),
        ("corrected_master", os.path.join(mastered_dir, "corrected_master.wav"), os.path.exists(os.path.join(mastered_dir, "corrected_master.wav"))),
        ("corrected_mix", os.path.join(mastered_dir, "corrected_mix.wav"), os.path.exists(os.path.join(mastered_dir, "corrected_mix.wav"))),
        ("june1_analysis", os.path.join(mastered_dir, JUNE1_ANALYSIS_FILE), os.path.exists(os.path.join(mastered_dir, JUNE1_ANALYSIS_FILE))),
        ("artifact_manifest", os.path.join(mastered_dir, MAY25_MANIFEST_FILE), os.path.exists(os.path.join(mastered_dir, MAY25_MANIFEST_FILE))),
        ("cleanup_manifest", os.path.join(mastered_dir, "cleanup_manifest.json"), os.path.exists(os.path.join(mastered_dir, "cleanup_manifest.json"))),
    ]

    failed = []
    for label, path, ok in checks:
        status = "OK" if ok else "MISSING"
        print(f"  {status}: {label} -> {path}")
        if not ok:
            failed.append(label)
    if failed:
        print(f"  Verification failed: {', '.join(failed)}")
        return False
    print("  Verification passed.")
    return True


def run_full_pipeline(sample_pack: bool = False,
                      workers: int = 8,
                      seed: int = None,
                      external_output_copy: bool = True,
                      external_output_root: str = None):
    """Run the complete pipeline: MIDI → Record → June 1 Mix/Master → Sample Pack."""
    print("=" * 70)
    print("FULL PIPELINE — MIDI → RECORD → JUNE 1 SKILL-BASED PROCESS")
    print("=" * 70)

    # Step 1: Generate MIDI
    midi_path, song_name, bpm, metadata = step_generate_midi()

    # Step 2: Record
    stems = step_record(midi_path, song_name, metadata)
    if not stems:
        print("  No stems recorded. Cannot proceed to production.")
        return

    # Step 3: June 1 skill-based mix/master
    recordings_dir = os.path.join(project_dir_for_song(song_name), "recordings")
    master_path = step_june1_mix_master(
        recordings_dir=recordings_dir,
        song_name=song_name, bpm=bpm,
        workers=workers, seed=seed,
    )

    if not master_path:
        print("  No master produced. Cannot proceed to consolidation.")
        return

    # Step 5: Sample Pack (optional)
    sample_dir = None
    if sample_pack:
        sample_dir = step_sample_pack(midi_path, song_name,
                                      recordings_dir=recordings_dir)

    # Step 6: Consolidate
    step_consolidate(song_name, midi_path, master_path, stems, metadata)

    # Step 7: Cleanup intermediate files
    cleanup_intermediates(song_name)
    prune_streaming_final_surface(mastered_dir_for_song(song_name), song_name, bpm)
    write_artifact_manifest(song_name, midi_path, master_path, stems, metadata)
    verified = verify_artifacts(song_name=song_name)
    if verified:
        export_completed_song_folder(
            song_name,
            destination_root=external_output_root,
            enabled=external_output_copy,
        )

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE: {song_name}")
    print(f"Master: {master_path}")
    if sample_dir:
        print(f"Sample Pack: {sample_dir}")
    print("=" * 70)


def run_production_only(stems_dir: str, song_name: str, bpm: float,
                        output_dir: str = None, sample_pack: bool = False,
                        midi_file: str = None,
                        exclude: list = None,
                        workers: int = 8,
                        seed: int = None,
                        external_output_copy: bool = True,
                        external_output_root: str = None):
    """Run production on existing stems, with optional sample pack."""
    recordings = Path(stems_dir)
    if not recordings.is_dir():
        print(f"Error: recordings dir not found: {stems_dir}")
        sys.exit(1)
    wavs = sorted(recordings.glob("*.wav"))
    if exclude:
        wavs = [w for w in wavs if not any(ex.lower() in w.name.lower() for ex in exclude)]
    if not wavs:
        print(f"Error: No WAV files found in {stems_dir}")
        sys.exit(1)

    print(f"Found {len(wavs)} stems in {stems_dir}")

    master_path = step_june1_mix_master(
        recordings_dir=stems_dir,
        song_name=song_name, bpm=bpm,
        output_dir=output_dir,
        workers=workers, seed=seed,
    )

    # Sample pack (optional) — always generates fresh MIDI via midi_orchestrator
    sample_dir = None
    if sample_pack:
        print("\n  Generating fresh MIDI for sample pack...")
        try:
            fresh_midi, _, _, _ = step_generate_midi()
            sample_dir = step_sample_pack(fresh_midi, song_name,
                                          recordings_dir=stems_dir)
        except Exception as e:
            print(f"  Sample pack failed: {e}")

    if master_path:
        # Cleanup intermediate files
        cleanup_intermediates(song_name, output_dir=output_dir, recordings_dir=stems_dir)
        mastered_dir = output_dir or mastered_dir_for_song(song_name)
        prune_streaming_final_surface(mastered_dir, song_name, bpm)
        write_artifact_manifest(
            song_name,
            midi_path=midi_file or find_midi_file(song_name),
            master_path=master_path,
            stems={w.name: str(w) for w in wavs},
            metadata={"bpm": bpm},
            output_dir=output_dir,
        )
        verified = verify_artifacts(
            song_name=song_name,
            output_dir=output_dir,
        )
        if verified:
            export_completed_song_folder(
                song_name,
                output_dir=output_dir,
                destination_root=external_output_root,
                enabled=external_output_copy,
            )

        print(f"\n{'='*70}")
        print(f"PRODUCTION COMPLETE: {song_name}")
        print(f"Master: {master_path}")
        if sample_dir:
            print(f"Sample Pack: {sample_dir}")
        print(f"{'='*70}")
    else:
        print("Production failed — no master produced.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="June 1 Pipeline — MIDI → Recording → Skill-Based Mix/Master"
    )

    # Mode selection
    parser.add_argument("--full", action="store_true", default=False,
                        help="Run full pipeline (MIDI → record → produce)")
    parser.add_argument("--batch", type=int, default=None,
                        help="Generate N tracks from scratch (MIDI → record → produce)")
    parser.add_argument("--midi-only", action="store_true",
                        help="Generate MIDI only")
    parser.add_argument("--produce-only", action="store_true",
                        help="Run production only on existing stems")
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify expected output artifacts without rendering audio")

    # Production options
    parser.add_argument("--stems", type=str, default=None,
                        help="Directory containing stem WAV files")
    parser.add_argument("--song-name", type=str, default=None,
                        help="Song name for output files")
    parser.add_argument("--bpm", type=float, default=None,
                        help="Beats per minute")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--exclude-stem", type=str, action='append', default=[],
                        help="Exclude stems matching this substring (can be used multiple times)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel stem workers (default: 8)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible panning and FX (default: random)")
    parser.add_argument("--no-external-output-copy", action="store_true",
                        help="Skip copying the completed song folder to the external RFLXN output root")
    parser.add_argument("--external-output-root", type=str, default=str(DEFAULT_EXTERNAL_OUTPUT_ROOT),
                        help="Destination root for completed song folder copies")

    # Sample pack options
    parser.add_argument("--sample-pack", action="store_true",
                        help="Generate companion sample pack (requires Fantom connected)")
    parser.add_argument("--midi-file", type=str, default=None,
                        help="MIDI file for sample pack generation (auto-detected if not provided)")

    args = parser.parse_args()

    # Default: run full pipeline with sample pack if no mode specified
    no_mode_selected = not any([args.full, args.batch, args.midi_only, args.produce_only, args.verify_only,
                                args.stems, args.sample_pack])

    if args.batch:
        successes = []
        failures = []
        for i in range(args.batch):
            print(f"\n{'='*70}")
            print(f"BATCH TRACK {i+1}/{args.batch}")
            print(f"{'='*70}")
            try:
                run_full_pipeline(
                    sample_pack=args.sample_pack,
                    workers=args.workers,
                    seed=args.seed,
                    external_output_copy=not args.no_external_output_copy,
                    external_output_root=args.external_output_root,
                )
                successes.append(i+1)
            except Exception as e:
                print(f"  Track {i+1} failed: {e}")
                failures.append(i+1)
                continue
        print(f"\n{'='*70}")
        print(f"BATCH COMPLETE: {len(successes)} succeeded, {len(failures)} failed")
        if failures:
            print(f"  Failed tracks: {failures}")
        print(f"{'='*70}")
    elif args.full or no_mode_selected:
        run_full_pipeline(
            sample_pack=True,
            workers=args.workers,
            seed=args.seed,
            external_output_copy=not args.no_external_output_copy,
            external_output_root=args.external_output_root,
        )
    elif args.midi_only:
        midi_path, song_name, bpm, _ = step_generate_midi()
        print(f"\nMIDI generated: {midi_path}")
    elif args.verify_only:
        ok = verify_artifacts(
            song_name=args.song_name,
            output_dir=args.output_dir,
        )
        sys.exit(0 if ok else 1)
    elif args.sample_pack:
        # Sample pack only mode
        if not args.song_name:
            print("Error: --song-name required for sample pack generation")
            sys.exit(1)
        midi_path = args.midi_file or find_midi_file(args.song_name)
        if not midi_path:
            print(f"Error: MIDI file not found for '{args.song_name}'. Use --midi-file to specify.")
            sys.exit(1)
        sample_dir = step_sample_pack(midi_path, args.song_name)
        print(f"\nSample pack: {sample_dir}")
    elif args.produce_only or args.stems:
        if not args.stems:
            print("Error: --stems required for production mode")
            sys.exit(1)
        if not args.song_name:
            args.song_name = os.path.basename(os.path.normpath(args.stems))
        if not args.bpm:
            midi_path = find_midi_file(args.song_name)
            if midi_path:
                args.bpm = get_bpm_from_midi(midi_path)
                print(f"BPM from MIDI: {args.bpm}")
            else:
                args.bpm = 90.0
                print(f"BPM defaulting to: {args.bpm}")
        run_production_only(
            args.stems, args.song_name, args.bpm, args.output_dir,
            sample_pack=args.sample_pack, midi_file=args.midi_file,
            exclude=args.exclude_stem,
            workers=args.workers, seed=args.seed,
            external_output_copy=not args.no_external_output_copy,
            external_output_root=args.external_output_root,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
