import os
import sys
import time
import copy
import random
import argparse
from datetime import datetime
from glob import glob

# Add project roots to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SCRIPTS_ROOT)
V10_DIR = os.path.join(SCRIPTS_ROOT, "v10_refactored")

sys.path.append(PROJECT_ROOT)
sys.path.append(SCRIPTS_ROOT)
sys.path.append(V10_DIR)
sys.path.append(SCRIPT_DIR)

# Import our modular pipeline components
import audio_recorder
from fantom_midi_control import FantomController
from audio_recorder import AudioRecorder, MultiPassOrchestrator
from post_production import ProductionEngine
from generate_sample_stems import generate_sample_pack
import shutil
import json

# Import the MIDI generator
import orchestrator as midi_gen

# Capture original patch pool at import time before any run can mutate it
ORIGINAL_MODEL_PATCHES = copy.deepcopy(audio_recorder._MODEL_EXPANSION_PATCHES)

EXPANSIONS = {
    'JUPITER-8': 'JP8 ',
    'SH-101':    'SH-101 ',
    'JX-8P':     'JX-8P ',
    'JUNO-106':  'JUNO-106 ',
}


def set_expansion_constraint(expansion_name: str):
    """Limit the synth patch pool to a specific Roland expansion."""
    if expansion_name not in EXPANSIONS:
        print(f"Warning: Unknown expansion {expansion_name}. Resetting to all patches.")
        audio_recorder._MODEL_EXPANSION_PATCHES = copy.deepcopy(ORIGINAL_MODEL_PATCHES)
        return

    prefix = EXPANSIONS[expansion_name]
    new_pool = {}

    all_expansion_patches = []
    for patches in ORIGINAL_MODEL_PATCHES.values():
        all_expansion_patches.extend([p for p in patches if p['name'].startswith(prefix)])

    for role, patches in ORIGINAL_MODEL_PATCHES.items():
        filtered = [p for p in patches if p['name'].startswith(prefix)]
        if filtered:
            new_pool[role] = filtered
        else:
            # Smart fallback: if a role (e.g. brass, bell) has no patches in this expansion,
            # borrow from poly/lead/bass within the same expansion before using anything else.
            fallback_roles = ['poly', 'lead', 'bass']
            found_fallback = False
            for f_role in fallback_roles:
                f_patches = [p for p in ORIGINAL_MODEL_PATCHES.get(f_role, []) if p['name'].startswith(prefix)]
                if f_patches:
                    new_pool[role] = f_patches
                    found_fallback = True
                    break
            if not found_fallback:
                new_pool[role] = all_expansion_patches if all_expansion_patches else patches

    print(f"Enforcing patch constraint: {expansion_name} (Prefix: '{prefix}')")
    audio_recorder._MODEL_EXPANSION_PATCHES = new_pool


def generate_sound_design_report(song_name: str, out_dir: str, recordings_dir: str = None):
    """Generate a Markdown report of the patches and modulations used per part."""
    report_path = os.path.join(out_dir, f"{song_name}_sound_design.md")

    manifest_roots = []
    if recordings_dir:
        manifest_roots.append(recordings_dir)
    manifest_roots.extend([
        os.path.join(get_song_project_dir(song_name), "recordings"),
        os.path.join(PROJECT_ROOT, "output", "recordings"),
    ])
    manifest_files = []
    for root in manifest_roots:
        manifest_files.extend(glob(os.path.join(root, f"{song_name}_pass*_manifest.json")))
    manifest_files = sorted(set(manifest_files))
    if not manifest_files:
        return

    lines = [f"# Sound Design Report: {song_name}", "",
             f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
             "## Part Assignments & Modulations", ""]

    for m_file in manifest_files:
        try:
            with open(m_file, 'r') as f:
                data = json.load(f)

            lines.append(f"### {data.get('batch', 'Unknown Pass').upper()}")
            lines.append("| Part | Track Name | Patch Name | Sound Design & LFO Matrix | MFX / Roland Mods |")
            lines.append("| :--- | :--- | :--- | :--- | :--- |")

            for part in data.get('parts', []):
                p_idx = part.get('part')
                t_name = part.get('recorded_track_name')
                patch = part.get('patch', {}).get('name', 'Unknown')
                design = part.get('sound_design', {})
                lfo = design.get('lfo_matrix', 'None')
                mfx = design.get('mfx_mod', 'None')
                lines.append(f"| {p_idx} | {t_name} | {patch} | {lfo} | {mfx} |")
            lines.append("")
        except Exception as e:
            print(f"  Warning: Could not parse manifest {m_file}: {e}")

    try:
        with open(report_path, 'w') as f:
            f.write("\n".join(lines))
        print(f"  ✓ Sound Design Report: {report_path}")
    except Exception as e:
        print(f"  Warning: Could not write sound design report: {e}")


def get_song_project_dir(song_name: str) -> str:
    """Return the per-song output folder used by all pipeline stages."""
    return os.path.join(PROJECT_ROOT, "output", song_name)


def prompt_interactive_options() -> tuple:
    """Interactively ask for expansion choice and track count. Returns (expansion, count)."""
    print("\nFANTOM PIPELINE — Setup")
    print("-" * 40)
    print("Constrain patches to a specific Roland expansion?")
    print("  0) No — use full patch pool (default)")
    for idx, name in enumerate(EXPANSIONS.keys(), start=1):
        print(f"  {idx}) {name}")

    expansion = None
    while True:
        raw = input(f"Select [0-{len(EXPANSIONS)}]: ").strip()
        if raw == "" or raw == "0":
            break
        if raw.isdigit() and 1 <= int(raw) <= len(EXPANSIONS):
            expansion = list(EXPANSIONS.keys())[int(raw) - 1]
            print(f"  → {expansion} selected")
            break
        print("  Invalid selection, try again.")

    count = 1
    while True:
        raw = input("How many tracks to generate? [1]: ").strip()
        if raw == "":
            break
        if raw.isdigit() and int(raw) >= 1:
            count = int(raw)
            break
        print("  Please enter a positive integer.")

    return expansion, count


def prompt_track_selection(midi_path: str) -> set:
    """
    Read track names from a MIDI file and let the user toggle tracks off.
    Returns a set of track name substrings to skip during recording.
    """
    import mido
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        print(f"  Warning: Could not read MIDI for track selection: {e}")
        return set()

    track_names = [t.name for t in mid.tracks if t.name and t.name != "__sync__"]
    if not track_names:
        return set()

    included = list(range(len(track_names)))  # indices of included tracks

    print("\nTRACK SELECTION — Uncheck any tracks to exclude from recording")
    print("-" * 60)
    print("All tracks included by default. Enter a number to toggle off.")
    print("Press Enter when done.\n")

    while True:
        for idx, name in enumerate(track_names):
            check = "✓" if idx in included else " "
            print(f"  [{check}] {idx + 1}  {name}")

        raw = input("\nToggle (number), or Enter to continue: ").strip()
        if raw == "":
            break
        if raw.isdigit():
            choice = int(raw) - 1
            if 0 <= choice < len(track_names):
                if choice in included:
                    included.remove(choice)
                    print(f"  → Excluded: {track_names[choice]}")
                else:
                    included.append(choice)
                    print(f"  → Included: {track_names[choice]}")
                continue
        print("  Invalid input.")

    skipped = {track_names[i] for i in range(len(track_names)) if i not in included}
    if skipped:
        print(f"\n  Skipping tracks: {', '.join(sorted(skipped))}")
    return skipped


def consolidate_project(song_name: str, latest_midi: str, master_wav: str,
                        stems: dict, sample_dir: str = None,
                        metadata: dict = None, out_dir: str = None):
    """Organize all song assets into a single project folder and generate documentation."""
    print(f"\n[STEP 5] Consolidating Project: {song_name}...")

    project_dir = get_song_project_dir(song_name)
    os.makedirs(project_dir, exist_ok=True)

    midi_dir = os.path.join(project_dir, "midi")
    os.makedirs(midi_dir, exist_ok=True)
    if latest_midi and os.path.exists(latest_midi):
        shutil.copy2(latest_midi, os.path.join(midi_dir, os.path.basename(latest_midi)))
        print(f"  ✓ Copied MIDI: {os.path.basename(latest_midi)}")

    mastered_dir = out_dir or os.path.join(PROJECT_ROOT, "output", "mastered")
    mastered_dest_dir = os.path.join(project_dir, "mastered")
    os.makedirs(mastered_dest_dir, exist_ok=True)
    master_files = glob(os.path.join(mastered_dir, f"{song_name}*.wav"))
    for f in master_files:
        dest = os.path.join(mastered_dest_dir, os.path.basename(f))
        if os.path.abspath(f) != os.path.abspath(dest):
            shutil.move(f, dest)
    print(f"  ✓ Organized {len(master_files)} mastered files in mastered/ folder")

    stems_dest_dir = os.path.join(project_dir, "stems")
    os.makedirs(stems_dest_dir, exist_ok=True)
    moved_stems = 0
    for stem_path in stems.values():
        if os.path.exists(stem_path):
            dest = os.path.join(stems_dest_dir, os.path.basename(stem_path))
            if os.path.abspath(stem_path) != os.path.abspath(dest):
                shutil.move(stem_path, dest)
            moved_stems += 1
    print(f"  ✓ Organized {moved_stems} stems in stems/ folder")

    if sample_dir and os.path.exists(sample_dir):
        sample_dest = os.path.join(project_dir, "sample_pack")
        sample_parent = os.path.dirname(os.path.abspath(sample_dir))
        if os.path.abspath(sample_dir) != os.path.abspath(sample_dest) and os.path.exists(sample_dest):
            shutil.rmtree(sample_dest)
        if os.path.abspath(sample_dir) != os.path.abspath(sample_dest):
            shutil.move(sample_dir, sample_dest)
            if os.path.basename(sample_parent) == "sample_pack_build" and os.path.isdir(sample_parent):
                try:
                    os.rmdir(sample_parent)
                except OSError:
                    pass
        print(f"  ✓ Organized sample pack in sample_pack/ folder")

    doc_path = os.path.join(project_dir, "DOCUMENTATION.md")
    manifest_roots = [
        os.path.join(project_dir, "recordings"),
        os.path.join(PROJECT_ROOT, "output", "recordings"),
    ]
    manifests = []
    for root in manifest_roots:
        manifests.extend(glob(os.path.join(root, f"{song_name}_pass*_manifest.json")))
    manifests = sorted(set(manifests))

    doc_lines = [
        f"# Project: {song_name}",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Musical Summary",
        f"- **Scale**: {metadata.get('scale', 'N/A') if metadata else 'N/A'}",
        f"- **Key**: {metadata.get('key', 'N/A') if metadata else 'N/A'}",
        f"- **BPM**: {metadata.get('bpm', 90) if metadata else '90'}",
        f"- **Armenian/Maqam**: {'Yes' if metadata and metadata.get('is_armenian') else 'No'}",
        "",
        "## Project Notes",
        "| Part | Track Name | Roland Patch | Sound Design & Modulations |",
        "| :--- | :--- | :--- | :--- |"
    ]

    for manifest_path in sorted(manifests):
        try:
            with open(manifest_path, "r") as f:
                data = json.load(f)
                for part in data.get("parts", []):
                    p_num = part.get("part")
                    p_name = part.get("recorded_track_name")
                    patch = part.get("patch", {}).get("name", "Unknown")
                    sd = part.get("sound_design", {})
                    sd_info = []
                    if sd.get("lfo_matrix"): sd_info.append(sd["lfo_matrix"])
                    if sd.get("mfx"): sd_info.append(f"MFX: {sd['mfx']}")
                    if sd.get("drum_fxm"): sd_info.append(f"Drum FXM: {sd['drum_fxm']}")
                    sd_str = " | ".join(sd_info) if sd_info else "Standard"
                    doc_lines.append(f"| {p_num} | {p_name} | **{patch}** | {sd_str} |")
        except Exception as e:
            print(f"  ! Error reading manifest {manifest_path}: {e}")

    with open(doc_path, "w") as f:
        f.write("\n".join(doc_lines))
    print(f"  ✓ Generated: DOCUMENTATION.md")
    print(f"\nPROJECT COMPLETE: {project_dir}")
    return project_dir


def load_existing_stems(song_name: str, recordings_dir: str = None) -> dict:
    """Scan recordings directory for stem files matching the song name."""
    stems = {}
    search_roots = []
    if recordings_dir:
        search_roots.append(recordings_dir)
    search_roots.extend([
        os.path.join(get_song_project_dir(song_name), "recordings"),
        os.path.join(get_song_project_dir(song_name), "stems"),
        os.path.join(PROJECT_ROOT, "output", "recordings"),
        "output/recordings",
    ])
    patterns = [
        os.path.join(root, pattern)
        for root in search_roots
        for pattern in [f"{song_name}_pass*_part*.wav", f"{song_name}_pass*.wav"]
    ]

    stem_files = []
    for pattern in patterns:
        stem_files.extend(sorted(glob(pattern)))

    if not stem_files:
        print(f"No stems found matching: {patterns[0] if patterns else song_name}")
        return stems

    for path in stem_files:
        basename = os.path.basename(path)
        if basename.endswith("_pass.wav"):
            continue
        key = basename.replace(".wav", "")
        stems[key] = path

    print(f"Found {len(stems)} existing stems for {song_name}")
    return stems


def find_midi_file(song_name: str) -> str:
    """Search for the MIDI file in common output directories."""
    search_dirs = ["output", "scripts/output", "../output", "../../output"]
    for dir_path in search_dirs:
        candidates = [
            os.path.join(dir_path, f"{song_name}.mid"),
            os.path.join(dir_path, f"{song_name}.midi"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        matches = sorted(glob(os.path.join(dir_path, f"{song_name}*.mid*")))
        if matches:
            return matches[-1]
    return None


def get_bpm_from_midi(midi_path: str) -> float:
    """Extract BPM from a MIDI file."""
    import mido
    if not os.path.exists(midi_path):
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")
    mid = mido.MidiFile(midi_path)
    tempo_us = 500000
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo_us = msg.tempo
                break
    return round(60_000_000 / tempo_us, 2)


def run_full_pipeline(resume_from_step: int = 1, song_name: str = None,
                      target_expansion: str = None, target_bpm: float = None,
                      exclude: list = None, interactive: bool = False):
    print("=" * 70)
    print("AUTONOMOUS FANTOM AUDIO PIPELINE")
    if target_expansion:
        print(f"EP TARGET: {target_expansion}")
    print("=" * 70)

    stems = {}
    bpm = target_bpm or 90.0
    latest_midi = None
    metadata = {}
    skip_tracks = set()

    if target_expansion:
        set_expansion_constraint(target_expansion)

    # STEP 1: Generate MIDI
    if resume_from_step <= 1:
        print(f"\n[STEP 1] Generating MIDI Sequence...")
        generated = midi_gen.main()
        if isinstance(generated, tuple):
            latest_midi = generated[0]
            metadata = generated[1]
        else:
            latest_midi = generated
            metadata = {}

        song_name = os.path.basename(latest_midi).replace(".mid", "")
        print(f"Generated MIDI: {latest_midi}")
        bpm = metadata.get('bpm', get_bpm_from_midi(latest_midi))
        print(f"BPM: {bpm}")

        # Interactive track selection after MIDI is generated
        if interactive:
            skip_tracks = prompt_track_selection(latest_midi)
    else:
        if not song_name:
            print("Error: --song-name required when resuming from step 2+")
            return
        latest_midi = find_midi_file(song_name)
        if not latest_midi:
            print(f"Error: MIDI file for '{song_name}' not found in output directories")
            return
        bpm = get_bpm_from_midi(latest_midi)
        print(f"Resuming with song: {song_name}, BPM: {bpm}")
        print(f"MIDI file: {latest_midi}")
        metadata = {'bpm': bpm, 'key': 'Unknown', 'scale': 'Unknown'}

    project_dir = get_song_project_dir(song_name)
    recordings_dir = os.path.join(project_dir, "recordings")
    mastered_dir = os.path.join(project_dir, "mastered")
    os.makedirs(recordings_dir, exist_ok=True)
    os.makedirs(mastered_dir, exist_ok=True)

    # STEP 2: Record Multi-Pass
    if resume_from_step <= 2:
        print("\n[STEP 2] Recording Phased Audio & Exporting Stems...")
        controller = FantomController()
        recorder = AudioRecorder(device_index=9, output_dir=recordings_dir)
        orch = MultiPassOrchestrator(recorder, controller, target_expansion=target_expansion)

        if not controller.output:
            print("Hardware connection failed. Aborting.")
            return

        stems = orch.run_multi_pass(latest_midi, song_name, skip_tracks=skip_tracks or None,
                                    metadata=metadata)
    else:
        print(f"\n[SKIPPING STEP 2] Loading existing stems for {song_name}...")
        stems = load_existing_stems(song_name, recordings_dir=recordings_dir)
        if not stems:
            print("No stems found. Cannot proceed to Step 3.")
            return

    # STEP 3: Advanced Production
    print("\n[STEP 3] Automated Mixing & Mastering...")
    out_dir = mastered_dir

    production = ProductionEngine(output_dir=out_dir)

    # Apply user exclusions (CLI --exclude flag or interactively skipped tracks)
    excluded_patterns = list(exclude or []) + [t for t in skip_tracks]
    if excluded_patterns:
        filtered_stems = {}
        for k, v in stems.items():
            if any(pattern in v for pattern in excluded_patterns):
                print(f"  Excluding stem: {os.path.basename(v)}")
                continue
            filtered_stems[k] = v
        stems = filtered_stems

    master_wav = production.process_full_mix(stems, song_name, bpm=bpm)
    production.process_pristine_mix(stems, song_name, bpm=bpm)

    generate_sound_design_report(song_name, out_dir, recordings_dir=recordings_dir)

    # STEP 4: Sample Pack Generation
    print("\n[STEP 4] Generating Companion Sample Pack...")
    sample_dir = None
    try:
        sample_dir = generate_sample_pack(
            latest_midi,
            output_root=os.path.join(project_dir, "sample_pack_build"),
            device_index=9,
        )
        print(f"Sample Pack: {sample_dir}")
    except Exception as e:
        print(f"Sample Generation Failed: {e}")

    # STEP 5: Project Consolidation
    consolidate_project(song_name, latest_midi, master_wav, stems, sample_dir, metadata,
                        out_dir=out_dir)

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE: {song_name}")
    print("=" * 70)
    return master_wav


def generate_all_eps():
    """Batch generate 4 EPs — one per expansion, 14 tracks each at random BPMs."""
    print("!!!" * 20)
    print("STARTING FULL EP GENERATION BATCH (4 EPs x 14 Tracks)")
    print("!!!" * 20)

    total_start = time.time()
    for expansion in EXPANSIONS.keys():
        print(f"\n\n>>> BEGINNING EP: {expansion}")
        for i in range(1, 15):
            target_bpm = round(random.uniform(70, 95), 1)
            print(f"\n--- EP {expansion} | Track {i}/14 | BPM: {target_bpm} ---")
            try:
                run_full_pipeline(target_expansion=expansion, target_bpm=target_bpm)
            except Exception as e:
                print(f"Error on track {i}: {e}")
                continue

    elapsed = (time.time() - total_start) / 3600
    print(f"\n\nBATCH COMPLETE. Total time: {elapsed:.2f} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fantom Audio Pipeline")
    parser.add_argument("--resume-from", type=int, choices=[1, 2, 3], default=1,
                        help="Resume pipeline from step 1, 2, or 3 (default: 1)")
    parser.add_argument("--song-name", type=str, default=None,
                        help="Song name to resume (required for --resume-from 2 or 3)")
    parser.add_argument("--exclude", type=str, nargs="+", default=[],
                        help="Stem name patterns to exclude from the mix (e.g. 'ClosedHat')")
    parser.add_argument("--expansion", type=str, choices=list(EXPANSIONS.keys()), default=None,
                        help="Constrain patches to a specific Roland expansion (e.g. 'SH-101')")
    parser.add_argument("--count", type=int, default=None,
                        help="Number of tracks to generate (default: 1, or prompted interactively)")
    parser.add_argument("--generate-eps", action="store_true",
                        help="Batch generate all 4 expansion EPs (14 tracks each)")
    args = parser.parse_args()

    if args.generate_eps:
        generate_all_eps()
    else:
        expansion = args.expansion
        count = args.count
        interactive = False

        # Offer interactive setup on a fresh run when no flags were provided
        if args.resume_from == 1 and args.song_name is None and expansion is None and count is None:
            expansion, count = prompt_interactive_options()
            interactive = True

        count = count or 1

        if count > 1:
            for i in range(count):
                target_bpm = round(random.uniform(70, 95), 1)
                print(f"\n--- Track {i+1}/{count} | Expansion: {expansion or 'Full Pool'} | BPM: {target_bpm} ---")
                run_full_pipeline(target_expansion=expansion, target_bpm=target_bpm,
                                  exclude=args.exclude, interactive=interactive)
        else:
            run_full_pipeline(
                resume_from_step=args.resume_from,
                song_name=args.song_name,
                target_expansion=expansion,
                exclude=args.exclude,
                interactive=interactive,
            )
