import os
import sys
import time
import argparse
import random
import copy
import subprocess
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

# Import the MIDI generator
import orchestrator as midi_gen

# Store the original patches for reliable monkey-patching
ORIGINAL_MODEL_PATCHES = copy.deepcopy(audio_recorder._MODEL_EXPANSION_PATCHES)

EXPANSIONS = {
    'JUPITER-8': 'JP8 ',
    'SH-101': 'SH-101 ',
    'JX-8P': 'JX-8P ',
    'JUNO-106': 'JUNO-106 '
}


def set_expansion_constraint(expansion_name: str):
    """Limit the synth patch pool to a specific Roland expansion."""
    if expansion_name not in EXPANSIONS:
        print(f"Warning: Unknown expansion {expansion_name}. Resetting to all patches.")
        audio_recorder._MODEL_EXPANSION_PATCHES = copy.deepcopy(ORIGINAL_MODEL_PATCHES)
        return

    prefix = EXPANSIONS[expansion_name]
    new_pool = {}
    
    # Pre-calculate all patches for this expansion to use as universal fallback
    all_expansion_patches = []
    for patches in ORIGINAL_MODEL_PATCHES.values():
        all_expansion_patches.extend([p for p in patches if p['name'].startswith(prefix)])

    for role, patches in ORIGINAL_MODEL_PATCHES.items():
        # Filter patches that start with the prefix
        filtered = [p for p in patches if p['name'].startswith(prefix)]
        
        if filtered:
            new_pool[role] = filtered
        else:
            # Smart Fallback: if 'brass' or 'bell' is missing for SH-101, 
            # try to use 'poly' or 'lead' from the SAME expansion.
            fallback_roles = ['poly', 'lead', 'bass']
            found_fallback = False
            for f_role in fallback_roles:
                f_patches = [p for p in ORIGINAL_MODEL_PATCHES.get(f_role, []) if p['name'].startswith(prefix)]
                if f_patches:
                    new_pool[role] = f_patches
                    found_fallback = True
                    break
            
            if not found_fallback:
                # Absolute last resort: any patch from this expansion
                new_pool[role] = all_expansion_patches if all_expansion_patches else patches
    
    print(f"Enforcing patch constraint: {expansion_name} (Prefix: '{prefix}')")
    audio_recorder._MODEL_EXPANSION_PATCHES = new_pool


def load_existing_stems(song_name: str, recordings_dir: str = "output/recordings") -> dict:
    """Scan recordings directory for stem files matching the song name."""
    stems = {}
    # Match patterns like: song_pass01_part01_usb01-02_name.wav or song_pass01_pass.wav
    patterns = [
        os.path.join(recordings_dir, f"{song_name}_pass*_part*.wav"),
        os.path.join(recordings_dir, f"{song_name}_pass*.wav"),
    ]

    stem_files = []
    for pattern in patterns:
        stem_files.extend(sorted(glob(pattern)))

    if not stem_files:
        print(f"No stems found matching: {patterns[0]}")
        return stems

    for path in stem_files:
        # Skip the multi-track pass files (they contain all channels)
        basename = os.path.basename(path)
        if basename.endswith("_pass.wav"):
            continue
        # Use the basename without extension as the key
        key = basename.replace(".wav", "")
        stems[key] = path

    print(f"Found {len(stems)} existing stems for {song_name}")
    return stems


def find_midi_file(song_name: str) -> str:
    """Search for the MIDI file in common output directories."""
    search_dirs = ["output", "scripts/output", "../output", "../../output"]
    for dir_path in search_dirs:
        # Check both with and without extension
        candidates = [
            os.path.join(dir_path, f"{song_name}.mid"),
            os.path.join(dir_path, f"{song_name}.midi"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        # Also try glob pattern
        pattern = os.path.join(dir_path, f"{song_name}*.mid*")
        matches = sorted(glob(pattern))
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


def generate_sound_design_report(song_name: str, out_dir: str):
    """Generate a Markdown report of the sound design and modulations used."""
    report_path = os.path.join(out_dir, f"{song_name}_sound_design.md")
    recordings_dir = "output/recordings"
    
    # Find all manifest files for this song
    manifest_pattern = os.path.join(recordings_dir, f"{song_name}_pass*_manifest.json")
    manifest_files = sorted(glob(manifest_pattern))
    
    if not manifest_files:
        return

    lines = [f"# Sound Design Report: {song_name}", ""]
    lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Part Assignments & Modulations")
    lines.append("")

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
            print(f"  Warning: Could not parse manifest {m_file} for report: {e}")

    try:
        with open(report_path, 'w') as f:
            f.write("\n".join(lines))
        print(f"✓ Sound Design Report generated: {report_path}")
    except Exception as e:
        print(f"  Warning: Could not write report: {e}")


def run_full_pipeline(resume_from_step: int = 1, song_name: str = None,
                      target_expansion: str = None, target_bpm: float = None):
    print("=" * 70)
    print("AUTONOMOUS FANTOM AUDIO PIPELINE")
    if target_expansion:
        print(f"EP TARGET: {target_expansion}")
    print("=" * 70)

    stems = {}
    bpm = target_bpm or 90.0
    latest_midi = None

    # Apply expansion constraint before any generation or recording
    if target_expansion:
        set_expansion_constraint(target_expansion)

    # STEP 1: Generate MIDI (if needed)
    if resume_from_step <= 1:
        print(f"\n[STEP 1] Generating MIDI Sequence (Target BPM: {bpm})...")
        import mido
        # Call parameterized orchestrator
        generated = midi_gen.main(target_bpm=bpm)
        latest_midi = generated[0] if isinstance(generated, tuple) else generated
        song_name = os.path.basename(latest_midi).replace(".mid", "")
        print(f"Generated MIDI: {latest_midi}")
        bpm = get_bpm_from_midi(latest_midi)
        print(f"BPM: {bpm}")
    else:
        # Need to find the MIDI file for the song
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

    # STEP 2: Record Multi-Pass (if needed)
    if resume_from_step <= 2:
        print("\n[STEP 2] Recording Phased Audio & Exporting Stems...")
        controller = FantomController()
        recorder = AudioRecorder(device_index=9, output_dir="output/recordings")
        orchestrator = MultiPassOrchestrator(recorder, controller, target_expansion=target_expansion)

        if not controller.output:
            print("Hardware connection failed. Aborting.")
            return

        stems = orchestrator.run_multi_pass(latest_midi, song_name)
    else:
        # Load existing stems
        print(f"\n[SKIPPING STEP 2] Loading existing stems for {song_name}...")
        stems = load_existing_stems(song_name)
        if not stems:
            print("No stems found. Cannot proceed to Step 3.")
            return

    # STEP 3: Advanced Production
    print("\n[STEP 3] Automated Mixing & Mastering...")
    # Update output directory if we have an expansion target
    out_dir = "output/mastered"
    if target_expansion:
        out_dir = os.path.join(out_dir, target_expansion.replace(" ", "_"))
    
    production = ProductionEngine(output_dir=out_dir)
    master_wav, pristine_wav = production.process_full_mix(stems, song_name, bpm=bpm)

    # STEP 4: Sample Generation
    print("\n[STEP 4] Generating Sample Pack (Loops & One-shots)...")
    try:
        sample_script = os.path.join(SCRIPT_DIR, "generate_sample_stems.py")
        python_exe = sys.executable
        # Pass the MIDI file generated in Step 1
        cmd = [python_exe, sample_script, latest_midi]
        print(f"  Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print("  Sample generation successful.")
    except Exception as e:
        print(f"  Warning: Sample generation failed: {e}")

    # STEP 5: Sound Design Documentation
    generate_sound_design_report(song_name, out_dir)

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE: {song_name}")
    print(f"Final Master : {master_wav}")
    print(f"Pristine Mix : {pristine_wav}")
    print("=" * 70)
    return master_wav, pristine_wav


def generate_all_eps():
    """Generate 4 EPs, 14 tracks each."""
    print("!!!" * 20)
    print("STARTING FULL EP GENERATION BATCH (4 EPs x 14 Tracks)")
    print("!!!" * 20)
    
    total_start = time.time()
    for expansion in EXPANSIONS.keys():
        print(f"\n\n>>> BEGINNING EP: {expansion}")
        for i in range(1, 15):
            track_name = f"EP_{expansion.replace(' ', '_')}_Track_{i:02d}"
            target_bpm = round(random.uniform(70, 95), 1)
            print(f"\n--- EP {expansion} | Track {i}/14 ({track_name}) | BPM: {target_bpm} ---")
            
            try:
                run_full_pipeline(target_expansion=expansion, target_bpm=target_bpm)
            except Exception as e:
                print(f"Error generating {track_name}: {e}")
                continue
    
    elapsed = (time.time() - total_start) / 3600
    print(f"\n\nBATCH COMPLETE. Total time: {elapsed:.2f} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fantom Audio Pipeline")
    parser.add_argument("--resume-from", type=int, choices=[1, 2, 3], default=1,
                        help="Resume pipeline from step 1, 2, or 3 (default: 1)")
    parser.add_argument("--song-name", type=str, default=None,
                        help="Song name to resume (required for --resume-from 2 or 3)")
    parser.add_argument("--generate-eps", action="store_true",
                        help="Generate all 4 expansion EPs (14 tracks each)")
    parser.add_argument("--expansion", type=str, choices=list(EXPANSIONS.keys()), default=None,
                        help="Target a specific Roland expansion (e.g., 'SH-101')")
    parser.add_argument("--count", type=int, default=1,
                        help="Number of tracks to generate (default: 1)")
    args = parser.parse_args()

    if args.generate_eps:
        generate_all_eps()
    elif args.expansion and args.count > 1:
        print(f"Starting test run for {args.expansion}: {args.count} tracks")
        for i in range(args.count):
            target_bpm = round(random.uniform(70, 95), 1)
            print(f"\n--- Test Track {i+1}/{args.count} | Expansion: {args.expansion} | BPM: {target_bpm} ---")
            try:
                run_full_pipeline(target_expansion=args.expansion, target_bpm=target_bpm)
            except Exception as e:
                print(f"Error: {e}")
                continue
    else:
        run_full_pipeline(
            resume_from_step=args.resume_from, 
            song_name=args.song_name, 
            target_expansion=args.expansion
        )
