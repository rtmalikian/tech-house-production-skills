import os
import sys
import glob

# Add project roots to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SCRIPTS_ROOT)

sys.path.append(PROJECT_ROOT)
sys.path.append(SCRIPTS_ROOT)
sys.path.append(SCRIPT_DIR)

from audio_recorder import AudioRecorder
from gain_staging import GainStager

def run_post_processing():
    recorder = AudioRecorder(output_dir="output/recordings")
    stager = GainStager(output_dir="output/mastered/staged")
    
    song_name = "05032026_015646_F#_F#_Hüseyni_v10_refactored"
    
    batches = {
        "melodic": ["Bass", "Harmonic Bass", "Main Melody", "Counter Melody", "Chorus Melody", "Melody FX", "Pad (Chords)"],
        "drum1": ["drum1_KickLow_n35", "drum1_Kick_n36", "drum1_SideStick_n37", "drum1_Snare_n38", "drum1_ClosedHat_n42", "drum1_LowTom_n45"],
        "drum2": ["drum2_KickLow_n35", "drum2_Kick_n36", "drum2_SideStick_n37", "drum2_Snare_n38", "drum2_Clap_n39", "drum2_SnareAlt_n40", "drum2_KickAlt_n41", "drum2_ClosedHat_n42", "drum2_LowTom_n45", "drum2_MidTom_n47"],
        "drum_aux": ["drum_aux_Tambourine_n54", "drum_aux_HighBongo_n60", "drum_aux_LowBongo_n61", "drum_aux_MuteConga_n62", "drum_aux_Maracas_n70"]
    }
    
    all_stems = []
    
    for batch_name, track_names in batches.items():
        pass_file = f"{song_name}_{batch_name}_pass.wav"
        print(f"\nExploding {pass_file}...")
        stem_paths_dict = recorder.split_stems(pass_file, track_names, batch_name)
        all_stems.extend(stem_paths_dict.values())
    
    print(f"\nExtracted {len(all_stems)} stems. Proceeding to Gain Staging...")
    
    results = stager.process_stems(all_stems)
    
    print("\nGain Staging Complete.")
    for r in results:
        print(f"  {os.path.basename(r['staged'])}: {r['original_lufs']:.2f} -> {r['target_lufs']:.2f} LUFS")

if __name__ == "__main__":
    run_post_processing()
