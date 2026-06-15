import os
import sys
import glob
import time

# Add project roots to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SCRIPTS_ROOT)

sys.path.append(PROJECT_ROOT)
sys.path.append(SCRIPTS_ROOT)
sys.path.append(SCRIPT_DIR)

from audio_recorder import AudioRecorder, MultiPassOrchestrator
from fantom_midi_control import FantomController

def re_record_melodic():
    controller = FantomController()
    recorder = AudioRecorder(device_index=9, output_dir="output/recordings")
    orchestrator = MultiPassOrchestrator(recorder, controller)
    
    midi_file = "scripts/output/05032026_015646_F#_F#_Hüseyni_v10_refactored.mid"
    song_name = "05032026_015646_F#_F#_Hüseyni_v10_refactored"
    
    # We only want to run the melodic pass for now as a test
    import mido
    mid = mido.MidiFile(midi_file)
    
    # Manually trigger melodic pass logic
    batch_name = "melodic"
    track_indices = [1, 2, 3, 4, 5, 6, 9] # From RECORDING_WORKFLOW.md
    
    print(f"\n=== RE-RECORDING PASS: {batch_name.upper()} ===")
    
    # 1. Prepare MIDI
    batch_mid = mido.MidiFile()
    batch_mid.ticks_per_beat = mid.ticks_per_beat
    batch_mid.tracks.append(mid.tracks[0])
    
    recorded_track_names = []
    for i, t_idx in enumerate(track_indices):
        new_track = mido.MidiTrack()
        new_track.name = mid.tracks[t_idx].name
        recorded_track_names.append(new_track.name)
        
        for msg in mid.tracks[t_idx]:
            if not msg.is_meta and hasattr(msg, 'channel'):
                new_track.append(msg.copy(channel=i))
            else:
                new_track.append(msg)
        batch_mid.tracks.append(new_track)
        
        # Configure Fantom
        patch_info = orchestrator.get_patch_for_track_name(new_track.name)
        print(f"  Mapping {new_track.name} to Part {i+1} ({patch_info['name']})")
        controller.select_patch(i, patch_info['msb'], patch_info['lsb'], patch_info['pc'])
        controller.set_part_level(i, 115)
    
    temp_midi = "temp_melodic_re.mid"
    batch_mid.save(temp_midi)
    
    output_filename = f"{song_name}_{batch_name}_pass.wav"
    #recorder.play_midi_and_record(temp_midi, output_filename, controller.port_name)
    # Manual recording of 30 seconds for test
    mid = mido.MidiFile(temp_midi)
    record_proc = recorder.record_pass(output_filename, 30.0)
    time.sleep(1.0)
    print(f"Playing MIDI to {controller.port_name} (30s test)...")
    try:
        with mido.open_output(controller.port_name) as port:
            start_time = time.time()
            for msg in mid.play():
                port.send(msg)
                if time.time() - start_time > 30: break
    except Exception as e:
        print(f"Playback error: {e}")
    record_proc.wait()
    
    # Check if recorded file is silent
    import subprocess
    cmd = ["sox", os.path.join("output/recordings", output_filename), "-n", "stat"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stderr)

if __name__ == "__main__":
    re_record_melodic()
