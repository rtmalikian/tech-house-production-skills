import mido
import time
import os
import sys
import sounddevice as sd
import numpy as np
from threading import Thread

# Add the current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from fantom_midi_control import FantomController

def play_midi_logic():
    project_root = os.path.dirname(os.path.dirname(current_dir))
    midi_path = os.path.join(project_root, 'scripts/output/05032026_012315_C_C_Locrian_v10_refactored.mid')
    
    controller = FantomController()
    if not controller.output: return

    # Config Track 1 (Bass)
    patch = {"name": "Saw Boost Bs", "msb": 87, "lsb": 64, "pc": 91, "channel": 0}
    controller.select_patch(patch['channel'], patch['msb'], patch['lsb'], patch['pc'])
    controller.set_part_level(patch['channel'], 127) # Max volume
    time.sleep(1.0)

    mid = mido.MidiFile(midi_path)
    start_tick = 19 * 4 * mid.ticks_per_beat
    
    play_mid = mido.MidiFile()
    play_mid.ticks_per_beat = mid.ticks_per_beat
    play_mid.tracks.append(mid.tracks[0])
    new_track = mido.MidiTrack()
    curr = 0
    for msg in mid.tracks[1]:
        curr += msg.time
        if curr < start_tick: continue
        m = msg.copy(time=0 if not new_track else msg.time)
        if not m.is_meta and hasattr(m, 'channel'): m.channel = 0
        new_track.append(m)
    play_mid.tracks.append(new_track)

    print("MIDI: Starting playback of Track 1...")
    for msg in play_mid.play():
        if not msg.is_meta:
            controller.output.send(msg)
        if time.time() - start_time > 10: break
    
    # Stop all
    for i in range(128):
        controller.output.send(mido.Message('note_off', note=i, velocity=0, channel=0))

def scan_audio():
    print("AUDIO: Scanning all 32 channels for 10 seconds...")
    device_index = 9 # FANTOM
    max_peaks = np.zeros(32)
    
    try:
        with sd.InputStream(device=device_index, channels=32, samplerate=48000) as stream:
            for _ in range(int(10 * 48000 / 1024)):
                data, overflowed = stream.read(1024)
                peaks = np.max(np.abs(data), axis=0)
                max_peaks = np.maximum(max_peaks, peaks)
                print(f"Max detected so far: {np.max(max_peaks):.6f}", end="\r")
    except Exception as e:
        print(f"\nAudio Error: {e}")

    print("\n\nScan Results:")
    found = False
    for i, p in enumerate(max_peaks):
        if p > 0.0001:
            print(f"  Channel {i+1}: {p:.6f}")
            found = True
    if not found:
        print("  NO SIGNAL DETECTED ON ANY CHANNEL.")
        print("\nPossible issues:")
        print("1. Terminal does not have Microphone permissions (System Settings > Privacy & Security > Microphone).")
        print("2. Fantom USB Driver is in 'Generic' mode (needs to be 'Vendor' for multi-channel audio).")
        print("3. Fantom USB Audio is not routed to USB (check Fantom System settings).")

if __name__ == "__main__":
    start_time = time.time()
    # Start MIDI in a separate thread
    midi_thread = Thread(target=play_midi_logic)
    midi_thread.start()
    
    # Run scanner in main thread
    scan_audio()
    
    midi_thread.join()
