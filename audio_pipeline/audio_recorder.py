import time
import os
import random
import re
import json
import mido
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
from typing import List, Optional, Dict, Tuple, Any
from scipy.signal import butter, sosfilt

# ---------------------------------------------------------------------------
# Octave-range checking constants (spectral centroid bounds in Hz)
# ---------------------------------------------------------------------------
_OCTAVE_CENTROID_BOUNDS = {
    'bass':   (40,  450),   # centroid above 450 Hz → patch is an octave too high
    'melody': (300, 4000),  # centroid below 300 Hz → patch is an octave too low
}
_MAX_OCTAVE_SHIFTS = 2  # maximum octave shifts in either direction per track
_OCTAVE_SETTLE_SECONDS = 0.75  # wait after MIDI pitch edits before audition/measurement
_OCTAVE_PLAY_SECONDS = 2.75
_OCTAVE_RECORD_SECONDS = 3.25
_CALIBRATION_RESET_SETTLE_SECONDS = 0.08
_FULL_PASS_RECORD_CHUNK_SECONDS = 0.5
_OCTAVE_DENSE_NOTE_LIMIT = 10
_MELODY_LOW_BAND_RATIO_MAX = 0.38
_BASS_RUMBLE_MAX_HZ = 40.0
_BASS_USABLE_MIN_HZ = 45.0
_BASS_USABLE_MAX_HZ = 180.0
_BASS_PITCH_MAX_HZ = 300.0
_BASS_RUMBLE_RATIO_MAX = 0.50
_BASS_USABLE_RATIO_MIN = 0.18
_BASS_PITCH_CONFIDENCE_MIN = 0.015
_BASS_EXPECTED_FUNDAMENTAL_RATIO_MIN = 0.055
_BASS_EXPECTED_FUNDAMENTAL_WEAK_RATIO = 0.12
_BASS_HARMONIC_DOMINANCE_MAX = 6.0
_BASS_BRIGHT_RATIO_MAX = 2.4
_SHARED_BASS_REGISTER = (33, 52)
_MAIN_BASS_REGISTER = _SHARED_BASS_REGISTER
_HARMONIC_BASS_REGISTER = _SHARED_BASS_REGISTER
_OCTAVE_RATIO_TOLERANCE = 0.35
_BASS_TRIGGER_PROBABILITY = 0.85
_HIHAT_TRIGGER_PROBABILITY = 0.85
_HIHAT_NOTES = {42, 44, 46}
_HIHAT_SPECIAL_PATCH_KEYS = {(86, 65, 69), (86, 65, 70)}
_HIHAT_BLOCKED_PATCH_KEYS = {
    (86, 65, pc) for pc in (63, 64, 65, 66, 67, 68, 71, 72, 73, 74)
} | {
    (86, 65, 8),   # Orchestra Kit — non-GM note mapping at 42/44/46
    (86, 65, 9),   # SFX Kit — sound effects at 42/44/46
    (86, 65, 17),  # Orchestra Kit w
    (86, 65, 18),  # SFX Kit w
}
_HIHAT_SPECIAL_NOTE_SHIFT_RANGE = (-10, 20)
_SNARE_TRIGGER_PROBABILITY = 0.92
_SNARE_NOTES = {37, 38, 39, 40}
_KICK_TRIGGER_PROBABILITY = 0.91
_KICK_NOTES = {35, 36, 41}
_DEFAULT_PATCH_ATTEMPTS = 4
_OCTAVE_CHECKED_PATCH_ATTEMPTS = 8
_BASS_PATCH_ATTEMPTS = 16
_MELODY_SYSEX_AUTOMATION_ENABLED = os.environ.get("FANTOM_MELODY_ENV_AUTOMATION", "1").strip().lower() not in {"0", "false", "no", "off"}
_SONG_TOTAL_BARS = 72

_LAYER_ELIGIBLE_TRACKS = {
    'main melody': {
        'family': 'Main_Melody',
        'categories': ['lead', 'poly', 'bell'],
        'probability': 1.0,
    },
    'counter melody': {
        'family': 'Counter_Melody',
        'categories': ['lead', 'poly', 'pluck', 'bell'],
        'probability': 1.0,
    },
    'chorus melody': {
        'family': 'Chorus_Melody',
        'categories': ['poly', 'brass', 'lead'],
        'probability': 1.0,
    },
    'pad (chords)': {
        'family': 'Pad_Chords',
        'categories': ['pad', 'strings', 'choir', 'poly'],
        'probability': 1.0,
    },
}

_LAYER_PAN_POSITIONS = {
    1: [64],
    2: [42, 86],
    3: [34, 64, 94],
}

_SNARE_LAYER_PROBABILITY = 1.0
_SNARE_LAYER_CATEGORIES = ['body', 'snap', 'air']
_SNARE_LAYER_PAN_POSITIONS = {
    2: [61, 67],
    3: [60, 64, 68],
}

_KICK_LAYER_PROBABILITY = 1.0
_KICK_LAYER_CATEGORIES = ['punch', 'sub', 'click']
_KICK_LAYER_PAN_POSITIONS = {
    2: [64, 64],
    3: [64, 64, 64],
}

_EASTERN_PERCUSSION_NOTES = {54, 60, 61, 62, 70}
_NOTE_NAME_TO_PC = {
    'C': 0, 'C#': 1, 'DB': 1, 'D': 2, 'D#': 3, 'EB': 3, 'E': 4, 'F': 5,
    'F#': 6, 'GB': 6, 'G': 7, 'G#': 8, 'AB': 8, 'A': 9, 'A#': 10,
    'BB': 10, 'B': 11,
}
_ARMENIAN_MAQAM_TUNING = {
    'Hijaz': {'intervals': [0, 1, 4, 5, 7, 8, 11], 'microtonal': {1: -50, 8: -50}},
    'Hüseyni': {'intervals': [0, 2, 3, 5, 7, 9, 10], 'microtonal': {2: -75, 10: -50}},
    'Kurdi': {'intervals': [0, 1, 3, 4, 6, 8, 10], 'microtonal': {1: -40, 6: -30}},
    'Rast': {'intervals': [0, 2, 4, 5, 7, 9, 10], 'microtonal': {4: -15, 10: -25}},
    'Bayati': {'intervals': [0, 2, 3, 5, 6, 8, 10], 'microtonal': {2: -55, 8: -40}},
    'Nahawand': {'intervals': [0, 2, 3, 5, 7, 8, 10], 'microtonal': {3: -20, 8: -35}},
}
_ARMENIAN_TUNED_TRACK_TOKENS = (
    'melody', 'chorus', 'counter', 'pad', 'chord', 'harmonic bass', 'duduk',
    'drone',
)
_EASTERN_PERCUSSION_PATCHES = {
    # EXZ006 World Instruments Z-Core tones. Requires EXZ006 installed on the Fantom.
    'tabla': [
        {'name': 'EXZ006 TablaBaya Menu', 'msb': 93, 'lsb': 21, 'pc': 115},
        {'name': 'EXZ006 Egypt Tablah', 'msb': 93, 'lsb': 21, 'pc': 119},
    ],
    'dholak': [
        {'name': 'EXZ006 Dholak Menu', 'msb': 93, 'lsb': 21, 'pc': 116},
        {'name': 'EXZ006 Dholak Menu 2', 'msb': 93, 'lsb': 21, 'pc': 117},
    ],
    'dhol': [
        {'name': 'EXZ006 Dhol Menu', 'msb': 93, 'lsb': 21, 'pc': 118},
    ],
    'madal': [
        {'name': 'EXZ006 Madal Menu', 'msb': 93, 'lsb': 21, 'pc': 120},
    ],
    'afro': [
        {'name': 'EXZ006 Afroperc Menu', 'msb': 93, 'lsb': 21, 'pc': 121},
    ],
    'conga': [
        {'name': 'EXZ006 Dyno Conga 1', 'msb': 93, 'lsb': 21, 'pc': 70},
        {'name': 'EXZ006 Dyno Conga 2', 'msb': 93, 'lsb': 21, 'pc': 71},
    ],
    'wood': [
        {'name': 'CMN Woodblock', 'msb': 87, 'lsb': 90, 'pc': 121},
    ],
}
_EASTERN_PERCUSSION_FALLBACK = [
    {'name': 'CMN Woodblock', 'msb': 87, 'lsb': 90, 'pc': 121},
]

_MODEL_EXPANSION_PATCHES = {
    # Roland model expansion tones. Requires the corresponding model expansions installed.
    'bass': [
        {'name': 'JP8 Reso Choke Bass', 'msb': 97, 'lsb': 64, 'pc': 56},
        {'name': 'JP8 Polarity Bass', 'msb': 97, 'lsb': 64, 'pc': 57},
        {'name': 'JP8 Gut Punch Bass', 'msb': 97, 'lsb': 64, 'pc': 60},
        {'name': 'JP8 Pulse Basser', 'msb': 97, 'lsb': 64, 'pc': 61},
        {'name': 'JP8 Saw Unison Bass', 'msb': 97, 'lsb': 64, 'pc': 67},
        {'name': 'JUNO-106 Big Boy Bass', 'msb': 97, 'lsb': 66, 'pc': 49},
        {'name': 'JUNO-106 Revisit Bass', 'msb': 97, 'lsb': 66, 'pc': 51},
        {'name': 'JUNO-106 Dizzy Bass', 'msb': 97, 'lsb': 66, 'pc': 52},
        {'name': 'JUNO-106 Halo Rez Bass', 'msb': 97, 'lsb': 66, 'pc': 56},
        {'name': 'JUNO-106 SubOSC Bass', 'msb': 97, 'lsb': 66, 'pc': 62},
        {'name': 'JX-8P Low Blow', 'msb': 97, 'lsb': 68, 'pc': 53},
        {'name': 'JX-8P Bit Basher', 'msb': 97, 'lsb': 68, 'pc': 54},
        {'name': 'JX-8P Dark Chorus Bass', 'msb': 97, 'lsb': 68, 'pc': 57},
        {'name': 'JX-8P DoubleFilter Bs', 'msb': 97, 'lsb': 68, 'pc': 60},
        {'name': 'SH-101 Oct Bass', 'msb': 97, 'lsb': 70, 'pc': 25},
        {'name': 'SH-101 THAbass', 'msb': 97, 'lsb': 70, 'pc': 28},
        {'name': 'SH-101 Filter Env Bs 1', 'msb': 97, 'lsb': 70, 'pc': 34},
        {'name': 'SH-101 PW+Saw Bass', 'msb': 97, 'lsb': 70, 'pc': 35},
        {'name': 'SH-101 Reso Sqr+Saw Bs', 'msb': 97, 'lsb': 70, 'pc': 40},
        {'name': 'SH-101 SubOSC Soft Bass', 'msb': 97, 'lsb': 70, 'pc': 44},
        {'name': 'SH-101 Porta Bass', 'msb': 97, 'lsb': 70, 'pc': 48},
        {'name': 'SH-101 Reso Seq Bs', 'msb': 97, 'lsb': 70, 'pc': 50},
    ],
    'lead': [
        {'name': 'JP8 Detuned Lead', 'msb': 97, 'lsb': 64, 'pc': 43},
        {'name': 'JP8 SBF Saw Lead', 'msb': 97, 'lsb': 64, 'pc': 44},
        {'name': 'JP8 Sync Lead JP', 'msb': 97, 'lsb': 64, 'pc': 46},
        {'name': 'JP8 Solid Lead Upper', 'msb': 97, 'lsb': 64, 'pc': 50},
        {'name': 'JP8 Saw Lead', 'msb': 97, 'lsb': 64, 'pc': 52},
        {'name': 'JUNO-106 JUNO Lead 1', 'msb': 97, 'lsb': 66, 'pc': 42},
        {'name': 'JUNO-106 JUNO Lead 2', 'msb': 97, 'lsb': 66, 'pc': 43},
        {'name': 'JUNO-106 Sacred Lead', 'msb': 97, 'lsb': 66, 'pc': 44},
        {'name': 'JUNO-106 Retroist Lead', 'msb': 97, 'lsb': 66, 'pc': 47},
        {'name': 'JX-8P Amazement Ld', 'msb': 97, 'lsb': 68, 'pc': 46},
        {'name': 'JX-8P Square Bottom', 'msb': 97, 'lsb': 68, 'pc': 47},
        {'name': 'JX-8P Sqr Lead', 'msb': 97, 'lsb': 68, 'pc': 49},
        {'name': 'JX-8P X-Mod Lead', 'msb': 97, 'lsb': 68, 'pc': 51},
        {'name': 'SH-101 Gimme Lead', 'msb': 97, 'lsb': 70, 'pc': 1},
        {'name': 'SH-101 Saw Boz', 'msb': 97, 'lsb': 70, 'pc': 4},
        {'name': 'SH-101 Pulse Leader', 'msb': 97, 'lsb': 70, 'pc': 11},
        {'name': 'SH-101 Porta Saw Lead', 'msb': 97, 'lsb': 70, 'pc': 13},
        {'name': 'SH-101 PWM LFO Lead', 'msb': 97, 'lsb': 70, 'pc': 18},
        {'name': 'SH-101 Bit Crusher Lead', 'msb': 97, 'lsb': 70, 'pc': 24},
    ],
    'poly': [
        {'name': 'JP8 Big Bite Pluck', 'msb': 97, 'lsb': 64, 'pc': 27},
        {'name': 'JP8 Fairy Tales', 'msb': 97, 'lsb': 64, 'pc': 28},
        {'name': 'JP8 Soft Pluck', 'msb': 97, 'lsb': 64, 'pc': 30},
        {'name': 'JUNO-106 Enchanted', 'msb': 97, 'lsb': 66, 'pc': 28},
        {'name': 'JUNO-106 Hard Pluck', 'msb': 97, 'lsb': 66, 'pc': 32},
        {'name': 'JUNO-106 Bright Pluck', 'msb': 97, 'lsb': 66, 'pc': 33},
        {'name': 'JX-8P Bright Keys', 'msb': 97, 'lsb': 68, 'pc': 32},
        {'name': 'JX-8P Syniano EP', 'msb': 97, 'lsb': 68, 'pc': 33},
        {'name': 'JX-8P Sqr Pluck 1', 'msb': 97, 'lsb': 68, 'pc': 61},
        {'name': 'SH-101 Porto Bells', 'msb': 97, 'lsb': 70, 'pc': 70},
        {'name': 'SH-101 Echo Pluck', 'msb': 97, 'lsb': 70, 'pc': 71},
        {'name': 'SH-101 Shorty /Mod', 'msb': 97, 'lsb': 70, 'pc': 72},
    ],
    'pad': [
        {'name': 'JP8 Berlin Night', 'msb': 97, 'lsb': 64, 'pc': 1},
        {'name': 'JP8 Sweep JP', 'msb': 97, 'lsb': 64, 'pc': 6},
        {'name': 'JP8 Reso Pad', 'msb': 97, 'lsb': 64, 'pc': 9},
        {'name': 'JP8 Bright Strings', 'msb': 97, 'lsb': 64, 'pc': 17},
        {'name': 'JP8 Strings JP', 'msb': 97, 'lsb': 64, 'pc': 18},
        {'name': 'JP8 Soft Saw Pad', 'msb': 97, 'lsb': 64, 'pc': 23},
        {'name': 'JUNO-106 Heater Pad', 'msb': 97, 'lsb': 66, 'pc': 1},
        {'name': 'JUNO-106 Saw Strings', 'msb': 97, 'lsb': 66, 'pc': 3},
        {'name': 'JUNO-106 Bright Pad', 'msb': 97, 'lsb': 66, 'pc': 7},
        {'name': 'JUNO-106 Juno Sweeper', 'msb': 97, 'lsb': 66, 'pc': 12},
        {'name': 'JUNO-106 Ambient Pad', 'msb': 97, 'lsb': 66, 'pc': 24},
        {'name': 'JX-8P Mass-5', 'msb': 97, 'lsb': 68, 'pc': 1},
        {'name': 'JX-8P Dynamic Lush Pad', 'msb': 97, 'lsb': 68, 'pc': 6},
        {'name': 'JX-8P Choir Pad', 'msb': 97, 'lsb': 68, 'pc': 11},
        {'name': 'JX-8P Soft Pad 1', 'msb': 97, 'lsb': 68, 'pc': 13},
        {'name': 'JX-8P Descender Pad', 'msb': 97, 'lsb': 68, 'pc': 18},
        {'name': 'SH-101 This Old Game', 'msb': 97, 'lsb': 70, 'pc': 64},
        {'name': 'SH-101 Poly 101 1', 'msb': 97, 'lsb': 70, 'pc': 66},
        {'name': 'SH-101 Shuno Pad', 'msb': 97, 'lsb': 70, 'pc': 67},
        {'name': 'SH-101 Simple Pad', 'msb': 97, 'lsb': 70, 'pc': 101},
    ],
    'brass': [
        {'name': 'JP8 Synth Brass JP', 'msb': 97, 'lsb': 64, 'pc': 40},
        {'name': 'JP8 PWM Env Brass', 'msb': 97, 'lsb': 64, 'pc': 41},
        {'name': 'JUNO-106 Bright Brass', 'msb': 97, 'lsb': 66, 'pc': 40},
        {'name': 'JUNO-106 Reso Soft Brass', 'msb': 97, 'lsb': 66, 'pc': 41},
        {'name': 'JX-8P JX Poly Brass', 'msb': 97, 'lsb': 68, 'pc': 38},
        {'name': 'JX-8P JX Powerbrass', 'msb': 97, 'lsb': 68, 'pc': 39},
        {'name': 'JX-8P Classic Poly JX', 'msb': 97, 'lsb': 68, 'pc': 44},
    ],
    'bell': [
        {'name': 'JP8 Delicate Bells', 'msb': 97, 'lsb': 64, 'pc': 14},
        {'name': 'JP8 CHIME', 'msb': 97, 'lsb': 64, 'pc': 94},
        {'name': 'JUNO-106 m0t0n0v0', 'msb': 97, 'lsb': 66, 'pc': 26},
        {'name': 'JX-8P Bell Chorus', 'msb': 97, 'lsb': 68, 'pc': 30},
        {'name': 'JX-8P Two Chimes', 'msb': 97, 'lsb': 68, 'pc': 31},
        {'name': 'SH-101 Reso Bell', 'msb': 97, 'lsb': 70, 'pc': 63},
    ],
    'fx': [
        {'name': 'JP8 XMod Spike', 'msb': 97, 'lsb': 64, 'pc': 71},
        {'name': 'JX-8P XMod Compu', 'msb': 97, 'lsb': 68, 'pc': 68},
        {'name': 'SH-101 SelfOSC Sweep', 'msb': 97, 'lsb': 70, 'pc': 89},
    ],
}


def _track_octave_type(track_name: str) -> Optional[str]:
    """Return 'bass', 'melody', or None (no octave checking for drums/other)."""
    n = track_name.lower()
    if any(x in n for x in ['kick', 'snare', 'hat', 'drum', 'perc', 'clap',
                              'bongo', 'conga', 'tambourine', 'maracas', 'stick', 'rim']):
        return None
    if 'bass' in n:
        return 'bass'
    if any(x in n for x in ['melody', 'lead', 'counter', 'chorus', 'fx', 'pad', 'chord']):
        return 'melody'
    return None

class AudioRecorder:
    def __init__(self, device_index: int = 9, output_dir: str = "output/recordings"):
        """
        device_index: Index for sounddevice (PortAudio). 
        Based on scan_fantom_audio.py, index 7 is typically the FANTOM.
        """
        self.device_index = device_index
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.recording_data = []
        self._recording = False

    def record_pass(self, filename: str, duration_seconds: float, channels: int = 32):
        """
        Record audio from the Fantom using sounddevice.
        """
        output_path = os.path.join(self.output_dir, filename)
        duration = duration_seconds + 5.0 # Pre-roll + Song + Tail
        samplerate = 48000
        total_frames = max(1, int(round(duration * samplerate)))
        chunk_frames = max(1, int(round(_FULL_PASS_RECORD_CHUNK_SECONDS * samplerate)))

        print(
            f"Starting recording (sounddevice blocking stream): {filename} "
            f"({duration:.2f}s total, device={self.device_index}, channels={channels}, "
            f"frames={total_frames}, chunk={chunk_frames})..."
        )

        # Query device to verify channel count matches expectations
        try:
            dev_info = sd.query_devices(self.device_index)
            dev_in_ch = dev_info.get('max_input_channels', 0)
            if dev_in_ch < channels:
                print(f"  WARNING: Device reports {dev_in_ch} input channels but {channels} requested")
                print(f"  Sync click detection on USB 31/32 may fail — recording will use {dev_in_ch} channels")
                channels = dev_in_ch
        except Exception:
            pass
        
        self.recording_data = []
        self._recording = True
        error_box = []
        
        def record_and_save():
            overflow_count = 0
            frames_remaining = total_frames
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with sf.SoundFile(
                    output_path,
                    mode='w',
                    samplerate=samplerate,
                    channels=channels,
                    subtype='FLOAT',
                ) as wav:
                    with sd.InputStream(
                        samplerate=samplerate,
                        channels=channels,
                        device=self.device_index,
                        dtype='float32',
                        callback=None,
                    ) as stream:
                        while frames_remaining > 0:
                            frames_to_read = min(chunk_frames, frames_remaining)
                            data, overflowed = stream.read(frames_to_read)
                            if overflowed:
                                overflow_count += 1
                            wav.write(data)
                            frames_remaining -= len(data)
                self._recording = False
                overflow_note = f" (input overflows={overflow_count})" if overflow_count else ""
                print(f"Finished recording: {filename}{overflow_note}")
            except Exception as exc:
                self._recording = False
                error_box.append(exc)
                print(f"Recording error while writing {filename}: {exc}")

        thread = threading.Thread(target=record_and_save, name=f"record_pass_{filename}")
        thread.recording_errors = error_box
        thread.start()
        return thread

    def split_stems(self, multi_channel_file: str, track_names: List[str], batch_name: str,
                    pre_roll_seconds: float = 0.0, count_in_seconds: float = 0.0,
                    trim_count_in: bool = True, return_sync_info: bool = False):
        """
        Explode a 32-channel WAV into individual stereo stems.
        FANTOM-6/7/8 USB Mapping (Parallel Mode):
        USB 1/2:   Part 1  …  USB 31/32: Part 16 (sync click)

        Trim strategy (two-stage):
          1. Coarse: remove pre_roll_seconds (sleep-based estimate)
          2. Fine:   detect the first count-in click transient on USB 31/32.
             Default export trims through the 4-beat count-in, so stem sample 0 is
             bar 1 of the song and downstream bar/section automation is direct.

        If return_sync_info is True, returns (stem_paths, sync_info) instead of just stem_paths.
        """
        source_path = os.path.join(self.output_dir, multi_channel_file)
        if not os.path.exists(source_path):
            print(f"Error: Source file {source_path} not found for splitting.")
            return ({}, _empty_sync_info()) if return_sync_info else {}

        print(f"Splitting stems for batch: {batch_name}...")

        data, samplerate = sf.read(source_path)
        recorded_channels = data.shape[1]
        print(f"  Recorded WAV: {recorded_channels} channels, {samplerate} Hz, "
              f"{data.shape[0] / samplerate:.2f}s")

        sync_info = {
            "channel_count": recorded_channels,
            "sync_channel_available": recorded_channels > _SYNC_USB_R,
            "click_detected": False,
            "sync_offset_samples": 0,
            "usb_latency_ms": 0.0,
            "coarse_trim_samples": 0,
            "fine_trim_applied": False,
            "fallback_used": False,
            "fallback_latency_ms": 0.0,
        }

        # Step 1: coarse pre-roll trim
        if pre_roll_seconds > 0.0:
            coarse_trim = int(pre_roll_seconds * samplerate)
            data = data[coarse_trim:]
            sync_info["coarse_trim_samples"] = coarse_trim

        # Step 2: fine sync — detect first count-in click transient on USB 31/32 (Part 16)
        # Then trim through the count-in so exported stems start at bar 1.
        bar1_sample = int(count_in_seconds * samplerate) if trim_count_in else 0

        if recorded_channels > _SYNC_USB_R:
            sync_mono = (np.abs(data[:, _SYNC_USB_L]) + np.abs(data[:, _SYNC_USB_R])) / 2
            peak = float(np.max(sync_mono)) if sync_mono.size > 0 else 0.0
            sync_info["sync_channel_peak_dbfs"] = round(20 * np.log10(max(peak, 1e-12)), 2)

            if peak < 1e-4:  # -80 dB — channel is essentially silent
                print(f"  WARNING: Sync channel USB 31/32 is silent (peak {sync_info['sync_channel_peak_dbfs']} dBFS)")
                print(f"  Count-in click NOT routed to USB 31/32 — applying fallback latency estimate")
                fallback_samples = int(_FALLBACK_USB_LATENCY_MS / 1000 * samplerate)
                trim_to = fallback_samples + bar1_sample
                data = data[trim_to:]
                sync_info["fallback_used"] = True
                sync_info["fallback_latency_ms"] = _FALLBACK_USB_LATENCY_MS
                print(f"  Applied fallback latency: {_FALLBACK_USB_LATENCY_MS:.0f}ms "
                      f"({fallback_samples} samples)")
                if trim_count_in:
                    print(f"  Trimmed 4-beat count-in ({bar1_sample / samplerate:.3f}s) + fallback; "
                          f"stem sample 0 ≈ song bar 1")
            else:
                # Adaptive threshold: 1% of peak, minimum -40 dBFS
                threshold = max(0.01, peak * 0.01)
                hits = np.where(sync_mono > threshold)[0]
                if len(hits) > 0:
                    sync_offset = int(hits[0])
                    usb_latency_ms = sync_offset / samplerate * 1000
                    trim_to = sync_offset + bar1_sample
                    data = data[trim_to:]
                    sync_info["click_detected"] = True
                    sync_info["sync_offset_samples"] = sync_offset
                    sync_info["usb_latency_ms"] = round(usb_latency_ms, 2)
                    sync_info["fine_trim_applied"] = True
                    sync_info["detection_threshold"] = round(threshold, 6)
                    print(f"  Click beat 1 detected: USB latency = {usb_latency_ms:.1f}ms "
                          f"(threshold {threshold:.4f}, peak {peak:.4f})")
                    if trim_count_in:
                        print(f"  Trimmed 4-beat count-in ({bar1_sample / samplerate:.3f}s); "
                              f"stem sample 0 = song bar 1")
                    else:
                        print(f"  Bar 1 (song start) = sample {bar1_sample} = "
                              f"{bar1_sample / samplerate:.3f}s into each stem")
                else:
                    # Peak exists but no samples exceed threshold — unusual, use fallback
                    print(f"  WARNING: Sync channel has signal (peak {peak:.4f}) but no "
                          f"transient above threshold {threshold:.4f}")
                    print(f"  Applying fallback latency estimate: {_FALLBACK_USB_LATENCY_MS:.0f}ms")
                    fallback_samples = int(_FALLBACK_USB_LATENCY_MS / 1000 * samplerate)
                    trim_to = fallback_samples + bar1_sample
                    data = data[trim_to:]
                    sync_info["fallback_used"] = True
                    sync_info["fallback_latency_ms"] = _FALLBACK_USB_LATENCY_MS
                    if trim_count_in:
                        print(f"  Trimmed 4-beat count-in + fallback; stem sample 0 ≈ song bar 1")
        else:
            print(f"  WARNING: Recording has {recorded_channels} channels (need > {_SYNC_USB_R}) — "
                  f"count-in sync unavailable")
            print(f"  Applying fallback latency estimate: {_FALLBACK_USB_LATENCY_MS:.0f}ms")
            fallback_samples = int(_FALLBACK_USB_LATENCY_MS / 1000 * samplerate)
            trim_to = fallback_samples + bar1_sample
            data = data[trim_to:]
            sync_info["fallback_used"] = True
            sync_info["fallback_latency_ms"] = _FALLBACK_USB_LATENCY_MS
            if trim_count_in:
                print(f"  Trimmed 4-beat count-in + fallback; stem sample 0 ≈ song bar 1")

        stem_paths = {}
        
        # Parallel Mode: Part 1 starts at Channel 0/1
        for i, name in enumerate(track_names):
            # Parallel Mode: Part i+1 on USB pair i+1 (channels i*2 and i*2+1)
            left = i * 2
            right = i * 2 + 1
            safe_name = self._safe_stem_label(name)
            stem_filename = (
                f"{multi_channel_file.replace('_pass.wav', '')}"
                f"_part{i+1:02d}_usb{left+1:02d}-{right+1:02d}_{safe_name}.wav"
            )
            stem_path = os.path.join(self.output_dir, stem_filename)
            
            if right < data.shape[1]:
                pair_data = data[:, [left, right]]
                # 20 Hz high-pass filter — removes sub-sonic DC and disruptive harmonics
                sos = butter(4, 20.0 / (samplerate / 2), btype='high', output='sos')
                pair_data = sosfilt(sos, pair_data, axis=0).astype(np.float32)
                sf.write(stem_path, pair_data, samplerate, subtype='FLOAT')
                print(f"  Exported Stem: {stem_filename} (Channels {left+1}-{right+1})")
                stem_paths[f"{batch_name}_part{i+1:02d}_{safe_name}"] = stem_path
            else:
                print(f"  Warning: Could not extract channels {left+1},{right+1} for {name} (Part {i+1})")
            
        if return_sync_info:
            return stem_paths, sync_info
        return stem_paths

    def _safe_stem_label(self, name: str) -> str:
        label = name.strip().replace(" ", "_")
        label = label.replace("(", "").replace(")", "")
        label = re.sub(r"[^A-Za-z0-9_-]+", "_", label)
        label = re.sub(r"_+", "_", label).strip("_")
        return label or "Unnamed"

    def play_midi_and_record(self, midi_file_path: str, output_filename: str, midi_port_name: str, total_duration: float) -> float:
        """
        Synchronized playback and recording.
        total_duration should already include the count-in duration.
        Returns the actual measured pre-roll in seconds so stems can be coarse-trimmed.
        Precise alignment is done in split_stems by detecting the count-in click.
        """
        mid = mido.MidiFile(midi_file_path)

        # 1. Start recording (song + count-in is already baked into total_duration; add release tail)
        record_thread = self.record_pass(output_filename, total_duration + 2.0)

        # 2. Measure pre-roll precisely so we can trim it later
        t0 = time.perf_counter()
        time.sleep(2.0)
        actual_pre_roll = time.perf_counter() - t0

        # 3. Play MIDI — time 0 of the song is right here
        print(f"Playing MIDI to {midi_port_name}...")
        try:
            with mido.open_output(midi_port_name) as port:
                port.send(mido.Message('start'))
                for msg in mid.play():
                    port.send(msg)
                port.send(mido.Message('stop'))
        except Exception as e:
            print(f"Playback error: {e}")

        # 4. Wait for recording to finish
        record_thread.join()
        if getattr(record_thread, 'recording_errors', None):
            raise record_thread.recording_errors[0]
        print(f"Finished recording pass: {output_filename}")
        return actual_pre_roll

    def measure_peak(self, duration: float, channel_indices: List[int]) -> float:
        """Measure peak amplitude on specific channels for a short duration."""
        samplerate = 48000
        recording = self._blocking_input_capture(duration, samplerate)
        relevant_data = recording[:, channel_indices]
        peak = np.max(np.abs(relevant_data)) if relevant_data.size > 0 else 0.0
        return peak

    def record_audio_snippet(self, duration: float, channel_indices: List[int]) -> Tuple[np.ndarray, int]:
        """Record and return raw 32-ch audio array plus samplerate."""
        samplerate = 48000
        recording = self._blocking_input_capture(duration, samplerate)
        return recording, samplerate

    def _blocking_input_capture(self, duration: float, samplerate: int) -> np.ndarray:
        """Capture input without sounddevice.rec()'s Python callback helper."""
        frames = max(1, int(duration * samplerate))
        with sd.InputStream(
            samplerate=samplerate,
            channels=32,
            device=self.device_index,
            dtype='float32',
            callback=None,
        ) as stream:
            recording, overflowed = stream.read(frames)
        if overflowed:
            print("    Audio input overflow during calibration capture.")
        return recording

    def spectral_centroid(self, audio: np.ndarray, samplerate: int,
                          channel_indices: List[int]) -> Optional[float]:
        """Return energy-weighted spectral centroid (Hz) in the 30–5000 Hz band.
        Returns None if signal is too quiet to analyse reliably."""
        ch_data = audio[:, channel_indices] if audio.ndim > 1 else audio.reshape(-1, 1)
        mono = np.mean(ch_data, axis=1).astype(np.float64)
        if np.max(np.abs(mono)) < 0.001:
            return None
        windowed = mono * np.hanning(len(mono))
        fft_mag = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(mono), 1.0 / samplerate)
        mask = (freqs >= 30) & (freqs <= 5000)
        total = np.sum(fft_mag[mask])
        if total < 1e-10:
            return None
        return float(np.sum(freqs[mask] * fft_mag[mask]) / total)

        # Part 16 / channel 15 is reserved as a 4-beat count-in click in every pass.

# USB 31/32 (channels 30/31, 0-indexed) captures it.
# The click provides a shared transient reference across all passes,
# automatically compensating for USB audio input latency and sleep jitter.
_SYNC_CH      = 15
_SYNC_NOTE    = 36   # C2 — kick on Standard Kit
_SYNC_SOUND   = {'name': 'Standard Kit', 'msb': 86, 'lsb': 65, 'pc': 1}
_SYNC_USB_L   = 30   # 0-indexed left channel in 32-ch recording
_SYNC_USB_R   = 31
_COUNT_IN_BEATS = 4  # beats of click before bar 1
_FALLBACK_USB_LATENCY_MS = 80.0  # conservative default when click detection fails


def _empty_sync_info() -> dict:
    return {
        "channel_count": 0,
        "sync_channel_available": False,
        "click_detected": False,
        "sync_offset_samples": 0,
        "usb_latency_ms": 0.0,
        "coarse_trim_samples": 0,
        "fine_trim_applied": False,
        "fallback_used": True,
        "fallback_latency_ms": _FALLBACK_USB_LATENCY_MS,
    }


class MultiPassOrchestrator:
    def __init__(self, recorder: AudioRecorder, controller, target_expansion: str = None):
        self.recorder = recorder
        self.controller = controller
        self.target_expansion = target_expansion
        self._used_patches: set = set()  # (msb, lsb, pc) used this session
        self._bad_patches: set = set()   # (msb, lsb, pc) rejected during audition/calibration

    def _infer_midi_tuning_context(self, midi_file_path: str, metadata: Dict = None) -> Dict:
        metadata = metadata or {}
        scale_label = str(metadata.get('scale') or '')
        key_label = str(metadata.get('key') or '')
        filename = os.path.basename(midi_file_path)

        mode = next((m for m in _ARMENIAN_MAQAM_TUNING if m in scale_label), None)
        if mode is None:
            mode = next((m for m in _ARMENIAN_MAQAM_TUNING if m in filename), None)
        is_armenian = bool(metadata.get('is_armenian')) or mode is not None
        if not is_armenian or mode is None:
            return {"is_armenian": False}

        if not key_label:
            # Generated filenames are e.g. 05062026_..._C#_C#_Hijaz_4-4_...
            key_match = re.search(r'_(C#|D#|F#|G#|A#|C|D|E|F|G|A|B)_.*_' + re.escape(mode), filename)
            if key_match:
                key_label = key_match.group(1)
        key_pc = _NOTE_NAME_TO_PC.get(key_label.upper(), 0)
        mode_info = _ARMENIAN_MAQAM_TUNING[mode]
        cent_offsets = {}
        for interval, cents in mode_info['microtonal'].items():
            cent_offsets[(key_pc + interval) % 12] = cents
        return {
            "is_armenian": True,
            "mode": mode,
            "key": key_label or 'C',
            "key_pc": key_pc,
            "cent_offsets": cent_offsets,
        }

    def _track_should_use_scale_tune(self, track_name: str, patch_info: Dict = None) -> bool:
        n = (track_name or '').lower().replace('_', ' ')
        if '__sync__' in n or 'drum' in n or 'kick' in n or 'snare' in n or 'hat' in n:
            return False
        if 'bass' in n and 'harmonic' not in n and 'drone' not in n:
            return False
        return any(token in n for token in _ARMENIAN_TUNED_TRACK_TOKENS)

    def _apply_armenian_zone_tuning(self, part_idx: int, track_name: str,
                                    patch_info: Dict, tuning_context: Dict) -> Dict:
        if not tuning_context.get("is_armenian"):
            return {}
        if not self._track_should_use_scale_tune(track_name, patch_info):
            return {}
        if not hasattr(self.controller, 'set_zone_scale_tune'):
            return {"skipped": "controller_missing_set_zone_scale_tune"}
        applied = self.controller.set_zone_scale_tune(
            part_idx + 1,
            tuning_context.get("key_pc", 0),
            tuning_context.get("cent_offsets", {}),
        ) or {}
        if applied:
            applied.update({
                "mode": tuning_context.get("mode"),
                "key": tuning_context.get("key"),
                "track": track_name,
            })
        return applied

    def _reset_zone_tuning(self, part_idx: int):
        if hasattr(self.controller, 'reset_zone_scale_tune'):
            self.controller.reset_zone_scale_tune(part_idx + 1)

    def get_batches(self, mid):
        """
        Pack every non-empty recordable MIDI track into 15-Part passes.
        Part 16 is reserved for the sync click, so each pass carries up to
        15 independent stereo stems through Fantom Parallel Mode.
        """
        recordable = self._build_recordable_specs(mid)

        batches = {}
        for batch_idx, start in enumerate(range(0, len(recordable), 15), start=1):
            batches[f"pass{batch_idx:02d}"] = recordable[start:start + 15]
        return batches

    def _build_recordable_specs(self, mid) -> List[Dict]:
        specs = []
        for i, track in enumerate(mid.tracks):
            if not self._is_recordable_track(track):
                continue
            specs.extend(self._expand_layer_specs(i, track, mid.ticks_per_beat))
        layered = [s for s in specs if s.get('layer_family')]
        if layered:
            families = {}
            for spec in layered:
                families.setdefault(spec['layer_family'], 0)
                families[spec['layer_family']] += 1
            summary = ", ".join(f"{name} x{count}" for name, count in sorted(families.items()))
            print(f"Track layering enabled: {summary}")
        return specs

    def _bar_ticks_from_midi(self, mid: mido.MidiFile) -> int:
        numerator, denominator = 4, 4
        for track in mid.tracks:
            for msg in track:
                if getattr(msg, 'type', None) == 'time_signature':
                    numerator = int(getattr(msg, 'numerator', 4))
                    denominator = int(getattr(msg, 'denominator', 4))
                    return int(round(mid.ticks_per_beat * numerator * 4 / max(1, denominator)))
        return mid.ticks_per_beat * 4

    def _build_part_sysex_automation(self, part_idx: int, track_name: str,
                                     patch_info: Dict, tpb: int,
                                     bar_ticks: int, count_in_ticks: int) -> Tuple[Optional[mido.MidiTrack], Dict]:
        if not _MELODY_SYSEX_AUTOMATION_ENABLED:
            return None, {"enabled": False, "skip_reason": "disabled_by_env"}
        if patch_info.get('msb') == 97:
            return None, {"enabled": False, "skip_reason": "model_expansion_patch"}
        builder = getattr(self.controller, 'build_melody_sysex_automation_track', None)
        if not builder:
            return None, {"enabled": False, "skip_reason": "controller_not_supported"}
        track, metadata = builder(part_idx, track_name, tpb, bar_ticks, count_in_ticks, _SONG_TOTAL_BARS)
        return track, metadata or {"enabled": False, "skip_reason": "no_metadata"}

    def _expand_layer_specs(self, source_idx: int, track: mido.MidiTrack, tpb: int) -> List[Dict]:
        track_name = track.name or ''
        
        # 1. Check for Snare Layering (Mandatory)
        if self._is_snare_layer_candidate(track_name):
            snare_specs = self._expand_snare_layer_specs(source_idx, track, tpb)
            if snare_specs:
                return snare_specs

        # 2. Check for Kick Layering (Mandatory)
        if self._is_kick_layer_candidate(track_name):
            kick_specs = self._expand_kick_layer_specs(source_idx, track, tpb)
            if kick_specs:
                return kick_specs

        # 3. Check for Melodic Layering (Config-driven, currently 1.0)
        cfg = _LAYER_ELIGIBLE_TRACKS.get(track_name.lower())
        base_spec = {
            'source_index': source_idx,
            'source_name': track_name,
            'track': track,
            'recorded_name': track_name,
            'layer_family': None,
            'layer_index': None,
            'layer_count': 1,
            'layer_category': None,
            'layer_variation': {},
        }
        if cfg is None:
            return [base_spec]

        # Layering is mandatory for eligible melodic tracks (probability 1.0)
        layer_count = 3 if random.random() < 0.30 else 2
        categories = self._choose_layer_categories(cfg['categories'], layer_count)
        pans = _LAYER_PAN_POSITIONS[layer_count]
        specs = []
        for layer_idx in range(1, layer_count + 1):
            category = categories[layer_idx - 1]
            variation = self._layer_variation(layer_idx, layer_count, category, tpb, pans[layer_idx - 1])
            label = f"{cfg['family']}_layer{layer_idx}_{category}"
            layer_track = self._make_layer_track(track, label, variation)
            specs.append({
                'source_index': source_idx,
                'source_name': track_name,
                'track': layer_track,
                'recorded_name': label,
                'layer_family': cfg['family'],
                'layer_index': layer_idx,
                'layer_count': layer_count,
                'layer_category': category,
                'layer_variation': variation,
            })
        return specs

    def _is_snare_layer_candidate(self, track_name: str) -> bool:
        n = track_name.lower()
        return (
            '_layer' not in n
            and 'snare' in n
            and any(x in n for x in ['drum1_', 'drum2_', 'drum_aux_'])
        )

    def _expand_snare_layer_specs(self, source_idx: int, track: mido.MidiTrack,
                                  tpb: int) -> List[Dict]:
        # Always layer snares
        layer_count = 3 if random.random() < 0.35 else 2
        categories = _SNARE_LAYER_CATEGORIES[:layer_count]
        pans = _SNARE_LAYER_PAN_POSITIONS[layer_count]
        specs = []
        for layer_idx, category in enumerate(categories, start=1):
            variation = self._snare_layer_variation(layer_idx, category, tpb, pans[layer_idx - 1])
            label = f"{track.name}_layer{layer_idx}_{category}"
            layer_track = self._make_layer_track(track, label, variation)
            specs.append({
                'source_index': source_idx,
                'source_name': track.name,
                'track': layer_track,
                'recorded_name': label,
                'layer_family': track.name,
                'layer_index': layer_idx,
                'layer_count': layer_count,
                'layer_category': category,
                'layer_variation': variation,
            })
        return specs

    def _is_kick_layer_candidate(self, track_name: str) -> bool:
        n = track_name.lower()
        return (
            '_layer' not in n
            and 'kick' in n
            and any(x in n for x in ['drum1_', 'drum2_', 'drum_aux_'])
        )

    def _expand_kick_layer_specs(self, source_idx: int, track: mido.MidiTrack,
                                 tpb: int) -> List[Dict]:
        # Always layer kicks
        layer_count = 3 if random.random() < 0.35 else 2
        categories = _KICK_LAYER_CATEGORIES[:layer_count]
        pans = _KICK_LAYER_PAN_POSITIONS[layer_count]
        specs = []
        for layer_idx, category in enumerate(categories, start=1):
            variation = self._kick_layer_variation(layer_idx, category, tpb, pans[layer_idx - 1])
            label = f"{track.name}_layer{layer_idx}_{category}"
            layer_track = self._make_layer_track(track, label, variation)
            specs.append({
                'source_index': source_idx,
                'source_name': track.name,
                'track': layer_track,
                'recorded_name': label,
                'layer_family': track.name,
                'layer_index': layer_idx,
                'layer_count': layer_count,
                'layer_category': category,
                'layer_variation': variation,
            })
        return specs

    def _kick_layer_variation(self, layer_idx: int, category: str, tpb: int, pan: int) -> Dict:
        # Kicks are usually more phase-sensitive, so timing offsets are smaller
        timing_ms = 0 if category == 'punch' else random.randint(1, 5)
        timing_ticks = max(0, int(round((timing_ms / 500.0) * tpb)))
        velocity_offsets = {
            'punch': 0,
            'sub': random.randint(-6, -2),
            'click': random.randint(-12, -6),
        }
        return {
            'velocity_offset': velocity_offsets.get(category, 0),
            'timing_offset_ticks': timing_ticks,
            'timing_offset_ms': timing_ms,
            'note_keep_probability': 1.0,
            'octave_shift': 0,
            'pan': pan,
        }

    def _snare_layer_variation(self, layer_idx: int, category: str, tpb: int, pan: int) -> Dict:
        timing_ms = 0 if category == 'body' else random.randint(1, 8)
        timing_ticks = max(0, int(round((timing_ms / 500.0) * tpb)))
        velocity_offsets = {
            'body': random.randint(-2, 3),
            'snap': random.randint(-8, -2),
            'air': random.randint(-18, -9),
        }
        return {
            'velocity_offset': velocity_offsets.get(category, 0),
            'timing_offset_ticks': timing_ticks,
            'timing_offset_ms': timing_ms,
            'note_keep_probability': 1.0,
            'octave_shift': 0,
            'pan': pan,
        }

    def _choose_layer_categories(self, categories: List[str], layer_count: int) -> List[str]:
        available = [cat for cat in categories if self.controller.sound_db.get(cat)]
        if not available:
            available = categories[:]
        first = available[0]
        rest = available[1:]
        random.shuffle(rest)
        chosen = [first] + rest[:max(0, layer_count - 1)]
        while len(chosen) < layer_count:
            chosen.append(random.choice(available))
        return chosen

    def _layer_variation(self, layer_idx: int, layer_count: int, category: str,
                         tpb: int, pan: int) -> Dict:
        if layer_idx == 1:
            return {
                'velocity_offset': 0,
                'timing_offset_ticks': 0,
                'timing_offset_ms': 0,
                'note_keep_probability': 1.0,
                'octave_shift': 0,
                'pan': pan,
            }

        timing_ms = random.randint(5, 20)
        timing_ticks = max(1, int(round((timing_ms / 500.0) * tpb)))  # 500ms/beat default at 120 BPM.
        octave_shift = 0
        if category in ['bell'] and random.random() < 0.45:
            octave_shift = 12
        elif category in ['strings', 'choir'] and random.random() < 0.25:
            octave_shift = -12

        return {
            'velocity_offset': random.randint(-14, -5),
            'timing_offset_ticks': timing_ticks,
            'timing_offset_ms': timing_ms,
            'note_keep_probability': random.uniform(0.68, 0.92) if layer_idx == 3 else random.uniform(0.78, 0.96),
            'octave_shift': octave_shift,
            'pan': pan,
        }

    def _make_layer_track(self, source: mido.MidiTrack, label: str, variation: Dict) -> mido.MidiTrack:
        events = []
        abs_tick = 0
        active_dropped = {}
        keep_prob = variation.get('note_keep_probability', 1.0)
        octave = variation.get('octave_shift', 0)
        velocity_offset = variation.get('velocity_offset', 0)
        timing_offset = variation.get('timing_offset_ticks', 0)

        for msg in source:
            abs_tick += msg.time
            if msg.is_meta:
                if msg.type not in ('track_name', 'end_of_track'):
                    events.append((abs_tick, msg.copy()))
                continue

            out = msg.copy()
            if out.type in ('note_on', 'note_off') and hasattr(out, 'note'):
                note_key = getattr(out, 'note')
                is_note_on = out.type == 'note_on' and getattr(out, 'velocity', 0) > 0
                is_note_end = out.type == 'note_off' or (out.type == 'note_on' and getattr(out, 'velocity', 0) == 0)
                if is_note_on:
                    keep = random.random() <= keep_prob
                    if not keep:
                        active_dropped[note_key] = active_dropped.get(note_key, 0) + 1
                        continue
                    out = out.copy(
                        note=max(0, min(127, out.note + octave)),
                        velocity=max(1, min(127, out.velocity + velocity_offset))
                    )
                elif is_note_end:
                    dropped = active_dropped.get(note_key, 0)
                    if dropped > 0:
                        active_dropped[note_key] = dropped - 1
                        continue
                    out = out.copy(note=max(0, min(127, out.note + octave)))

            events.append((abs_tick + timing_offset, out))

        new_track = mido.MidiTrack()
        new_track.name = label
        new_track.append(mido.MetaMessage('track_name', name=label, time=0))
        new_track.append(mido.Message('control_change', control=10, value=variation.get('pan', 64), time=0))

        events.sort(key=lambda item: item[0])
        last_tick = 0
        for abs_time, msg in events:
            abs_time = max(0, abs_time)
            new_track.append(msg.copy(time=max(0, abs_time - last_tick)))
            last_tick = abs_time
        return new_track

    def _is_recordable_track(self, track) -> bool:
        if not track.name or track.name == '__sync__':
            return False
        return any(
            msg.type == 'note_on' and getattr(msg, 'velocity', 0) > 0
            for msg in track
        )

    def play_snippet(self, midi_obj: mido.MidiFile, duration_seconds: float):
        """Play a snippet of MIDI for calibration."""
        if not self.controller.output: return
        start_time = time.perf_counter()
        try:
            # mid.play() handles tempo correctly
            for msg in midi_obj.play():
                if time.perf_counter() - start_time > duration_seconds:
                    break
                self.controller.output.send(msg)
        except Exception as e:
            print(f"    Snippet play error: {e}")
        # All Notes Off
        for ch in range(16):
            self.controller.output.send(mido.Message('control_change', channel=ch, control=123, value=0))

    def _reset_calibration_controllers(self):
        """Clear notes and common latch controllers before calibration auditions."""
        if not self.controller.output:
            return
        for ch in range(16):
            self.controller.output.send(mido.Message('control_change', channel=ch, control=123, value=0))
            self.controller.output.send(mido.Message('control_change', channel=ch, control=64, value=0))
            self.controller.output.send(mido.Message('pitchwheel', channel=ch, pitch=0))
        reset_portamento = getattr(self.controller, 'reset_zone_portamento', None)
        if reset_portamento:
            for zone in range(1, 17):
                reset_portamento(zone)
        time.sleep(_CALIBRATION_RESET_SETTLE_SECONDS)

    @staticmethod
    def _shift_track_notes(track: mido.MidiTrack, semitones: int) -> mido.MidiTrack:
        """Return a copy of track with every note pitch shifted by semitones (clamped 0–127)."""
        new_track = mido.MidiTrack()
        for msg in track:
            if msg.type in ('note_on', 'note_off') and hasattr(msg, 'note'):
                new_track.append(msg.copy(note=max(0, min(127, msg.note + semitones))))
            else:
                new_track.append(msg)
        return new_track

    @staticmethod
    def _track_note_range(track: mido.MidiTrack) -> Optional[Tuple[int, int]]:
        notes = [
            msg.note for msg in track
            if msg.type == 'note_on' and getattr(msg, 'velocity', 0) > 0
        ]
        if not notes:
            return None
        return min(notes), max(notes)

    @staticmethod
    def _track_note_frequencies(track: Optional[mido.MidiTrack]) -> List[float]:
        if track is None:
            return []
        notes = sorted({
            msg.note for msg in track
            if msg.type == 'note_on' and getattr(msg, 'velocity', 0) > 0
        })
        return [440.0 * (2.0 ** ((note - 69) / 12.0)) for note in notes]

    @staticmethod
    def _bass_register_for_track(track_name: str) -> Tuple[int, int]:
        return _SHARED_BASS_REGISTER

    @classmethod
    def _bass_register_ok(cls, track: Optional[mido.MidiTrack], track_name: str) -> bool:
        note_range = cls._track_note_range(track) if track is not None else None
        if note_range is None:
            return False
        lo, hi = cls._bass_register_for_track(track_name)
        return note_range[0] >= lo and note_range[1] <= hi

    @classmethod
    def _bass_register_shift(cls, track: Optional[mido.MidiTrack], track_name: str) -> int:
        note_range = cls._track_note_range(track) if track is not None else None
        if note_range is None:
            return 0
        target_lo, target_hi = cls._bass_register_for_track(track_name)
        low, high = note_range
        offset = 0
        for _ in range(_MAX_OCTAVE_SHIFTS):
            if high > target_hi:
                offset -= 12
                low -= 12
                high -= 12
                continue
            if low < target_lo:
                offset += 12
                low += 12
                high += 12
                continue
            break
        return offset

    @staticmethod
    def _safe_label(label: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
        return cleaned.strip("_") or "track"

    def _build_calibration_snippet(self, track: mido.MidiTrack, part_idx: int, tpb: int,
                                   tempo_us: int) -> mido.MidiFile:
        """Build a short dense note audition so octave decisions are based on real pitches."""
        abs_tick = 0
        active: Dict[Tuple[int, int], Tuple[int, int]] = {}
        note_pairs: List[Tuple[int, int, int, int]] = []
        for msg in track:
            abs_tick += msg.time
            if msg.is_meta or not hasattr(msg, 'note'):
                continue
            channel = getattr(msg, 'channel', 0)
            key = (channel, msg.note)
            if msg.type == 'note_on' and getattr(msg, 'velocity', 0) > 0:
                active[key] = (abs_tick, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and getattr(msg, 'velocity', 0) == 0):
                start = active.pop(key, None)
                if start:
                    start_tick, velocity = start
                    duration = max(tpb // 4, abs_tick - start_tick)
                    note_pairs.append((start_tick, msg.note, velocity, duration))

        snippet_mid = mido.MidiFile()
        snippet_mid.ticks_per_beat = tpb
        tempo_track = mido.MidiTrack()
        tempo_track.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))
        snippet_mid.tracks.append(tempo_track)

        snippet_track = mido.MidiTrack()
        snippet_track.name = getattr(track, 'name', 'calibration_snippet')
        snippet_track.append(mido.MetaMessage('track_name', name=snippet_track.name, time=0))
        if note_pairs:
            note_pairs.sort(key=lambda item: item[0])
            step = max(tpb // 2, 180)
            last_tick = 0
            for idx, (_, note, velocity, duration) in enumerate(note_pairs[:_OCTAVE_DENSE_NOTE_LIMIT]):
                start_tick = idx * step
                note_duration = max(tpb // 4, min(duration, int(tpb * 0.90)))
                snippet_track.append(mido.Message(
                    'note_on', channel=part_idx, note=note,
                    velocity=max(1, min(127, velocity)), time=max(0, start_tick - last_tick)
                ))
                snippet_track.append(mido.Message(
                    'note_off', channel=part_idx, note=note,
                    velocity=0, time=note_duration
                ))
                last_tick = start_tick + note_duration
        else:
            current_abs = 0
            started = False
            for msg in track:
                if msg.is_meta:
                    continue
                current_abs += msg.time
                if msg.type == 'note_on' and getattr(msg, 'velocity', 0) > 0:
                    started = True
                if started:
                    snippet_track.append(msg.copy(channel=part_idx, time=0 if len(snippet_track) == 1 else msg.time))

        snippet_track.append(mido.MetaMessage('end_of_track', time=tpb))
        snippet_mid.tracks.append(snippet_track)
        return snippet_mid

    def _save_octave_audit_midi(self, snippet_mid: mido.MidiFile, audit_dir: Optional[str],
                                part_idx: int, track_name: str, label: str) -> Optional[str]:
        if not audit_dir:
            return None
        try:
            os.makedirs(audit_dir, exist_ok=True)
            filename = f"part{part_idx + 1:02d}_{self._safe_label(track_name)}_{label}.mid"
            path = os.path.join(audit_dir, filename)
            snippet_mid.save(path)
            return path
        except Exception as exc:
            print(f"    [Octave] Could not save audit MIDI: {exc}")
            return None

    @staticmethod
    def _copy_midi_with_track(midi_obj: mido.MidiFile, track_idx: int,
                              replacement_track: mido.MidiTrack) -> mido.MidiFile:
        copied = mido.MidiFile()
        copied.ticks_per_beat = midi_obj.ticks_per_beat
        for idx, track in enumerate(midi_obj.tracks):
            new_track = mido.MidiTrack()
            source = replacement_track if idx == track_idx else track
            for msg in source:
                new_track.append(msg.copy())
            copied.tracks.append(new_track)
        return copied

    @staticmethod
    def _extract_mono(audio: np.ndarray, channel_indices: List[int]) -> Optional[np.ndarray]:
        if audio is None or len(audio) == 0:
            return None
        try:
            selected = audio[:, channel_indices]
        except Exception:
            selected = audio
        if selected.ndim > 1:
            selected = np.mean(selected, axis=1)
        mono = selected.astype(float)
        if np.max(np.abs(mono)) < 1e-7:
            return None
        return mono

    @staticmethod
    def _band_ratio(audio: np.ndarray, samplerate: int, channel_indices: List[int],
                    low_hz: float, high_hz: float, total_low_hz: float,
                    total_high_hz: float) -> Optional[float]:
        selected = MultiPassOrchestrator._extract_mono(audio, channel_indices)
        if selected is None:
            return None
        window = np.hanning(len(selected))
        spectrum = np.fft.rfft(selected * window)
        power = np.abs(spectrum) ** 2
        freqs = np.fft.rfftfreq(len(selected), 1.0 / samplerate)
        target = (freqs >= low_hz) & (freqs < high_hz)
        total = (freqs >= total_low_hz) & (freqs < total_high_hz)
        total_power = float(np.sum(power[total]))
        if total_power <= 1e-12:
            return None
        return float(np.sum(power[target]) / total_power)

    @staticmethod
    def _estimate_pitch_profile(audio: np.ndarray, samplerate: int, channel_indices: List[int],
                                octave_type: str, track: Optional[mido.MidiTrack] = None,
                                track_name: str = '') -> Dict[str, Any]:
        mono = MultiPassOrchestrator._extract_mono(audio, channel_indices)
        if mono is None:
            return {"fundamental_hz": None, "pitch_confidence": 0.0, "classification": "no_signal"}

        window = np.hanning(len(mono))
        spectrum = np.fft.rfft(mono * window)
        power = np.abs(spectrum) ** 2
        freqs = np.fft.rfftfreq(len(mono), 1.0 / samplerate)

        if octave_type == 'bass':
            total_low, total_high = 20.0, 700.0
            search_low, search_high = 25.0, _BASS_PITCH_MAX_HZ
            target_low, target_high = _BASS_USABLE_MIN_HZ, _BASS_USABLE_MAX_HZ
        else:
            total_low, total_high = 80.0, 5000.0
            search_low, search_high = 80.0, 2000.0
            target_low, target_high = 160.0, 1400.0

        total_mask = (freqs >= total_low) & (freqs < total_high)
        total_power = float(np.sum(power[total_mask]))
        if total_power <= 1e-12:
            return {"fundamental_hz": None, "pitch_confidence": 0.0, "classification": "no_signal"}

        search_mask = (freqs >= search_low) & (freqs <= search_high)
        candidates = freqs[search_mask]
        if len(candidates) == 0:
            return {"fundamental_hz": None, "pitch_confidence": 0.0, "classification": "uncertain"}

        scores = []
        for freq in candidates:
            width = max(2.0, freq * 0.025)
            score = 0.0
            for harmonic, weight in ((1, 1.0), (2, 0.55), (3, 0.30), (4, 0.18)):
                center = freq * harmonic
                if center > total_high:
                    continue
                band = (freqs >= center - width) & (freqs <= center + width)
                score += weight * float(np.sum(power[band]))
            scores.append(score)

        scores = np.asarray(scores)
        if scores.size == 0 or float(np.max(scores)) <= 1e-12:
            return {"fundamental_hz": None, "pitch_confidence": 0.0, "classification": "uncertain"}

        best_idx = int(np.argmax(scores))
        fundamental = float(candidates[best_idx])
        confidence = min(1.0, float(scores[best_idx] / total_power))

        if octave_type == 'bass':
            rumble_ratio = MultiPassOrchestrator._band_ratio(
                audio, samplerate, channel_indices, 20.0, _BASS_RUMBLE_MAX_HZ, 20.0, 700.0
            )
            usable_ratio = MultiPassOrchestrator._band_ratio(
                audio, samplerate, channel_indices, _BASS_USABLE_MIN_HZ, _BASS_USABLE_MAX_HZ, 20.0, 700.0
            )
            harmonic_ratio = MultiPassOrchestrator._band_ratio(
                audio, samplerate, channel_indices, _BASS_USABLE_MAX_HZ, 700.0, 20.0, 700.0
            )
            rumble_ratio = rumble_ratio or 0.0
            usable_ratio = usable_ratio or 0.0
            harmonic_ratio = harmonic_ratio or 0.0
            expected_freqs = MultiPassOrchestrator._track_note_frequencies(track)
            expected_power = 0.0
            expected_low = expected_high = None
            if expected_freqs:
                expected_low = min(expected_freqs)
                expected_high = max(expected_freqs)
                for expected_freq in expected_freqs:
                    width = max(2.0, expected_freq * 0.035)
                    band = (freqs >= expected_freq - width) & (freqs <= expected_freq + width)
                    expected_power += float(np.sum(power[band]))
            expected_ratio = expected_power / total_power if total_power > 1e-12 else 0.0
            harmonic_dominance = harmonic_ratio / max(expected_ratio, 1e-6)
            bright_ratio = harmonic_ratio / max(usable_ratio, 1e-6)
            register_ok = MultiPassOrchestrator._bass_register_ok(track, track_name)

            if register_ok and (
                expected_ratio < _BASS_EXPECTED_FUNDAMENTAL_RATIO_MIN
                and harmonic_dominance > _BASS_HARMONIC_DOMINANCE_MAX
            ):
                classification = "too_bright"
            elif register_ok and (
                bright_ratio > _BASS_BRIGHT_RATIO_MAX
                and expected_ratio < _BASS_EXPECTED_FUNDAMENTAL_WEAK_RATIO
            ):
                classification = "too_bright"
            elif register_ok and usable_ratio < _BASS_USABLE_RATIO_MIN and harmonic_ratio > 0.35:
                classification = "too_bright"
            elif register_ok and rumble_ratio > _BASS_RUMBLE_RATIO_MAX and expected_ratio < _BASS_EXPECTED_FUNDAMENTAL_RATIO_MIN:
                classification = "rumble_heavy"
            elif register_ok:
                classification = "usable_bass"
            elif confidence < _BASS_PITCH_CONFIDENCE_MIN and usable_ratio < _BASS_USABLE_RATIO_MIN:
                classification = "uncertain"
            elif rumble_ratio > _BASS_RUMBLE_RATIO_MAX and usable_ratio < _BASS_USABLE_RATIO_MIN:
                classification = "rumble_heavy"
            elif fundamental < _BASS_USABLE_MIN_HZ:
                classification = "too_low"
            elif fundamental > _BASS_USABLE_MAX_HZ:
                classification = "too_high"
            else:
                classification = "usable_bass"
            return {
                "fundamental_hz": fundamental,
                "pitch_confidence": confidence,
                "classification": classification,
                "rumble_ratio": rumble_ratio,
                "usable_bass_ratio": usable_ratio,
                "harmonic_ratio": harmonic_ratio,
                "expected_midi_fundamental_range_hz": [expected_low, expected_high] if expected_freqs else None,
                "expected_fundamental_energy_ratio": expected_ratio,
                "harmonic_dominance_ratio": harmonic_dominance,
                "bass_bright_ratio": bright_ratio,
                "bass_register_ok": register_ok,
            }

        if confidence < 0.01:
            classification = "uncertain"
        elif fundamental < target_low:
            classification = "too_low"
        elif fundamental > target_high:
            classification = "too_high"
        else:
            classification = "usable_melody"
        return {
            "fundamental_hz": fundamental,
            "pitch_confidence": confidence,
            "classification": classification,
        }

    def _octave_metrics(self, audio: np.ndarray, samplerate: int, channel_indices: List[int],
                        octave_type: str, track: Optional[mido.MidiTrack] = None,
                        track_name: str = '') -> Dict[str, Any]:
        centroid = self.recorder.spectral_centroid(audio, samplerate, channel_indices)
        profile = self._estimate_pitch_profile(audio, samplerate, channel_indices, octave_type, track, track_name)
        if octave_type == 'bass':
            metrics = {"centroid_hz": centroid}
            metrics.update(profile)
            return metrics
        ratio = self._band_ratio(audio, samplerate, channel_indices, 20.0, 300.0, 20.0, 5000.0)
        metrics = {"centroid_hz": centroid, "low_band_ratio": ratio}
        metrics.update(profile)
        return metrics

    @staticmethod
    def _round_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
        rounded: Dict[str, Any] = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.floating)):
                rounded[key] = round(float(value), 4)
            else:
                rounded[key] = value
        return rounded

    def _octave_decision(self, metrics: Dict[str, Any], octave_type: str) -> Tuple[int, str]:
        centroid = metrics.get("centroid_hz")
        if centroid is None:
            return 0, "no_signal"
        if octave_type == 'bass':
            classification = metrics.get("classification")
            fundamental = metrics.get("fundamental_hz")
            if metrics.get("bass_register_ok") and classification == "too_bright":
                return 0, "bass_patch_too_bright"
            if metrics.get("bass_register_ok") and classification == "rumble_heavy":
                return 0, "bass_patch_rumble_heavy"
            if metrics.get("bass_register_ok"):
                return 0, "bass_register_ok"
            if classification in ("rumble_heavy", "too_low"):
                f_label = "unknown" if fundamental is None else f"{fundamental:.1f}hz"
                return 12, f"bass_{classification}_{f_label}"
            if classification == "too_high":
                return -12, f"bass_fundamental_above_{_BASS_USABLE_MAX_HZ:g}hz"
            if classification == "uncertain":
                return 0, "bass_pitch_uncertain"
            return 0, "bass_range_ok"
        min_hz, max_hz = _OCTAVE_CENTROID_BOUNDS[octave_type]
        if centroid < min_hz:
            return 12, f"centroid_below_{min_hz:g}hz"
        if centroid > max_hz:
            return -12, f"centroid_above_{max_hz:g}hz"
        if octave_type == 'melody':
            low_ratio = metrics.get("low_band_ratio")
            if low_ratio is not None and low_ratio > _MELODY_LOW_BAND_RATIO_MAX:
                return 12, f"melody_low_band_ratio_{low_ratio:.2f}"
        return 0, "range_ok"

    def _verify_octave_shift(self, before_metrics: Dict[str, Any], after_metrics: Dict[str, Any],
                             shift: int, octave_type: str) -> Tuple[bool, str]:
        before_f = before_metrics.get("fundamental_hz")
        after_f = after_metrics.get("fundamental_hz")
        if before_f is None or after_f is None or before_f <= 0:
            return False, "fundamental_missing"

        ratio = after_f / before_f
        expected = 2.0 if shift > 0 else 0.5
        lo = expected * (1.0 - _OCTAVE_RATIO_TOLERANCE)
        hi = expected * (1.0 + _OCTAVE_RATIO_TOLERANCE)
        if not (lo <= ratio <= hi):
            return False, f"fundamental_ratio_{ratio:.2f}_outside_{lo:.2f}_{hi:.2f}"

        if octave_type == 'bass':
            after_class = after_metrics.get("classification")
            if shift < 0 and after_metrics.get("bass_register_ok") and after_class == "too_bright":
                return False, "downshift_landed_on_too_bright_patch"
            if shift < 0 and after_class in ("rumble_heavy", "too_low"):
                return False, f"downshift_landed_in_{after_class}"
            if shift > 0 and after_class == "uncertain":
                return False, "upshift_pitch_uncertain"

        return True, f"fundamental_ratio_{ratio:.2f}_verified"

    def _check_octave_range(self, part_idx: int, snippet_mid: mido.MidiFile,
                             octave_type: str, track_name: str = '',
                             audit_dir: Optional[str] = None) -> Tuple[int, Dict[str, Any]]:
        """Play the snippet, measure spectral centroid, and transpose ±1 octave until
        the track falls within the expected band. Returns offset and diagnostics."""
        min_hz, max_hz = _OCTAVE_CENTROID_BOUNDS[octave_type]
        ch_indices = [part_idx * 2, part_idx * 2 + 1]
        total_offset = 0
        diagnostics: Dict[str, Any] = {
            "type": octave_type,
            "target_centroid_hz": [min_hz, max_hz],
            "target_bass_fundamental_hz": [_BASS_USABLE_MIN_HZ, _BASS_USABLE_MAX_HZ] if octave_type == 'bass' else None,
            "settle_delay_seconds": _OCTAVE_SETTLE_SECONDS,
            "bass_register_target_midi": list(self._bass_register_for_track(track_name)) if octave_type == 'bass' else None,
            "midi_register_prefit_semitones": 0,
            "before_note_range": None,
            "after_note_range": None,
            "before_centroid_hz": None,
            "after_centroid_hz": None,
            "fundamental_hz_before": None,
            "fundamental_hz_after": None,
            "shift_verified": False,
            "rejected_shift_reason": None,
            "patch_rejected_reason": None,
            "decision": "not_checked",
            "attempts": [],
            "audit_midi": {},
            "verification_passed": False,
        }

        if len(snippet_mid.tracks) > 1:
            diagnostics["before_note_range"] = self._track_note_range(snippet_mid.tracks[1])
            before_path = self._save_octave_audit_midi(
                snippet_mid, audit_dir, part_idx, track_name, "before"
            )
            if before_path:
                diagnostics["audit_midi"]["before"] = before_path

            if octave_type == 'bass':
                prefit_shift = self._bass_register_shift(snippet_mid.tracks[1], track_name)
                if prefit_shift != 0:
                    before_range = self._track_note_range(snippet_mid.tracks[1])
                    snippet_mid.tracks[1] = self._shift_track_notes(snippet_mid.tracks[1], prefit_shift)
                    total_offset += prefit_shift
                    after_range = self._track_note_range(snippet_mid.tracks[1])
                    diagnostics["midi_register_prefit_semitones"] = prefit_shift
                    diagnostics["after_note_range"] = after_range
                    prefit_path = self._save_octave_audit_midi(
                        snippet_mid, audit_dir, part_idx, track_name,
                        f"prefit_{prefit_shift:+d}st".replace("+", "plus").replace("-", "minus")
                    )
                    if prefit_path:
                        diagnostics["audit_midi"]["prefit"] = prefit_path
                    if before_range and after_range:
                        print(
                            f"    [Octave] MIDI register prefit {prefit_shift:+d} st: "
                            f"{before_range[0]}-{before_range[1]} -> {after_range[0]}-{after_range[1]}"
                        )
                    print(f"    [Octave] Waiting {_OCTAVE_SETTLE_SECONDS:.2f}s before auditioning prefit MIDI.")
                    time.sleep(_OCTAVE_SETTLE_SECONDS)

        for attempt in range(_MAX_OCTAVE_SHIFTS):
            self._reset_calibration_controllers()
            play_thread = threading.Thread(target=self.play_snippet, args=(snippet_mid, _OCTAVE_PLAY_SECONDS))
            play_thread.start()
            audio, samplerate = self.recorder.record_audio_snippet(_OCTAVE_RECORD_SECONDS, ch_indices)
            play_thread.join()

            current_track = snippet_mid.tracks[1] if len(snippet_mid.tracks) > 1 else None
            metrics = self._octave_metrics(audio, samplerate, ch_indices, octave_type, current_track, track_name)
            centroid = metrics.get("centroid_hz")
            if centroid is None:
                print(f"    [Octave] No signal — skipping range check.")
                diagnostics["decision"] = "no_signal"
                break

            if diagnostics["before_centroid_hz"] is None:
                diagnostics["before_centroid_hz"] = round(float(centroid), 2)
                if metrics.get("fundamental_hz") is not None:
                    diagnostics["fundamental_hz_before"] = round(float(metrics["fundamental_hz"]), 2)

            shift, reason = self._octave_decision(metrics, octave_type)
            attempt_info = {
                "attempt": attempt + 1,
                "offset_before_semitones": total_offset,
                "metrics": self._round_metrics(metrics),
                "decision": reason,
                "shift_semitones": shift,
            }
            diagnostics["attempts"].append(attempt_info)

            extra = ""
            if octave_type == 'bass':
                if metrics.get("fundamental_hz") is not None:
                    extra += f" | f0={metrics['fundamental_hz']:.1f} Hz"
                if metrics.get("rumble_ratio") is not None:
                    extra += f" | rumble={metrics['rumble_ratio']:.2f}"
                if metrics.get("usable_bass_ratio") is not None:
                    extra += f" | usable={metrics['usable_bass_ratio']:.2f}"
                if metrics.get("expected_fundamental_energy_ratio") is not None:
                    extra += f" | expected={metrics['expected_fundamental_energy_ratio']:.2f}"
                if metrics.get("bass_bright_ratio") is not None:
                    extra += f" | bright={metrics['bass_bright_ratio']:.2f}"
                if metrics.get("classification"):
                    extra += f" | {metrics['classification']}"
            elif octave_type == 'melody' and metrics.get("low_band_ratio") is not None:
                extra = f" | <300Hz ratio={metrics['low_band_ratio']:.2f}"

            print(f"    [Octave] Centroid: {centroid:.1f} Hz  (target {min_hz}-{max_hz} Hz){extra}")

            if shift == 0:
                print(f"    [Octave] Frequency range OK.")
                diagnostics["decision"] = reason
                if octave_type == 'bass' and metrics.get("classification") == "too_bright":
                    diagnostics["patch_rejected_reason"] = "bass_patch_too_bright"
                    diagnostics["verification_passed"] = False
                    print("    [Octave] Bass patch rejected: expected MIDI fundamentals are weak versus harmonics.")
                elif octave_type == 'bass' and metrics.get("classification") == "rumble_heavy":
                    diagnostics["patch_rejected_reason"] = "bass_patch_rumble_heavy"
                    diagnostics["verification_passed"] = False
                    print("    [Octave] Bass patch rejected: sub-rumble dominates the expected bass range.")
                break

            direction = "UP" if shift > 0 else "DOWN"
            print(f"    [Octave] {reason} — auditioning {direction} one octave ({shift:+d} st)")
            before_range = self._track_note_range(snippet_mid.tracks[1]) if len(snippet_mid.tracks) > 1 else None
            candidate_track = None
            if len(snippet_mid.tracks) > 1:
                candidate_track = self._shift_track_notes(snippet_mid.tracks[1], shift)
            candidate_mid = self._copy_midi_with_track(snippet_mid, 1, candidate_track) if candidate_track else snippet_mid
            candidate_offset = total_offset + shift
            after_range = self._track_note_range(candidate_track) if candidate_track else None
            diagnostics["decision"] = reason
            if before_range and after_range:
                print(f"    [Octave] Candidate MIDI note range {before_range[0]}-{before_range[1]} -> {after_range[0]}-{after_range[1]}")
            print(f"    [Octave] Waiting {_OCTAVE_SETTLE_SECONDS:.2f}s before auditioning candidate shift.")
            time.sleep(_OCTAVE_SETTLE_SECONDS)

            self._reset_calibration_controllers()
            play_thread = threading.Thread(target=self.play_snippet, args=(candidate_mid, _OCTAVE_PLAY_SECONDS))
            play_thread.start()
            audio, samplerate = self.recorder.record_audio_snippet(_OCTAVE_RECORD_SECONDS, ch_indices)
            play_thread.join()
            candidate_metrics = self._octave_metrics(audio, samplerate, ch_indices, octave_type, candidate_track, track_name)
            verified, verify_reason = self._verify_octave_shift(metrics, candidate_metrics, shift, octave_type)
            attempt_info["candidate_metrics"] = self._round_metrics(candidate_metrics)
            attempt_info["shift_verified"] = verified
            attempt_info["verification_reason"] = verify_reason

            candidate_centroid = candidate_metrics.get("centroid_hz")
            if candidate_centroid is not None:
                diagnostics["after_centroid_hz"] = round(float(candidate_centroid), 2)
            if candidate_metrics.get("fundamental_hz") is not None:
                diagnostics["fundamental_hz_after"] = round(float(candidate_metrics["fundamental_hz"]), 2)

            if verified and candidate_track is not None:
                snippet_mid.tracks[1] = candidate_track
                total_offset = candidate_offset
                diagnostics["after_note_range"] = after_range
                diagnostics["shift_verified"] = True
                diagnostics["verification_passed"] = True
                label = f"after_{total_offset:+d}st".replace("+", "plus").replace("-", "minus")
                after_path = self._save_octave_audit_midi(snippet_mid, audit_dir, part_idx, track_name, label)
                if after_path:
                    diagnostics["audit_midi"][label] = after_path
                print(f"    [Octave] Shift verified ({verify_reason}); committed {total_offset:+d} st.")
                if octave_type == 'bass' and candidate_metrics.get("bass_register_ok"):
                    if candidate_metrics.get("classification") == "too_bright":
                        diagnostics["patch_rejected_reason"] = "bass_patch_too_bright"
                        diagnostics["decision"] = "bass_patch_too_bright"
                        diagnostics["verification_passed"] = False
                        print("    [Octave] Bass patch rejected after verified shift: register is valid but expected fundamentals are weak.")
                    elif candidate_metrics.get("classification") == "rumble_heavy":
                        diagnostics["patch_rejected_reason"] = "bass_patch_rumble_heavy"
                        diagnostics["decision"] = "bass_patch_rumble_heavy"
                        diagnostics["verification_passed"] = False
                        print("    [Octave] Bass patch rejected after verified shift: sub-rumble dominates the expected bass range.")
                    else:
                        diagnostics["decision"] = "bass_register_ok_after_shift"
                    break
                continue

            diagnostics["rejected_shift_reason"] = verify_reason
            diagnostics["decision"] = "shift_not_audibly_verified"
            if octave_type == 'bass' and candidate_metrics.get("classification") == "too_bright":
                diagnostics["patch_rejected_reason"] = "bass_patch_too_bright"
                diagnostics["decision"] = "bass_patch_too_bright"
            elif octave_type == 'bass' and candidate_metrics.get("classification") == "rumble_heavy":
                diagnostics["patch_rejected_reason"] = "bass_patch_rumble_heavy"
                diagnostics["decision"] = "bass_patch_rumble_heavy"
            label = f"rejected_{candidate_offset:+d}st".replace("+", "plus").replace("-", "minus")
            rejected_path = self._save_octave_audit_midi(candidate_mid, audit_dir, part_idx, track_name, label)
            if rejected_path:
                diagnostics["audit_midi"][label] = rejected_path
            print(f"    [Octave] Shift rejected: {verify_reason}. Keeping offset {total_offset:+d} st.")
            break

        if len(snippet_mid.tracks) > 1:
            diagnostics["after_note_range"] = self._track_note_range(snippet_mid.tracks[1])
        if total_offset == 0 and diagnostics.get("after_centroid_hz") is None:
            diagnostics["after_centroid_hz"] = diagnostics.get("before_centroid_hz")
        if total_offset == 0 and diagnostics["decision"] in ("range_ok", "bass_range_ok", "not_checked"):
            diagnostics["verification_passed"] = True
        if total_offset != 0 and diagnostics["decision"] in ("bass_register_ok", "bass_register_ok_after_shift"):
            diagnostics["verification_passed"] = True

        diagnostics["total_offset_semitones"] = total_offset
        return total_offset, diagnostics

    def calibrate_part_gain(self, part_idx: int, track: mido.MidiTrack, tpb: int, tempo_us: int,
                             track_name: str = '', audit_dir: Optional[str] = None):
        """Iteratively adjust Fantom EQ gain until peak is near -6dBFS (0.5).
        For bass and melody tracks, also checks octave range and transposes if needed.
        Returns (gain_db, octave_offset_semitones, octave_diagnostics)."""
        # 1. Find the first note_on with velocity > 0 to avoid calibrating on silence
        abs_tick = 0
        first_note_tick = None
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                first_note_tick = abs_tick
                break
        
        if first_note_tick is None:
            print(f"  [Calibrating] Part {part_idx+1}: No notes found — using default gain.")
            return 0.0, 0, {"decision": "no_notes"}

        print(f"  [Calibrating] Part {part_idx+1} Gain (Notes start at tick {first_note_tick})...")
        self.controller.set_zone_eq_switch(part_idx + 1, True)

        # 2. Create a short dense musical snippet so sparse loops still calibrate reliably.
        snippet_mid = self._build_calibration_snippet(track, part_idx, tpb, tempo_us)

        # 3. Octave range check (bass and melody tracks only)
        octave_offset = 0
        octave_diagnostics: Dict[str, Any] = {"decision": "not_applicable"}
        octave_type = _track_octave_type(track_name)
        if octave_type is not None:
            print(f"  [Octave Check] Part {part_idx+1} ({track_name}) — type: {octave_type}")
            octave_offset, octave_diagnostics = self._check_octave_range(
                part_idx, snippet_mid, octave_type, track_name=track_name, audit_dir=audit_dir
            )
            if octave_offset != 0:
                print(f"  [Octave] Applied {octave_offset:+d} semitones to Part {part_idx+1}")
            if octave_diagnostics.get("patch_rejected_reason"):
                print(f"    ! OCTAVE CALIBRATION REJECTED PATCH: {octave_diagnostics['patch_rejected_reason']}")
                return None, octave_offset, octave_diagnostics

        # 4. Smart Calibration Loop (Proportional Control)
        current_gain = 0.0
        target_peak = 0.5  # -6dBFS
        max_attempts = 6
        tolerance = 0.05   # ±0.05 amplitude deviation

        for attempt in range(max_attempts):
            self.controller.set_zone_eq_gain(part_idx + 1, 'input', current_gain)
            time.sleep(0.1) # Wait for SysEx processing

            # Play and measure in parallel
            play_thread = threading.Thread(target=self.play_snippet, args=(snippet_mid, 2.5))
            play_thread.start()

            # Measure slightly longer than play to capture tails
            peak = self.recorder.measure_peak(duration=3.0, channel_indices=[part_idx*2, part_idx*2+1])
            play_thread.join()

            # SILENCE DETECTION:
            # If peak is nearly zero on first attempt, skip track (unassigned key)
            if attempt == 0 and peak < 0.005:
                print(f"    ! SILENCE DETECTED (Peak={peak:.4f}). Key unassigned? Skipping Part {part_idx+1}.")
                return None, octave_offset, octave_diagnostics

            peak_db = 20 * np.log10(max(1e-5, peak))
            target_db = 20 * np.log10(target_peak)
            diff_db = target_db - peak_db

            print(f"    - Attempt {attempt+1}: Peak={peak:.3f} ({peak_db:.1f}dB) | Need {diff_db:+.1f}dB | Gain={current_gain:+.1f}dB")

            # Check if we are within tolerance
            if abs(peak - target_peak) < tolerance:
                print(f"      * Target reached (within tolerance).")
                break

            # Proportional adjustment
            if peak < 0.001:
                current_gain += 12.0
            else:
                current_gain += diff_db

            # Clamp to Fantom's hardware limits (-24 to +24)
            current_gain = max(-24.0, min(24.0, current_gain))

            if (current_gain == 24.0 and diff_db > 0) or (current_gain == -24.0 and diff_db < 0):
                print(f"      * Hardware gain limits reached.")
                break

        print(f"    - Final Calibration: Part {part_idx+1} Gain={current_gain:+.1f}dB  Octave offset={octave_offset:+d}st")
        return current_gain, octave_offset, octave_diagnostics

    def _trigger_probability_for_track(self, track_name: str) -> Optional[float]:
        n = track_name.lower()
        if 'bass' in n and 'drum' not in n:
            return _BASS_TRIGGER_PROBABILITY
        if any(token in n for token in ['closedhat', 'openhat', 'pedalhat', 'hihat', 'hi_hat', 'hi-hat']):
            return _HIHAT_TRIGGER_PROBABILITY
        if any(token in n for token in ['kick', 'kicklow', 'kickalt']):
            return _KICK_TRIGGER_PROBABILITY
        if any(token in n for token in ['snare', 'snarealt']):
            return _SNARE_TRIGGER_PROBABILITY
        match = re.search(r"_n(\d+)(?:_|$)", n)
        if match:
            note = int(match.group(1))
            if note in _HIHAT_NOTES:
                return _HIHAT_TRIGGER_PROBABILITY
            if note in _KICK_NOTES:
                return _KICK_TRIGGER_PROBABILITY
            if note in _SNARE_NOTES:
                return _SNARE_TRIGGER_PROBABILITY
        return None

    def _max_patch_attempts_for_track(self, track_name: str) -> int:
        octave_type = _track_octave_type(track_name)
        if octave_type == 'bass':
            return _BASS_PATCH_ATTEMPTS
        if octave_type == 'melody':
            return _OCTAVE_CHECKED_PATCH_ATTEMPTS
        return _DEFAULT_PATCH_ATTEMPTS

    def _thin_track_triggers(self, source_track: mido.MidiTrack, track_name: str) -> Tuple[mido.MidiTrack, Dict]:
        probability = self._trigger_probability_for_track(track_name)
        if probability is None:
            return source_track, {"applied": False}

        thinned = mido.MidiTrack()
        thinned.name = getattr(source_track, 'name', track_name)
        dropped_note_counts: Dict[int, int] = {}
        pending_time = 0
        triggered = 0
        kept = 0
        dropped = 0

        for msg in source_track:
            pending_time += msg.time

            if msg.is_meta:
                thinned.append(msg.copy(time=pending_time))
                pending_time = 0
                continue

            is_note_on = msg.type == 'note_on' and getattr(msg, 'velocity', 0) > 0
            is_note_end = msg.type == 'note_off' or (msg.type == 'note_on' and getattr(msg, 'velocity', 0) == 0)

            if is_note_on:
                triggered += 1
                if random.random() > probability:
                    dropped += 1
                    dropped_note_counts[msg.note] = dropped_note_counts.get(msg.note, 0) + 1
                    continue
                kept += 1
                thinned.append(msg.copy(time=pending_time))
                pending_time = 0
                continue

            if is_note_end and dropped_note_counts.get(msg.note, 0) > 0:
                dropped_note_counts[msg.note] -= 1
                if dropped_note_counts[msg.note] <= 0:
                    del dropped_note_counts[msg.note]
                continue

            thinned.append(msg.copy(time=pending_time))
            pending_time = 0

        if pending_time and len(thinned) > 0:
            thinned[-1].time += pending_time

        if triggered > 0 and kept == 0:
            print(f"    Trigger probability would mute all notes on {track_name}; using original track.")
            return source_track, {
                "applied": False,
                "probability": probability,
                "reason": "all_notes_would_drop",
                "original_triggers": triggered,
            }

        if triggered:
            print(f"    Trigger probability: kept {kept}/{triggered} notes ({probability:.0%}) on {track_name}")

        return thinned, {
            "applied": triggered > 0,
            "probability": probability,
            "original_triggers": triggered,
            "kept_triggers": kept,
            "dropped_triggers": dropped,
        }

    def run_multi_pass(self, midi_file_path: str, song_name: str, skip_tracks: set = None,
                       metadata: Dict = None):
        """
        Implements the Multi-Pass Strategy with individual stem export.
        """
        mid = mido.MidiFile(midi_file_path)
        total_song_duration = mid.length
        batches = self.get_batches(mid)
        all_stem_paths = {}
        tuning_context = self._infer_midi_tuning_context(midi_file_path, metadata)

        print(f"Total Song Duration: {total_song_duration:.2f}s")
        if tuning_context.get("is_armenian"):
            offsets = tuning_context.get("cent_offsets", {})
            print(f"Armenian/Maqam zone tuning: {tuning_context.get('key')} {tuning_context.get('mode')} "
                  f"({len(offsets)} tuned pitch classes)")

        for batch_name, track_specs in batches.items():
            if not track_specs:
                continue

            # Filter user-excluded tracks before building batch MIDI and part assignments
            if skip_tracks:
                track_specs = [s for s in track_specs
                               if not any(t.lower() in s['recorded_name'].lower() for t in skip_tracks)]
            if not track_specs:
                print(f"  Skipping pass {batch_name} — all tracks excluded by user.")
                continue
                
            print(f"\n=== STARTING PASS: {batch_name.upper()} ===")
            
            # 1. Prepare MIDI for this batch
            batch_mid = mido.MidiFile()
            batch_mid.ticks_per_beat = mid.ticks_per_beat
            
            # Copy all meta-messages from track 0 (Tempo, etc.)
            tempo_track = mido.MidiTrack()
            for msg in mid.tracks[0]:
                if msg.is_meta:
                    tempo_track.append(msg)
            batch_mid.tracks.append(tempo_track)
            
            # --- Count-in setup ---
            # All song messages are shifted forward by _COUNT_IN_BEATS so the click
            # precedes bar 1. The click is on Part 16 (channel 15, USB 31/32) and is
            # used post-recording to trim every pass to sample-accurate alignment,
            # compensating for both USB audio latency and sleep-based pre-roll jitter.
            tpb = mid.ticks_per_beat
            bar_ticks = self._bar_ticks_from_midi(mid)
            count_in_ticks = _COUNT_IN_BEATS * tpb

            # Extract tempo for count-in duration calculation
            song_tempo_us = 500000  # default 120 BPM
            for track in mid.tracks:
                for msg in track:
                    if msg.type == 'set_tempo':
                        song_tempo_us = msg.tempo
                        break
                else:
                    continue
                break
            count_in_seconds = mido.tick2second(count_in_ticks, tpb, song_tempo_us)

            recorded_track_names = []
            pass_manifest = {
                "batch": batch_name,
                "midi_file": midi_file_path,
                "song_name": song_name,
                "parts": [],
                "sync": {
                    "part": 16,
                    "midi_channel": _SYNC_CH + 1,
                    "usb_pair": "31/32",
                    "track_name": "__sync__",
                }
            }
            calibration_audit_dir = os.path.join(
                self.recorder.output_dir,
                "calibration_audit",
                self._safe_label(batch_name)
            )
            for i, spec in enumerate(track_specs):
                if i >= 15:  # channel 15 reserved for sync click
                    print(f"  Warning: Skipping {spec['recorded_name']} (Exceeds 15 musical Part limit)")
                    break

                new_track = mido.MidiTrack()
                new_track.name = spec['recorded_name']
                source_track, trigger_probability = self._thin_track_triggers(spec['track'], new_track.name)

                # --- 1. SELECT PATCH + AUDITION/CALIBRATE ---
                patch_info = {}
                sound_design = {}
                calibrated_gain = None
                octave_offset = 0
                hihat_note_offset = 0
                hihat_patch_policy: Dict[str, Any] = self._hihat_patch_policy(new_track.name, {}, 0)
                octave_diagnostics: Dict[str, Any] = {"decision": "not_checked"}
                patch_attempt_history: List[Dict[str, Any]] = []
                max_patch_attempts = self._max_patch_attempts_for_track(new_track.name)
                for patch_attempt in range(max_patch_attempts):
                    patch_info = self.get_patch_for_track_name(new_track.name)
                    attempt_hihat_offset = self._hihat_note_offset_for_patch(new_track.name, patch_info)
                    attempt_source_track = (
                        self._shift_track_notes(source_track, attempt_hihat_offset)
                        if attempt_hihat_offset else source_track
                    )
                    hihat_patch_policy = self._hihat_patch_policy(
                        new_track.name, patch_info, attempt_hihat_offset
                    )
                    print(
                        f"  Mapping {new_track.name} to Part {i+1} "
                        f"(USB {i*2+1}/{i*2+2}) -> {patch_info['name']} "
                        f"[patch attempt {patch_attempt + 1}/{max_patch_attempts}]"
                    )
                    if hihat_patch_policy.get("is_hihat"):
                        if attempt_hihat_offset:
                            print(
                                f"    Hi-hat patch policy: PC {patch_info.get('pc')} "
                                f"allows note articulation shift {attempt_hihat_offset:+d}"
                            )
                        else:
                            print(
                                f"    Hi-hat patch policy: PC {patch_info.get('pc')} "
                                "keeps original MIDI notes"
                            )
                    self.controller.select_patch(i, patch_info['msb'], patch_info['lsb'], patch_info['pc'])
                    self.controller.set_part_level(i, 115)
                    sound_design = {}
                    if patch_info.get('msb') == 97:
                        sound_design = {"model_expansion": patch_info.get("name"), "zcore_edits": "skipped"}
                        print("    Sound design: model expansion selected; skipping Z-Core partial SysEx edits")
                    elif hasattr(self.controller, 'apply_track_sound_design'):
                        sound_design = self.controller.apply_track_sound_design(i, new_track.name) or {}
                        applied_design = [str(v) for v in sound_design.values() if v]
                        if applied_design:
                            print(f"    Sound design: {' | '.join(applied_design)}")

                    tuning_applied = self._apply_armenian_zone_tuning(
                        i, new_track.name, patch_info, tuning_context
                    )
                    if tuning_applied:
                        sound_design["tuning"] = tuning_applied
                        if tuning_applied.get("type") == "custom_scale_tune":
                            print(f"    Tuning: CUSTOM {tuning_applied.get('key')} {tuning_applied.get('mode')} "
                                  f"on Zone {i+1}")
                    else:
                        self._reset_zone_tuning(i)

                    calibrated_gain, octave_offset, octave_diagnostics = self.calibrate_part_gain(
                        i, attempt_source_track, mid.ticks_per_beat, song_tempo_us,
                        track_name=new_track.name,
                        audit_dir=calibration_audit_dir
                    )
                    patch_attempt_history.append({
                        "attempt": patch_attempt + 1,
                        "patch": {
                            "name": patch_info.get("name", "Unknown"),
                            "msb": patch_info.get("msb"),
                            "lsb": patch_info.get("lsb"),
                            "pc": patch_info.get("pc"),
                        },
                        "accepted": calibrated_gain is not None,
                        "gain_db": None if calibrated_gain is None else round(float(calibrated_gain), 2),
                        "octave_offset_semitones": octave_offset,
                        "hihat_note_offset_semitones": attempt_hihat_offset,
                        "hihat_patch_policy": hihat_patch_policy,
                        "octave_decision": octave_diagnostics.get("decision"),
                        "patch_rejected_reason": octave_diagnostics.get("patch_rejected_reason"),
                    })
                    if calibrated_gain is not None:
                        hihat_note_offset = attempt_hihat_offset
                        break

                    bad_key = (patch_info.get('msb'), patch_info.get('lsb'), patch_info.get('pc'))
                    self._bad_patches.add(bad_key)
                    self._used_patches.discard(bad_key)
                    if patch_attempt < max_patch_attempts - 1:
                        retry_reason = octave_diagnostics.get("patch_rejected_reason") or "silent_or_gain_calibration_failed"
                        print(
                            f"    ! Patch rejected ({retry_reason}); "
                            f"trying alternate patch ({patch_attempt + 2}/{max_patch_attempts})"
                        )

                # Handle skipped tracks only after alternate patch attempts are exhausted.
                if calibrated_gain is None:
                    print(f"    ! No audible patch found for {new_track.name}; skipping Part {i+1}.")
                    continue
                recorded_track_names.append(new_track.name)

                # --- 2. PREPARE TRACK FOR PASS ---
                # Push all events forward by count_in_ticks; apply octave and hi-hat articulation offsets if set.
                total_note_offset = octave_offset + hihat_note_offset
                first = True
                for msg in source_track:
                    if first:
                        if not msg.is_meta:
                            if total_note_offset != 0 and msg.type in ('note_on', 'note_off') and hasattr(msg, 'note'):
                                new_track.append(msg.copy(channel=i, note=max(0, min(127, msg.note + total_note_offset)), time=msg.time + count_in_ticks))
                            else:
                                new_track.append(msg.copy(channel=i, time=msg.time + count_in_ticks))
                        else:
                            new_track.append(msg.copy(time=msg.time + count_in_ticks))
                        first = False
                    else:
                        if not msg.is_meta:
                            if total_note_offset != 0 and msg.type in ('note_on', 'note_off') and hasattr(msg, 'note'):
                                new_track.append(msg.copy(channel=i, note=max(0, min(127, msg.note + total_note_offset))))
                            else:
                                new_track.append(msg.copy(channel=i))
                        else:
                            new_track.append(msg)
                batch_mid.tracks.append(new_track)
                automation_track, sysex_automation = self._build_part_sysex_automation(
                    i, new_track.name, patch_info, tpb, bar_ticks, count_in_ticks
                )
                if automation_track is not None:
                    batch_mid.tracks.append(automation_track)
                    print(
                        f"    Melody SysEx automation: {sysex_automation.get('event_count', 0)} events "
                        f"on {automation_track.name}"
                    )

                pass_manifest["parts"].append({
                    "source_track_index": spec['source_index'],
                    "source_track_name": spec['source_name'],
                    "recorded_track_name": new_track.name,
                    "part": i + 1,
                    "midi_channel": i + 1,
                    "usb_pair": f"{i*2+1}/{i*2+2}",
                    "calibrated_gain_db": calibrated_gain,
                    "octave_offset_semitones": octave_offset,
                    "hihat_note_offset_semitones": hihat_note_offset,
                    "total_note_offset_semitones": total_note_offset,
                    "hihat_patch_policy": hihat_patch_policy,
                    "octave_calibration": octave_diagnostics,
                    "layer": {
                        "family": spec.get('layer_family'),
                        "index": spec.get('layer_index'),
                        "count": spec.get('layer_count', 1),
                        "category": spec.get('layer_category'),
                        "variation": spec.get('layer_variation', {}),
                    },
                    "trigger_probability": trigger_probability,
                    "sound_design": sound_design,
                    "sound_design_raw": sound_design.get("_raw"),
                    "melody_sysex_automation": sysex_automation,
                    "patch": {
                        "name": patch_info.get("name", "Unknown"),
                        "msb": patch_info.get("msb"),
                        "lsb": patch_info.get("lsb"),
                        "pc": patch_info.get("pc"),
                    },
                    "patch_attempts": patch_attempt_history,
                    "max_patch_attempts": max_patch_attempts,
                })

            # Build 4-beat count-in click track on channel 15 / Part 16
            # Beat 1 = accent (vel 127), beats 2-4 = softer (vel 90)
            note_dur = tpb // 4   # 1/16-note duration for sharp transient
            gap      = tpb - note_dur
            sync_track = mido.MidiTrack()
            sync_track.append(mido.MetaMessage('track_name', name='__sync__', time=0))
            for beat in range(_COUNT_IN_BEATS):
                vel = 127 if beat == 0 else 90
                delta_on = 0 if beat == 0 else gap
                sync_track.append(mido.Message('note_on',  channel=_SYNC_CH, note=_SYNC_NOTE, velocity=vel,  time=delta_on))
                sync_track.append(mido.Message('note_off', channel=_SYNC_CH, note=_SYNC_NOTE, velocity=0,    time=note_dur))
            batch_mid.tracks.append(sync_track)
            self.controller.select_patch(_SYNC_CH, _SYNC_SOUND['msb'], _SYNC_SOUND['lsb'], _SYNC_SOUND['pc'])
            self.controller.set_part_level(_SYNC_CH, 127)
            print(f"  4-beat count-in → Part 16 (USB 31/32) [{_SYNC_SOUND['name']}] | bar 1 at +{count_in_seconds:.3f}s")

            # Save the exact MIDI used for this recording pass so octave offsets can be audited later.
            recording_midi = os.path.join(
                self.recorder.output_dir,
                f"{song_name}_{batch_name}_recording_pass.mid"
            )
            batch_mid.save(recording_midi)
            pass_manifest["recording_midi"] = recording_midi
            manifest_path = os.path.join(self.recorder.output_dir, f"{song_name}_{batch_name}_manifest.json")
            with open(manifest_path, "w") as f:
                json.dump(pass_manifest, f, indent=2)
            print(f"  Pass manifest: {manifest_path}")
            
            # 2. Record this pass (32-ch master); include count-in in total duration
            output_filename = f"{song_name}_{batch_name}_pass.wav"
            actual_pre_roll = self.recorder.play_midi_and_record(
                recording_midi,
                output_filename,
                self.controller.port_name,
                total_song_duration + count_in_seconds
            )
            self._reset_calibration_controllers()

            # 3. Split stems — coarse pre-roll trim then fine sync via click detection
            batch_stems, sync_info = self.recorder.split_stems(
                output_filename, recorded_track_names, batch_name,
                pre_roll_seconds=actual_pre_roll,
                count_in_seconds=count_in_seconds,
                return_sync_info=True,
            )
            pass_manifest["sync"].update(sync_info)
            # Re-save manifest with sync diagnostics
            with open(manifest_path, "w") as f:
                json.dump(pass_manifest, f, indent=2)
            all_stem_paths.update(batch_stems)
            
        print("\nMulti-Pass Phased Recording & Stem Export Complete.")
        return all_stem_paths

    def _pick_unique(self, categories: list, extra_pool: Optional[List[dict]] = None) -> dict:
        """Pick a random patch from the given category list, avoiding repeats within the session."""
        pool = []
        for cat in categories:
            pool.extend(self.controller.sound_db.get(cat, []))
        if extra_pool:
            pool.extend(extra_pool)
        random.shuffle(pool)
        for p in pool:
            key = (p['msb'], p['lsb'], p['pc'])
            if key in self._bad_patches:
                continue
            if key not in self._used_patches:
                self._used_patches.add(key)
                return p.copy()
        # All good sounds exhausted — allow repeats, but do not revisit known-silent patches.
        usable_pool = [p for p in pool if (p['msb'], p['lsb'], p['pc']) not in self._bad_patches]
        return random.choice(usable_pool).copy() if usable_pool else {"name": "Default", "msb": 87, "lsb": 64, "pc": 1}

    def _pick_unique_from_pool(self, pool: List[dict]) -> dict:
        candidates = [p.copy() for p in pool]
        random.shuffle(candidates)
        for p in candidates:
            key = (p['msb'], p['lsb'], p['pc'])
            if key in self._bad_patches:
                continue
            if key not in self._used_patches:
                self._used_patches.add(key)
                return p
        usable_candidates = [p for p in candidates if (p['msb'], p['lsb'], p['pc']) not in self._bad_patches]
        if usable_candidates:
            return random.choice(usable_candidates)
        return {"name": "Default", "msb": 87, "lsb": 64, "pc": 1}

    def _model_expansion_pool(self, roles: List[str]) -> List[dict]:
        pool = []
        aliases = {
            'strings': ['pad'],
            'choir': ['pad'],
            'pluck': ['poly'],
        }
        for role in roles:
            for mapped_role in aliases.get(role, [role]):
                pool.extend(_MODEL_EXPANSION_PATCHES.get(mapped_role, []))
        return [p.copy() for p in pool]

    def _pick_role_patch(self, categories: List[str], model_roles: Optional[List[str]] = None) -> dict:
        return self._pick_unique(categories, self._model_expansion_pool(model_roles or categories))

    @staticmethod
    def _patch_key(patch_info: Dict) -> Tuple[Any, Any, Any]:
        return (patch_info.get('msb'), patch_info.get('lsb'), patch_info.get('pc'))

    @staticmethod
    def _is_hihat_track_name(track_name: str) -> bool:
        n = (track_name or '').lower().replace('-', '_').replace(' ', '_')
        if any(token in n for token in ('closedhat', 'openhat', 'pedalhat', 'hihat', 'closed_hat', 'open_hat')):
            return True
        return re.search(r'(^|_)hat(_|$)', n) is not None

    def _pick_hihat_patch(self) -> dict:
        pool = [
            p.copy() for p in self.controller.sound_db.get('drums', [])
            if self._patch_key(p) not in _HIHAT_BLOCKED_PATCH_KEYS
        ]
        if not pool:
            pool = [
                {'name': 'HiHat Cymbal', 'msb': 86, 'lsb': 65, 'pc': 69},
                {'name': 'HiHat Cymbal w', 'msb': 86, 'lsb': 65, 'pc': 70},
            ]
        return self._pick_unique_from_pool(pool)

    def _hihat_note_offset_for_patch(self, track_name: str, patch_info: Dict) -> int:
        return 0

    def _hihat_patch_policy(self, track_name: str, patch_info: Dict, note_offset: int) -> Dict[str, Any]:
        is_hihat = self._is_hihat_track_name(track_name)
        patch_key = self._patch_key(patch_info) if patch_info else None
        return {
            "is_hihat": is_hihat,
            "blocked_patch_pcs": [63, 64, 65, 66, 67, 68, 71, 72, 73, 74],
            "special_articulation_patch_pcs": [69, 70],
            "selected_patch_pc": patch_info.get("pc") if patch_info else None,
            "selected_patch_allowed": (patch_key not in _HIHAT_BLOCKED_PATCH_KEYS) if is_hihat and patch_info else None,
            "note_shift_allowed": bool(is_hihat and patch_key in _HIHAT_SPECIAL_PATCH_KEYS),
            "note_shift_range": list(_HIHAT_SPECIAL_NOTE_SHIFT_RANGE),
            "note_offset_semitones": int(note_offset),
        }

    def _eastern_percussion_note(self, track_name: str) -> Optional[int]:
        match = re.search(r"_n(\d+)(?:_|$)", track_name.lower())
        if not match:
            return None
        note = int(match.group(1))
        return note if note in _EASTERN_PERCUSSION_NOTES else None

    def _eastern_percussion_patch_pool(self, track_name: str) -> Optional[List[dict]]:
        n = track_name.lower()
        note = self._eastern_percussion_note(track_name)
        if note is None or 'drum_aux_' not in n:
            return None

        if 'bongo' in n or 'conga' in n or note in {60, 61, 62}:
            groups = ['tabla', 'dholak', 'dhol', 'madal', 'conga']
        elif 'tambourine' in n or 'maracas' in n or note in {54, 70}:
            groups = ['tabla', 'afro', 'wood']
        else:
            groups = ['tabla', 'dholak', 'dhol', 'madal', 'afro', 'conga', 'wood']

        pool = []
        for group in groups:
            pool.extend(_EASTERN_PERCUSSION_PATCHES.get(group, []))
        return pool or _EASTERN_PERCUSSION_FALLBACK

    def _layer_patch_categories(self, track_name: str) -> Optional[List[str]]:
        match = re.search(r"_layer\d+_([a-z]+)$", track_name.lower())
        if not match:
            return None
        category = match.group(1)
        if category in self.controller.sound_db:
            return [category]
        return None

    def get_patch_for_track_name(self, track_name):
        n = track_name.lower()
        layer_categories = self._layer_patch_categories(track_name)
        if layer_categories:
            return self._pick_role_patch(layer_categories)

        eastern_pool = self._eastern_percussion_patch_pool(track_name)
        if eastern_pool:
            return self._pick_unique_from_pool(eastern_pool)

        if 'harmonic bass' in n:
            # Harmonic bass: synth bass pool
            return self._pick_role_patch(['bass'])
        elif 'bass' in n:
            # Bass: synth + electric bass pool
            return self._pick_role_patch(['bass'])
        elif 'fx' in n or 'melody fx' in n:
            # FX track: pulsating / synth FX
            return self._pick_role_patch(['fx', 'poly'])
        elif 'chorus melody' in n:
            # Chorus melody: immediate poly/brass patches; pads have slow attacks.
            return self._pick_role_patch(['poly', 'brass', 'lead'])
        elif 'counter melody' in n:
            # Counter melody: leads or poly keys
            return self._pick_role_patch(['lead', 'poly'])
        elif 'melody' in n or 'lead' in n:
            # Main melody: leads, poly keys, brass
            return self._pick_role_patch(['lead', 'poly', 'brass'])
        elif 'pad' in n or 'chord' in n:
            # Pads / chords: pad, strings, choir
            return self._pick_role_patch(['pad', 'strings', 'choir'], ['pad'])
        elif self._is_hihat_track_name(n):
            return self._pick_hihat_patch()
        elif any(x in n for x in ['drum', 'kick', 'snare', 'hat', 'perc',
                                   'stick', 'clap', 'bongo', 'conga',
                                   'tambourine', 'maracas']):
            return self._pick_unique(['drums'])
        else:
            return self._pick_unique(['poly', 'lead'])

if __name__ == "__main__":
    pass
