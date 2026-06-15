#!/usr/bin/env python3
"""
Tech House Stem Recorder
========================
Records individual stems from the Roland Fantom via USB audio.
Each MIDI track gets its own Fantom Part → USB stereo pair → WAV stem.

Fantom USB Audio Mapping (Parallel Mode):
  Part 1  → USB 1/2    Part 9  → USB 17/18
  Part 2  → USB 3/4    Part 10 → USB 19/20
  Part 3  → USB 5/6    Part 11 → USB 21/22
  Part 4  → USB 7/8    Part 12 → USB 23/24
  Part 5  → USB 9/10   Part 13 → USB 25/26
  Part 6  → USB 11/12  Part 14 → USB 27/28
  Part 7  → USB 13/14  Part 15 → USB 29/30
  Part 8  → USB 15/16  Part 16 → USB 31/32 (sync click)

Usage:
    cd "/Volumes/Raphael/Tech House"
    source venv/bin/activate
    python3 record_stems.py output/TH_0613_1529_128_Gmin.mid --bpm 128
"""
import os
import sys
import json
import time
import threading
import argparse
import random
from pathlib import Path
from typing import List, Dict, Optional

import mido
import numpy as np
import sounddevice as sd
import soundfile as sf

sys.path.insert(0, 'audio_pipeline')
from fantom_midi_control import (
    FantomController, _addr_add, _nibbles, _signed_100, _signed_63,
    _percent_to_127, create_roland_sysex
)

# ============================================================================
# CONFIGURATION
# ============================================================================

FANTOM_DEVICE_INDEX = 7       # sounddevice index for FANTOM-6 7 8
SAMPLE_RATE = 48000            # Fantom USB audio sample rate
CHANNELS = 32                  # 16 stereo pairs
SYNC_CH = 15                   # Channel 15 (Part 16) for sync click
SYNC_NOTE = 37                 # Rimshot for sync click
COUNT_IN_BEATS = 4             # 4-beat count-in before song
USB_PAIRS_PER_PART = 2         # Each Part gets a stereo USB pair

# Track name → patch category mapping
TRACK_PATCH_MAP = {
    'bass':       {'category': 'house_bass',   'msb': 87, 'lsb': 92, 'pc': 122},
    'sub':        {'category': 'sub_bass',      'msb': 87, 'lsb': 92, 'pc': 104},
    'acid':       {'category': 'acid_bass',     'msb': 87, 'lsb': 68, 'pc': 89},
    'stab':       {'category': 'chord_stab',    'msb': 87, 'lsb': 92, 'pc': 85},
    'pad':        {'category': 'dark_pad',      'msb': 87, 'lsb': 65, 'pc': 26},
    'fx':         {'category': 'fx',            'msb': 87, 'lsb': 92, 'pc': 126},
    'kick':       {'category': 'drum_kit',      'msb': 86, 'lsb': 65, 'pc': 45},
    'snare':      {'category': 'drum_kit',      'msb': 86, 'lsb': 65, 'pc': 45},
    'clap':       {'category': 'drum_kit',      'msb': 86, 'lsb': 65, 'pc': 45},
    'hat':        {'category': 'drum_kit',      'msb': 86, 'lsb': 65, 'pc': 45},
    'crash':      {'category': 'drum_kit',      'msb': 86, 'lsb': 65, 'pc': 45},
    'ride':       {'category': 'drum_kit',      'msb': 86, 'lsb': 65, 'pc': 45},
    'tambourine': {'category': 'drum_kit',      'msb': 86, 'lsb': 65, 'pc': 45},
    'shaker':     {'category': 'drum_kit',      'msb': 86, 'lsb': 65, 'pc': 45},
    'maracas':    {'category': 'drum_kit',      'msb': 86, 'lsb': 65, 'pc': 45},
    'sidestick':  {'category': 'drum_kit',      'msb': 86, 'lsb': 65, 'pc': 45},
}

# Load tech house patches
PATCH_DB = None
def load_patch_db():
    global PATCH_DB
    if PATCH_DB is None:
        with open('skills/tech_house_fantom_patches.json') as f:
            PATCH_DB = json.load(f)
    return PATCH_DB


# ============================================================================
# TRACK CLASSIFICATION
# ============================================================================

def classify_track(track_name: str) -> Dict:
    """Classify a MIDI track name into a role with patch info."""
    n = track_name.lower()
    db = load_patch_db()

    # Melodic/synth tracks
    if 'bass' in n and 'sub' in n:
        return {'role': 'sub_bass', 'channel_type': 'melodic', 'patch_key': 'sub_bass'}
    elif 'bass' in n and 'acid' in n:
        return {'role': 'acid_bass', 'channel_type': 'melodic', 'patch_key': 'acid_bass'}
    elif 'bass' in n:
        return {'role': 'bass', 'channel_type': 'melodic', 'patch_key': 'house_bass'}
    elif 'stab' in n or 'chord' in n:
        return {'role': 'stab', 'channel_type': 'melodic', 'patch_key': 'chord_stab'}
    elif 'pad' in n:
        return {'role': 'pad', 'channel_type': 'melodic', 'patch_key': 'dark_pad'}
    elif 'acid' in n:
        return {'role': 'acid', 'channel_type': 'melodic', 'patch_key': 'acid_bass'}
    elif 'fx' in n:
        return {'role': 'fx', 'channel_type': 'melodic', 'patch_key': 'fx'}

    # Drum tracks (from exploded drum system)
    elif 'drum1_' in n or 'drum2_' in n or 'drum_aux_' in n:
        # Determine drum sound from track name
        if 'kick' in n:
            return {'role': 'kick', 'channel_type': 'drum', 'patch_key': 'drum_kit'}
        elif 'snare' in n:
            return {'role': 'snare', 'channel_type': 'drum', 'patch_key': 'drum_kit'}
        elif 'clap' in n:
            return {'role': 'clap', 'channel_type': 'drum', 'patch_key': 'drum_kit'}
        elif 'closedhat' in n or 'closed_hat' in n:
            return {'role': 'closed_hat', 'channel_type': 'drum', 'patch_key': 'drum_kit'}
        elif 'openhat' in n or 'open_hat' in n:
            return {'role': 'open_hat', 'channel_type': 'drum', 'patch_key': 'drum_kit'}
        elif 'crash' in n:
            return {'role': 'crash', 'channel_type': 'drum', 'patch_key': 'drum_kit'}
        elif 'ride' in n:
            return {'role': 'ride', 'channel_type': 'drum', 'patch_key': 'drum_kit'}
        elif 'tambourine' in n:
            return {'role': 'tambourine', 'channel_type': 'drum', 'patch_key': 'drum_kit'}
        elif 'shaker' in n or 'maracas' in n:
            return {'role': 'shaker', 'channel_type': 'drum', 'patch_key': 'drum_kit'}
        elif 'sidestick' in n or 'rim' in n:
            return {'role': 'sidestick', 'channel_type': 'drum', 'patch_key': 'drum_kit'}
        elif 'tom' in n:
            return {'role': 'tom', 'channel_type': 'drum', 'patch_key': 'drum_kit'}
        else:
            return {'role': 'drum_other', 'channel_type': 'drum', 'patch_key': 'drum_kit'}

    return {'role': 'unknown', 'channel_type': 'melodic', 'patch_key': 'house_bass'}


def select_patch_for_track(track_info: Dict) -> Dict:
    """Select a random patch from the curated tech house database."""
    db = load_patch_db()
    key = track_info['patch_key']

    # Map patch_key to actual JSON structure
    key_map = {
        'house_bass': ('tech_house_bass', 'house'),
        'sub_bass': ('tech_house_bass', 'sub'),
        'acid_bass': ('tech_house_bass', 'acid'),
        'wobble_bass': ('tech_house_bass', 'wobble'),
        'chord_stab': ('tech_house_stab', None),
        'dark_pad': ('tech_house_pad', None),
        'acid_lead': ('tech_house_acid_lead', None),
        'drum_kit': ('tech_house_drums', None),
        'fx': ('tech_house_pad', None),  # Fallback to pad for FX
    }

    mapping = key_map.get(key)
    if mapping:
        category, subcategory = mapping
        if subcategory:
            pool = db.get(category, {}).get(subcategory, [])
        else:
            pool = db.get(category, [])
    else:
        pool = []

    if not pool:
        # Fallback
        for fallback in ['tech_house_bass.house', 'tech_house_drums']:
            parts = fallback.split('.')
            if len(parts) == 2:
                pool = db.get(parts[0], {}).get(parts[1], [])
            else:
                pool = db.get(parts[0], [])
            if pool:
                break

    if not pool:
        return {'name': 'Default', 'msb': 87, 'lsb': 64, 'pc': 1}

    return random.choice(pool)


# ============================================================================
# SOUND DESIGN
# ============================================================================

def apply_sound_design(fc: FantomController, part_idx: int, track_info: Dict):
    """Apply tech house sound design to a Fantom Part."""
    role = track_info['role']
    base = fc._zcore_base(part_idx)

    if role in ('bass', 'sub_bass', 'acid_bass', 'acid'):
        # Bass tone: HPF filter type (type 3) to cut sub rumble
        # Use HPF instead of LPF for bass — lets mids/highs through
        fc._send_dt1(_addr_add(base, [0x00, 0x20, 0x00]), [3])  # HPF type
        fc._send_dt1(_addr_add(base, [0x00, 0x20, 0x01]), [random.randint(50, 70)])  # Cutoff: 50-70 (higher = more bass cut)
        fc._send_dt1(_addr_add(base, [0x00, 0x20, 0x02]), [random.randint(10, 25)])  # Resonance
        # Envelope
        fc._send_dt1(_addr_add(base, [0x00, 0x24, 0x00]), [0])
        fc._send_dt1(_addr_add(base, [0x00, 0x24, 0x01]), [random.randint(35, 55)])
        fc._send_dt1(_addr_add(base, [0x00, 0x24, 0x02]), [random.randint(55, 80)])
        fc._send_dt1(_addr_add(base, [0x00, 0x24, 0x03]), [random.randint(15, 35)])
        # Step LFO matrix
        _apply_step_lfo(fc, part_idx, role)
        # Super Filter MFX for acid (303-style resonant filter)
        if role == 'acid':
            _apply_super_filter_mfx(fc, part_idx)
        # EQ: Aggressive low cut, boost mids and highs
        fc.set_zone_eq_switch(part_idx + 1, True)
        fc.set_zone_eq_gain(part_idx + 1, 'low', -12.0)   # Heavy low cut
        fc.set_zone_eq_gain(part_idx + 1, 'mid', +6.0)    # Strong mid boost
        fc.set_zone_eq_gain(part_idx + 1, 'high', +4.0)   # High boost

    elif role == 'stab':
        # Short envelope
        fc._send_dt1(_addr_add(base, [0x00, 0x24, 0x00]), [0])
        fc._send_dt1(_addr_add(base, [0x00, 0x24, 0x01]), [random.randint(40, 55)])
        fc._send_dt1(_addr_add(base, [0x00, 0x24, 0x02]), [0])
        fc._send_dt1(_addr_add(base, [0x00, 0x24, 0x03]), [random.randint(10, 20)])
        # Chorus MFX
        _apply_chorus_mfx(fc, part_idx)
        # EQ: Boost presence
        fc.set_zone_eq_switch(part_idx + 1, True)
        fc.set_zone_eq_gain(part_idx + 1, 'low', -3.0)    # Cut lows
        fc.set_zone_eq_gain(part_idx + 1, 'mid', +4.0)    # Boost mids for presence
        fc.set_zone_eq_gain(part_idx + 1, 'high', +3.0)   # Boost highs for air

    elif role == 'pad':
        # Long envelope, slow LFO
        fc._send_dt1(_addr_add(base, [0x00, 0x24, 0x00]), [random.randint(20, 50)])
        fc._send_dt1(_addr_add(base, [0x00, 0x24, 0x01]), [random.randint(60, 90)])
        fc._send_dt1(_addr_add(base, [0x00, 0x24, 0x02]), [random.randint(70, 100)])
        fc._send_dt1(_addr_add(base, [0x00, 0x24, 0x03]), [random.randint(40, 70)])
        # Slow S&H LFO for evolution
        _apply_pad_lfo(fc, part_idx)
        # EQ: Cut lows, boost highs
        fc.set_zone_eq_switch(part_idx + 1, True)
        fc.set_zone_eq_gain(part_idx + 1, 'low', -4.0)
        fc.set_zone_eq_gain(part_idx + 1, 'high', +2.0)


def _apply_step_lfo(fc, part_idx, role):
    """Apply S&H step LFO matrix — psychedelic filter modulation."""
    block = _addr_add(fc._zcore_base(part_idx), [0x00, 0x30, 0x00])

    # LFO1: S&H — tempo-synced for rhythmic filter patterns
    # Rate determines how fast the filter sweeps (lower = slower, more psychedelic)
    if role in ('acid_bass', 'acid'):
        rate1 = 7  # Slower for acid — more hypnotic
    elif role in ('pad', 'stab'):
        rate1 = 10  # Medium for pads — evolving texture
    else:
        rate1 = 12  # Faster for bass — rhythmic pump
    
    fc._send_dt1(_addr_add(block, [0x00, 0x00]), [6])  # S&H waveform
    fc._send_dt1(_addr_add(block, [0x00, 0x01]), [1])  # Sync ON (tempo-synced)
    fc._send_dt1(_addr_add(block, [0x00, 0x02]), [rate1])  # Rate (tempo division)
    
    # LFO1 depth — AGGRESSIVE for psychedelic filter sweep
    # This controls how much the filter opens/closes
    fc._send_dt1(_addr_add(block, [0x00, 0x19]), _nibbles(_signed_100(random.randint(25, 45)), 2))  # Filter depth
    fc._send_dt1(_addr_add(block, [0x00, 0x1B]), _nibbles(_signed_100(random.randint(8, 15)), 2))  # Resonance depth

    # LFO2: Smooth — for subtle movement
    fc._send_dt1(_addr_add(block, [0x00, 0x4F]), [random.choice([0, 1, 2])])  # Sin/Tri/Saw
    fc._send_dt1(_addr_add(block, [0x00, 0x50]), [1])  # Sync ON
    fc._send_dt1(_addr_add(block, [0x00, 0x51]), [random.choice([15, 18, 21])])  # Slower rate
    fc._send_dt1(_addr_add(block, [0x00, 0x68]), _nibbles(_signed_100(random.randint(10, 20)), 2))  # Depth
    fc._send_dt1(_addr_add(block, [0x00, 0x6A]), _nibbles(_signed_100(random.randint(4, 10)), 2))

    # Matrix — AGGRESSIVE modulation routing for psychedelic flavor
    # LFO1 → CUTOFF (main filter sweep)
    fc._send_dt1(_addr_add(block, [0x00, 0x56]), [104])  # LFO1 source
    fc._send_dt1(_addr_add(block, [0x00, 0x57]), [2])    # → CUT (filter cutoff)
    fc._send_dt1(_addr_add(block, [0x00, 0x58]), [_signed_63(random.randint(25, 45))])  # AGGRESSIVE depth
    
    # LFO1 → LFO2 RATE (cross-modulation for evolving patterns)
    fc._send_dt1(_addr_add(block, [0x00, 0x59]), [17])   # → LFO2-RATE
    fc._send_dt1(_addr_add(block, [0x00, 0x5A]), [_signed_63(random.randint(10, 20))])  # Cross-mod depth

    # LFO2 → FAT (harmonic richness)
    fc._send_dt1(_addr_add(block, [0x00, 0x5F]), [105])  # LFO2 source
    fc._send_dt1(_addr_add(block, [0x00, 0x60]), [36])   # → FAT
    fc._send_dt1(_addr_add(block, [0x00, 0x61]), [_signed_63(random.randint(8, 18))])  # Depth
    
    # LFO2 → PAN (stereo movement)
    fc._send_dt1(_addr_add(block, [0x00, 0x62]), [5])    # → PAN
    fc._send_dt1(_addr_add(block, [0x00, 0x63]), [_signed_63(random.randint(6, 14))])  # Pan depth
    fc._send_dt1(_addr_add(block, [0x00, 0x63]), [_signed_63(random.randint(4, 8))])


def _apply_chorus_mfx(fc, part_idx):
    """Apply chorus MFX for stabs."""
    fc._set_mfx_type_and_params(fc._tone_mfx_base(part_idx), 23, {
        1: 1, 2: 10, 3: 15, 4: 1, 6: 12,
        7: random.randint(20, 30), 8: 120,
        9: 15, 10: 16, 11: random.randint(20, 28), 12: 100,
    })


def _apply_super_filter_mfx(fc, part_idx):
    """Apply Super Filter MFX (type 5) for acid bass — 303-style resonant filter."""
    fc._set_mfx_type_and_params(fc._tone_mfx_base(part_idx), 5, {
        0: random.randint(70, 80),    # Cutoff (70-80 = moderate-open)
        1: random.randint(25, 40),    # Resonance (25-40 = squelchy)
        2: 1,                         # Filter type (1 = LPF)
        3: random.randint(15, 25),    # Depth (modulation depth)
        4: 1,                         # Sync on (tempo-synced)
        5: random.randint(6, 10),     # Rate sync (6=1/8, 8=1/4, 10=1/2)
        6: random.randint(0, 3),      # Waveform (0=sin, 1=tri, 2=saw, 3=sqr)
        7: 100,                       # Balance (100% wet)
        8: 100,                       # Output level
    })


def _apply_pad_lfo(fc, part_idx):
    """Apply slow S&H LFO for pad evolution."""
    block = _addr_add(fc._zcore_base(part_idx), [0x00, 0x30, 0x00])
    fc._send_dt1(_addr_add(block, [0x00, 0x00]), [6])  # S&H
    fc._send_dt1(_addr_add(block, [0x00, 0x01]), [1])
    fc._send_dt1(_addr_add(block, [0x00, 0x02]), [18])  # 1 bar
    fc._send_dt1(_addr_add(block, [0x00, 0x19]), _nibbles(_signed_100(random.randint(8, 16)), 2))

    fc._send_dt1(_addr_add(block, [0x00, 0x4F]), [0])  # LFO2: SIN
    fc._send_dt1(_addr_add(block, [0x00, 0x50]), [1])
    fc._send_dt1(_addr_add(block, [0x00, 0x51]), [21])  # 2 bars
    fc._send_dt1(_addr_add(block, [0x00, 0x68]), _nibbles(_signed_100(random.randint(6, 12)), 2))

    fc._send_dt1(_addr_add(block, [0x00, 0x56]), [104])
    fc._send_dt1(_addr_add(block, [0x00, 0x57]), [2])   # CUT
    fc._send_dt1(_addr_add(block, [0x00, 0x58]), [_signed_63(random.randint(6, 14))])
    fc._send_dt1(_addr_add(block, [0x00, 0x59]), [36])  # FAT
    fc._send_dt1(_addr_add(block, [0x00, 0x5A]), [_signed_63(random.randint(4, 8))])

    fc._send_dt1(_addr_add(block, [0x00, 0x5F]), [105])
    fc._send_dt1(_addr_add(block, [0x00, 0x60]), [5])   # PAN
    fc._send_dt1(_addr_add(block, [0x00, 0x61]), [_signed_63(random.randint(4, 10))])
    fc._send_dt1(_addr_add(block, [0x00, 0x62]), [3])   # RES
    fc._send_dt1(_addr_add(block, [0x00, 0x63]), [_signed_63(random.randint(3, 8))])


# ============================================================================
# LEVEL CALIBRATION
# ============================================================================

def measure_peak(duration: float = 3.0, channel_indices: List[int] = None) -> float:
    """Record a short snippet and return peak amplitude on specified channels."""
    if channel_indices is None:
        channel_indices = [0, 1]
    frames = int(duration * SAMPLE_RATE)
    chunk = 4096
    recorded = []
    try:
        with sd.InputStream(device=FANTOM_DEVICE_INDEX, channels=CHANNELS,
                           samplerate=SAMPLE_RATE, dtype='float32',
                           blocksize=chunk) as stream:
            remaining = frames
            while remaining > 0:
                to_read = min(chunk, remaining)
                data, _ = stream.read(to_read)
                recorded.append(data)
                remaining -= len(data)
    except Exception as e:
        print(f"      measure_peak error: {e}")
        return 0.0
    if not recorded:
        return 0.0
    audio = np.concatenate(recorded, axis=0)
    relevant = audio[:, channel_indices]
    return float(np.max(np.abs(relevant))) if relevant.size > 0 else 0.0


def build_calibration_snippet(track: mido.MidiTrack, part_idx: int,
                               tpb: int, bpm: float) -> mido.MidiFile:
    """
    Build a short dense MIDI snippet for gain calibration.
    Extracts unique notes from the track and plays them in rapid succession
    so even sparse loops produce a reliable peak measurement.
    """
    # Collect unique note/velocity pairs from the track
    seen_notes = set()
    note_events = []
    for msg in track:
        if msg.is_meta or not hasattr(msg, 'note'):
            continue
        if msg.type == 'note_on' and msg.velocity > 0:
            key = msg.note
            if key not in seen_notes:
                seen_notes.add(key)
                note_events.append((msg.note, msg.velocity))

    if not note_events:
        return None

    # Build a 2-bar snippet with notes packed densely
    snippet = mido.MidiFile()
    snippet.ticks_per_beat = tpb
    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))
    snippet.tracks.append(tempo_track)

    note_track = mido.MidiTrack()
    note_dur = tpb // 2  # 8th note duration
    gap = tpb // 4       # 16th note gap between notes

    for i, (note, vel) in enumerate(note_events[:16]):  # Max 16 unique notes
        # On
        note_track.append(mido.Message('note_on', channel=part_idx,
                                       note=note, velocity=vel,
                                       time=gap if i > 0 else 0))
        # Off
        note_track.append(mido.Message('note_off', channel=part_idx,
                                       note=note, velocity=0, time=note_dur))

    # Pad to 2 bars
    remaining = tpb * 4 - (len(note_events[:16]) * (gap + note_dur))
    if remaining > 0:
        note_track.append(mido.Message('note_off', channel=part_idx,
                                       note=0, velocity=0, time=remaining))

    snippet.tracks.append(note_track)
    return snippet


def play_snippet(snippet: mido.MidiFile, port_name: str, duration: float = 2.5):
    """Play a MIDI snippet through the Fantom."""
    outport = mido.open_output(port_name)
    start = time.time()
    for msg in snippet.play():
        outport.send(msg)
    elapsed = time.time() - start
    # All notes off
    for ch in range(16):
        outport.send(mido.Message('control_change', channel=ch, control=123, value=0))
    outport.close()


def calibrate_part_gain(fc: FantomController, part_idx: int, track: mido.MidiTrack,
                        tpb: int, bpm: float, port_name: str,
                        is_drum: bool = False) -> Optional[float]:
    """
    Iteratively adjust Fantom zone EQ gain until peak is near -6 dBFS.
    Returns calibrated gain in dB, or None if silent/rejected.
    For drum parts, skip calibration (drums are velocity-controlled).
    """
    # Drums: skip calibration, use default gain
    if is_drum:
        fc.set_zone_eq_switch(part_idx + 1, True)
        fc.set_zone_eq_gain(part_idx + 1, 'input', 0.0)  # No gain boost — prevent clipping
        return 0.0

    target_peak = 0.5   # -6 dBFS
    tolerance = 0.05     # ±0.05 amplitude
    max_attempts = 6
    current_gain = 0.0

    # Build calibration snippet
    snippet = build_calibration_snippet(track, part_idx, tpb, bpm)
    if snippet is None:
        print(f"    Cal: Part {part_idx+1} — no notes, skipping")
        return None

    # Enable zone EQ
    fc.set_zone_eq_switch(part_idx + 1, True)
    time.sleep(0.05)

    usb_left = part_idx * 2
    usb_right = part_idx * 2 + 1

    print(f"    Cal: Part {part_idx+1} (USB {usb_left+1}/{usb_right+1}) ", end='')

    for attempt in range(max_attempts):
        # Set gain
        fc.set_zone_eq_gain(part_idx + 1, 'input', current_gain)
        time.sleep(0.1)

        # Play and measure
        play_thread = threading.Thread(target=play_snippet,
                                       args=(snippet, port_name, 2.5))
        play_thread.start()
        peak = measure_peak(duration=3.0, channel_indices=[usb_left, usb_right])
        play_thread.join()

        # Silence detection on first attempt
        if attempt == 0 and peak < 0.005:
            print(f"— SILENT (peak={peak:.4f}), skipping")
            return None

        peak_db = 20 * np.log10(max(1e-5, peak))
        target_db = 20 * np.log10(target_peak)
        diff_db = target_db - peak_db

        # Check tolerance
        if abs(peak - target_peak) < tolerance:
            print(f"→ {current_gain:+.1f}dB (peak={peak_db:.1f}dB) ✓")
            return current_gain

        # Proportional adjustment
        if peak < 0.001:
            current_gain += 12.0
        else:
            current_gain += diff_db

        # Clamp to hardware limits
        current_gain = max(-24.0, min(24.0, current_gain))

        if (current_gain == 24.0 and diff_db > 0) or (current_gain == -24.0 and diff_db < 0):
            break

    print(f"→ {current_gain:+.1f}dB (limit)")
    return current_gain


# ============================================================================
# MIDI TRACK ANALYSIS
# ============================================================================

def get_recordable_tracks(midi_path: str) -> List[Dict]:
    """Extract recordable tracks from a MIDI file."""
    mid = mido.MidiFile(midi_path)
    tracks = []

    for i, track in enumerate(mid.tracks):
        name = track.name.strip()
        if not name or name.startswith('__'):
            continue
        # Check if track has note events
        has_notes = any(msg.type in ('note_on', 'note_off') and msg.velocity > 0
                       for msg in track if not msg.is_meta)
        if not has_notes:
            continue

        info = classify_track(name)
        info['track_index'] = i
        info['track_name'] = name
        info['track'] = track
        info['note_count'] = sum(1 for msg in track
                                if msg.type == 'note_on' and msg.velocity > 0)
        tracks.append(info)

    return tracks


def get_batches(tracks: List[Dict], max_per_batch: int = 15) -> List[List[Dict]]:
    """Split tracks into batches of max_per_batch."""
    batches = []
    for i in range(0, len(tracks), max_per_batch):
        batches.append(tracks[i:i + max_per_batch])
    return batches


# ============================================================================
# RECORDING
# ============================================================================

def record_and_split(midi_path: str, output_dir: str, bpm: float,
                     fc: FantomController, port_name: str) -> Dict[str, str]:
    """
    Record individual stems from the Fantom.
    Returns dict of {track_name: wav_path}.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get recordable tracks
    tracks = get_recordable_tracks(midi_path)
    print(f"\nFound {len(tracks)} recordable tracks:")
    for t in tracks:
        print(f"  {t['track_name']}: {t['role']} ({t['note_count']} notes)")

    # Batch into passes (15 tracks per pass)
    batches = get_batches(tracks, max_per_batch=15)
    print(f"\n{len(batches)} recording pass(es) needed")

    mid = mido.MidiFile(midi_path)
    song_duration = mid.length
    tpb = mid.ticks_per_beat

    # Extract tempo
    song_tempo_us = mido.bpm2tempo(bpm)
    count_in_seconds = COUNT_IN_BEATS * (60.0 / bpm)

    all_stems = {}
    calibration_gains = {}

    for pass_idx, batch in enumerate(batches):
        print(f"\n{'='*60}")
        print(f"PASS {pass_idx + 1}/{len(batches)}: {len(batch)} tracks")
        print(f"{'='*60}")

        # Build batch MIDI
        batch_mid = mido.MidiFile()
        batch_mid.ticks_per_beat = tpb

        # Copy tempo track
        tempo_track = mido.MidiTrack()
        for msg in mid.tracks[0]:
            if msg.is_meta:
                tempo_track.append(msg)
        batch_mid.tracks.append(tempo_track)

        track_names = []
        for part_idx, track_info in enumerate(batch):
            if part_idx >= 15:
                print(f"  Skipping {track_info['track_name']} (exceeds 15-part limit)")
                break

            name = track_info['track_name']
            track_names.append(name)
            print(f"\n  Part {part_idx + 1} → USB {part_idx*2+1}/{part_idx*2+2}: {name}")

            # Select patch
            patch = select_patch_for_track(track_info)
            print(f"    Patch: {patch['name']} (MSB:{patch['msb']} LSB:{patch['lsb']} PC:{patch['pc']})")
            fc.select_patch(part_idx, patch['msb'], patch['lsb'], patch['pc'])

            # Apply sound design
            apply_sound_design(fc, part_idx, track_info)
            print(f"    Sound design applied ({track_info['role']})")

            # Calibrate level to -6 dBFS (drums skip calibration)
            is_drum = track_info['channel_type'] == 'drum'
            gain_db = calibrate_part_gain(fc, part_idx, track_info['track'],
                                          tpb, bpm, port_name, is_drum=is_drum)
            if gain_db is None:
                print(f"    ! Silent or rejected — skipping Part {part_idx+1}")
                continue
            track_info['calibrated_gain_db'] = gain_db
            calibration_gains[name] = gain_db

            # Build MIDI track for this Part
            new_track = mido.MidiTrack()
            new_track.name = name

            # Add count-in offset
            first = True
            for msg in track_info['track']:
                if msg.is_meta:
                    new_track.append(msg.copy())
                else:
                    if msg.type in ('note_on', 'note_off'):
                        new_track.append(msg.copy(
                            channel=part_idx,
                            time=msg.time + (tpb * COUNT_IN_BEATS if first else 0)
                        ))
                    else:
                        new_track.append(msg.copy(
                            channel=part_idx,
                            time=msg.time + (tpb * COUNT_IN_BEATS if first else 0)
                        ))
                    first = False

            batch_mid.tracks.append(new_track)

        # Add sync click on Part 16 (channel 15)
        sync_track = mido.MidiTrack()
        sync_track.name = '__sync__'
        note_dur = tpb // 4
        gap = tpb - note_dur
        for beat in range(COUNT_IN_BEATS):
            vel = 127 if beat == 0 else 90
            delta = 0 if beat == 0 else gap
            sync_track.append(mido.Message('note_on', channel=SYNC_CH,
                                          note=SYNC_NOTE, velocity=vel, time=delta))
            sync_track.append(mido.Message('note_off', channel=SYNC_CH,
                                          note=SYNC_NOTE, velocity=0, time=note_dur))
        batch_mid.tracks.append(sync_track)

        # Select sync patch
        fc.select_patch(SYNC_CH, 86, 65, 45)  # Use drum kit for click

        # Save batch MIDI
        batch_midi_path = os.path.join(output_dir, f"pass{pass_idx + 1:02d}_recording.mid")
        batch_mid.save(batch_midi_path)
        print(f"\n  Batch MIDI: {batch_midi_path}")

        # Record
        total_duration = song_duration + count_in_seconds + 5.0  # pre-roll + song + tail
        output_filename = f"pass{pass_idx + 1:02d}_all.wav"
        output_path = os.path.join(output_dir, output_filename)

        print(f"  Recording {total_duration:.1f}s ({len(batch)} stems + sync)...")
        _play_midi_and_record(batch_midi_path, output_path, port_name, total_duration)

        # Split into individual stems
        print(f"  Splitting stems...")
        stems = _split_stems(output_path, track_names, output_dir, pass_idx,
                            count_in_seconds)
        all_stems.update(stems)

    return all_stems, calibration_gains


def _play_midi_and_record(midi_path: str, output_path: str,
                          port_name: str, duration: float):
    """Play MIDI through Fantom and record 32-channel audio simultaneously."""
    recorded_chunks = []
    recording_active = False

    def audio_callback(indata, frames, time_info, status):
        if recording_active:
            recorded_chunks.append(indata.copy())

    def record_thread():
        nonlocal recording_active
        recording_active = True
        with sd.InputStream(device=FANTOM_DEVICE_INDEX, channels=CHANNELS,
                           samplerate=SAMPLE_RATE, callback=audio_callback,
                           blocksize=2048):
            time.sleep(duration)
        recording_active = False

    # Start recording
    rec = threading.Thread(target=record_thread)
    rec.start()
    time.sleep(0.3)  # Let recording stabilize

    # Play MIDI with precise timing (replace mido.play() with manual timing)
    outport = mido.open_output(port_name)
    mid = mido.MidiFile(midi_path)
    print(f"    Playing MIDI (precise timing)...")

    # Extract all messages with absolute times
    # First, get the tempo from the MIDI file
    tempo_us = mido.bpm2tempo(128)  # Default
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo_us = msg.tempo
                break
    
    all_msgs = []
    for track in mid.tracks:
        abs_time = 0.0
        for msg in track:
            abs_time += mido.tick2second(msg.time, mid.ticks_per_beat, tempo_us)
            if not msg.is_meta:
                all_msgs.append((abs_time, msg))
    
    # Sort by absolute time
    all_msgs.sort(key=lambda x: x[0])
    
    # Play with busy-wait timing
    start_perf = time.perf_counter()
    for abs_time, msg in all_msgs:
        target = start_perf + abs_time
        while time.perf_counter() < target:
            pass  # Busy-wait for precise timing
        outport.send(msg)
    
    elapsed = time.perf_counter() - start_perf
    outport.close()
    print(f"    MIDI done ({elapsed:.1f}s, {len(all_msgs)} messages)")

    # Wait for recording to finish
    rec.join(timeout=duration + 10)

    # Save
    if recorded_chunks:
        audio = np.concatenate(recorded_chunks, axis=0)
        sf.write(output_path, audio, SAMPLE_RATE, subtype='PCM_24')
        print(f"    Saved: {output_path} ({audio.shape[0]/SAMPLE_RATE:.1f}s, {audio.shape[1]}ch)")
    else:
        print(f"    ERROR: No audio recorded!")


def _split_stems(multi_ch_path: str, track_names: List[str],
                 output_dir: str, pass_idx: int, count_in_seconds: float) -> Dict[str, str]:
    """Split 32-channel WAV into individual stereo stems with sync click detection."""
    data, sr = sf.read(multi_ch_path)
    stems = {}

    # === SYNC CLICK DETECTION ===
    # The sync click is on USB 31/32 (channels 30-31).
    # Detect the first transient to find the exact song start.
    sync_ch_left = 30   # USB 31 = channel index 30
    sync_ch_right = 31  # USB 32 = channel index 31

    trim_start = 0  # Default: no trim

    if data.shape[1] > sync_ch_left:
        sync_signal = data[:, sync_ch_left]
        sync_peak = np.max(np.abs(sync_signal))

        if sync_peak > 0.01:  # Sync click detected
            # Find first transient: envelope exceeds 50% of peak
            envelope = np.abs(sync_signal)
            threshold = sync_peak * 0.5
            above_threshold = np.where(envelope > threshold)[0]

            if len(above_threshold) > 0:
                # First click position = song start
                first_click = above_threshold[0]
                # Skip past the click (count-in beats before song)
                # The click marks beat 1 of count-in. Song starts after COUNT_IN_BEATS.
                # But we want to trim to the START of the song, not the click.
                # The click is 4 beats before song start.
                beats_per_second = 1.0 / (count_in_seconds / 4)  # 4 beats = count-in
                click_to_song_samples = int(4 * sr / beats_per_second)  # 4 beats

                # Trim to song start (after count-in)
                trim_start = max(0, first_click + click_to_song_samples)
                print(f"    Sync click detected at sample {first_click} ({first_click/sr:.3f}s)")
                print(f"    Song start at sample {trim_start} ({trim_start/sr:.3f}s)")
            else:
                # Fallback: coarse trim
                trim_start = max(0, int(count_in_seconds * sr) - int(0.1 * sr))
                print(f"    No sync click found, using coarse trim ({trim_start/sr:.3f}s)")
        else:
            # No sync signal, coarse trim
            trim_start = max(0, int(count_in_seconds * sr) - int(0.1 * sr))
            print(f"    No sync signal, using coarse trim ({trim_start/sr:.3f}s)")
    else:
        trim_start = max(0, int(count_in_seconds * sr) - int(0.1 * sr))

    for i, name in enumerate(track_names):
        usb_left = i * 2
        usb_right = i * 2 + 1

        if usb_right >= data.shape[1]:
            print(f"    Skipping {name} (USB {usb_left+1}/{usb_right+1} not available)")
            continue

        # Extract stereo pair
        stem_data = data[trim_start:, usb_left:usb_right+1]

        # Check if there's signal
        peak = np.max(np.abs(stem_data))
        if peak < 0.001:
            print(f"    {name}: silent (USB {usb_left+1}/{usb_right+1})")
            continue

        # Save stem
        safe_name = name.replace('/', '_').replace(' ', '_').replace('\\', '_')
        stem_path = os.path.join(output_dir, f"{safe_name}.wav")
        sf.write(stem_path, stem_data, sr, subtype='PCM_24')

        peak_db = 20 * np.log10(peak + 1e-10)
        print(f"    {name}: peak={peak_db:.1f} dBFS → {stem_path}")
        stems[name] = stem_path

    return stems


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Tech House Stem Recorder')
    parser.add_argument('midi_file', help='MIDI file to record')
    parser.add_argument('--bpm', type=float, default=126, help='BPM')
    parser.add_argument('--output-dir', default=None, help='Output directory')
    parser.add_argument('--port', default='FANTOM-6 7 8', help='MIDI port name')
    args = parser.parse_args()

    # Default output dir: same name as MIDI file without extension
    if args.output_dir is None:
        stem = Path(args.midi_file).stem
        args.output_dir = os.path.join(os.path.dirname(args.midi_file) or '.', stem, 'recordings')

    # Connect to Fantom
    print("Connecting to Roland Fantom...")
    fc = FantomController(port_name=args.port)
    if not fc.output:
        print("ERROR: Could not connect to Fantom")
        sys.exit(1)
    print(f"Connected: {fc.port_name}")

    # Record stems
    stems, calibration_gains = record_and_split(args.midi_file, args.output_dir, args.bpm, fc, args.port)

    # Summary
    print(f"\n{'='*60}")
    print(f"STEM RECORDING COMPLETE")
    print(f"{'='*60}")
    print(f"Stems recorded: {len(stems)}")
    print(f"Output: {args.output_dir}")
    for name, path in sorted(stems.items()):
        gain = calibration_gains.get(name, 'N/A')
        gain_str = f" (gain={gain:+.1f}dB)" if isinstance(gain, float) else ""
        print(f"  {name}: {path}{gain_str}")

    # Save manifest
    manifest = {
        'midi_file': args.midi_file,
        'bpm': args.bpm,
        'stems': stems,
        'total_stems': len(stems),
        'calibration': {name: {'gain_db': gain, 'target_peak_dbfs': -6.0}
                       for name, gain in calibration_gains.items()
                       if isinstance(gain, float)},
    }
    manifest_path = os.path.join(args.output_dir, 'stem_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}")


if __name__ == '__main__':
    main()
