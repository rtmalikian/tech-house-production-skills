import mido
import random
import csv
import os
from typing import List, Dict, Tuple
from midi_config import SWING_VALUES, HUMANIZATION
from midi_theory import cents_to_pitch_bend
from midi_models import MicrotonalNote

# ============================================================================
# MIDI UTILITIES
# ============================================================================

def apply_swing(timing: int, position_in_bar: int, swing: float = 0.58) -> int:
    if position_in_bar % 2 == 1:
        delay = int(timing * (swing - 0.5))
        return timing + delay
    else:
        delay = int(timing * (swing - 0.5))
        return timing - delay

def humanize_timing(base_ticks: int, style: str = 'lofi') -> int:
    range_min, range_max = HUMANIZATION.get(f'timing_{style}', (-10, 10))
    return random.randint(range_min, range_max)

def humanize_velocity(base_velocity: int, style: str = 'lofi') -> int:
    range_min, range_max = HUMANIZATION.get(f'velocity_{style}', (-8, 8))
    return random.randint(range_min, range_max)

def add_note(track: mido.MidiTrack, note: int, vel: int, dur: int, dt: int = 0, 
             ch: int = 0, swing: float = 0.58, humanize: bool = True, style: str = 'lofi'):
    position = (dt // 120) % 8
    swung_dt = apply_swing(dt, position, swing) if swing > 0.5 else dt
    if humanize:
        t_off, v_off = humanize_timing(swung_dt, style), humanize_velocity(vel, style)
    else: t_off, v_off = 0, 0
    final_vel, final_dt = max(1, min(127, vel + v_off)), max(0, swung_dt + t_off)
    track.append(mido.Message('note_on', note=note, velocity=final_vel, time=final_dt, channel=ch))
    track.append(mido.Message('note_off', note=note, velocity=0, time=max(60, dur - t_off), channel=ch))

def add_microtonal_note(track: mido.MidiTrack, micro_note: MicrotonalNote, dt: int = 0, ch: int = 0):
    if micro_note.cents_offset != 0:
        pitch = int(max(-8192, min(8191, (micro_note.cents_offset / 100.0) * 4096)))
        track.append(mido.Message('pitchwheel', pitch=pitch, time=dt, channel=ch))
        track.append(mido.Message('note_on', note=micro_note.note, velocity=micro_note.velocity, time=0, channel=ch))
    else:
        track.append(mido.Message('note_on', note=micro_note.note, velocity=micro_note.velocity, time=dt, channel=ch))
    track.append(mido.Message('note_off', note=micro_note.note, velocity=0, time=micro_note.duration, channel=ch))
    if micro_note.cents_offset != 0:
        track.append(mido.Message('pitchwheel', pitch=0, time=0, channel=ch))

def add_pad_chord(pad_tr: mido.MidiTrack, chord: List[int], dt: int, duration: int,
                  vel: int = 70, bar: int = 0):
    if (bar % 4 == 3 or bar > 24) and random.random() < 0.4:
        ext = random.choice([chord[0] + 14, chord[0] + 17, chord[0] + 21])
        if ext not in chord: chord.append(ext)
    chord = sorted(chord)
    for i, note in enumerate(chord):
        pad_tr.append(mido.Message('note_on', note=note, velocity=vel, time=dt if i==0 else 0))
    for i, note in enumerate(chord):
        pad_tr.append(mido.Message('note_off', note=note, velocity=0, time=duration if i==0 else 0))

def load_fantom_sounds(filename: str) -> List[Dict]:
    sounds = []
    try:
        with open(filename, 'r', newline='') as f:
            for row in csv.DictReader(f): sounds.append(row)
    except FileNotFoundError: pass
    return sounds

def load_drum_kits(filename: str) -> List[Dict]:
    kits = []
    try:
        with open(filename, 'r', newline='') as f:
            for row in csv.DictReader(f):
                try:
                    row['MSB'], row['LSB'], row['PC'] = int(row['MSB']), int(row['LSB']), int(row['PC'])
                    kits.append(row)
                except: pass
    except FileNotFoundError: pass
    return kits

def init_spatial(track, pan=64, expr=100):
    track.append(mido.Message('control_change', control=10, value=pan, time=0))
    track.append(mido.Message('control_change', control=11, value=expr, time=0))

def write_performance_to_track(track, events, drums_main_tr, drums_chorus_tr, force_drum=False, articulation='standard'):
    track_time = 0
    for e in events:
        dt = max(0, e['time'] - track_time)
        chan = 9 if (track in [drums_main_tr, drums_chorus_tr] or force_drum) else 0
        if e.get('type') == 'pitchwheel':
            track.append(mido.Message('pitchwheel', pitch=int(e.get('pitch', 0)), time=int(dt), channel=chan))
            track_time += dt
            continue
        if not force_drum and e['vel'] > 0 and dt > 480: 
            track.append(mido.Message('pitchwheel', pitch=-500, time=int(dt)))
            track.append(mido.Message('pitchwheel', pitch=0, time=0))
            track_time += dt
            dt = 0
        track.append(mido.Message('note_on' if e['vel'] > 0 else 'note_off', note=e['note'], velocity=e['vel'], time=int(dt), channel=chan))
        track_time += dt
        if not force_drum and e['vel'] > 0 and articulation == 'legato':
            for v in range(5): track.append(mido.Message('control_change', control=1, value=v*15, time=0))
    return track_time
