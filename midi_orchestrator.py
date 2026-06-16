#!/usr/bin/env python3
"""
Tech House MIDI Generator v1.0
Generates tech house MIDI from scratch — 909 drums, syncopated bass,
acid lines, chord stabs, and proper arrangement structure.

Based on the final_pipeline_june2026 architecture, adapted for tech house.
"""
import mido
import random
import os
import sys
import json
from datetime import datetime
from typing import List, Dict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from midi_config import (
    TOTAL_BARS, PROD_DIR, SWING_VALUES, NOTE_NAMES, REGISTER_RANGES,
    TECH_HOUSE_BPM_MIN, TECH_HOUSE_BPM_MAX,
    PREFERRED_KEYS, PREFERRED_SCALES,
    get_bar_length_ticks, get_song_length_ticks, get_random_bpm
)
from midi_models import VoiceLeadingContext, TensionState, MelodyNote
from midi_theory import (
    MODE_INTERVALS, get_chord_quality, get_chord_notes,
    clamp_to_register, parse_chord_symbol, build_bar_harmony,
    nearest_chord_or_scale_tone
)
from midi_composition import (
    generate_4bar_loop, generate_euclidean, generate_bass,
    generate_harmonic_bass, generate_counter_melody_2bar
)
from midi_composition_blueprint import (
    blueprint_to_metadata, create_composition_blueprint, select_progression_name
)
from midi_musical_devices import (
    render_counter_melody_device_events
)
from midi_pad_composition import (
    derive_later_section_progressions, render_pad_bar_events, select_pad_profile
)
from midi_song_structure import (
    VERSE_PROGRESSIONS, CHORUS_PROGRESSIONS, INTRO_OUTRO_PROGRESSIONS,
    FILL_PROGRESSIONS, get_bar_type, get_passing_chord,
    transform_loop
)
from midi_engine import (
    apply_swing, init_spatial, write_performance_to_track
)
from midi_analysis import analyze_melody_intervals, analyze_voice_leading

# Drum imports
from midi_drum_sequences import (
    get_pattern_funcs as get_drum_pattern_funcs,
    GM_DRUM_MAP, PATTERN_FAMILIES, PATTERN_FAMILY_MAP
)


def main():
    print("=" * 70)
    print("Tech House MIDI Generator v1.0")
    print("=" * 70)

    # === BPM: Tech house range 124-128 ===
    song_bpm = get_random_bpm()
    print(f"BPM: {song_bpm}")

    # Always 4/4 for tech house
    time_sig = '4-4'
    bar_length = get_bar_length_ticks(time_sig)
    song_length = get_song_length_ticks(time_sig)
    beats_per_bar = 4
    print(f"Time Signature: 4/4")

    # === KEY: Minor keys preferred for tech house ===
    scales_map = {}
    for ro in range(12):
        for mn, iv in MODE_INTERVALS.items():
            scales_map[f"{NOTE_NAMES[ro]} {mn}"] = [ro + i for i in iv]

    # 90% minor, 10% other (dorian, phrygian for darker vibes)
    if random.random() < 0.90:
        minor_scales = [k for k in scales_map.keys() if 'Minor' in k]
        scale_name = random.choice(minor_scales)
    else:
        dark_scales = [k for k in scales_map.keys()
                       if any(m in k for m in ('Minor', 'Dorian', 'Phrygian'))]
        scale_name = random.choice(dark_scales)

    scale_notes = scales_map[scale_name]
    base = scale_notes[0]
    key = NOTE_NAMES[base % 12]
    blueprint = create_composition_blueprint(scale_name, time_sig)

    # === CHORD PROGRESSIONS (minimal for tech house) ===
    verse_prog_name = select_progression_name(VERSE_PROGRESSIONS.keys(), 'statement', blueprint)
    chorus_prog_name = select_progression_name(CHORUS_PROGRESSIONS.keys(), 'lift', blueprint)
    intro_prog_name = select_progression_name(INTRO_OUTRO_PROGRESSIONS.keys(), 'setup', blueprint)
    fill_prog_name = select_progression_name(FILL_PROGRESSIONS.keys(), 'tension', blueprint)

    verse_prog = VERSE_PROGRESSIONS[verse_prog_name]
    chorus_prog = CHORUS_PROGRESSIONS[chorus_prog_name]
    intro_prog = INTRO_OUTRO_PROGRESSIONS[intro_prog_name]
    fill_prog = FILL_PROGRESSIONS[fill_prog_name]
    section_progressions, progression_variants = derive_later_section_progressions(
        intro_prog, verse_prog, chorus_prog, fill_prog,
        scale_notes, base, enable_reharmonization=False,
    )

    print(f"\nKey: {key} {scale_name}")
    print(f"Blueprint: {blueprint.mood} | harmonic={blueprint.harmonic_complexity:.2f} | density={blueprint.melodic_density:.2f}")
    print(f"\nProgressions:")
    print(f"  Intro: {intro_prog_name}")
    print(f"  Verse: {verse_prog_name}")
    print(f"  Chorus: {chorus_prog_name}")
    print(f"  Fill: {fill_prog_name}")

    # === MIDI TRACKS ===
    mid = mido.MidiFile(type=1)
    tempo_tr = mido.MidiTrack()
    bass_tr = mido.MidiTrack()
    sub_bass_tr = mido.MidiTrack()
    acid_tr = mido.MidiTrack()
    stab_tr = mido.MidiTrack()
    pad_tr = mido.MidiTrack()
    fx_tr = mido.MidiTrack()
    drums_main_tr = mido.MidiTrack()
    drums_chorus_tr = mido.MidiTrack()
    aux_tr = mido.MidiTrack()
    aux_tr.name = 'Aux Percussion'

    mid.tracks.extend([tempo_tr, bass_tr, sub_bass_tr, acid_tr, stab_tr,
                       pad_tr, fx_tr, drums_main_tr, drums_chorus_tr, aux_tr])

    tempo_tr.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(song_bpm), time=0))
    tempo_tr.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    tempo_tr.append(mido.MetaMessage('end_of_track', time=song_length))

    # === HARMONY PLAN ===
    def progression_for_section(section_type):
        return section_progressions.get(section_type, verse_prog)

    def chord_degree(chord_token):
        return parse_chord_symbol(chord_token, scale_notes, base).root_offset

    def section_intent(section_type):
        return blueprint.section_intents.get(section_type, blueprint.section_intents.get('verse1', None))

    harmony_plan = []
    for bar in range(TOTAL_BARS):
        bt = get_bar_type(bar)
        prog = progression_for_section(bt)
        if bt in ['intro', 'outro'] and len(prog) <= 2:
            chord_token = prog[bar % len(prog)]
        else:
            chord_token = prog[bar % len(prog)]
        harmony_plan.append(build_bar_harmony(
            bar, bt, chord_token, base, scale_notes, scale_name
        ))

    # === TRACK NAMES ===
    bass_tr.name = 'Bass'
    bass_tr.append(mido.MetaMessage('track_name', name='Bass', time=0))
    sub_bass_tr.name = 'Sub Bass'
    sub_bass_tr.append(mido.MetaMessage('track_name', name='Sub Bass', time=0))
    acid_tr.name = 'Acid Line'
    acid_tr.append(mido.MetaMessage('track_name', name='Acid Line', time=0))
    stab_tr.name = 'Chord Stab'
    stab_tr.append(mido.MetaMessage('track_name', name='Chord Stab', time=0))
    pad_tr.name = 'Pad'
    pad_tr.append(mido.MetaMessage('track_name', name='Pad', time=0))
    fx_tr.name = 'FX'
    fx_tr.append(mido.MetaMessage('track_name', name='FX', time=0))

    # === GENERATE LOOPS ===
    verse_loop = generate_4bar_loop(
        scale_notes, base, 'western',
        False, bar_length, harmony_window=harmony_plan[16:20],
        section_intent=section_intent('drop1'),
        motif_seed=blueprint.motif_seed
    )
    chorus_loop = generate_4bar_loop(
        scale_notes, base, 'western',
        True, bar_length, harmony_window=harmony_plan[24:28],
        section_intent=section_intent('drop2'),
        motif_seed=blueprint.motif_seed
    )
    print(f"  Melody personas: verse={verse_loop.get('persona', 'legacy')}, chorus={chorus_loop.get('persona', 'legacy')}")

    # === EVENT LISTS ===
    bass_ev, sub_bass_ev, acid_ev, stab_ev, pad_ev, fx_ev = [], [], [], [], [], []
    aux_perc_ev = []
    current_swing = SWING_VALUES['none']  # Tech house: fully quantized, no swing
    gvc = VoiceLeadingContext()

    # === BAR-BY-BAR GENERATION ===
    for bar in range(TOTAL_BARS):
        bt = get_bar_type(bar)
        harmony = harmony_plan[bar]
        root = harmony.root
        intent = section_intent(bt)

        # Energy levels per section — DJ-friendly, commercial length
        energy_map = {
            'intro': 0.3 + (bar / 16) * 0.2,           # Drums only, building energy
            'drop1': 0.85 + (random.random() * 0.1),    # Full energy
            'breakdown': max(0.2, 0.8 - ((bar - 48) / 32) * 0.6),  # 32-bar dramatic breakdown
            'drop2': 0.9 + (random.random() * 0.1),     # Full energy
            'outro': max(0.2, 0.9 - ((bar - 112) / 16) * 0.7),  # Fading out
        }
        energy = energy_map.get(bt, 0.5)

        qual = 'minor' if harmony.spec.quality in ['minor', 'min7', 'min9'] else (
            'dom7' if harmony.spec.quality == 'dom7' else
            get_chord_quality(root, scale_notes)
        )
        bs = bar * bar_length

        # === DJ-FRIENDLY ELEMENT RULES ===
        # Intro/Outro: drums + percussion ONLY (no bass, acid, stabs, pads)
        # Breakdown: NO kick, NO bass — pad + atmosphere + FX + filter sweeps
        # Drop: STAGGERED element entry (not everything at once!)

        is_drums_only = bt in ['intro', 'outro']
        is_breakdown = bt == 'breakdown'
        is_drop = bt in ['drop1', 'drop2']
        
        # Calculate position within drop for staggered entry
        if is_drop:
            drop_start = 16 if bt == 'drop1' else 80
            bar_in_drop = bar - drop_start  # 0-31 within drop
        else:
            bar_in_drop = -1
        
        # Calculate position within breakdown for filter sweeps
        if is_breakdown:
            breakdown_start = 48
            bar_in_breakdown = bar - breakdown_start  # 0-31 within breakdown
        else:
            bar_in_breakdown = -1
        
        # === A-A-A-B PHRASE STRUCTURE ===
        # Tech house melodies loop A for 3 bars, then B (variation) on bar 4
        bar_in_phrase = bar % 4  # 0, 1, 2, 3
        is_switch_up = (bar_in_phrase == 3)  # B bar — aggressive modulation

        # === BASS GENERATION ===
        # Reference tracks: bass DROPS OUT during breakdown, only preview in last 4 bars
        # Dennis Ferrer: bass enters LATE (bar 42!)
        bass_should_play = False
        bass_is_preview = False

        if is_drop and bar_in_drop >= 0:
            bass_should_play = True  # Bass plays throughout drop
        elif is_breakdown and bar >= 76:
            # Bass preview: last 4 bars of breakdown only (reference pattern)
            bass_should_play = True
            bass_is_preview = True
        elif bt == 'intro':
            # Bass preview from bar 0 — full groove from the start
            bass_should_play = True
            bass_is_preview = True
        # Note: bass is SILENT during bars 48-75 of breakdown (reference pattern)

        if bass_should_play:
            # Tech house bass styles per section
            if bt in ['drop1', 'drop2']:
                bass_style = random.choice(['active', 'syncopated', 'standard'])
            else:
                bass_style = 'standard'  # Intro/breakdown preview: simple pattern

            next_harmony = harmony_plan[bar + 1] if bar + 1 < len(harmony_plan) else None
            bcell = generate_bass(root, qual, scale_notes, gvc, None,
                                   bass_style, bar, bar_length, time_sig,
                                   harmony=harmony, next_harmony=next_harmony,
                                   section_intent=intent)
            tick = 0
            for i, note in enumerate(bcell.notes):
                dur = bcell.rhythm[i]
                if tick + dur > bar_length:
                    dur = bar_length - tick
                if note is not None:
                    # Bass preview in breakdown is quieter
                    if bass_is_preview:
                        vel = int(30 * energy)  # Sub-only preview
                    else:
                        vel = int(50 * energy)  # Consistent bass velocity (tight, quantized)
                    if dur <= 120:
                        vel = int(vel * 0.6)
                    abs_t = bs + tick
                    if abs_t + dur <= song_length:
                        fn = clamp_to_register(note, 'bass')
                        bass_ev.extend([
                            {'time': abs_t, 'note': fn, 'vel': vel},
                            {'time': abs_t + dur, 'note': fn, 'vel': 0}
                        ])
                tick += dur

        # === SUB BASS (drops only, beat 1 only, quiet) ===
        if is_drop and bar_in_drop >= 0:
            sub_note = clamp_to_register(root, 'sub_bass')
            if sub_note < 28:
                sub_note += 12
            # Sub bass preview is quieter
            if bass_is_preview:
                sub_vel = int(30 * energy)  # Very quiet preview
            else:
                sub_vel = int(60 * energy)
            sub_t = bs
            sub_dur = 960  # Half note
            if sub_t + sub_dur <= song_length:
                sub_bass_ev.extend([
                    {'time': sub_t, 'note': sub_note, 'vel': sub_vel},
                    {'time': sub_t + sub_dur, 'note': sub_note, 'vel': 0}
                ])

        # === ACID LINE (A-A-A-B structure: bars 0-2 = A, bar 3 = B switch-up) ===
        if is_drop and bar_in_drop >= 8 and random.random() < 0.6:
            acid_root = clamp_to_register(root, 'acid_line')
            
            # A bars: standard acid notes pool
            acid_notes_A = [acid_root, acid_root + 7, acid_root + 12,
                           acid_root + 10, acid_root + 5, acid_root + 3]
            
            # B bar (switch-up): add chromatic tension, octave up, more notes
            acid_notes_B = [acid_root + 12, acid_root + 11, acid_root + 13,  # Octave + chromatic
                           acid_root + 10, acid_root + 5, acid_root + 3,
                           acid_root + 14, acid_root + 15]  # 9th, b3 up
            
            sixteenth = 120
            
            if is_switch_up:
                # B BAR: Aggressive switch-up — more notes, higher, chromatic
                acid_notes_pool = acid_notes_B
                note_chance = 0.75  # More notes (75% vs 55%)
                acid_vel = int(120 * energy)  # Louder on switch-up
            else:
                # A BARS: Standard acid pattern
                acid_notes_pool = acid_notes_A
                note_chance = 0.55
                acid_vel = int(100 * energy)
            
            # 16th note acid pattern
            for step in range(16):
                if random.random() < note_chance:
                    an = random.choice(acid_notes_pool)
                    av = acid_vel
                    at = bs + step * sixteenth
                    ad = sixteenth // 2
                    acid_ev.extend([
                        {'time': at, 'note': an, 'vel': av},
                        {'time': at + ad, 'note': an, 'vel': 0}
                    ])

        # === ARPEGGIO PATTERNS (A-A-A-B: bars 0-2 = A, bar 3 = B switch-up) ===
        # Not just 1-3-5 — use 7ths, 9ths, chromatic passing tones, varied rhythms
        if is_drop and bar_in_drop >= 4:
            chord = harmony.chord_tones
            root = chord[0] if chord else harmony.root
            
            # Build moody note pool — minor intervals, chromatic tension
            arp_pool_A = []
            for n in chord[:4]:  # Root, 3rd, 5th, 7th
                arp_pool_A.append(clamp_to_register(n + 12, 'pad'))
            if len(chord) >= 1:
                arp_pool_A.append(clamp_to_register(root + 14, 'pad'))
            arp_pool_A.append(clamp_to_register(root + 11, 'pad'))  # maj7
            arp_pool_A.append(clamp_to_register(root + 13, 'pad'))  # b9
            
            # B pool: higher octave, more chromatic, aggressive
            arp_pool_B = []
            for n in chord[:4]:
                arp_pool_B.append(clamp_to_register(n + 24, 'pad'))  # 2 octaves up
            arp_pool_B.append(clamp_to_register(root + 23, 'pad'))  # maj7 high
            arp_pool_B.append(clamp_to_register(root + 25, 'pad'))  # b9 high
            arp_pool_B.append(clamp_to_register(root + 26, 'pad'))  # 9th high
            arp_pool_B.append(clamp_to_register(root + 22, 'pad'))  # b7 high
            
            # A pattern: standard rhythms
            arp_patterns_A = {
                'dotted': [(0, 360), (360, 120), (480, 240), (720, 120),
                          (960, 360), (1320, 120), (1440, 240), (1680, 120)],
                'long_short': [(0, 480), (480, 120), (600, 480), (1080, 120),
                              (1200, 480), (1680, 120)],
                'sparse': [(0, 240), (480, 240), (960, 240), (1440, 240)],
                'triplet': [(0, 160), (160, 160), (320, 160), (480, 160),
                           (640, 160), (800, 160), (960, 160), (1120, 160),
                           (1280, 160), (1440, 160), (1600, 160), (1760, 160)],
                'offbeat': [(120, 240), (360, 240), (600, 240), (840, 240),
                           (1080, 240), (1320, 240), (1560, 240), (1800, 240)],
            }
            
            # B pattern: aggressive, dense, fast
            arp_patterns_B = {
                'rapid': [(0, 120), (120, 120), (240, 120), (360, 120),
                         (480, 120), (600, 120), (720, 120), (840, 120),
                         (960, 120), (1080, 120), (1200, 120), (1320, 120),
                         (1440, 120), (1560, 120), (1680, 120), (1800, 120)],
                'burst': [(0, 60), (60, 60), (120, 60), (180, 60),
                         (480, 60), (540, 60), (600, 60), (660, 60),
                         (960, 60), (1020, 60), (1080, 60), (1140, 60),
                         (1440, 60), (1500, 60), (1560, 60), (1620, 60)],
                'stutter': [(0, 120), (120, 120), (240, 120), (360, 120),
                           (480, 240), (720, 240), (960, 120), (1080, 120),
                           (1200, 120), (1320, 120), (1440, 240), (1680, 240)],
            }
            
            if is_switch_up:
                # B BAR: Aggressive switch-up
                arp_pool = arp_pool_B
                pattern = random.choice(list(arp_patterns_B.values()))
                note_seq = list(reversed(arp_pool))  # Descending for darkness
                vel_base = 90  # Louder
            else:
                # A BARS: Standard moody pattern
                arp_pool = arp_pool_A
                pattern_name = random.choice(list(arp_patterns_A.keys()))
                pattern = arp_patterns_A[pattern_name]
                note_sequences = [
                    arp_pool,
                    list(reversed(arp_pool)),
                    [arp_pool[0], arp_pool[2], arp_pool[3] if len(arp_pool) > 3 else arp_pool[0], arp_pool[4] if len(arp_pool) > 4 else arp_pool[1]],
                    [arp_pool[i] for i in range(len(arp_pool)-1, -1, -1)],
                    [random.choice(arp_pool) for _ in range(len(arp_pool))],
                ]
                note_seq = random.choice(note_sequences)
                vel_base = 60
            
            for i, (pos, dur) in enumerate(pattern):
                note = note_seq[i % len(note_seq)]
                vel = int((vel_base + (pos % 480 == 0) * 25 + (pos % 240 == 0) * 10) * energy)
                t = bs + pos
                if t + dur <= bs + bar_length:
                    pad_ev.extend([
                        {'time': t, 'note': note, 'vel': vel},
                        {'time': t + dur, 'note': note, 'vel': 0}
                    ])

        # === CHORD STABS (staggered: enter at bar 4+ of drop) ===
        if is_drop and bar_in_drop >= 4:
            chord_notes = harmony.chord_tones[:3]  # Triads only
            stab_vel = int(100 * energy)  # Consistent stab velocity (tight, quantized)
            # Stab on beat 1, short duration
            for ci, cn in enumerate(chord_notes):
                sn = clamp_to_register(cn + 12, 'pad')
                stab_ev.extend([
                    {'time': bs, 'note': sn, 'vel': stab_vel},
                    {'time': bs + 240, 'note': sn, 'vel': 0}  # 8th note duration
                ])
            # Stab on beat 3 (60% chance — more frequent)
            if random.random() < 0.6:
                for ci, cn in enumerate(chord_notes):
                    sn = clamp_to_register(cn + 12, 'pad')
                    stab_ev.extend([
                        {'time': bs + 2 * 480, 'note': sn, 'vel': int(stab_vel * 0.8)},
                        {'time': bs + 2 * 480 + 240, 'note': sn, 'vel': 0}
                    ])

        # === PAD (breakdown — atmospheric, LOUDER to fill space) ===
        # Reference tracks: pads/synths INCREASE during breakdown
        # When drums/bass drop out, pads become the dominant element
        if is_breakdown:
            # Override energy: pads are LOUDER during breakdown (reference pattern)
            pad_energy = min(1.0, energy * 1.5)  # 50% louder than section energy
            profile = select_pad_profile(bt, intent, False,
                                          previous_profile=None, repeat_count=0)
            profile_events, _ = render_pad_bar_events(
                harmony=harmony,
                next_harmony=harmony_plan[bar + 1] if bar + 1 < len(harmony_plan) else None,
                previous_voicing=None,
                profile=profile,
                bar_start=bs,
                bar_length=bar_length,
                energy=pad_energy,  # Louder during breakdown
                intent=intent,
                scale_notes=scale_notes,
                qual=qual,
                is_armenian=False,
                armenian_scale_name=None,
                cadence=False,
            )
            pad_ev.extend(profile_events)

        # === AUX PERCUSSION (euclidean patterns) ===
        if bar >= 8 and bt not in ['intro', 'breakdown']:
            e_steps = 16
            e_pattern = generate_euclidean(random.choice([2, 3, 4, 5]), e_steps)
            perc_note = random.choice([GM_DRUM_MAP['TAMBOURINE'],
                                        GM_DRUM_MAP['SHAKER'],
                                        GM_DRUM_MAP['RIMSHOT']])
            for i, pulse in enumerate(e_pattern):
                if pulse:
                    p_abs = bs + i * 120
                    p_vel = int(random.randint(30, 55) * energy)
                    aux_perc_ev.extend([
                        {'time': p_abs, 'note': perc_note, 'vel': p_vel},
                        {'time': p_abs + 60, 'note': perc_note, 'vel': 0}
                    ])

        # === FX (atmospheric layers, risers, filter sweeps) — DJ-friendly ===
        # White noise riser: ascending notes with increasing velocity
        # Filter sweep: simulates LPF opening (low → high frequencies)
        # Reference tracks show sub drops from 66% to 26% during breakdowns

        if is_breakdown:
            # Filter sweep: ascending notes that simulate LPF opening
            # First half of breakdown: low notes (filter closed)
            # Second half: high notes (filter opening)
            if bar_in_breakdown >= 0:
                # Filter sweep position (0.0 = closed, 1.0 = open)
                filter_position = bar_in_breakdown / 32.0  # 0-1 over 32 bars
                
                # Generate filter sweep notes
                # Low frequencies (filter closed) → High frequencies (filter open)
                sweep_base = 48 + int(filter_position * 48)  # C3 to C7
                sweep_vel = int((40 + filter_position * 60) * energy)  # Getting louder
                
                # 8th note filter sweep pattern
                for step in range(8):
                    fx_note = sweep_base + step * 2
                    if fx_note > 108:
                        fx_note = 108
                    fx_t = bs + step * 240  # 8th note spacing
                    fx_dur = 180
                    if fx_t + fx_dur <= bs + bar_length:
                        fx_ev.extend([
                            {'time': fx_t, 'note': fx_note, 'vel': sweep_vel},
                            {'time': fx_t + fx_dur, 'note': fx_note, 'vel': 0}
                        ])
            
            # Riser in last 8 bars of breakdown (bars 72-79)
            if bar >= 72:
                riser_bar = bar - 72  # 0-7 within riser section
                for step in range(8):
                    fx_note = 84 + step * 3 + riser_bar * 2  # Ascending
                    if fx_note > 108:
                        fx_note = 108
                    fx_vel = int((30 + step * 8 + riser_bar * 5) * energy)
                    fx_vel = min(127, fx_vel)
                    fx_t = bs + step * (bar_length // 8)
                    fx_ev.extend([
                        {'time': fx_t, 'note': fx_note, 'vel': fx_vel},
                        {'time': fx_t + 120, 'note': fx_note, 'vel': 0}
                    ])

        # Downlifter at drop transitions (first bar of drops)
        if bt == 'drop1' and bar == 16:
            # Impact: low note burst
            for step in range(4):
                fx_note = 36 + step * 2  # Descending
                fx_vel = int(90 * energy)
                fx_t = bs + step * 120
                fx_ev.extend([
                    {'time': fx_t, 'note': fx_note, 'vel': fx_vel},
                    {'time': fx_t + 240, 'note': fx_note, 'vel': 0}
                ])

        if bt == 'drop2' and bar == 80:
            # Impact: low note burst
            for step in range(4):
                fx_note = 36 + step * 2
                fx_vel = int(100 * energy)
                fx_t = bs + step * 120
                fx_ev.extend([
                    {'time': fx_t, 'note': fx_note, 'vel': fx_vel},
                    {'time': fx_t + 240, 'note': fx_note, 'vel': 0}
                ])

        # === ATMOSPHERIC "OTHER" ELEMENTS (reference pattern) ===
        # John Summit: "other" enters at bar 47, just before drop at bar 52
        # These elements prime the listener for the drop
        if is_breakdown and bar_in_breakdown >= 28:
            # Last 4 bars of breakdown: atmospheric elements enter
            # Creates anticipation for the drop
            atm_root = clamp_to_register(root + 12, 'pad')
            atm_notes = [atm_root, atm_root + 7, atm_root + 12]  # Root, 5th, octave
            
            # Sparse atmospheric pattern — every 2 beats
            for beat in [0, 2]:
                note = atm_notes[beat % len(atm_notes)]
                vel = int(50 * energy)  # Subtle, not dominant
                t = bs + beat * 480
                dur = 480  # Half note
                if t + dur <= bs + bar_length:
                    pad_ev.extend([
                        {'time': t, 'note': note, 'vel': vel},
                        {'time': t + dur, 'note': note, 'vel': 0}
                    ])

    # === DRUM GENERATION ===
    main_ev, chorus_ev = [], []
    bbA, bbB, bbC, bbD, main_family, chorus_family = get_drum_pattern_funcs()
    print(f"  Drum families: main={main_family}, chorus={chorus_family}")

    for bar in range(TOTAL_BARS):
        bt = get_bar_type(bar)
        bs = bar * bar_length

        # Drum intensity per section — DJ-friendly
        variation_map = {
            'intro': 0,       # Minimal: kick + hats only
            'drop1': 1,       # Full drums
            'breakdown': 0,   # Minimal: hats/perc only, NO kick
            'drop2': 2,       # Full drums, more variation
            'outro': 0,       # Minimal: kick + hats only, fading
        }
        variation_level = variation_map.get(bt, 0)

        # Select drum pattern based on section — DJ-friendly
        if bt in ['drop1', 'drop2']:
            dbar = bbD(480, random.randint(0, 3), variation_level=variation_level,
                        is_chorus=True, time_sig=time_sig)
        elif bt == 'breakdown':
            # === BREAKDOWN DRUM PATTERN ===
            # First half (bars 48-63): sparse, atmospheric — NO kick
            # Second half (bars 64-71): drum BUILD — kick enters, velocity increases
            # Last 8 bars (bars 72-79): RAPID BUILD — kick intensifies to 16ths
            
            bar_in_breakdown = bar - 48  # 0-31 within breakdown
            dbar = []
            
            if bar_in_breakdown < 16:
                # First half: sparse hats only, NO kick
                for i in range(0, 16, 2):
                    dbar.append({'note': GM_DRUM_MAP['CLOSED_HAT'], 'velocity': random.randint(40, 60),
                                 'time': i * 120})
                # Occasional open hat
                if random.random() < 0.3:
                    dbar.append({'note': GM_DRUM_MAP['OPEN_HAT'], 'velocity': random.randint(35, 50),
                                 'time': 7 * 120})
            
            elif bar_in_breakdown < 24:
                # BUILD PHASE 1 (bars 64-71): Kick on quarter notes, velocity building
                build_progress = (bar_in_breakdown - 16) / 8.0  # 0.0 to 1.0
                kick_vel = int(60 + build_progress * 67)  # 60 → 127
                
                # Kick on every beat (quarter notes)
                for beat in range(4):
                    dbar.append({'note': GM_DRUM_MAP['KICK'], 'velocity': kick_vel,
                                 'time': beat * 480})
                
                # Hats on 8th notes, velocity building
                hat_vel = int(50 + build_progress * 50)  # 50 → 100
                for i in range(8):
                    dbar.append({'note': GM_DRUM_MAP['CLOSED_HAT'], 'velocity': hat_vel,
                                 'time': i * 240})
                
                # Clap on beats 2&4 starting at bar 68
                if bar_in_breakdown >= 20:
                    clap_vel = int(60 + build_progress * 60)
                    dbar.append({'note': GM_DRUM_MAP['CLAP'], 'velocity': clap_vel,
                                 'time': 1 * 480})
                    dbar.append({'note': GM_DRUM_MAP['CLAP'], 'velocity': clap_vel,
                                 'time': 3 * 480})
            
            else:
                # BUILD PHASE 2 (bars 72-79): RAPID BUILD — kick on 16ths, velocity max
                build_progress = (bar_in_breakdown - 24) / 8.0  # 0.0 to 1.0
                
                # Kick on 16th notes — classic build technique
                kick_vel_start = int(80 + build_progress * 47)  # 80 → 127
                for i in range(16):
                    # Velocity increases within the bar too
                    vel = int(kick_vel_start + (i / 16.0) * (127 - kick_vel_start))
                    vel = min(127, vel)
                    dbar.append({'note': GM_DRUM_MAP['KICK'], 'velocity': vel,
                                 'time': i * 120})
                
                # Snare roll on 16ths (last 4 bars only)
                if bar_in_breakdown >= 28:
                    snare_vel = int(70 + build_progress * 57)
                    for i in range(16):
                        dbar.append({'note': GM_DRUM_MAP['SNARE'], 'velocity': snare_vel,
                                     'time': i * 120})
                
                # Hats on 16ths
                hat_vel = int(80 + build_progress * 47)
                for i in range(16):
                    dbar.append({'note': GM_DRUM_MAP['CLOSED_HAT'], 'velocity': hat_vel,
                                 'time': i * 120})
        elif bt == 'intro':
            # Intro: kick + hats + clap + bass preview from bar 0
            # Reference tracks have full groove from the start
            dbar = []
            for beat in range(4):
                dbar.append({'note': GM_DRUM_MAP['KICK'], 'velocity': random.randint(122, 127),
                             'time': beat * 480})
            # Clap on beats 2&4 from the start
            dbar.append({'note': GM_DRUM_MAP['CLAP'], 'velocity': 110,
                         'time': 1 * 480})
            dbar.append({'note': GM_DRUM_MAP['CLAP'], 'velocity': 110,
                         'time': 3 * 480})
            # Hats on 16ths
            for i in range(16):
                vel = 120 if i % 4 == 0 else 110 if i % 4 == 2 else 100
                dbar.append({'note': GM_DRUM_MAP['CLOSED_HAT'], 'velocity': vel,
                             'time': i * 120})
            else:
                # Last 8 bars: add clap on 2&4
                dbar = bbA(480, random.randint(0, 1), variation_level=0,
                            is_chorus=False, time_sig=time_sig)
        elif bt == 'outro':
            # Outro: kick + hats only, gradually strip
            if bar >= 72:
                # Last 8 bars: kick + hat only, fading
                dbar = []
                for beat in range(4):
                    dbar.append({'note': GM_DRUM_MAP['KICK'], 'velocity': random.randint(115, 122),
                                 'time': beat * 480})
                for i in range(0, 16, 2):
                    dbar.append({'note': GM_DRUM_MAP['CLOSED_HAT'], 'velocity': random.randint(60, 80),
                                 'time': i * 120})
            else:
                # First 8 bars: full pattern, fading
                dbar = bbA(480, random.randint(0, 1), variation_level=0,
                            is_chorus=False, time_sig=time_sig)
        else:
            dbar = bbA(480, random.randint(0, 3), variation_level=variation_level,
                        is_chorus=False, time_sig=time_sig)

        active_ev = chorus_ev if bt in ['drop1', 'drop2'] else main_ev
        last_hat = -100
        for n in sorted(dbar, key=lambda x: x['time']):
            msg = dict(n)
            # Avoid overlapping hats
            if msg['note'] == 42 and abs(msg['time'] - last_hat) < 120:
                continue
            if msg['note'] == 46:
                last_hat = msg['time']

            abs_t = bs + msg['time']
            active_ev.extend([
                {'time': abs_t, 'note': msg['note'], 'vel': msg['velocity']},
                {'time': abs_t + 60, 'note': msg['note'], 'vel': 0}
            ])

    # === SORT ALL EVENTS ===
    for ev_list in [main_ev, chorus_ev, bass_ev, sub_bass_ev, acid_ev,
                     stab_ev, pad_ev, fx_ev, aux_perc_ev]:
        ev_list.sort(key=lambda x: x['time'])

    # === ANALYSIS ===
    print(f"\n{'='*70}")
    print("TECH HOUSE MIDI GENERATION COMPLETE")
    print(f"{'='*70}")

    # === INIT SPATIAL ===
    init_spatial(bass_tr, 64)
    init_spatial(sub_bass_tr, 64)
    init_spatial(acid_tr, 80)
    init_spatial(stab_tr, 100)
    init_spatial(pad_tr, 50)
    init_spatial(fx_tr, 64)

    # === WRITE TRACKS ===
    lt_b = write_performance_to_track(bass_tr, bass_ev, drums_main_tr, drums_chorus_tr)
    lt_sb = write_performance_to_track(sub_bass_tr, sub_bass_ev, drums_main_tr, drums_chorus_tr)
    lt_ac = write_performance_to_track(acid_tr, acid_ev, drums_main_tr, drums_chorus_tr)
    lt_st = write_performance_to_track(stab_tr, stab_ev, drums_main_tr, drums_chorus_tr)
    lt_p = write_performance_to_track(pad_tr, pad_ev, drums_main_tr, drums_chorus_tr)
    lt_fx = write_performance_to_track(fx_tr, fx_ev, drums_main_tr, drums_chorus_tr)

    # === MIDI CC AUTOMATION: Filter Cutoff + Resonance Builds ===
    # During build phase (bars 64-79), send CC 74 (cutoff) and CC 71 (resonance)
    # that gradually increase to create tension before the drop
    def add_filter_build_automation(track, channel, bar_length, build_start_bar=64, build_end_bar=80):
        """Add CC 74 (cutoff) and CC 71 (resonance) automation during build."""
        cc_interval = bar_length // 4  # Send CC 4 times per bar (every beat)
        
        for bar in range(build_start_bar, build_end_bar):
            build_progress = (bar - build_start_bar) / (build_end_bar - build_start_bar)  # 0.0 → 1.0
            
            for beat in range(4):
                t = (bar * bar_length) + (beat * cc_interval)
                beat_progress = (beat / 4.0)  # 0.0, 0.25, 0.5, 0.75
                total_progress = min(1.0, build_progress + beat_progress * (1.0 / (build_end_bar - build_start_bar)))
                
                # CC 74: Filter Cutoff (0-127, ramp from 40 to 127)
                cutoff_val = int(40 + total_progress * 87)
                cutoff_val = min(127, cutoff_val)
                track.append(mido.Message('control_change', channel=channel, control=74, value=cutoff_val, time=t))
                
                # CC 71: Resonance (0-127, ramp from 30 to 100)
                resonance_val = int(30 + total_progress * 70)
                resonance_val = min(127, resonance_val)
                track.append(mido.Message('control_change', channel=channel, control=71, value=resonance_val, time=t))
        
        # Reset after build (at drop)
        drop_t = build_end_bar * bar_length
        track.append(mido.Message('control_change', channel=channel, control=74, value=64, time=drop_t))
        track.append(mido.Message('control_change', channel=channel, control=71, value=40, time=drop_t))

    # Filter build automation is handled by SysEx automation during playback
    # (not embedded in MIDI to avoid huge delta times)

    # === EXPLODED DRUMS ===
    DRUM_NAME_MAP = {
        35: "KickLow", 36: "Kick", 41: "KickAlt",
        37: "SideStick", 38: "Snare", 39: "Clap", 40: "SnareAlt", 43: "FloorTom",
        42: "ClosedHat", 44: "PedalHat", 45: "LowTom", 46: "OpenHat", 47: "MidTom",
        49: "Crash", 51: "Ride", 54: "Tambourine", 56: "Cowbell", 70: "Shaker/Maracas",
    }

    drum_tracks_data = {}
    main_ev.sort(key=lambda x: x['time'])
    chorus_ev.sort(key=lambda x: x['time'])
    aux_perc_ev.sort(key=lambda x: x['time'])

    def add_to_exploded(ev_list, prefix):
        for ev in ev_list:
            note = ev['note']
            key = (prefix, note)
            if key not in drum_tracks_data:
                drum_tracks_data[key] = []
            drum_tracks_data[key].append(ev)

    add_to_exploded(main_ev, "drum1")
    add_to_exploded(chorus_ev, "drum2")
    add_to_exploded(aux_perc_ev, "drum_aux")

    print(f"Exploding drums into individual instrument tracks...")

    tracks_to_keep = [tempo_tr, bass_tr, sub_bass_tr, acid_tr, stab_tr,
                       pad_tr, fx_tr]
    mid.tracks = tracks_to_keep

    for (prefix, note) in sorted(drum_tracks_data.keys()):
        events = sorted(drum_tracks_data[(prefix, note)], key=lambda x: x['time'])
        instr_name = DRUM_NAME_MAP.get(note, "Instr")
        track_name = f"{prefix}_{instr_name}_n{note}"

        dtr = mido.MidiTrack()
        dtr.name = track_name
        dtr.append(mido.MetaMessage('track_name', name=track_name, time=0))
        mid.tracks.append(dtr)

        pan_val = random.randint(44, 84)
        init_spatial(dtr, pan=pan_val)

        lt = write_performance_to_track(dtr, events, None, None, force_drum=True)
        dtr.append(mido.MetaMessage('end_of_track', time=max(0, song_length - lt)))

    # === END OF TRACK MARKERS ===
    bass_tr.append(mido.MetaMessage('end_of_track', time=max(0, song_length - lt_b)))
    sub_bass_tr.append(mido.MetaMessage('end_of_track', time=max(0, song_length - lt_sb)))
    acid_tr.append(mido.MetaMessage('end_of_track', time=max(0, song_length - lt_ac)))
    stab_tr.append(mido.MetaMessage('end_of_track', time=max(0, song_length - lt_st)))
    pad_tr.append(mido.MetaMessage('end_of_track', time=max(0, song_length - lt_p)))
    fx_tr.append(mido.MetaMessage('end_of_track', time=max(0, song_length - lt_fx)))

    # === SAVE (clean names + JSON sidecar) ===
    ts = datetime.now().strftime("%m%d_%H%M")
    # Scale type abbreviation
    if 'Minor' in scale_name: scale_type = 'min'
    elif 'Major' in scale_name: scale_type = 'maj'
    elif 'Dorian' in scale_name: scale_type = 'dor'
    elif 'Phrygian' in scale_name: scale_type = 'phr'
    else: scale_type = scale_name[:3].lower()
    # Clean key: A# → As, C# → Cs, etc.
    clean_key = key.replace('#', 's')
    stem = f"TH_{ts}_{song_bpm}_{clean_key}{scale_type}"

    out = os.path.join(PROD_DIR, f"{stem}.mid")
    mid.save(out)
    print(f"\n✓ Generated: {out}")

    # === JSON SIDECAR ===
    section_offsets = {
        'intro': 0, 'build1': 16, 'drop1': 24,
        'breakdown': 40, 'build2': 48, 'drop2': 56, 'outro': 72
    }
    metadata = {
        'bpm': song_bpm,
        'key': key,
        'scale': scale_name,
        'genre': 'tech_house',
        'time_signature': '4/4',
        'total_bars': TOTAL_BARS,
        'duration_sec': round(TOTAL_BARS * 4 * (60.0 / song_bpm), 1),
        'drums': {
            'main': main_family,
            'chorus': chorus_family,
        },
        'progressions': {
            'intro': intro_prog_name,
            'verse': verse_prog_name,
            'chorus': chorus_prog_name,
        },
        'arrangement': 'tech_house_standard',
        'sections': {
            s_name: {
                'bar': bar_idx,
                'time_sec': round(bar_idx * beats_per_bar * (60.0 / song_bpm), 1),
            }
            for s_name, bar_idx in section_offsets.items()
        },
        'tracks': [
            'Bass', 'Sub Bass', 'Acid Line', 'Chord Stab', 'Pad', 'FX',
            'drum1_Kick', 'drum1_Clap', 'drum1_ClosedHat', 'drum1_OpenHat',
            'drum2_Kick', 'drum2_Clap', 'drum2_ClosedHat', 'drum2_OpenHat',
        ],
    }

    json_path = os.path.join(PROD_DIR, f"{stem}.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata: {json_path}")

    return out, metadata


if __name__ == "__main__":
    main()
