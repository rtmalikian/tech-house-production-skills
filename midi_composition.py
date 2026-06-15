import random
from typing import List, Dict, Optional
from midi_models import VoiceLeadingContext, TensionState, MelodicCell, Motif, BarHarmony, SectionIntent
from midi_config import BASS_REGISTER_RANGE, get_bar_length_ticks, TIME_SIGNATURES, TICKS_PER_BEAT
from midi_theory import (
    clamp_to_register, ARMENIAN_MAQAM_SCALES, get_chord_notes, get_chord_quality,
    calculate_tension, score_interval, score_voice_leading, check_parallel_perfects,
    correct_parallel_perfects, get_motion_direction, get_phrase_cadence_intervals,
    get_emphasis_intervals, scale_degree_to_pitch, get_major_minor_mode,
    is_major_minor_scale
)
from midi_musical_devices import (
    apply_melodic_devices, bass_approach_note, choose_melody_persona,
    diversify_motif, phrase_role, should_use_pedal, theory_rhythm
)

# ============================================================================
# ABAC MELODIC CELL GENERATION
# ============================================================================

def generate_abac_cells(scale: List[int], root: int, scale_name: str = 'western',
                        cell_set: str = 'verse') -> Dict[str, MelodicCell]:
    if cell_set == 'chorus':
        templates = [
            {'intervals': [0, 3, 5, 3, 1, 0, -1], 'rhythm': [360, 240, 240, 360, 240, 240, 480], 'contour': 'arch'},
            {'intervals': [0, 2, 4, 5, 3, 2, 0], 'rhythm': [480, 240, 240, 240, 240, 240, 480], 'contour': 'ascending'},
            {'intervals': [0, 0, 2, 3, 2, 0, 0], 'rhythm': [240, 240, 360, 360, 360, 240, 480], 'contour': 'wave'},
            {'intervals': [0, 5, 3, 2, 1, 0, -2], 'rhythm': [480, 240, 240, 240, 240, 240, 480], 'contour': 'descending'},
        ]
    else:
        templates = [
            {'intervals': [0, 2, 1, 0, -1, 2, 0], 'rhythm': [480, 240, 240, 480, 240, 240, 480], 'contour': 'arch'},
            {'intervals': [0, 1, 2, 1, 0, -1, 0], 'rhythm': [360, 360, 240, 480, 240, 240, 480], 'contour': 'wave'},
            {'intervals': [0, 1, 0, 2, 1, 2, 3], 'rhythm': [480, 240, 240, 360, 360, 240, 480], 'contour': 'ascending'},
            {'intervals': [0, -1, 1, 0, 2, 1, 0], 'rhythm': [480, 480, 240, 240, 240, 240, 480], 'contour': 'descending'},
        ]
    
    base = random.choice(templates)
    return {
        'A': _make_cell(base, root, scale, scale_name),
        'B': _make_cell(_light_var(base), root, scale, scale_name),
        'C': _make_cell(_sig_var(base), root, scale, scale_name),
    }

def _light_var(t: Dict) -> Dict:
    new_int = t['intervals'].copy()
    for idx in random.sample(range(len(new_int)), random.choice([1, 2])):
        new_int[idx] += random.choice([-2, -1, 1, 2])
    new_rhy = t['rhythm'].copy()
    if random.random() < 0.5 and len(new_rhy) > 3:
        i = random.randint(1, len(new_rhy) - 2)
        new_rhy[i], new_rhy[i+1] = new_rhy[i+1], new_rhy[i]
    return {'intervals': new_int, 'rhythm': new_rhy, 'contour': t['contour']}

def _sig_var(t: Dict) -> Dict:
    if random.random() < 0.5:
        new_int = [-i for i in t['intervals']]
    else:
        new_int = t['intervals'][::-1]
        if len(new_int) > 1:
            new_int[-1] = new_int[0] - random.choice([0, 1, 2])
    return {
        'intervals': new_int,
        'rhythm': random.choice([[480, 480, 240, 240, 240, 240, 480], [240, 240, 480, 360, 240, 360, 480], [360, 240, 240, 480, 240, 240, 480]]),
        'contour': random.choice(['arch', 'valley', 'ascending', 'descending'])
    }

def _make_cell(t: Dict, root: int, scale: List[int], scale_name: str) -> MelodicCell:
    notes = [clamp_to_register(root + i, 'main_melody') for i in t['intervals']]
    if t['contour'] == 'arch':
        mid = len(notes) // 2
        for i in range(mid, len(notes)):
            if notes[i] > notes[mid]: notes[i] -= 12
    elif t['contour'] == 'valley':
        mid = len(notes) // 2
        for i in range(mid):
            if notes[i] < notes[mid]: notes[i] += 12
    elif t['contour'] == 'ascending':
        for i in range(1, len(notes) - 1):
            if notes[i] <= notes[i-1]: notes[i] += 12
    elif t['contour'] == 'descending':
        for i in range(1, len(notes)):
            if notes[i] >= notes[i-1]: notes[i] -= 12
    notes = [clamp_to_register(n, 'main_melody') for n in notes]
    ornaments = [{'type': 'armenian', 'scale': scale_name}] if scale_name in ARMENIAN_MAQAM_SCALES else []
    return MelodicCell(notes=notes, rhythm=t['rhythm'][:len(notes)], contour=t['contour'],
                       phrase_type='antecedent', ornaments=ornaments)

# ============================================================================
# UNIQUE 4-BAR LOOP GENERATION
# ============================================================================

def generate_4bar_loop(scale: List[int], root: int, scale_name: str = 'western',
                       is_chorus: bool = False, bar_length: int = 1920,
                       harmony_window: Optional[List[BarHarmony]] = None,
                       section_intent: Optional[SectionIntent] = None,
                       motif_seed: Optional[List[int]] = None) -> Dict:
    persona = choose_melody_persona(is_chorus, section_intent)
    motif = list(motif_seed) if motif_seed else _generate_seed_motif(scale, root, scale_name, is_chorus)
    motif = _section_motif_transform(motif, section_intent)
    motif = _apply_motif_transformations(motif, scale, root, scale_name)
    motif = diversify_motif(motif, [], is_chorus)
    full_notes, full_rhythm = [], []
    loop_bars = []
    vc, ts = VoiceLeadingContext(), TensionState()
    
    for bar_in_loop in range(4):
        harmony = harmony_window[bar_in_loop % len(harmony_window)] if harmony_window else None
        next_harmony = harmony_window[(bar_in_loop + 1) % len(harmony_window)] if harmony_window else None
        if bar_in_loop == 0:
            phrase_type, cadence_bar, contour = 'antecedent', False, random.choice(persona['contours'])
            motif_variant = motif
        elif bar_in_loop == 1:
            phrase_type, cadence_bar, contour = 'antecedent', False, random.choice(persona['contours'])
            motif_variant = _sequence_motif(motif, random.choice([2, 3]), scale)
        elif bar_in_loop == 2:
            phrase_type, cadence_bar, contour = 'consequent', False, random.choice(persona['contours'])
            motif_variant = _invert_motif(motif, scale) if random.random() < 0.5 else _generate_contrasting_motif(scale, root, scale_name, is_chorus)
        else:
            phrase_type, cadence_bar, contour = 'consequent', True, 'descending'
            motif_variant = motif
        
        if harmony_window:
            cell_rhythm, rest_flags = theory_rhythm(is_chorus, bar_in_loop, bar_length, persona)
        else:
            cell_rhythm, rest_flags = _generate_bar_rhythm(is_chorus, bar_in_loop, bar_length), None
        active_root = harmony.root if harmony else root
        register_shift = section_intent.register_shift if section_intent else 0
        active_chord = harmony.chord_tones if harmony else get_chord_notes(root, get_chord_quality(root, scale))
        cell = generate_melodic_cell(
            base=clamp_to_register(active_root + (36 if is_chorus else 24) + register_shift, 'main_melody' if not is_chorus else 'chorus_melody'),
            scale=scale, rhythm=cell_rhythm, chord=active_chord,
            bass_root=active_root, vc=vc, ts=ts, contour=contour, phrase_type=phrase_type,
            cadence_bar=cadence_bar, scale_name=scale_name, bar_length=bar_length,
            motif=motif_variant, cadence_type='authentic', is_chorus=is_chorus,
            voice='chorus_melody' if is_chorus else 'main_melody'
        )
        notes = cell.notes
        if harmony:
            notes = apply_melodic_devices(
                notes, cell.rhythm, harmony, next_harmony, scale,
                'chorus_melody' if is_chorus else 'main_melody',
                persona, phrase_role(bar_in_loop), rest_flags
            )
        loop_bars.append({
            'notes': notes,
            'rhythm': cell.rhythm,
            'role': phrase_role(bar_in_loop),
            'persona': persona['name'],
            'harmony_aware': bool(harmony),
            'section_role': section_intent.role if section_intent else None,
        })
        full_notes.extend(notes)
        full_rhythm.extend(cell.rhythm)
    
    return {
        'notes': full_notes,
        'rhythm': full_rhythm,
        'bars': loop_bars,
        'is_chorus': is_chorus,
        'motif': motif,
        'persona': persona['name'],
        'harmony_aware': bool(harmony_window),
    }

def _generate_seed_motif(scale: List[int], root: int, scale_name: str, is_chorus: bool) -> List[int]:
    if scale_name in ARMENIAN_MAQAM_SCALES:
        char_intervals = _get_scale_characteristic_intervals(scale_name, is_chorus)
    elif is_chorus: char_intervals = [0, 2, 4, 5, 7, 5, 4, 2, 0]
    else: char_intervals = [0, 1, 2, 1, 0, -1, 1, 0]
    
    motif_len = random.randint(4, 6)
    start_idx = random.randint(0, max(0, len(char_intervals) - motif_len))
    motif = char_intervals[start_idx:start_idx + motif_len]
    for i in range(len(motif)):
        if random.random() < 0.3: motif[i] += random.choice([-2, -1, 1, 2])
    return motif

def _section_motif_transform(motif: List[int], section_intent: Optional[SectionIntent]) -> List[int]:
    if not section_intent:
        return motif
    mode = section_intent.motif_transform
    out = motif[:]
    if mode == 'fragment':
        return out[:max(3, min(len(out), 4))]
    if mode == 'sequence_up':
        return [n + 2 for n in out]
    if mode == 'invert':
        return _invert_motif(out, [])
    if mode == 'retrograde':
        return list(reversed(out))
    if mode == 'compress':
        return [0 if n == 0 else int(n * 0.65) for n in out]
    if mode == 'thin':
        return out[::2] or out
    return out

def _get_scale_characteristic_intervals(scale_name: str, is_chorus: bool) -> List[int]:
    motifs = {
        'Hijaz': {'verse': [0, 1, 4, 3, 1, 0, -1], 'chorus': [0, 4, 5, 7, 5, 4, 1, 0]},
        'Hüseyni': {'verse': [0, 2, 3, 5, 3, 2, 0], 'chorus': [0, 3, 5, 7, 9, 7, 5, 3]},
        'Kurdi': {'verse': [0, 1, 3, 4, 3, 1, 0], 'chorus': [0, 3, 6, 8, 6, 3, 1, 0]},
        'Rast': {'verse': [0, 2, 4, 5, 7, 5, 4, 2], 'chorus': [0, 4, 5, 7, 9, 7, 5, 4]},
        'Bayati': {'verse': [0, 2, 3, 5, 6, 5, 3, 2], 'chorus': [0, 3, 5, 6, 8, 6, 5, 3]},
        'Nahawand': {'verse': [0, 2, 3, 5, 7, 8, 7, 5], 'chorus': [0, 3, 5, 7, 8, 10, 8, 7]},
    }
    if scale_name in motifs: return motifs[scale_name]['chorus' if is_chorus else 'verse']
    return [0, 2, 4, 5, 7, 5, 4, 2] if is_chorus else [0, 1, 2, 1, 0, -1, 1, 0]

def _apply_motif_transformations(motif: List[int], scale: List[int], root: int, scale_name: str) -> List[int]:
    result = motif[:]
    if random.random() < 0.3: result = _sequence_motif(result, random.choice([2, 3, 5]), scale)
    if random.random() < 0.2: result = _invert_motif(result, scale)
    if random.random() < 0.25 and len(result) > 3: result = result[:random.randint(2, len(result) - 1)]
    return result

def _sequence_motif(motif: List[int], interval_shift: int, scale: List[int]) -> List[int]:
    return [m + interval_shift for m in motif]

def _invert_motif(motif: List[int], scale: List[int]) -> List[int]:
    if len(motif) < 2: return motif
    inverted = [motif[0]]
    for i in range(1, len(motif)):
        inverted.append(inverted[-1] - (motif[i] - motif[i-1]))
    return inverted

def _generate_contrasting_motif(scale: List[int], root: int, scale_name: str, is_chorus: bool) -> List[int]:
    if is_chorus: return _generate_seed_motif(scale, root, scale_name, False)
    motif = _generate_seed_motif(scale, root, scale_name, True)
    for i in range(1, len(motif)):
        if random.random() < 0.4: motif[i] += random.choice([-3, -2, 2, 3])
    return motif

def _motif_targets(motif: List[int], root: int, base: int, count: int,
                   voice: str) -> List[int]:
    if not motif:
        return []
    targets = []
    for i in range(count):
        interval = motif[i % len(motif)]
        raw = base + interval
        pc = (root + interval) % 12
        target = min([pc + 12 * octv for octv in range(11)], key=lambda n: abs(n - raw))
        targets.append(clamp_to_register(target, voice))
    return targets

def _apply_phrase_cadence(notes: List[int], root: int, scale_name: str,
                          cadence_type: str, voice: str, final: bool = False) -> List[int]:
    if not notes:
        return notes
    cadence = get_phrase_cadence_intervals(scale_name, cadence_type, final=final)
    if not cadence:
        return notes
    out = notes[:]
    start = max(0, len(out) - len(cadence))
    reference = out[start - 1] if start > 0 else out[-1]
    lo, hi = (52, 86) if voice == 'chorus_melody' else (48, 84)
    for offset, interval in enumerate(cadence[-len(out[start:]):]):
        out[start + offset] = clamp_to_register(scale_degree_to_pitch(root, interval, reference, lo, hi), voice)
        reference = out[start + offset]
    return out

def _generate_bar_rhythm(is_chorus: bool, bar_in_loop: int, bar_length: int = 1920) -> List[int]:
    q, e, dq = 480, 240, 720
    if is_chorus:
        rhy_templates = [[q, e, e, q, e, e, q, e], [dq, e, q, e, q, e, e], [q, q, e, e, q, q], [e, e, q, dq, e, e, q], [q, e, dq, e, q, e]]
    else:
        rhy_templates = [[q, q, q, q], [q, e, e, q, q], [dq, e, q, q], [q, q, e, e, q], [e, e, q, q, q]]
    rhythm = random.choice(rhy_templates).copy()
    if random.random() < 0.3 and len(rhythm) > 3:
        if random.random() < 0.5: rhythm.insert(random.randint(1, len(rhythm) - 1), e)
        elif len(rhythm) > 4: rhythm.pop(random.randint(1, len(rhythm) - 1))
    return _scale_rhythm(rhythm, bar_length)

# ============================================================================
# COUNTER MELODY GENERATION
# ============================================================================

def generate_counter_melody_2bar(scale: List[int], root: int, scale_name: str,
                                  chord_root: int, chord_quality: List[int],
                                  main_melody_notes: List[int] = None,
                                  bar_length: int = 1920,
                                  harmony: Optional[BarHarmony] = None,
                                  harmony_window: Optional[List[BarHarmony]] = None) -> Dict:
    motif = _generate_counter_motif(scale, root, scale_name)
    persona = choose_melody_persona(False)
    persona['rest_chance'] = max(persona.get('rest_chance', 0.2), 0.32)
    full_notes, full_rhythm = [], []
    vc, ts = VoiceLeadingContext(), TensionState()
    for bar_in_loop in range(2):
        active_harmony = harmony_window[bar_in_loop % len(harmony_window)] if harmony_window else harmony
        next_harmony = harmony_window[(bar_in_loop + 1) % len(harmony_window)] if harmony_window else None
        if bar_in_loop == 0: phrase_type, cadence_bar, contour = 'antecedent', False, random.choice(['arch', 'ascending', 'wave'])
        else: phrase_type, cadence_bar, contour = 'consequent', True, 'descending'
        if harmony_window:
            cell_rhythm, rest_flags = theory_rhythm(False, bar_in_loop + 1, bar_length, persona)
        else:
            cell_rhythm, rest_flags = _generate_counter_rhythm(bar_in_loop, bar_length), None
        cell = generate_melodic_cell(
            base=clamp_to_register(root + 48, 'counter_melody'), scale=scale, rhythm=cell_rhythm,
            chord=active_harmony.chord_tones if active_harmony else get_chord_notes(chord_root, chord_quality),
            bass_root=active_harmony.root if active_harmony else chord_root, vc=vc, ts=ts,
            contour=contour, phrase_type=phrase_type, cadence_bar=cadence_bar, scale_name=scale_name,
            bar_length=bar_length, motif=motif, cadence_type='authentic', voice='counter_melody'
        )
        notes = cell.notes
        if active_harmony:
            notes = apply_melodic_devices(
                notes, cell.rhythm, active_harmony, next_harmony, scale,
                'counter_melody', persona, phrase_role(bar_in_loop + 1), rest_flags
            )
        full_notes.extend(notes)
        full_rhythm.extend(cell.rhythm)
    return {'notes': full_notes, 'rhythm': full_rhythm, 'motif': motif}

def _generate_counter_motif(scale: List[int], root: int, scale_name: str) -> List[int]:
    if scale_name in ARMENIAN_MAQAM_SCALES:
        motifs = {
            'Hijaz': [0, 1, 4, 3, 1, 0], 'Hüseyni': [0, 2, 3, 5, 3, 2], 'Kurdi': [0, 1, 3, 4, 3, 1],
            'Rast': [0, 2, 4, 5, 7, 5], 'Bayati': [0, 2, 3, 5, 6, 5], 'Nahawand': [0, 2, 3, 5, 7, 8],
        }
        char_intervals = motifs.get(scale_name, [0, 2, 4, 5, 7, 5])
    else: char_intervals = [0, 2, 4, 5, 7, 5, 4, 2]
    motif_len = random.randint(4, 6)
    start_idx = random.randint(0, max(0, len(char_intervals) - motif_len))
    motif = char_intervals[start_idx:start_idx + motif_len]
    for i in range(len(motif)):
        if random.random() < 0.2: motif[i] += random.choice([-1, 1])
    return motif

def _generate_counter_rhythm(bar_in_loop: int, bar_length: int = 1920) -> List[int]:
    q, e, dq, h = 480, 240, 720, 960
    rhy_templates = [[q, q, q, q], [dq, e, q, q], [q, e, e, h], [e, e, q, dq], [q, q, h], [h, q, q]]
    rhythm = random.choice(rhy_templates).copy()
    if random.random() < 0.2 and len(rhythm) > 3:
        if random.random() < 0.5: rhythm.insert(random.randint(1, len(rhythm) - 1), e)
        elif len(rhythm) > 3: rhythm.pop(random.randint(1, len(rhythm) - 1))
    return _scale_rhythm(rhythm, bar_length)

# ============================================================================
# RHYTHM SCALING
# ============================================================================

def _scale_rhythm(rhythm, bar_length):
    """Scale rhythm template proportionally to fit bar_length."""
    original_total = sum(rhythm)
    if original_total == 0:
        return rhythm
    scale = bar_length / original_total
    scaled = [max(120, int(r * scale)) for r in rhythm]
    diff = bar_length - sum(scaled)
    if scaled:
        scaled[-1] += diff
    return scaled

# ============================================================================
# TIME-SIG-SPECIFIC BASS PATTERNS
# ============================================================================

BASS_PATTERNS = {
    '4-4': {
        # Original patterns
        'whole': [1920], 'dotted_half': [1440, 480], 'half_quarter': [960, 960],
        'lofi_pocket': [720, 240, 480, 480], 'syncopated': [480, 480, 720, 240],
        'standard': [960, 480, 480], 'root_fifth': [960, 960],
        'active': [480, 240, 240, 480, 480],
        # Tech house bass patterns
        'tech_house_offbeat': [480, 240, 240, 480, 480],  # Offbeat emphasis
        'tech_house_driving': [240, 240, 480, 240, 240, 480],  # Driving 8th notes
        'tech_house_minimal': [960, 480, 480],  # Minimal, root-fifth
        'tech_house_rolling': [480, 240, 240, 480, 240, 240],  # Rolling 16ths
        'tech_house_sub': [1920],  # Long sub notes
        'tech_house_chop': [240, 240, 240, 240, 480, 480],  # Chopped/staccato
    },
    '3-4': {
        'waltz': [960, 480], 'flowing': [480, 480, 480],
        'lofi_pocket': [720, 240, 480], 'syncopated': [480, 240, 240, 480],
        'standard': [960, 480], 'root_fifth': [480, 480, 480],
        'active': [480, 240, 240, 480],
    },
    '5-4': {
        'money': [480, 480, 480, 480, 480], 'dotted': [720, 720, 480, 480],
        'lofi_pocket': [720, 240, 480, 480, 480], 'syncopated': [480, 480, 720, 240, 480],
        'standard': [960, 480, 480, 480], 'root_fifth': [960, 960, 480],
        'active': [480, 240, 240, 480, 480, 480],
    },
    '5-8': {
        'tight': [480, 240, 240, 240], 'bouncy': [240, 240, 480, 240],
        'lofi_pocket': [360, 120, 240, 240], 'syncopated': [240, 240, 360, 120],
        'standard': [480, 240, 240], 'root_fifth': [480, 480],
        'active': [240, 120, 120, 240, 240],
    },
}

# ============================================================================
# BASS GENERATION
# ============================================================================

def generate_bass(root: int, quality: str, scale: List[int], vc: VoiceLeadingContext,
                  rhythm: Optional[List[int]] = None, pattern_style: str = 'standard',
                  bar: int = 0, bar_length: int = 1920, time_sig: str = '4-4',
                  harmony: Optional[BarHarmony] = None,
                  next_harmony: Optional[BarHarmony] = None,
                  section_intent: Optional[SectionIntent] = None) -> MelodicCell:
    patterns = BASS_PATTERNS.get(time_sig, BASS_PATTERNS['4-4'])
    bar_mod = bar % 4
    if rhythm is None:
        if bar_mod == 0 or bar_mod == 2:
            rhythm = patterns.get('standard', patterns.get('root_fifth'))
        elif bar_mod == 1:
            rhythm = patterns.get('lofi_pocket', patterns.get('standard'))
        else:
            rhythm = patterns.get('syncopated', patterns.get('standard'))

    if sum(rhythm) != bar_length:
        rhythm = _scale_rhythm(rhythm, bar_length)

    chord = harmony.bass_tones if harmony else get_chord_notes(root, quality)
    root_note = harmony.root if harmony else chord[0]
    bass_lo, bass_hi = BASS_REGISTER_RANGE
    while root_note < bass_lo: root_note += 12
    while root_note > bass_hi: root_note -= 12
    chord_pcs = {n % 12 for n in chord}
    fifth = next((root_note + interval for interval in (7, 10, 12, 14) if (root_note + interval) % 12 in chord_pcs), root_note + 7)
    octave = root_note + 12
    approach_tones = [n for n in chord if n % 12 not in {root_note % 12, fifth % 12}]

    notes = []
    pedal_active = bool(harmony and should_use_pedal(harmony.section, bar))
    approach_note = bass_approach_note(root_note, next_harmony, scale)
    density = section_intent.density if section_intent else 0.55
    is_peak = bool(section_intent and section_intent.role in {'lift', 'peak', 'tension', 'release_setup'})
    for i in range(len(rhythm)):
        if i == 0: notes.append(root_note)
        elif density < 0.45 and bar_mod in [1, 3] and random.random() < 0.34: notes.append(None)
        elif i == len(rhythm) - 1 and approach_note is not None and random.random() < (0.72 if is_peak else 0.55): notes.append(approach_note)
        elif pedal_active and random.random() < 0.65: notes.append(root_note)
        elif i == len(rhythm) - 1 and bar_mod == 3: notes.append(octave)
        elif pattern_style == 'active' and approach_tones and random.random() < (0.36 if is_peak else 0.25): notes.append(random.choice(approach_tones))
        elif random.random() < (0.38 if density > 0.62 else 0.3): notes.append(fifth)
        else: notes.append(root_note)

    vc.bass_last_note = next((n for n in reversed(notes) if n is not None), None)
    return MelodicCell(notes=notes, rhythm=rhythm[:len(notes)])

def generate_harmonic_bass(root: int, quality: str, scale: List[int], vc: VoiceLeadingContext,
                           rhythm: Optional[List[int]] = None, pattern_style: str = 'standard',
                           bar: int = 0, bar_length: int = 1920, time_sig: str = '4-4',
                           harmony: Optional[BarHarmony] = None,
                           section_intent: Optional[SectionIntent] = None) -> MelodicCell:
    bar_mod = bar % 4
    if rhythm is None:
        patterns = BASS_PATTERNS.get(time_sig, BASS_PATTERNS['4-4'])
        if bar_mod == 3:
            rhythm = patterns.get('active', [480, 480, 240, 240, 480])
        else:
            rhythm = patterns.get('root_fifth', [960, 960])
    if sum(rhythm) != bar_length:
        rhythm = _scale_rhythm(rhythm, bar_length)

    chord = harmony.chord_tones if harmony else get_chord_notes(root, quality)
    root_h = chord[0]
    bass_lo, bass_hi = BASS_REGISTER_RANGE
    while root_h < bass_lo: root_h += 12
    while root_h > bass_hi: root_h -= 12
    chord_h = [n for n in chord]
    for i in range(len(chord_h)):
        while chord_h[i] < root_h: chord_h[i] += 12

    notes = []
    density = section_intent.density if section_intent else 0.55
    guide_pcs = {n % 12 for n in harmony.guide_tones} if harmony else set()
    for i in range(len(rhythm)):
        if density < 0.44 and i > 0 and random.random() < 0.35:
            notes.append(None)
        elif guide_pcs and i > 0 and random.random() < 0.45:
            guides = [n for n in chord_h if n % 12 in guide_pcs]
            notes.append(random.choice(guides or chord_h))
        elif i == 0 or not (len(rhythm) > 2 and i >= 2): notes.append(random.choice(chord_h))
        else:
            last = notes[-1] if notes[-1] is not None else root_h
            try:
                idx = scale.index(last % 12)
                notes.append(scale[(idx + random.choice([-1, 1])) % len(scale)] + (last // 12) * 12)
            except: notes.append(random.choice(chord_h))
    return MelodicCell(notes=notes, rhythm=rhythm[:len(notes)])

# ============================================================================
# PAD CHORD GENERATION
# ============================================================================

def get_chord_extensions(root: int, quality: str, scale: List[int], bar_type: str, 
                         abac_position: str) -> List[int]:
    if quality == 'major': base = [root, root+4, root+7]
    else: base = [root, root+3, root+7]
    
    if abac_position == 'A': return [root, root+4, root+7, root+11] if quality == 'major' else [root, root+3, root+7, root+10]
    elif abac_position == 'B': return [root, root+4, root+7, root+11, root+14] if quality == 'major' else [root, root+3, root+7, root+10, root+14]
    elif abac_position == 'C': return [root, root+4, root+7, root+11, root+14, root+21] if quality == 'major' else [root, root+3, root+7, root+10, root+14, root+17]
    
    if bar_type == 'intro': return [root, root+4, root+7, root+14] if quality == 'major' else [root, root+3, root+7, root+14]
    elif bar_type.startswith('chorus'): return [root, root+4, root+7, root+11, root+14] if quality == 'major' else [root, root+3, root+7, root+10, root+14]
    return base

def get_chord_inversion(chord: List[int], bar_type: str, beat: int) -> List[int]:
    if len(chord) < 3 or beat == 0: return chord
    if beat == 2: return chord[1:] + [chord[0] + 12]
    return [chord[2]] + chord[:2] + [chord[0] + 12]

def get_chord_voicing(chord: List[int], register: str = 'pad') -> List[int]:
    if len(chord) < 4: return sorted(chord)
    sorted_chord = sorted(chord)
    voiced = [sorted_chord[-2] - 12] + sorted_chord[:-2] + [sorted_chord[-1]]
    return sorted(voiced)

def generate_pad_chord(root: int, quality: str, scale: List[int], bar_type: str,
                       abac_position: str, beat: int = 0) -> List[int]:
    chord = get_chord_extensions(root, quality, scale, bar_type, abac_position)
    chord = get_chord_inversion(chord, bar_type, beat)
    chord = get_chord_voicing(chord, 'pad')
    final_chord = []
    for note in chord:
        while note < 72: note += 12
        while note > 96: note -= 12
        final_chord.append(note)
    return sorted(final_chord)

# ============================================================================
# MELODY GENERATION & VOICE LEADING
# ============================================================================

def select_next_note(cur: int, scale: List[int], chord: List[int], prev_bass: Optional[int],
                     prev_mel: Optional[int], bass: int, beat: int, ts: TensionState,
                     phrase_pos: int = 0, cadence_bar: bool = False,
                     scale_name: str = 'western', motif_target: Optional[int] = None,
                     phrase_target: Optional[int] = None, is_chorus: bool = False) -> int:
    candidates = []
    for oct_off in [-12, 0, 12]:
        for sn in scale:
            c = (cur // 12) * 12 + (sn % 12) + oct_off
            if 0 < abs(c - cur) <= 12 and 48 <= c <= 84: candidates.append(c)
    if not candidates: return cur
    scored = []
    for c in candidates:
        s = 0.0
        if prev_bass and prev_mel and check_parallel_perfects(bass, prev_bass, c, prev_mel): s -= 2.0
        s += score_voice_leading(get_motion_direction(bass, prev_bass) if prev_bass else 0, get_motion_direction(c, prev_mel) if prev_mel else 0) * 0.5
        s += score_interval(c, cur, chord, beat)
        if motif_target is not None:
            pc_distance = min((c - motif_target) % 12, (motif_target - c) % 12)
            if c % 12 == motif_target % 12:
                s += 1.4
            elif pc_distance <= 2:
                s += 0.6
        if phrase_target is not None:
            remaining_weight = max(0.2, phrase_pos / 8.0)
            s += max(0.0, 1.2 - (abs(c - phrase_target) / 12.0)) * remaining_weight
        if cadence_bar:
            if c % 12 == chord[0] % 12: s += 1.5
            elif len(chord) > 2 and c % 12 == chord[2] % 12: s += 1.0
        emphasis = get_emphasis_intervals(scale_name)
        if (c - bass) % 12 in emphasis and beat % 2 == 0:
            s += 0.4
        if is_chorus and beat in (0, 2) and c % 12 in [ch % 12 for ch in chord]:
            s += 0.5
        scored.append((c, s))
    mx = max(sc for _, sc in scored)
    return random.choices([c for c,_ in scored], weights=[max(0.01, sc - mx + 1.0) for _, sc in scored], k=1)[0]

def generate_melodic_cell(base: int, scale: List[int], rhythm: List[int], chord: List[int],
                          bass_root: int, vc: VoiceLeadingContext, ts: TensionState, 
                          contour: str = "arch", phrase_type: str = "antecedent",
                          cadence_bar: bool = False, scale_name: str = 'western',
                          bar_length: int = 1920, motif: Optional[List[int]] = None,
                          cadence_type: str = 'authentic', is_chorus: bool = False,
                          voice: str = 'main_melody') -> MelodicCell:
    notes, cur, tick = [], base, 0
    motif_targets = _motif_targets(motif or [], bass_root, base, len(rhythm), voice)
    final_interval = 0 if cadence_bar else random.choice(get_emphasis_intervals(scale_name))
    phrase_target = scale_degree_to_pitch(bass_root, final_interval, base, 48, 86)
    for i, dur in enumerate(rhythm):
        if tick + dur > bar_length: break
        motif_target = motif_targets[i] if i < len(motif_targets) else None
        phrase_pos = int((i / max(1, len(rhythm) - 1)) * 8)
        nxt = select_next_note(
            cur, scale, chord, vc.bass_last_note, vc.melody_last_note, bass_root, i, ts,
            phrase_pos, cadence_bar, scale_name, motif_target, phrase_target, is_chorus
        )
        if vc.bass_last_note is not None and vc.melody_last_note is not None:
            nxt = correct_parallel_perfects(nxt, bass_root, vc.melody_last_note, vc.bass_last_note, scale)
        if phrase_type == "antecedent":
            if i < len(rhythm) // 2 and nxt < cur: nxt += 12
            elif i >= len(rhythm) // 2 and nxt > cur and not cadence_bar: nxt -= 12
        elif phrase_type == "consequent":
            if i < len(rhythm) * 2 // 3 and nxt < cur: nxt += 12
            elif i >= len(rhythm) * 2 // 3 and nxt > cur: nxt -= 12
        notes.append(nxt)
        ts.current_tension, vc.melody_last_note, cur, tick = calculate_tension(nxt, chord), nxt, nxt, tick + dur
    if cadence_bar:
        notes = _apply_phrase_cadence(notes, bass_root, scale_name, cadence_type, voice, final=True)
    return MelodicCell(notes=notes, rhythm=rhythm[:len(notes)], contour=contour, phrase_type=phrase_type, ornaments=[{'type': 'armenian', 'scale': scale_name}] if scale_name in ARMENIAN_MAQAM_SCALES else [])

def generate_euclidean(pulses: int, steps: int) -> List[int]:
    if pulses > steps: pulses = steps
    pattern, counts, remainders = [], [], []
    divisor = steps - pulses
    remainders.append(pulses)
    level = 0
    while True:
        counts.append(divisor // remainders[level])
        remainders.append(divisor % remainders[level])
        divisor = remainders[level]
        level += 1
        if remainders[level] <= 1: break
    counts.append(divisor)
    def build(l):
        if l == -1: pattern.append(0)
        elif l == -2: pattern.append(1)
        else:
            for _ in range(counts[l]): build(l - 1)
            if remainders[l] != 0: build(l - 2)
    build(level)
    pattern.reverse()
    return pattern[:steps]
