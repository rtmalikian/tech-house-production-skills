import re
from typing import List, Dict, Tuple, Optional, Union
from midi_models import IntervalQuality, MicrotonalNote, CadencePlan, ChordSpec, BarHarmony
from midi_config import REGISTER_RANGES

# ============================================================================
# MUSIC THEORY CONSTANTS
# ============================================================================

INTERVAL_QUALITIES = {
    0: IntervalQuality.PERFECT_CONSONANCE, 1: IntervalQuality.DISSONANCE,
    2: IntervalQuality.STRONG_TENSION, 3: IntervalQuality.IMPERFECT_CONSONANCE,
    4: IntervalQuality.IMPERFECT_CONSONANCE, 5: IntervalQuality.MILD_TENSION,
    6: IntervalQuality.DISSONANCE, 7: IntervalQuality.PERFECT_CONSONANCE,
    8: IntervalQuality.STRONG_TENSION, 9: IntervalQuality.IMPERFECT_CONSONANCE,
    10: IntervalQuality.STRONG_TENSION, 11: IntervalQuality.STRONG_TENSION,
}

CADENCE_TYPES = {'authentic': [5, 0], 'plagal': [3, 0], 'half': [0, 5], 'deceptive': [5, 6]}

ARMENIAN_MAQAM_SCALES = {
    'Hijaz': {
        'intervals': [0, 1, 4, 5, 7, 8, 11],
        'characteristic': 'Augmented 2nd (1-4)',
        'microtonal': {1: -50, 8: -50},
        'directional': True,
        'ornaments': ['mordent', 'grace_note'],
        'mood': 'exotic, melancholic',
    },
    'Hüseyni': {
        'intervals': [0, 2, 3, 5, 7, 9, 10],
        'characteristic': 'Very flat 2nd',
        'microtonal': {2: -75, 10: -50},
        'directional': True,
        'ornaments': ['trill', 'mordent'],
        'mood': 'mystical, spiritual',
    },
    'Kurdi': {
        'intervals': [0, 1, 3, 4, 6, 8, 10],
        'characteristic': 'Flat 2nd, flat 5th',
        'microtonal': {1: -40, 6: -30},
        'directional': True,
        'ornaments': ['grace_note', 'turn'],
        'mood': 'sad, lamenting',
    },
    'Rast': {
        'intervals': [0, 2, 4, 5, 7, 9, 10],
        'characteristic': 'Slightly flat major 3rd',
        'microtonal': {4: -15, 10: -25},
        'directional': True,
        'ornaments': ['trill'],
        'mood': 'heroic, majestic',
    },
    'Bayati': {
        'intervals': [0, 2, 3, 5, 6, 8, 10],
        'characteristic': 'Quarter-tone 2nd',
        'microtonal': {2: -55, 8: -40},
        'directional': True,
        'ornaments': ['mordent', 'grace_note', 'turn'],
        'mood': 'folk, pastoral',
    },
    'Nahawand': {
        'intervals': [0, 2, 3, 5, 7, 8, 10],
        'characteristic': 'Augmented 2nd (6-7)',
        'microtonal': {3: -20, 8: -35},
        'directional': True,
        'ornaments': ['trill', 'mordent'],
        'mood': 'dramatic, passionate',
    },
}

MODE_INTERVALS = {
    "Major": [0, 2, 4, 5, 7, 9, 11], "Dorian": [0, 2, 3, 5, 7, 9, 10],
    "Phrygian": [0, 1, 3, 5, 7, 8, 10], "Lydian": [0, 2, 4, 6, 7, 9, 11],
    "Mixolydian": [0, 2, 4, 5, 7, 9, 10], "Minor": [0, 2, 3, 5, 7, 8, 10],
    "Locrian": [0, 1, 3, 4, 6, 8, 10]
}

ARMENIAN_CADENCE_INTERVALS = {
    'Hijaz': {
        'final': [4, 1, 0], 'half': [1, 4, 5], 'turnaround': [7, 5, 4, 1, 0],
        'emphasis': [0, 1, 4, 5, 7],
    },
    'Hüseyni': {
        'final': [3, 2, 0], 'half': [5, 7, 5], 'turnaround': [7, 5, 3, 2, 0],
        'emphasis': [0, 2, 3, 5, 7],
    },
    'Kurdi': {
        'final': [3, 1, 0], 'half': [4, 6, 4], 'turnaround': [6, 4, 3, 1, 0],
        'emphasis': [0, 1, 3, 4, 6],
    },
    'Rast': {
        'final': [5, 4, 2, 0], 'half': [4, 5, 7], 'turnaround': [7, 5, 4, 2, 0],
        'emphasis': [0, 2, 4, 5, 7],
    },
    'Bayati': {
        'final': [3, 2, 0], 'half': [5, 6, 5], 'turnaround': [6, 5, 3, 2, 0],
        'emphasis': [0, 2, 3, 5, 6],
    },
    'Nahawand': {
        'final': [5, 3, 2, 0], 'half': [7, 8, 7], 'turnaround': [10, 8, 7, 5, 3, 2, 0],
        'emphasis': [0, 2, 3, 5, 7],
    },
}

MAJOR_MINOR_CADENCE_INTERVALS = {
    'major': {
        'authentic': [2, 0], 'half': [2, 5], 'plagal': [9, 7, 4], 'deceptive': [11, 9],
        'emphasis': [0, 2, 4, 7, 9],
    },
    'minor': {
        'authentic': [2, 0], 'leading_authentic': [11, 0], 'half': [2, 7],
        'plagal': [8, 7, 3], 'deceptive': [10, 8],
        'emphasis': [0, 2, 3, 7, 8, 10],
    },
}

def pitch_class_set(scale: List[int]) -> set:
    return {n % 12 for n in scale}

def nearest_pitch_for_pc(reference: int, pc: int, low: int = 0, high: int = 127) -> int:
    candidates = [pc + 12 * octave for octave in range(11)]
    candidates = [c for c in candidates if low <= c <= high]
    if not candidates:
        return max(low, min(high, pc))
    return min(candidates, key=lambda n: abs(n - reference))

def scale_degree_to_pitch(root: int, interval: int, reference: int,
                          low: int = 48, high: int = 84) -> int:
    return nearest_pitch_for_pc(reference, (root + interval) % 12, low, high)

def get_scale_degree_for_note(note: int, root: int) -> int:
    return (note - root) % 12

def is_major_minor_scale(scale_name: str) -> bool:
    return 'Major' in scale_name or 'Minor' in scale_name

def get_major_minor_mode(scale_name: str) -> str:
    return 'minor' if 'Minor' in scale_name else 'major'

def get_phrase_cadence_intervals(scale_name: str, cadence_type: str = 'authentic',
                                 final: bool = False) -> List[int]:
    if scale_name in ARMENIAN_CADENCE_INTERVALS:
        table = ARMENIAN_CADENCE_INTERVALS[scale_name]
        if final:
            return table.get('final', [2, 0])
        return table.get('half' if cadence_type == 'half' else 'final', table.get('final', [2, 0]))
    if is_major_minor_scale(scale_name):
        mode = get_major_minor_mode(scale_name)
        table = MAJOR_MINOR_CADENCE_INTERVALS[mode]
        if mode == 'minor' and cadence_type == 'authentic':
            return table['leading_authentic'] if final else table['authentic']
        return table.get(cadence_type, table['authentic'])
    return [2, 0] if cadence_type != 'half' else [2, 7]

def get_emphasis_intervals(scale_name: str) -> List[int]:
    if scale_name in ARMENIAN_CADENCE_INTERVALS:
        return ARMENIAN_CADENCE_INTERVALS[scale_name]['emphasis']
    if is_major_minor_scale(scale_name):
        return MAJOR_MINOR_CADENCE_INTERVALS[get_major_minor_mode(scale_name)]['emphasis']
    return [0, 2, 4, 5, 7]

def filter_scale_tones(notes: List[int], scale: List[int]) -> List[int]:
    pcs = pitch_class_set(scale)
    return [n for n in notes if n % 12 in pcs]

def get_modal_pad_chord(root: int, quality: str, scale: List[int], scale_name: str,
                        section_energy: float = 0.6) -> List[int]:
    pcs = pitch_class_set(scale)
    root_pc = root % 12
    intervals = [0, 7]
    if scale_name in ARMENIAN_MAQAM_SCALES:
        color_by_mode = {
            'Hijaz': [1, 5], 'Hüseyni': [2, 3, 7], 'Kurdi': [1, 3, 6],
            'Rast': [2, 4, 7], 'Bayati': [2, 3, 5], 'Nahawand': [3, 5, 7],
        }
        intervals.extend(color_by_mode.get(scale_name, [2, 5]))
    elif quality == 'minor':
        intervals.extend([3, 10 if section_energy > 0.75 else 5])
    else:
        intervals.extend([4, 9 if section_energy > 0.75 else 5])
    notes = []
    for interval in intervals:
        pc = (root_pc + interval) % 12
        if pc in pcs:
            notes.append(root + interval)
    return notes or [root, root + 7]

def get_microtonal_offset_for_root(note: int, root: int, scale_name: str,
                                   ascending: bool = True) -> int:
    if scale_name not in ARMENIAN_MAQAM_SCALES:
        return 0
    interval = (note - root) % 12
    scale = ARMENIAN_MAQAM_SCALES[scale_name]
    for idx, scale_interval in enumerate(scale['intervals']):
        if scale_interval % 12 == interval:
            base_offset = scale.get('microtonal', {}).get(idx, 0)
            if base_offset and scale.get('directional', False):
                return base_offset + 15 if ascending else base_offset - 15
            return base_offset
    return 0

def get_chord_notes(root: int, quality: str, inv: int = 0) -> List[int]:
    """Get chord notes. Tech house uses minor 7ths as default."""
    intervals = {
        'major': [0, 4, 7],
        'minor': [0, 3, 7],
        'dom7': [0, 4, 7, 10],
        'min7': [0, 3, 7, 10],      # Minor 7th — THE tech house chord
        'maj7': [0, 4, 7, 11],      # Major 7th — dreamy, lush
        'min9': [0, 3, 7, 10, 14],  # Minor 9th — rich, sophisticated
        'add9': [0, 4, 7, 14],      # Add9 — brighter than plain triad
        'sus2': [0, 2, 7],          # Suspended 2nd — ambiguous, floating
        'sus4': [0, 5, 7],          # Suspended 4th — tension
        'dim': [0, 3, 6],           # Diminished — dark
        'aug': [0, 4, 8],           # Augmented — tense
    }.get(quality, [0, 3, 7, 10])  # Default to minor 7th for tech house
    chord = [root + i for i in intervals]
    if inv == 1 and len(chord) >= 3:
        chord = chord[1:] + [chord[0] + 12]
    elif inv == 2 and len(chord) >= 3:
        chord = chord[2:] + [chord[0] + 12, chord[1] + 12]
    return sorted(chord)

def get_chord_quality(root: int, scale: List[int]) -> str:
    rc = root % 12
    sc = [n % 12 for n in scale]
    if (rc + 4) % 12 in sc:
        return 'major'
    return 'minor' if (rc + 3) % 12 in sc else 'major'

ROMAN_TO_OFFSET = {
    'I': 0, 'II': 2, 'III': 4, 'IV': 5, 'V': 7, 'VI': 9, 'VII': 11,
}

OFFSET_TO_ROMAN = {
    0: 'I', 1: 'bII', 2: 'ii', 3: 'bIII', 4: 'iii', 5: 'IV',
    6: 'bV', 7: 'V', 8: 'bVI', 9: 'vi', 10: 'bVII', 11: 'vii',
}

def symbol_for_degree(degree: int) -> str:
    return OFFSET_TO_ROMAN.get(degree % 12, 'I')

def _parse_roman_root(symbol: str) -> Tuple[int, str, str]:
    accidental = 0
    rest = symbol.strip()
    while rest.startswith(('b', '#')):
        accidental += -1 if rest[0] == 'b' else 1
        rest = rest[1:]
    match = re.match(r'(vii|VII|vi|VI|iv|IV|iii|III|ii|II|i|I|v|V)(.*)$', rest)
    if not match:
        return 0, 'I', rest
    roman, suffix = match.groups()
    return (ROMAN_TO_OFFSET[roman.upper()] + accidental) % 12, roman, suffix

def parse_chord_symbol(chord: Union[int, str], scale: List[int], root: int = 0) -> ChordSpec:
    if isinstance(chord, int):
        degree = chord % 12
        symbol = symbol_for_degree(degree)
        quality = get_chord_quality(root + degree, scale)
        return ChordSpec(symbol=symbol, root_degree=degree, root_offset=degree, quality=quality)

    symbol = chord.strip()
    if not symbol:
        return parse_chord_symbol(0, scale, root)

    is_secondary = '/' in symbol
    target_degree = None
    primary = symbol
    if is_secondary:
        primary, target = symbol.split('/', 1)
        target_degree, _, _ = _parse_roman_root(target)

    degree, roman, suffix = _parse_roman_root(primary)
    if is_secondary and target_degree is not None:
        degree = (target_degree + degree) % 12

    suffix_l = suffix.lower().replace('-', '')
    quality = 'minor' if roman.islower() else 'major'
    if 'dim' in suffix_l or roman == 'vii':
        quality = 'dim'
    elif 'aug' in suffix_l:
        quality = 'aug'
    elif 'maj7' in suffix_l:
        quality = 'maj7'
    elif 'm9' in suffix_l or (roman.islower() and '9' in suffix_l):
        quality = 'min9'
    elif 'm7' in suffix_l or (roman.islower() and '7' in suffix_l):
        quality = 'min7'
    elif is_secondary or suffix_l.startswith('7') or 'dom' in suffix_l:
        quality = 'dom7'

    suspension = None
    if 'sus2' in suffix_l:
        suspension = 2
    elif 'sus4' in suffix_l or 'sus' in suffix_l:
        suspension = 5

    extensions = []
    if 'add9' in suffix_l:
        extensions.append(14)
    if '6' in suffix_l and 'sus' not in suffix_l:
        extensions.append(9)
    if '9' in suffix_l and 'add9' not in suffix_l:
        extensions.append(14)
    if '11' in suffix_l:
        extensions.append(17)
    if '13' in suffix_l:
        extensions.append(21)

    return ChordSpec(
        symbol=symbol,
        root_degree=degree,
        root_offset=degree,
        quality=quality,
        extensions=extensions,
        suspension=suspension,
        is_secondary=is_secondary,
        target_degree=target_degree,
    )

def build_chord_tones(root_note: int, spec: ChordSpec, scale: List[int],
                      scale_name: str = 'western') -> List[int]:
    if spec.quality in {'minor', 'min7', 'min9'}:
        intervals = [0, 3, 7]
    elif spec.quality == 'dim':
        intervals = [0, 3, 6]
    elif spec.quality == 'aug':
        intervals = [0, 4, 8]
    else:
        intervals = [0, 4, 7]

    if spec.suspension is not None:
        intervals = [i for i in intervals if i not in (3, 4)]
        intervals.append(spec.suspension)

    if spec.quality in {'maj7'}:
        intervals.append(11)
    elif spec.quality in {'dom7', 'min7', 'min9'}:
        intervals.append(10)
    elif any(ext in spec.extensions for ext in (14, 17, 21)) and spec.quality == 'major':
        intervals.append(11)

    intervals.extend(spec.extensions)
    intervals = sorted(set(intervals))

    pcs = pitch_class_set(scale)
    tones = []
    for interval in intervals:
        note = root_note + interval
        is_core = interval % 12 in {0, 3, 4, 5, 6, 7, 8, 10, 11}
        allow_color = spec.is_secondary or scale_name not in ARMENIAN_MAQAM_SCALES
        if is_core or allow_color or note % 12 in pcs:
            tones.append(note)
    return sorted(set(tones)) or get_chord_notes(root_note, get_chord_quality(root_note, scale))

def _fit_notes_near(notes: List[int], low: int, high: int) -> List[int]:
    fitted = []
    for note in notes:
        n = note
        while n < low:
            n += 12
        while n > high:
            n -= 12
        fitted.append(n)
    return sorted(set(fitted))

def guide_tones_for_chord(chord_tones: List[int]) -> List[int]:
    if len(chord_tones) <= 2:
        return chord_tones
    roots = {chord_tones[0] % 12, chord_tones[2] % 12 if len(chord_tones) > 2 else chord_tones[0] % 12}
    guides = [n for n in chord_tones if n % 12 not in roots]
    return guides or chord_tones[1:]

def build_bar_harmony(bar: int, section: str, chord: Union[int, str], base: int,
                      scale: List[int], scale_name: str = 'western') -> BarHarmony:
    spec = parse_chord_symbol(chord, scale, base)
    root_note = base + spec.root_offset
    while root_note > 48:
        root_note -= 12
    while root_note < 24:
        root_note += 12
    chord_tones = build_chord_tones(root_note, spec, scale, scale_name)
    return BarHarmony(
        bar=bar,
        section=section,
        root=root_note,
        degree=spec.root_offset,
        spec=spec,
        chord_tones=chord_tones,
        guide_tones=guide_tones_for_chord(chord_tones),
        bass_tones=_fit_notes_near(chord_tones, 24, 52),
        scale_tones=scale,
    )

def nearest_chord_or_scale_tone(note: int, harmony: BarHarmony, voice: str,
                                strong: bool = False) -> int:
    pool = harmony.chord_tones if strong else sorted(set(harmony.chord_tones + harmony.scale_tones))
    if not pool:
        return note
    pcs = {p % 12 for p in pool}
    candidates = [pc + 12 * octave for pc in pcs for octave in range(11)]
    fitted = [clamp_to_register(c, voice) for c in candidates]
    return min(fitted, key=lambda n: abs(n - note))

def get_interval_class(note1: int, note2: int) -> int:
    return abs(note1 - note2) % 12

def get_interval_type(interval: int) -> str:
    if interval <= 2:
        return 'stepwise'
    elif interval <= 5:
        return 'small_leap'
    else:
        return 'large_leap'

def clamp_to_register(note: int, voice: str) -> int:
    if voice not in REGISTER_RANGES:
        return note
    lo, hi = REGISTER_RANGES[voice]
    pc = note % 12
    for oct in range(10):
        c = pc + oct * 12
        if lo <= c <= hi:
            return c
    return lo if note < lo else hi

def calculate_tension(note: int, chord: List[int]) -> float:
    if not chord:
        return 0.5
    dist = min(abs((note - c) % 12) for c in chord)
    return 1.0 - INTERVAL_QUALITIES.get(dist, IntervalQuality.MILD_TENSION).value

def plan_cadences(total: int) -> List[CadencePlan]:
    return [CadencePlan(b, t, CADENCE_TYPES.get(t, [0, 5]))
            for b, t in [
                (7, 'half'), (15, 'half'), (23, 'authentic'), (31, 'authentic'),
                (35, 'half'), (43, 'half'), (51, 'authentic'), (59, 'authentic'),
                (63, 'half'), (71, 'plagal')
            ] if b < total]

def score_interval(candidate: int, current_note: int, chord: List[int], beat: int) -> float:
    interval = abs(candidate - current_note)
    interval_type = get_interval_type(interval)
    
    base_scores = {
        'stepwise': 2.0,
        'small_leap': 1.0,
        'large_leap': 0.3,
    }
    score = base_scores.get(interval_type, 0.5)
    
    if beat % 2 == 0:
        dist = min(abs((candidate - ch) % 12) for ch in chord) if chord else 7
        if dist in [0, 3, 4, 7]:
            score += 0.5
    
    if interval_type == 'large_leap':
        score -= 0.3
    
    return score

# ============================================================================
# VOICE LEADING
# ============================================================================

def is_parallel_perfect(interval1: int, interval2: int) -> bool:
    perfect = {0, 7}
    return interval1 in perfect and interval2 in perfect and interval1 == interval2

def check_parallel_perfects(bass_current: int, bass_prev: int, melody_current: int, melody_prev: int) -> bool:
    if bass_prev is None or melody_prev is None:
        return False
    curr_interval = (melody_current - bass_current) % 12
    prev_interval = (melody_prev - bass_prev) % 12
    return is_parallel_perfect(curr_interval, prev_interval)

def correct_parallel_perfects(melody_note: int, bass_note: int, 
                               prev_melody: int, prev_bass: int,
                               scale: List[int]) -> int:
    if not check_parallel_perfects(bass_note, prev_bass, melody_note, prev_melody):
        return melody_note
    bass_motion = bass_note - prev_bass
    if bass_motion > 0:
        for offset in [-1, -2, 1, 2]:
            candidate = melody_note + offset
            if candidate % 12 in [s % 12 for s in scale]:
                if not is_parallel_perfect((candidate - bass_note) % 12, (prev_melody - prev_bass) % 12):
                    return candidate
    else:
        for offset in [1, 2, -1, -2]:
            candidate = melody_note + offset
            if candidate % 12 in [s % 12 for s in scale]:
                if not is_parallel_perfect((candidate - bass_note) % 12, (prev_melody - prev_bass) % 12):
                    return candidate
    return melody_note

def score_voice_leading(bass_motion: int, melody_motion: int) -> float:
    if bass_motion * melody_motion < 0: return 1.0
    if bass_motion == 0 or melody_motion == 0: return 0.7
    if bass_motion == melody_motion: return 0.4
    return 0.3

def get_motion_direction(current: int, previous: int) -> int:
    if previous is None: return 0
    if current > previous: return 1
    elif current < previous: return -1
    return 0

# ============================================================================
# MICROTONAL & ORNAMENTATION
# ============================================================================

def get_microtonal_offset(note: int, scale_name: str, ascending: bool = True) -> int:
    if scale_name not in ARMENIAN_MAQAM_SCALES:
        return 0
    
    scale = ARMENIAN_MAQAM_SCALES[scale_name]
    microtonal = scale.get('microtonal', {})
    note_pc = note % 12
    root_pc = scale['intervals'][0]
    
    for i, interval in enumerate(scale['intervals']):
        if (root_pc + interval) % 12 == note_pc:
            if i in microtonal:
                base_offset = microtonal[i]
                if scale.get('directional', False):
                    return base_offset + 15 if ascending else base_offset - 15
                return base_offset
    return 0

def cents_to_pitch_bend(cents: int) -> Tuple[int, int]:
    cents = max(-100, min(100, cents))
    bend_value = int((cents / 100.0) * 4096) + 0x2000
    return (bend_value & 0x7F, (bend_value >> 7) & 0x7F)

def add_mordent(note: int, scale_name: str, duration: int) -> List[MicrotonalNote]:
    if scale_name in ARMENIAN_MAQAM_SCALES:
        intervals = ARMENIAN_MAQAM_SCALES[scale_name]['intervals']
        upper_neighbor = note + (intervals[2] - intervals[0]) if len(intervals) > 2 else note + 2
    else:
        upper_neighbor = note + 2
    
    orn_dur = min(60, duration // 4)
    return [
        MicrotonalNote(note, 80, orn_dur, 0, 'static'),
        MicrotonalNote(upper_neighbor, 75, orn_dur, 0, 'static'),
        MicrotonalNote(note, 85, duration - 2*orn_dur, 0, 'static'),
    ]

def add_trill(note: int, scale_name: str, duration: int) -> List[MicrotonalNote]:
    if scale_name in ARMENIAN_MAQAM_SCALES:
        intervals = ARMENIAN_MAQAM_SCALES[scale_name]['intervals']
        upper = note + (intervals[2] - intervals[0]) if len(intervals) > 2 else note + 2
    else:
        upper = note + 1
    
    orn_dur = min(40, duration // 6)
    notes = []
    rem = duration
    for i in range(6):
        if rem <= 0: break
        n = note if i % 2 == 0 else upper
        dur = min(orn_dur, rem)
        notes.append(MicrotonalNote(n, 70 + (i * 2), dur, 0, 'static'))
        rem -= dur
    if rem > 0: notes[-1].duration += rem
    return notes

def add_grace_note(note: int, prev_note: int, scale_name: str, duration: int) -> List[MicrotonalNote]:
    grace_dur = min(40, duration // 8)
    grace = note - 1
    micro = get_microtonal_offset(grace, scale_name, True) if scale_name in ARMENIAN_MAQAM_SCALES else 0
    return [
        MicrotonalNote(grace, 50, grace_dur, micro, 'ascending'),
        MicrotonalNote(note, 85, duration - grace_dur, 0, 'static'),
    ]

def add_turn(note: int, scale_name: str, duration: int) -> List[MicrotonalNote]:
    if scale_name in ARMENIAN_MAQAM_SCALES:
        intervals = ARMENIAN_MAQAM_SCALES[scale_name]['intervals']
        upper = note + (intervals[2] - intervals[0]) if len(intervals) > 2 else note + 2
        lower = note - 1
    else:
        upper, lower = note + 1, note - 1
    
    orn_dur = min(50, duration // 5)
    return [
        MicrotonalNote(upper, 70, orn_dur, 0, 'static'),
        MicrotonalNote(note, 75, orn_dur, 0, 'static'),
        MicrotonalNote(lower, 65, orn_dur, 0, 'static'),
        MicrotonalNote(note, 80, duration - 3*orn_dur, 0, 'static'),
    ]

def apply_armenian_ornaments(notes: List[int], scale_name: str, rhythms: List[int]) -> List[MicrotonalNote]:
    if scale_name not in ARMENIAN_MAQAM_SCALES:
        return [MicrotonalNote(n, 80, r, 0, 'static') for n, r in zip(notes, rhythms)]
    
    ornaments = ARMENIAN_MAQAM_SCALES[scale_name].get('ornaments', [])
    result = []
    for i, (note, rhythm) in enumerate(zip(notes, rhythms)):
        if rhythm >= 480 and i > 0:
            if 'trill' in ornaments and rhythm >= 720: result.extend(add_trill(note, scale_name, rhythm))
            elif 'mordent' in ornaments: result.extend(add_mordent(note, scale_name, rhythm))
            elif 'turn' in ornaments and rhythm >= 600: result.extend(add_turn(note, scale_name, rhythm))
            else: result.append(MicrotonalNote(note, 80, rhythm, 0, 'static'))
        elif i > 0 and abs(note - notes[i-1]) > 3:
            if 'grace_note' in ornaments: result.extend(add_grace_note(note, notes[i-1], scale_name, rhythm))
            else: result.append(MicrotonalNote(note, 80, rhythm, 0, 'static'))
        else: result.append(MicrotonalNote(note, 80, rhythm, 0, 'static'))
    return result
