import random
from typing import List, Dict, Optional, Union

ChordToken = Union[int, str]

# ============================================================================
# TECH HOUSE CHORD PROGRESSIONS
# ============================================================================
# Tech house uses minimal harmonic content — root notes, fifths, minor triads
# Chord progressions are simple, repetitive, and hypnotic

# Main groove progressions (2-4 chords, very repetitive)
VERSE_PROGRESSIONS = {
    "i-iv-i-v": [0, 5, 0, 7],
    "i-bVII-i-v": [0, 10, 0, 7],
    "i-bVI-bVII-i": [0, 8, 10, 0],
    "i-iv-bVII-i": [0, 5, 10, 0],
    "i-v-bVI-bVII": [0, 7, 8, 10],
    "i-bIII-bVII-iv": [0, 3, 10, 5],
    "i-i-i-i": [0, 0, 0, 0],  # Static/drone — very common in tech house
    "i-iv-i-iv": [0, 5, 0, 5],
    "i-v-i-v": [0, 7, 0, 7],
    "i-bVII-iv-i": [0, 10, 5, 0],
}

# Drop/peak energy progressions — slightly more movement
CHORUS_PROGRESSIONS = {
    "i-iv-v-i": [0, 5, 7, 0],
    "i-bVII-bVI-v": [0, 10, 8, 7],
    "i-bIII-iv-v": [0, 3, 5, 7],
    "i-iv-bVII-v": [0, 5, 10, 7],
    "i-bVI-bVII-i": [0, 8, 10, 0],
    "i-v-bVI-bVII": [0, 7, 8, 10],
    "i-i-i-i": [0, 0, 0, 0],
    "i-iv-i-v": [0, 5, 0, 7],
}

# Intro/Outro — minimal, DJ-friendly
INTRO_OUTRO_PROGRESSIONS = {
    "i": [0],
    "i-i": [0, 0],
    "i-v": [0, 7],
    "i-iv": [0, 5],
    "i-i-i-i": [0, 0, 0, 0],
}

# Transition/fill progressions
FILL_PROGRESSIONS = {
    "v-i": [7, 0],
    "iv-v": [5, 7],
    "bVII-i": [10, 0],
    "v-bVI": [7, 8],
    "iv-i": [5, 0],
}

PASSING_CHORDS = {
    ('intro', 'drop1'): [0, 7],
    ('drop1', 'breakdown'): [0],
    ('breakdown', 'drop2'): [7, 0],
    ('drop2', 'outro'): [0],
    ('outro', None): [0],
}

LOFI_PROGRESSIONS = {**VERSE_PROGRESSIONS, **CHORUS_PROGRESSIONS}

def get_section_progression(section_type: str) -> Dict[str, List[ChordToken]]:
    if section_type.startswith('drop'): return CHORUS_PROGRESSIONS
    elif section_type in ['intro', 'outro']: return INTRO_OUTRO_PROGRESSIONS
    elif section_type.startswith('build'): return VERSE_PROGRESSIONS
    elif section_type == 'breakdown': return VERSE_PROGRESSIONS
    elif section_type.startswith('fill'): return FILL_PROGRESSIONS
    return VERSE_PROGRESSIONS

def _dominant_of(target_degree: int) -> int:
    return (target_degree + 7) % 12

def _target_symbol(target_degree: int) -> str:
    return {0: 'i', 1: 'bII', 2: 'ii', 3: 'bIII', 4: 'iii', 5: 'iv',
            6: 'bV', 7: 'v', 8: 'bVI', 9: 'vi', 10: 'bVII', 11: 'vii'}.get(target_degree % 12, 'i')

def get_passing_chord(from_section: str, to_section: str, root: int, scale: List[int],
                      target_degree: Optional[int] = None) -> ChordToken:
    if to_section is None:
        return 0
    target = 0 if target_degree is None else target_degree % 12
    context_pool = list(PASSING_CHORDS.get((from_section, to_section), []))
    if not context_pool:
        return 0
    return random.choice(context_pool)

def transform_loop(loop: Dict, mode: str) -> Dict:
    new_loop = {
        'notes': list(loop['notes']),
        'rhythm': list(loop['rhythm']),
        'bars': [
            {
                'notes': list(bar.get('notes', [])),
                'rhythm': list(bar.get('rhythm', [])),
                'role': bar.get('role'),
                'persona': bar.get('persona'),
                'harmony_aware': bar.get('harmony_aware', loop.get('harmony_aware', False)),
            }
            for bar in loop.get('bars', [])
        ],
        'is_chorus': loop.get('is_chorus', False),
        'motif': loop.get('motif'),
        'persona': loop.get('persona'),
        'harmony_aware': loop.get('harmony_aware', False),
    }
    if mode == 'retrograde':
        new_loop['notes'].reverse()
        new_loop['rhythm'].reverse()
        for bar in new_loop['bars']:
            bar['notes'].reverse()
            bar['rhythm'].reverse()
    elif mode == 'inversion':
        center = next((n for n in loop['notes'] if n is not None), None)
        if center is not None:
            new_loop['notes'] = [None if n is None else center - (n - center) for n in loop['notes']]
            for bar in new_loop['bars']:
                bar['notes'] = [None if n is None else center - (n - center) for n in bar['notes']]
    elif mode == 'augmentation':
        new_loop['rhythm'] = [r * 2 for r in loop['rhythm']]
        for bar in new_loop['bars']:
            bar['rhythm'] = [r * 2 for r in bar['rhythm']]
    if new_loop['bars']:
        new_loop['notes'] = [n for bar in new_loop['bars'] for n in bar['notes']]
        new_loop['rhythm'] = [r for bar in new_loop['bars'] for r in bar['rhythm']]
    return new_loop

# ============================================================================
# TECH HOUSE ARRANGEMENT STRUCTURE — DJ-FRIENDLY, COMMERCIAL LENGTH
# ============================================================================
# All sections are 16 bars (DJ phrasing is always in 16-bar blocks)
# Intro(16) + Drop1(32) + Breakdown(32) + Drop2(32) + Outro(16) = 128 bars
# At 128 BPM: 128 bars × 4 beats × 60/128 = 4:00 (commercial minimum)
# Intro: drums + percussion only (let DJ beatmatch)
# Outro: drums + percussion only (let DJ mix out)
# Breakdown: 32 bars for dramatic tension (reference tracks use 16-32 bar breakdowns)

def get_bar_type(bar: int) -> str:
    """DJ-friendly tech house arrangement — commercial length, all 16-bar sections."""
    if bar < 16: return 'intro'        # 16 bars: kick + hats only
    elif bar < 48: return 'drop1'      # 32 bars: full energy, staggered entry
    elif bar < 80: return 'breakdown'  # 32 bars: dramatic tension, filter sweeps
    elif bar < 112: return 'drop2'     # 32 bars: full energy, staggered entry
    return 'outro'                      # 16 bars: drums + percussion only (DJ mix-out)

def get_phrase_position(bar: int) -> int:
    bt = get_bar_type(bar)
    offsets = {
        'intro': 0, 'drop1': 16, 'breakdown': 48, 'drop2': 80, 'outro': 112
    }
    return (bar - offsets.get(bt, 0)) % 8

def get_abac_position(bar: int, bar_type: str) -> str:
    offsets = {
        'intro': 0, 'build1': 16, 'drop1': 24,
        'breakdown': 40, 'build2': 48, 'drop2': 56, 'outro': 72
    }
    section_start = offsets.get(bar_type, 0)
    phrase_bar = (bar - section_start) % 4
    return {0: 'A', 1: 'B', 2: 'A', 3: 'C'}.get(phrase_bar, 'A')
