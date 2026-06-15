import random
from typing import List, Dict, Optional

# --- GM Drum Map (909-style mapping) ---
GM_DRUM_MAP = {
    'KICK': 36, 'SNARE': 38, 'RIMSHOT': 37, 'CLOSED_HAT': 42,
    'OPEN_HAT': 46, 'LOW_TOM': 45, 'MID_TOM': 47, 'HIGH_TOM': 50,
    'CRASH_CYMBAL': 49, 'RIDE_CYMBAL': 51,
    'COWBELL': 56, 'CLAVES': 75, 'TAMBOURINE': 54, 'MARACAS': 70,
    'SIDE_STICK': 37, 'CLAP': 39, 'PEDAL_HAT': 44,
    # 909-specific percussion
    'SHAKER': 70, 'RIM': 37, 'CRASH': 49,
}

# Tech house pattern families
PATTERN_FAMILIES = [
    'four_on_floor', 'tech_house_groove', 'minimal_tech',
    'driving_tech', 'acid_tech', 'percussive_tech',
    'deep_tech', 'peak_time_tech',
]

# ============================================================================
# HELPERS
# ============================================================================

def _get_drum_positions(time_sig='4-4', tpb=480):
    """Tech house: always 4/4. Kick on every beat, clap on 2 and 4."""
    return [1 * tpb, 3 * tpb], [0, 1 * tpb, 2 * tpb, 3 * tpb], 16

def invert_kick_snare(notes):
    """Swap kick (36) and snare (38) positions."""
    return [
        {'note': 38, 'velocity': n['velocity'], 'time': n['time']} if n['note'] == 36
        else {'note': 36, 'velocity': n['velocity'], 'time': n['time']} if n['note'] == 38
        else n for n in notes
    ]

def apply_micro_techniques(notes, tpb):
    """Tech house micro-techniques: subtle groove enhancements."""
    sixteenth = tpb // 4
    result = list(notes)

    # Ghost kick anticipation (15% - subtle)
    if random.random() < 0.15:
        for n in list(result):
            if n['note'] == 36 and n['velocity'] > 80 and n['time'] >= sixteenth:
                result.append({'note': 36, 'velocity': random.randint(25, 40),
                               'time': n['time'] - sixteenth})

    # Open hat choke (20%)
    if random.random() < 0.20:
        for n in list(result):
            if n['note'] == 46:
                result.append({'note': 42, 'velocity': 70, 'time': n['time'] + 3})

    # Velocity crescendo on hats over 8-bar phrase (15%)
    if random.random() < 0.15:
        hats = sorted([n for n in result if n['note'] in (42, 44)], key=lambda x: x['time'])
        for i, h in enumerate(hats):
            h['velocity'] = max(30, min(127, int(50 + (i / max(len(hats), 1)) * 60)))

    # Crash on beat 1 (10%)
    if random.random() < 0.10:
        result.append({'note': 49, 'velocity': random.randint(35, 55), 'time': 0})

    # Ride accent (8%)
    if random.random() < 0.08:
        for beat in range(4):
            result.append({'note': 51, 'velocity': random.randint(40, 60),
                           'time': beat * tpb + sixteenth})

    return result


def _finalize(notes, tpb, inverted=False, time_sig='4-4', quantize_drums=True):
    """Apply micro-techniques to hats/perc only. Kick and clap stay on grid."""
    if quantize_drums:
        # Split into quantized (kick, clap, snare) and groovable (hats, perc)
        quantized = [n for n in notes if n['note'] in (36, 38, 39, 49)]  # kick, snare, clap, crash
        groovable = [n for n in notes if n['note'] not in (36, 38, 39, 49)]
        groovable = apply_micro_techniques(groovable, tpb)
        notes = quantized + groovable
    else:
        notes = apply_micro_techniques(notes, tpb)
    if inverted:
        notes = invert_kick_snare(notes)
    seen = set()
    deduped = []
    for n in notes:
        key = (n['note'], n['time'])
        if key not in seen:
            seen.add(key)
            deduped.append(n)
    return deduped


# ============================================================================
# 1. FOUR ON THE FLOOR (Foundation)
# ============================================================================

def create_four_on_floor_bar(tpb, base_pattern_id=0, variation_level=0,
                              is_chorus=False, inverted=False, time_sig='4-4'):
    """
    Classic 4-on-the-floor kick with 909-style clap on 2&4.
    16th note hi-hats with velocity variation for groove.
    """
    notes = []
    sixteenth = tpb // 4
    bar_len = tpb * 4

    # === KICK: 4-on-the-floor, NO ghost kicks ===
    for beat in range(4):
        notes.append({'note': GM_DRUM_MAP['KICK'], 'velocity': random.randint(122, 127),
                       'time': beat * tpb})

    # === CLAP: Beats 2 and 4 ===
    for st in [1 * tpb, 3 * tpb]:
        clap_vel = 120 if is_chorus else 110
        notes.append({'note': GM_DRUM_MAP['CLAP'], 'velocity': clap_vel, 'time': st})
        # Subtle snare layer on chorus
        if is_chorus:
            notes.append({'note': GM_DRUM_MAP['SNARE'], 'velocity': int(clap_vel * 0.6),
                           'time': st + 2})

    # === HI-HATS: 16th notes, tight velocity (quantized, no humanization) ===
    for i in range(16):
        pos = i * sixteenth
        # Consistent velocity — groove from pattern, not velocity variation
        if i % 4 == 0:
            vel = 120  # Downbeats
        elif i % 4 == 2:
            vel = 110  # Upbeats
        else:
            vel = 100  # Ghost 16ths — audible but consistent
        notes.append({'note': GM_DRUM_MAP['CLOSED_HAT'], 'velocity': vel, 'time': pos})

    # Open hat on offbeat of beat 2 (classic tech house)
    if base_pattern_id in [0, 2]:
        notes.append({'note': GM_DRUM_MAP['OPEN_HAT'], 'velocity': 110,
                       'time': 1 * tpb + 2 * sixteenth})

    # Open hat on offbeat of beat 4 (chorus variation)
    if is_chorus and random.random() < 0.5:
        notes.append({'note': GM_DRUM_MAP['OPEN_HAT'], 'velocity': 105,
                       'time': 3 * tpb + 2 * sixteenth})

    return _finalize(notes, tpb, inverted, time_sig)


# ============================================================================
# 2. TECH HOUSE GROOVE
# ============================================================================

def create_tech_house_groove_bar(tpb, base_pattern_id=0, variation_level=0,
                                  is_chorus=False, inverted=False, time_sig='4-4'):
    """
    Syncopated kick pattern with offbeat bass stabs.
    Shaker/tambourine layer for rhythmic drive.
    """
    notes = []
    sixteenth = tpb // 4
    bar_len = tpb * 4

    # === KICK: 4-on-the-floor, NO syncopated kicks ===
    notes.append({'note': GM_DRUM_MAP['KICK'], 'velocity': random.randint(122, 127), 'time': 0})
    notes.append({'note': GM_DRUM_MAP['KICK'], 'velocity': random.randint(122, 127), 'time': 1 * tpb})
    notes.append({'note': GM_DRUM_MAP['KICK'], 'velocity': random.randint(122, 127), 'time': 2 * tpb})
    notes.append({'note': GM_DRUM_MAP['KICK'], 'velocity': random.randint(122, 127), 'time': 3 * tpb})

    # === CLAP: Beats 2 and 4 ===
    for st in [1 * tpb, 3 * tpb]:
        notes.append({'note': GM_DRUM_MAP['CLAP'], 'velocity': 115 if is_chorus else 105,
                       'time': st})

    # === HI-HATS: Offbeat pattern ===
    for i in range(8):
        pos = i * 2 * sixteenth + sixteenth  # Offbeat 8ths
        vel = random.randint(80, 100)
        if i % 2 == 0:
            notes.append({'note': GM_DRUM_MAP['CLOSED_HAT'], 'velocity': vel, 'time': pos})
        else:
            notes.append({'note': GM_DRUM_MAP['CLOSED_HAT'],
                           'velocity': random.randint(60, 75), 'time': pos})

    # === TAMBOURINE/SHAKER: Steady 16ths ===
    for i in range(16):
        pos = i * sixteenth
        vel = random.randint(40, 65) if i % 2 else random.randint(50, 75)
        notes.append({'note': GM_DRUM_MAP['TAMBOURINE'], 'velocity': vel, 'time': pos})

    # Chorus: add ride cymbal
    if is_chorus:
        for beat in range(4):
            notes.append({'note': GM_DRUM_MAP['RIDE_CYMBAL'], 'velocity': random.randint(60, 80),
                           'time': beat * tpb})

    return _finalize(notes, tpb, inverted, time_sig)


# ============================================================================
# 3. MINIMAL TECH
# ============================================================================

def create_minimal_tech_bar(tpb, base_pattern_id=0, variation_level=0,
                             is_chorus=False, inverted=False, time_sig='4-4'):
    """
    Stripped-back minimal tech house. Sparse hats, no clap on some bars.
    More space, more groove through subtraction.
    """
    notes = []
    sixteenth = tpb // 4
    bar_len = tpb * 4

    # === KICK: 4-on-floor, consistent ===
    for beat in range(4):
        notes.append({'note': GM_DRUM_MAP['KICK'], 'velocity': random.randint(122, 127),
                       'time': beat * tpb})

    # === CLAP: Only on beat 4 (minimal approach) or 2&4 ===
    if base_pattern_id in [0, 1]:
        notes.append({'note': GM_DRUM_MAP['CLAP'], 'velocity': 110, 'time': 3 * tpb})
    else:
        for st in [1 * tpb, 3 * tpb]:
            notes.append({'note': GM_DRUM_MAP['CLAP'], 'velocity': 110, 'time': st})

    # === HI-HATS: Sparse, offbeat only ===
    hat_positions = [sixteenth, 3 * sixteenth, 5 * sixteenth, 7 * sixteenth,
                     9 * sixteenth, 11 * sixteenth, 13 * sixteenth, 15 * sixteenth]
    for pos in hat_positions:
        if random.random() < 0.6:  # 60% density — sparse
            vel = random.randint(65, 90)
            notes.append({'note': GM_DRUM_MAP['CLOSED_HAT'], 'velocity': vel, 'time': pos})

    # === RIMSHOT: Syncopated accent ===
    if variation_level > 0:
        rim_positions = [sixteenth * 3, sixteenth * 7, sixteenth * 11, sixteenth * 15]
        for pos in rim_positions:
            if random.random() < 0.25:
                notes.append({'note': GM_DRUM_MAP['RIMSHOT'], 'velocity': random.randint(45, 65),
                               'time': pos})

    return _finalize(notes, tpb, inverted, time_sig)


# ============================================================================
# 4. DRIVING TECH
# ============================================================================

def create_driving_tech_bar(tpb, base_pattern_id=0, variation_level=0,
                             is_chorus=False, inverted=False, time_sig='4-4'):
    """
    Driving, energetic tech house. More percussion layers,
    ride cymbal, and snare fills for peak-time energy.
    """
    notes = []
    sixteenth = tpb // 4
    bar_len = tpb * 4

    # === KICK: 4-on-floor, consistent velocity ===
    for beat in range(4):
        notes.append({'note': GM_DRUM_MAP['KICK'], 'velocity': random.randint(122, 127),
                       'time': beat * tpb})

    # === CLAP + SNARE: Layered on 2&4 ===
    for st in [1 * tpb, 3 * tpb]:
        notes.append({'note': GM_DRUM_MAP['CLAP'], 'velocity': 120, 'time': st})
        notes.append({'note': GM_DRUM_MAP['SNARE'], 'velocity': 90, 'time': st})

    # === CLOSED HATS: Driving 16ths ===
    for i in range(16):
        pos = i * sixteenth
        vel = random.randint(90, 110) if i % 4 == 0 else random.randint(65, 85)
        notes.append({'note': GM_DRUM_MAP['CLOSED_HAT'], 'velocity': vel, 'time': pos})

    # === OPEN HAT: Accent pattern ===
    notes.append({'note': GM_DRUM_MAP['OPEN_HAT'], 'velocity': 95,
                   'time': 1 * tpb + 2 * sixteenth})
    if is_chorus:
        notes.append({'note': GM_DRUM_MAP['OPEN_HAT'], 'velocity': 90,
                       'time': 3 * tpb + 2 * sixteenth})

    # === RIDE: Steady 8ths for drive ===
    if is_chorus or variation_level > 1:
        for i in range(8):
            pos = i * 2 * sixteenth
            vel = random.randint(60, 80)
            notes.append({'note': GM_DRUM_MAP['RIDE_CYMBAL'], 'velocity': vel, 'time': pos})

    # === CRASH: On first beat of phrases ===
    if variation_level > 0 and random.random() < 0.25:
        notes.append({'note': GM_DRUM_MAP['CRASH_CYMBAL'], 'velocity': random.randint(80, 100),
                       'time': 0})

    return _finalize(notes, tpb, inverted, time_sig)


# ============================================================================
# 5. ACID TECH
# ============================================================================

def create_acid_tech_bar(tpb, base_pattern_id=0, variation_level=0,
                          is_chorus=False, inverted=False, time_sig='4-4'):
    """
    Acid-flavored tech house. 16th-note hi-hat patterns,
    heavier kick, more aggressive clap layering.
    """
    notes = []
    sixteenth = tpb // 4
    bar_len = tpb * 4

    # === KICK: 4-on-floor, consistent ===
    for beat in range(4):
        notes.append({'note': GM_DRUM_MAP['KICK'], 'velocity': random.randint(122, 127),
                       'time': beat * tpb})
    # === CLAP: Beats 2&4 with snare flam ===
    for st in [1 * tpb, 3 * tpb]:
        notes.append({'note': GM_DRUM_MAP['CLAP'], 'velocity': 120, 'time': st})
        notes.append({'note': GM_DRUM_MAP['SNARE'], 'velocity': 80, 'time': st + 3})

    # === HATS: Aggressive 16ths with accent pattern ===
    accent_pattern = [1, 0, 0.7, 0.5, 1, 0, 0.7, 0.5, 1, 0, 0.8, 0.6, 1, 0, 0.7, 0.4]
    for i in range(16):
        pos = i * sixteenth
        vel = int(60 + accent_pattern[i] * 55)
        vel += random.randint(-5, 5)
        vel = max(30, min(127, vel))
        notes.append({'note': GM_DRUM_MAP['CLOSED_HAT'], 'velocity': vel, 'time': pos})

    # === OPEN HAT: 303-style accent positions ===
    open_positions = [2 * sixteenth, 6 * sixteenth, 10 * sixteenth, 14 * sixteenth]
    for pos in open_positions:
        if random.random() < 0.35:
            notes.append({'note': GM_DRUM_MAP['OPEN_HAT'], 'velocity': random.randint(70, 90),
                           'time': pos})

    # === RIMSHOT: Offbeat accents ===
    if variation_level > 0:
        for i in [1, 5, 9, 13]:
            if random.random() < 0.3:
                notes.append({'note': GM_DRUM_MAP['RIMSHOT'], 'velocity': random.randint(50, 70),
                               'time': i * sixteenth})

    return _finalize(notes, tpb, inverted, time_sig)


# ============================================================================
# 6. PERCUSSIVE TECH
# ============================================================================

def create_percussive_tech_bar(tpb, base_pattern_id=0, variation_level=0,
                                is_chorus=False, inverted=False, time_sig='4-4'):
    """
    Percussion-heavy tech house with conga/bongo-style hits,
    shakers, and complex rhythmic layering.
    """
    notes = []
    sixteenth = tpb // 4
    bar_len = tpb * 4

    # === KICK: 4-on-floor, consistent ===
    for beat in range(4):
        notes.append({'note': GM_DRUM_MAP['KICK'], 'velocity': random.randint(122, 127),
                       'time': beat * tpb})

    # === CLAP: Beat 2&4 ===
    for st in [1 * tpb, 3 * tpb]:
        notes.append({'note': GM_DRUM_MAP['CLAP'], 'velocity': 110, 'time': st})

    # === HATS: Straight 16ths, moderate velocity ===
    for i in range(16):
        pos = i * sixteenth
        vel = random.randint(70, 90)
        notes.append({'note': GM_DRUM_MAP['CLOSED_HAT'], 'velocity': vel, 'time': pos})

    # === TAMBOURINE: Offbeat 8ths ===
    for i in range(8):
        pos = i * 2 * sixteenth + sixteenth
        notes.append({'note': GM_DRUM_MAP['TAMBOURINE'], 'velocity': random.randint(55, 75),
                       'time': pos})

    # === RIMSHOT: Syncopated pattern ===
    rim_pattern = [sixteenth * 3, sixteenth * 7, sixteenth * 10, sixteenth * 14]
    for pos in rim_pattern:
        if random.random() < 0.6:
            notes.append({'note': GM_DRUM_MAP['RIMSHOT'], 'velocity': random.randint(50, 70),
                           'time': pos})

    # === COWBELL: Sparse accents (optional) ===
    if base_pattern_id == 1:
        cowbell_positions = [0, 2 * sixteenth, 4 * tpb // 4, 6 * sixteenth]
        for pos in cowbell_positions:
            if random.random() < 0.4:
                notes.append({'note': GM_DRUM_MAP['COWBELL'], 'velocity': random.randint(50, 70),
                               'time': pos})

    return _finalize(notes, tpb, inverted, time_sig)


# ============================================================================
# 7. DEEP TECH
# ============================================================================

def create_deep_tech_bar(tpb, base_pattern_id=0, variation_level=0,
                          is_chorus=False, inverted=False, time_sig='4-4'):
    """
    Deep, hypnotic tech house. Very minimal percussion,
    emphasis on kick and bass interplay.
    """
    notes = []
    sixteenth = tpb // 4
    bar_len = tpb * 4

    # === KICK: Subtle variations on 4-on-floor ===
    notes.append({'note': GM_DRUM_MAP['KICK'], 'velocity': 127, 'time': 0})
    notes.append({'note': GM_DRUM_MAP['KICK'], 'velocity': 122, 'time': tpb})
    notes.append({'note': GM_DRUM_MAP['KICK'], 'velocity': 125, 'time': 2 * tpb})
    notes.append({'note': GM_DRUM_MAP['KICK'], 'velocity': 120, 'time': 3 * tpb})

    # === CLAP: Beat 4 only (very minimal) ===
    notes.append({'note': GM_DRUM_MAP['CLAP'], 'velocity': 100, 'time': 3 * tpb})

    # === HATS: Very sparse, every other 8th note ===
    for i in range(4):
        pos = i * 4 * sixteenth + 2 * sixteenth  # "and" of each beat
        notes.append({'note': GM_DRUM_MAP['CLOSED_HAT'], 'velocity': random.randint(60, 80),
                       'time': pos})

    # === PEDAL HAT: Subtle pulse ===
    if variation_level > 0:
        for beat in range(4):
            notes.append({'note': GM_DRUM_MAP['PEDAL_HAT'], 'velocity': random.randint(40, 55),
                           'time': beat * tpb + 2 * sixteenth})

    return _finalize(notes, tpb, inverted, time_sig)


# ============================================================================
# 8. PEAK TIME TECH
# ============================================================================

def create_peak_time_tech_bar(tpb, base_pattern_id=0, variation_level=0,
                               is_chorus=False, inverted=False, time_sig='4-4'):
    """
    Maximum energy tech house for peak moments.
    All layers active, crash cymbals, snare rolls.
    """
    notes = []
    sixteenth = tpb // 4
    bar_len = tpb * 4

    # === KICK: Heavy 4-on-floor, consistent ===
    for beat in range(4):
        notes.append({'note': GM_DRUM_MAP['KICK'], 'velocity': random.randint(122, 127),
                       'time': beat * tpb})

    # === CLAP + SNARE: Layered ===
    for st in [1 * tpb, 3 * tpb]:
        notes.append({'note': GM_DRUM_MAP['CLAP'], 'velocity': 125, 'time': st})
        notes.append({'note': GM_DRUM_MAP['SNARE'], 'velocity': 100, 'time': st + 2})

    # === HATS: Fast 16ths ===
    for i in range(16):
        pos = i * sixteenth
        vel = random.randint(85, 110) if i % 2 == 0 else random.randint(60, 80)
        notes.append({'note': GM_DRUM_MAP['CLOSED_HAT'], 'velocity': vel, 'time': pos})

    # === OPEN HAT: Prominent ===
    notes.append({'note': GM_DRUM_MAP['OPEN_HAT'], 'velocity': 100,
                   'time': 1 * tpb + 2 * sixteenth})
    notes.append({'note': GM_DRUM_MAP['OPEN_HAT'], 'velocity': 95,
                   'time': 3 * tpb + 2 * sixteenth})

    # === RIDE: Driving ===
    for i in range(8):
        notes.append({'note': GM_DRUM_MAP['RIDE_CYMBAL'], 'velocity': random.randint(70, 90),
                       'time': i * 2 * sixteenth})

    # === CRASH: On beat 1 ===
    notes.append({'note': GM_DRUM_MAP['CRASH_CYMBAL'], 'velocity': 100, 'time': 0})

    # === TOM FILLS: Bar-end variations ===
    if variation_level > 0 and random.random() < 0.3:
        fill_start = bar_len - 4 * sixteenth
        for i in range(4):
            tom = random.choice([GM_DRUM_MAP['LOW_TOM'], GM_DRUM_MAP['MID_TOM'],
                                  GM_DRUM_MAP['HIGH_TOM']])
            notes.append({'note': tom, 'velocity': random.randint(70, 90),
                           'time': fill_start + i * sixteenth})

    return _finalize(notes, tpb, inverted, time_sig)


# ============================================================================
# PATTERN FAMILY MAP
# ============================================================================

PATTERN_FAMILY_MAP = {
    'four_on_floor': create_four_on_floor_bar,
    'tech_house_groove': create_tech_house_groove_bar,
    'minimal_tech': create_minimal_tech_bar,
    'driving_tech': create_driving_tech_bar,
    'acid_tech': create_acid_tech_bar,
    'percussive_tech': create_percussive_tech_bar,
    'deep_tech': create_deep_tech_bar,
    'peak_time_tech': create_peak_time_tech_bar,
}


def get_pattern_funcs():
    """
    Return pattern functions for a tech house track.
    Returns (pattern_A, pattern_B, pattern_C, pattern_D, main_family, chorus_family)
    """
    main_family = random.choice(['four_on_floor', 'tech_house_groove', 'minimal_tech',
                                  'deep_tech', 'percussive_tech'])
    chorus_family = random.choice(['driving_tech', 'peak_time_tech', 'acid_tech',
                                    'four_on_floor'])
    # Ensure main and chorus are different
    while chorus_family == main_family:
        chorus_family = random.choice(['driving_tech', 'peak_time_tech', 'acid_tech'])

    main_func = PATTERN_FAMILY_MAP[main_family]
    chorus_func = PATTERN_FAMILY_MAP[chorus_family]

    return main_func, main_func, chorus_func, chorus_func, main_family, chorus_family
