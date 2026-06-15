import random
from typing import Dict, List, Optional, Tuple

from midi_models import BarHarmony, SectionIntent
from midi_musical_devices import resolve_suspension_voicing, voice_lead_pad_chord
from midi_song_structure import ChordToken
from midi_theory import (
    filter_scale_tones,
    get_modal_pad_chord,
    nearest_pitch_for_pc,
    parse_chord_symbol,
    symbol_for_degree,
)


PAD_PROFILES = (
    "sustained_wash",
    "offbeat_stabs",
    "broken_8th_pulses",
    "sparse_two_note_shells",
    "octave_bloom",
    "call_response_suspensions",
)

PAD_LOW = 72
PAD_HIGH = 96


def derive_later_section_progressions(
    intro_prog: List[ChordToken],
    verse_prog: List[ChordToken],
    chorus_prog: List[ChordToken],
    fill_prog: List[ChordToken],
    scale: List[int],
    base: int,
    enable_reharmonization: bool = True,
) -> Tuple[Dict[str, List[ChordToken]], Dict[str, str]]:
    section_progressions = {
        "intro": list(intro_prog),
        "outro": list(intro_prog),
        "verse1": list(verse_prog),
        "verse2": list(verse_prog),
        "chorus1": list(chorus_prog),
        "chorus2": list(chorus_prog),
        "fill1": list(fill_prog),
        "fill2": list(fill_prog),
    }
    labels = {
        "verse2": "same progression; pad texture variation only",
        "chorus2": "same progression; pad texture variation only",
    }

    if not enable_reharmonization:
        return section_progressions, labels

    verse2 = _rotate_progression(verse_prog, 1)
    chorus2 = _rotate_progression(chorus_prog, 2 if len(chorus_prog) > 2 else 1)

    verse2, verse_label = _recolor_progression(
        verse2,
        scale=scale,
        base=base,
        role="verse2",
        strategy=random.choice(("similar_function", "modal_mixture")),
    )
    chorus2, chorus_label = _recolor_progression(
        chorus2,
        scale=scale,
        base=base,
        role="chorus2",
        strategy=random.choice(("applied_dominant", "suspension_cadence", "similar_function")),
    )

    section_progressions["verse2"] = verse2
    section_progressions["chorus2"] = chorus2
    labels["verse2"] = f"rotated + {verse_label}"
    labels["chorus2"] = f"rotated + {chorus_label}"
    return section_progressions, labels


def select_pad_profile(
    section: str,
    intent: SectionIntent,
    is_armenian: bool,
    previous_profile: Optional[str] = None,
    repeat_count: int = 0,
) -> str:
    role_weights = _profile_weights(section, intent.energy, is_armenian)
    if previous_profile and repeat_count >= 2:
        role_weights = {
            name: weight for name, weight in role_weights.items() if name != previous_profile
        }
    profiles = list(role_weights)
    weights = [role_weights[name] for name in profiles]
    return random.choices(profiles, weights=weights, k=1)[0]


def render_pad_bar_events(
    harmony: BarHarmony,
    next_harmony: Optional[BarHarmony],
    previous_voicing: Optional[List[int]],
    profile: str,
    bar_start: int,
    bar_length: int,
    energy: float,
    intent: SectionIntent,
    scale_notes: List[int],
    qual: str,
    is_armenian: bool = False,
    armenian_scale_name: Optional[str] = None,
    cadence: bool = False,
) -> Tuple[List[Dict], Optional[List[int]]]:
    if profile not in PAD_PROFILES:
        profile = "sustained_wash"

    raw_chord = _base_pad_chord(
        harmony,
        scale_notes=scale_notes,
        qual=qual,
        energy=energy,
        is_armenian=is_armenian,
        armenian_scale_name=armenian_scale_name,
        add_color=profile in {"sustained_wash", "octave_bloom", "call_response_suspensions"},
    )
    previous = previous_voicing or None
    events: List[Dict] = []

    if profile == "sustained_wash":
        chord = voice_lead_pad_chord(_fit_pad_notes(raw_chord), previous, max_notes=5)
        velocity = _velocity(54, 70, energy, is_armenian)
        duration = _clip_duration(bar_length + (120 if cadence else random.randint(-90, 90)), 360, bar_length + 240)
        _add_chord(events, bar_start, chord, velocity, duration)
        return events, chord

    if profile == "offbeat_stabs":
        chord = voice_lead_pad_chord(_fit_pad_notes(raw_chord), previous, max_notes=3)
        velocity = _velocity(50, 66, energy, is_armenian)
        for offset in _choose_offsets(_offbeat_offsets(bar_length), 2 if energy > 0.65 else 1):
            _add_chord(events, bar_start + offset, chord, velocity + random.randint(-4, 4), random.randint(180, 360))
        return events, chord

    if profile == "broken_8th_pulses":
        chord = voice_lead_pad_chord(_fit_pad_notes(raw_chord), previous, max_notes=4)
        shell = _shell_voicing(harmony, previous, max_notes=2)
        pool = chord if len(chord) >= 3 else shell
        velocity = _velocity(43, 58, energy, is_armenian)
        pulse_offsets = [offset for offset in range(0, max(0, bar_length - 180), 240)]
        for idx, offset in enumerate(pulse_offsets):
            if random.random() < (0.42 if intent.role in {"statement", "variation"} else 0.28):
                continue
            notes = _rotating_dyad(pool, idx)
            _add_chord(events, bar_start + offset, notes, velocity + random.randint(-5, 5), random.randint(135, 210))
        return events, pool

    if profile == "sparse_two_note_shells":
        shell = _shell_voicing(harmony, previous, max_notes=2)
        velocity = _velocity(47, 61, energy, is_armenian)
        offsets = [0]
        if energy > 0.56:
            offsets.append(max(240, min(bar_length - 360, bar_length // 2)))
        if next_harmony and random.random() < 0.35:
            offsets.append(max(0, bar_length - 240))
        for offset in sorted(set(offsets)):
            _add_chord(events, bar_start + offset, shell, velocity + random.randint(-4, 4), min(720, max(240, bar_length - offset)))
        return events, shell

    if profile == "octave_bloom":
        chord = voice_lead_pad_chord(_fit_pad_notes(raw_chord), previous, max_notes=4)
        shell = chord[: max(2, min(3, len(chord)))]
        bloom = _bloom_notes(chord, harmony)
        velocity = _velocity(48, 64, energy, is_armenian)
        bloom_offset = max(360, min(bar_length - 240, bar_length // 2))
        _add_chord(events, bar_start, shell, velocity, bar_length)
        if bloom:
            _add_chord(events, bar_start + bloom_offset, bloom, max(1, velocity - 8), bar_length - bloom_offset)
        return events, sorted(set(shell + bloom))

    suspended = _suspended_chord(raw_chord, harmony, qual)
    sus_voicing = voice_lead_pad_chord(_fit_pad_notes(suspended), previous, max_notes=4)
    resolved = resolve_suspension_voicing(sus_voicing, _fit_pad_notes(raw_chord))
    velocity = _velocity(49, 64, energy, is_armenian)
    response_offset = 480 if bar_length >= 1440 else max(240, bar_length // 2)
    response_offset = min(response_offset, max(120, bar_length - 240))
    _add_chord(events, bar_start, sus_voicing, velocity, response_offset)
    _add_chord(events, bar_start + response_offset, resolved, max(1, velocity - 3), bar_length - response_offset)
    return events, resolved


def _profile_weights(section: str, energy: float, is_armenian: bool) -> Dict[str, float]:
    if section == "intro":
        weights = {
            "sustained_wash": 4.0,
            "octave_bloom": 1.8,
            "sparse_two_note_shells": 1.0,
            "call_response_suspensions": 0.6,
        }
    elif section == "outro":
        weights = {
            "sustained_wash": 3.2,
            "sparse_two_note_shells": 2.0,
            "octave_bloom": 1.0,
            "call_response_suspensions": 0.8,
        }
    elif section.startswith("chorus"):
        weights = {
            "broken_8th_pulses": 2.8,
            "octave_bloom": 2.4,
            "call_response_suspensions": 2.0,
            "offbeat_stabs": 1.2,
            "sustained_wash": 0.8,
        }
    elif section == "verse2":
        weights = {
            "offbeat_stabs": 2.3,
            "sparse_two_note_shells": 2.0,
            "broken_8th_pulses": 1.7,
            "call_response_suspensions": 1.3,
            "octave_bloom": 0.9,
        }
    else:
        weights = {
            "sparse_two_note_shells": 2.6,
            "offbeat_stabs": 2.1,
            "sustained_wash": 1.3,
            "broken_8th_pulses": 1.0,
            "call_response_suspensions": 0.9,
        }

    if energy > 0.82:
        weights["octave_bloom"] = weights.get("octave_bloom", 0.5) + 1.0
        weights["broken_8th_pulses"] = weights.get("broken_8th_pulses", 0.5) + 0.9
    if is_armenian:
        weights["sustained_wash"] = weights.get("sustained_wash", 0.0) + 1.0
        weights["call_response_suspensions"] = max(0.3, weights.get("call_response_suspensions", 0.5) * 0.65)
    return weights


def _rotate_progression(progression: List[ChordToken], amount: int) -> List[ChordToken]:
    if not progression:
        return []
    amount %= len(progression)
    return list(progression[amount:] + progression[:amount])


def _recolor_progression(
    progression: List[ChordToken],
    scale: List[int],
    base: int,
    role: str,
    strategy: str,
) -> Tuple[List[ChordToken], str]:
    if not progression:
        return [], "empty"
    out = list(progression)
    index = 1 if len(out) > 2 else 0
    if role == "chorus2" and strategy in {"applied_dominant", "suspension_cadence"}:
        index = max(0, len(out) - 2)
    original = out[index]
    out[index] = _replacement_chord(original, scale, base, strategy)
    return out, f"{strategy} at slot {index + 1}"


def _replacement_chord(chord: ChordToken, scale: List[int], base: int, strategy: str) -> ChordToken:
    spec = parse_chord_symbol(chord, scale, base)
    degree = spec.root_offset % 12
    if strategy == "modal_mixture":
        return random.choice(["bVImaj7", "bVII7", "bIIImaj7", "iv7"])
    if strategy == "applied_dominant":
        target = symbol_for_degree(degree if degree not in (0, 7) else 0)
        return "Vsus4" if target == "I" and random.random() < 0.45 else f"V7/{target}"
    if strategy == "suspension_cadence":
        return "Vsus4" if degree in (0, 7) else f"{symbol_for_degree(degree)}sus4"

    if degree in (0, 4, 9):
        return random.choice(["Iadd9", "vi9", "iii7", "Imaj7"])
    if degree in (2, 5):
        return random.choice(["ii7", "IVmaj7", "IVadd9"])
    if degree in (7, 10, 11):
        return random.choice(["V7", "Vsus4", "bVII7"])
    return random.choice(["Iadd9", "IVmaj7", "vi9"])


def _base_pad_chord(
    harmony: BarHarmony,
    scale_notes: List[int],
    qual: str,
    energy: float,
    is_armenian: bool,
    armenian_scale_name: Optional[str],
    add_color: bool,
) -> List[int]:
    if is_armenian:
        raw = get_modal_pad_chord(harmony.root, qual, scale_notes, armenian_scale_name or "", energy)
        raw = filter_scale_tones(raw, scale_notes) or raw
    else:
        raw = list(harmony.chord_tones)
    has_color = bool(harmony.spec.extensions or harmony.spec.suspension or harmony.spec.quality in {"maj7", "min7", "min9", "dom7"})
    if add_color and not has_color and random.random() < (0.22 if is_armenian else 0.45):
        color_pool = [harmony.root + 14, harmony.root + 17, harmony.root + 21]
        if not is_armenian:
            color_pool.append(harmony.root + 9)
        extension = random.choice(color_pool)
        if extension not in raw and (not is_armenian or extension % 12 in {n % 12 for n in scale_notes}):
            raw.append(extension)
    return sorted(set(raw))


def _fit_pad_notes(notes: List[int]) -> List[int]:
    fitted = []
    for note in notes:
        fitted.append(nearest_pitch_for_pc(84, note % 12, PAD_LOW, PAD_HIGH))
    return sorted(set(fitted))


def _shell_voicing(harmony: BarHarmony, previous: Optional[List[int]], max_notes: int = 2) -> List[int]:
    root_pc = harmony.root % 12
    by_interval = {}
    for note in harmony.chord_tones:
        by_interval[(note - harmony.root) % 12] = note
    preferred = []
    for interval in (3, 4, 5, 10, 11, 9, 14, 2, 7):
        note = by_interval.get(interval % 12)
        if note is not None and note % 12 not in {n % 12 for n in preferred}:
            preferred.append(note)
        if len(preferred) >= max_notes:
            break
    if len(preferred) < max_notes:
        for note in harmony.chord_tones:
            if note % 12 != root_pc and note % 12 not in {n % 12 for n in preferred}:
                preferred.append(note)
            if len(preferred) >= max_notes:
                break
    if len(preferred) < max_notes:
        preferred.append(harmony.root + 7)
    return voice_lead_pad_chord(_fit_pad_notes(preferred), previous, max_notes=max_notes)


def _suspended_chord(raw_chord: List[int], harmony: BarHarmony, qual: str) -> List[int]:
    third = harmony.root + (4 if qual == "major" else 3)
    sus_interval = 5 if random.random() < 0.72 else 2
    sus = harmony.root + sus_interval
    suspended = [note for note in raw_chord if note % 12 != third % 12]
    if sus % 12 not in {n % 12 for n in suspended}:
        suspended.append(sus)
    return sorted(suspended)


def _velocity(low: int, high: int, energy: float, is_armenian: bool) -> int:
    value = random.randint(low, high)
    value = int(value * (0.82 if is_armenian else 1.0) * (0.78 + energy * 0.28))
    return max(1, min(96, value))


def _add_chord(events: List[Dict], start: int, notes: List[int], velocity: int, duration: int) -> None:
    duration = max(60, int(duration))
    clean_notes = sorted(set(n for n in notes if PAD_LOW <= n <= PAD_HIGH))
    for note in clean_notes:
        events.append({"time": int(start), "note": int(note), "vel": int(max(1, min(127, velocity)))})
    for note in clean_notes:
        events.append({"time": int(start + duration), "note": int(note), "vel": 0})


def _clip_duration(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def _offbeat_offsets(bar_length: int) -> List[int]:
    return [offset for offset in (240, 720, 1200, 1680, 2160) if offset < bar_length - 180]


def _choose_offsets(offsets: List[int], target_count: int) -> List[int]:
    if not offsets:
        return [0]
    count = max(1, min(len(offsets), target_count + random.choice([0, 1])))
    return sorted(random.sample(offsets, count))


def _rotating_dyad(pool: List[int], index: int) -> List[int]:
    if not pool:
        return []
    if len(pool) == 1:
        return pool
    first = pool[index % len(pool)]
    second = pool[(index + 1 + (index % 2)) % len(pool)]
    return sorted({first, second})


def _bloom_notes(chord: List[int], harmony: BarHarmony) -> List[int]:
    candidates = []
    for note in chord:
        upper = note + 12
        if PAD_LOW <= upper <= PAD_HIGH:
            candidates.append(upper)
    for interval in (14, 17, 21, 9):
        candidates.append(nearest_pitch_for_pc(91, (harmony.root + interval) % 12, PAD_LOW, PAD_HIGH))
    random.shuffle(candidates)
    return sorted(set(candidates[: random.choice([1, 2])]))
