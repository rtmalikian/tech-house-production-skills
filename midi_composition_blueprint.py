import random
from typing import Dict, Iterable, List

from midi_models import CompositionBlueprint, SectionIntent


SECTION_ORDER = ["intro", "verse1", "chorus1", "fill1", "verse2", "chorus2", "fill2", "outro"]


MOOD_ARCHETYPES = {
    "introspective": {
        "harmonic_complexity": 0.72,
        "melodic_density": 0.48,
        "chorus_lift": 5,
        "motif": [0, 2, 1, 0, -2, 0],
    },
    "hook_forward": {
        "harmonic_complexity": 0.55,
        "melodic_density": 0.68,
        "chorus_lift": 7,
        "motif": [0, 2, 4, 5, 4, 2, 0],
    },
    "modal_shadow": {
        "harmonic_complexity": 0.82,
        "melodic_density": 0.56,
        "chorus_lift": 4,
        "motif": [0, 1, 4, 3, 1, 0],
    },
    "lofi_pocket": {
        "harmonic_complexity": 0.62,
        "melodic_density": 0.42,
        "chorus_lift": 5,
        "motif": [0, -2, 0, 2, 1, 0],
    },
}


SECTION_SHAPES = {
    "intro": ("setup", 0.36, 0.36, 0, "medium", "fragment", 0.03),
    "verse1": ("statement", 0.58, 0.52, 0, "medium", "identity", 0.08),
    "chorus1": ("lift", 0.88, 0.72, 5, "strong", "sequence_up", 0.02),
    "fill1": ("tension", 0.78, 0.62, 2, "half", "compress", 0.12),
    "verse2": ("variation", 0.66, 0.58, 0, "medium", "invert", 0.06),
    "chorus2": ("peak", 0.98, 0.80, 7, "strong", "sequence_up", 0.01),
    "fill2": ("release_setup", 0.84, 0.66, 2, "half", "retrograde", 0.10),
    "outro": ("resolve", 0.38, 0.34, -2, "final", "thin", 0.18),
}


def create_composition_blueprint(scale_name: str, time_sig: str) -> CompositionBlueprint:
    mood = "modal_shadow" if _is_modal_or_maqam(scale_name) else random.choice(list(MOOD_ARCHETYPES))
    archetype = MOOD_ARCHETYPES[mood]
    density_adjust = -0.08 if time_sig in {"5-4", "5-8"} else 0.0
    section_intents = {}
    tension_arc = []
    for section in SECTION_ORDER:
        role, energy, density, register_shift, cadence, transform, dropout = SECTION_SHAPES[section]
        if section.startswith("chorus"):
            register_shift += archetype["chorus_lift"] - 5
        intent = SectionIntent(
            name=section,
            role=role,
            energy=_clamp(energy + random.uniform(-0.04, 0.04), 0.2, 1.0),
            density=_clamp(density + archetype["melodic_density"] * 0.16 - 0.08 + density_adjust, 0.25, 0.95),
            register_shift=register_shift,
            cadence_strength=cadence,
            motif_transform=transform,
            dropout_chance=_clamp(dropout + random.uniform(-0.02, 0.02), 0.0, 0.3),
        )
        section_intents[section] = intent
        tension_arc.append(intent.energy)
    return CompositionBlueprint(
        mood=mood,
        harmonic_complexity=archetype["harmonic_complexity"],
        melodic_density=_clamp(archetype["melodic_density"] + density_adjust, 0.25, 0.95),
        chorus_lift=archetype["chorus_lift"],
        tension_arc=tension_arc,
        section_intents=section_intents,
        motif_seed=list(archetype["motif"]),
    )


def select_progression_name(names: Iterable[str], section_role: str,
                            blueprint: CompositionBlueprint) -> str:
    candidates = list(names)
    if not candidates:
        raise ValueError("No progression names supplied")
    scored = []
    for name in candidates:
        score = 1.0
        color = _progression_color_score(name)
        cadence = _cadence_score(name)
        if blueprint.harmonic_complexity > 0.7:
            score += color * 1.2
        else:
            score += max(0.0, 1.0 - color) * 0.7
        if section_role in {"lift", "peak", "resolve"}:
            score += cadence * 1.4
        if section_role in {"setup", "statement"} and name.startswith(("I-", "i-", "vi", "Iadd9", "iadd9")):
            score += 0.8
        if section_role in {"tension", "release_setup"} and ("V" in name or "bII" in name):
            score += 1.0
        scored.append((name, max(0.05, score)))
    return random.choices([n for n, _ in scored], weights=[s for _, s in scored], k=1)[0]


def blueprint_to_metadata(blueprint: CompositionBlueprint) -> Dict:
    return {
        "mood": blueprint.mood,
        "harmonic_complexity": round(blueprint.harmonic_complexity, 3),
        "melodic_density": round(blueprint.melodic_density, 3),
        "chorus_lift": blueprint.chorus_lift,
        "tension_arc": [round(v, 3) for v in blueprint.tension_arc],
        "section_intents": {
            name: {
                "role": intent.role,
                "energy": round(intent.energy, 3),
                "density": round(intent.density, 3),
                "register_shift": intent.register_shift,
                "cadence_strength": intent.cadence_strength,
                "motif_transform": intent.motif_transform,
                "dropout_chance": round(intent.dropout_chance, 3),
            }
            for name, intent in blueprint.section_intents.items()
        },
        "motif_seed": list(blueprint.motif_seed),
    }


def _progression_color_score(name: str) -> float:
    markers = ["maj7", "add9", "9", "7", "sus", "bVI", "bVII", "bII", "iii"]
    return min(1.0, sum(1 for marker in markers if marker in name) / 3.0)


def _cadence_score(name: str) -> float:
    if name.endswith(("V-I", "V7", "I", "I6", "Iadd9")):
        return 1.0
    if "V" in name:
        return 0.7
    return 0.25


def _is_modal_or_maqam(scale_name: str) -> bool:
    return not any(mode in scale_name for mode in ("Major", "Minor"))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
