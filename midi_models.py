from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional

class IntervalQuality(Enum):
    PERFECT_CONSONANCE = 1.0
    IMPERFECT_CONSONANCE = 0.7
    MILD_TENSION = 0.4
    STRONG_TENSION = 0.2
    DISSONANCE = 0.1

@dataclass
class VoiceLeadingContext:
    bass_last_note: Optional[int] = None
    melody_last_note: Optional[int] = None
    phrase_position: int = 0
    last_motion: int = 0  # Direction of last melodic motion

@dataclass
class TensionState:
    current_tension: float = 0.5
    phrase_tension_arc: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.5, 0.7, 0.8, 0.3])

@dataclass
class CadencePlan:
    bar: int
    cadence_type: str
    chords: List[int]

@dataclass
class ChordSpec:
    """Structured harmonic intent for one chord slot."""
    symbol: str
    root_degree: int
    root_offset: int
    quality: str
    extensions: List[int] = field(default_factory=list)
    suspension: Optional[int] = None
    is_secondary: bool = False
    target_degree: Optional[int] = None

@dataclass
class BarHarmony:
    """Resolved harmony used by every pitched MIDI part in a bar."""
    bar: int
    section: str
    root: int
    degree: int
    spec: ChordSpec
    chord_tones: List[int]
    guide_tones: List[int]
    bass_tones: List[int]
    scale_tones: List[int]

@dataclass
class SectionIntent:
    """Song-level musical intent for one arranged section."""
    name: str
    role: str
    energy: float
    density: float
    register_shift: int = 0
    cadence_strength: str = "medium"
    motif_transform: str = "identity"
    dropout_chance: float = 0.0

@dataclass
class CompositionBlueprint:
    """High-level plan that keeps harmony, melody, bass, and structure aligned."""
    mood: str
    harmonic_complexity: float
    melodic_density: float
    chorus_lift: int
    tension_arc: List[float]
    section_intents: Dict[str, SectionIntent]
    motif_seed: List[int]

@dataclass
class MelodicCell:
    notes: List[int]
    rhythm: List[int]
    articulation: str = "normal"
    contour: str = "random"
    phrase_type: str = "antecedent"
    ornaments: List[Dict] = field(default_factory=list)

@dataclass
class MelodyNote:
    """Stores a single melody note for FX generation."""
    abs_time: int
    bar: int
    note: int
    velocity: int
    source: str
    microtone_cents: int = 0

@dataclass
class Motif:
    """Represents a melodic motif with intervals and rhythm."""
    intervals: List[int]
    rhythm: List[int]
    contour: str = "arch"
    scale_context: str = "western"
    
    def sequence(self, interval_shift: int) -> 'Motif':
        return Motif(
            intervals=[i + interval_shift for i in self.intervals],
            rhythm=self.rhythm,
            contour=self.contour,
            scale_context=self.scale_context
        )
    
    def invert(self) -> 'Motif':
        return Motif(
            intervals=[-i for i in self.intervals],
            rhythm=self.rhythm,
            contour=self.contour,
            scale_context=self.scale_context
        )
    
    def augment(self) -> 'Motif':
        return Motif(
            intervals=self.intervals,
            rhythm=[r * 2 for r in self.rhythm],
            contour=self.contour,
            scale_context=self.scale_context
        )
    
    def diminish(self) -> 'Motif':
        return Motif(
            intervals=self.intervals,
            rhythm=[max(60, r // 2) for r in self.rhythm],
            contour=self.contour,
            scale_context=self.scale_context
        )
    
    def to_notes(self, root: int) -> List[int]:
        return [root + i for i in self.intervals]

@dataclass
class MicrotonalNote:
    """Note with microtonal pitch bend information."""
    note: int
    velocity: int
    duration: int
    cents_offset: int = 0
    direction: str = 'static'
