import os
import random

# ============================================================================
# TECH HOUSE CONFIGURATION
# ============================================================================

TICKS_PER_BEAT = 480  # MIDI standard

# Tech house: 124-128 BPM (sweet spot 125-126)
TECH_HOUSE_BPM_MIN = 124
TECH_HOUSE_BPM_MAX = 128

# Tech house arrangement: DJ-friendly, commercial length, all 16-bar sections
# Intro(16) + Drop1(32) + Breakdown(32) + Drop2(32) + Outro(16) = 128 bars
# At 128 BPM: 128 bars × 4 beats × 60/128 = 4:00 (commercial minimum)
TOTAL_BARS = 128

TIME_SIGNATURES = {
    '4-4': {'numerator': 4, 'denominator': 4, 'beats_per_bar': 4},
}

def get_bar_length_ticks(ts_key='4-4'):
    return int(TICKS_PER_BEAT * TIME_SIGNATURES[ts_key]['beats_per_bar'])

def get_song_length_ticks(ts_key='4-4'):
    return get_bar_length_ticks(ts_key) * TOTAL_BARS

def get_random_bpm():
    """Return a random tech house BPM in the 124-128 range."""
    return random.randint(TECH_HOUSE_BPM_MIN, TECH_HOUSE_BPM_MAX)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROD_DIR = "/Volumes/Raphael/Tech House/output"
os.makedirs(PROD_DIR, exist_ok=True)

SCALES_FILE = os.path.join(PROD_DIR, "scales.txt")

# Tech house: NO swing. Fully quantized. Groove comes from velocity,
# not timing displacement. Kick and bass must be locked to the grid.
SWING_VALUES = {
    'none': 0.50,       # No swing (straight — the tech house standard)
    'light': 0.50,      # No swing — even "light" is straight for tech house
    'medium': 0.52,     # Barely perceptible (2%) — only for hats if at all
    'heavy': 0.54,      # Max 4% — still very tight
}

# Tech house humanization: ZERO. Perfect quantization.
# Groove comes from velocity patterns, not timing displacement.
HUMANIZATION = {
    'timing_subtle': (0, 0),          # No timing jitter
    'timing_hat': (0, 0),             # No timing jitter on hats
    'velocity_subtle': (-4, 4),       # Minimal velocity variation
    'velocity_hat': (-15, 8),         # Hat velocity groove (this IS the groove)
    'velocity_perc': (-6, 6),         # Percussion velocity variation
}

# Tech house: more stepwise, less leaping than lofi
INTERVAL_WEIGHTS = {
    'stepwise': 0.70,
    'small_leap': 0.25,
    'large_leap': 0.05,
}

# Register ranges for tech house instruments
BASS_REGISTER_RANGE = (33, 52)  # Above sub rumble, below low-mid mud

REGISTER_RANGES = {
    'bass': BASS_REGISTER_RANGE,
    'sub_bass': (28, 40),          # Tight sub range — only for foundation
    'main_melody': (60, 84),       # Stabs/leads
    'counter_melody': (60, 78),
    'chorus_melody': (60, 84),
    'pad': (60, 84),               # Chord stabs, not lush pads
    'acid_line': (48, 72),         # 303-style acid range
}

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Tech house preferred keys (minor keys dominate)
PREFERRED_KEYS = ['G', 'A', 'F', 'D', 'C', 'A#']
PREFERRED_SCALES = ['Minor']  # Almost exclusively minor for tech house
