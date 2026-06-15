"""
Configuration presets and constants for the Python Revamp Pipeline.

All tuning values in one place — edit here, not in processing code.
"""

# ──────────────────────────────────────────────────────────────────────
# Compression Presets — MINIMAL PROCESSING
# ──────────────────────────────────────────────────────────────────────
# Philosophy: "Do no harm." Raw recordings sound good — preserve them.
# attack_ms: 20-30ms for drums to preserve transients
# ratio: gentle 1.5-2:1 (was 3:1 — too aggressive)
# gr_min/gr_max: 1.5-3 dB max (was 3-5 dB — too aggressive)

COMP_PRESETS = {
    'kick':    {'attack_ms': 25.0, 'release_ms': 50.0,  'ratio': 2.0, 'gr_min': 2.0, 'gr_max': 3.0},
    'snare':   {'attack_ms': 25.0, 'release_ms': 50.0,  'ratio': 2.0, 'gr_min': 2.0, 'gr_max': 3.0},
    'hat':     {'attack_ms': 10.0, 'release_ms': 60.0,  'ratio': 1.5, 'gr_min': 1.5, 'gr_max': 2.5},
    'clap':    {'attack_ms': 20.0, 'release_ms': 80.0,  'ratio': 2.0, 'gr_min': 2.0, 'gr_max': 3.0},
    'bass':    {'attack_ms': 20.0, 'release_ms': 0.0,   'ratio': 2.0, 'gr_min': 2.0, 'gr_max': 3.0, 'release_note': 0.5},
    'pad':     {'attack_ms': 30.0, 'release_ms': 0.0,   'ratio': 1.5, 'gr_min': 1.5, 'gr_max': 2.5, 'release_note': 0.5},
    'chord':   {'attack_ms': 30.0, 'release_ms': 0.0,   'ratio': 1.5, 'gr_min': 1.5, 'gr_max': 2.5, 'release_note': 0.5},
    'melody':  {'attack_ms': 15.0, 'release_ms': 0.0,   'ratio': 2.0, 'gr_min': 2.0, 'gr_max': 3.0, 'release_note': 0.125},
    'counter': {'attack_ms': 15.0, 'release_ms': 0.0,   'ratio': 2.0, 'gr_min': 2.0, 'gr_max': 3.0, 'release_note': 0.125},
    'chorus':  {'attack_ms': 15.0, 'release_ms': 0.0,   'ratio': 2.0, 'gr_min': 2.0, 'gr_max': 3.0, 'release_note': 0.125},
    'default': {'attack_ms': 20.0, 'release_ms': 0.0,   'ratio': 2.0, 'gr_min': 2.0, 'gr_max': 3.0, 'release_note': 0.125},
}

# Section-specific compression for automation effects (more aggressive to tame peaks)
SECTION_COMP = {
    'ratio': 5.0,
    'attack_ms': 5.0,
    'release_ms': 80.0,
}

# Drum bus parallel compression
DRUM_PARALLEL_COMP = {
    'ratio': 10.0,
    'attack_ms': 2.0,
    'release_ms': 40.0,
    'blend': 0.50,  # 50% crush + 50% dry — NY-style for presence
}

# Dynamic soft clipping (replaces fixed ceiling)
DRUM_DYNAMIC_SOFT_CLIP = {
    'stem_headroom_db': 5.0,    # gentle on individual stems
    'bus_headroom_db': 4.0,     # slightly more on bus
    'block_ms': 25,             # matches kick transient (20-30ms)
}

# Kick-bass sidechain ducking
KICK_BASS_SIDECHAIN = {
    'depth_db': 2.0,
    'release_ms': 20.0,         # fast release, no pumping
    'threshold_db': -30.0,
    'freq_range': (40, 120),
}

# ──────────────────────────────────────────────────────────────────────
# Absolute LUFS Targets Per Role (replaces pink noise gain staging)
# ──────────────────────────────────────────────────────────────────────

STEM_LUFS_TARGETS = {
    'kick':    -12.0,
    'snare':   -14.0,
    'hat':     -18.0,
    'clap':    -16.0,
    'bass':    -18.0,
    'pad':     -22.0,
    'chord':   -22.0,
    'melody':  -20.0,
    'counter': -22.0,
    'chorus':  -22.0,
    'fx':      -24.0,
    'default': -20.0,
}

# ──────────────────────────────────────────────────────────────────────
# Reverb Categories (like Roland Fantom MFX types)
# ──────────────────────────────────────────────────────────────────────
# dry_level=0.0 because dry signal is already in the mix via processed stem.
# Only the wet reverb tail goes to the return.

REVERB_CATEGORIES = {
    'drum': {
        'room_size': 0.35,
        'damping': 0.6,
        'wet_level': 0.8,
        'dry_level': 0.0,
        'width': 1.0,
    },
    'melodic': {
        'room_size': 0.60,
        'damping': 0.5,
        'wet_level': 0.8,
        'dry_level': 0.0,
        'width': 1.0,
    },
    'pad': {
        'room_size': 0.80,
        'damping': 0.35,
        'wet_level': 0.8,
        'dry_level': 0.0,
        'width': 1.0,
    },
    'fx': {
        'room_size': 0.75,
        'damping': 0.4,
        'wet_level': 0.8,
        'dry_level': 0.0,
        'width': 1.0,
    },
}

# ──────────────────────────────────────────────────────────────────────
# Layer-Specific Handling
# ──────────────────────────────────────────────────────────────────────
# The Fantom records kick/snare/melody as separate layers with distinct roles.
# Each layer needs different LUFS targets, send levels, and compression.

LAYER_PRESETS = {
    # Kick layers
    'kick_sub': {
        'lufs_target': -18.0,      # Sub is foundation, keep it controlled
        'reverb_send': 0.00,       # Completely dry
        'delay_send': 0.00,
        'reverb_category': None,
        'saturation': 1.08,        # Slight warmth for sub harmonics
        'comp_ratio': 2.5,         # Less compression on sub
        'trim_db': -4.0,           # Quieter than punch
        'eq': [
            {'type': 'highpass', 'freq': 28},
            {'type': 'lowpass', 'freq': 105},
            {'type': 'bell', 'freq': 58, 'gain_db': 2.0, 'q': 1.1},
            {'type': 'bell', 'freq': 240, 'gain_db': -3.0, 'q': 1.3},
        ],
    },
    'kick_punch': {
        'lufs_target': -14.0,      # Punch is the main kick body
        'reverb_send': 0.00,       # Dry
        'delay_send': 0.00,
        'reverb_category': None,
        'saturation': 1.05,
        'comp_ratio': 3.0,
        'trim_db': 0.0,            # Reference level
        'eq': [
            {'type': 'highpass', 'freq': 35},
            {'type': 'lowpass', 'freq': 4200},
            {'type': 'bell', 'freq': 85, 'gain_db': 2.0, 'q': 1.0},
            {'type': 'bell', 'freq': 280, 'gain_db': -3.0, 'q': 1.3},
            {'type': 'bell', 'freq': 2200, 'gain_db': 1.5, 'q': 1.2},
        ],
    },
    'kick_click': {
        'lufs_target': -20.0,      # Click is transient detail
        'reverb_send': 0.08,       # Tiny bit of room for click
        'delay_send': 0.00,
        'reverb_category': 'drum',
        'saturation': 1.03,
        'comp_ratio': 2.0,         # Light compression to preserve transient
        'trim_db': -5.0,           # Much quieter than punch
        'eq': [
            {'type': 'highpass', 'freq': 850},
            {'type': 'lowpass', 'freq': 9000},
            {'type': 'bell', 'freq': 3500, 'gain_db': 3.0, 'q': 1.1},
            {'type': 'bell', 'freq': 550, 'gain_db': -2.5, 'q': 1.4},
        ],
    },

    # Snare layers
    'snare_body': {
        'lufs_target': -16.0,      # Body is the main snare
        'reverb_send': 0.00,       # Dry — body provides the weight
        'delay_send': 0.00,
        'reverb_category': None,
        'saturation': 1.05,
        'comp_ratio': 3.0,
        'trim_db': 0.0,            # Reference level
        'eq': [
            {'type': 'lowpass', 'freq': 5200},
            {'type': 'bell', 'freq': 220, 'gain_db': 2.0, 'q': 1.2},
            {'type': 'bell', 'freq': 3500, 'gain_db': -2.0, 'q': 1.5},
        ],
    },
    'snare_snap': {
        'lufs_target': -18.0,      # Snap is the transient crack
        'reverb_send': 0.15,       # Small room for snap
        'delay_send': 0.00,
        'reverb_category': 'drum',
        'saturation': 1.03,
        'comp_ratio': 2.5,
        'trim_db': -3.0,           # Quieter than body
        'eq': [
            {'type': 'highpass', 'freq': 180},
            {'type': 'bell', 'freq': 2200, 'gain_db': 2.5, 'q': 1.1},
            {'type': 'bell', 'freq': 450, 'gain_db': -2.0, 'q': 1.2},
        ],
    },
    'snare_air': {
        'lufs_target': -22.0,      # Air is the sizzle/texture
        'reverb_send': 0.30,       # More reverb on air — it's the "space" layer
        'delay_send': 0.00,
        'reverb_category': 'drum',
        'saturation': 1.02,        # Minimal saturation — preserve the air
        'comp_ratio': 2.0,         # Light compression
        'trim_db': -6.0,           # Much quieter than body
        'eq': [
            {'type': 'highpass', 'freq': 4500},
            {'type': 'lowpass', 'freq': 14000},
        ],
    },

    # Melody layers
    'melody_lead': {
        'lufs_target': -20.0,      # Lead is the primary melody voice
        'reverb_send': 0.30,
        'delay_send': 0.25,
        'reverb_category': 'melodic',
        'saturation': 1.05,
        'comp_ratio': 2.5,
        'trim_db': 0.0,            # Reference level
    },
    'melody_poly': {
        'lufs_target': -23.0,      # Poly is harmonic support
        'reverb_send': 0.28,
        'delay_send': 0.20,
        'reverb_category': 'melodic',
        'saturation': 1.05,
        'comp_ratio': 2.0,
        'trim_db': -3.0,           # Quieter than lead
    },

    # Counter melody layers
    'counter_lead': {
        'lufs_target': -22.0,
        'reverb_send': 0.35,
        'delay_send': 0.35,
        'reverb_category': 'melodic',
        'saturation': 1.05,
        'comp_ratio': 2.5,
        'trim_db': 0.0,
    },
    'counter_bell': {
        'lufs_target': -24.0,      # Bells are shimmer, keep them back
        'reverb_send': 0.32,
        'delay_send': 0.30,
        'reverb_category': 'melodic',
        'saturation': 1.03,
        'comp_ratio': 2.0,
        'trim_db': -2.0,
    },
    'counter_pluck': {
        'lufs_target': -23.0,
        'reverb_send': 0.28,
        'delay_send': 0.25,
        'reverb_category': 'melodic',
        'saturation': 1.05,
        'comp_ratio': 2.5,
        'trim_db': -1.0,
    },

    # Chorus melody layers
    'chorus_poly': {
        'lufs_target': -22.0,
        'reverb_send': 0.32,
        'delay_send': 0.20,
        'reverb_category': 'melodic',
        'saturation': 1.05,
        'comp_ratio': 2.5,
        'trim_db': 0.0,
    },
    'chorus_brass': {
        'lufs_target': -23.0,
        'reverb_send': 0.28,
        'delay_send': 0.15,
        'reverb_category': 'melodic',
        'saturation': 1.08,        # Brass benefits from slight warmth
        'comp_ratio': 2.5,
        'trim_db': -2.0,
    },

    # Pad layers
    'pad_layer1': {
        'lufs_target': -22.0,
        'reverb_send': 0.40,
        'delay_send': 0.00,
        'reverb_category': 'pad',
        'saturation': 1.05,
        'comp_ratio': 2.0,
        'trim_db': -2.0,
    },
    'pad_layer2': {
        'lufs_target': -24.0,
        'reverb_send': 0.35,
        'delay_send': 0.00,
        'reverb_category': 'pad',
        'saturation': 1.05,
        'comp_ratio': 2.0,
        'trim_db': -4.0,
    },
    'pad_layer3': {
        'lufs_target': -25.0,
        'reverb_send': 0.30,
        'delay_send': 0.00,
        'reverb_category': 'pad',
        'saturation': 1.03,
        'comp_ratio': 2.0,
        'trim_db': -5.0,
    },
}

# ──────────────────────────────────────────────────────────────────────
# Per-Stem Send Levels (base — overridden by LAYER_PRESETS for layers)
# ──────────────────────────────────────────────────────────────────────
# Maps substring matches to (reverb_category, reverb_send, delay_send)
# ORDER MATTERS: more specific matches first (layer-specific overrides generic)

STEM_SEND_MAP = [
    # Kick layers (specific first, then generic fallback)
    ('kick_sub',    None,   0.00, 0.00),
    ('kick_punch',  None,   0.00, 0.00),
    ('kick_click',  'drum', 0.08, 0.00),
    ('kick',        None,   0.00, 0.00),
    # Snare layers
    ('snare_body',  None,   0.00, 0.00),
    ('snare_snap',  'drum', 0.15, 0.00),
    ('snare_air',   'drum', 0.30, 0.00),
    ('snare',       'drum', 0.22, 0.00),
    # Hat
    ('hat',         'drum', 0.12, 0.00),
    # Clap
    ('clap',        'drum', 0.25, 0.00),
    # Tambourine / Maracas / Perc
    ('tambourine',  'drum', 0.18, 0.00),
    ('maracas',     'drum', 0.15, 0.00),
    ('perc',        'drum', 0.15, 0.00),
    # Bass — dry
    ('bass',        None,   0.00, 0.00),
    # Melody layers
    ('melody_lead', 'melodic', 0.30, 0.25),
    ('melody_poly', 'melodic', 0.28, 0.20),
    ('melody',      'melodic', 0.30, 0.25),
    # Counter melody layers
    ('counter_lead', 'melodic', 0.35, 0.35),
    ('counter_bell', 'melodic', 0.32, 0.30),
    ('counter_pluck','melodic', 0.28, 0.25),
    ('counter',     'melodic', 0.35, 0.35),
    # Chorus melody layers
    ('chorus_poly', 'melodic', 0.32, 0.20),
    ('chorus_brass','melodic', 0.28, 0.15),
    ('chorus',      'melodic', 0.32, 0.20),
    # Pad layers
    ('pad_layer1',  'pad',  0.40, 0.00),
    ('pad_layer2',  'pad',  0.35, 0.00),
    ('pad_layer3',  'pad',  0.30, 0.00),
    ('pad',         'pad',  0.40, 0.00),
    ('chord',       'pad',  0.35, 0.00),
    # FX
    ('fx',          'fx',      0.25, 0.18),
]

# ──────────────────────────────────────────────────────────────────────
# Delay Settings
# ──────────────────────────────────────────────────────────────────────

DELAY_PRESETS = {
    'melodic': {'feedback': 0.30, 'mix': 0.25},
    'counter': {'feedback': 0.35, 'mix': 0.35},
    'chorus':  {'feedback': 0.25, 'mix': 0.20},
    'fx':      {'feedback': 0.30, 'mix': 0.18},
}

# Test-run preservation defaults: reduce delay ring-out and wet density before
# pink-noise rechecks and arrangement summing.
for _preset in DELAY_PRESETS.values():
    _preset['feedback'] *= 0.75
    _preset['mix'] *= 0.67
for _preset in REVERB_CATEGORIES.values():
    if isinstance(_preset, dict) and 'send' in _preset:
        _preset['send'] *= 0.67
    if isinstance(_preset, dict) and 'mix' in _preset:
        _preset['mix'] *= 0.67
    if isinstance(_preset, dict) and 'wet_level' in _preset:
        _preset['wet_level'] *= 0.67

# ──────────────────────────────────────────────────────────────────────
# Per-Band Stereo Width Targets
# ──────────────────────────────────────────────────────────────────────
# (low_hz, high_hz, width_multiplier)
# width 0.0 = mono, 1.0 = normal, >1.0 = wider

STEREO_WIDTH_BANDS = [
    (20,    80,   0.0),   # Sub: mono
    (80,   200,   0.5),   # Bass: narrow
    (200,  2000,  1.0),   # Mid: normal
    (2000, 8000,  1.3),   # High: wide
    (8000, 20000, 1.4),   # Air: wide
]

# ──────────────────────────────────────────────────────────────────────
# Dynamic EQ Bands
# ──────────────────────────────────────────────────────────────────────
# Only activates when band energy exceeds threshold_db

DYNAMIC_EQ_BANDS = [
    {
        'name': 'mud',
        'freq': 275.0,
        'q': 1.5,
        'threshold_db': -20.0,
        'ratio': 2.0,
        'direction': 'cut',
        'max_gain_db': 1.5,
    },
    {
        'name': 'harsh',
        'freq': 3200.0,
        'q': 2.0,
        'threshold_db': -18.0,
        'ratio': 2.0,
        'direction': 'cut',
        'max_gain_db': 1.5,
    },
    {
        'name': 'clarity',
        'freq': 5000.0,
        'q': 1.2,
        'threshold_db': -22.0,
        'ratio': 1.5,
        'direction': 'boost',
        'max_gain_db': 1.5,
    },
]

# ──────────────────────────────────────────────────────────────────────
# Mastering Targets
# ──────────────────────────────────────────────────────────────────────

MASTER_TARGET_LUFS = -11.0        # Match reference (-10.2 LUFS)
MASTER_TRUE_PEAK_DB = -1.0        # True peak ceiling
MASTER_TOLERANCE_DB = 1.0         # Acceptable LUFS deviation
MASTER_SPECTRAL_VARIANCE = 0.3    # 64-band convergence threshold

# Streaming-optimized mastering target (second pass)
STREAMING_TARGET_LUFS = -14.0     # Spotify/Apple/YouTube/Tidal
STREAMING_TRUE_PEAK_DB = -1.0     # -1 dBTP per platform specs

# ──────────────────────────────────────────────────────────────────────
# Monitoring Thresholds
# ──────────────────────────────────────────────────────────────────────
# What a professional mastering engineer checks against

MONITOR_THRESHOLDS = {
    'lufs_range': (-20.0, -6.0),          # Acceptable integrated LUFS range
    'true_peak_max_db': -0.5,             # True peak ceiling
    'crest_min': 6.0,                     # Over-compression threshold
    'crest_max': 18.0,                    # Under-compression threshold
    'lra_min': 3.0,                       # Too compressed (fatiguing)
    'lra_max': 20.0,                      # Very dynamic (may get turned down)
    'plr_min': 5.0,                       # Heavily compressed
    'plr_max': 16.0,                      # Very dynamic
    'stereo_corr_min': 0.0,               # Phase issue threshold
    'stereo_corr_max': 0.95,              # Too mono threshold
    'mono_compat_min': 0.7,               # Mono cancellation threshold (70% energy preserved)
    'clip_ratio_max': 0.001,              # Max clipping samples before flagging
    'dc_offset_max': 0.01,                # Max DC offset
    'side_mid_ratio_min': 0.05,           # Too narrow
    'side_mid_ratio_max': 1.5,            # Too wide / phase issues
    'lr_balance_max': 0.15,               # Lopsided threshold
    'spectral_flatness_max': 0.5,         # Noisy/washed threshold
}

# ──────────────────────────────────────────────────────────────────────
# Iteration Limits
# ──────────────────────────────────────────────────────────────────────

MAX_STEM_ITERATIONS = 3
MAX_BUS_ITERATIONS = 3
MAX_MASTER_ITERATIONS = 5

# Cumulative gain cap per iteration (prevents filter stacking artifacts)
MAX_CUMULATIVE_GAIN_DB = 3.0

# ──────────────────────────────────────────────────────────────────────
# EQ Safety Limits
# ──────────────────────────────────────────────────────────────────────

EQ_MIN_Q = 0.7
EQ_MAX_Q = 1.2
EQ_MAX_GAIN_DB = 1.5             # Per-band max gain per attempt (1.5 dB max — wave interference safety)
EQ_BYPASS_VARIANCE = 0.3         # Bypass if spectral variance below this

# ──────────────────────────────────────────────────────────────────────
# Saturation
# ──────────────────────────────────────────────────────────────────────

SATURATION_PRESETS = {
    'drum':    1.05,   # Ultra-subtle warmth
    'bass':    1.10,   # Slightly more
    'pad':     1.05,   # Subtle tape
    'default': 1.05,
}

# ──────────────────────────────────────────────────────────────────────
# Reference Track Path
# ──────────────────────────────────────────────────────────────────────

import os
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR = os.path.join(PIPELINE_DIR, "reference")
PINK_NOISE_DIR = os.path.join(PIPELINE_DIR, "pink_noise")
REFERENCE_TRACK_PATH = os.path.join(
    REFERENCE_DIR, "TheAlchemist_Tight_812814020637_1_5.mp3"
)
PINK_NOISE_REFERENCE_PATH = os.path.join(REFERENCE_DIR, "pink_noise_ref.wav")
PINK_NOISE_CALIBRATION_PATH = os.path.join(PINK_NOISE_DIR, "pink_noise_5min_-20LUFS.wav")

# ──────────────────────────────────────────────────────────────────────
# Bus Automation FX Presets
# ──────────────────────────────────────────────────────────────────────

# Song structure (bars) — must match midi_song_structure.py
SONG_SECTIONS = {
    'intro':   (0, 8),
    'verse1':  (8, 24),
    'pre_chorus1_build': (20, 24),
    'chorus1': (24, 32),
    'fill1':   (32, 36),
    'verse2':  (36, 52),
    'pre_chorus2_build': (48, 52),
    'chorus2': (52, 60),
    'fill2':   (60, 64),
    'outro':   (64, 72),
}

# ──────────────────────────────────────────────────────────────────────
# Objective Listening Engine Safety Defaults
# ──────────────────────────────────────────────────────────────────────
# Conservative caps for measured, confidence-gated mix corrections.

OBJECTIVE_LISTENING_PROFILE = 'lofi_warm'

OBJECTIVE_LISTENING = {
    'confidence_floor': 0.62,
    'max_eq_gain_db': {
        'kick': 1.0,
        'snare': 1.2,
        'hat': 1.2,
        'clap': 1.2,
        'bass': 1.4,
        'pad': 1.6,
        'chord': 1.6,
        'melody': 1.2,
        'counter': 1.3,
        'chorus': 1.3,
        'fx': 1.2,
        'default': 1.2,
    },
    'max_sidechain_depth_db': {
        'bass': 4.0,
        'return': 2.5,
        'default': 2.0,
    },
    'profiles': {
        # Warmth and texture are allowed; corrective moves stay especially gentle.
        'lofi_warm': {
            'confidence_floor': 0.64,
            'low_mid_margin': 0.070,
            'harsh_margin': 0.085,
            'return_target_db': -18.0,
            'max_eq_scale': 0.90,
        },
        # Keeps drums and bass forward without over-washing ambience.
        'hiphop_punchy': {
            'confidence_floor': 0.62,
            'low_mid_margin': 0.060,
            'harsh_margin': 0.075,
            'return_target_db': -19.5,
            'max_eq_scale': 1.00,
        },
        # Allows richer ambience and modal instruments while protecting harsh reeds/leads.
        'armenian_cinematic': {
            'confidence_floor': 0.66,
            'low_mid_margin': 0.080,
            'harsh_margin': 0.095,
            'return_target_db': -16.5,
            'max_eq_scale': 0.85,
        },
        # Most conservative profile for clean, transparent renders.
        'clean_pristine': {
            'confidence_floor': 0.68,
            'low_mid_margin': 0.050,
            'harsh_margin': 0.065,
            'return_target_db': -21.0,
            'max_eq_scale': 0.75,
        },
    },
    'sections': SONG_SECTIONS,
}

# Verse transition points (relative to verse start, 16-bar verse)
VERSE_TRANSITION_BARS = [4, 8, 12, 16]  # bar 4→5, 8→9, 12, 16→17

# Sectional energy
CHORUS_ENERGY_BOOST_DB = 1.0
CHORUS_STEREO_WIDTH = 1.1

# ── Per-Bus Effect Palettes Per Section ──

# DRUMS BUS
DRUM_EFFECTS = {
    'intro': [
        'tape_start',
        'hpf_sweep',
        'lpf_sweep',
        'isolate_low',
    ],
    'verse': [
        'lpf_on_off',
        'riser',
        'filter_drop',
        'phaser_sweep',
        'bitcrush_sweep',
        'isolate_mid',
        'isolate_high',
        'hpf_sweep',
        'lpf_sweep',
    ],
    'chorus': [
        'tremolo',
        'reverb_freeze',
        'impact',
        'hpf_sweep',
        'lpf_sweep',
    ],
    'fill': [
        'tape_stop',
        'beat_repeat',
        'silence_drop',
        'isolate_mid',
        'hpf_sweep',
    ],
    'outro': [
        'vinyl_stop',
        'hpf_sweep',
        'lpf_sweep',
        'isolate_high',
    ],
}

# BASS BUS
BASS_EFFECTS = {
    'intro': [
        'hpf_sweep',
        'lpf_sweep',
        'isolate_low',
    ],
    'verse': [
        'hpf_sweep',
        'hpf_on_off',
        'filter_drop',
        'lpf_sweep',
        'isolate_low',
        'isolate_mid',
    ],
    'chorus': [
        'distortion_build',
        'sidechain_pump',
        'lpf_sweep',
    ],
    'fill': [
        'hpf_sweep',
        'isolate_low',
    ],
    'outro': [
        'hpf_sweep',
        'lpf_sweep',
        'isolate_mid',
    ],
}

# MELODY BUS
MELODY_EFFECTS = {
    'intro': [
        'gain_fade',
        'reverb_wash',
        'hpf_sweep',
        'isolate_mid',
    ],
    'verse': [
        'lpf_sweep',
        'reverb_throw',
        'resonance_sweep',
        'tape_wobble',
        'isolate_mid',
        'isolate_high',
    ],
    'chorus': [
        'chorus_swell',
        'reverb_freeze',
        'stereo_widen',
        'hpf_sweep',
    ],
    'fill': [
        'delay_feedback_swell',
        'bitcrush_sweep',
        'stereo_narrow',
        'hpf_sweep',
        'isolate_mid',
    ],
    'outro': [
        'gain_fade',
        'reverb_wash',
        'lpf_sweep',
        'vinyl_crackle',
        'isolate_high',
    ],
}

# PADS BUS
PAD_EFFECTS = {
    'intro': [
        'gain_rise',
        'reverb_wash',
        'hpf_sweep',
        'stereo_widen',
    ],
    'verse': [
        'lpf_sweep',
        'reverb_throw',
        'resonance_sweep',
        'tape_wobble',
        'isolate_mid',
    ],
    'chorus': [
        'chorus_swell',
        'reverb_freeze',
        'stereo_widen',
        'lpf_sweep',
    ],
    'fill': [
        'delay_feedback_swell',
        'bitcrush_sweep',
        'stereo_narrow',
        'hpf_sweep',
    ],
    'outro': [
        'outro_fade',
        'reverb_wash',
        'lpf_sweep',
        'vinyl_crackle',
    ],
}

# PRE-CHORUS BUILD EFFECTS (filter sweeps are primary)
BUILD_EFFECTS = {
    'drums': ['riser', 'hpf_sweep', 'lpf_sweep', 'bitcrush_sweep'],
    'bass': ['hpf_sweep', 'lpf_sweep', 'isolate_low'],
    'pads': ['lpf_sweep', 'hpf_sweep', 'stereo_widen', 'resonance_sweep'],
    'melody': ['lpf_sweep', 'hpf_sweep', 'stereo_widen', 'resonance_sweep'],
}

# Map bus names to effect palettes
BUS_EFFECT_PALETTES = {
    'drums': DRUM_EFFECTS,
    'bass': BASS_EFFECTS,
    'pads': PAD_EFFECTS,
    'melody': MELODY_EFFECTS,
    'build': BUILD_EFFECTS,
}

# Fill muting: stems to mute during fill
FILL_MUTE_PATTERNS = ['pad', 'chord', 'counter']

# FX bus effects (applied to fx bus if present)
FX_EFFECTS = {
    'intro': ['reverb_wash', 'gain_fade'],
    'verse': ['isolate_high', 'tape_wobble'],
    'chorus': ['reverb_freeze', 'stereo_widen'],
    'fill': ['delay_feedback_swell'],
    'outro': ['reverb_wash', 'gain_fade'],
}

# ── Effect Depth Constants (very pronounced) ──
EFFECT_DEPTH = {
    'tape_wobble_depth': 0.020,
    'vinyl_crackle_amount': 0.12,
    'sidechain_pump_depth': 0.95,
    'reverb_wash_drums_wet': 0.9,
    'reverb_wash_melody_wet': 0.7,
    'stereo_widen_max': 2.5,
    'chorus_swell_end_depth': 0.9,
    'delay_feedback_end_feedback': 0.95,
    'delay_feedback_end_mix': 0.9,
    'distortion_build_peak_db': 30.0,
    'bitcrush_sweep_end_bits': 2,
    'riser_gain_peak_db': 12.0,
    'hpf_sweep_bass_cutoff': 6000.0,
    'lpf_sweep_end_cutoff': 100.0,
    'transition_gain_bump_db': 3.0,
}

# ──────────────────────────────────────────────────────────────────────
# Optimization
# ──────────────────────────────────────────────────────────────────────

OPTIMIZER_MAX_EVALS = 25
DRUM_OPTIMIZER_MAX_EVALS = 5

# Transient percussion — skip optimizer entirely, just gain stage
TRANSIENT_PERCUSSION_TOKENS = ['crash', 'ride', 'tambourine', 'maracas', 'cowbell',
                                'openhat', 'open_hat', 'cymbal']

OPTIMIZER_LOSS_WEIGHTS = {
    'lufs': 0.5,
    'spectral': 0.3,
    'crest': 1.0,
    'peak_penalty': 3.0,
    'context': 0.0,
    'stereo': 0.3,
    'over_compress_penalty': 0.5,
    # New monitoring-driven weights
    'lra': 0.4,                     # Loudness range (dynamic expression)
    'plr': 0.3,                     # Peak-to-loudness ratio
    'mono_compat': 0.2,             # Mono compatibility
    'tonal_balance': 0.3,           # Boomy/thin/harsh/muddy/bright penalties
    'true_peak': 2.0,               # True peak (ISP) penalty
    'onset_strength': 0.2,          # Transient preservation
}

BUSY_SECTION_PERCENTILE = 80      # Top 20% busiest sections
BUSY_SECTION_MIN_BARS = 2         # Minimum 2 bars per section
STEREO_TARGET = 0.7               # Target stereo correlation (slightly wide)
CREST_MIN = 6.0                   # Over-compression threshold (crest below this = too compressed)
COMP_OPT_RANGE = 2.0              # Optimizer can adjust GR by ±2dB around presets

# ──────────────────────────────────────────────────────────────────────
# Auto-Pan for Drum/Percussion Stems
# ──────────────────────────────────────────────────────────────────────

AUTO_PAN = {
    'pan_range': 0.05,              # ±5% of stereo field
    'rate_triplets': 2,             # 2 triplet notes per step
    'irregularity': 0.6,            # 0.0 = smooth sine, 1.0 = fully random
    'seed_base': 42,                # Base random seed (per-stem offset added)
}
