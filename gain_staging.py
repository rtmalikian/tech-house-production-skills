"""
Absolute LUFS gain staging — replaces pink noise matching.

Measures input LUFS, applies single gain move to reach per-role target.
No pink noise, no iterative matching, no cascading filters.
"""

import os
import sys
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

# Ensure this package's directory is first on sys.path
import numpy as np
import pyloudnorm as pyln

import config


def gain_stage_to_target(y: np.ndarray, sr: int, role: str = 'default') -> tuple:
    """
    Measure input LUFS, apply single gain move to reach per-role target.

    Args:
        y: audio signal (numpy array)
        sr: sample rate
        role: stem role (kick, snare, bass, pad, melody, etc.)

    Returns:
        (y_gained, gain_db): gained signal and the gain applied in dB
    """
    y = _ensure_stereo(y)
    target_lufs = config.STEM_LUFS_TARGETS.get(role, config.STEM_LUFS_TARGETS['default'])

    meter = pyln.Meter(sr)
    current_lufs = meter.integrated_loudness(y)

    if current_lufs <= -70.0 or not np.isfinite(current_lufs):
        return y, 0.0

    gain_db = target_lufs - current_lufs
    gain_db = float(np.clip(gain_db, -24.0, 24.0))

    if abs(gain_db) < 0.1:
        return y, 0.0

    y_out = y * (10.0 ** (gain_db / 20.0))
    return y_out.astype(np.float32), gain_db


def measure_lufs(y: np.ndarray, sr: int) -> float:
    """Measure integrated LUFS of signal."""
    y = _ensure_stereo(y)
    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(y)
    if not np.isfinite(lufs):
        return -70.0
    return float(lufs)


def _ensure_stereo(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return np.stack([y, y], axis=1)
    if y.shape[1] == 1:
        return np.repeat(y, 2, axis=1)
    return y[:, :2]
