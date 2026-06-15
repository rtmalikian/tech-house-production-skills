"""
Reference track analysis and 64-band Match EQ.

Analyzes a reference track's spectral profile and iteratively matches
the mix to it. Single-pass correction per attempt — no filter stacking.
"""

# Ensure this package's directory is first on sys.path
import os
import sys
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

import numpy as np
import pyloudnorm as pyln
import librosa
import soundfile as sf
from scipy.signal import lfilter

import config


class ReferenceAnalyzer:
    """64-band reference matching (like iZotope Match EQ)."""

    # 64 logarithmically-spaced bands from 20Hz to 20kHz
    N_BANDS = 64
    FMIN = 20.0
    FMAX = 20000.0

    def __init__(self):
        self.reference_analysis = None

    def analyze_reference(self, ref_path: str) -> dict:
        """Load and analyze reference track."""
        if not os.path.exists(ref_path):
            print(f"  [Reference] Missing: {ref_path}")
            return None

        try:
            y, sr = sf.read(ref_path, always_2d=True)
            y = _ensure_stereo(np.asarray(y, dtype=np.float32))
        except Exception as e:
            print(f"  [Reference] Read failed: {e}")
            return None

        # LUFS
        meter = pyln.Meter(sr)
        try:
            lufs = float(meter.integrated_loudness(y))
        except Exception:
            lufs = -70.0

        # Crest factor
        peak = float(np.max(np.abs(y)))
        rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-12)
        crest_db = 20.0 * np.log10(max(peak, 1e-12) / rms)

        # 64-band spectral profile
        spectral_profile = self._get_narrowband_profile(y, sr)

        # Stereo correlation
        if y.ndim == 2 and y.shape[1] >= 2:
            denom = float(np.std(y[:, 0]) * np.std(y[:, 1]))
            stereo_corr = float(np.corrcoef(y[:, 0], y[:, 1])[0, 1]) if denom > 1e-12 else 1.0
        else:
            stereo_corr = 1.0

        self.reference_analysis = {
            'lufs': lufs,
            'peak_db': 20.0 * np.log10(max(peak, 1e-10)),
            'crest_db': crest_db,
            'stereo_corr': stereo_corr,
            'spectral_profile': spectral_profile,
        }

        print(f"  [Reference] Loaded: {os.path.basename(ref_path)}")
        print(f"    LUFS={lufs:+.1f} crest={crest_db:.1f}dB stereo={stereo_corr:.2f}")
        return self.reference_analysis

    def match_eq(self, y: np.ndarray, sr: int,
                 reference_profile: dict = None,
                 max_iterations: int = 5,
                 dsp_engine=None) -> np.ndarray:
        """
        Iteratively match spectral profile to reference.

        CRITICAL: Each iteration REPLACES the previous correction (never stacks).
        Single combined filter per attempt. Cumulative gain capped.
        """
        if reference_profile is None:
            reference_profile = self.reference_analysis
        if reference_profile is None:
            return y

        ref_profile = reference_profile.get('spectral_profile', {})
        if not ref_profile:
            return y

        y = _ensure_stereo(y)
        best_y = y.copy()
        best_distance = float('inf')

        for attempt in range(max_iterations):
            # Measure current profile
            current_profile = self._get_narrowband_profile(best_y, sr)

            # Compute deltas
            deltas = self._compute_deltas(current_profile, ref_profile)
            max_delta = max(abs(d) for d in deltas.values())

            # Convergence check
            if max_delta < 0.3:
                break

            # Compute corrections (top 3 worst bands only)
            sorted_deltas = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)
            corrections = sorted_deltas[:3]

            # Apply cumulative gain cap
            total_gain = sum(abs(d) for _, d in corrections)
            if total_gain > config.MAX_CUMULATIVE_GAIN_DB:
                scale = config.MAX_CUMULATIVE_GAIN_DB / total_gain
                corrections = [(k, v * scale) for k, v in corrections]

            # Build single combined filter
            y_candidate = best_y.copy()
            for band_key, delta in corrections:
                center_hz = self._band_center(band_key)
                gain_db = float(np.clip(delta * 0.5, -config.EQ_MAX_GAIN_DB, config.EQ_MAX_GAIN_DB))
                q = float(np.clip(1.0, config.EQ_MIN_Q, config.EQ_MAX_Q))
                y_candidate = self._apply_bell(y_candidate, sr, center_hz, gain_db, q)

            # Gain match to original level
            if dsp_engine:
                y_candidate, _ = dsp_engine.gain_match(best_y, y_candidate, max_correction_db=2.0)

            # Verify
            new_profile = self._get_narrowband_profile(y_candidate, sr)
            new_distance = self._profile_distance(new_profile, ref_profile)

            if new_distance < best_distance:
                best_distance = new_distance
                best_y = y_candidate
            else:
                break  # Don't degrade

        return best_y

    def _get_narrowband_profile(self, y: np.ndarray, sr: int) -> dict:
        """Get spectral profile with N narrow bands."""
        mono = np.mean(y, axis=1) if y.ndim > 1 else y
        if len(mono) < 2048:
            return {}

        S = np.abs(librosa.stft(mono))
        freqs = librosa.fft_frequencies(sr=sr)

        # Logarithmically spaced band edges
        band_edges = np.logspace(
            np.log10(self.FMIN), np.log10(self.FMAX), self.N_BANDS + 1
        )

        band_energies = {}
        for i in range(self.N_BANDS):
            low, high = band_edges[i], band_edges[i + 1]
            idx = (freqs >= low) & (freqs < high)
            if np.any(idx):
                band_energies[f"{low:.0f}-{high:.0f}"] = float(np.mean(S[idx, :]))
            else:
                band_energies[f"{low:.0f}-{high:.0f}"] = 0.0

        # Normalize
        max_energy = max(band_energies.values()) if band_energies else 1e-10
        if max_energy < 1e-10:
            return band_energies

        return {k: v / max_energy for k, v in band_energies.items()}

    def _compute_deltas(self, current: dict, reference: dict) -> dict:
        """Compute per-band deltas (reference - current)."""
        deltas = {}
        for key in reference:
            if key in current:
                deltas[key] = reference[key] - current[key]
        return deltas

    def _profile_distance(self, a: dict, b: dict) -> float:
        """RMS distance between two spectral profiles."""
        keys = set(a.keys()) & set(b.keys())
        if not keys:
            return 0.0
        diffs = [(a[k] - b[k]) ** 2 for k in sorted(keys)]
        return float(np.sqrt(np.mean(diffs)))

    def _band_center(self, band_key: str) -> float:
        """Get center frequency from band key like '100-200'."""
        parts = band_key.split('-')
        try:
            low = float(parts[0])
            high = float(parts[1])
            return (low + high) / 2.0
        except (ValueError, IndexError):
            return 1000.0

    def _apply_bell(self, y: np.ndarray, sr: int,
                    center_freq: float, gain_db: float, q: float) -> np.ndarray:
        """Apply a single bell EQ filter."""
        if abs(gain_db) < 0.05:
            return y
        A = 10 ** (gain_db / 40)
        omega = 2 * np.pi * center_freq / sr
        alpha = np.sin(omega) / (2 * q)

        b = [1 + alpha * A, -2 * np.cos(omega), 1 - alpha * A]
        a = [1 + alpha / A, -2 * np.cos(omega), 1 - alpha / A]

        orig_dtype = y.dtype
        y_f64 = y.astype(np.float64)
        return lfilter(b, a, y_f64, axis=0).astype(orig_dtype)


def _ensure_stereo(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return np.stack([y, y], axis=1)
    if y.shape[1] == 1:
        return np.repeat(y, 2, axis=1)
    return y[:, :2]
