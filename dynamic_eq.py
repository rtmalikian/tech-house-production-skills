"""
Dynamic EQ — frequency-dependent processing (like iZotope Neutron Unmask).

Only corrects when specific frequency bands exceed a threshold.
Single-pass per band — no cascading filters, no iterative stacking.
"""

import os
import sys
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

# Ensure this package's directory is first on sys.path
import numpy as np
from scipy.signal import butter, sosfilt, lfilter

import config


class DynamicEQ:
    """Frequency-dependent EQ that only activates when bands exceed thresholds."""

    def __init__(self, sr: int = 48000):
        self.sr = sr
        self.bands = config.DYNAMIC_EQ_BANDS

    def process(self, y: np.ndarray, sr: int = None) -> np.ndarray:
        """
        Apply dynamic EQ bands. Only corrects when band energy exceeds threshold.
        Single-pass — no cascading, no iteration.
        """
        if sr is not None:
            self.sr = sr
        y = _ensure_stereo(y)
        y_out = y.copy()

        for band in self.bands:
            energy_db = self._measure_band_energy(y_out, band['freq'], band['q'])
            threshold_db = band['threshold_db']

            if energy_db > threshold_db:
                # Calculate correction magnitude
                excess_db = energy_db - threshold_db
                gain_db = min(excess_db / band['ratio'], band['max_gain_db'])

                if band['direction'] == 'cut':
                    gain_db = -gain_db

                # Apply single bell filter
                y_out = self._apply_bell(y_out, band['freq'], gain_db, band['q'])

        return y_out.astype(np.float32)

    def unmask(self, target_y: np.ndarray, masker_y: np.ndarray,
               sr: int, freq_range: tuple, max_cut_db: float = 2.0,
               target_label: str = None, masker_label: str = None,
               report: list = None) -> np.ndarray:
        """
        Reduce target in freq_range when masker is active.
        Like iZotope Neutron Unmask — dynamic spectral carving.
        """
        target_y = _ensure_stereo(target_y)
        masker_y = _ensure_stereo(masker_y)

        # Measure masker energy in the frequency range
        masker_energy = self._measure_band_range_energy(masker_y, freq_range[0], freq_range[1])
        target_energy = self._measure_band_range_energy(target_y, freq_range[0], freq_range[1])

        applied = False
        cut_db = 0.0

        # Only cut if both signals have significant energy in this range
        if masker_energy > -40.0 and target_energy > -40.0:
            # Cut proportional to masker energy
            cut_db = min(max_cut_db, (masker_energy + 40.0) / 20.0 * max_cut_db)
            center = (freq_range[0] + freq_range[1]) / 2.0
            q = 1.5
            target_y = self._apply_bell(target_y, center, -cut_db, q)
            applied = True

        target_after = self._measure_band_range_energy(target_y, freq_range[0], freq_range[1])
        if report is not None:
            report.append({
                "target_bus": target_label or "target",
                "masker_bus": masker_label or "masker",
                "freq_range_hz": [float(freq_range[0]), float(freq_range[1])],
                "requested_depth_db": float(max_cut_db),
                "actual_cut_db": float(cut_db if applied else 0.0),
                "target_before_db": float(target_energy),
                "target_after_db": float(target_after),
                "masker_db": float(masker_energy),
                "applied": bool(applied),
            })

        return target_y.astype(np.float32)

    def _measure_band_energy(self, y: np.ndarray, freq: float, q: float) -> float:
        """Measure energy in a specific frequency band via bandpass filter."""
        mono = np.mean(y, axis=1) if y.ndim > 1 else y
        nyq = self.sr / 2.0
        bw = freq / q
        lo = max((freq - bw / 2) / nyq, 0.001)
        hi = min((freq + bw / 2) / nyq, 0.999)
        if lo >= hi:
            return -70.0
        sos = butter(2, [lo, hi], btype='band', output='sos')
        filtered = sosfilt(sos, mono.astype(np.float64))
        rms = np.sqrt(np.mean(filtered ** 2) + 1e-12)
        return float(20.0 * np.log10(rms))

    def _measure_band_range_energy(self, y: np.ndarray, low_hz: float, high_hz: float) -> float:
        """Measure energy in a frequency range."""
        mono = np.mean(y, axis=1) if y.ndim > 1 else y
        nyq = self.sr / 2.0
        lo = max(low_hz / nyq, 0.001)
        hi = min(high_hz / nyq, 0.999)
        if lo >= hi:
            return -70.0
        sos = butter(2, [lo, hi], btype='band', output='sos')
        filtered = sosfilt(sos, mono.astype(np.float64))
        rms = np.sqrt(np.mean(filtered ** 2) + 1e-12)
        return float(20.0 * np.log10(rms))

    def _apply_bell(self, y: np.ndarray, center_freq: float,
                    gain_db: float, q: float) -> np.ndarray:
        """Apply a single bell EQ filter."""
        if abs(gain_db) < 0.05:
            return y
        A = 10 ** (gain_db / 40)
        omega = 2 * np.pi * center_freq / self.sr
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
