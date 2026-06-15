"""
Per-band stereo width control and M/S processing.

Mono sub, narrow bass, wide highs — mono-compatible width control.
"""

import os
import sys
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

# Ensure this package's directory is first on sys.path
import numpy as np
from scipy.signal import butter, sosfilt

import config


class StereoProcessor:
    """Per-band stereo width and M/S processing."""

    def __init__(self, sr: int = 48000):
        self.sr = sr

    def per_band_width(self, y: np.ndarray, sr: int = None) -> np.ndarray:
        """
        Control stereo width per frequency band.
        Mono sub, narrow bass, normal mid, wide highs.
        """
        if sr is not None:
            self.sr = sr
        y = _ensure_stereo(y)
        if y.ndim < 2 or y.shape[1] < 2:
            return y

        out = np.zeros_like(y)
        prev_hi = np.zeros_like(y)

        for low_hz, high_hz, width in config.STEREO_WIDTH_BANDS:
            band = self._bandpass(y, low_hz, high_hz)
            band = self._apply_width(band, width)
            out += band

        # Normalize to prevent gain buildup from overlapping bands
        peak = np.max(np.abs(out))
        orig_peak = np.max(np.abs(y))
        if peak > 1e-6 and orig_peak > 1e-6:
            out = out * (orig_peak / peak)

        return out.astype(np.float32)

    def mono_sub(self, y: np.ndarray, sr: int = None, cutoff: float = 80.0) -> np.ndarray:
        """Mono everything below cutoff."""
        if sr is not None:
            self.sr = sr
        y = _ensure_stereo(y)
        if y.ndim < 2 or y.shape[1] < 2:
            return y

        # Extract sub band
        sub = self._lowpass(y, cutoff)
        mono_sub = np.mean(sub, axis=1, keepdims=True)
        mono_sub = np.repeat(mono_sub, 2, axis=1)

        # Remove sub from original, add mono sub back
        high = y - sub
        return (high + mono_sub).astype(np.float32)

    def ms_eq(self, y: np.ndarray, sr: int = None,
              side_highshelf_freq: float = 3000.0,
              side_highshelf_gain: float = 1.5) -> np.ndarray:
        """M/S EQ: boost side channel highs for perceived width."""
        if sr is not None:
            self.sr = sr
        y = _ensure_stereo(y)
        if y.ndim < 2 or y.shape[1] < 2:
            return y

        mid = (y[:, 0] + y[:, 1]) * 0.5
        side = (y[:, 0] - y[:, 1]) * 0.5

        # High shelf boost on side channel
        side = self._high_shelf_numpy(side, side_highshelf_freq, side_highshelf_gain)

        out = np.stack([mid + side, mid - side], axis=1)

        # Peak protection
        peak = np.max(np.abs(out))
        if peak > 0.98:
            out = out * (0.95 / peak)

        return out.astype(np.float32)

    def side_hpf(self, y: np.ndarray, sr: int = None, cutoff: float = 90.0) -> np.ndarray:
        """High-pass side channel to focus low-end in mono center."""
        if sr is not None:
            self.sr = sr
        y = _ensure_stereo(y)
        if y.ndim < 2 or y.shape[1] < 2:
            return y

        mid = (y[:, 0] + y[:, 1]) * 0.5
        side = (y[:, 0] - y[:, 1]) * 0.5

        # HPF side channel
        side_hpf = side - self._lowpass_numpy(side, cutoff)

        out = np.stack([mid + side_hpf, mid - side_hpf], axis=1)
        return out.astype(np.float32)

    # ── Internal Helpers ─────────────────────────────────────────────

    def _bandpass(self, y: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
        nyq = self.sr / 2.0
        lo = max(low_hz / nyq, 0.001)
        hi = min(high_hz / nyq, 0.999)
        if lo >= hi:
            return np.zeros_like(y)
        sos = butter(2, [lo, hi], btype='band', output='sos')
        if y.ndim == 2:
            return np.column_stack([sosfilt(sos, y[:, ch]) for ch in range(y.shape[1])])
        return sosfilt(sos, y)

    def _lowpass(self, y: np.ndarray, cutoff: float) -> np.ndarray:
        nyq = self.sr / 2.0
        c = min(cutoff / nyq, 0.999)
        sos = butter(2, c, btype='low', output='sos')
        if y.ndim == 2:
            return np.column_stack([sosfilt(sos, y[:, ch]) for ch in range(y.shape[1])])
        return sosfilt(sos, y)

    def _lowpass_numpy(self, x: np.ndarray, cutoff: float) -> np.ndarray:
        """Simple IIR lowpass for M/S processing."""
        cutoff = max(20.0, min(cutoff, self.sr * 0.45))
        alpha = (2.0 * np.pi * cutoff) / (2.0 * np.pi * cutoff + self.sr)
        out = np.zeros_like(x)
        out[0] = x[0]
        for i in range(1, len(x)):
            out[i] = out[i - 1] + alpha * (x[i] - out[i - 1])
        return out

    def _high_shelf_numpy(self, x: np.ndarray, freq: float, gain_db: float) -> np.ndarray:
        """Simple high shelf via lowpass subtraction + gain."""
        if abs(gain_db) < 0.05:
            return x
        low = self._lowpass_numpy(x, freq)
        high = x - low
        gain_lin = 10.0 ** (gain_db / 20.0)
        return low + high * gain_lin

    def _apply_width(self, y: np.ndarray, width: float) -> np.ndarray:
        """Apply stereo width to a band. width=0 mono, 1 normal, >1 wider."""
        if y.ndim < 2 or y.shape[1] < 2:
            return y
        if abs(width - 1.0) < 0.01:
            return y

        mid = (y[:, 0] + y[:, 1]) * 0.5
        side = (y[:, 0] - y[:, 1]) * 0.5 * width

        out = np.stack([mid + side, mid - side], axis=1)
        return out


def _ensure_stereo(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return np.stack([y, y], axis=1)
    if y.shape[1] == 1:
        return np.repeat(y, 2, axis=1)
    return y[:, :2]
