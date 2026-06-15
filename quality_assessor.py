"""
Objective audio quality assessment — the virtual assistant engineer.

Measures everything a professional mastering engineer sees on their meters:
  - LUFS (integrated, momentary, short-term)
  - LRA (Loudness Range)
  - True Peak (inter-sample peak via 4x oversampling)
  - PLR (Peak-to-Loudness Ratio)
  - Crest factor
  - Stereo correlation (overall + per-band)
  - Spectral profile (centroid, rolloff, bandwidth, flatness, contrast)
  - MFCC (timbre fingerprint)
  - Onset strength (transient punch)
  - HPSS (harmonic/percussive energy ratio)
  - Clipping detection
  - DC offset
  - Mono compatibility
  - Side/Mid energy ratio (stereo width)
  - L/R balance
  - Band energy ratios (boomy/thin/harsh/muddy/bright)
"""

import os
import sys
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

import numpy as np
import pyloudnorm as pyln
import librosa
from scipy.signal import resample_poly, butter, sosfilt

import config


class QualityAssessor:
    """Objective audio quality metrics and pass/fail assessment."""

    GRANULAR_BANDS = [
        (20, 60), (60, 150), (150, 300), (300, 600), (600, 1200),
        (1200, 2500), (2500, 4500), (4500, 8000), (8000, 12000), (12000, 20000),
    ]

    # Per-band correlation bands
    CORR_BANDS = [
        ('sub', 20, 120),
        ('low', 120, 400),
        ('mid', 400, 2500),
        ('high', 2500, 8000),
        ('air', 8000, 20000),
    ]

    # Perceptual tonal balance bands
    TONAL_BANDS = {
        'sub_bass':   (20, 80),
        'bass':       (80, 250),
        'low_mid':    (250, 500),
        'mid':        (500, 2000),
        'upper_mid':  (2000, 4000),
        'presence':   (4000, 6000),
        'brilliance': (6000, 20000),
    }

    def assess_stem(self, y: np.ndarray, sr: int, role: str = 'default') -> dict:
        """Assess a processed stem against quality targets."""
        y = _ensure_stereo(y)
        mono = np.mean(y, axis=1)

        meter = pyln.Meter(sr)

        # ── Loudness ──
        try:
            lufs = float(meter.integrated_loudness(y))
        except Exception:
            lufs = -70.0

        try:
            lra = float(meter.loudness_range(y))
        except Exception:
            lra = 0.0

        # Momentary LUFS (400ms windows)
        momentary_lufs = self._windowed_lufs(y, sr, meter, window_sec=0.4)
        short_term_lufs = self._windowed_lufs(y, sr, meter, window_sec=3.0)

        target_lufs = config.STEM_LUFS_TARGETS.get(role, config.STEM_LUFS_TARGETS['default'])
        lufs_pass = abs(lufs - target_lufs) < 6.0

        # ── Peak / True Peak ──
        peak = float(np.max(np.abs(y)))
        peak_db = 20.0 * np.log10(max(peak, 1e-10))

        true_peak = self._true_peak(y, sr)
        true_peak_db = 20.0 * np.log10(max(true_peak, 1e-10))

        peak_pass = true_peak_db < -0.5

        # ── PLR (Peak-to-Loudness Ratio) ──
        plr = true_peak_db - lufs if lufs > -70.0 else 0.0

        # ── Crest factor ──
        rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-12)
        crest_db = 20.0 * np.log10(max(peak, 1e-12) / rms)
        crest_pass = 6.0 < crest_db < 16.0

        # ── Clipping detection ──
        clip_ratio = float(np.sum(np.abs(y) >= 0.99) / y.size)
        is_clipped = clip_ratio > 0.001

        # ── DC offset ──
        dc_offset = float(np.mean(y))

        # ── Stereo analysis ──
        if y.ndim == 2 and y.shape[1] >= 2:
            left, right = y[:, 0], y[:, 1]
            denom = float(np.std(left) * np.std(right))
            stereo_corr = float(np.corrcoef(left, right)[0, 1]) if denom > 1e-12 else 1.0

            # L/R balance
            rms_l = float(np.sqrt(np.mean(left ** 2)))
            rms_r = float(np.sqrt(np.mean(right ** 2)))
            lr_balance = (rms_l - rms_r) / max(rms_l + rms_r, 1e-12)

            # Side/Mid energy ratio (stereo width metric)
            mid = (left + right) / 2.0
            side = (left - right) / 2.0
            mid_rms = float(np.sqrt(np.mean(mid ** 2)))
            side_rms = float(np.sqrt(np.mean(side ** 2)))
            side_mid_ratio = side_rms / max(mid_rms, 1e-12)

            # Per-band correlation
            per_band_corr = self._per_band_correlation(left, right, sr)

            # Mono compatibility
            mono_sum = mid * 2.0
            stereo_energy = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
            mono_energy = float(np.sqrt(np.mean(mono_sum.astype(np.float64) ** 2)))
            mono_compat = mono_energy / max(stereo_energy, 1e-12)
        else:
            stereo_corr = 1.0
            lr_balance = 0.0
            side_mid_ratio = 0.0
            per_band_corr = {name: 1.0 for name, _, _ in self.CORR_BANDS}
            mono_compat = 1.0

        stereo_pass = 0.2 < stereo_corr < 0.95

        # ── Spectral analysis ──
        spectral_profile = self._get_spectral_profile(y, sr)

        S = np.abs(librosa.stft(mono)) if len(mono) >= 2048 else None
        if S is not None:
            spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
            spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr)))
            spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr)))
            spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(S=S)))
            spectral_contrast = float(np.mean(librosa.feature.spectral_contrast(S=S, sr=sr)))
            mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S ** 2), sr=sr, n_mfcc=13)
            mfcc_means = [float(np.mean(mfcc[i])) for i in range(13)]
        else:
            spectral_centroid = 0.0
            spectral_rolloff = 0.0
            spectral_bandwidth = 0.0
            spectral_flatness = 0.0
            spectral_contrast = 0.0
            mfcc_means = [0.0] * 13

        # ── Onset strength (transient punch) ──
        onset_env = librosa.onset.onset_strength(y=mono, sr=sr)
        onset_strength_mean = float(np.mean(onset_env))
        onset_strength_max = float(np.max(onset_env))

        # ── HPSS (harmonic/percussive ratio) ──
        if len(mono) >= 2048:
            harmonic, percussive = librosa.effects.hpss(y if y.ndim == 1 else mono)
            h_energy = float(np.sqrt(np.mean(harmonic ** 2)))
            p_energy = float(np.sqrt(np.mean(percussive ** 2)))
            harmonic_ratio = h_energy / max(h_energy + p_energy, 1e-12)
        else:
            harmonic_ratio = 0.5

        # ── Band energy ratios (perceptual tonal balance) ──
        tonal_balance = self._tonal_balance(y, sr)
        boomy = tonal_balance.get('boomy', 0.0)
        thin = tonal_balance.get('thin', 0.0)
        muddy = tonal_balance.get('muddy', 0.0)
        harsh = tonal_balance.get('harsh', 0.0)
        bright = tonal_balance.get('bright', 0.0)

        return {
            # Loudness
            'lufs': lufs,
            'lufs_target': target_lufs,
            'lufs_pass': lufs_pass,
            'lra': lra,
            'momentary_lufs': momentary_lufs,
            'short_term_lufs': short_term_lufs,
            # Peak
            'peak_db': peak_db,
            'true_peak_db': true_peak_db,
            'peak_pass': peak_pass,
            # PLR
            'plr': plr,
            # Crest
            'crest_db': crest_db,
            'crest_pass': crest_pass,
            # Clipping
            'clip_ratio': clip_ratio,
            'is_clipped': is_clipped,
            # DC
            'dc_offset': dc_offset,
            # Stereo
            'stereo_corr': stereo_corr,
            'stereo_pass': stereo_pass,
            'lr_balance': lr_balance,
            'side_mid_ratio': side_mid_ratio,
            'per_band_corr': per_band_corr,
            'mono_compat': mono_compat,
            # Spectral
            'spectral_profile': spectral_profile,
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_flatness': spectral_flatness,
            'spectral_contrast': spectral_contrast,
            'mfcc': mfcc_means,
            # Transients
            'onset_strength_mean': onset_strength_mean,
            'onset_strength_max': onset_strength_max,
            # Harmonic
            'harmonic_ratio': harmonic_ratio,
            # Tonal balance
            'tonal_balance': tonal_balance,
            'boomy': boomy,
            'thin': thin,
            'muddy': muddy,
            'harsh': harsh,
            'bright': bright,
            # Pass/fail
            'all_pass': lufs_pass and peak_pass and crest_pass and stereo_pass,
        }

    def assess_mix(self, y: np.ndarray, sr: int,
                   reference_analysis: dict = None) -> dict:
        """Assess a mix with full monitoring suite."""
        y = _ensure_stereo(y)
        mono = np.mean(y, axis=1)
        meter = pyln.Meter(sr)

        try:
            lufs = float(meter.integrated_loudness(y))
        except Exception:
            lufs = -70.0

        try:
            lra = float(meter.loudness_range(y))
        except Exception:
            lra = 0.0

        momentary_lufs = self._windowed_lufs(y, sr, meter, window_sec=0.4)
        short_term_lufs = self._windowed_lufs(y, sr, meter, window_sec=3.0)

        peak = float(np.max(np.abs(y)))
        peak_db = 20.0 * np.log10(max(peak, 1e-10))
        true_peak = self._true_peak(y, sr)
        true_peak_db = 20.0 * np.log10(max(true_peak, 1e-10))
        plr = true_peak_db - lufs if lufs > -70.0 else 0.0

        rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-12)
        crest_db = 20.0 * np.log10(max(peak, 1e-12) / rms)

        clip_ratio = float(np.sum(np.abs(y) >= 0.99) / y.size)

        if y.ndim == 2 and y.shape[1] >= 2:
            left, right = y[:, 0], y[:, 1]
            denom = float(np.std(left) * np.std(right))
            stereo_corr = float(np.corrcoef(left, right)[0, 1]) if denom > 1e-12 else 1.0
            mid = (left + right) / 2.0
            side = (left - right) / 2.0
            side_mid_ratio = float(np.sqrt(np.mean(side ** 2))) / max(float(np.sqrt(np.mean(mid ** 2))), 1e-12)
            rms_l = float(np.sqrt(np.mean(left ** 2)))
            rms_r = float(np.sqrt(np.mean(right ** 2)))
            lr_balance = (rms_l - rms_r) / max(rms_l + rms_r, 1e-12)
            per_band_corr = self._per_band_correlation(left, right, sr)
            mono_sum = mid * 2.0
            mono_compat = float(np.sqrt(np.mean(mono_sum.astype(np.float64) ** 2))) / max(float(np.sqrt(np.mean(y.astype(np.float64) ** 2))), 1e-12)
        else:
            stereo_corr = 1.0
            side_mid_ratio = 0.0
            lr_balance = 0.0
            per_band_corr = {name: 1.0 for name, _, _ in self.CORR_BANDS}
            mono_compat = 1.0

        spectral_profile = self._get_spectral_profile(y, sr)

        S = np.abs(librosa.stft(mono)) if len(mono) >= 2048 else None
        if S is not None:
            spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
            spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr)))
            spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(S=S)))
        else:
            spectral_centroid = 0.0
            spectral_rolloff = 0.0
            spectral_flatness = 0.0

        tonal_balance = self._tonal_balance(y, sr)

        result = {
            'lufs': lufs, 'lra': lra,
            'momentary_lufs': momentary_lufs, 'short_term_lufs': short_term_lufs,
            'peak_db': peak_db, 'true_peak_db': true_peak_db, 'plr': plr,
            'crest_db': crest_db, 'clip_ratio': clip_ratio,
            'stereo_corr': stereo_corr, 'side_mid_ratio': side_mid_ratio,
            'lr_balance': lr_balance, 'per_band_corr': per_band_corr, 'mono_compat': mono_compat,
            'spectral_profile': spectral_profile,
            'spectral_centroid': spectral_centroid, 'spectral_rolloff': spectral_rolloff,
            'spectral_flatness': spectral_flatness,
            'tonal_balance': tonal_balance,
        }

        if reference_analysis:
            result['lufs_delta'] = lufs - reference_analysis.get('lufs', lufs)
            result['crest_delta'] = crest_db - reference_analysis.get('crest_db', crest_db)
            result['plr_delta'] = plr - reference_analysis.get('plr', plr)
            result['lra_delta'] = lra - reference_analysis.get('lra', lra)
            result['spectral_delta'] = self._spectral_distance(
                spectral_profile, reference_analysis.get('spectral_profile', {})
            )
            result['width_delta'] = side_mid_ratio - reference_analysis.get('side_mid_ratio', side_mid_ratio)
            result['lufs_pass'] = abs(result['lufs_delta']) < config.MASTER_TOLERANCE_DB
            result['spectral_pass'] = result['spectral_delta'] < config.MASTER_SPECTRAL_VARIANCE
        else:
            result['lufs_delta'] = 0.0
            result['crest_delta'] = 0.0
            result['plr_delta'] = 0.0
            result['lra_delta'] = 0.0
            result['spectral_delta'] = 0.0
            result['width_delta'] = 0.0
            result['lufs_pass'] = True
            result['spectral_pass'] = True

        return result

    def loudness_match(self, y_before: np.ndarray, y_after: np.ndarray,
                       sr: int, max_correction_db: float = 12.0) -> tuple:
        """Verify processing didn't destroy level. Correct if delta > 3 dB."""
        y_before = _ensure_stereo(y_before)
        y_after = _ensure_stereo(y_after)

        meter = pyln.Meter(sr)
        try:
            lufs_before = float(meter.integrated_loudness(y_before))
            lufs_after = float(meter.integrated_loudness(y_after))
        except Exception:
            return y_after, 0.0

        if lufs_before <= -70.0 or lufs_after <= -70.0:
            return y_after, 0.0

        delta = lufs_before - lufs_after
        if abs(delta) < 3.0:
            return y_after, 0.0

        correction = float(np.clip(delta * 0.7, -max_correction_db, max_correction_db))
        y_out = y_after * (10.0 ** (correction / 20.0))
        return y_out.astype(np.float32), correction

    # ── Internal analysis methods ──

    def _true_peak(self, y: np.ndarray, sr: int, oversample: int = 4) -> float:
        """True peak via oversampling (inter-sample peak detection)."""
        mono = np.mean(y, axis=1) if y.ndim > 1 else y
        if len(mono) < 10:
            return float(np.max(np.abs(y)))
        upsampled = resample_poly(mono, oversample, 1)
        return float(np.max(np.abs(upsampled)))

    def _windowed_lufs(self, y: np.ndarray, sr: int, meter: pyln.Meter,
                       window_sec: float = 0.4) -> list:
        """Compute LUFS in overlapping windows (momentary=0.4s, short-term=3s)."""
        window_samples = int(sr * window_sec)
        hop = window_samples // 2
        results = []
        for start in range(0, len(y) - window_samples + 1, hop):
            chunk = y[start:start + window_samples]
            try:
                l = float(meter.integrated_loudness(chunk))
                if l > -70.0:
                    results.append(l)
            except Exception:
                pass
        return results

    def _per_band_correlation(self, left: np.ndarray, right: np.ndarray, sr: int) -> dict:
        """L/R correlation per frequency band."""
        result = {}
        for name, low, high in self.CORR_BANDS:
            if high >= sr / 2:
                result[name] = 1.0
                continue
            sos = butter(4, [low, high], btype='band', fs=sr, output='sos')
            l_filt = sosfilt(sos, left)
            r_filt = sosfilt(sos, right)
            denom = float(np.std(l_filt) * np.std(r_filt))
            if denom < 1e-12:
                result[name] = 1.0
            else:
                result[name] = float(np.corrcoef(l_filt, r_filt)[0, 1])
        return result

    def _tonal_balance(self, y: np.ndarray, sr: int) -> dict:
        """Perceptual tonal balance: boomy/thin/muddy/harsh/bright scores."""
        mono = np.mean(y, axis=1) if y.ndim > 1 else y
        if len(mono) < 2048:
            return {'boomy': 0.0, 'thin': 0.0, 'muddy': 0.0, 'harsh': 0.0, 'bright': 0.0}

        S = np.abs(librosa.stft(mono))
        freqs = librosa.fft_frequencies(sr=sr)

        band_energy = {}
        for name, (lo, hi) in self.TONAL_BANDS.items():
            idx = (freqs >= lo) & (freqs <= hi)
            band_energy[name] = float(np.mean(S[idx, :])) if np.any(idx) else 0.0

        total = sum(band_energy.values()) or 1e-12

        sub_ratio = band_energy['sub_bass'] / total
        bass_ratio = band_energy['bass'] / total
        low_mid_ratio = band_energy['low_mid'] / total
        upper_mid_ratio = band_energy['upper_mid'] / total
        presence_ratio = band_energy['presence'] / total
        brilliance_ratio = band_energy['brilliance'] / total

        return {
            'boomy': float(max(0, sub_ratio + bass_ratio - 0.3)),
            'thin': float(max(0, 0.2 - bass_ratio - low_mid_ratio)),
            'muddy': float(max(0, low_mid_ratio - 0.15)),
            'harsh': float(max(0, upper_mid_ratio + presence_ratio - 0.25)),
            'bright': float(max(0, brilliance_ratio - 0.15)),
        }

    def _get_spectral_profile(self, y: np.ndarray, sr: int) -> dict:
        """Get normalized spectral profile across granular bands."""
        mono = np.mean(y, axis=1) if y.ndim > 1 else y
        if len(mono) < 2048:
            return {f"{lo}-{hi}": 0.0 for lo, hi in self.GRANULAR_BANDS}

        S = np.abs(librosa.stft(mono))
        freqs = librosa.fft_frequencies(sr=sr)

        band_energies = {}
        for low, high in self.GRANULAR_BANDS:
            idx = (freqs >= low) & (freqs <= high)
            if np.any(idx):
                band_energies[f"{low}-{high}"] = float(np.mean(S[idx, :]))
            else:
                band_energies[f"{low}-{high}"] = 0.0

        max_energy = max(band_energies.values()) if band_energies else 1e-10
        if max_energy < 1e-10:
            return band_energies

        return {k: v / max_energy for k, v in band_energies.items()}

    def _spectral_distance(self, profile_a: dict, profile_b: dict) -> float:
        """RMS difference between two spectral profiles."""
        keys = set(profile_a.keys()) & set(profile_b.keys())
        if not keys:
            return 0.0
        diffs = [(profile_a[k] - profile_b[k]) ** 2 for k in sorted(keys)]
        return float(np.sqrt(np.mean(diffs)))


def _ensure_stereo(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return np.stack([y, y], axis=1)
    if y.shape[1] == 1:
        return np.repeat(y, 2, axis=1)
    return y[:, :2]
