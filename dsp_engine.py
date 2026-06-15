"""
Core DSP engine using pedalboard and numpy.

All audio processing goes through this module. No FFmpeg subprocess calls.
"""

import os
import sys
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

# Ensure this package's directory is first on sys.path
import numpy as np
import pyloudnorm as pyln
from scipy.signal import resample_poly
from pedalboard import (
    Compressor, Limiter, Reverb, Chorus, Delay, Phaser, Gain, Clipping,
    HighpassFilter, LowpassFilter, LadderFilter, Distortion, Bitcrush,
)

import config


class DSPEngine:
    """Pure Python audio processing engine."""

    # ── Compression ──────────────────────────────────────────────────

    def adaptive_compress(self, y: np.ndarray, sr: int, role: str = 'default',
                          threshold_offset_db: float = 0.0,
                          bpm: float = 90.0) -> dict:
        """
        Analyze signal → binary-search threshold → pedalboard Compressor.
        Makeup gain = threshold_db × (1 - 1/ratio).

        Release time is BPM-synced for bass/melody/pad roles:
          - bass/pad/chord: 1/2 note duration
          - melody/counter/chorus: 1/8 note duration
          - kick/snare: fixed 50ms for punchiness

        Args:
            threshold_offset_db: shifts the GR target range ±dB (for optimizer).
                                 Positive = more compression, negative = less.
            bpm: beats per minute (for BPM-synced release)

        Returns dict with y, gr_achieved, makeup_db, debug info.
        """
        y = self._ensure_stereo(y)
        preset = config.COMP_PRESETS.get(role, config.COMP_PRESETS['default'])
        ratio = preset['ratio']
        attack_ms = preset['attack_ms']
        gr_min = preset['gr_min']
        gr_max = preset['gr_max']

        # Compute release time — BPM-synced or fixed
        release_note = preset.get('release_note')
        if release_note and bpm > 0:
            # BPM-synced: note_duration in seconds = (60/bpm) * note_value
            # note_value: 0.5 = half note, 0.125 = eighth note, etc.
            release_ms = (60.0 / bpm) * release_note * 1000.0
        else:
            release_ms = preset['release_ms']

        # Measure input
        input_peak_db = self._peak_db(y)
        input_rms = self._rms(y)

        # Binary search for threshold that achieves target GR
        # threshold_offset_db shifts the GR target range (not the threshold)
        # so the optimizer actually controls compression amount
        effective_gr_min = max(0.5, gr_min + threshold_offset_db)
        effective_gr_max = max(1.0, gr_max + threshold_offset_db)
        target_mid = (effective_gr_min + effective_gr_max) / 2.0
        threshold_db = input_peak_db - 3.0
        best_threshold_db = threshold_db
        best_gr = 0.0
        best_diff = float('inf')

        for _ in range(10):
            compressor = Compressor(
                threshold_db=threshold_db,
                ratio=ratio,
                attack_ms=attack_ms,
                release_ms=release_ms,
            )
            y_test = compressor(y, sr)
            output_rms = self._rms(y_test)
            actual_gr = self._gr_from_rms(input_rms, output_rms)

            diff = abs(actual_gr - target_mid)
            if diff < best_diff:
                best_diff = diff
                best_threshold_db = threshold_db
                best_gr = actual_gr

            if effective_gr_min <= actual_gr <= effective_gr_max:
                break
            if actual_gr < effective_gr_min:
                threshold_db -= 2.0
            else:
                threshold_db += 2.0

        # Apply final compression
        compressor = Compressor(
            threshold_db=best_threshold_db,
            ratio=ratio,
            attack_ms=attack_ms,
            release_ms=release_ms,
        )
        y_compressed = compressor(y, sr)

        # Peak protection INSIDE compressor chain — catch clips from compressor
        y_compressed = self._peak_protect(y_compressed, ceiling_db=-3.0)

        # Makeup gain = threshold × (1 - 1/ratio) × 0.5
        # Reduced to 50% to avoid over-driving (was 80% — too aggressive)
        makeup_db = -(best_threshold_db * (1.0 - 1.0 / ratio)) * 0.5
        makeup_db = np.clip(makeup_db, 0.0, 8.0)
        y_out = y_compressed * (10.0 ** (makeup_db / 20.0))

        # Final peak protection
        y_out = self._peak_protect(y_out, ceiling_db=-1.0)

        # Verify actual GR
        output_rms = self._rms(y_out)
        actual_gr = self._gr_from_rms(input_rms, output_rms)

        return {
            'y': y_out.astype(np.float32),
            'gr_achieved': best_gr,
            'makeup_db': makeup_db,
            'threshold_db': best_threshold_db,
            'ratio': ratio,
            'attack_ms': attack_ms,
            'release_ms': release_ms,
        }

    # ── Reverb ───────────────────────────────────────────────────────

    def reverb(self, y: np.ndarray, sr: int, category: str = 'drum') -> np.ndarray:
        """Pedalboard algorithmic reverb with category presets. Returns wet-only signal."""
        y = self._ensure_stereo(y)
        preset = config.REVERB_CATEGORIES.get(category, config.REVERB_CATEGORIES['drum'])
        rvb = Reverb(
            room_size=preset['room_size'],
            damping=preset['damping'],
            wet_level=preset['wet_level'],
            dry_level=preset['dry_level'],
            width=preset['width'],
        )
        return rvb(y, sr).astype(np.float32)

    # ── Delay ────────────────────────────────────────────────────────

    def delay(self, y: np.ndarray, sr: int, bpm: float,
              delay_type: str = 'melodic') -> np.ndarray:
        """BPM-synced delay. Returns wet-only signal."""
        y = self._ensure_stereo(y)
        preset = config.DELAY_PRESETS.get(delay_type, config.DELAY_PRESETS['melodic'])
        # Dotted eighth note
        q_ms = 60000.0 / bpm
        delay_sec = (q_ms * 0.75) / 1000.0
        dly = Delay(
            delay_seconds=delay_sec,
            feedback=preset['feedback'],
            mix=preset['mix'],
        )
        return dly(y, sr).astype(np.float32)

    # ── Chorus ───────────────────────────────────────────────────────

    def chorus(self, y: np.ndarray, sr: int,
               rate_hz: float = 1.5, depth: float = 0.25) -> np.ndarray:
        """Pedalboard chorus."""
        y = self._ensure_stereo(y)
        ch = Chorus(
            rate_hz=rate_hz,
            depth=depth,
            centre_delay_ms=7.0,
            feedback=0.0,
            mix=0.5,
        )
        return ch(y, sr).astype(np.float32)

    # ── Phaser ───────────────────────────────────────────────────────

    def phaser(self, y: np.ndarray, sr: int,
               rate_hz: float = 0.5, depth: float = 0.5) -> np.ndarray:
        """Pedalboard phaser."""
        y = self._ensure_stereo(y)
        ph = Phaser(
            rate_hz=rate_hz,
            depth=depth,
            centre_frequency_hz=1300.0,
            feedback=0.0,
            mix=0.5,
        )
        return ph(y, sr).astype(np.float32)

    # ── Saturation ───────────────────────────────────────────────────

    def saturate(self, y: np.ndarray, role: str = 'default',
                 amount_override: float = None) -> np.ndarray:
        """Tanh waveshaping with oversampling. Subtle per-role amounts."""
        amount = amount_override if amount_override is not None else config.SATURATION_PRESETS.get(role, config.SATURATION_PRESETS['default'])
        y = self._ensure_stereo(y)
        y64 = y.astype(np.float64)
        saturated = np.tanh(y64 * amount) / np.tanh(amount)
        return saturated.astype(np.float32)

    # ── Soft Clipping ────────────────────────────────────────────────

    def soft_clip(self, y: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
        """Tanh soft clipping to catch peaks before limiter."""
        y = self._ensure_stereo(y)
        ceiling = 10.0 ** (ceiling_db / 20.0)
        y64 = y.astype(np.float64)
        # Soft clip: tanh waveshaping scaled to ceiling
        clipped = np.tanh(y64 / ceiling) * ceiling
        return clipped.astype(np.float32)

    def parallel_compress(self, y: np.ndarray, sr: int,
                          ratio: float = 10.0, attack_ms: float = 2.0,
                          release_ms: float = 40.0, blend: float = 0.25) -> np.ndarray:
        """
        NY-style parallel compression: heavy compression blended with dry.
        Raises average level without squashing transients.
        """
        y = self._ensure_stereo(y)
        compressor = Compressor(
            threshold_db=-20.0,
            ratio=ratio,
            attack_ms=attack_ms,
            release_ms=release_ms,
        )
        crush = compressor(y, sr)
        # Peak protect the crushed signal
        crush = self._peak_protect(crush, ceiling_db=-3.0)
        # Blend
        out = y * (1.0 - blend) + crush * blend
        return self._peak_protect(out, ceiling_db=-1.0).astype(np.float32)

    def soft_clip_bus(self, y: np.ndarray, ceiling_db: float = -3.0) -> np.ndarray:
        """Soft clip a bus signal to tame transient peaks."""
        return self.soft_clip(y, ceiling_db)

    def dynamic_soft_clip(self, y: np.ndarray, sr: int,
                           headroom_db: float = 5.0, block_ms: float = 25) -> np.ndarray:
        """
        Dynamic soft clipping — ceiling follows signal's local peak.

        Divides signal into blocks, measures peak per block,
        sets ceiling at peak - headroom_db, applies tanh waveshaping.
        """
        y = self._ensure_stereo(y)
        block_size = max(1, int(sr * block_ms / 1000))
        n_blocks = max(1, len(y) // block_size)
        out = np.zeros_like(y, dtype=np.float64)

        for i in range(n_blocks):
            start = i * block_size
            end = min(start + block_size, len(y))
            block = y[start:end].astype(np.float64)

            block_peak = np.max(np.abs(block))
            if block_peak < 1e-10:
                out[start:end] = block
                continue

            block_peak_db = 20.0 * np.log10(block_peak)
            ceiling_db = block_peak_db - headroom_db
            ceiling = 10.0 ** (ceiling_db / 20.0)

            if block_peak > ceiling:
                clipped = np.tanh(block / ceiling) * ceiling
                out[start:end] = clipped
            else:
                out[start:end] = block

        return out.astype(np.float32)

    def kick_bass_sidechain(self, bass_y: np.ndarray, kick_y: np.ndarray,
                             sr: int, depth_db: float = 3.0,
                             release_ms: float = 20.0,
                             threshold_db: float = -30.0,
                             freq_range: tuple = (40, 120)) -> np.ndarray:
        """
        Kick-triggered bass ducking.

        Detects kick transients in kick_y, ducks bass_y in freq_range
        proportionally. Mild depth, 20ms release (no pumping).
        """
        bass_y = self._ensure_stereo(bass_y)
        kick_y = self._ensure_stereo(kick_y)

        kick_mono = np.mean(kick_y, axis=1) if kick_y.ndim > 1 else kick_y
        kick_env = np.abs(kick_mono)

        release_coeff = np.exp(-1.0 / (sr * release_ms / 1000.0))
        smoothed = np.zeros_like(kick_env)
        prev = 0.0
        for i in range(len(kick_env)):
            c = 0.99 if kick_env[i] > prev else release_coeff
            smoothed[i] = (1 - c) * kick_env[i] + c * prev
            prev = smoothed[i]

        threshold = 10.0 ** (threshold_db / 20.0)
        depth_lin = 10.0 ** (depth_db / 20.0)

        gr = np.ones(len(smoothed))
        mask = smoothed > threshold
        gr[mask] = 1.0 / (1.0 + (smoothed[mask] / threshold - 1.0) * (depth_lin - 1.0))
        gr = np.clip(gr, 1.0 / depth_lin, 1.0)

        bass_low = self._bandpass(bass_y, sr, freq_range[0], freq_range[1])
        bass_high = bass_y - bass_low

        # Truncate gr to match bass length (kick and bass buses may differ in length)
        gr = gr[:len(bass_y)]

        if bass_low.ndim == 2:
            gr = gr[:, np.newaxis]
        bass_low_ducked = bass_low * gr

        return (bass_low_ducked + bass_high).astype(np.float32)

    def _overlap_add_process(self, y: np.ndarray, sr: int,
                              process_fn, block_size: int = 4096,
                              overlap: float = 0.5) -> np.ndarray:
        """
        Overlap-add processing to eliminate block boundary discontinuities.
        Each block overlaps the previous by 50%, multiplied by a Hanning window.
        Overlapping windows sum to unity — no gain change, no discontinuities.
        """
        y = self._ensure_stereo(y)
        hop = int(block_size * (1 - overlap))
        n_samples = len(y)
        n_channels = y.shape[1] if y.ndim > 1 else 1
        n_blocks = max(1, (n_samples - block_size) // hop + 1)

        out = np.zeros((n_samples + block_size, n_channels), dtype=np.float64)
        norm = np.zeros((n_samples + block_size, 1), dtype=np.float64)
        window = np.hanning(block_size).astype(np.float64).reshape(-1, 1)

        for i in range(n_blocks):
            start = i * hop
            end = min(start + block_size, n_samples)
            actual_len = end - start
            block = y[start:end].astype(np.float64)
            processed = process_fn(block.astype(np.float32), sr).astype(np.float64)
            out[start:end] += processed[:actual_len] * window[:actual_len]
            norm[start:end] += window[:actual_len]

        # Normalize by overlap count
        norm = np.maximum(norm, 1e-10)
        result = out[:n_samples] / norm[:n_samples]
        return result.astype(np.float32)

    def serial_soft_clip(self, y: np.ndarray, passes: int = 3,
                         ceiling_db: float = -1.0) -> np.ndarray:
        """Multiple soft clip passes for gradual peak reduction. Less aggressive than single pass."""
        y = self._ensure_stereo(y)
        for _ in range(passes):
            y = self.soft_clip(y, ceiling_db)
        return y

    def multiband_compress(self, y: np.ndarray, sr: int,
                           bands: list = None) -> np.ndarray:
        """
        Simple 3-band compression to tame transients before limiting.
        Splits into sub/mid/high, compresses each, sums back.
        """
        y = self._ensure_stereo(y)
        if bands is None:
            bands = [
                {'name': 'sub',   'low': 20,    'high': 200,   'threshold_db': -18, 'ratio': 2.0, 'attack_ms': 20, 'release_ms': 150},
                {'name': 'mid',   'low': 200,   'high': 4000,  'threshold_db': -15, 'ratio': 2.5, 'attack_ms': 10, 'release_ms': 120},
                {'name': 'high',  'low': 4000,  'high': 20000, 'threshold_db': -18, 'ratio': 2.0, 'attack_ms': 5,  'release_ms': 80},
            ]

        out = np.zeros_like(y, dtype=np.float64)
        for band in bands:
            # Extract band
            band_y = self._bandpass(y, sr, band['low'], band['high'])
            # Compress
            comp_result = self.adaptive_compress(
                band_y.astype(np.float32), sr, 'default'
            )
            out += comp_result['y'].astype(np.float64)

        # Peak protect
        peak = np.max(np.abs(out))
        if peak > 0.98:
            out *= 0.95 / peak

        return out.astype(np.float32)

    # ── Brick-Wall Limiter ───────────────────────────────────────────

    def limit(self, y: np.ndarray, sr: int,
              threshold_db: float = -1.0) -> np.ndarray:
        """Pedalboard brick-wall limiter."""
        y = self._ensure_stereo(y)
        limiter = Limiter(threshold_db=threshold_db, release_ms=100.0)
        return limiter(y, sr).astype(np.float32)

    def brick_wall_limit(self, y: np.ndarray, sr: int,
                         ceiling_db: float = -1.0,
                         lookahead_ms: float = 5.0,
                         release_ms: float = 50.0) -> np.ndarray:
        """
        Custom brick-wall limiter with lookahead. No pops, no discontinuities.
        Reads ahead to detect peaks before they arrive, applies smooth gain reduction.
        """
        y = self._ensure_stereo(y)
        ceiling = 10.0 ** (ceiling_db / 20.0)
        lookahead = int(sr * lookahead_ms / 1000.0)

        # Envelope: max of absolute value across channels
        mono = np.max(np.abs(y.astype(np.float64)), axis=1)

        # Lookahead envelope: max of next N samples
        env = np.copy(mono)
        for i in range(len(mono) - 1, -1, -1):
            end = min(i + lookahead, len(mono))
            env[i] = np.max(mono[i:end])

        # Gain reduction: where env > ceiling, reduce
        gain = np.ones(len(mono), dtype=np.float64)
        mask = env > ceiling
        gain[mask] = ceiling / np.maximum(env[mask], 1e-10)

        # Smooth gain: instant attack, smooth release
        release_coef = np.exp(-1.0 / (sr * release_ms / 1000.0))
        smoothed = np.zeros_like(gain)
        smoothed[0] = gain[0]
        for i in range(1, len(gain)):
            if gain[i] < smoothed[i - 1]:
                smoothed[i] = gain[i]  # Instant attack
            else:
                smoothed[i] = smoothed[i - 1] + (1.0 - release_coef) * (gain[i] - smoothed[i - 1])

        # Compensate for lookahead: shift gain back
        gain_compensated = np.ones(len(mono), dtype=np.float64)
        if lookahead > 0 and lookahead < len(smoothed):
            gain_compensated[:len(smoothed) - lookahead] = smoothed[lookahead:]

        return (y * gain_compensated[:, np.newaxis]).astype(np.float32)

    # ── Reference-Matched Gain ───────────────────────────────────────

    def reference_matched_gain(self, y: np.ndarray, sr: int,
                                reference_lufs: float = -10.2,
                                max_correction_db: float = 6.0) -> tuple:
        """
        Small corrective gain to match reference loudness.
        Unlike normalization, this preserves the relative balance.
        Returns (y_corrected, correction_db).
        """
        y = self._ensure_stereo(y)
        meter = pyln.Meter(sr)
        current_lufs = meter.integrated_loudness(y)
        if current_lufs <= -70.0 or not np.isfinite(current_lufs):
            return y, 0.0

        delta = reference_lufs - current_lufs
        correction = float(np.clip(delta, -max_correction_db, max_correction_db))

        if abs(correction) < 0.1:
            return y, 0.0

        y_out = y * (10.0 ** (correction / 20.0))
        return y_out.astype(np.float32), correction

    # ── Automation Effects ───────────────────────────────────────────

    def filter_sweep(self, y: np.ndarray, sr: int,
                     start_hz: float, end_hz: float,
                     filter_type: str = 'lowpass') -> np.ndarray:
        """
        Automated filter cutoff sweep across the signal.
        Processes in 1024-sample blocks with smoothly varying cutoff.
        """
        y = self._ensure_stereo(y)
        block_size = 1024
        n_blocks = len(y) // block_size
        if n_blocks < 1:
            return y

        cutoffs = np.geomspace(max(start_hz, 20.0), min(end_hz, sr * 0.45), n_blocks)
        out = np.zeros_like(y, dtype=np.float64)

        for i in range(n_blocks):
            block = y[i * block_size:(i + 1) * block_size].astype(np.float64)
            cutoff = float(cutoffs[i])

            if filter_type == 'lowpass':
                alpha = (2.0 * np.pi * cutoff) / (2.0 * np.pi * cutoff + sr)
                # Simple IIR lowpass per channel
                for ch in range(block.shape[1]):
                    state = block[0, ch]
                    for j in range(1, len(block)):
                        state = state + alpha * (block[j, ch] - state)
                        block[j, ch] = state
            elif filter_type == 'highpass':
                # High = original - lowpass
                low_block = block.copy()
                alpha = (2.0 * np.pi * cutoff) / (2.0 * np.pi * cutoff + sr)
                for ch in range(low_block.shape[1]):
                    state = low_block[0, ch]
                    for j in range(1, len(low_block)):
                        state = state + alpha * (low_block[j, ch] - state)
                        low_block[j, ch] = state
                block = block - low_block

            out[i * block_size:(i + 1) * block_size] = block

        return out.astype(np.float32)

    def reverb_wash(self, y: np.ndarray, sr: int,
                    start_wet: float = 0.0, end_wet: float = 0.8,
                    room_size: float = 0.7) -> np.ndarray:
        """
        Reverb wash: process entire signal through ONE reverb, blend wet/dry with gain curve.
        No block boundaries — natural reverb tail preserved.
        """
        y = self._ensure_stereo(y)
        # Process entire signal through one reverb at full wet
        rvb = Reverb(room_size=room_size, damping=0.5, wet_level=1.0, dry_level=0.0, width=1.0)
        wet_signal = rvb(y, sr).astype(np.float32)
        # Blend wet/dry using gain curve
        wet_curve = np.linspace(start_wet, end_wet, len(y), dtype=np.float32)
        out = y * (1.0 - wet_curve[:, np.newaxis]) + wet_signal * wet_curve[:, np.newaxis]
        return out.astype(np.float32)

    def delay_feedback_swell(self, y: np.ndarray, sr: int,
                              delay_sec: float = 0.375,
                              start_feedback: float = 0.1,
                              end_feedback: float = 0.6,
                              start_mix: float = 0.1,
                              end_mix: float = 0.4) -> np.ndarray:
        """Automated delay feedback and mix swell using overlap-add."""
        y = self._ensure_stereo(y)
        block_size = 4096
        n_blocks = max(1, len(y) // block_size)
        feedbacks = np.linspace(start_feedback, end_feedback, n_blocks)
        mixes = np.linspace(start_mix, end_mix, n_blocks)

        def process_block(block, sr, idx=0):
            i = min(idx, n_blocks - 1)
            dly = Delay(delay_seconds=delay_sec, feedback=float(feedbacks[i]), mix=float(mixes[i]))
            return dly(block, sr)

        hop = int(block_size * 0.5)
        n_samples = len(y)
        n_channels = y.shape[1] if y.ndim > 1 else 1
        n_blocks_oa = max(1, (n_samples - block_size) // hop + 1)
        out = np.zeros((n_samples + block_size, n_channels), dtype=np.float64)
        norm = np.zeros((n_samples + block_size, 1), dtype=np.float64)
        window = np.hanning(block_size).astype(np.float64).reshape(-1, 1)
        for i in range(n_blocks_oa):
            start = i * hop
            end = min(start + block_size, n_samples)
            actual_len = end - start
            block = y[start:end]
            idx = int(i * n_blocks / max(n_blocks_oa, 1))
            processed = process_block(block, sr, idx)
            out[start:end] += processed[:actual_len].astype(np.float64) * window[:actual_len]
            norm[start:end] += window[:actual_len]
        norm = np.maximum(norm, 1e-10)
        return (out[:n_samples] / norm[:n_samples]).astype(np.float32)

    def chorus_swell(self, y: np.ndarray, sr: int,
                     start_depth: float = 0.0, end_depth: float = 0.4,
                     rate_hz: float = 1.5) -> np.ndarray:
        """Automated chorus depth swell."""
        y = self._ensure_stereo(y)
        block_size = 4096
        n_blocks = max(1, len(y) // block_size)
        depths = np.linspace(start_depth, end_depth, n_blocks)

        def process_block(block, sr, idx=0):
            i = min(idx, n_blocks - 1)
            ch = Chorus(rate_hz=rate_hz, depth=float(depths[i]), centre_delay_ms=7.0, feedback=0.0, mix=0.5)
            return ch(block, sr)

        hop = int(block_size * 0.5)
        n_samples = len(y)
        n_channels = y.shape[1] if y.ndim > 1 else 1
        n_blocks_oa = max(1, (n_samples - block_size) // hop + 1)
        out = np.zeros((n_samples + block_size, n_channels), dtype=np.float64)
        norm = np.zeros((n_samples + block_size, 1), dtype=np.float64)
        window = np.hanning(block_size).astype(np.float64).reshape(-1, 1)
        for i in range(n_blocks_oa):
            start = i * hop
            end = min(start + block_size, n_samples)
            actual_len = end - start
            block = y[start:end]
            idx = int(i * n_blocks / max(n_blocks_oa, 1))
            processed = process_block(block, sr, idx)
            out[start:end] += processed[:actual_len].astype(np.float64) * window[:actual_len]
            norm[start:end] += window[:actual_len]
        norm = np.maximum(norm, 1e-10)
        return (out[:n_samples] / norm[:n_samples]).astype(np.float32)

    def gain_automation(self, y: np.ndarray, gain_curve: np.ndarray) -> np.ndarray:
        """Apply a time-varying gain curve. gain_curve should be 1D, same length as y."""
        y = self._ensure_stereo(y)
        if len(gain_curve) < len(y):
            gain_curve = np.pad(gain_curve, (0, len(y) - len(gain_curve)), constant_values=1.0)
        elif len(gain_curve) > len(y):
            gain_curve = gain_curve[:len(y)]
        return (y * gain_curve[:, np.newaxis]).astype(np.float32)

    def stereo_width_automation(self, y: np.ndarray, width_curve: np.ndarray) -> np.ndarray:
        """Apply time-varying stereo width. width_curve: 1D, 0=mono, 1=normal, >1=wider."""
        y = self._ensure_stereo(y)
        if y.ndim < 2 or y.shape[1] < 2:
            return y
        if len(width_curve) < len(y):
            width_curve = np.pad(width_curve, (0, len(y) - len(width_curve)), constant_values=1.0)
        elif len(width_curve) > len(y):
            width_curve = width_curve[:len(y)]

        mid = (y[:, 0] + y[:, 1]) * 0.5
        side = (y[:, 0] - y[:, 1]) * 0.5
        side = side * width_curve
        out = np.stack([mid + side, mid - side], axis=1)
        peak = np.max(np.abs(out))
        if peak > 0.98:
            out *= 0.95 / peak
        return out.astype(np.float32)

    def auto_pan(self, y: np.ndarray, sr: int, bpm: float,
                 pan_range: float = 0.05, rate_triplets: int = 2,
                 irregularity: float = 0.6, seed: int = None) -> np.ndarray:
        """
        Musical auto-panning with irregular rhythm.

        Uses a random-walk pattern at triplet-note intervals for organic,
        non-mechanical stereo movement. Constant-power panning law.

        Args:
            y: stereo audio (N x 2)
            sr: sample rate
            bpm: song tempo
            pan_range: max pan offset (0.0-1.0, where 0.05 = ±5% of stereo field)
            rate_triplets: rate in triplet eighth notes (2 = two triplets per step)
            irregularity: 0.0 = smooth sine, 1.0 = fully random walk
            seed: random seed for reproducibility per stem
        """
        y = self._ensure_stereo(y)
        if y.ndim < 2 or y.shape[1] < 2:
            return y

        rng = np.random.RandomState(seed)

        # Duration of one triplet eighth note
        triplet_sec = (60.0 / bpm) / 3.0
        step_sec = triplet_sec * rate_triplets
        step_samples = max(1, int(sr * step_sec))

        n_steps = max(1, len(y) // step_samples + 1)

        # Generate pan positions via random walk
        pan_positions = np.zeros(n_steps)
        pan_positions[0] = rng.uniform(-pan_range * 0.5, pan_range * 0.5)

        for i in range(1, n_steps):
            if rng.random() < irregularity:
                # Random step
                step = rng.uniform(-pan_range * 0.4, pan_range * 0.4)
            else:
                # Continue in same direction with decay
                step = pan_positions[i - 1] * 0.6

            pan_positions[i] = np.clip(pan_positions[i - 1] + step, -pan_range, pan_range)

        # Smooth transitions between steps (crossfade)
        pan_curve = np.zeros(len(y))
        for i in range(n_steps):
            start = i * step_samples
            end = min(start + step_samples, len(y))
            if start >= len(y):
                break
            # Linear interpolation between steps
            if i < n_steps - 1:
                t = np.linspace(0, 1, end - start, dtype=np.float64)
                pan_curve[start:end] = pan_positions[i] * (1 - t) + pan_positions[i + 1] * t
            else:
                pan_curve[start:end] = pan_positions[i]

        # Apply constant-power panning law
        # pan: -1.0 = full left, 0.0 = center, +1.0 = full right
        # theta: 0 = full left, pi/2 = center, pi = full right
        theta = (pan_curve + 1.0) * (np.pi / 4.0)  # map -1..+1 to 0..pi/2
        l_gain = np.cos(theta)
        r_gain = np.sin(theta)

        out = np.stack([y[:, 0] * l_gain, y[:, 1] * r_gain], axis=1)

        # Peak protection
        peak = np.max(np.abs(out))
        if peak > 0.98:
            out *= 0.95 / peak

        return out.astype(np.float32)

    def riser(self, y: np.ndarray, sr: int, start_sample: int, end_sample: int,
              bpm: float) -> np.ndarray:
        """
        Build tension: filter sweep up + gain rise + reverb build.
        Applied to the segment between start_sample and end_sample.
        """
        y = self._ensure_stereo(y)
        out = y.copy()
        seg = out[start_sample:end_sample].copy()
        if len(seg) < sr // 10:
            return out

        # 1. Filter sweep: 800Hz → 12kHz
        seg = self.filter_sweep(seg, sr, 800.0, 12000.0, 'lowpass')

        # 2. Gain rise: +3dB over the segment
        gain_curve = np.linspace(0.0, 3.0, len(seg), dtype=np.float32)
        seg = seg * (10.0 ** (gain_curve[:, np.newaxis] / 20.0))

        # 3. Reverb build
        seg = self.reverb_wash(seg, sr, start_wet=0.0, end_wet=0.3, room_size=0.6)

        out[start_sample:end_sample] = self._peak_protect(seg, -1.0)
        return out

    def downlifter(self, y: np.ndarray, sr: int, start_sample: int, end_sample: int) -> np.ndarray:
        """
        Release tension: filter sweep down + reverb tail.
        """
        y = self._ensure_stereo(y)
        out = y.copy()
        seg = out[start_sample:end_sample].copy()
        if len(seg) < sr // 10:
            return out

        # 1. Filter sweep: 12kHz → 400Hz
        seg = self.filter_sweep(seg, sr, 12000.0, 400.0, 'lowpass')

        # 2. Gain fade: -6dB
        gain_curve = np.linspace(0.0, -6.0, len(seg), dtype=np.float32)
        seg = seg * (10.0 ** (gain_curve[:, np.newaxis] / 20.0))

        # 3. Reverb wash out
        seg = self.reverb_wash(seg, sr, start_wet=0.1, end_wet=0.5, room_size=0.7)

        out[start_sample:end_sample] = seg
        return out

    def impact(self, y: np.ndarray, sr: int, sample_pos: int,
               duration_sec: float = 0.5) -> np.ndarray:
        """
        Transient boost + sub boom at a specific moment.
        """
        y = self._ensure_stereo(y)
        out = y.copy()
        half = int(duration_sec * sr / 2)
        start = max(0, sample_pos - half)
        end = min(len(out), sample_pos + half)
        seg = out[start:end].copy()

        if len(seg) < sr // 20:
            return out

        # Transient boost: +4dB attack, decay over segment
        boost_curve = np.exp(-np.linspace(0, 5, len(seg))) * 4.0
        seg = seg * (10.0 ** (boost_curve[:, np.newaxis] / 20.0))

        # Sub boom: add a short low-frequency burst
        t = np.linspace(0, duration_sec, len(seg), dtype=np.float32)
        sub = np.sin(2 * np.pi * 45 * t) * 0.15 * np.exp(-t * 8.0)
        sub_stereo = np.stack([sub, sub], axis=1)
        seg = seg + sub_stereo

        out[start:end] = self._peak_protect(seg, -0.5)
        return out

    def silence_drop(self, y: np.ndarray, sr: int, start_sample: int,
                     duration_samples: int) -> np.ndarray:
        """Mute a section of audio (silence drop before impact)."""
        y = self._ensure_stereo(y)
        out = y.copy()
        end = min(len(out), start_sample + duration_samples)

        # Apply short fade out and fade in to avoid clicks
        fade_len = min(int(sr * 0.01), (end - start_sample) // 4)  # 10ms fade
        if fade_len > 0:
            fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            out[start_sample:start_sample + fade_len] *= fade_out[:, np.newaxis]
            out[end - fade_len:end] *= fade_in[:, np.newaxis]

        out[start_sample + fade_len:end - fade_len] = 0.0
        return out

    # ── Tape/Vinyl Effects ───────────────────────────────────────────

    def tape_stop(self, y: np.ndarray, sr: int, duration_sec: float = 1.5) -> np.ndarray:
        """Tape stop: pitch drops, speed slows, signal fades."""
        y = self._ensure_stereo(y)
        stop_len = min(int(sr * duration_sec), len(y))
        if stop_len < sr // 10:
            return y
        out = y.copy()
        seg = out[-stop_len:].copy()
        src_idx = np.linspace(0, stop_len - 1, stop_len)
        curve = np.cumsum(np.linspace(1.0, 0.08, stop_len))
        curve = curve / curve[-1] * (stop_len - 1)
        stopped = np.zeros_like(seg)
        for ch in range(seg.shape[1]):
            stopped[:, ch] = np.interp(src_idx, curve, seg[:, ch], left=seg[0, ch], right=0.0)
        amp = np.linspace(1.0, 0.0, stop_len, dtype=np.float32)[:, None]
        out[-stop_len:] = stopped * amp
        return out.astype(np.float32)

    def vinyl_stop(self, y: np.ndarray, sr: int, duration_sec: float = 2.0) -> np.ndarray:
        """Vinyl stop: tape stop + wow/flutter + crackle."""
        y = self._ensure_stereo(y)
        stop_len = min(int(sr * duration_sec), len(y))
        if stop_len < sr // 10:
            return y
        out = y.copy()
        seg = out[-stop_len:].copy()
        t = np.linspace(0, duration_sec, stop_len, dtype=np.float32)
        wow = np.sin(2 * np.pi * 0.5 * t) * 0.003
        flutter = np.sin(2 * np.pi * 6.0 * t) * 0.001
        pitch_mod = 1.0 + wow + flutter
        src_idx = np.cumsum(1.0 / pitch_mod)
        src_idx = src_idx / src_idx[-1] * (stop_len - 1)
        stopped = np.zeros_like(seg)
        flat_idx = np.linspace(0, stop_len - 1, stop_len)
        for ch in range(seg.shape[1]):
            stopped[:, ch] = np.interp(src_idx, flat_idx, seg[:, ch], left=seg[0, ch], right=0.0)
        crackle = np.random.randn(stop_len).astype(np.float32) * 0.02 * np.linspace(0, 1, stop_len)
        stopped += crackle[:, None]
        amp = np.linspace(1.0, 0.0, stop_len, dtype=np.float32)[:, None]
        out[-stop_len:] = stopped * amp
        return out.astype(np.float32)

    def tape_start(self, y: np.ndarray, sr: int, duration_sec: float = 1.5) -> np.ndarray:
        """Tape start: signal speeds up from slow to normal."""
        y = self._ensure_stereo(y)
        start_len = min(int(sr * duration_sec), len(y))
        if start_len < sr // 10:
            return y
        out = y.copy()
        seg = out[:start_len].copy()
        src_idx = np.linspace(0, start_len - 1, start_len)
        curve = np.cumsum(np.linspace(0.08, 1.0, start_len))
        curve = curve / curve[-1] * (start_len - 1)
        started = np.zeros_like(seg)
        for ch in range(seg.shape[1]):
            started[:, ch] = np.interp(src_idx, curve, seg[:, ch], left=0.0, right=seg[-1, ch])
        amp = np.linspace(0.0, 1.0, start_len, dtype=np.float32)[:, None]
        out[:start_len] = started * amp
        return out.astype(np.float32)

    def tape_wobble(self, y: np.ndarray, sr: int, depth: float = 0.003) -> np.ndarray:
        """Tape wobble: wow/flutter pitch wobble without stopping."""
        y = self._ensure_stereo(y)
        t = np.linspace(0, len(y) / sr, len(y), dtype=np.float32)
        wow = np.sin(2 * np.pi * 0.5 * t) * depth
        flutter = np.sin(2 * np.pi * 6.0 * t) * depth * 0.3
        pitch_mod = 1.0 + wow + flutter
        src_idx = np.cumsum(1.0 / pitch_mod)
        src_idx = src_idx / src_idx[-1] * (len(y) - 1)
        flat_idx = np.linspace(0, len(y) - 1, len(y))
        out = np.zeros_like(y)
        for ch in range(y.shape[1]):
            out[:, ch] = np.interp(src_idx, flat_idx, y[:, ch])
        return out.astype(np.float32)

    def vinyl_crackle(self, y: np.ndarray, sr: int, amount: float = 0.02) -> np.ndarray:
        """Add vinyl crackle noise overlay."""
        y = self._ensure_stereo(y)
        crackle = np.random.randn(len(y)).astype(np.float32) * amount
        # Make crackle sparse (random spikes)
        mask = np.random.random(len(y)) > 0.98
        crackle *= mask
        return (y + crackle[:, None]).astype(np.float32)

    # ── Rhythmic Effects ─────────────────────────────────────────────

    def sidechain_pump(self, y: np.ndarray, sr: int, bpm: float,
                       depth: float = 0.8, release_ms: float = 200.0) -> np.ndarray:
        """Rhythmic volume ducking synced to BPM."""
        y = self._ensure_stereo(y)
        beat_samples = int(60.0 / bpm * sr)
        phase = np.mod(np.arange(len(y)), beat_samples) / float(beat_samples)
        pump = 1.0 - depth * np.exp(-phase * (1000.0 / release_ms))
        return (y * pump[:, np.newaxis]).astype(np.float32)

    def beat_repeat(self, y: np.ndarray, sr: int, bpm: float,
                    start_sample: int, repeats: int = 4,
                    division: float = 0.25) -> np.ndarray:
        """Repeat a short segment (1/4, 1/8 note) multiple times."""
        y = self._ensure_stereo(y)
        beat_len = int(60.0 / bpm * sr * division)
        out = y.copy()
        seg = out[start_sample:start_sample + beat_len].copy()
        if len(seg) < 10:
            return out
        for i in range(repeats):
            pos = start_sample + (i + 1) * beat_len
            if pos + beat_len > len(out):
                break
            # Fade to avoid clicks
            fade = min(int(sr * 0.005), beat_len // 4)
            out[pos:pos + beat_len] = seg
            if fade > 0 and i > 0:
                out[pos:pos + fade] *= np.linspace(0, 1, fade, dtype=np.float32)[:, None]
        return out.astype(np.float32)

    def stutter(self, y: np.ndarray, sr: int, start_sample: int,
                stutter_len: int = 2048, repeats: int = 8) -> np.ndarray:
        """Rapid repeats of a short buffer."""
        y = self._ensure_stereo(y)
        out = y.copy()
        seg = out[start_sample:start_sample + stutter_len].copy()
        if len(seg) < 10:
            return out
        fade = min(int(sr * 0.002), stutter_len // 4)
        for i in range(repeats):
            pos = start_sample + i * stutter_len
            if pos + stutter_len > len(out):
                break
            out[pos:pos + stutter_len] = seg
            if fade > 0:
                out[pos:pos + fade] *= np.linspace(0, 1, fade, dtype=np.float32)[:, None]
                out[pos + stutter_len - fade:pos + stutter_len] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
        return out.astype(np.float32)

    # ── Filter Effects ───────────────────────────────────────────────

    def ladder_filter_sweep(self, y: np.ndarray, sr: int,
                            start_hz: float, end_hz: float,
                            mode: str = 'lpf', resonance: float = 0.5) -> np.ndarray:
        """LadderFilter cutoff sweep using overlap-add to eliminate block boundary pops."""
        y = self._ensure_stereo(y)

        mode_map = {
            'lpf': LadderFilter.Mode.LPF24,
            'hpf': LadderFilter.Mode.HPF24,
            'bpf': LadderFilter.Mode.BPF24,
        }
        ladder_mode = mode_map.get(mode, LadderFilter.Mode.LPF24)

        block_size = 4096
        n_blocks = max(1, len(y) // block_size)
        cutoffs = np.geomspace(max(start_hz, 20.0), min(end_hz, sr * 0.45), n_blocks)

        def process_block(block, sr, cutoff_idx=0):
            cutoff = float(cutoffs[min(cutoff_idx, len(cutoffs) - 1)])
            lf = LadderFilter(mode=ladder_mode, cutoff_hz=cutoff, resonance=resonance, drive=1.0)
            return lf(block, sr)

        # Overlap-add with cutoff varying per block
        hop = int(block_size * 0.5)
        n_samples = len(y)
        n_channels = y.shape[1] if y.ndim > 1 else 1
        n_blocks_oa = max(1, (n_samples - block_size) // hop + 1)
        out = np.zeros((n_samples + block_size, n_channels), dtype=np.float64)
        norm = np.zeros((n_samples + block_size, 1), dtype=np.float64)
        window = np.hanning(block_size).astype(np.float64).reshape(-1, 1)

        for i in range(n_blocks_oa):
            start = i * hop
            end = min(start + block_size, n_samples)
            actual_len = end - start
            block = y[start:end]
            # Map this block to the nearest cutoff index
            cutoff_idx = int(i * len(cutoffs) / max(n_blocks_oa, 1))
            processed = process_block(block, sr, cutoff_idx)
            out[start:end] += processed[:actual_len].astype(np.float64) * window[:actual_len]
            norm[start:end] += window[:actual_len]

        norm = np.maximum(norm, 1e-10)
        return (out[:n_samples] / norm[:n_samples]).astype(np.float32)

    def filter_drop(self, y: np.ndarray, sr: int, start_sample: int,
                    drop_hz: float = 200.0, hold_sec: float = 0.5) -> np.ndarray:
        """LPF drops from high to low, holds, then snaps back."""
        y = self._ensure_stereo(y)
        out = y.copy()
        hold_len = int(hold_sec * sr)
        end_sample = min(start_sample + hold_len, len(out))
        seg = out[start_sample:end_sample].copy()
        if len(seg) < sr // 20:
            return out
        # Sweep down
        seg = self.ladder_filter_sweep(seg, sr, 18000.0, drop_hz, 'lpf', 0.7)
        out[start_sample:end_sample] = seg
        return out.astype(np.float32)

    def filter_riser(self, y: np.ndarray, sr: int, start_sample: int,
                     end_sample: int) -> np.ndarray:
        """LPF sweep up + gain build for riser effect."""
        y = self._ensure_stereo(y)
        out = y.copy()
        seg = out[start_sample:end_sample].copy()
        if len(seg) < sr // 10:
            return out
        seg = self.ladder_filter_sweep(seg, sr, 400.0, 18000.0, 'lpf', 0.5)
        gain_curve = np.linspace(0.0, 3.0, len(seg), dtype=np.float32)
        seg = seg * (10.0 ** (gain_curve[:, np.newaxis] / 20.0))
        out[start_sample:end_sample] = self._peak_protect(seg, -1.0)
        return out

    def filter_downlifter(self, y: np.ndarray, sr: int, start_sample: int,
                          end_sample: int) -> np.ndarray:
        """LPF sweep down + gain fade for downlifter."""
        y = self._ensure_stereo(y)
        out = y.copy()
        seg = out[start_sample:end_sample].copy()
        if len(seg) < sr // 10:
            return out
        seg = self.ladder_filter_sweep(seg, sr, 18000.0, 300.0, 'lpf', 0.5)
        gain_curve = np.linspace(0.0, -6.0, len(seg), dtype=np.float32)
        seg = seg * (10.0 ** (gain_curve[:, np.newaxis] / 20.0))
        out[start_sample:end_sample] = seg
        return out

    # ── Frequency Isolation ──────────────────────────────────────────

    def isolate_low(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Isolate low frequencies only (20-200Hz)."""
        y = self._ensure_stereo(y)
        return self._bandpass(y, sr, 20.0, 200.0)

    def isolate_mid(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Isolate mid frequencies only (200-4000Hz)."""
        y = self._ensure_stereo(y)
        return self._bandpass(y, sr, 200.0, 4000.0)

    def isolate_high(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Isolate high frequencies only (4000-20000Hz)."""
        y = self._ensure_stereo(y)
        return self._bandpass(y, sr, 4000.0, 20000.0)

    # ── Reverb Freeze ────────────────────────────────────────────────

    def reverb_freeze(self, y: np.ndarray, sr: int, wet_level: float = 0.6) -> np.ndarray:
        """Reverb with freeze_mode=1.0 — sustains the reverb tail forever."""
        y = self._ensure_stereo(y)
        rvb = Reverb(room_size=0.9, damping=0.3, wet_level=wet_level, dry_level=0.0,
                     width=1.0, freeze_mode=1.0)
        return rvb(y, sr).astype(np.float32)

    # ── Distortion Build ─────────────────────────────────────────────

    def distortion_build(self, y: np.ndarray, sr: int,
                         peak_drive_db: float = 20.0) -> np.ndarray:
        """Distortion drive ramps up then back down. Overlap-add for smooth transitions."""
        y = self._ensure_stereo(y)
        block_size = 4096
        n_blocks = max(1, len(y) // block_size)
        half = n_blocks // 2
        drives = np.concatenate([
            np.linspace(0, peak_drive_db, half),
            np.linspace(peak_drive_db, 0, n_blocks - half)
        ])

        def process_block(block, sr, idx=0):
            i = min(idx, len(drives) - 1)
            drive = float(drives[i])
            if drive < 0.5:
                return block
            d = Distortion(drive_db=drive)
            return d(block, sr)

        hop = int(block_size * 0.5)
        n_samples = len(y)
        n_channels = y.shape[1] if y.ndim > 1 else 1
        n_blocks_oa = max(1, (n_samples - block_size) // hop + 1)
        out = np.zeros((n_samples + block_size, n_channels), dtype=np.float64)
        norm = np.zeros((n_samples + block_size, 1), dtype=np.float64)
        window = np.hanning(block_size).astype(np.float64).reshape(-1, 1)
        for i in range(n_blocks_oa):
            start = i * hop
            end = min(start + block_size, n_samples)
            actual_len = end - start
            block = y[start:end]
            idx = int(i * n_blocks / max(n_blocks_oa, 1))
            processed = process_block(block, sr, idx)
            out[start:end] += processed[:actual_len].astype(np.float64) * window[:actual_len]
            norm[start:end] += window[:actual_len]
        norm = np.maximum(norm, 1e-10)
        return (out[:n_samples] / norm[:n_samples]).astype(np.float32)

    # ── Bitcrush Sweep ───────────────────────────────────────────────

    def bitcrush_sweep(self, y: np.ndarray, sr: int,
                       start_bits: int = 12, end_bits: int = 4) -> np.ndarray:
        """Bitcrush bit_depth sweeps from start to end. Overlap-add for smooth transitions."""
        y = self._ensure_stereo(y)
        block_size = 4096
        n_blocks = max(1, len(y) // block_size)
        bits_range = np.linspace(start_bits, end_bits, n_blocks)

        def process_block(block, sr, idx=0):
            i = min(idx, len(bits_range) - 1)
            bits = int(round(float(bits_range[i])))
            bits = max(4, min(16, bits))
            bc = Bitcrush(bit_depth=bits)
            return bc(block, sr)

        hop = int(block_size * 0.5)
        n_samples = len(y)
        n_channels = y.shape[1] if y.ndim > 1 else 1
        n_blocks_oa = max(1, (n_samples - block_size) // hop + 1)
        out = np.zeros((n_samples + block_size, n_channels), dtype=np.float64)
        norm = np.zeros((n_samples + block_size, 1), dtype=np.float64)
        window = np.hanning(block_size).astype(np.float64).reshape(-1, 1)
        for i in range(n_blocks_oa):
            start = i * hop
            end = min(start + block_size, n_samples)
            actual_len = end - start
            block = y[start:end]
            idx = int(i * n_blocks / max(n_blocks_oa, 1))
            processed = process_block(block, sr, idx)
            out[start:end] += processed[:actual_len].astype(np.float64) * window[:actual_len]
            norm[start:end] += window[:actual_len]
        norm = np.maximum(norm, 1e-10)
        return (out[:n_samples] / norm[:n_samples]).astype(np.float32)

    def upward_compress(self, y: np.ndarray, sr: int,
                        threshold_db: float = -30.0, ratio: float = 2.0) -> np.ndarray:
        """Boost quiet parts, leave peaks alone. Adds density without affecting transients."""
        y = self._ensure_stereo(y)
        y64 = y.astype(np.float64)

        # Envelope follower
        env = np.abs(y64)
        if env.ndim == 2:
            env = np.max(env, axis=1)
        env_db = 20.0 * np.log10(np.maximum(env, 1e-10))

        # Gain curve: boost below threshold
        gain_db = np.zeros_like(env_db)
        below = env_db < threshold_db
        gain_db[below] = (threshold_db - env_db[below]) * (1.0 - 1.0 / ratio)
        gain_db = np.clip(gain_db, 0.0, 12.0)  # Max +12dB boost
        gain_lin = 10.0 ** (gain_db / 20.0)

        if y64.ndim == 2:
            gain_lin = gain_lin[:, np.newaxis]

        y_out = y64 * gain_lin
        return self._peak_protect(y_out.astype(np.float32), ceiling_db=-1.0)

    # ── Transient/Sustain Split ──────────────────────────────────────

    def transient_sustain_split(self, y: np.ndarray, sr: int) -> tuple:
        """Separate signal into transient and sustain components."""
        y = self._ensure_stereo(y)
        y64 = y.astype(np.float64)
        env = np.abs(y64)
        if env.ndim == 2:
            env = np.max(env, axis=1)

        # Fast and slow envelopes
        fast = self._iir_envelope(env, sr, 1.0, 10.0)
        slow = self._iir_envelope(env, sr, 10.0, 100.0)

        transient = np.clip(fast - slow, 0, None)
        max_t = transient.max()
        if max_t > 1e-10:
            transient /= max_t
        sustain = 1.0 - transient

        if y64.ndim == 2:
            transient = transient[:, np.newaxis]
            sustain = sustain[:, np.newaxis]

        y_transient = (y64 * transient).astype(np.float32)
        y_sustain = (y64 * sustain).astype(np.float32)
        return y_transient, y_sustain

    # ── Biquad Filter (for EQ corrections) ───────────────────────────

    def peaking_filter(self, y: np.ndarray, sr: int,
                       center_freq: float, gain_db: float, q: float = 1.0) -> np.ndarray:
        """Bell/Notch EQ filter via scipy."""
        if abs(gain_db) < 0.05:
            return y
        from scipy.signal import lfilter
        A = 10 ** (gain_db / 40)
        omega = 2 * np.pi * center_freq / sr
        alpha = np.sin(omega) / (2 * q)

        b = [1 + alpha * A, -2 * np.cos(omega), 1 - alpha * A]
        a = [1 + alpha / A, -2 * np.cos(omega), 1 - alpha / A]

        orig_dtype = y.dtype
        y_f64 = y.astype(np.float64)
        return lfilter(b, a, y_f64, axis=0).astype(orig_dtype)

    def low_shelf(self, y: np.ndarray, sr: int,
                  shelf_freq: float, gain_db: float, q: float = 0.707) -> np.ndarray:
        """Low shelf EQ filter."""
        if abs(gain_db) < 0.05:
            return y
        from scipy.signal import lfilter
        A = 10 ** (gain_db / 40)
        omega = 2 * np.pi * shelf_freq / sr
        alpha = np.sin(omega) / (2 * q)

        b = [A * ((A + 1) - (A - 1) * np.cos(omega) + 2 * np.sqrt(A) * alpha),
             2 * A * ((A - 1) - (A + 1) * np.cos(omega)),
             A * ((A + 1) - (A - 1) * np.cos(omega) - 2 * np.sqrt(A) * alpha)]
        a = [(A + 1) + (A - 1) * np.cos(omega) + 2 * np.sqrt(A) * alpha,
             -2 * ((A - 1) + (A + 1) * np.cos(omega)),
             (A + 1) + (A - 1) * np.cos(omega) - 2 * np.sqrt(A) * alpha]

        orig_dtype = y.dtype
        y_f64 = y.astype(np.float64)
        return lfilter(b, a, y_f64, axis=0).astype(orig_dtype)

    def high_shelf(self, y: np.ndarray, sr: int,
                   shelf_freq: float, gain_db: float, q: float = 0.707) -> np.ndarray:
        """High shelf EQ filter."""
        if abs(gain_db) < 0.05:
            return y
        from scipy.signal import lfilter
        A = 10 ** (gain_db / 40)
        omega = 2 * np.pi * shelf_freq / sr
        alpha = np.sin(omega) / (2 * q)

        b = [A * ((A + 1) + (A - 1) * np.cos(omega) + 2 * np.sqrt(A) * alpha),
             -2 * A * ((A - 1) + (A + 1) * np.cos(omega)),
             A * ((A + 1) + (A - 1) * np.cos(omega) - 2 * np.sqrt(A) * alpha)]
        a = [(A + 1) - (A - 1) * np.cos(omega) + 2 * np.sqrt(A) * alpha,
             2 * ((A - 1) - (A + 1) * np.cos(omega)),
             (A + 1) - (A - 1) * np.cos(omega) - 2 * np.sqrt(A) * alpha]

        orig_dtype = y.dtype
        y_f64 = y.astype(np.float64)
        return lfilter(b, a, y_f64, axis=0).astype(orig_dtype)

    # ── DC Removal ───────────────────────────────────────────────────

    def remove_dc(self, y: np.ndarray) -> np.ndarray:
        """Remove DC offset from signal."""
        return (y - np.mean(y, axis=0)).astype(np.float32)

    # ── Gain Match ───────────────────────────────────────────────────

    def layer_eq(self, y: np.ndarray, sr: int, eq_chain: list) -> np.ndarray:
        """
        Apply a layer-specific EQ chain from config.

        eq_chain is a list of dicts:
            {'type': 'highpass', 'freq': 28}
            {'type': 'lowpass', 'freq': 105}
            {'type': 'bell', 'freq': 58, 'gain_db': 2.0, 'q': 1.1}
            {'type': 'lowshelf', 'freq': 100, 'gain_db': 2.0}
            {'type': 'highshelf', 'freq': 8000, 'gain_db': 1.5}
        """
        if not eq_chain:
            return y

        y = self._ensure_stereo(y)
        y_out = y.copy()

        for eq in eq_chain:
            eq_type = eq['type']
            freq = eq['freq']

            if eq_type == 'highpass':
                y_out = self._apply_highpass(y_out, sr, freq)
            elif eq_type == 'lowpass':
                y_out = self._apply_lowpass(y_out, sr, freq)
            elif eq_type == 'bell':
                y_out = self.peaking_filter(y_out, sr, freq, eq['gain_db'], eq.get('q', 1.0))
            elif eq_type == 'lowshelf':
                y_out = self.low_shelf(y_out, sr, freq, eq['gain_db'], eq.get('q', 0.707))
            elif eq_type == 'highshelf':
                y_out = self.high_shelf(y_out, sr, freq, eq['gain_db'], eq.get('q', 0.707))

        return y_out.astype(np.float32)

    def apply_bell_eq(self, y: np.ndarray, sr: int, freq: float,
                      gain_db: float, q: float = 1.0) -> np.ndarray:
        """Convenience: apply a single bell EQ band. Used by optimizer."""
        if abs(gain_db) < 0.05:
            return y
        return self.peaking_filter(y, sr, freq, gain_db, q).astype(np.float32)

    def detect_eq_bands(self, y: np.ndarray, sr: int, role: str = 'default',
                        n_bands: int = 4) -> list:
        """
        Detect EQ bands from stem's spectral profile.

        For bass instruments: returns sub/low/low-mid bands (no air).
        For others: finds top N peak regions in spectral profile.

        Returns list of dicts: [{'center_hz': float, 'q': float, 'band_type': str}, ...]
        """
        y = self._ensure_stereo(y)
        # Convert to mono for analysis
        mono = np.mean(np.abs(y.astype(np.float64)), axis=1) if y.ndim == 2 else np.abs(y.astype(np.float64))

        # Compute FFT magnitude spectrum
        n_fft = 4096
        mono_mean = np.mean(y.astype(np.float64), axis=1) if y.ndim == 2 else y.astype(np.float64)
        spectrum = np.abs(np.fft.rfft(mono_mean, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        # Skip DC and very low frequencies
        valid_mask = freqs >= 20.0
        spectrum = spectrum[valid_mask]
        freqs = freqs[valid_mask]

        # Smooth spectrum (1/3 octave smoothing)
        smoothed = np.zeros_like(spectrum)
        for i in range(len(freqs)):
            bw = freqs[i] / 3.0  # 1/3 octave bandwidth
            mask = (freqs >= freqs[i] - bw) & (freqs <= freqs[i] + bw)
            if np.any(mask):
                smoothed[i] = np.mean(spectrum[mask])
            else:
                smoothed[i] = spectrum[i]

        # Normalize to 0-1
        max_val = np.max(smoothed)
        if max_val > 0:
            smoothed = smoothed / max_val

        # Bass-specific: force sub/low/low-mid bands
        bass_roles = ['bass', 'sub_bass']
        if role.lower() in bass_roles or 'bass' in role.lower():
            return [
                {'center_hz': 60.0, 'q': 0.8, 'band_type': 'sub'},
                {'center_hz': 120.0, 'q': 0.8, 'band_type': 'low'},
                {'center_hz': 300.0, 'q': 0.9, 'band_type': 'low-mid'},
                {'center_hz': 800.0, 'q': 1.0, 'band_type': 'mid'},
            ]

        # Find peaks in smoothed spectrum
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(smoothed, height=0.3, distance=10, prominence=0.1)

        if len(peaks) == 0:
            # Fallback: use fixed bands
            return [
                {'center_hz': 200.0, 'q': 0.9, 'band_type': 'low-mid'},
                {'center_hz': 800.0, 'q': 1.0, 'band_type': 'mid'},
                {'center_hz': 3000.0, 'q': 1.0, 'band_type': 'high-mid'},
                {'center_hz': 8000.0, 'q': 0.8, 'band_type': 'high'},
            ]

        # Sort peaks by energy (descending)
        peak_energies = smoothed[peaks]
        sorted_indices = np.argsort(peak_energies)[::-1]
        top_peaks = peaks[sorted_indices[:n_bands]]

        # Sort by frequency for consistent ordering
        top_peaks = np.sort(top_peaks)

        # Build band definitions
        bands = []
        for peak_idx in top_peaks:
            center_hz = float(freqs[peak_idx])
            # Determine band type based on frequency
            if center_hz < 100:
                band_type = 'sub'
            elif center_hz < 300:
                band_type = 'low'
            elif center_hz < 1000:
                band_type = 'low-mid'
            elif center_hz < 3000:
                band_type = 'mid'
            elif center_hz < 6000:
                band_type = 'high-mid'
            elif center_hz < 12000:
                band_type = 'high'
            else:
                band_type = 'air'

            # Q based on frequency: wider Q for lows, tighter for highs
            q = 0.9 if center_hz < 500 else 1.0 if center_hz < 3000 else 0.8

            bands.append({'center_hz': center_hz, 'q': q, 'band_type': band_type})

        return bands

    def _apply_highpass(self, y: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
        """Apply high-pass filter via scipy butterworth."""
        from scipy.signal import butter, sosfilt
        nyq = sr / 2.0
        c = min(max(cutoff / nyq, 0.001), 0.999)
        sos = butter(2, c, btype='high', output='sos')
        if y.ndim == 2:
            return np.column_stack([sosfilt(sos, y[:, ch].astype(np.float64)).astype(np.float32)
                                    for ch in range(y.shape[1])])
        return sosfilt(sos, y.astype(np.float64)).astype(np.float32)

    def _apply_lowpass(self, y: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
        """Apply low-pass filter via scipy butterworth."""
        from scipy.signal import butter, sosfilt
        nyq = sr / 2.0
        c = min(max(cutoff / nyq, 0.001), 0.999)
        sos = butter(2, c, btype='low', output='sos')
        if y.ndim == 2:
            return np.column_stack([sosfilt(sos, y[:, ch].astype(np.float64)).astype(np.float32)
                                    for ch in range(y.shape[1])])
        return sosfilt(sos, y.astype(np.float64)).astype(np.float32)

    def gain_match(self, y_before: np.ndarray, y_after: np.ndarray,
                   max_correction_db: float = 3.0) -> tuple:
        """Restore output RMS to match input RMS. Returns (y_corrected, correction_db)."""
        rms_before = self._rms(y_before)
        rms_after = self._rms(y_after)
        if rms_after < 1e-10:
            return y_after, 0.0
        gain_db = 20.0 * np.log10(rms_before / rms_after)
        gain_db = float(np.clip(gain_db, -max_correction_db, max_correction_db))
        if abs(gain_db) < 0.05:
            return y_after, 0.0
        y_out = y_after * (10.0 ** (gain_db / 20.0))
        return y_out.astype(np.float32), gain_db

    # ── Internal Helpers ─────────────────────────────────────────────

    def _ensure_stereo(self, y: np.ndarray) -> np.ndarray:
        if y.ndim == 1:
            return np.stack([y, y], axis=1)
        if y.shape[1] == 1:
            return np.repeat(y, 2, axis=1)
        return y[:, :2]

    def _mono(self, y: np.ndarray) -> np.ndarray:
        return np.mean(y, axis=1) if y.ndim > 1 else y

    def _rms(self, y: np.ndarray) -> float:
        return float(np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-12)

    def _peak_db(self, y: np.ndarray) -> float:
        return float(20.0 * np.log10(np.max(np.abs(y)) + 1e-10))

    def _true_peak(self, y: np.ndarray, sr: int, oversample: int = 4) -> float:
        """True peak via oversampling for inter-sample peak detection."""
        mono = self._mono(y)
        if len(mono) < 10:
            return float(np.max(np.abs(y)))
        upsampled = resample_poly(mono, oversample, 1)
        return float(np.max(np.abs(upsampled)))

    def _bandpass(self, y: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
        """Extract a frequency band via butterworth bandpass filter."""
        from scipy.signal import butter, sosfilt
        nyq = sr / 2.0
        lo = max(low_hz / nyq, 0.001)
        hi = min(high_hz / nyq, 0.999)
        if lo >= hi:
            return np.zeros_like(y)
        sos = butter(2, [lo, hi], btype='band', output='sos')
        if y.ndim == 2:
            return np.column_stack([sosfilt(sos, y[:, ch].astype(np.float64)).astype(np.float32)
                                    for ch in range(y.shape[1])])
        return sosfilt(sos, y.astype(np.float64)).astype(np.float32)

    def _gr_from_rms(self, input_rms: float, output_rms: float) -> float:
        if output_rms < 1e-10:
            return 0.0
        return float(20.0 * np.log10(input_rms / output_rms))

    def _peak_protect(self, y: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
        peak = np.max(np.abs(y))
        ceiling = 10.0 ** (ceiling_db / 20.0)
        if peak > ceiling:
            y = y * (ceiling / peak)
        return y.astype(np.float32)

    def _iir_envelope(self, x: np.ndarray, sr: int,
                      attack_ms: float, release_ms: float) -> np.ndarray:
        a_c = np.exp(-1.0 / (sr * attack_ms / 1000.0))
        r_c = np.exp(-1.0 / (sr * release_ms / 1000.0))
        out = np.zeros_like(x)
        prev = 0.0
        for i in range(len(x)):
            c = a_c if x[i] > prev else r_c
            out[i] = (1 - c) * x[i] + c * prev
            prev = out[i]
        return out
