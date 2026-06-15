"""
Iterative processor — iZotope-style Analyze → Process → Verify feedback loops.

CRITICAL DESIGN: Each attempt REPLACES the previous (never stacks filters).
Single combined filter per attempt. Immediate stop on quality degradation.
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
from dsp_engine import DSPEngine
from dynamic_eq import DynamicEQ
from stereo_processor import StereoProcessor
from quality_assessor import QualityAssessor
from reference_analyzer import ReferenceAnalyzer


class IterativeProcessor:
    """Safe feedback loops for stems, buses, and master."""

    def __init__(self, dsp_engine: DSPEngine = None):
        self.dsp = dsp_engine or DSPEngine()
        self.dyn_eq = DynamicEQ()
        self.stereo = StereoProcessor()
        self.assessor = QualityAssessor()
        self.ref_analyzer = ReferenceAnalyzer()

    # ── Stem Processing ──────────────────────────────────────────────

    def iterative_stem(self, y: np.ndarray, sr: int, role: str = 'default',
                       reference_profile: dict = None,
                       layer: str = None, context_suggestions: list = None,
                       eq_bands: list = None, bpm: float = 120.0,
                       use_optimizer: bool = True, stem_name: str = '') -> tuple:
        """
        Process a stem with iterative feedback.

        If use_optimizer=True and optimizer params are provided, uses Nelder-Mead
        optimization to find best compression threshold and EQ gains.
        Otherwise falls back to the simple loop.

        Returns (y_processed, debug_info).
        """
        y = _ensure_stereo(y)

        # Skip optimizer for transient percussion (crash, ride, tambourine, etc.)
        name_lower = stem_name.lower()
        is_transient_perc = any(tok in name_lower for tok in config.TRANSIENT_PERCUSSION_TOKENS)

        if is_transient_perc:
            # Just gain stage — no optimizer, no compression
            target_lufs = config.STEM_LUFS_TARGETS.get(role, config.STEM_LUFS_TARGETS.get('default', -18.0))
            from gain_staging import gain_stage_to_target
            best_y, gain_db = gain_stage_to_target(y, sr, role)
            debug = {
                'mode': 'skip_transient',
                'attempts': 0,
                'best_params': {},
                'history': [],
                'corrections': [f"transient percussion: skipped optimizer, gain={gain_db:+.1f}dB"],
            }
            return best_y, debug

        # Determine max evals based on role
        max_evals = None
        if role == 'drums':
            max_evals = config.DRUM_OPTIMIZER_MAX_EVALS

        if use_optimizer and bpm > 0:
            try:
                from optimizer import optimize_stem
                best_y, best_params, history = optimize_stem(
                    y, sr, role, layer, reference_profile,
                    context_suggestions, eq_bands, bpm,
                    dsp=self.dsp, assessor=self.assessor,
                    ref_analyzer=self.ref_analyzer, verbose=True,
                    max_evals=max_evals
                )
                debug = {
                    'mode': 'optimizer',
                    'attempts': len(history),
                    'best_params': best_params,
                    'history': history,
                    'corrections': [f"eval {h['eval']}: loss={h['loss']:.3f}" for h in history[-3:]],
                }
                # Dynamic soft clip for drum stems
                if role == 'drums':
                    clip_cfg = config.DRUM_DYNAMIC_SOFT_CLIP
                    best_y = self.dsp.dynamic_soft_clip(best_y, sr,
                        headroom_db=clip_cfg['stem_headroom_db'],
                        block_ms=clip_cfg['block_ms'])
                return best_y, debug
            except Exception as e:
                print(f"  [WARN] Optimizer failed ({e}), falling back to simple loop")

        # Fallback: simple iterative loop
        best_y = y.copy()
        best_assessment = self.assessor.assess_stem(best_y, sr, role)
        debug = {'mode': 'simple', 'attempts': 0, 'corrections': []}

        for attempt in range(config.MAX_STEM_ITERATIONS):
            debug['attempts'] = attempt + 1

            # Process: adaptive compress → saturate → dynamic EQ
            y_candidate = best_y.copy()
            comp_result = self.dsp.adaptive_compress(y_candidate, sr, role, bpm=bpm)
            y_candidate = comp_result['y']
            y_candidate = self.dsp.saturate(y_candidate, role)

            # Dynamic EQ (mud/harshness control)
            y_candidate = self.dyn_eq.process(y_candidate, sr)

            # Gain match to original level
            y_candidate, gm_db = self.dsp.gain_match(best_y, y_candidate, max_correction_db=3.0)

            # Verify
            assessment = self.assessor.assess_stem(y_candidate, sr, role)

            if assessment['all_pass']:
                best_y = y_candidate
                debug['corrections'].append(f"attempt {attempt+1}: all_pass")
                break

            # Check if quality improved
            if self._assessment_score(assessment) > self._assessment_score(best_assessment):
                best_y = y_candidate
                best_assessment = assessment
                debug['corrections'].append(f"attempt {attempt+1}: improved")
            else:
                debug['corrections'].append(f"attempt {attempt+1}: no improvement, stopping")
                break

        return best_y, debug

    # ── Bus Processing ───────────────────────────────────────────────

    def iterative_bus(self, bus_y: np.ndarray, sr: int, bus_name: str,
                      reference_profile: dict = None) -> tuple:
        """
        Process a bus with iterative feedback. Max 3 attempts.

        Returns (y_processed, debug_info).
        """
        bus_y = _ensure_stereo(bus_y)
        best_y = bus_y.copy()
        best_score = self._compute_spectral_score(best_y, sr, reference_profile)
        debug = {'attempts': 0, 'corrections': []}

        for attempt in range(config.MAX_BUS_ITERATIONS):
            debug['attempts'] = attempt + 1

            y_candidate = best_y.copy()

            # Spectral profile matching (if reference available)
            if reference_profile:
                y_candidate = self.ref_analyzer.match_eq(
                    y_candidate, sr, reference_profile,
                    max_iterations=2, dsp_engine=self.dsp
                )

            # Gain match
            y_candidate, gm_db = self.dsp.gain_match(best_y, y_candidate, max_correction_db=2.0)

            # Verify
            new_score = self._compute_spectral_score(y_candidate, sr, reference_profile)

            if new_score < best_score:
                best_score = new_score
                best_y = y_candidate
                debug['corrections'].append(f"attempt {attempt+1}: improved (score={new_score:.3f})")
            else:
                debug['corrections'].append(f"attempt {attempt+1}: no improvement, stopping")
                break

        return best_y, debug

    # ── Master Processing ────────────────────────────────────────────

    def iterative_master(self, y: np.ndarray, sr: int,
                         reference_analysis: dict = None,
                         bpm: float = 90.0) -> tuple:
        """
        Master with iterative feedback. Max 5 attempts.

        Each attempt replaces previous. Final chain: soft clip → limit → normalize.

        Returns (y_mastered, debug_info).
        """
        y = _ensure_stereo(y)
        best_y = y.copy()
        best_distance = float('inf')
        debug = {'attempts': 0, 'corrections': []}

        ref_profile = None
        if reference_analysis:
            ref_profile = reference_analysis.get('spectral_profile', {})

        for attempt in range(config.MAX_MASTER_ITERATIONS):
            debug['attempts'] = attempt + 1

            y_candidate = best_y.copy()

            # 1. Match EQ (reference-guided)
            if ref_profile:
                y_candidate = self.ref_analyzer.match_eq(
                    y_candidate, sr, reference_analysis,
                    max_iterations=2, dsp_engine=self.dsp
                )

            # 2. M/S processing
            y_candidate = self.stereo.side_hpf(y_candidate, sr, cutoff=90.0)
            y_candidate = self.stereo.ms_eq(y_candidate, sr)

            # 3. Gain match
            y_candidate, gm_db = self.dsp.gain_match(best_y, y_candidate, max_correction_db=2.0)

            # Verify
            if ref_profile:
                current_profile = self.ref_analyzer._get_narrowband_profile(y_candidate, sr)
                distance = self.ref_analyzer._profile_distance(current_profile, ref_profile)
            else:
                distance = 0.0

            if distance < best_distance:
                best_distance = distance
                best_y = y_candidate
                debug['corrections'].append(f"attempt {attempt+1}: improved (dist={distance:.3f})")
            else:
                debug['corrections'].append(f"attempt {attempt+1}: no improvement, stopping")
                break

            # Convergence check
            if distance < config.MASTER_SPECTRAL_VARIANCE:
                debug['corrections'].append(f"attempt {attempt+1}: converged")
                break

        # Final chain: compress → brick-wall limit → reference-matched gain
        # No normalization — preserves gain staging balance

        # 1. Gentle glue compression (1.5:1, 2-3 dB GR)
        comp_result = self.dsp.adaptive_compress(best_y, sr, 'default', bpm=bpm)
        best_y = comp_result['y']

        # 2. Custom brick-wall limiter (no pops, with lookahead)
        best_y = self.dsp.brick_wall_limit(best_y, sr, ceiling_db=config.MASTER_TRUE_PEAK_DB)

        # 3. Reference-matched corrective gain (preserves balance)
        ref_lufs = 10.2  # Default reference LUFS
        if reference_analysis and 'lufs' in reference_analysis:
            ref_lufs = reference_analysis['lufs']
        best_y, correction_db = self.dsp.reference_matched_gain(best_y, sr, ref_lufs, max_correction_db=6.0)

        return best_y, debug

    def streaming_master(self, y_mastered: np.ndarray, sr: int) -> tuple:
        """
        Create a streaming-optimized master from an existing mastered signal.

        Target: -14 LUFS, -1 dBTP (Spotify/Apple/YouTube/Tidal).
        Applies: gentle limiting + gain adjustment. No re-EQ — preserves the
        original master's tonal balance.

        Returns (y_streaming, debug_info).
        """
        y = _ensure_stereo(y_mastered)
        debug = {'target_lufs': config.STREAMING_TARGET_LUFS,
                 'target_true_peak': config.STREAMING_TRUE_PEAK_DB}

        # Measure current levels
        meter = pyln.Meter(sr)
        try:
            current_lufs = float(meter.integrated_loudness(y))
        except Exception:
            current_lufs = -70.0

        current_tp = self.dsp._true_peak(y, sr)
        current_tp_db = 20.0 * np.log10(max(current_tp, 1e-10))

        debug['input_lufs'] = current_lufs
        debug['input_true_peak_db'] = current_tp_db

        # If already at or below target, just ensure true peak compliance
        if current_lufs <= config.STREAMING_TARGET_LUFS + 1.0:
            y_out = self.dsp.brick_wall_limit(y, sr, ceiling_db=config.STREAMING_TRUE_PEAK_DB)
            debug['action'] = 'limit_only'
        else:
            # Need to reduce loudness. Apply gain reduction, then limit.
            gain_db = config.STREAMING_TARGET_LUFS - current_lufs
            gain_db = max(gain_db, -6.0)  # cap at -6 dB
            y_out = y * (10.0 ** (gain_db / 20.0))
            y_out = self.dsp.brick_wall_limit(y_out, sr, ceiling_db=config.STREAMING_TRUE_PEAK_DB)
            debug['action'] = f'gain {gain_db:+.1f} dB + limit'

        # Verify
        try:
            final_lufs = float(meter.integrated_loudness(y_out))
        except Exception:
            final_lufs = -70.0
        final_tp = self.dsp._true_peak(y_out, sr)
        final_tp_db = 20.0 * np.log10(max(final_tp, 1e-10))

        debug['output_lufs'] = final_lufs
        debug['output_true_peak_db'] = final_tp_db

        return y_out.astype(np.float32), debug

    # ── Helpers ──────────────────────────────────────────────────────

    def _assessment_score(self, assessment: dict) -> float:
        """Compute a single quality score from assessment (higher is better)."""
        score = 0.0
        if assessment.get('lufs_pass'):
            score += 1.0
        if assessment.get('peak_pass'):
            score += 1.0
        if assessment.get('crest_pass'):
            score += 1.0
        if assessment.get('stereo_pass'):
            score += 1.0
        return score

    def _compute_spectral_score(self, y: np.ndarray, sr: int,
                                 reference_profile: dict) -> float:
        """Compute spectral distance to reference (lower is better)."""
        if not reference_profile:
            return 0.0
        current = self.assessor._get_spectral_profile(y, sr)
        return self.assessor._spectral_distance(current, reference_profile)


def _ensure_stereo(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return np.stack([y, y], axis=1)
    if y.shape[1] == 1:
        return np.repeat(y, 2, axis=1)
    return y[:, :2]
