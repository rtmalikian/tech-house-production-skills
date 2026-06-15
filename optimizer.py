"""
Stem Optimizer — feedback-driven parameter optimization using Nelder-Mead.

Optimizes peak control AND processing parameters for each stem by:
1. Applying the processing chain in memory with current params
2. Measuring comprehensive metrics on busiest sections
3. Computing a weighted loss incorporating monitoring metrics
4. Letting scipy.optimize propose new params

Chain: compress(threshold + offset) → saturate(amount) → EQ(gains) →
       soft_clip(ceiling) → limit(ceiling)
"""

import os
import sys
import numpy as np

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

import config
from dsp_engine import DSPEngine
from quality_assessor import QualityAssessor
from reference_analyzer import ReferenceAnalyzer
from section_analyzer import analyze_sections


def optimize_stem(y: np.ndarray, sr: int, role: str, layer: str,
                  reference_profile: dict, context_suggestions: list,
                  eq_bands: list, bpm: float,
                  dsp: DSPEngine = None, assessor: QualityAssessor = None,
                  ref_analyzer: ReferenceAnalyzer = None,
                  verbose: bool = True, max_evals: int = None) -> tuple:
    """
    Optimize stem processing parameters using Nelder-Mead.

    Search space:
      [0] soft_clip_ceiling_db   (-6 to 0)
      [1] limiter_ceiling_db     (-3 to 0)
      [2] comp_gr_offset         (-2 to +2, shifts GR target range)
      [3] saturation_amount      (1.0 to 1.3)
      [4-7] eq_gain_1..4         (-3 to +3 per band)

    Args:
        y: audio signal (stereo)
        sr: sample rate
        role: stem role (bass, melody, drums, etc.)
        layer: layer name (lead, poly, bell, etc.) or None
        reference_profile: spectral profile dict from reference track
        context_suggestions: list of (band_idx, cut_db) from context analyzer
        eq_bands: list of EQ band dicts (freq, q, type)
        bpm: beats per minute
        dsp: DSPEngine instance (created if None)
        assessor: QualityAssessor instance (created if None)
        ref_analyzer: ReferenceAnalyzer instance (created if None)
        verbose: print each evaluation
        max_evals: override evaluation limit

    Returns:
        (best_y, best_params, history)
    """
    from scipy.optimize import minimize

    if dsp is None:
        dsp = DSPEngine()
    if assessor is None:
        assessor = QualityAssessor()
    if ref_analyzer is None:
        ref_analyzer = ReferenceAnalyzer()

    y = _ensure_stereo(y)

    target_lufs = config.STEM_LUFS_TARGETS.get(role, config.STEM_LUFS_TARGETS.get('default', -18.0))

    # Detect busy sections for evaluation
    sections = analyze_sections(y, sr, bpm, percentile=config.BUSY_SECTION_PERCENTILE,
                                min_bars=config.BUSY_SECTION_MIN_BARS)

    if not sections:
        eval_audio = y
        eval_label = "full"
    else:
        eval_segments = []
        for sec in sections:
            eval_segments.append(y[sec['start_sample']:sec['end_sample']])
        eval_audio = np.concatenate(eval_segments, axis=0)
        eval_label = f"{len(sections)} busy sections"

    # Prepare EQ band structure from detected bands
    eq_band_freqs = []
    if eq_bands:
        for band in eq_bands[:4]:
            eq_band_freqs.append(band.get('center_hz', 1000.0))
    while len(eq_band_freqs) < 4:
        eq_band_freqs.append([200.0, 800.0, 3000.0, 8000.0][len(eq_band_freqs)])

    # Context analyzer band center frequencies (10 log-spaced bands from 20-20kHz)
    context_band_freqs = np.logspace(np.log10(20), np.log10(20000), 10)
    context_band_freqs = [(context_band_freqs[i] + context_band_freqs[i+1]) / 2
                          for i in range(9)] + [float(context_band_freqs[-1])]

    # 8 params: soft_clip, limiter, comp_gr_offset, sat_amount, eq_gain_1..4
    x0 = [-1.0, -1.0, 0.0, 1.05, 0.0, 0.0, 0.0, 0.0]

    # Parameter bounds (used for clamping since Nelder-Mead ignores bounds)
    param_lo = np.array([-6.0, -3.0, -config.COMP_OPT_RANGE, 1.0, -3.0, -3.0, -3.0, -3.0])
    param_hi = np.array([0.0, 0.0, config.COMP_OPT_RANGE, 1.3, 3.0, 3.0, 3.0, 3.0])

    eval_count = [0]
    history = []
    best_loss = [float('inf')]
    stall_count = [0]
    STALL_LIMIT = 8  # stop after 8 consecutive evals with no improvement

    weights = config.OPTIMIZER_LOSS_WEIGHTS

    class _Stalled(Exception):
        """Raised when optimizer is stuck — stop early."""
        pass

    def objective(params):
        # Clamp params to valid bounds (Nelder-Mead ignores scipy bounds)
        params = np.clip(params, param_lo, param_hi)
        eval_count[0] += 1
        soft_clip_ceiling = params[0]
        limiter_ceiling = params[1]
        comp_gr_offset = params[2]
        sat_amount = params[3]
        eq_gains = params[4:8]

        try:
            # Apply expanded processing chain
            y_proc = _apply_chain(
                eval_audio, sr, role, dsp,
                soft_clip_ceiling, limiter_ceiling,
                comp_gr_offset, sat_amount,
                eq_band_freqs, eq_gains, bpm=bpm
            )

            # Comprehensive assessment
            metrics = assessor.assess_stem(y_proc, sr, role)

            # Spectral distance to reference
            spectral_dist = 0.0
            if reference_profile:
                current_profile = assessor._get_spectral_profile(y_proc, sr)
                spectral_dist = assessor._spectral_distance(current_profile, reference_profile)

            # Extract metrics
            lufs = metrics.get('lufs', -100.0)
            crest_db = metrics.get('crest_db', 0.0)
            peak_db = metrics.get('peak_db', -100.0)
            true_peak_db = metrics.get('true_peak_db', -100.0)
            lra = metrics.get('lra', 0.0)
            plr = metrics.get('plr', 0.0)
            stereo_corr = metrics.get('stereo_corr', 1.0)
            mono_compat = metrics.get('mono_compat', 1.0)
            onset_str = metrics.get('onset_strength_mean', 0.0)
            boomy = metrics.get('boomy', 0.0)
            thin = metrics.get('thin', 0.0)
            muddy = metrics.get('muddy', 0.0)
            harsh = metrics.get('harsh', 0.0)
            bright = metrics.get('bright', 0.0)

            # ── Loss computation ──
            loss = 0.0

            # LUFS deviation
            loss += weights['lufs'] * abs(lufs - target_lufs)

            # Spectral distance to reference
            loss += weights['spectral'] * spectral_dist

            # Crest factor (target: 10-14 dB for most genres)
            crest_target = 12.0
            loss += weights['crest'] * abs(crest_db - crest_target)

            # Peak control — mild compression should lower crest to ~10-11 dB
            if crest_db > 12.0:
                loss += 0.3 * (crest_db - 12.0)

            # True peak penalty (ISP)
            loss += weights['true_peak'] * max(0, true_peak_db - config.MASTER_TRUE_PEAK_DB)

            # Over-compression penalty
            loss += weights['over_compress_penalty'] * max(0, config.CREST_MIN - crest_db)

            # LRA penalty (target: 5-12 LU for most genres)
            if lra < 5.0:
                loss += weights['lra'] * (5.0 - lra)
            elif lra > 15.0:
                loss += weights['lra'] * (lra - 15.0) * 0.5

            # PLR penalty (target: 8-14 dB)
            if plr < 6.0:
                loss += weights['plr'] * (6.0 - plr)
            elif plr > 16.0:
                loss += weights['plr'] * (plr - 16.0) * 0.3

            # Stereo correlation penalty
            if stereo_corr < 0.0:
                loss += weights['stereo'] * abs(stereo_corr) * 2.0  # Phase issue
            elif stereo_corr > 0.95:
                loss += weights['stereo'] * (stereo_corr - 0.95) * 5.0  # Too mono

            # Mono compatibility penalty
            if mono_compat < 0.7:
                loss += weights['mono_compat'] * (0.7 - mono_compat) * 3.0

            # Tonal balance penalties
            tonal_penalty = (boomy + thin + muddy + harsh + bright) * 0.5
            loss += weights['tonal_balance'] * tonal_penalty

            # Context suggestion penalty — encourage optimizer to apply suggested cuts
            if context_suggestions:
                ctx_penalty = 0.0
                for band_idx, cut_db in context_suggestions:
                    # Map context band to nearest optimizer EQ band by frequency
                    if band_idx < len(context_band_freqs):
                        ctx_freq = context_band_freqs[band_idx]
                        distances = [abs(f - ctx_freq) for f in eq_band_freqs]
                        nearest_band = distances.index(min(distances))
                        ctx_penalty += abs(eq_gains[nearest_band] - cut_db) * 0.3
                loss += weights.get('context', 0.3) * ctx_penalty

            # Log history
            attempt = {
                'eval': eval_count[0],
                'params': {
                    'soft_clip_ceiling_db': soft_clip_ceiling,
                    'limiter_ceiling_db': limiter_ceiling,
                    'comp_gr_offset': comp_gr_offset,
                    'saturation_amount': sat_amount,
                    'eq_gains': list(eq_gains),
                },
                'metrics': {
                    'lufs': lufs, 'crest_db': crest_db, 'peak_db': peak_db,
                    'true_peak_db': true_peak_db, 'lra': lra, 'plr': plr,
                    'stereo_corr': stereo_corr, 'mono_compat': mono_compat,
                    'tonal_penalty': tonal_penalty,
                },
                'spectral_dist': spectral_dist,
                'loss': loss,
            }
            history.append(attempt)

            if verbose:
                print(f"  eval {eval_count[0]:2d}: "
                      f"sclip={soft_clip_ceiling:+.1f} lim={limiter_ceiling:+.1f} "
                      f"comp={comp_gr_offset:+.1f} sat={sat_amount:.2f} "
                      f"eq={','.join(f'{g:+.1f}' for g in eq_gains)} "
                      f"→ LUFS={lufs:+.1f} crest={crest_db:.1f} LRA={lra:.1f} "
                      f"TP={true_peak_db:+.1f} loss={loss:.3f}")

            # Early termination if stuck
            if loss < best_loss[0] - 0.01:
                best_loss[0] = loss
                stall_count[0] = 0
            else:
                stall_count[0] += 1
                if stall_count[0] >= STALL_LIMIT:
                    raise _Stalled(f"stalled after {eval_count[0]} evals")

            return loss

        except _Stalled:
            raise
        except Exception as e:
            if verbose:
                print(f"  eval {eval_count[0]:2d}: ERROR {e}")
            return 100.0

    # Build initial simplex to ensure meaningful exploration of all params
    simplex = np.tile(x0, (9, 1))
    simplex[1][0] += 1.0   # soft_clip +1 dB
    simplex[2][1] += 0.5   # limiter +0.5 dB
    simplex[3][2] += 1.5   # comp +1.5 dB GR shift
    simplex[4][3] += 0.05  # sat +0.05
    simplex[5][4] += 1.0   # eq band 1 +1 dB
    simplex[6][5] += 1.0   # eq band 2
    simplex[7][6] += 1.0   # eq band 3
    simplex[8][7] += 1.0   # eq band 4

    # Run optimizer — catch early termination
    effective_max_evals = max_evals if max_evals is not None else config.OPTIMIZER_MAX_EVALS

    try:
        result = minimize(
            objective, x0, method='Nelder-Mead',
            options={
                'maxfev': effective_max_evals,
                'xatol': 0.05,
                'fatol': 0.01,
                'initial_simplex': simplex,
                'adaptive': True,
            }
        )
    except _Stalled:
        # Build a fake result from the best we found
        class _Result:
            pass
        result = _Result()
        result.x = history[-1]['params']['soft_clip_ceiling_db'] if history else x0[0]
        # Use the best params from history
        best_hist = min(history, key=lambda h: h['loss'])
        result.x = [
            best_hist['params']['soft_clip_ceiling_db'],
            best_hist['params']['limiter_ceiling_db'],
            best_hist['params']['comp_gr_offset'],
            best_hist['params']['saturation_amount'],
            *best_hist['params']['eq_gains'],
        ]
        result.fun = best_hist['loss']
        if verbose:
            print(f"  Early termination: stuck after {eval_count[0]} evals, "
                  f"best loss={best_hist['loss']:.3f}")

    best_params = {
        'soft_clip_ceiling_db': result.x[0],
        'limiter_ceiling_db': result.x[1],
        'comp_gr_offset': result.x[2],
        'saturation_amount': result.x[3],
        'eq_gains': list(result.x[4:8]),
    }

    # Apply best params to full signal
    best_y = _apply_chain(
        y, sr, role, dsp,
        best_params['soft_clip_ceiling_db'],
        best_params['limiter_ceiling_db'],
        best_params['comp_gr_offset'],
        best_params['saturation_amount'],
        eq_band_freqs, best_params['eq_gains'], bpm=bpm
    )

    if verbose:
        print(f"  Optimizer converged: {eval_count[0]} evals, best loss={result.fun:.3f}")

    return best_y, best_params, history


def _apply_chain(y: np.ndarray, sr: int, role: str, dsp: DSPEngine,
                 soft_clip_ceiling: float, limiter_ceiling: float,
                 comp_gr_offset: float, sat_amount: float,
                 eq_freqs: list, eq_gains: list,
                 bpm: float = 90.0) -> np.ndarray:
    """
    Expanded processing chain: compress → saturate → EQ → soft_clip → limit.
    """
    y_proc = y.astype(np.float32)

    # A. Always apply adaptive compression (threshold auto-adjusts to signal)
    result = dsp.adaptive_compress(y_proc, sr, role,
                                   threshold_offset_db=comp_gr_offset,
                                   bpm=bpm)
    y_proc = result['y']

    # B. Signal-dependent saturation
    if sat_amount > 1.01:
        y_proc = dsp.saturate(y_proc, role=role, amount_override=sat_amount)

    # C. Per-band EQ with optimized gains
    for freq, gain_db in zip(eq_freqs, eq_gains):
        if abs(gain_db) > 0.05:
            y_proc = dsp.apply_bell_eq(y_proc, sr, freq, gain_db, q=1.0)

    # D. Soft clip to catch peaks gently
    y_proc = dsp.soft_clip(y_proc, ceiling_db=soft_clip_ceiling)

    # E. Hard limiter for final ceiling
    y_proc = dsp.limit(y_proc, sr, threshold_db=limiter_ceiling)

    return y_proc.astype(np.float32)


def _ensure_stereo(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return np.stack([y, y], axis=1)
    if y.shape[1] == 1:
        return np.repeat(y, 2, axis=1)
    return y[:, :2]
