"""
Production Engine — main orchestrator for the Python Revamp Pipeline.

Same interface as the existing ProductionEngine (process_full_mix, process_pristine_mix)
so it can be swapped into pipeline_orchestrator.py without changes.
"""

import os
import sys
import shutil
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

import gc
import hashlib
import json
import math
import re
import tempfile
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

# Ensure this package's directory is first on sys.path
import config
from dsp_engine import DSPEngine
from gain_staging import gain_stage_to_target
from dynamic_eq import DynamicEQ
from stereo_processor import StereoProcessor
from quality_assessor import QualityAssessor
from reference_analyzer import ReferenceAnalyzer
from automation_engine import BusAutomationEngine
from iterative_processor import IterativeProcessor
from stem_context_analyzer import analyze_stem_context
from optimization_logger import OptimizationLogger
from sanity_checker import SanityChecker
from objective_listening_engine import ObjectiveListeningEngine
from golden_post_processor import GoldenPostProcessor, load_params
from preservation_mastering import render_preservation_master_file


def _atomic_write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=os.path.dirname(path),
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _stable_name_offset(name: str, modulo: int = 1000) -> int:
    digest = hashlib.sha256(str(name).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % int(modulo)

SECTION_BARS = {
    "intro": (0, 8),
    "verse1_a": (8, 16),
    "verse1_b": (16, 20),
    "pre_chorus1_build": (20, 24),
    "chorus1": (24, 32),
    "fill1": (32, 36),
    "verse2_a": (36, 44),
    "verse2_b": (44, 48),
    "pre_chorus2_build": (48, 52),
    "chorus2": (52, 60),
    "fill2": (60, 64),
    "outro": (64, 72),
}

SECTION_ALLOWED_LIFT_DB = {
    "intro": -1.5,
    "verse": 0.0,
    "pre_chorus": 1.5,
    "chorus": 2.75,
    "fill": 3.0,
    "outro": -0.75,
}

SECTION_PEAK_CEILINGS_DB = {
    "intro": -4.5,
    "verse": -3.5,
    "pre_chorus": -3.0,
    "chorus": -2.5,
    "fill": -2.5,
    "outro": -4.0,
}

SECTION_PEAK_MAX_TRIM_DB = {
    "intro": -4.0,
    "verse": -4.0,
    "pre_chorus": -3.5,
    "chorus": -3.0,
    "fill": -3.0,
    "outro": -4.0,
}

SECTION_RETURN_TARGETS_DB = {
    "intro": -10.0,
    "verse": -12.0,
    "pre_chorus": -13.0,
    "chorus": -14.0,
    "fill": -14.0,
    "outro": -10.0,
}

BUS_SECTION_TARGETS = {
    "bass": {
        "lift_db": {"intro": -1.0, "verse": 0.0, "pre_chorus": 1.0, "chorus": 1.5, "fill": 1.5, "outro": -0.75},
        "max_trim_db": -7.0,
        "max_boost_db": 1.25,
        "bands": [(40.0, 90.0, 1.35), (90.0, 180.0, 1.0), (180.0, 350.0, 0.55)],
    },
    "melody": {
        "lift_db": {"intro": -0.75, "verse": 0.0, "pre_chorus": 1.0, "chorus": 1.5, "fill": 1.25, "outro": -0.75},
        "max_trim_db": -8.0,
        "max_boost_db": 1.25,
        "bands": [(180.0, 450.0, 0.8), (450.0, 1200.0, 1.1), (1200.0, 4200.0, 1.3), (4200.0, 8500.0, 0.45)],
    },
    "pads": {
        "lift_db": {"intro": -0.75, "verse": 0.0, "pre_chorus": 0.75, "chorus": 1.0, "fill": 1.0, "outro": -0.25},
        "max_trim_db": -8.0,
        "max_boost_db": 0.75,
        "bands": [(120.0, 350.0, 0.9), (350.0, 1200.0, 1.15), (1200.0, 4200.0, 1.0), (4200.0, 12000.0, 0.65)],
    },
    "drums": {
        "lift_db": {"intro": -1.0, "verse": 0.0, "pre_chorus": 0.75, "chorus": 1.0, "fill": 1.5, "outro": -0.75},
        "max_trim_db": -3.0,
        "max_boost_db": 2.5,
        "bands": [(45.0, 120.0, 1.0), (160.0, 350.0, 0.8), (850.0, 4200.0, 1.2), (4200.0, 11000.0, 0.75)],
    },
}


class ProductionEngine:
    """
    Advanced Production Engine — Pure Python DSP.

    Replaces FFmpeg-based processing with pedalboard and pyloudnorm.
    Uses iterative feedback loops (iZotope-style) at stem, bus, and master levels.
    """

    def __init__(self, output_dir: str = "output/mastered",
                 golden_post_enabled: bool = True,
                 golden_post_params_path: str = None,
                 golden_pan_seed: int = None,
                 section_aware_pink_enabled: bool = True,
                 publish_full_master: bool = True,
                 stem_optimizer_enabled: bool = True):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.golden_post_enabled = bool(golden_post_enabled)
        self.golden_post_params_path = golden_post_params_path
        self.golden_pan_seed = golden_pan_seed
        self.section_aware_pink_enabled = bool(section_aware_pink_enabled)
        self.publish_full_master = bool(publish_full_master)
        self.stem_optimizer_enabled = bool(stem_optimizer_enabled)

        self.dsp = DSPEngine()
        self.dyn_eq = DynamicEQ()
        self.stereo = StereoProcessor()
        self.assessor = QualityAssessor()
        self.ref_analyzer = ReferenceAnalyzer()
        self.iterative = IterativeProcessor(self.dsp)
        self.automation = BusAutomationEngine(self.dsp)
        self.optimization_logger = OptimizationLogger(output_dir)
        self.sanity_checker = SanityChecker()
        self.listener = ObjectiveListeningEngine(
            settings=getattr(config, 'OBJECTIVE_LISTENING', {}),
            profile=getattr(config, 'OBJECTIVE_LISTENING_PROFILE', 'lofi_warm'),
        )

        # Load reference track
        self.reference_analysis = self.ref_analyzer.analyze_reference(
            config.REFERENCE_TRACK_PATH
        )

    # ── State Management ──────────────────────────────────────────────

    def _state_path(self) -> str:
        return os.path.join(self.output_dir, "pipeline_state.json")

    def _save_state(self, step: int, song_name: str, bpm: float,
                    processed_paths: list = None, reverb_return_paths: list = None,
                    delay_return_paths: list = None, bus_paths: list = None,
                    automated_bus_paths: list = None, mix_path: str = None,
                    master_path: str = None, stem_paths: list = None):
        state = {
            'step': step, 'song_name': song_name, 'bpm': bpm,
            'stem_paths': stem_paths or [],
            'processed_paths': processed_paths or [],
            'reverb_return_paths': reverb_return_paths or [],
            'delay_return_paths': delay_return_paths or [],
            'bus_paths': bus_paths or [],
            'automated_bus_paths': automated_bus_paths or [],
            'mix_path': mix_path, 'master_path': master_path,
        }
        _atomic_write_json(self._state_path(), state)

    def _load_state(self) -> dict:
        path = self._state_path()
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def resume_from_step(self, start_step: int, song_name: str = None,
                         bpm: float = None) -> str:
        """Resume pipeline from a saved state at the given step."""
        state = self._load_state()
        if not state:
            raise FileNotFoundError(f"No pipeline state found at {self._state_path()}")

        song_name = song_name or state['song_name']
        bpm = bpm or state['bpm']
        stem_paths = state.get('stem_paths', [])
        processed_paths = state.get('processed_paths', [])
        reverb_return_paths = state.get('reverb_return_paths', [])
        delay_return_paths = state.get('delay_return_paths', [])
        bus_paths = state.get('bus_paths', [])
        automated_bus_paths = state.get('automated_bus_paths', [])
        mix_path = state.get('mix_path')
        master_path = state.get('master_path')

        # Get sample rate from available audio files
        sr = self._get_sr_from_paths(processed_paths or stem_paths)

        print(f"\n--- RESUMING: {song_name} from Step {start_step} (BPM: {bpm}) ---")

        # Step 2: Bus processing
        if start_step <= 2:
            print("[Step 2] Bus Processing & Unmasking...")
            bus_paths = self._process_buses(processed_paths, sr, bpm)
            self._save_state(2, song_name, bpm, processed_paths, reverb_return_paths,
                           delay_return_paths, bus_paths, stem_paths=stem_paths)

        # Step 2.5: Bus automation
        if start_step <= 2.5:
            print("[Step 2.5] Bus Automation FX...")
            automation_dir = os.path.join(self.output_dir, "automated")
            os.makedirs(automation_dir, exist_ok=True)
            bus_dict = {}
            for bp in bus_paths:
                bus_name = os.path.basename(bp).replace("bus_", "").replace(".wav", "")
                bus_dict[bus_name] = bp
            automated_bus_dict = self.automation.apply_bus_automation(
                bus_dict, sr, bpm, automation_dir, song_name=song_name
            )
            automated_bus_paths = list(automated_bus_dict.values())
            self._save_state(2.5, song_name, bpm, processed_paths, reverb_return_paths,
                           delay_return_paths, bus_paths, automated_bus_paths,
                           stem_paths=stem_paths)

        # Step 3: Sum to mix
        if start_step <= 3:
            print("[Step 3] Summing to Mix...")
            reverb_return_paths, delay_return_paths = self._process_returns_objectively(
                processed_paths, reverb_return_paths, delay_return_paths, sr, bpm
            )
            bus_paths, automated_bus_paths, reverb_return_paths, delay_return_paths = (
                self._apply_section_aware_pink_staging(
                    processed_paths, reverb_return_paths, delay_return_paths,
                    bus_paths, automated_bus_paths, sr, bpm, song_name
                )
            )
            mix_path = self._sum_with_optional_golden_post(
                processed_paths, reverb_return_paths, delay_return_paths,
                bus_paths, automated_bus_paths, song_name, bpm, sr
            )
            mix_path = self._prepare_premaster_objectively(mix_path, song_name, sr, bpm)
            self._save_state(3, song_name, bpm, processed_paths, reverb_return_paths,
                           delay_return_paths, bus_paths, automated_bus_paths, mix_path,
                           stem_paths=stem_paths)

        # Step 4: Mastering
        if start_step <= 4:
            print("[Step 4] Iterative Mastering...")
            master_path = self._master(mix_path, song_name, bpm=bpm)
            self._save_state(4, song_name, bpm, processed_paths, reverb_return_paths,
                           delay_return_paths, bus_paths, automated_bus_paths, mix_path,
                           master_path, stem_paths)

        # Step 5: Variants
        if start_step <= 5:
            print("[Step 5] Creating Mix Variants...")
            self._create_variants(processed_paths, reverb_return_paths,
                                delay_return_paths, bus_paths, song_name, sr, bpm=bpm)

        # Copy final masters to final/
        final_dir = os.path.join(self.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        final_copied = []
        if master_path and os.path.exists(master_path):
            dest = os.path.join(final_dir, os.path.basename(master_path))
            shutil.copy2(master_path, dest)
            final_copied.append(dest)
        for f in os.listdir(self.output_dir):
            if _is_retained_final_master(f) and f != os.path.basename(master_path or ""):
                src = os.path.join(self.output_dir, f)
                dest = os.path.join(final_dir, f)
                shutil.copy2(src, dest)
                final_copied.append(dest)
        if final_copied:
            print(f"\n  Final outputs ({len(final_copied)}):")
            for p in final_copied:
                print(f"    {p}")

        self.listener.write_reports(self.output_dir)

        print(f"\n✓ Production complete: {master_path}")
        return master_path

    def _get_sr_from_paths(self, paths: list) -> int:
        """Get sample rate from first available audio file."""
        for p in paths:
            if p and os.path.exists(p):
                _, sr = sf.read(p, always_2d=True)
                return sr
        return 44100

    # ── Main Entry Point ─────────────────────────────────────────────

    def process_full_mix(self, stems: dict, song_name: str, bpm: float = 90.0) -> str:
        """
        Full production pipeline with iterative processing.

        Args:
            stems: dict of {key: filepath}
            song_name: song name for output files
            bpm: beats per minute

        Returns:
            path to mastered WAV file
        """
        print(f"\n--- PRODUCING: {song_name} (BPM: {bpm}) ---")

        stem_paths = [p for p in stems.values() if p and os.path.exists(p)]
        if not stem_paths:
            print("  No stems found — skipping production.")
            return None

        # ── Step 1: Per-stem iterative processing ────────────────────
        print("[Step 1] Per-Stem Optimization...")
        processed_dir = os.path.join(self.output_dir, "processed")
        reverb_dir = os.path.join(self.output_dir, "reverb_returns")
        delay_dir = os.path.join(self.output_dir, "delay_returns")
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(reverb_dir, exist_ok=True)
        os.makedirs(delay_dir, exist_ok=True)

        # Save initial state so resume is possible even if Step 1 crashes
        sr = self._get_sr_from_paths(stem_paths)
        self._save_state(0, song_name, bpm, stem_paths=stem_paths)

        # Analyze stem context (spectral collisions between stems)
        print("  Analyzing stem context...")
        ref_profile = None
        if self.reference_analysis:
            ref_profile = self.reference_analysis.get('spectral_profile')
        context = analyze_stem_context(stem_paths, sr)
        context_suggestions = context.get('suggestions', {})
        stem_diagnosis = self.listener.analyze_stems(stem_paths, sr, bpm)
        stem_actions = self._actions_by_target(stem_diagnosis.actions)

        processed_paths = []
        reverb_return_paths = []
        delay_return_paths = []

        for path in stem_paths:
            name = os.path.basename(path)
            role = self._detect_role(name)
            layer = self._detect_layer(name)
            layer_preset = self._get_layer_preset(name)

            # Load
            y, sr = sf.read(path, always_2d=True)
            y = _ensure_stereo(np.asarray(y, dtype=np.float32))

            if np.max(np.abs(y)) < 1e-6:
                print(f"  {name:48s}  [SILENT] skipping")
                continue

            # A. Gain stage — use layer-specific LUFS target if available
            if layer_preset:
                # Temporarily override LUFS target for this layer
                orig_target = config.STEM_LUFS_TARGETS.get(role)
                config.STEM_LUFS_TARGETS[role] = layer_preset['lufs_target']
                y_staged, gain_db = gain_stage_to_target(y, sr, role)
                if orig_target is not None:
                    config.STEM_LUFS_TARGETS[role] = orig_target
            else:
                y_staged, gain_db = gain_stage_to_target(y, sr, role)

            # B. Detect EQ bands from spectral profile
            eq_bands = self.dsp.detect_eq_bands(y_staged, sr, role, n_bands=4)

            # C. Get context suggestions for this stem
            stem_suggestions = context_suggestions.get(name, [])

            # D. Optimized iterative processing
            print(f"\n  Processing: {name}")
            print(f"    Role: {role}, Layer: {layer or 'none'}")
            print(f"    EQ bands: {[b['band_type'] for b in eq_bands]}")
            if stem_suggestions:
                print(f"    Context cuts: {stem_suggestions}")

            y_processed, debug = self.iterative.iterative_stem(
                y_staged, sr, role, ref_profile,
                layer=layer, context_suggestions=stem_suggestions,
                eq_bands=eq_bands, bpm=bpm, use_optimizer=self.stem_optimizer_enabled,
                stem_name=name
            )

            # E. Apply layer-specific EQ (frequency shaping per layer role)
            if layer_preset and 'eq' in layer_preset:
                y_before_eq = y_processed.copy()
                y_processed = self.dsp.layer_eq(y_processed, sr, layer_preset['eq'])
                y_processed, _ = self.dsp.gain_match(y_before_eq, y_processed, max_correction_db=2.0)

            # E2. Objective listening corrections for this stem
            listen_actions = stem_actions.get(name, [])
            if listen_actions:
                y_processed = self._apply_listening_actions(
                    y_processed, sr, listen_actions, source_name=name, bpm=bpm
                )

            # F. Apply layer trim (balances stacked layers musically)
            if layer_preset and layer_preset['trim_db'] != 0.0:
                trim_db = layer_preset['trim_db']
                y_processed = y_processed * (10.0 ** (trim_db / 20.0))

            # G. Quality assessment
            assessment = self.assessor.assess_stem(y_processed, sr, role)

            # H. Sanity check against historical runs
            sanity_result = self.sanity_checker.check_stem(name, assessment)

            # I. Loudness match to source
            y_processed, lm_correction = self.assessor.loudness_match(y, y_processed, sr)

            # I2. Auto-pan for drum/percussion stems (not kick)
            if _is_autopan_eligible(name):
                pan_cfg = config.AUTO_PAN
                seed = pan_cfg['seed_base'] + _stable_name_offset(name)
                y_processed = self.dsp.auto_pan(
                    y_processed, sr, bpm,
                    pan_range=pan_cfg['pan_range'],
                    rate_triplets=pan_cfg['rate_triplets'],
                    irregularity=pan_cfg['irregularity'],
                    seed=seed
                )

            # J. Save processed stem
            proc_path = os.path.join(processed_dir, f"proc_{name}")
            sf.write(proc_path, y_processed, sr, subtype='FLOAT')
            processed_paths.append(proc_path)

            # K. Log optimization results
            best_params = debug.get('best_params', {})
            history = debug.get('history', [])
            self.optimization_logger.log_stem_optimization(
                name, role, best_params, history, sanity_result
            )

            # L. Reverb send — use layer-specific send level if available
            send_config = self._get_send_config(name)
            if layer_preset:
                send_config['reverb_send'] = layer_preset['reverb_send']
                send_config['delay_send'] = layer_preset['delay_send']
                send_config['reverb_category'] = layer_preset['reverb_category']

            if send_config['reverb_send'] > 0:
                y_send = y_processed * send_config['reverb_send']
                category = send_config['reverb_category']
                y_wet = self.dsp.reverb(y_send, sr, category)
                rev_path = os.path.join(reverb_dir, f"reverb_return_{name}")
                sf.write(rev_path, y_wet, sr, subtype='FLOAT')
                reverb_return_paths.append(rev_path)
            else:
                rev_path = None

            # H. Delay send
            if send_config['delay_send'] > 0:
                y_dly_send = y_processed * send_config['delay_send']
                delay_type = send_config.get('delay_type', 'melodic')
                y_dly = self.dsp.delay(y_dly_send, sr, bpm, delay_type)
                dly_path = os.path.join(delay_dir, f"delay_return_{name}")
                sf.write(dly_path, y_dly, sr, subtype='FLOAT')
                delay_return_paths.append(dly_path)

            # Log
            layer_str = f" [{layer}]" if layer else ""
            status = "OK" if assessment['all_pass'] else "WARN"
            clip_flag = " CLIP!" if assessment.get('is_clipped') else ""
            pan_str = " [autopan]" if _is_autopan_eligible(name) else ""
            print(
                f"  {name:48s}  gain:{gain_db:+.1f}dB | "
                f"LUFS={assessment['lufs']:+.1f} TP={assessment['true_peak_db']:+.1f} "
                f"crest={assessment['crest_db']:.1f} LRA={assessment['lra']:.1f} | "
                f"stereo={assessment['stereo_corr']:.2f}{pan_str} | "
                f"[{status}]{clip_flag}{layer_str}"
            )

            del y, y_staged, y_processed
            gc.collect()

        # Log session summary
        self.optimization_logger.log_session_summary()
        summary = self.optimization_logger.get_session_summary()
        if summary:
            print(f"\n  Optimization Summary:")
            print(f"    Total stems: {summary.get('total_stems', 0)}")
            print(f"    Avg evaluations: {summary.get('avg_evaluations_per_stem', 0)}")
            print(f"    All pass: {summary.get('stems_all_pass', 0)}")
            print(f"    With warnings: {summary.get('stems_with_warnings', 0)}")

        # Save state after Step 1
        self._save_state(1, song_name, bpm, processed_paths, reverb_return_paths,
                         delay_return_paths, stem_paths=stem_paths)

        # ── Step 2: Bus processing with unmasking ────────────────────
        print("[Step 2] Bus Processing & Unmasking...")
        bus_paths = self._process_buses(processed_paths, sr, bpm)
        self._save_state(2, song_name, bpm, processed_paths, reverb_return_paths,
                         delay_return_paths, bus_paths, stem_paths=stem_paths)

        # ── Step 2.5: Bus Automation (intro/outro, phrase FX, transitions) ──
        print("[Step 2.5] Bus Automation FX...")
        automation_dir = os.path.join(self.output_dir, "automated")
        os.makedirs(automation_dir, exist_ok=True)
        # Convert bus_paths list to dict for automation engine
        bus_dict = {}
        for bp in bus_paths:
            bus_name = os.path.basename(bp).replace("bus_", "").replace(".wav", "")
            bus_dict[bus_name] = bp
        automated_bus_dict = self.automation.apply_bus_automation(
            bus_dict, sr, bpm, automation_dir, song_name=song_name
        )
        automated_bus_paths = list(automated_bus_dict.values())
        self._save_state(2.5, song_name, bpm, processed_paths, reverb_return_paths,
                         delay_return_paths, bus_paths, automated_bus_paths,
                         stem_paths=stem_paths)

        # ── Step 3: Sum to mix ───────────────────────────────────────
        print("[Step 3] Summing to Mix...")
        reverb_return_paths, delay_return_paths = self._process_returns_objectively(
            processed_paths, reverb_return_paths, delay_return_paths, sr, bpm
        )
        bus_paths, automated_bus_paths, reverb_return_paths, delay_return_paths = (
            self._apply_section_aware_pink_staging(
                processed_paths, reverb_return_paths, delay_return_paths,
                bus_paths, automated_bus_paths, sr, bpm, song_name
            )
        )
        mix_path = self._sum_with_optional_golden_post(
            processed_paths, reverb_return_paths, delay_return_paths,
            bus_paths, automated_bus_paths, song_name, bpm, sr
        )
        mix_path = self._prepare_premaster_objectively(mix_path, song_name, sr, bpm)
        self._save_state(3, song_name, bpm, processed_paths, reverb_return_paths,
                         delay_return_paths, bus_paths, automated_bus_paths, mix_path,
                         stem_paths=stem_paths)

        # ── Step 4: Iterative mastering ──────────────────────────────
        print("[Step 4] Iterative Mastering...")
        master_path = self._master(mix_path, song_name, bpm=bpm)
        self._save_state(4, song_name, bpm, processed_paths, reverb_return_paths,
                         delay_return_paths, bus_paths, automated_bus_paths, mix_path,
                         master_path, stem_paths)

        # Clean up mix
        if mix_path and os.path.exists(mix_path):
            os.remove(mix_path)

        # ── Step 5: Mix variants ─────────────────────────────────────
        print("[Step 5] Creating Mix Variants...")
        self._create_variants(processed_paths, reverb_return_paths,
                              delay_return_paths, bus_paths, song_name, sr, bpm=bpm)

        # ── Copy final masters to final/ ────────────────────────────
        final_dir = os.path.join(self.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        final_copied = []
        if master_path and os.path.exists(master_path):
            dest = os.path.join(final_dir, os.path.basename(master_path))
            shutil.copy2(master_path, dest)
            final_copied.append(dest)
        # Copy variant masters
        for f in os.listdir(self.output_dir):
            if _is_retained_final_master(f) and f != os.path.basename(master_path or ""):
                src = os.path.join(self.output_dir, f)
                dest = os.path.join(final_dir, f)
                shutil.copy2(src, dest)
                final_copied.append(dest)
        if final_copied:
            print(f"\n  Final outputs ({len(final_copied)}):")
            for p in final_copied:
                print(f"    {p}")

        self.listener.write_reports(self.output_dir)

        print(f"\n✓ Production complete: {master_path}")
        return master_path

    def process_pristine_mix(self, stems: dict, song_name: str, bpm: float = 90.0) -> str:
        """Pristine mix — simplified path without creative FX."""
        print(f"\n--- PRODUCING PRISTINE: {song_name} ---")

        stem_paths = [p for p in stems.values() if p and os.path.exists(p)]
        if not stem_paths:
            return None

        pristine_dir = os.path.join(self.output_dir, "pristine_processed")
        os.makedirs(pristine_dir, exist_ok=True)

        processed_paths = []
        for path in stem_paths:
            name = os.path.basename(path)
            role = self._detect_role(name)

            y, sr = sf.read(path, always_2d=True)
            y = _ensure_stereo(np.asarray(y, dtype=np.float32))
            if np.max(np.abs(y)) < 1e-6:
                continue

            y_staged, _ = gain_stage_to_target(y, sr, role)
            y_comp = self.dsp.adaptive_compress(y_staged, sr, role, bpm=bpm)['y']
            y_comp, _ = self.dsp.gain_match(y_staged, y_comp)

            proc_path = os.path.join(pristine_dir, f"pristine_{name}")
            sf.write(proc_path, y_comp, sr, subtype='FLOAT')
            processed_paths.append(proc_path)

        # Sum
        mix_path = self._sum_to_mix(processed_paths, f"{self._output_song_name(song_name, bpm)}_pristine_sum")
        if not mix_path:
            return None

        # Master with preservation chain
        output_name = self._output_song_name(song_name, bpm)
        output_path = os.path.join(self.output_dir, f"{output_name}_pristine-mix.wav")
        render_preservation_master_file(
            mix_path,
            output_path,
            bpm=bpm,
            streaming_path=None,
            premaster_path=None,
            report_path=os.path.join(self.output_dir, "preservation_mastering_report_pristine.json"),
        )
        if os.path.exists(mix_path):
            os.remove(mix_path)

        # Copy to final/
        final_dir = os.path.join(self.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        dest = os.path.join(final_dir, os.path.basename(output_path))
        shutil.copy2(output_path, dest)

        print(f"✓ Pristine mix: {output_path}")
        return output_path

    # ── Bus Processing ───────────────────────────────────────────────

    def _process_buses(self, processed_paths: list, sr: int, bpm: float) -> list:
        """Group stems into buses, apply unmasking and iterative processing."""
        buses = self._group_into_buses(processed_paths)
        bus_dir = os.path.join(self.output_dir, "buses")
        os.makedirs(bus_dir, exist_ok=True)

        bus_paths = []
        bus_y_cache = {}
        stem_outlier_report = []

        for bus_name, paths in buses.items():
            if not paths:
                continue

            # Sum bus
            bus_y = None
            for p in paths:
                y, _ = sf.read(p, always_2d=True)
                y = _ensure_stereo(np.asarray(y, dtype=np.float64))
                y, stem_report = _protect_stem_outlier(y, sr, p, bus_name)
                if stem_report:
                    stem_outlier_report.append(stem_report)
                bus_y = y if bus_y is None else self._pad_and_add(bus_y, y)

            # Peak protect — let transients through to parallel comp
            peak = np.max(np.abs(bus_y))
            if peak > 0.99:
                bus_y = bus_y * (0.99 / peak)

            bus_y = bus_y.astype(np.float32)
            bus_y_cache[bus_name] = bus_y

        if stem_outlier_report:
            report_path = os.path.join(self.output_dir, "stem_outlier_report.json")
            _atomic_write_json(report_path, {
                "stage": "pre_bus_stem_outlier_protection",
                "reports": stem_outlier_report,
            })
            print(f"  Stem outlier protection: {len(stem_outlier_report)} trim(s), report {report_path}")

        # Unmasking between buses
        unmask_report = []
        for support_bus in ['melody', 'pads']:
            if support_bus in bus_y_cache and 'drums' in bus_y_cache:
                max_cut = self._adaptive_unmask_depth(
                    bus_y_cache[support_bus], bus_y_cache['drums'], sr, (2200, 3200), cap_db=1.5
                )
                bus_y_cache[support_bus] = self.dyn_eq.unmask(
                    bus_y_cache[support_bus], bus_y_cache['drums'], sr, (2200, 3200),
                    max_cut_db=max_cut, target_label=support_bus, masker_label="drums",
                    report=unmask_report
                )

            if support_bus in bus_y_cache and 'bass' in bus_y_cache:
                max_cut = self._adaptive_unmask_depth(
                    bus_y_cache[support_bus], bus_y_cache['bass'], sr, (120, 240), cap_db=1.0
                )
                bus_y_cache[support_bus] = self.dyn_eq.unmask(
                    bus_y_cache[support_bus], bus_y_cache['bass'], sr, (120, 240),
                    max_cut_db=max_cut, target_label=support_bus, masker_label="bass",
                    report=unmask_report
                )

        # Bass-drums unmasking — cut bass where kick lives
        if 'bass' in bus_y_cache and 'drums' in bus_y_cache:
            max_cut = self._adaptive_unmask_depth(
                bus_y_cache['bass'], bus_y_cache['drums'], sr, (50, 120), cap_db=3.5
            )
            bus_y_cache['bass'] = self.dyn_eq.unmask(
                bus_y_cache['bass'], bus_y_cache['drums'], sr, (50, 120),
                max_cut_db=max_cut, target_label="bass", masker_label="drums",
                report=unmask_report
            )

        if unmask_report:
            report_path = os.path.join(self.output_dir, "unmask_report.json")
            _atomic_write_json(report_path, {
                "stage": "bus_unmasking",
                "events": unmask_report,
            })

        # Objective listening bus corrections: measured low-end, masking, mono safety
        bus_diagnosis = self.listener.analyze_buses(bus_y_cache, sr, bpm)
        bus_actions = self._actions_by_target(bus_diagnosis.actions)
        objective_sidechain = None
        for action in bus_actions.get('bass', []):
            if action.action_type == 'sidechain':
                objective_sidechain = action
                break

        # Kick-bass sidechain — measured when confidence is high, configured fallback otherwise
        if 'bass' in bus_y_cache and 'drums' in bus_y_cache:
            ks_cfg = dict(config.KICK_BASS_SIDECHAIN)
            before_sidechain = bus_y_cache['bass'].copy()
            if objective_sidechain:
                params = objective_sidechain.params
                ks_cfg.update({
                    'depth_db': params.get('depth_db', ks_cfg['depth_db']),
                    'freq_range': tuple(params.get('freq_range', ks_cfg['freq_range'])),
                })
            ks_cfg['depth_db'] = min(max(float(ks_cfg.get('depth_db', 0.0)), 0.0), 3.5)
            ks_cfg['release_ms'] = config.KICK_BASS_SIDECHAIN['release_ms']
            bus_y_cache['bass'] = self.dsp.kick_bass_sidechain(
                bus_y_cache['bass'], bus_y_cache['drums'], sr,
                depth_db=ks_cfg['depth_db'],
                release_ms=ks_cfg['release_ms'],
                threshold_db=ks_cfg['threshold_db'],
                freq_range=ks_cfg['freq_range'],
            )
            if objective_sidechain:
                before_metrics = self.listener.measure_audio(before_sidechain, sr, "bus_bass_before")
                after_metrics = self.listener.measure_audio(bus_y_cache['bass'], sr, "bus_bass_after")
                accepted = self.listener.validate_action(objective_sidechain, before_metrics, after_metrics)
                if not accepted:
                    bus_y_cache['bass'] = before_sidechain
                result = "accepted" if accepted else "rejected"
                print(f"    Listening action: bus_bass {objective_sidechain.issue} -> sidechain "
                      f"conf={objective_sidechain.confidence:.2f} {objective_sidechain.params} [{result}]")

        for bus_name, actions in bus_actions.items():
            if bus_name in bus_y_cache:
                actions = [a for a in actions if a.action_type != 'sidechain']
                bus_y_cache[bus_name] = self._apply_listening_actions(
                    bus_y_cache[bus_name], sr, actions, source_name=f"bus_{bus_name}", bpm=bpm
                )

        # Save buses
        for bus_name, bus_y in bus_y_cache.items():
            out_path = os.path.join(bus_dir, f"bus_{bus_name}.wav")
            sf.write(out_path, bus_y, sr, subtype='FLOAT')
            bus_paths.append(out_path)

        return bus_paths

    def _group_into_buses(self, stem_paths: list) -> dict:
        """Categorize stems into logical buses."""
        buses = {'drums': [], 'bass': [], 'pads': [], 'melody': [], 'fx': []}
        for path in stem_paths:
            name = os.path.basename(path).lower()
            if any(x in name for x in ['kick', 'snare', 'hat', 'clap', 'drum',
                                        'bongo', 'conga', 'tambourine', 'maracas',
                                        'perc', 'side_stick']):
                buses['drums'].append(path)
            elif 'bass' in name:
                buses['bass'].append(path)
            elif any(x in name for x in ['pad', 'chord']):
                buses['pads'].append(path)
            elif any(x in name for x in ['melody', 'chorus', 'counter']):
                buses['melody'].append(path)
            else:
                buses['fx'].append(path)
        return buses

    def _apply_section_aware_pink_staging(self, processed_paths: list,
                                          reverb_return_paths: list,
                                          delay_return_paths: list,
                                          bus_paths: list,
                                          automated_bus_paths: list,
                                          sr: int,
                                          bpm: float,
                                          song_name: str) -> tuple:
        """Post-automation section pink staging plus section-aware wet-return control."""
        if not self.section_aware_pink_enabled:
            return bus_paths, automated_bus_paths, reverb_return_paths, delay_return_paths

        source_bus_paths = automated_bus_paths or bus_paths
        try:
            buses, bus_sr = _read_bus_paths(source_bus_paths)
        except Exception as exc:
            print(f"  Section-aware pink staging skipped: {exc}")
            return bus_paths, automated_bus_paths, reverb_return_paths, delay_return_paths

        sr = bus_sr or sr
        print("  Section-aware pink staging: post-automation buses, sidechain keys, wet returns")
        section_dir = os.path.join(self.output_dir, "section_automated_buses")
        key_dir = os.path.join(self.output_dir, "sidechain_keys")
        os.makedirs(section_dir, exist_ok=True)
        os.makedirs(key_dir, exist_ok=True)

        bus_pink_buses, bus_section_metrics, bus_section_deltas = _apply_bus_section_pink_trims(
            buses, sr, bpm, song_name
        )
        section_buses, section_metrics = _apply_full_dry_section_trims(
            bus_pink_buses, sr, bpm, song_name
        )
        peak_buses, section_peak_report = _apply_section_peak_controller(
            section_buses, sr, bpm, song_name
        )
        sidechain_buses, sidechain_report = _apply_sidechain_keys(
            peak_buses, processed_paths, key_dir, sr
        )

        section_bus_paths = []
        for name, y in sorted(sidechain_buses.items()):
            out_path = os.path.join(section_dir, f"auto_{name}.wav")
            sf.write(out_path, y.astype(np.float32), sr, subtype="FLOAT")
            section_bus_paths.append(out_path)

        dry_mix = None
        for y in sidechain_buses.values():
            dry_mix = y.copy() if dry_mix is None else _pad_and_add_arrays(dry_mix, y)
        if dry_mix is None:
            return bus_paths, automated_bus_paths, reverb_return_paths, delay_return_paths
        dry_mix = _peak_protect(dry_mix, ceiling_db=-7.0)

        raw_returns = _sum_audio_paths(list(reverb_return_paths or []) + list(delay_return_paths or []))
        raw_returns_path = None
        calibrated_returns_path = None
        return_report = []
        if raw_returns is not None:
            raw_returns = _pad_to(raw_returns, len(dry_mix))
            raw_returns_path = os.path.join(self.output_dir, "raw_summed_returns.wav")
            sf.write(raw_returns_path, raw_returns.astype(np.float32), sr, subtype="FLOAT")
            calibrated_returns, return_report = _calibrate_returns_by_section(
                dry_mix, raw_returns, sr, bpm, song_name
            )
            if calibrated_returns is not None:
                calibrated_returns_path = os.path.join(self.output_dir, "section_calibrated_returns.wav")
                sf.write(calibrated_returns_path, calibrated_returns.astype(np.float32), sr, subtype="FLOAT")
                reverb_return_paths = [calibrated_returns_path]
                delay_return_paths = []

        report = {
            "stage": "section_aware_pink_staging",
            "enabled": True,
            "bus_section_deltas": bus_section_deltas,
            "bus_section_pink_metrics": bus_section_metrics,
            "section_metrics": section_metrics,
            "section_peak_control": section_peak_report,
            "sidechain": sidechain_report,
            "return_calibration": return_report,
            "outputs": {
                "section_bus_paths": section_bus_paths,
                "raw_summed_returns": raw_returns_path,
                "section_calibrated_returns": calibrated_returns_path,
            },
        }
        report_path = os.path.join(self.output_dir, "section_aware_pink_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        _write_section_aware_markdown(report, os.path.join(self.output_dir, "SECTION_AWARE_PINK_REPORT.md"))
        print(f"  Section-aware pink report: {report_path}")
        return bus_paths, section_bus_paths, reverb_return_paths, delay_return_paths

    def _sum_with_optional_golden_post(self, processed_paths: list,
                                       reverb_return_paths: list,
                                       delay_return_paths: list,
                                       bus_paths: list,
                                       automated_bus_paths: list,
                                       song_name: str,
                                       bpm: float,
                                       sr: int) -> str:
        """Sum using the promoted golden post pathway when enabled."""
        if self.golden_post_enabled:
            try:
                params = load_params(
                    self.golden_post_params_path,
                    pan_seed_override=self.golden_pan_seed,
                    song_name=song_name,
                )
                print("  Golden post path enabled: dry stereo wall, return tightening, kick/bass ducking")
                processor = GoldenPostProcessor(self, params)
                report = processor.render_mix(
                    processed_paths=processed_paths,
                    reverb_paths=reverb_return_paths,
                    delay_paths=delay_return_paths,
                    bus_paths=bus_paths,
                    automated_bus_paths=automated_bus_paths,
                    song_name=song_name,
                    bpm=bpm,
                    sr=sr,
                )
                mix_path = report.get("mix_path")
                if mix_path:
                    print(f"  Golden post report: {os.path.join(self.output_dir, 'golden_post', 'golden_post_analysis.md')}")
                    return mix_path
                print(f"  Golden post unavailable ({report.get('reason', 'unknown')}); falling back to standard sum.")
            except Exception as exc:
                print(f"  WARNING: Golden post path failed: {exc}. Falling back to standard sum.")

        main_sources = automated_bus_paths or bus_paths or processed_paths
        all_mix_paths = list(reverb_return_paths or []) + list(delay_return_paths or []) + list(main_sources or [])
        return self._sum_to_mix(all_mix_paths, song_name, bpm=bpm)

    # ── Mastering ────────────────────────────────────────────────────

    def _master(self, mix_path: str, song_name: str, bpm: float = 90.0) -> str:
        """Preservation mastering with premaster and streaming-normalized output."""
        if not mix_path or not os.path.exists(mix_path):
            return None
        if not self.publish_full_master:
            print("  Full-song baseline master suppressed; May19 streaming master will be published by the professional controller.")
            return None

        output_name = self._output_song_name(song_name, bpm)
        premaster_path = os.path.join(self.output_dir, f"{output_name}_premaster.wav")
        output_path = os.path.join(self.output_dir, f"{output_name}_master.wav")
        streaming_path = os.path.join(self.output_dir, f"{output_name}_streaming_master.wav")
        report_path = os.path.join(self.output_dir, "preservation_mastering_report.json")
        _, report = render_preservation_master_file(
            mix_path,
            output_path,
            bpm=bpm,
            streaming_path=streaming_path,
            premaster_path=premaster_path,
            report_path=report_path,
        )

        print(f"  Premaster: {premaster_path}")
        print(f"  Master: LUFS={report.get('master_lufs', 0):+.1f} | "
              f"Peak={report.get('master_peak_db', 0):+.1f} dBFS | preservation chain")
        for band, stats in report.get("multiband_peak_control", {}).items():
            print(f"    {band:4s} comp: avg GR={stats.get('avg_gr_db', 0):.2f}dB "
                  f"max GR={stats.get('max_gr_db', 0):.2f}dB "
                  f"makeup={stats.get('makeup_db', 0):.2f}dB")
        streaming_debug = report.get("streaming") or {}
        print(f"  Streaming master: LUFS={streaming_debug.get('output_lufs', 0):+.1f} | "
              f"Peak={streaming_debug.get('output_peak_db', 0):+.1f} dBFS | "
              f"gain={streaming_debug.get('lufs_gain_db', 0):+.1f}dB "
              f"peak safety={streaming_debug.get('peak_safety_db', 0):+.1f}dB")

        return output_path

    # ── Mix Variants ─────────────────────────────────────────────────

    def _create_variants(self, processed_paths, reverb_paths, delay_paths,
                         bus_paths, song_name, sr, bpm: float = 90.0):
        """Create retained utility variants: drums+bass and pads+drums+bass."""
        variants = [
            {
                "suffix": "drums-bass",
                "label": "Drums + Bass Mix",
                "include_pads": False,
            },
            {
                "suffix": "pads-drums-bass",
                "label": "Pads/Chords + Drums + Bass Mix",
                "include_pads": True,
            },
        ]

        fx_paths = list(reverb_paths or []) + list(delay_paths or [])
        for var in variants:
            v_stems = [
                p for p in processed_paths
                if self._is_any_bass_stem(p)
                or self._is_prefixed_drum_stem(p)
                or (var["include_pads"] and self._is_pad_chord_stem(p))
            ]
            v_fx = [
                p for p in fx_paths
                if self._is_prefixed_drum_stem(self._variant_return_source_name(p))
                or (var["include_pads"] and self._is_pad_chord_stem(self._variant_return_source_name(p)))
            ]
            v_all = self._unique_paths(v_stems + v_fx)
            if not v_all:
                continue

            print(f"  Generating {var['label']}...")
            v_mix = self._sum_to_mix(v_all, song_name, suffix=var['suffix'], bpm=bpm)
            if v_mix:
                self._master_variant(v_mix, song_name, var['suffix'], bpm=bpm)

    def _variant_basename(self, path_or_name: str) -> str:
        return os.path.basename(path_or_name or "").lower()

    def _variant_return_source_name(self, path_or_name: str) -> str:
        name = self._variant_basename(path_or_name)
        for prefix in ("reverb_return_", "delay_return_", "proc_"):
            if name.startswith(prefix):
                return name[len(prefix):]
        return name

    def _is_any_bass_stem(self, path_or_name: str) -> bool:
        name = self._variant_return_source_name(path_or_name)
        return "_bass" in name

    def _is_bass1_stem(self, path_or_name: str) -> bool:
        name = self._variant_return_source_name(path_or_name)
        return "_bass" in name and "_harmonic_bass" not in name

    def _is_bass2_stem(self, path_or_name: str) -> bool:
        return "_harmonic_bass" in self._variant_return_source_name(path_or_name)

    def _is_pad_chord_stem(self, path_or_name: str) -> bool:
        name = self._variant_return_source_name(path_or_name)
        return "pad" in name or "chord" in name

    def _is_prefixed_drum_stem(self, path_or_name: str) -> bool:
        name = self._variant_return_source_name(path_or_name)
        return any(prefix in name for prefix in ("drum1_", "drum2_", "drum_aux_"))

    def _unique_paths(self, paths: list) -> list:
        seen = set()
        result = []
        for path in paths:
            if not path or path in seen:
                continue
            seen.add(path)
            result.append(path)
        return result

    def _master_variant(self, mix_path: str, song_name: str, suffix: str,
                        bpm: float = 90.0) -> str:
        """Master a mix variant with the preservation chain."""
        output_name = self._output_song_name(song_name, bpm)
        output_path = os.path.join(self.output_dir, f"{output_name}_{suffix}_streaming_master.wav")
        report_path = os.path.join(self.output_dir, f"preservation_mastering_report_{suffix}.json")
        render_preservation_master_file(
            mix_path,
            None,
            bpm=bpm,
            streaming_path=output_path,
            premaster_path=None,
            report_path=report_path,
        )
        if os.path.exists(mix_path):
            os.remove(mix_path)
        return output_path

    # ── Summing ──────────────────────────────────────────────────────

    def _sum_to_mix(self, paths: list, song_name: str, suffix: str = "",
                    bpm: float = None) -> str:
        """Sum all paths into a single stereo mix."""
        s_part = f"_{suffix}" if suffix else ""
        output_name = self._output_song_name(song_name, bpm)
        output_path = os.path.join(self.output_dir, f"{output_name}{s_part}_mix.wav")

        valid = [p for p in paths if p and os.path.exists(p)]
        if not valid:
            return None

        mix = None
        sr = 48000
        for p in valid:
            y, sr = sf.read(p, always_2d=True)
            y = np.asarray(y, dtype=np.float64)
            mix = y if mix is None else self._pad_and_add(mix, y)

        # Peak protect
        peak = np.max(np.abs(mix))
        headroom = 10 ** (-6.0 / 20.0)
        if peak > headroom:
            mix *= headroom / peak

        sf.write(output_path, mix.astype(np.float32), sr, subtype='FLOAT')
        return output_path

    # ── Helpers ──────────────────────────────────────────────────────

    def _detect_role(self, name: str) -> str:
        """Detect stem role from filename."""
        n = name.lower()
        if 'kick' in n:
            return 'kick'
        if 'snare' in n:
            return 'snare'
        if 'hat' in n:
            return 'hat'
        if 'clap' in n:
            return 'clap'
        if 'bass' in n:
            return 'bass'
        if 'pad' in n:
            return 'pad'
        if 'chord' in n:
            return 'chord'
        if 'melody' in n:
            return 'melody'
        if 'counter' in n:
            return 'counter'
        if 'chorus' in n:
            return 'chorus'
        if 'fx' in n:
            return 'fx'
        return 'default'

    def _detect_layer(self, name: str) -> str:
        """Detect layer-specific role from filename. Returns layer key or None."""
        n = name.lower()

        # Kick layers
        if 'kick' in n:
            if '_sub' in n:
                return 'kick_sub'
            if '_punch' in n:
                return 'kick_punch'
            if '_click' in n:
                return 'kick_click'

        # Snare layers
        if 'snare' in n:
            if '_body' in n:
                return 'snare_body'
            if '_snap' in n:
                return 'snare_snap'
            if '_air' in n:
                return 'snare_air'

        # Melody layers
        if 'main_melody' in n or ('melody' in n and 'counter' not in n and 'chorus' not in n and 'fx' not in n):
            if 'layer1' in n or '_lead' in n:
                return 'melody_lead'
            if 'layer2' in n or '_poly' in n:
                return 'melody_poly'

        # Counter melody layers
        if 'counter' in n:
            if 'layer1' in n or '_lead' in n:
                return 'counter_lead'
            if 'layer2' in n or '_bell' in n:
                return 'counter_bell'
            if 'layer3' in n or '_pluck' in n:
                return 'counter_pluck'

        # Chorus melody layers
        if 'chorus' in n:
            if 'layer1' in n or '_poly' in n:
                return 'chorus_poly'
            if 'layer2' in n or '_brass' in n:
                return 'chorus_brass'

        # Pad layers
        if 'pad' in n or 'chord' in n:
            if 'layer1' in n:
                return 'pad_layer1'
            if 'layer2' in n:
                return 'pad_layer2'
            if 'layer3' in n:
                return 'pad_layer3'

        return None

    def _get_layer_preset(self, name: str) -> dict:
        """Get layer-specific preset. Returns None if no layer detected."""
        layer = self._detect_layer(name)
        if layer and layer in config.LAYER_PRESETS:
            return config.LAYER_PRESETS[layer]
        return None

    def _get_send_config(self, name: str) -> dict:
        """Get reverb/delay send configuration for a stem."""
        n = name.lower()
        result = {
            'reverb_category': None,
            'reverb_send': 0.0,
            'delay_send': 0.0,
            'delay_type': 'melodic',
        }

        for match, category, rev_send, dly_send in config.STEM_SEND_MAP:
            if match in n:
                result['reverb_category'] = category
                result['reverb_send'] = rev_send
                result['delay_send'] = dly_send
                if 'counter' in n:
                    result['delay_type'] = 'counter'
                elif 'chorus' in n:
                    result['delay_type'] = 'chorus'
                elif 'fx' in n:
                    result['delay_type'] = 'fx'
                break

        return result

    def _output_song_name(self, song_name: str, bpm: float = None) -> str:
        """Append BPM to output file stems when the song name does not already include it."""
        base = str(song_name or "song")
        if re.search(r'(?i)(^|[_-])\d+(?:\.\d+)?bpm($|[_-])', base):
            return base
        if bpm is None:
            return base
        return f"{base}_{int(round(float(bpm)))}bpm"

    def _actions_by_target(self, actions: list) -> dict:
        grouped = {}
        for action in actions or []:
            grouped.setdefault(action.target, []).append(action)
        return grouped

    def _adaptive_unmask_depth(self, target_y: np.ndarray, masker_y: np.ndarray,
                               sr: int, freq_range: tuple, cap_db: float) -> float:
        self.dyn_eq.sr = sr
        target_db = self.dyn_eq._measure_band_range_energy(target_y, freq_range[0], freq_range[1])
        masker_db = self.dyn_eq._measure_band_range_energy(masker_y, freq_range[0], freq_range[1])
        if target_db <= -45.0 or masker_db <= -45.0:
            return 0.0
        overlap_conf = np.clip((min(target_db, masker_db) + 45.0) / 24.0, 0.0, 1.0)
        dominance = max(0.0, masker_db - target_db)
        depth = float(np.clip((0.25 + overlap_conf * cap_db + dominance * 0.04), 0.0, cap_db))
        if depth >= 0.25:
            print(f"    Adaptive unmask {freq_range[0]}-{freq_range[1]}Hz: "
                  f"target={target_db:+.1f}dB masker={masker_db:+.1f}dB depth={depth:.2f}dB")
        return depth

    def _apply_width(self, y: np.ndarray, width: float) -> np.ndarray:
        y = _ensure_stereo(y)
        if abs(width - 1.0) < 0.01:
            return y
        mid = (y[:, 0] + y[:, 1]) * 0.5
        side = (y[:, 0] - y[:, 1]) * 0.5 * width
        out = np.stack([mid + side, mid - side], axis=1)
        peak = np.max(np.abs(out))
        if peak > 0.98:
            out = out * (0.95 / peak)
        return out.astype(np.float32)

    def _apply_listening_actions(self, y: np.ndarray, sr: int, actions: list,
                                 source_name: str = "", bpm: float = 90.0) -> np.ndarray:
        if not actions:
            return y
        y_out = _ensure_stereo(np.asarray(y, dtype=np.float32))
        for action in actions:
            before = y_out.copy()
            before_metrics = self.listener.measure_audio(before, sr, f"{source_name}_before")
            params = action.params
            if action.action_type == 'dynamic_eq':
                y_out = self.dsp.apply_bell_eq(
                    y_out, sr,
                    freq=float(params.get('freq', 1000.0)),
                    gain_db=float(params.get('gain_db', 0.0)),
                    q=float(params.get('q', 1.0)),
                )
                y_out, _ = self.dsp.gain_match(before, y_out, max_correction_db=1.0)
            elif action.action_type == 'gain_trim':
                gain_db = float(params.get('gain_db', 0.0))
                y_out = y_out * (10.0 ** (gain_db / 20.0))
            elif action.action_type == 'mono_sub':
                y_out = self.stereo.mono_sub(y_out, sr, cutoff=float(params.get('cutoff', 90.0)))
            elif action.action_type == 'saturate':
                y_out = self.dsp.saturate(y_out, role='default',
                                          amount_override=float(params.get('amount', 1.03)))
                y_out, _ = self.dsp.gain_match(before, y_out, max_correction_db=1.0)
            elif action.action_type == 'duck_return':
                depth_db = float(params.get('depth_db', 1.0))
                y_out = y_out * (10.0 ** (-depth_db / 20.0))
            elif action.action_type == 'width_adjust':
                width = float(params.get('width', 1.0))
                y_out = self._apply_width(y_out, width)
            elif action.action_type == 'skip':
                action.accepted = False
                action.validation = {'reason': 'analysis_only'}
                print(f"    Listening action: {source_name} {action.issue} -> skip "
                      f"conf={action.confidence:.2f} {params} [analysis]")
                continue
            peak = np.max(np.abs(y_out))
            if peak > 0.99:
                y_out = y_out * (0.98 / peak)
            after_metrics = self.listener.measure_audio(y_out, sr, f"{source_name}_after")
            accepted = self.listener.validate_action(action, before_metrics, after_metrics)
            if not accepted:
                y_out = before
            result = "accepted" if accepted else "rejected"
            print(f"    Listening action: {source_name} {action.issue} -> {action.action_type} "
                  f"conf={action.confidence:.2f} {params} [{result}]")
        return y_out.astype(np.float32)

    def _process_returns_objectively(self, processed_paths: list, reverb_paths: list,
                                     delay_paths: list, sr: int, bpm: float) -> tuple:
        return_paths = list(reverb_paths or []) + list(delay_paths or [])
        diagnosis = self.listener.analyze_returns(processed_paths, return_paths, sr, bpm)
        actions_by_target = self._actions_by_target(diagnosis.actions)
        if not actions_by_target:
            return reverb_paths, delay_paths

        def process_group(paths):
            out_paths = []
            for path in paths:
                name = os.path.basename(path)
                actions = actions_by_target.get(name, [])
                if not actions:
                    out_paths.append(path)
                    continue
                y, file_sr = sf.read(path, always_2d=True)
                y = _ensure_stereo(np.asarray(y, dtype=np.float32))
                for action in actions:
                    if action.action_type == 'filter_return':
                        eq_chain = [
                            {'type': 'highpass', 'freq': action.params.get('highpass', 150.0)},
                            {'type': 'lowpass', 'freq': action.params.get('lowpass', 8500.0)},
                        ]
                        y_before = y.copy()
                        before_metrics = self.listener.measure_audio(y_before, file_sr, f"{name}_before")
                        y = self.dsp.layer_eq(y, file_sr, eq_chain)
                        y, _ = self.dsp.gain_match(y_before, y, max_correction_db=1.0)
                        gain_db = float(action.params.get('gain_db', 0.0))
                        y = y * (10.0 ** (gain_db / 20.0))
                        after_metrics = self.listener.measure_audio(y, file_sr, f"{name}_after")
                        accepted = self.listener.validate_action(action, before_metrics, after_metrics)
                        if not accepted:
                            y = y_before
                        result = "accepted" if accepted else "rejected"
                        print(f"  Return listening action: {name} {action.issue} "
                              f"conf={action.confidence:.2f} {action.params} [{result}]")
                sf.write(path, y.astype(np.float32), file_sr, subtype='FLOAT')
                out_paths.append(path)
            return out_paths

        return process_group(reverb_paths or []), process_group(delay_paths or [])

    def _prepare_premaster_objectively(self, mix_path: str, song_name: str,
                                       sr: int, bpm: float) -> str:
        if not mix_path or not os.path.exists(mix_path):
            return mix_path
        y, file_sr = sf.read(mix_path, always_2d=True)
        y = _ensure_stereo(np.asarray(y, dtype=np.float32))
        diagnosis = self.listener.analyze_premaster(y, file_sr, bpm, self.reference_analysis)
        if diagnosis.actions:
            y = self._apply_listening_actions(y, file_sr, diagnosis.actions,
                                              source_name='premaster', bpm=bpm)
            sf.write(mix_path, y.astype(np.float32), file_sr, subtype='FLOAT')
        return mix_path

    def _pad_and_add(self, a, b):
        """Add two arrays of potentially different lengths."""
        if len(a) < len(b):
            res = np.zeros_like(b)
            res[:len(a)] = a
            res += b
        else:
            res = np.zeros_like(a)
            res[:len(b)] = b
            res += a
        return res


def _db_to_amp(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def _amp_to_db(x) -> float:
    return 20.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float64), 1e-12))


def _rms_db(y: np.ndarray) -> float:
    y = _ensure_stereo(np.asarray(y, dtype=np.float32))
    return float(_amp_to_db(np.sqrt(np.mean(np.square(y), dtype=np.float64))))


def _peak_db(y: np.ndarray) -> float:
    y = _ensure_stereo(np.asarray(y, dtype=np.float32))
    return float(_amp_to_db(np.max(np.abs(y))))


def _beats_to_ms(bpm: float, beats: float) -> float:
    return (60.0 / max(float(bpm), 1.0)) * float(beats) * 1000.0


def _pad_to(y: np.ndarray, n: int) -> np.ndarray:
    y = _ensure_stereo(np.asarray(y, dtype=np.float32))
    if len(y) >= n:
        return y[:n]
    return np.pad(y, ((0, n - len(y)), (0, 0)))


def _pad_and_add_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = max(len(a), len(b))
    return _pad_to(a, n) + _pad_to(b, n)


def _peak_protect(y: np.ndarray, ceiling_db: float = -6.0) -> np.ndarray:
    pk = _peak_db(y)
    if pk <= ceiling_db:
        return y.astype(np.float32)
    return (y * _db_to_amp(ceiling_db - pk)).astype(np.float32)


def _section_kind(name: str) -> str:
    if "pre_chorus" in name or "build" in name:
        return "pre_chorus"
    if "chorus" in name:
        return "chorus"
    if "fill" in name:
        return "fill"
    if "intro" in name:
        return "intro"
    if "outro" in name:
        return "outro"
    return "verse"


def _beats_per_bar_from_name(song_name: str) -> float:
    if "_5-8_" in song_name:
        return 2.5
    for token, beats in (("_3-4_", 3.0), ("_5-4_", 5.0), ("_4-4_", 4.0)):
        if token in song_name:
            return beats
    return 4.0


def _section_slices(length: int, sr: int, bpm: float, song_name: str) -> dict:
    samples_per_bar = int(sr * (60.0 / max(float(bpm), 1.0)) * _beats_per_bar_from_name(song_name))
    out = {}
    for name, (bar_start, bar_end) in SECTION_BARS.items():
        start = min(length, bar_start * samples_per_bar)
        end = min(length, bar_end * samples_per_bar)
        if end > start:
            out[name] = (start, end)
    return out


def _smooth_gain_curve(gain_db: np.ndarray, sr: int, fade_ms: float = 70.0) -> np.ndarray:
    fade = max(16, int(sr * fade_ms / 1000.0))
    curve = np.power(10.0, gain_db / 20.0).astype(np.float32)
    if len(curve) <= fade * 2:
        return curve
    kernel = np.hanning(fade * 2 + 1).astype(np.float32)
    kernel /= np.sum(kernel)
    return np.convolve(curve, kernel, mode="same").astype(np.float32)


def _read_bus_paths(paths: list) -> tuple:
    buses = {}
    sr = None
    for path in sorted(paths or []):
        if not path or not os.path.exists(path):
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        name = name.replace("auto_", "").replace("bus_", "")
        y, file_sr = sf.read(path, always_2d=True)
        sr = file_sr
        buses[name] = _ensure_stereo(np.asarray(y, dtype=np.float32))
    if not buses:
        raise FileNotFoundError("no bus paths found")
    length = max(len(y) for y in buses.values())
    return {name: _pad_to(y, length) for name, y in buses.items()}, sr


def _sum_audio_paths(paths: list) -> np.ndarray | None:
    total = None
    for path in paths or []:
        if not path or not os.path.exists(path):
            continue
        y, _ = sf.read(path, always_2d=True)
        y = _ensure_stereo(np.asarray(y, dtype=np.float32))
        total = y if total is None else _pad_and_add_arrays(total, y)
    return total.astype(np.float32) if total is not None else None


def _band_energy_db(y: np.ndarray, sr: int, low_hz: float, high_hz: float) -> float:
    y = _ensure_stereo(y)
    nyq = sr * 0.5
    low = max(20.0, float(low_hz)) / nyq
    high = min(float(high_hz), nyq * 0.98) / nyq
    if high <= low:
        return _rms_db(y)
    sos = butter(3, [low, high], btype="bandpass", output="sos")
    filt = np.column_stack([sosfilt(sos, y[:, ch]) for ch in range(2)])
    return _rms_db(filt)


def _role_pink_score_db(y: np.ndarray, sr: int, bus_name: str) -> float:
    cfg = BUS_SECTION_TARGETS.get(bus_name, BUS_SECTION_TARGETS["melody"])
    values = []
    weights = []
    for low_hz, high_hz, weight in cfg["bands"]:
        center = math.sqrt(float(low_hz) * float(high_hz))
        pink_comp = 3.0103 * math.log2(center / 1000.0)
        values.append(_band_energy_db(y, sr, low_hz, high_hz) + pink_comp)
        weights.append(float(weight))
    return float(np.average(np.asarray(values, dtype=np.float64), weights=np.asarray(weights, dtype=np.float64)))


def _section_anchor(section: str) -> str | None:
    if section.startswith("verse"):
        return section
    if section == "intro":
        return "verse1_a"
    if section == "outro":
        return "verse2_b"
    if section.endswith("1") or "1_" in section or "_1" in section:
        return "verse1_a"
    if section.endswith("2") or "2_" in section or "_2" in section:
        return "verse2_a"
    return None


def _metric_lookup(metrics: list, bus_name: str, section: str | None) -> dict | None:
    if not section:
        return None
    for row in metrics:
        if row["bus"] == bus_name and row["section"] == section:
            return row
    return None


def _apply_bus_section_pink_trims(buses: dict, sr: int, bpm: float, song_name: str) -> tuple:
    length = max(len(y) for y in buses.values())
    slices = _section_slices(length, sr, bpm, song_name)
    out = {name: _pad_to(y, length).copy() for name, y in buses.items()}
    metrics = []

    for bus_name, y in out.items():
        if bus_name not in BUS_SECTION_TARGETS:
            continue
        for section, (start, end) in slices.items():
            seg = y[start:end]
            metrics.append({
                "bus": bus_name,
                "section": section,
                "kind": _section_kind(section),
                "rms_before_db": _rms_db(seg),
                "pink_score_before_db": _role_pink_score_db(seg, sr, bus_name),
            })

    for bus_name, y in list(out.items()):
        cfg = BUS_SECTION_TARGETS.get(bus_name)
        if not cfg:
            continue
        bus_rows = [m for m in metrics if m["bus"] == bus_name]
        verse_pink = [m["pink_score_before_db"] for m in bus_rows if m["kind"] == "verse"]
        verse_rms = [m["rms_before_db"] for m in bus_rows if m["kind"] == "verse"]
        ref_pink = float(np.median(verse_pink)) if verse_pink else -24.0
        ref_rms = float(np.median(verse_rms)) if verse_rms else -24.0
        gain_db = np.zeros(length, dtype=np.float32)
        for row in bus_rows:
            section = row["section"]
            anchor_row = _metric_lookup(metrics, bus_name, _section_anchor(section))
            local_ref_pink = float(anchor_row["pink_score_before_db"]) if anchor_row else ref_pink
            local_ref_rms = float(anchor_row["rms_before_db"]) if anchor_row else ref_rms
            allowed_lift = float(cfg["lift_db"].get(row["kind"], 0.0))
            pink_gain = (local_ref_pink + allowed_lift) - row["pink_score_before_db"]
            rms_gain = (local_ref_rms + allowed_lift) - row["rms_before_db"]
            trim = float(np.clip(min(pink_gain, rms_gain), float(cfg["max_trim_db"]), float(cfg["max_boost_db"])))
            start, end = slices[section]
            gain_db[start:end] = trim
            row["allowed_lift_db"] = allowed_lift
            row["bus_section_gain_db"] = trim
        curve = _smooth_gain_curve(gain_db, sr, fade_ms=70.0)
        out[bus_name] = (y * curve[:, None]).astype(np.float32)

    for row in metrics:
        start, end = slices[row["section"]]
        seg = out[row["bus"]][start:end]
        row["rms_after_db"] = _rms_db(seg)
        row["pink_score_after_db"] = _role_pink_score_db(seg, sr, row["bus"])
    return out, metrics, _bus_section_deltas(metrics)


def _bus_section_deltas(metrics: list) -> dict:
    by_bus_section = {(m["bus"], m["section"]): m for m in metrics}
    out = {}
    for bus in sorted({m["bus"] for m in metrics}):
        rows = []
        for verse, chorus in [("verse1_a", "chorus1"), ("verse2_a", "chorus2")]:
            v = by_bus_section.get((bus, verse))
            c = by_bus_section.get((bus, chorus))
            if not v or not c:
                continue
            rows.append({
                "verse": verse,
                "chorus": chorus,
                "pink_delta_before_db": c["pink_score_before_db"] - v["pink_score_before_db"],
                "pink_delta_after_db": c["pink_score_after_db"] - v["pink_score_after_db"],
                "rms_delta_before_db": c["rms_before_db"] - v["rms_before_db"],
                "rms_delta_after_db": c["rms_after_db"] - v["rms_after_db"],
            })
        out[bus] = rows
    return out


def _role_rms_ceiling_db(bus_name: str) -> float:
    if bus_name == "drums":
        return -16.0
    if bus_name == "bass":
        return -15.0
    if bus_name == "pads":
        return -17.0
    if bus_name == "melody":
        return -17.0
    return -18.0


def _protect_stem_outlier(y: np.ndarray, sr: int, path: str, bus_name: str) -> tuple:
    y = _ensure_stereo(y)
    peak_before = _peak_db(y)
    rms_before = _rms_db(y)
    peak_trim = -1.0 - peak_before if peak_before > -1.0 else 0.0
    rms_ceiling = _role_rms_ceiling_db(bus_name)
    rms_trim = rms_ceiling - rms_before if rms_before > rms_ceiling else 0.0
    trim = float(np.clip(min(0.0, peak_trim, rms_trim), -12.0, 0.0))
    if trim >= -0.05:
        return y, None
    out = (y * _db_to_amp(trim)).astype(np.float32)
    report = {
        "path": path,
        "file": os.path.basename(path),
        "bus": bus_name,
        "peak_before_db": float(peak_before),
        "rms_before_db": float(rms_before),
        "peak_ceiling_db": -1.0,
        "rms_ceiling_db": float(rms_ceiling),
        "trim_db": trim,
        "peak_after_db": float(_peak_db(out)),
        "rms_after_db": float(_rms_db(out)),
    }
    return out, report


def _apply_full_dry_section_trims(buses: dict, sr: int, bpm: float, song_name: str) -> tuple:
    length = max(len(y) for y in buses.values())
    slices = _section_slices(length, sr, bpm, song_name)
    dry = None
    for y in buses.values():
        dry = _pad_to(y, length) if dry is None else dry + _pad_to(y, length)
    verse_levels = [_rms_db(dry[s:e]) for sec, (s, e) in slices.items() if _section_kind(sec) == "verse"]
    verse_ref = float(np.median(verse_levels)) if verse_levels else _rms_db(dry)
    gain_db = np.zeros(length, dtype=np.float32)
    metrics = []
    for sec, (start, end) in slices.items():
        kind = _section_kind(sec)
        dry_level = _rms_db(dry[start:end])
        allowed = SECTION_ALLOWED_LIFT_DB[kind]
        trim = float(np.clip(min(0.0, (verse_ref + allowed) - dry_level), -5.5, 0.0))
        gain_db[start:end] = trim
        metrics.append({"section": sec, "kind": kind, "dry_rms_db_before": dry_level, "section_trim_db": trim})
    curve = _smooth_gain_curve(gain_db, sr)
    out = {name: (y * curve[:, None]).astype(np.float32) for name, y in buses.items()}
    dry_after = None
    for y in out.values():
        dry_after = y if dry_after is None else dry_after + y
    for row in metrics:
        start, end = slices[row["section"]]
        row["dry_rms_db_after"] = _rms_db(dry_after[start:end])
        row["peak_db_after"] = _peak_db(dry_after[start:end])
    return out, metrics


def _apply_section_peak_controller(buses: dict, sr: int, bpm: float, song_name: str) -> tuple:
    length = max(len(y) for y in buses.values())
    slices = _section_slices(length, sr, bpm, song_name)
    dry = None
    for y in buses.values():
        dry = _pad_to(y, length) if dry is None else dry + _pad_to(y, length)
    if dry is None:
        return buses, []

    verse_rms = [
        _rms_db(dry[start:end])
        for sec, (start, end) in slices.items()
        if _section_kind(sec) == "verse"
    ]
    verse_ref = float(np.median(verse_rms)) if verse_rms else _rms_db(dry)
    gain_db = np.zeros(length, dtype=np.float32)
    report = []

    for sec, (start, end) in slices.items():
        kind = _section_kind(sec)
        seg = dry[start:end]
        peak_before = _peak_db(seg)
        rms_before = _rms_db(seg)
        crest_before = peak_before - rms_before
        peak_ceiling = SECTION_PEAK_CEILINGS_DB[kind]
        allowed_lift = SECTION_ALLOWED_LIFT_DB[kind]
        peak_trim = peak_ceiling - peak_before if peak_before > peak_ceiling else 0.0
        lift_trim = (verse_ref + allowed_lift) - rms_before if rms_before > verse_ref + allowed_lift else 0.0
        requested_trim = min(0.0, peak_trim, lift_trim)
        trim = float(np.clip(requested_trim, SECTION_PEAK_MAX_TRIM_DB[kind], 0.0))
        gain_db[start:end] = trim
        report.append({
            "section": sec,
            "kind": kind,
            "peak_before_db": float(peak_before),
            "rms_before_db": float(rms_before),
            "crest_before_db": float(crest_before),
            "verse_ref_rms_db": float(verse_ref),
            "allowed_lift_db": float(allowed_lift),
            "peak_ceiling_db": float(peak_ceiling),
            "requested_trim_db": float(requested_trim),
            "section_peak_trim_db": trim,
            "hit_trim_cap": bool(requested_trim < SECTION_PEAK_MAX_TRIM_DB[kind]),
        })

    curve = _smooth_gain_curve(gain_db, sr, fade_ms=_beats_to_ms(bpm, 1.0))
    out = {name: (_pad_to(y, length) * curve[:, None]).astype(np.float32) for name, y in buses.items()}
    dry_after = None
    for y in out.values():
        dry_after = y if dry_after is None else dry_after + y
    for row in report:
        start, end = slices[row["section"]]
        seg = dry_after[start:end]
        row["peak_after_db"] = float(_peak_db(seg))
        row["rms_after_db"] = float(_rms_db(seg))
        row["crest_after_db"] = float(row["peak_after_db"] - row["rms_after_db"])
    return out, report


def _collect_key_bus(processed_paths: list, keywords: tuple) -> tuple:
    paths = [p for p in processed_paths if p and os.path.exists(p) and any(k in os.path.basename(p).lower() for k in keywords)]
    return _sum_audio_paths(paths), [os.path.basename(p) for p in paths]


def _envelope_from_key(key: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    key = _ensure_stereo(key)
    nyq = sr * 0.5
    sos = butter(3, [max(20.0, low_hz) / nyq, min(high_hz, nyq * 0.98) / nyq], btype="bandpass", output="sos")
    band = np.column_stack([sosfilt(sos, key[:, ch]) for ch in range(2)])
    mono = np.max(np.abs(band), axis=1)
    attack = math.exp(-1.0 / max(1.0, sr * 0.006))
    release = math.exp(-1.0 / max(1.0, sr * 0.020))
    env = np.zeros_like(mono, dtype=np.float32)
    for i, value in enumerate(mono):
        prev = env[i - 1] if i else value
        coeff = attack if value > prev else release
        env[i] = coeff * prev + (1.0 - coeff) * value
    active = env[env > np.percentile(env, 70)] if env.size else env
    scale = float(np.percentile(active, 95)) if active.size else 1.0
    return np.clip(env / max(scale, 1e-6), 0.0, 1.0)


def _dynamic_band_duck(target: np.ndarray, key: np.ndarray, sr: int, low_hz: float, high_hz: float, depth_db: float) -> np.ndarray:
    target = _ensure_stereo(target)
    key = _pad_to(key, len(target))
    nyq = sr * 0.5
    low = max(20.0, low_hz) / nyq
    high = min(high_hz, nyq * 0.98) / nyq
    if high <= low:
        return target
    band_sos = butter(4, [low, high], btype="bandpass", output="sos")
    low_sos = butter(4, low, btype="lowpass", output="sos")
    high_sos = butter(4, high, btype="highpass", output="sos")
    band = np.column_stack([sosfilt(band_sos, target[:, ch]) for ch in range(2)])
    low_part = np.column_stack([sosfilt(low_sos, target[:, ch]) for ch in range(2)])
    high_part = np.column_stack([sosfilt(high_sos, target[:, ch]) for ch in range(2)])
    env = _envelope_from_key(key, sr, low_hz, high_hz)
    gain = np.power(10.0, (-abs(depth_db) * env) / 20.0).astype(np.float32)
    return (low_part + band * gain[:, None] + high_part).astype(np.float32)


def _apply_sidechain_keys(buses: dict, processed_paths: list, key_dir: str, sr: int) -> tuple:
    out = {name: y.copy() for name, y in buses.items()}
    report = {}
    kick_key, kick_sources = _collect_key_bus(processed_paths, ("kick",))
    if kick_key is not None:
        sf.write(os.path.join(key_dir, "key_kick.wav"), kick_key.astype(np.float32), sr, subtype="FLOAT")
        kick_depth_db = float(config.KICK_BASS_SIDECHAIN.get("depth_db", 2.4))
        for target in ("bass",):
            if target in out:
                before = out[target].copy()
                out[target] = _dynamic_band_duck(out[target], kick_key, sr, 45.0, 140.0, depth_db=kick_depth_db)
                report[f"kick_to_{target}"] = {"sources": kick_sources, "band_hz": [45.0, 140.0], "depth_db": kick_depth_db, "target_rms_delta_db": _rms_db(out[target]) - _rms_db(before)}
    snare_key, snare_sources = _collect_key_bus(processed_paths, ("snare", "clap", "side_stick", "sidestick"))
    if snare_key is not None:
        sf.write(os.path.join(key_dir, "key_snare.wav"), snare_key.astype(np.float32), sr, subtype="FLOAT")
        for target in ("melody", "pads"):
            if target in out:
                before = out[target].copy()
                out[target] = _dynamic_band_duck(out[target], snare_key, sr, 850.0, 4200.0, depth_db=1.8)
                report[f"snare_to_{target}"] = {"sources": snare_sources, "band_hz": [850.0, 4200.0], "depth_db": 1.8, "target_rms_delta_db": _rms_db(out[target]) - _rms_db(before)}
    return out, report


def _calibrate_returns_by_section(dry_mix: np.ndarray, returns: np.ndarray | None, sr: int, bpm: float, song_name: str) -> tuple:
    if returns is None:
        return None, []
    max_cut_db = -22.0
    returns = _pad_to(returns, len(dry_mix))
    slices = _section_slices(len(dry_mix), sr, bpm, song_name)
    gain_db = np.zeros(len(dry_mix), dtype=np.float32)
    report = []
    for sec, (start, end) in slices.items():
        ratio = _rms_db(returns[start:end]) - _rms_db(dry_mix[start:end])
        kind = _section_kind(sec)
        max_ratio = SECTION_RETURN_TARGETS_DB[kind]
        requested_trim = min(0.0, max_ratio - ratio)
        trim = float(np.clip(requested_trim, max_cut_db, 0.0))
        gain_db[start:end] = trim
        report.append({
            "section": sec,
            "kind": kind,
            "return_to_dry_db_before": ratio,
            "max_return_to_dry_db": max_ratio,
            "requested_return_trim_db": requested_trim,
            "max_return_cut_db": max_cut_db,
            "return_trim_db": trim,
            "return_to_dry_db_after": ratio + trim,
            "hit_return_trim_cap": bool(requested_trim < max_cut_db),
        })
    curve = _smooth_gain_curve(gain_db, sr, fade_ms=90.0)
    calibrated = (returns * curve[:, None]).astype(np.float32)
    for row in report:
        start, end = slices[row["section"]]
        row["return_to_dry_db_after_smoothed"] = _rms_db(calibrated[start:end]) - _rms_db(dry_mix[start:end])
    return calibrated, report


def _write_section_aware_markdown(report: dict, path: str) -> None:
    lines = ["# Section-Aware Pink Staging Report", "", "## Outputs", ""]
    for key, value in report.get("outputs", {}).items():
        if isinstance(value, list):
            continue
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Bus Pink Section Deltas", ""])
    for bus, rows in report.get("bus_section_deltas", {}).items():
        for row in rows:
            lines.append(
                f"- {bus} {row['chorus']} vs {row['verse']}: "
                f"pink {row['pink_delta_before_db']:+.2f} -> {row['pink_delta_after_db']:+.2f} dB, "
                f"RMS {row['rms_delta_before_db']:+.2f} -> {row['rms_delta_after_db']:+.2f} dB"
            )
    lines.extend(["", "## Return Calibration", ""])
    for row in report.get("return_calibration", []):
        cap_note = " cap-hit" if row.get("hit_return_trim_cap") else ""
        lines.append(
            f"- {row['section']}: return/dry {row['return_to_dry_db_before']:+.2f} -> "
            f"{row['return_to_dry_db_after_smoothed']:+.2f} dB, "
            f"target <= {row['max_return_to_dry_db']:+.2f} dB, "
            f"trim {row['return_trim_db']:+.2f} dB{cap_note}"
        )
    lines.extend(["", "## Section Peak Control", ""])
    for row in report.get("section_peak_control", []):
        cap_note = " cap-hit" if row.get("hit_trim_cap") else ""
        lines.append(
            f"- {row['section']}: peak {row['peak_before_db']:+.2f} -> "
            f"{row['peak_after_db']:+.2f} dB, RMS {row['rms_before_db']:+.2f} -> "
            f"{row['rms_after_db']:+.2f} dB, trim {row['section_peak_trim_db']:+.2f} dB{cap_note}"
        )
    unmask_path = os.path.join(os.path.dirname(path), "unmask_report.json")
    if os.path.exists(unmask_path):
        try:
            with open(unmask_path) as fh:
                unmask_report = json.load(fh).get("events", [])
        except Exception:
            unmask_report = []
        if unmask_report:
            lines.extend(["", "## Bus Unmasking", ""])
            for row in unmask_report:
                lines.append(
                    f"- {row['target_bus']} vs {row['masker_bus']} "
                    f"{row['freq_range_hz'][0]:.0f}-{row['freq_range_hz'][1]:.0f} Hz: "
                    f"requested {row['requested_depth_db']:.2f} dB, actual {row['actual_cut_db']:.2f} dB, "
                    f"target {row['target_before_db']:+.2f} -> {row['target_after_db']:+.2f} dB"
                )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _is_retained_final_master(filename: str) -> bool:
    if not filename.endswith("_master.wav"):
        return False
    return any(token in filename for token in (
        "_drums-bass_streaming_master.wav",
        "_pads-drums-bass_streaming_master.wav",
    ))


def _ensure_stereo(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return np.stack([y, y], axis=1)
    if y.shape[1] == 1:
        return np.repeat(y, 2, axis=1)
    return y[:, :2]


# Auto-pan eligible drum/percussion stems (exclude kick — keep centered)
_AUTOPAN_TARGETS = [
    'snare', 'snarealt', 'clap',
    'closedhat', 'openhat', 'pedalhat', 'hihat', 'hat',
    'crash', 'ride', 'cymbal',
    'tambourine', 'maracas', 'perc',
    'drum_aux', 'bongo', 'conga',
    'floortom', 'lowtom', 'midtom', 'tom',
    'sidestick', 'cowbell',
]


def _is_autopan_eligible(name: str) -> bool:
    """Check if a stem should receive auto-pan (drum/percussion, not kick)."""
    n = name.lower()
    if any(x in n for x in ['kick', 'kicklow', 'kickalt']):
        return False
    return any(x in n for x in _AUTOPAN_TARGETS)
