
import contextlib
import concurrent.futures
import gc
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import soundfile as sf

import production_engine as pe
import config
from gain_staging import gain_stage_to_target


_COPY_DIR = None
_WORKERS = 1
_QUIET = False
_RUN_METADATA_PATH = None
_GAIN_MODE = "pink"


def install(engine_module, copy_dir, workers=1, quiet_optimizer=False, run_metadata_path=None, gain_mode="pink"):
    global _COPY_DIR, _WORKERS, _QUIET, _RUN_METADATA_PATH, _GAIN_MODE
    _COPY_DIR = copy_dir
    _WORKERS = max(1, int(workers or 1))
    _QUIET = bool(quiet_optimizer)
    _RUN_METADATA_PATH = run_metadata_path
    _GAIN_MODE = gain_mode
    engine_module.ProductionEngine.process_full_mix = process_full_mix_parallel


def _worker_init(copy_dir):
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_var, "1")
    if copy_dir and copy_dir not in sys.path:
        sys.path.insert(0, copy_dir)


def process_full_mix_parallel(self, stems: dict, song_name: str, bpm: float = 90.0) -> str:
    print(f"\n--- PRODUCING: {song_name} (BPM: {bpm}) ---")
    stem_paths = [p for p in stems.values() if p and os.path.exists(p)]
    if not stem_paths:
        print("  No stems found - skipping production.")
        return None

    print(f"[Step 1] Per-Stem Optimization ({_WORKERS} parallel workers)...")
    processed_dir = os.path.join(self.output_dir, "processed")
    reverb_dir = os.path.join(self.output_dir, "reverb_returns")
    delay_dir = os.path.join(self.output_dir, "delay_returns")
    worker_log_dir = os.path.join(self.output_dir, "worker_logs")
    for d in (processed_dir, reverb_dir, delay_dir, worker_log_dir):
        os.makedirs(d, exist_ok=True)

    sr = self._get_sr_from_paths(stem_paths)
    self._save_state(0, song_name, bpm, stem_paths=stem_paths)

    print("  Analyzing stem context...")
    ref_profile = None
    if self.reference_analysis:
        ref_profile = self.reference_analysis.get("spectral_profile")
    context = pe.analyze_stem_context(stem_paths, sr)
    context_suggestions = context.get("suggestions", {})
    stem_diagnosis = self.listener.analyze_stems(stem_paths, sr, bpm)
    stem_actions = self._actions_by_target(stem_diagnosis.actions)

    tasks = []
    for index, path in enumerate(stem_paths):
        name = os.path.basename(path)
        tasks.append({
            "index": index,
            "path": path,
            "name": name,
            "output_dir": self.output_dir,
            "processed_dir": processed_dir,
            "reverb_dir": reverb_dir,
            "delay_dir": delay_dir,
            "worker_log_dir": worker_log_dir,
            "ref_profile": ref_profile,
            "context_suggestions": context_suggestions.get(name, []),
            "stem_actions": stem_actions.get(name, []),
            "bpm": bpm,
            "quiet": _QUIET,
            "gain_mode": _GAIN_MODE,
        })

    results = []
    if _WORKERS <= 1:
        for task in tasks:
            results.append(_process_one_stem(task))
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=_WORKERS,
            initializer=_worker_init,
            initargs=(_COPY_DIR,),
        ) as pool:
            futures = {pool.submit(_process_one_stem, task): task for task in tasks}
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    failure = {
                        "stem": task["name"],
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                    _append_jsonl(_RUN_METADATA_PATH, {"event": "worker_failed", **failure})
                    raise RuntimeError(f"Parallel stem worker failed for {task['name']}: {exc}") from exc

    results.sort(key=lambda r: r["index"])
    processed_paths = []
    reverb_return_paths = []
    delay_return_paths = []
    for result in results:
        if result.get("silent"):
            print(f"  {result['name']:48s}  [SILENT] skipping")
            continue
        processed_paths.append(result["proc_path"])
        if result.get("rev_path"):
            reverb_return_paths.append(result["rev_path"])
        if result.get("dly_path"):
            delay_return_paths.append(result["dly_path"])
        self.optimization_logger.log_stem_optimization(
            result["name"], result["role"], result.get("best_params", {}),
            result.get("history", []), result.get("sanity_result")
        )
        assessment = result["assessment"]
        layer_str = f" [{result['layer']}]" if result.get("layer") else ""
        status = "OK" if assessment["all_pass"] else "WARN"
        clip_flag = " CLIP!" if assessment.get("is_clipped") else ""
        pan_str = " [autopan]" if result.get("autopan") else ""
        print(
            f"  {result['name']:48s}  gain:{result['gain_db']:+.1f}dB | "
            f"LUFS={assessment['lufs']:+.1f} TP={assessment['true_peak_db']:+.1f} "
            f"crest={assessment['crest_db']:.1f} LRA={assessment['lra']:.1f} | "
            f"stereo={assessment['stereo_corr']:.2f}{pan_str} | "
            f"[{status}]{clip_flag}{layer_str} | {result['duration_sec']:.1f}s"
        )
        _append_jsonl(_RUN_METADATA_PATH, {
            "event": "stem_processed",
            "stem": result["name"],
            "role": result["role"],
            "layer": result.get("layer"),
            "duration_sec": result["duration_sec"],
            "proc_path": result["proc_path"],
            "rev_path": result.get("rev_path"),
            "dly_path": result.get("dly_path"),
            "worker_log": result.get("worker_log"),
        })

    self.optimization_logger.log_session_summary()
    summary = self.optimization_logger.get_session_summary()
    if summary:
        print(f"\n  Optimization Summary:")
        print(f"    Total stems: {summary.get('total_stems', 0)}")
        print(f"    Avg evaluations: {summary.get('avg_evaluations_per_stem', 0)}")
        print(f"    All pass: {summary.get('stems_all_pass', 0)}")
        print(f"    With warnings: {summary.get('stems_with_warnings', 0)}")

    self._save_state(1, song_name, bpm, processed_paths, reverb_return_paths,
                     delay_return_paths, stem_paths=stem_paths)

    print("[Step 2] Bus Processing & Unmasking...")
    bus_paths = self._process_buses(processed_paths, sr, bpm)
    self._save_state(2, song_name, bpm, processed_paths, reverb_return_paths,
                     delay_return_paths, bus_paths, stem_paths=stem_paths)

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

    print("[Step 4] Iterative Mastering...")
    master_path = self._master(mix_path, song_name, bpm=bpm)
    self._save_state(4, song_name, bpm, processed_paths, reverb_return_paths,
                     delay_return_paths, bus_paths, automated_bus_paths, mix_path,
                     master_path, stem_paths)

    if mix_path and os.path.exists(mix_path):
        shutil.copy2(mix_path, os.path.join(self.output_dir, "premaster_full_mix.wav"))

    print("[Step 5] Creating Mix Variants...")
    self._create_variants(processed_paths, reverb_return_paths,
                          delay_return_paths, bus_paths, song_name, sr, bpm=bpm)

    final_dir = os.path.join(self.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    final_copied = []
    if master_path and os.path.exists(master_path):
        dest = os.path.join(final_dir, os.path.basename(master_path))
        shutil.copy2(master_path, dest)
        final_copied.append(dest)
    for f in os.listdir(self.output_dir):
        if pe._is_retained_final_master(f) and f != os.path.basename(master_path or ""):
            src = os.path.join(self.output_dir, f)
            dest = os.path.join(final_dir, f)
            shutil.copy2(src, dest)
            final_copied.append(dest)
    if final_copied:
        print(f"\n  Final outputs ({len(final_copied)}):")
        for p in final_copied:
            print(f"    {p}")

    self.listener.write_reports(self.output_dir)
    print(f"\nProduction complete: {master_path}")
    return master_path


def _process_one_stem(task):
    start = time.time()
    name = task["name"]
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    worker_log = os.path.join(task["worker_log_dir"], f"{task['index']:02d}_{safe_name}.log")
    try:
        with open(worker_log, "w") as log_f, contextlib.redirect_stdout(log_f), contextlib.redirect_stderr(log_f):
            return _process_one_stem_inner(task, worker_log, start)
    except Exception:
        with open(worker_log, "a") as log_f:
            log_f.write("\nWORKER FAILURE\n")
            log_f.write(traceback.format_exc())
        raise


def _process_one_stem_inner(task, worker_log, start):
    engine = pe.ProductionEngine(output_dir=task["output_dir"])
    path = task["path"]
    name = task["name"]
    role = engine._detect_role(name)
    layer = engine._detect_layer(name)
    layer_preset = engine._get_layer_preset(name)

    y, sr = sf.read(path, always_2d=True)
    y = pe._ensure_stereo(np.asarray(y, dtype=np.float32))
    if np.max(np.abs(y)) < 1e-6:
        return {
            "index": task["index"],
            "name": name,
            "silent": True,
            "duration_sec": time.time() - start,
            "worker_log": worker_log,
        }

    if layer_preset:
        orig_target = config.STEM_LUFS_TARGETS.get(role)
        config.STEM_LUFS_TARGETS[role] = layer_preset["lufs_target"]
        y_staged, gain_db = gain_stage_to_target(y, sr, role)
        if orig_target is not None:
            config.STEM_LUFS_TARGETS[role] = orig_target
    else:
        y_staged, gain_db = gain_stage_to_target(y, sr, role)

    eq_bands = engine.dsp.detect_eq_bands(y_staged, sr, role, n_bands=4)
    print(f"Processing: {name}")
    print(f"Role: {role}, Layer: {layer or 'none'}")
    print(f"EQ bands: {[b['band_type'] for b in eq_bands]}")
    if task["context_suggestions"]:
        print(f"Context cuts: {task['context_suggestions']}")

    y_processed, debug = engine.iterative.iterative_stem(
        y_staged, sr, role, task["ref_profile"],
        layer=layer, context_suggestions=task["context_suggestions"],
        eq_bands=eq_bands, bpm=task["bpm"], use_optimizer=True,
        stem_name=name,
    )

    if layer_preset and "eq" in layer_preset:
        y_before_eq = y_processed.copy()
        y_processed = engine.dsp.layer_eq(y_processed, sr, layer_preset["eq"])
        y_processed, _ = engine.dsp.gain_match(y_before_eq, y_processed, max_correction_db=2.0)

    listen_actions = task["stem_actions"]
    if listen_actions:
        y_processed = engine._apply_listening_actions(
            y_processed, sr, listen_actions, source_name=name, bpm=task["bpm"]
        )

    if layer_preset and layer_preset["trim_db"] != 0.0:
        y_processed = y_processed * (10.0 ** (layer_preset["trim_db"] / 20.0))

    from arrangement_density import apply_arrangement_envelope
    y_processed, arrangement_report = apply_arrangement_envelope(
        y_processed, sr, task["bpm"], name, role, layer
    )

    assessment = engine.assessor.assess_stem(y_processed, sr, role)
    sanity_result = engine.sanity_checker.check_stem(name, assessment)

    if task.get("gain_mode") == "baseline":
        from gain_flow_logging import current_loudness_match_with_log
        y_processed, lm_correction = current_loudness_match_with_log(
            engine.assessor, y, y_processed, sr, name, role, layer
        )
    else:
        from pink_noise_stage import pink_noise_stage_processed_stem
        y_processed, lm_correction = pink_noise_stage_processed_stem(
            y_processed, sr, role=role, layer=layer, stem_name=name
        )

    autopan = pe._is_autopan_eligible(name)
    if autopan:
        pan_cfg = config.AUTO_PAN
        seed = pan_cfg["seed_base"] + hash(name) % 1000
        y_processed = engine.dsp.auto_pan(
            y_processed, sr, task["bpm"],
            pan_range=pan_cfg["pan_range"],
            rate_triplets=pan_cfg["rate_triplets"],
            irregularity=pan_cfg["irregularity"],
            seed=seed,
        )

    proc_path = os.path.join(task["processed_dir"], f"proc_{name}")
    sf.write(proc_path, y_processed, sr, subtype="FLOAT")

    send_config = engine._get_send_config(name)
    if layer_preset:
        send_config["reverb_send"] = layer_preset["reverb_send"]
        send_config["delay_send"] = layer_preset["delay_send"]
        send_config["reverb_category"] = layer_preset["reverb_category"]

    rev_path = None
    if send_config["reverb_send"] > 0:
        y_send = y_processed * send_config["reverb_send"]
        y_wet = engine.dsp.reverb(y_send, sr, send_config["reverb_category"]) * 0.67
        rev_path = os.path.join(task["reverb_dir"], f"reverb_return_{name}")
        sf.write(rev_path, y_wet, sr, subtype="FLOAT")

    dly_path = None
    if send_config["delay_send"] > 0:
        y_dly_send = y_processed * send_config["delay_send"]
        delay_type = send_config.get("delay_type", "melodic")
        y_dly = engine.dsp.delay(y_dly_send, sr, task["bpm"], delay_type) * 0.67
        dly_path = os.path.join(task["delay_dir"], f"delay_return_{name}")
        sf.write(dly_path, y_dly, sr, subtype="FLOAT")

    del y, y_staged, y_processed
    gc.collect()

    return {
        "index": task["index"],
        "name": name,
        "role": role,
        "layer": layer,
        "gain_db": float(gain_db),
        "lm_correction": float(lm_correction),
        "assessment": assessment,
        "sanity_result": sanity_result,
        "best_params": debug.get("best_params", {}),
        "history": debug.get("history", []),
        "proc_path": proc_path,
        "rev_path": rev_path,
        "dly_path": dly_path,
        "autopan": autopan,
        "duration_sec": time.time() - start,
        "worker_log": worker_log,
        "arrangement": arrangement_report,
        "silent": False,
    }


def _append_jsonl(path, payload):
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(payload, default=str) + "\n")
