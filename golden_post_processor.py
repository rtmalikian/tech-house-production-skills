"""
Promoted golden post-audio processor for the Python Revamp pipeline.

This module contains the production version of the dry stereo-wall processing
that was proven in the isolated full-song test environment on 2026-05-08.
It intentionally avoids importing test scripts so orchestrator.py can use it
as a normal production pathway.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf


DEFAULT_PARAM_PATH = Path(__file__).resolve().parent / "golden_post_params.json"


def load_params(params_path: Optional[str] = None,
                pan_seed_override: Optional[int] = None,
                song_name: str = "") -> Dict:
    path = Path(params_path).expanduser().resolve() if params_path else DEFAULT_PARAM_PATH
    with open(path) as f:
        params = json.load(f)
    params["_params_path"] = str(path)
    params["_base_pan_seed"] = int(params.get("pan_seed_base", 60233))
    if pan_seed_override is not None:
        params["pan_seed_base"] = int(pan_seed_override)
        params["pan_seed_mode"] = "override"
    elif params.get("pan_seed_mode") == "song":
        digest = hashlib.sha1(str(song_name).encode("utf-8")).hexdigest()
        offset = int(digest[:8], 16) % 20000
        params["pan_seed_base"] = int(params["_base_pan_seed"]) + offset
    elif params.get("pan_seed_mode") == "run":
        digest = hashlib.sha1(f"{song_name}:{time.time_ns()}".encode("utf-8")).hexdigest()
        offset = int(digest[:8], 16) % 20000
        params["pan_seed_base"] = int(params["_base_pan_seed"]) + offset
    return params


class GoldenPostProcessor:
    """Apply the promoted dry stereo-wall and objective post-balance path."""

    def __init__(self, engine, params: Dict):
        self.engine = engine
        self.params = dict(params)

    def render_mix(self, processed_paths: List[str], reverb_paths: List[str],
                   delay_paths: List[str], bus_paths: List[str],
                   automated_bus_paths: List[str], song_name: str,
                   bpm: float, sr: int) -> Dict:
        max_passes = max(1, int(self.params.get("golden_post_max_passes", 1)))
        best_report = None
        pass_reports = []
        working_params = dict(self.params)
        for pass_index in range(1, max_passes + 1):
            self.params = dict(working_params)
            report = self._render_pass(
                processed_paths, reverb_paths, delay_paths, bus_paths,
                automated_bus_paths, song_name, bpm, sr, pass_index
            )
            pass_reports.append(report)
            if best_report is None or score_golden_report(report) > score_golden_report(best_report):
                best_report = report
            if report.get("objective_gate_results", {}).get("all_pass"):
                break
            working_params = next_pass_params(working_params, report)

        if best_report is None:
            return {"mix_path": None, "enabled": False, "reason": "no_report"}
        best_report["pass_reports"] = summarize_pass_reports(pass_reports)
        best_report["selected_pass"] = best_report.get("pass_index")
        self.params = dict(best_report.get("params", self.params))
        self._write_reports(Path(self.engine.output_dir) / "golden_post", best_report)
        return best_report

    def _render_pass(self, processed_paths: List[str], reverb_paths: List[str],
                     delay_paths: List[str], bus_paths: List[str],
                     automated_bus_paths: List[str], song_name: str,
                     bpm: float, sr: int, pass_index: int) -> Dict:
        post_dir = Path(self.engine.output_dir) / "golden_post" / f"pass_{pass_index:02d}"
        sources_dir = post_dir / "adjusted_sources"
        returns_dir = post_dir / "section_adjusted_returns"
        target_returns_dir = post_dir / "target_adjusted_returns"
        dry_drums_dir = post_dir / "dry_stereo_drums"
        for path in (post_dir, sources_dir, returns_dir, target_returns_dir, dry_drums_dir):
            path.mkdir(parents=True, exist_ok=True)

        raw_buses = load_named_buses(automated_bus_paths or bus_paths)
        if not raw_buses:
            return {"mix_path": None, "enabled": False, "reason": "no_bus_paths"}

        raw_metrics = {name: simple_audio_stats(y, file_sr)
                       for name, (y, file_sr) in raw_buses.items()}
        kick_paths = [p for p in processed_paths if "kick" in Path(p).name.lower()]
        kick_bus, kick_sr = load_sum_paths(kick_paths)
        if kick_bus is None:
            first_y, kick_sr = next(iter(raw_buses.values()))
            kick_bus = np.zeros_like(first_y, dtype=np.float32)

        trial_metrics = self._trial_bus_metrics(raw_buses)
        bass_dynamic_gain_db = self._dynamic_role_gain(
            trial_metrics, "bass", "melody",
            "bass_target_vs_melody_db", "bass_trim_min_db", "bass_trim_max_db"
        )
        pads_dynamic_gain_db = self._dynamic_role_gain(
            trial_metrics, "pads", "melody",
            "pads_target_vs_melody_db", "pads_trim_min_db", "pads_trim_max_db"
        )

        bass_low = raw_metrics.get("bass", {}).get("low_40_120_rms_db", -120.0)
        kick_stats = simple_audio_stats(kick_bus, kick_sr)
        kick_low = kick_stats.get("low_40_120_rms_db", -120.0)
        duck_depth_db = float(np.clip(
            (bass_low - kick_low) - float(self.params["kick_duck_target_margin_db"]),
            float(self.params["kick_duck_min_db"]),
            float(self.params["kick_duck_max_db"]),
        ))

        pan_report = build_panned_drum_bus(
            processed_paths, dry_drums_dir, self.params, engine=self.engine
        )

        adjusted_main = []
        adjusted_metrics = {}
        applied_moves = []
        for name, (y, file_sr) in raw_buses.items():
            if name == "drums" and pan_report.get("drums_path"):
                yy, file_sr = sf.read(pan_report["drums_path"], always_2d=True)
                yy = np.asarray(yy, dtype=np.float32)
            else:
                yy = self._apply_role_tone(y, file_sr, name)

            role_gain = float(self.params.get(f"{name}_gain_db", 0.0))
            if name == "bass":
                role_gain += bass_dynamic_gain_db
                yy = apply_kick_ducking(yy, kick_bus, duck_depth_db)
            elif name == "pads":
                role_gain += pads_dynamic_gain_db

            yy = apply_gain_db(yy, role_gain)
            if name == "drums":
                yy = add_kick_parallel(
                    self.engine, yy, file_sr, kick_bus, kick_sr, self.params
                )
                yy, post_pan_gain = post_pan_gain_stage_drums(
                    yy, file_sr, trial_metrics, self.params
                )
                role_gain += post_pan_gain
                pan_report["post_pan_drums_gain_db"] = post_pan_gain

            out = sources_dir / f"{name}.wav"
            sf.write(out, yy.astype(np.float32), file_sr, subtype="FLOAT")
            adjusted_main.append(str(out))
            adjusted_metrics[name] = simple_audio_stats(yy, file_sr)
            applied_moves.append({"target": name, "gain_db": role_gain})

        adjusted_reverb = adjust_section_returns(
            reverb_paths, returns_dir, "reverb",
            float(self.params["reverb_body_trim_db"]),
            float(self.params["reverb_outro_trim_db"]),
            float(self.params["return_duck_body_db"]),
            float(self.params["return_duck_outro_db"]),
            adjusted_main,
            float(self.params["outro_fraction"]),
        )
        adjusted_delay = adjust_section_returns(
            delay_paths, returns_dir, "delay",
            float(self.params["delay_body_trim_db"]),
            float(self.params["delay_outro_trim_db"]),
            float(self.params["return_duck_body_db"]) * 0.5,
            float(self.params["return_duck_outro_db"]) * 0.5,
            adjusted_main,
            float(self.params["outro_fraction"]),
        )
        return_normalization = normalize_returns_to_target(
            adjusted_reverb,
            adjusted_delay,
            adjusted_main,
            target_returns_dir,
            float(self.params["return_to_dry_target_db"]),
        )
        adjusted_reverb = return_normalization["reverb_paths"]
        adjusted_delay = return_normalization["delay_paths"]
        snare_duck_report = {"enabled": False}
        if self.params.get("snare_duck_enabled", True):
            snare_duck_report = apply_snare_ducking_to_paths(
                adjusted_main,
                adjusted_reverb,
                adjusted_delay,
                pan_report,
                post_dir / "snare_ducked",
                self.params,
            )
            adjusted_main = snare_duck_report.get("main_paths", adjusted_main)
            adjusted_reverb = snare_duck_report.get("reverb_paths", adjusted_reverb)
            adjusted_delay = snare_duck_report.get("delay_paths", adjusted_delay)

        all_mix_paths = adjusted_reverb + adjusted_delay + adjusted_main
        mix_path = self.engine._sum_to_mix(all_mix_paths, song_name, bpm=bpm)
        if mix_path and Path(mix_path).exists():
            pass_mix_path = post_dir / "candidate_mix.wav"
            shutil.copy2(mix_path, pass_mix_path)
            mix_path = str(pass_mix_path)
        final_analysis = analyze_post_recording_paths(
            adjusted_main, adjusted_reverb, adjusted_delay
        )
        kick_analysis = analyze_kick_bass(
            kick_bus, kick_sr, raw_buses, adjusted_main, duck_depth_db, len(kick_paths)
        )
        report = {
            "enabled": True,
            "params": self.params,
            "pass_index": pass_index,
            "raw_bus_metrics": raw_metrics,
            "adjusted_bus_metrics": adjusted_metrics,
            "bass_dynamic_gain_db": bass_dynamic_gain_db,
            "pads_dynamic_gain_db": pads_dynamic_gain_db,
            "kick_paths": kick_paths,
            "kick_duck_depth_db": duck_depth_db,
            "kick_analysis": kick_analysis,
            "dry_stereo_wall": pan_report,
            "applied_moves": applied_moves,
            "return_normalization": return_normalization,
            "snare_ducking": snare_duck_report,
            "after": final_analysis,
            "mix_path": mix_path,
            "post_dir": str(post_dir),
        }
        report["stereo_wall_analysis"] = analyze_stereo_wall(
            mix_path, final_analysis, pan_report
        )
        report["snare_hat_audibility"] = analyze_snare_hat_audibility(
            final_analysis, pan_report
        )
        report["objective_gate_results"] = evaluate_gates(report)
        self._write_reports(post_dir, report)
        return report

    def _trial_bus_metrics(self, raw_buses: Dict[str, Tuple[np.ndarray, int]]) -> Dict:
        trial_metrics = {}
        for name, (y, sr) in raw_buses.items():
            yy = self._apply_role_tone(y, sr, name)
            yy = apply_gain_db(yy, float(self.params.get(f"{name}_gain_db", 0.0)))
            trial_metrics[name] = simple_audio_stats(yy, sr)
        return trial_metrics

    def _dynamic_role_gain(self, metrics: Dict, target: str, anchor: str,
                           target_key: str, min_key: str, max_key: str) -> float:
        current = bus_ratio_db(metrics, target, anchor)
        desired = float(self.params[target_key])
        return float(np.clip(
            desired - current,
            float(self.params[min_key]),
            float(self.params[max_key]),
        ))

    def _apply_role_tone(self, y: np.ndarray, sr: int, name: str) -> np.ndarray:
        yy = ensure_stereo(np.asarray(y, dtype=np.float32))
        cut = float(self.params.get("low_mid_cut_db", 0.0))
        if name in {"melody", "pads"} and abs(cut) > 0.01:
            yy = self.engine.dsp.apply_bell_eq(yy, sr, 260.0, cut, q=0.9)
        return yy.astype(np.float32)

    def _write_reports(self, post_dir: Path, report: Dict) -> None:
        json_path = post_dir / "golden_post_analysis.json"
        json_path.write_text(json.dumps(report, indent=2, default=str))
        lines = [
            "# Golden Post Analysis",
            "",
            f"- Source preset: `{report['params'].get('_params_path')}`",
            f"- Selected pass: `{report.get('selected_pass', report.get('pass_index'))}`",
            f"- Pan seed: `{report['params'].get('pan_seed_base')}`",
            f"- Return / dry: `{report['after']['return_to_dry_rms_db']:.2f} dB`",
            f"- Drums / melody: `{report['after']['drum_to_melody_rms_db']:.2f} dB`",
            f"- Bass / melody: `{report['after']['bass_to_melody_rms_db']:.2f} dB`",
            f"- Pads / melody: `{report['after']['pads_to_melody_rms_db']:.2f} dB`",
            f"- Kick ducking max: `{report['kick_duck_depth_db']:.2f} dB`",
            f"- Snare ducking max: `{report.get('snare_ducking', {}).get('max_reduction_db', 0.0):.2f} dB`",
            f"- Bass minus kick low band: `{report['kick_analysis']['bass_minus_kick_low_db']:.2f} dB`",
            f"- Snare mean pan: `{report['stereo_wall_analysis'].get('snare_mean_pan', 0.0):+.2f}`",
            f"- Hat mean pan: `{report['stereo_wall_analysis'].get('hats_mean_pan', 0.0):+.2f}`",
            f"- Drum side/mid ratio: `{report['stereo_wall_analysis'].get('drums_side_mid_ratio', 0.0):.3f}`",
            f"- Snare RMS vs drums: `{report.get('snare_hat_audibility', {}).get('snare_rms_vs_drums_db', -120.0):+.2f} dB`",
            f"- Hats RMS vs drums: `{report.get('snare_hat_audibility', {}).get('hats_rms_vs_drums_db', -120.0):+.2f} dB`",
            f"- Snare presence vs melody: `{report.get('snare_hat_audibility', {}).get('snare_presence_vs_melody_db', -120.0):+.2f} dB`",
            f"- Objective gates all pass: `{report['objective_gate_results']['all_pass']}`",
            "",
            "## Iterative Passes",
        ]
        for item in report.get("pass_reports", []):
            lines.append(
                f"- Pass `{item['pass_index']}` score `{item['score']:.2f}` "
                f"all_pass `{item['all_pass']}` snare/hat gain "
                f"`{item.get('snare_hat_gain_db', 0.0):+.1f} dB`"
            )
        lines += [
            "",
            "## Policy",
            "- Objective gates are logged as monitoring, not hard render blockers.",
            "- Reverb/delay body is tightened while the outro washout is retained.",
            "- Snare/clap remains left-of-center; hats remain right-of-center.",
            "- Snare/clap can briefly duck melody, pads, FX, reverb, and delay with a fast 1ms/10ms envelope.",
        ]
        (post_dir / "golden_post_analysis.md").write_text("\n".join(lines) + "\n")


def build_panned_drum_bus(processed_paths: List[str], out_dir: Path, params: Dict, engine=None) -> Dict:
    rng = np.random.default_rng(int(params["pan_seed_base"]))
    result = {"pan_seed": int(params["pan_seed_base"]), "stems": [], "groups": {}, "drums_path": None}
    summed = None
    sr0 = None
    for src in processed_paths:
        src_path = Path(src)
        role = drum_component_role(src_path.name)
        if role is None:
            continue
        y, sr = sf.read(src_path, always_2d=True)
        y = ensure_stereo(np.asarray(y, dtype=np.float32))
        if sr0 is None:
            sr0 = sr
        if sr != sr0:
            continue
        pan, depth = choose_component_pan(role, rng, params)
        gain_db = component_gain_db(role, params)
        if role == "snare_clap":
            y = enhance_snare_clap(y, sr, params, engine=engine)
        y = apply_gain_db(y, gain_db)
        y = apply_static_pan(y, pan)
        if depth > 0.0:
            y = apply_centered_autopan(
                y, sr, base_pan=pan, depth=depth,
                cycles=float(params["autopan_cycles"])
            )
        out = out_dir / f"panned_{src_path.name}"
        sf.write(out, y.astype(np.float32), sr, subtype="FLOAT")
        summed = y if summed is None else sum_fit(summed, y)
        result["stems"].append({
            "name": src_path.name,
            "role": role,
            "pan": pan,
            "autopan_depth": depth,
            "gain_db": gain_db,
            "transient_boost_db": float(params.get("snare_transient_boost_db", 0.0)) if role == "snare_clap" else 0.0,
            "path": str(out),
            "stats": simple_audio_stats(y, sr),
        })
    if summed is None:
        return result
    drums_path = out_dir / "dry_stereo_drums.wav"
    sf.write(drums_path, summed.astype(np.float32), sr0, subtype="FLOAT")
    result["drums_path"] = str(drums_path)
    result["drums_stats_before_post_pan_gain"] = simple_audio_stats(summed, sr0)
    for role in sorted({item["role"] for item in result["stems"]}):
        role_paths = [item["path"] for item in result["stems"] if item["role"] == role]
        role_y, role_sr = load_sum_paths(role_paths)
        if role_y is not None:
            role_items = [item for item in result["stems"] if item["role"] == role]
            result["groups"][role] = {
                "count": len(role_paths),
                "stats": simple_audio_stats(role_y, role_sr),
                "mean_pan": float(np.mean([item["pan"] for item in role_items])),
            }
    return result


def drum_component_role(name: str) -> Optional[str]:
    n = name.lower()
    if "kick" in n:
        return "kick"
    if any(x in n for x in ("snare", "clap", "sidestick")):
        return "snare_clap"
    if "hat" in n:
        return "hats"
    if any(x in n for x in ("drum_aux", "tambourine", "maracas", "perc", "shaker", "bongo", "conga")):
        return "aux_perc"
    if any(x in n for x in ("crash", "ride", "tom", "cowbell", "drum")):
        return "other_perc"
    return None


def choose_component_pan(role: str, rng: np.random.Generator, params: Dict) -> Tuple[float, float]:
    if role == "kick":
        return 0.0, 0.0
    if role == "snare_clap":
        lo, hi = params["snare_pan_range"]
        return float(rng.uniform(lo, hi)), float(params["snare_autopan_depth"])
    if role == "hats":
        lo, hi = params["hat_pan_range"]
        return float(rng.uniform(lo, hi)), float(params["hat_autopan_depth"])
    if role == "aux_perc":
        lo, hi = params["aux_pan_range"]
        pan = float(rng.uniform(lo, hi))
        if abs(pan) < 0.25:
            pan = float(np.sign(pan or 1.0) * rng.uniform(0.25, 0.55))
        return pan, float(params["aux_autopan_depth"])
    lo, hi = params["other_perc_pan_range"]
    return float(rng.uniform(lo, hi)), float(params["other_perc_autopan_depth"])


def component_gain_db(role: str, params: Dict) -> float:
    if role == "kick":
        return float(params.get("kick_gain_db", 0.0))
    if role in {"snare_clap", "hats"}:
        return float(params["snare_hat_gain_db"])
    if role == "aux_perc":
        return float(params["drum_aux_gain_db"])
    return 0.0


def enhance_snare_clap(y: np.ndarray, sr: int, params: Dict, engine=None) -> np.ndarray:
    y = ensure_stereo(np.asarray(y, dtype=np.float32))
    transient_boost_db = float(params.get("snare_transient_boost_db", 0.0))
    presence_boost_db = float(params.get("snare_presence_boost_db", 0.0))
    out = y
    if abs(presence_boost_db) > 0.01 and engine is not None:
        try:
            out = engine.dsp.apply_bell_eq(out, sr, 2500.0, presence_boost_db, q=1.0)
        except Exception:
            pass
    if transient_boost_db > 0.01:
        try:
            if engine is not None and hasattr(engine.dsp, "transient_sustain_split"):
                transient, sustain = engine.dsp.transient_sustain_split(out, sr)
                out = sustain + apply_gain_db(transient, transient_boost_db)
            else:
                transient, sustain = transient_sustain_split(out)
                out = sustain + apply_gain_db(transient, transient_boost_db)
        except Exception:
            transient, sustain = transient_sustain_split(out)
            out = sustain + apply_gain_db(transient, transient_boost_db)
    peak = float(np.max(np.abs(out)) + 1e-12)
    if peak > 0.98:
        out = out * (0.98 / peak)
    return out.astype(np.float32)


def transient_sustain_split(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = ensure_stereo(np.asarray(y, dtype=np.float32))
    mono = np.mean(np.abs(y), axis=1)
    fast = smooth_envelope(mono, attack=16, release=220)
    slow = smooth_envelope(mono, attack=256, release=3000)
    mask = np.clip((fast - slow) / (np.max(fast - slow) + 1e-12), 0.0, 1.0)
    mask = mask[:, None].astype(np.float32)
    transient = y * mask
    sustain = y * (1.0 - mask)
    return transient.astype(np.float32), sustain.astype(np.float32)


def apply_static_pan(y: np.ndarray, pan: float) -> np.ndarray:
    y = ensure_stereo(y)
    mono = np.mean(y[:, :2], axis=1)
    pan = float(np.clip(pan, -0.95, 0.95))
    angle = (pan + 1.0) * np.pi * 0.25
    return np.stack([np.cos(angle) * mono, np.sin(angle) * mono], axis=1).astype(np.float32)


def apply_centered_autopan(y: np.ndarray, sr: int, base_pan: float,
                           depth: float, cycles: float) -> np.ndarray:
    if len(y) == 0 or depth <= 0.0:
        return y
    mono = np.mean(y[:, :2], axis=1)
    t = np.linspace(0.0, 1.0, len(y), endpoint=False, dtype=np.float32)
    lfo = np.sin(2.0 * np.pi * float(cycles) * t)
    pan_curve = np.clip(float(base_pan) + float(depth) * lfo, -0.78, 0.78)
    angle = (pan_curve + 1.0) * np.pi * 0.25
    return np.stack([np.cos(angle) * mono, np.sin(angle) * mono], axis=1).astype(np.float32)


def post_pan_gain_stage_drums(y: np.ndarray, sr: int, trial_metrics: Dict, params: Dict) -> Tuple[np.ndarray, float]:
    drum_stats = simple_audio_stats(y, sr)
    melody_stats = trial_metrics.get("melody")
    if not melody_stats:
        return y, 0.0
    current = drum_stats["rms_db"] - melody_stats["rms_db"]
    gain_db = float(np.clip(
        float(params["post_pan_drums_vs_melody_target_db"]) - current,
        float(params["post_pan_drums_trim_min_db"]),
        float(params["post_pan_drums_trim_max_db"]),
    ))
    candidate = apply_gain_db(y, gain_db)
    peak = float(np.max(np.abs(candidate)) + 1e-12)
    if peak > 0.98:
        protect_db = 20.0 * np.log10(0.98 / peak)
        candidate = apply_gain_db(candidate, protect_db)
        gain_db += float(protect_db)
    return candidate.astype(np.float32), gain_db


def add_kick_parallel(engine, drums: np.ndarray, sr: int, kick_bus: np.ndarray,
                      kick_sr: int, params: Dict) -> np.ndarray:
    if kick_bus is None or kick_sr != sr:
        return drums
    kick = fit_length(kick_bus, len(drums), drums.shape[1])
    try:
        kick = engine.dsp.saturate(kick, role="default", amount_override=1.035)
    except Exception:
        pass
    kick = apply_gain_db(kick, float(params["kick_parallel_gain_db"]))
    mixed = drums + (kick * float(params["kick_parallel_blend"]))
    peak = float(np.max(np.abs(mixed)) + 1e-12)
    if peak > 0.98:
        mixed = mixed * (0.98 / peak)
    return mixed.astype(np.float32)


def apply_kick_ducking(bass: np.ndarray, kick_bus: np.ndarray, depth_db: float) -> np.ndarray:
    if kick_bus is None or depth_db <= 0.0:
        return bass
    kick = fit_length(kick_bus, len(bass), bass.shape[1])
    mono = np.mean(np.abs(kick), axis=1)
    if not np.any(mono > 0.0):
        return bass
    env = smooth_envelope(mono, attack=96, release=4200)
    threshold = float(np.percentile(env, 88))
    if threshold <= 1e-9:
        return bass
    control = np.clip((env - threshold) / max(np.max(env) - threshold, 1e-9), 0.0, 1.0)
    gain = np.power(10.0, (-float(depth_db) * control) / 20.0).astype(np.float32)
    return (bass * gain[:, None]).astype(np.float32)


def adjust_section_returns(paths: List[str], out_dir: Path, label: str,
                           body_trim_db: float, outro_trim_db: float,
                           body_duck_db: float, outro_duck_db: float,
                           dry_paths: List[str], outro_fraction: float) -> List[str]:
    out_paths = []
    dry_env = combined_envelope(dry_paths)
    for src in paths or []:
        src_path = Path(src)
        y, sr = sf.read(src_path, always_2d=True)
        y = ensure_stereo(np.asarray(y, dtype=np.float32))
        section_gain = section_gain_curve(len(y), body_trim_db, outro_trim_db, outro_fraction)
        duck_gain = return_duck_curve(len(y), dry_env, body_duck_db, outro_duck_db, outro_fraction)
        out_y = y * section_gain[:, None] * duck_gain[:, None]
        out = out_dir / f"{label}_{src_path.name}"
        sf.write(out, out_y.astype(np.float32), sr, subtype="FLOAT")
        out_paths.append(str(out))
    return out_paths


def normalize_returns_to_target(reverb_paths: List[str], delay_paths: List[str],
                                dry_paths: List[str], out_dir: Path,
                                target_db: float) -> Dict:
    before_db = return_to_dry_ratio_db((reverb_paths or []) + (delay_paths or []), dry_paths)
    extra_trim_db = float(np.clip(target_db - before_db, -10.0, 0.0)) if before_db > target_db else 0.0
    if abs(extra_trim_db) < 0.001:
        return {
            "before_return_to_dry_db": before_db,
            "target_return_to_dry_db": target_db,
            "extra_trim_db": 0.0,
            "after_return_to_dry_db": before_db,
            "reverb_paths": reverb_paths or [],
            "delay_paths": delay_paths or [],
        }
    result = {"reverb": [], "delay": []}
    for label, paths in (("reverb", reverb_paths or []), ("delay", delay_paths or [])):
        for src in paths:
            src_path = Path(src)
            y, sr = sf.read(src_path, always_2d=True)
            y = apply_gain_db(np.asarray(y, dtype=np.float32), extra_trim_db)
            out = out_dir / src_path.name
            sf.write(out, y, sr, subtype="FLOAT")
            result[label].append(str(out))
    after_db = return_to_dry_ratio_db(result["reverb"] + result["delay"], dry_paths)
    return {
        "before_return_to_dry_db": before_db,
        "target_return_to_dry_db": target_db,
        "extra_trim_db": extra_trim_db,
        "after_return_to_dry_db": after_db,
        "reverb_paths": result["reverb"],
        "delay_paths": result["delay"],
    }


def apply_snare_ducking_to_paths(main_paths: List[str], reverb_paths: List[str],
                                 delay_paths: List[str], pan_report: Dict,
                                 out_dir: Path, params: Dict) -> Dict:
    snare_paths = [
        item["path"] for item in pan_report.get("stems", [])
        if item.get("role") == "snare_clap"
    ]
    snare_y, snare_sr = load_sum_paths(snare_paths)
    if snare_y is None:
        return {
            "enabled": False,
            "reason": "no_snare_control",
            "main_paths": main_paths,
            "reverb_paths": reverb_paths,
            "delay_paths": delay_paths,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    targets = set(params.get("snare_duck_targets", []))
    depth_db = float(params.get("snare_duck_depth_db", 0.0))
    attack_ms = float(params.get("snare_duck_attack_ms", 1.0))
    release_ms = float(params.get("snare_duck_release_ms", 10.0))
    ducked = {"main": [], "reverb": [], "delay": []}
    stats = []

    def maybe_duck(path: str, label: str, target_name: str) -> str:
        if target_name not in targets or depth_db <= 0.0:
            return path
        y, sr = sf.read(path, always_2d=True)
        y = ensure_stereo(np.asarray(y, dtype=np.float32))
        if sr != snare_sr:
            return path
        out_y, duck_stats = apply_snare_ducking(
            y, snare_y, depth_db=depth_db,
            attack_ms=attack_ms, release_ms=release_ms, sr=sr
        )
        out_path = out_dir / label / Path(path).name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, out_y.astype(np.float32), sr, subtype="FLOAT")
        stats.append({"target": target_name, "path": str(out_path), **duck_stats})
        return str(out_path)

    for path in main_paths or []:
        name = bus_name_from_path(Path(path))
        ducked["main"].append(maybe_duck(path, "main", name))
    for path in reverb_paths or []:
        ducked["reverb"].append(maybe_duck(path, "reverb", "reverb"))
    for path in delay_paths or []:
        ducked["delay"].append(maybe_duck(path, "delay", "delay"))

    max_reduction = max([s["max_reduction_db"] for s in stats], default=0.0)
    avg_reduction = float(np.mean([s["avg_reduction_db"] for s in stats])) if stats else 0.0
    return {
        "enabled": True,
        "depth_db": depth_db,
        "attack_ms": attack_ms,
        "release_ms": release_ms,
        "control_paths": snare_paths,
        "targets": sorted(targets),
        "stats": stats,
        "max_reduction_db": max_reduction,
        "avg_reduction_db": avg_reduction,
        "main_paths": ducked["main"],
        "reverb_paths": ducked["reverb"],
        "delay_paths": ducked["delay"],
    }


def apply_snare_ducking(y: np.ndarray, snare_y: np.ndarray, depth_db: float,
                        attack_ms: float, release_ms: float, sr: int) -> Tuple[np.ndarray, Dict]:
    control_src = fit_length(snare_y, len(y), y.shape[1])
    mono = np.mean(np.abs(control_src), axis=1)
    if not np.any(mono > 0.0):
        return y, {"max_reduction_db": 0.0, "avg_reduction_db": 0.0}
    attack = max(1, int(sr * attack_ms / 1000.0))
    release = max(1, int(sr * release_ms / 1000.0))
    env = smooth_envelope(mono, attack=attack, release=release)
    threshold = float(np.percentile(env, 86))
    ceiling = float(np.percentile(env, 99))
    if threshold <= 1e-9 or ceiling <= threshold:
        return y, {"max_reduction_db": 0.0, "avg_reduction_db": 0.0}
    control = np.clip((env - threshold) / max(ceiling - threshold, 1e-9), 0.0, 1.0)
    reduction_db = float(depth_db) * control
    gain = np.power(10.0, -reduction_db / 20.0).astype(np.float32)
    active = reduction_db > 0.01
    avg_reduction = float(np.mean(reduction_db[active])) if np.any(active) else 0.0
    return (y * gain[:, None]).astype(np.float32), {
        "max_reduction_db": float(np.max(reduction_db)),
        "avg_reduction_db": avg_reduction,
    }


def analyze_post_recording_paths(dry_paths: List[str], reverb_paths: List[str],
                                 delay_paths: List[str]) -> Dict:
    buses = {}
    for path in dry_paths:
        name = bus_name_from_path(Path(path))
        y, sr = sf.read(path, always_2d=True)
        buses[name] = simple_audio_stats(np.asarray(y, dtype=np.float32), sr)
    return {
        "bus_metrics": buses,
        "return_to_dry_rms_db": return_to_dry_ratio_db((reverb_paths or []) + (delay_paths or []), dry_paths),
        "drum_to_melody_rms_db": bus_ratio_db(buses, "drums", "melody"),
        "bass_to_melody_rms_db": bus_ratio_db(buses, "bass", "melody"),
        "pads_to_melody_rms_db": bus_ratio_db(buses, "pads", "melody"),
        "reverb_paths": reverb_paths or [],
        "delay_paths": delay_paths or [],
    }


def analyze_kick_bass(kick_bus: np.ndarray, kick_sr: int, raw_buses: Dict,
                      adjusted_main: List[str], duck_depth_db: float,
                      kick_path_count: int) -> Dict:
    adjusted = {}
    for path in adjusted_main:
        y, sr = sf.read(path, always_2d=True)
        adjusted[bus_name_from_path(Path(path))] = (np.asarray(y, dtype=np.float32), sr)
    bass_y, bass_sr = adjusted.get("bass", raw_buses.get("bass"))
    drums_y, drums_sr = adjusted.get("drums", raw_buses.get("drums"))
    kick_stats = simple_audio_stats(kick_bus, kick_sr)
    bass_stats = simple_audio_stats(bass_y, bass_sr)
    drums_stats = simple_audio_stats(drums_y, drums_sr)
    return {
        "kick_path_count": kick_path_count,
        "duck_depth_db": duck_depth_db,
        "kick_low_40_120_rms_db": kick_stats["low_40_120_rms_db"],
        "bass_low_40_120_rms_db": bass_stats["low_40_120_rms_db"],
        "drums_low_40_120_rms_db": drums_stats["low_40_120_rms_db"],
        "bass_minus_kick_low_db": bass_stats["low_40_120_rms_db"] - kick_stats["low_40_120_rms_db"],
        "bass_minus_drums_low_db": bass_stats["low_40_120_rms_db"] - drums_stats["low_40_120_rms_db"],
    }


def analyze_stereo_wall(render_path: Optional[str], final_analysis: Dict, pan_report: Dict) -> Dict:
    groups = pan_report.get("groups", {})
    master_stats = {}
    if render_path and Path(render_path).exists():
        y, sr = sf.read(render_path, always_2d=True)
        master_stats = simple_audio_stats(np.asarray(y, dtype=np.float32), sr)
    drum_stats = final_analysis.get("bus_metrics", {}).get("drums", {})
    return {
        "master_side_mid_ratio": master_stats.get("side_mid_ratio"),
        "master_stereo_corr": master_stats.get("stereo_corr"),
        "drums_side_mid_ratio": drum_stats.get("side_mid_ratio", 0.0),
        "snare_mean_pan": groups.get("snare_clap", {}).get("mean_pan", 0.0),
        "hats_mean_pan": groups.get("hats", {}).get("mean_pan", 0.0),
        "aux_mean_pan": groups.get("aux_perc", {}).get("mean_pan", 0.0),
        "kick_mean_pan": groups.get("kick", {}).get("mean_pan", 0.0),
        "snare_side_mid_ratio": groups.get("snare_clap", {}).get("stats", {}).get("side_mid_ratio", 0.0),
        "hats_side_mid_ratio": groups.get("hats", {}).get("stats", {}).get("side_mid_ratio", 0.0),
        "aux_side_mid_ratio": groups.get("aux_perc", {}).get("stats", {}).get("side_mid_ratio", 0.0),
    }


def analyze_snare_hat_audibility(final_analysis: Dict, pan_report: Dict) -> Dict:
    groups = pan_report.get("groups", {})
    bus_metrics = final_analysis.get("bus_metrics", {})
    drums = bus_metrics.get("drums", {})
    melody = bus_metrics.get("melody", {})
    snare = groups.get("snare_clap", {}).get("stats", {})
    hats = groups.get("hats", {}).get("stats", {})
    return {
        "snare_rms_vs_drums_db": snare.get("rms_db", -120.0) - drums.get("rms_db", -120.0),
        "hats_rms_vs_drums_db": hats.get("rms_db", -120.0) - drums.get("rms_db", -120.0),
        "snare_presence_vs_melody_db": (
            snare.get("presence_1200_5000_rms_db", -120.0)
            - melody.get("presence_1200_5000_rms_db", -120.0)
        ),
        "hats_air_vs_melody_db": (
            hats.get("air_6500_14000_rms_db", -120.0)
            - melody.get("air_6500_14000_rms_db", -120.0)
        ),
        "snare_rms_db": snare.get("rms_db"),
        "hats_rms_db": hats.get("rms_db"),
    }


def evaluate_gates(report: Dict) -> Dict:
    after = report["after"]
    wall = report.get("stereo_wall_analysis", {})
    kick = report.get("kick_analysis", {})
    audible = report.get("snare_hat_audibility", {})
    params = report.get("params", {})
    gates = {
        "bass_in_user_window": 2.4 <= after["bass_to_melody_rms_db"] <= 11.5,
        "melody_not_over_drums": after["drum_to_melody_rms_db"] >= -2.0,
        "pads_below_melody": after["pads_to_melody_rms_db"] <= -1.5,
        "wet_dry_tighter": after["return_to_dry_rms_db"] <= -24.0,
        "kick_not_dominated_by_bass_low": kick.get("bass_minus_kick_low_db", 99.0) <= 2.0,
        "snare_left_not_extreme": -0.58 <= wall.get("snare_mean_pan", 0.0) <= -0.12,
        "hats_right_not_extreme": 0.12 <= wall.get("hats_mean_pan", 0.0) <= 0.58,
        "master_mono_compatible": wall.get("master_stereo_corr", 1.0) >= 0.35,
        "snare_audible": audible.get("snare_rms_vs_drums_db", -120.0) >= float(params.get("snare_min_rms_vs_drums_db", -10.0)),
        "snare_presence_not_buried": audible.get("snare_presence_vs_melody_db", -120.0) >= float(params.get("snare_min_presence_vs_melody_db", -6.0)),
        "hats_audible": audible.get("hats_rms_vs_drums_db", -120.0) >= float(params.get("hats_min_rms_vs_drums_db", -22.0)),
    }
    gates["all_pass"] = all(gates.values())
    return gates


def score_golden_report(report: Dict) -> float:
    if not report:
        return -1e9
    gates = report.get("objective_gate_results", {})
    audible = report.get("snare_hat_audibility", {})
    after = report.get("after", {})
    score = 0.0
    score += sum(4.0 for value in gates.values() if value is True)
    score -= abs(after.get("return_to_dry_rms_db", -26.5) - (-26.5)) * 0.4
    score -= abs(after.get("drum_to_melody_rms_db", 3.0) - 3.0) * 0.5
    score -= max(0.0, -22.0 - audible.get("hats_rms_vs_drums_db", -120.0)) * 1.0
    score -= max(0.0, -10.0 - audible.get("snare_rms_vs_drums_db", -120.0)) * 0.8
    score -= max(0.0, -6.0 - audible.get("snare_presence_vs_melody_db", -120.0)) * 0.8
    score -= max(0.0, report.get("snare_ducking", {}).get("max_reduction_db", 0.0) - 2.0) * 2.0
    return float(score)


def next_pass_params(params: Dict, report: Dict) -> Dict:
    new_params = dict(params)
    gates = report.get("objective_gate_results", {})
    if not gates.get("hats_audible", True) or not gates.get("snare_audible", True):
        lift = float(new_params.get("snare_hat_pass_lift_db", 1.0))
        new_params["snare_hat_gain_db"] = float(min(
            float(new_params.get("snare_hat_gain_max_db", 4.5)),
            float(new_params.get("snare_hat_gain_db", 2.5)) + lift,
        ))
    if not gates.get("snare_presence_not_buried", True):
        new_params["snare_transient_boost_db"] = float(min(
            float(new_params.get("snare_transient_boost_max_db", 2.5)),
            float(new_params.get("snare_transient_boost_db", 1.5)) + 0.5,
        ))
    if not gates.get("snare_audible", True) or not gates.get("snare_presence_not_buried", True):
        new_params["snare_duck_depth_db"] = float(min(
            float(new_params.get("snare_duck_depth_max_db", 2.0)),
            float(new_params.get("snare_duck_depth_db", 1.0)) + 0.5,
        ))
    return new_params


def summarize_pass_reports(pass_reports: List[Dict]) -> List[Dict]:
    summary = []
    for report in pass_reports:
        summary.append({
            "pass_index": report.get("pass_index"),
            "score": score_golden_report(report),
            "all_pass": report.get("objective_gate_results", {}).get("all_pass"),
            "snare_hat_gain_db": report.get("params", {}).get("snare_hat_gain_db"),
            "snare_duck_depth_db": report.get("params", {}).get("snare_duck_depth_db"),
            "snare_transient_boost_db": report.get("params", {}).get("snare_transient_boost_db"),
            "snare_hat_audibility": report.get("snare_hat_audibility", {}),
        })
    return summary


def simple_audio_stats(y: np.ndarray, sr: int) -> Dict:
    y = ensure_stereo(np.asarray(y, dtype=np.float32))
    mono = np.mean(y, axis=1)
    mid = (y[:, 0] + y[:, 1]) * 0.5
    side = (y[:, 0] - y[:, 1]) * 0.5
    corr = float(np.corrcoef(y[:, 0], y[:, 1])[0, 1]) if len(y) > 2 else 1.0
    if not np.isfinite(corr):
        corr = 1.0
    return {
        "peak_db": amp_to_db(np.max(np.abs(y))),
        "rms_db": rms_db(y),
        "low_40_120_rms_db": band_rms_db(mono, sr, 40.0, 120.0),
        "low_mid_120_350_rms_db": band_rms_db(mono, sr, 120.0, 350.0),
        "presence_1200_5000_rms_db": band_rms_db(mono, sr, 1200.0, 5000.0),
        "air_6500_14000_rms_db": band_rms_db(mono, sr, 6500.0, 14000.0),
        "side_mid_ratio": float(np.sqrt(np.mean(side ** 2)) / (np.sqrt(np.mean(mid ** 2)) + 1e-12)),
        "stereo_corr": corr,
    }


def band_rms_db(mono: np.ndarray, sr: int, low: float, high: float) -> float:
    if len(mono) < 32:
        return -120.0
    n = int(min(len(mono), sr * 20))
    x = mono[:n] * np.hanning(n)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    spec = np.fft.rfft(x)
    mask = (freqs >= low) & (freqs <= min(high, sr * 0.49))
    if not np.any(mask):
        return -120.0
    energy = np.mean(np.abs(spec[mask]) ** 2) / max(n, 1)
    return amp_to_db(np.sqrt(energy))


def load_named_buses(paths: List[str]) -> Dict[str, Tuple[np.ndarray, int]]:
    buses = {}
    for src in paths or []:
        src_path = Path(src)
        if not src_path.exists():
            continue
        y, sr = sf.read(src_path, always_2d=True)
        buses[bus_name_from_path(src_path)] = (ensure_stereo(np.asarray(y, dtype=np.float32)), sr)
    return buses


def load_sum_paths(paths: Iterable[str]) -> Tuple[Optional[np.ndarray], Optional[int]]:
    result = None
    sr_out = None
    for path in paths or []:
        if not path or not os.path.exists(path):
            continue
        y, sr = sf.read(path, always_2d=True)
        y = ensure_stereo(np.asarray(y, dtype=np.float32))
        if result is None:
            result = np.zeros_like(y)
            sr_out = sr
        if sr != sr_out:
            continue
        result = sum_fit(result, y)
    return result, sr_out


def return_to_dry_ratio_db(return_paths: List[str], dry_paths: List[str]) -> float:
    return paths_rms_db(return_paths) - paths_rms_db(dry_paths)


def paths_rms_db(paths: List[str]) -> float:
    if not paths:
        return -120.0
    values = []
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        y, _ = sf.read(path, always_2d=True)
        values.append(float(np.mean(np.asarray(y, dtype=np.float64) ** 2)))
    if not values:
        return -120.0
    rms = float(np.sqrt(np.mean(values)) + 1e-12)
    return amp_to_db(rms)


def bus_ratio_db(metrics: Dict, target: str, anchor: str) -> float:
    if target not in metrics or anchor not in metrics:
        return 0.0
    return float(metrics[target]["rms_db"] - metrics[anchor]["rms_db"])


def bus_name_from_path(path: Path) -> str:
    name = path.stem.lower()
    if name.startswith("bus_"):
        name = name[4:]
    if name.startswith("automated_bus_"):
        name = name[len("automated_bus_"):]
    if name.startswith("auto_"):
        name = name[len("auto_"):]
    for suffix in ("_automated", "_mix"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name


def combined_envelope(paths: List[str]) -> np.ndarray:
    env = None
    for path in paths or []:
        y, _ = sf.read(path, always_2d=True)
        mono = np.mean(np.abs(np.asarray(y, dtype=np.float32)), axis=1)
        this_env = smooth_envelope(mono, attack=256, release=6000)
        if env is None:
            env = this_env
        else:
            n = max(len(env), len(this_env))
            padded = np.zeros(n, dtype=np.float32)
            padded[:len(env)] += env
            padded[:len(this_env)] += this_env
            env = padded
    return env if env is not None else np.zeros(0, dtype=np.float32)


def section_gain_curve(length: int, body_db: float, outro_db: float, outro_fraction: float) -> np.ndarray:
    curve = np.full(length, 10.0 ** (body_db / 20.0), dtype=np.float32)
    start = int(length * max(0.0, min(1.0, 1.0 - outro_fraction)))
    if start < length:
        ramp = np.linspace(0.0, 1.0, length - start, dtype=np.float32)
        body = 10.0 ** (body_db / 20.0)
        outro = 10.0 ** (outro_db / 20.0)
        curve[start:] = body + (outro - body) * ramp
    return curve


def return_duck_curve(length: int, dry_env: np.ndarray, body_db: float,
                      outro_db: float, outro_fraction: float) -> np.ndarray:
    if len(dry_env) == 0:
        return np.ones(length, dtype=np.float32)
    env = fit_1d(dry_env, length)
    threshold = float(np.percentile(env, 65))
    if threshold <= 1e-9:
        return np.ones(length, dtype=np.float32)
    control = np.clip((env - threshold) / max(np.percentile(env, 98) - threshold, 1e-9), 0.0, 1.0)
    body_gain_db = body_db * control
    outro_gain_db = outro_db * control
    start = int(length * max(0.0, min(1.0, 1.0 - outro_fraction)))
    gain_db = body_gain_db
    if start < length:
        ramp = np.linspace(0.0, 1.0, length - start, dtype=np.float32)
        gain_db[start:] = body_gain_db[start:] + (outro_gain_db[start:] - body_gain_db[start:]) * ramp
    return np.power(10.0, gain_db / 20.0).astype(np.float32)


def smooth_envelope(x: np.ndarray, attack: int, release: int) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    current = 0.0
    attack = max(1, int(attack))
    release = max(1, int(release))
    for i, value in enumerate(x.astype(np.float32)):
        coeff = 1.0 / (attack if value > current else release)
        current += (float(value) - current) * coeff
        out[i] = current
    return out


def fit_length(y: np.ndarray, length: int, channels: int) -> np.ndarray:
    y = ensure_stereo(np.asarray(y, dtype=np.float32))
    out = np.zeros((length, channels), dtype=np.float32)
    n = min(length, len(y))
    ch = min(channels, y.shape[1])
    out[:n, :ch] = y[:n, :ch]
    if channels > ch and ch == 1:
        out[:, 1:] = out[:, :1]
    return out


def fit_1d(y: np.ndarray, length: int) -> np.ndarray:
    out = np.zeros(length, dtype=np.float32)
    n = min(length, len(y))
    out[:n] = y[:n]
    return out


def sum_fit(base_y: np.ndarray, add_y: np.ndarray) -> np.ndarray:
    base_y = ensure_stereo(base_y)
    add_y = ensure_stereo(add_y)
    n = max(len(base_y), len(add_y))
    out = np.zeros((n, max(base_y.shape[1], add_y.shape[1])), dtype=np.float32)
    out[:len(base_y), :base_y.shape[1]] += base_y
    out[:len(add_y), :add_y.shape[1]] += add_y
    return out


def ensure_stereo(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return np.stack([y, y], axis=1)
    if y.shape[1] == 1:
        return np.repeat(y, 2, axis=1)
    return y[:, :2]


def apply_gain_db(y: np.ndarray, gain_db: float) -> np.ndarray:
    return (np.asarray(y, dtype=np.float32) * (10.0 ** (float(gain_db) / 20.0))).astype(np.float32)


def rms_db(y: np.ndarray) -> float:
    return amp_to_db(np.sqrt(np.mean(np.asarray(y, dtype=np.float64) ** 2)) if len(y) else 0.0)


def amp_to_db(value: float) -> float:
    return float(20.0 * np.log10(max(float(value), 1e-12)))


def copy_final_aliases(output_dir: str, master_path: Optional[str],
                       streaming_path: Optional[str] = None) -> Dict:
    """Create stable aliases matching the successful test-run names."""
    aliases = {}
    final_dir = Path(output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    if master_path and Path(master_path).exists():
        dest = final_dir / "full_song_master.wav"
        shutil.copy2(master_path, dest)
        aliases["full_song_master"] = str(dest)
    if streaming_path and Path(streaming_path).exists():
        dest = final_dir / "full_song_streaming_master.wav"
        shutil.copy2(streaming_path, dest)
        aliases["full_song_streaming_master"] = str(dest)
    return aliases
