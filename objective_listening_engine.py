"""
Objective Listening Engine.

Turns measurable WAV evidence into conservative mix decisions. This is not a
second DSP pipeline; it is a decision layer that recommends small actions for
the existing processors.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt


@dataclass
class MixAction:
    target: str
    action_type: str
    confidence: float
    params: Dict
    issue: str
    evidence: Dict
    accepted: Optional[bool] = None
    validation: Dict = None


@dataclass
class MixDiagnosis:
    stage: str
    actions: List[MixAction]
    observations: List[Dict]

    def to_dict(self) -> Dict:
        return {
            "stage": self.stage,
            "actions": [asdict(a) for a in self.actions],
            "observations": self.observations,
        }


class ObjectiveListeningEngine:
    """Analyze mix context and propose evidence-based corrective moves."""

    DEFAULT_SETTINGS = {
        "confidence_floor": 0.62,
        "max_eq_gain_db": {
            "kick": 1.0, "snare": 1.2, "hat": 1.2, "clap": 1.2,
            "bass": 1.4, "pad": 1.6, "chord": 1.6,
            "melody": 1.2, "counter": 1.3, "chorus": 1.3,
            "fx": 1.2, "default": 1.2,
        },
        "max_sidechain_depth_db": {"bass": 4.0, "return": 2.5, "default": 2.0},
        "profiles": {
            "lofi_warm": {
                "confidence_floor": 0.64,
                "low_mid_margin": 0.070,
                "harsh_margin": 0.085,
                "return_target_db": -18.0,
                "max_eq_scale": 0.90,
            },
            "hiphop_punchy": {
                "confidence_floor": 0.62,
                "low_mid_margin": 0.060,
                "harsh_margin": 0.075,
                "return_target_db": -19.5,
                "max_eq_scale": 1.00,
            },
            "armenian_cinematic": {
                "confidence_floor": 0.66,
                "low_mid_margin": 0.080,
                "harsh_margin": 0.095,
                "return_target_db": -16.5,
                "max_eq_scale": 0.85,
            },
            "clean_pristine": {
                "confidence_floor": 0.68,
                "low_mid_margin": 0.050,
                "harsh_margin": 0.065,
                "return_target_db": -21.0,
                "max_eq_scale": 0.75,
            },
        },
        "sections": {},
    }

    ROLE_PRIORITY = {
        "kick": 1, "snare": 2, "bass": 3, "melody": 4, "chorus": 4,
        "counter": 5, "pad": 6, "chord": 6, "hat": 7, "clap": 7,
        "perc": 7, "fx": 8, "default": 9,
    }

    BANDS = {
        "sub": (25, 80),
        "bass": (80, 160),
        "low_mid": (180, 500),
        "body": (500, 1200),
        "presence": (1200, 5000),
        "harsh": (2500, 6500),
        "air": (6500, 14000),
    }

    def __init__(self, confidence_floor: float = None,
                 settings: Dict = None, profile: str = "lofi_warm"):
        self.settings = _deep_merge(self.DEFAULT_SETTINGS, settings or {})
        self.profile_name = profile if profile in self.settings.get("profiles", {}) else "lofi_warm"
        self.profile = self.settings.get("profiles", {}).get(self.profile_name, {})
        self.confidence_floor = float(
            confidence_floor
            if confidence_floor is not None
            else self.profile.get("confidence_floor", self.settings.get("confidence_floor", 0.62))
        )
        self.reports: List[MixDiagnosis] = []

    # ------------------------------------------------------------------
    # Public analysis API
    # ------------------------------------------------------------------

    def analyze_stems(self, stem_paths: List[str], sr: int, bpm: float) -> MixDiagnosis:
        metrics = {}
        for path in stem_paths:
            if not path or not os.path.exists(path):
                continue
            name = os.path.basename(path)
            y, file_sr = sf.read(path, always_2d=True)
            y = _ensure_stereo(np.asarray(y, dtype=np.float32))
            if np.max(np.abs(y)) < 1e-7:
                continue
            metrics[name] = self._stem_metrics(y, file_sr, name)

        actions: List[MixAction] = []
        observations = [self._metric_observation(name, m) for name, m in metrics.items()]
        if not metrics:
            diag = MixDiagnosis("stems", actions, observations)
            self.reports.append(diag)
            return diag

        actions.extend(self._stem_tonal_actions(metrics))
        actions.extend(self._melody_audibility_actions(metrics))
        diag = MixDiagnosis("stems", self._gate_actions(actions), observations)
        self.reports.append(diag)
        return diag

    def analyze_buses(self, bus_audio: Dict[str, np.ndarray], sr: int,
                      bpm: float) -> MixDiagnosis:
        metrics = {
            name: self._audio_metrics(_ensure_stereo(y), sr, name)
            for name, y in bus_audio.items() if y is not None
        }
        actions: List[MixAction] = []
        observations = [self._metric_observation(name, m) for name, m in metrics.items()]

        if "bass" in metrics and "drums" in metrics:
            bass_e = metrics["bass"]["bands"]
            drum_e = metrics["drums"]["bands"]
            sub_overlap = min(bass_e["sub"], drum_e["sub"])
            bass_overlap = min(bass_e["bass"], drum_e["bass"])
            overlap = max(sub_overlap, bass_overlap)
            dominance = max(drum_e["sub"] - bass_e["sub"], drum_e["bass"] - bass_e["bass"])
            confidence = _scale01((overlap + 42.0) / 18.0)
            if confidence >= 0.55:
                max_depth = float(self.settings.get("max_sidechain_depth_db", {}).get("bass", 4.0))
                depth = float(np.clip(1.0 + confidence * 2.5 + max(0, dominance) * 0.08, 1.0, max_depth))
                release_ms = 45.0
                actions.append(MixAction(
                    target="bass",
                    action_type="sidechain",
                    confidence=confidence,
                    params={"depth_db": depth, "release_ms": release_ms, "freq_range": (40, 125)},
                    issue="kick_bass_masking",
                    evidence={
                        "overlap_db": overlap,
                        "sub_overlap_db": sub_overlap,
                        "bass_overlap_db": bass_overlap,
                        "kick_low_minus_bass_low_db": dominance,
                    },
                ))

        for bus_name, m in metrics.items():
            corr = m.get("per_band_corr", {})
            if bus_name in {"bass", "drums"} and corr.get("sub", 1.0) < 0.82:
                confidence = _scale01((0.82 - corr.get("sub", 1.0)) / 0.35)
                actions.append(MixAction(
                    target=bus_name,
                    action_type="mono_sub",
                    confidence=confidence,
                    params={"cutoff": 95.0 if bus_name == "bass" else 80.0},
                    issue="low_end_mono_translation",
                    evidence={"sub_correlation": corr.get("sub", 1.0)},
                ))

            bus_low_mid_baseline = self._relative_ratio_baseline(metrics, "low_mid", fallback=0.26)
            if m["ratios"]["low_mid"] > bus_low_mid_baseline and bus_name in {"melody", "pads", "bass"}:
                excess = m["ratios"]["low_mid"] - bus_low_mid_baseline
                confidence = _scale01(excess / 0.16)
                role = "bass" if bus_name == "bass" else ("pad" if bus_name == "pads" else "melody")
                actions.append(MixAction(
                    target=bus_name,
                    action_type="dynamic_eq",
                    confidence=confidence,
                    params={"freq": 285.0, "gain_db": -self._capped_gain(role, confidence, 1.8), "q": 1.0},
                    issue="bus_low_mid_buildup",
                    evidence={"low_mid_ratio": m["ratios"]["low_mid"], "relative_baseline": bus_low_mid_baseline},
                ))

        diag = MixDiagnosis("buses", self._gate_actions(actions), observations)
        self.reports.append(diag)
        return diag

    def analyze_returns(self, dry_paths: List[str], return_paths: List[str],
                        sr: int, bpm: float) -> MixDiagnosis:
        dry_rms = self._sum_rms_from_paths(dry_paths)
        actions: List[MixAction] = []
        observations = []

        for path in return_paths:
            if not path or not os.path.exists(path):
                continue
            name = os.path.basename(path)
            y, file_sr = sf.read(path, always_2d=True)
            y = _ensure_stereo(np.asarray(y, dtype=np.float32))
            if np.max(np.abs(y)) < 1e-7:
                continue
            m = self._audio_metrics(y, file_sr, name)
            rel_db = _db(m["rms"] / max(dry_rms, 1e-12))
            tail_density = m["ratios"]["low_mid"] + m["ratios"]["presence"]
            observations.append({
                **self._metric_observation(name, m),
                "return_to_dry_db": rel_db,
                "tail_density": tail_density,
            })
            return_target_db = float(self.profile.get("return_target_db", -18.0))
            if rel_db > return_target_db or tail_density > 0.45:
                confidence = max(
                    _scale01((rel_db - return_target_db) / 10.0),
                    _scale01((tail_density - 0.38) / 0.20),
                )
                actions.append(MixAction(
                    target=name,
                    action_type="filter_return",
                    confidence=confidence,
                    params={
                        "highpass": 170.0 if "reverb" in name.lower() else 140.0,
                        "lowpass": 7600.0 if "reverb" in name.lower() else 9000.0,
                        "gain_db": -float(np.clip(confidence * 2.2, 0.5, 2.2)),
                    },
                    issue="ambience_wash",
                    evidence={"return_to_dry_db": rel_db, "tail_density": tail_density},
                ))

        diag = MixDiagnosis("returns", self._gate_actions(actions), observations)
        self.reports.append(diag)
        return diag

    def analyze_premaster(self, mix_y: np.ndarray, sr: int, bpm: float,
                          reference_analysis: Dict = None) -> MixDiagnosis:
        y = _ensure_stereo(np.asarray(mix_y, dtype=np.float32))
        m = self._audio_metrics(y, sr, "premaster")
        actions: List[MixAction] = []
        observations = [self._metric_observation("premaster", m)]
        section_obs = self._section_contrast_observation(y, sr, bpm)
        if section_obs:
            observations.append(section_obs)

        peak_db = m["summary"]["peak_db"]
        if peak_db > -3.0:
            gain_db = float(np.clip(-6.0 - peak_db, -4.0, -0.5))
            confidence = _scale01((peak_db + 3.0) / 5.0)
            actions.append(MixAction(
                target="premaster",
                action_type="gain_trim",
                confidence=confidence,
                params={"gain_db": gain_db},
                issue="premaster_headroom",
                evidence={"peak_db": peak_db},
            ))

        if m["per_band_corr"].get("sub", 1.0) < 0.86:
            confidence = _scale01((0.86 - m["per_band_corr"]["sub"]) / 0.30)
            actions.append(MixAction(
                target="premaster",
                action_type="mono_sub",
                confidence=confidence,
                params={"cutoff": 95.0},
                issue="premaster_low_end_mono_translation",
                evidence={"sub_correlation": m["per_band_corr"]["sub"]},
            ))

        if m["ratios"]["harsh"] > 0.30:
            confidence = _scale01((m["ratios"]["harsh"] - 0.26) / 0.16)
            actions.append(MixAction(
                target="premaster",
                action_type="dynamic_eq",
                confidence=confidence,
                params={"freq": 3900.0, "gain_db": -float(np.clip(confidence * 1.2, 0.3, 1.2)), "q": 0.9},
                issue="premaster_harshness",
                evidence={"harsh_ratio": m["ratios"]["harsh"]},
            ))

        diag = MixDiagnosis("premaster", self._gate_actions(actions), observations)
        self.reports.append(diag)
        return diag

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def write_reports(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "diagnosis_report.json")
        md_path = os.path.join(output_dir, "diagnosis_report.md")
        report_dicts = [diag.to_dict() for diag in self.reports]
        with open(json_path, "w") as f:
            json.dump({"profile": self.profile_name, "diagnoses": report_dicts}, f, indent=2)
        with open(md_path, "w") as f:
            f.write("# Objective Listening Report\n\n")
            f.write(f"Profile: `{self.profile_name}`\n\n")
            for diag in report_dicts:
                f.write(f"## {diag['stage'].title()}\n\n")
                actions = diag.get("actions", [])
                if not actions:
                    f.write("No high-confidence actions.\n\n")
                    continue
                f.write("| Target | Issue | Action | Confidence | Result | Evidence | Validation |\n")
                f.write("|---|---|---|---:|---|---|---|\n")
                for action in actions:
                    evidence = ", ".join(f"{k}={_fmt(v)}" for k, v in action.get("evidence", {}).items())
                    validation = action.get("validation") or {}
                    validation_text = ", ".join(f"{k}={_fmt(v)}" for k, v in validation.items())
                    result = "pending"
                    if action.get("accepted") is True:
                        result = "accepted"
                    elif action.get("accepted") is False:
                        result = "rejected"
                    f.write(
                        f"| {action['target']} | {action['issue']} | {action['action_type']} "
                        f"{action.get('params', {})} | {action['confidence']:.2f} | "
                        f"{result} | {evidence} | {validation_text} |\n"
                    )
                f.write("\n")
        return json_path, md_path

    # ------------------------------------------------------------------
    # Validation API
    # ------------------------------------------------------------------

    def measure_audio(self, y: np.ndarray, sr: int, name: str = "candidate") -> Dict:
        """Public wrapper used by the production engine for before/after checks."""
        return self._audio_metrics(_ensure_stereo(np.asarray(y, dtype=np.float32)), sr, name)

    def validate_action(self, action: MixAction, before: Dict, after: Dict) -> bool:
        """Accept only changes that improve the metric the action was meant to fix."""
        accepted = True
        validation = {}

        if action.action_type == "skip":
            accepted = False
            validation = {"reason": "analysis_only"}
        elif action.action_type == "dynamic_eq":
            metric = self._validation_ratio_for_issue(action.issue)
            before_value = before["ratios"].get(metric, 0.0)
            after_value = after["ratios"].get(metric, 0.0)
            accepted = after_value <= before_value + 0.006
            validation = {
                f"before_{metric}_ratio": before_value,
                f"after_{metric}_ratio": after_value,
                "delta": after_value - before_value,
            }
        elif action.action_type == "gain_trim":
            before_peak = before["summary"]["peak_db"]
            after_peak = after["summary"]["peak_db"]
            accepted = after_peak <= before_peak + 0.10
            validation = {
                "before_peak_db": before_peak,
                "after_peak_db": after_peak,
                "delta_db": after_peak - before_peak,
            }
        elif action.action_type == "mono_sub":
            before_corr = before["per_band_corr"].get("sub", 1.0)
            after_corr = after["per_band_corr"].get("sub", 1.0)
            accepted = after_corr >= before_corr - 0.02
            validation = {
                "before_sub_correlation": before_corr,
                "after_sub_correlation": after_corr,
                "delta": after_corr - before_corr,
            }
        elif action.action_type in {"filter_return", "duck_return"}:
            before_density = before["ratios"]["low_mid"] + before["ratios"]["presence"]
            after_density = after["ratios"]["low_mid"] + after["ratios"]["presence"]
            before_rms = before["summary"]["rms_db"]
            after_rms = after["summary"]["rms_db"]
            accepted = after_density <= before_density + 0.012 or after_rms <= before_rms + 0.10
            validation = {
                "before_tail_density": before_density,
                "after_tail_density": after_density,
                "before_rms_db": before_rms,
                "after_rms_db": after_rms,
            }
        elif action.action_type == "sidechain":
            before_sub = before["bands"].get("sub", -90.0)
            after_sub = after["bands"].get("sub", -90.0)
            before_rms = before["summary"]["rms_db"]
            after_rms = after["summary"]["rms_db"]
            accepted = after_sub <= before_sub + 0.25 and after_rms >= before_rms - 4.5
            validation = {
                "before_sub_db": before_sub,
                "after_sub_db": after_sub,
                "before_rms_db": before_rms,
                "after_rms_db": after_rms,
            }
        elif action.action_type in {"width_adjust", "saturate"}:
            before_peak = before["summary"]["peak_db"]
            after_peak = after["summary"]["peak_db"]
            accepted = after_peak <= max(-0.5, before_peak + 1.0)
            validation = {"before_peak_db": before_peak, "after_peak_db": after_peak}

        action.accepted = bool(accepted)
        action.validation = validation
        return bool(accepted)

    # ------------------------------------------------------------------
    # Internal decisions
    # ------------------------------------------------------------------

    def _stem_tonal_actions(self, metrics: Dict[str, Dict]) -> List[MixAction]:
        actions: List[MixAction] = []
        role_groups = {}
        for _, m in metrics.items():
            role_groups.setdefault(m["role"], []).append(m)

        for name, m in metrics.items():
            role = m["role"]
            ratios = m["ratios"]
            if role in {"pad", "chord", "melody", "counter", "chorus", "bass"}:
                low_mid_baseline = self._role_ratio_baseline(
                    role_groups, role, "low_mid",
                    floor=0.24,
                    margin=float(self.profile.get("low_mid_margin", 0.07)),
                )
            else:
                low_mid_baseline = 1.0
            if ratios["low_mid"] > low_mid_baseline:
                excess = ratios["low_mid"] - low_mid_baseline
                confidence = _scale01(excess / 0.17)
                actions.append(MixAction(
                    target=name, action_type="dynamic_eq", confidence=confidence,
                    params={"freq": 300.0, "gain_db": -self._capped_gain(role, confidence, 1.6), "q": 1.1},
                    issue="stem_mud_boxiness",
                    evidence={"low_mid_ratio": ratios["low_mid"], "relative_baseline": low_mid_baseline, "role": role},
                ))

            harsh_limit = 0.30 if role in {"hat", "snare", "clap"} else 0.26
            if role in {"hat", "snare", "clap", "melody", "counter", "chorus"}:
                harsh_baseline = self._role_ratio_baseline(
                    role_groups, role, "harsh",
                    floor=harsh_limit,
                    margin=float(self.profile.get("harsh_margin", 0.08)),
                )
            else:
                harsh_baseline = 1.0
            if ratios["harsh"] > harsh_baseline:
                excess = ratios["harsh"] - harsh_baseline
                confidence = _scale01(excess / 0.16)
                actions.append(MixAction(
                    target=name, action_type="dynamic_eq", confidence=confidence,
                    params={"freq": 4200.0, "gain_db": -self._capped_gain(role, confidence, 1.4), "q": 1.0},
                    issue="stem_harshness_fatigue",
                    evidence={"harsh_ratio": ratios["harsh"], "relative_baseline": harsh_baseline, "role": role},
                ))
        return actions

    def _melody_audibility_actions(self, metrics: Dict[str, Dict]) -> List[MixAction]:
        actions: List[MixAction] = []
        melody_items = [(n, m) for n, m in metrics.items() if m["role"] in {"melody", "counter", "chorus"}]
        mask_items = [(n, m) for n, m in metrics.items() if m["role"] in {"pad", "chord", "hat", "snare", "clap"}]
        if not melody_items or not mask_items:
            return actions

        mel_presence = np.median([m["bands"]["presence"] for _, m in melody_items])
        for name, m in mask_items:
            excess_db = m["bands"]["presence"] - mel_presence
            priority_ok = self.ROLE_PRIORITY.get(m["role"], 9) >= 6
            if priority_ok and excess_db > 2.5:
                confidence = _scale01((excess_db - 2.5) / 8.0)
                actions.append(MixAction(
                    target=name, action_type="dynamic_eq", confidence=confidence,
                    params={"freq": 2600.0, "gain_db": -self._capped_gain(m["role"], confidence, 1.5), "q": 1.0},
                    issue="melody_presence_masking",
                    evidence={"mask_presence_minus_melody_db": excess_db, "role": m["role"]},
                ))
        return actions

    def _gate_actions(self, actions: List[MixAction]) -> List[MixAction]:
        return [a for a in actions if a.confidence >= self.confidence_floor]

    def _role_ratio_baseline(self, role_groups: Dict[str, List[Dict]], role: str,
                             ratio_name: str, floor: float, margin: float) -> float:
        role_values = [m["ratios"][ratio_name] for m in role_groups.get(role, [])]
        all_values = [m["ratios"][ratio_name] for group in role_groups.values() for m in group]
        candidates = []
        if len(role_values) > 1:
            candidates.append(float(np.median(role_values)) + margin)
        if len(all_values) > 1:
            candidates.append(float(np.median(all_values)) + margin)
        if candidates:
            return max(floor, min(candidates))
        return floor + margin

    def _relative_ratio_baseline(self, metrics: Dict[str, Dict],
                                 ratio_name: str, fallback: float) -> float:
        values = [m["ratios"][ratio_name] for m in metrics.values()]
        if not values:
            return fallback
        margin = float(self.profile.get("low_mid_margin", 0.07))
        return max(fallback, float(np.median(values)) + margin)

    def _capped_gain(self, role: str, confidence: float, nominal_max: float) -> float:
        role_caps = self.settings.get("max_eq_gain_db", {})
        role_cap = float(role_caps.get(role, role_caps.get("default", nominal_max)))
        profile_scale = float(self.profile.get("max_eq_scale", 1.0))
        cap = max(0.2, min(nominal_max, role_cap) * profile_scale)
        return float(np.clip(confidence * cap, 0.3, cap))

    def _metric_observation(self, name: str, metrics: Dict) -> Dict:
        return {
            "target": name,
            "role": metrics.get("role"),
            **metrics["summary"],
            "ratios": {k: float(v) for k, v in metrics["ratios"].items()},
            "per_band_corr": {k: float(v) for k, v in metrics.get("per_band_corr", {}).items()},
        }

    def _section_contrast_observation(self, y: np.ndarray, sr: int, bpm: float) -> Dict:
        sections = self.settings.get("sections") or {}
        if not sections or bpm <= 0:
            return {}

        bar_samples = int(sr * (60.0 / bpm) * 4.0)
        if bar_samples <= 0:
            return {}

        mono = np.mean(_ensure_stereo(y), axis=1)
        energies = {}
        for section_name, span in sections.items():
            if not isinstance(span, (list, tuple)) or len(span) != 2:
                continue
            start = max(0, int(span[0]) * bar_samples)
            end = min(len(mono), int(span[1]) * bar_samples)
            if end <= start + 64:
                continue
            section = mono[start:end]
            energies[section_name] = _db(float(np.sqrt(np.mean(section.astype(np.float64) ** 2)) + 1e-12))

        verse_vals = [v for k, v in energies.items() if "verse" in k]
        chorus_vals = [v for k, v in energies.items() if "chorus" in k and "pre" not in k]
        contrast = None
        if verse_vals and chorus_vals:
            contrast = float(np.median(chorus_vals) - np.median(verse_vals))

        return {
            "metric": "section_contrast",
            "section_rms_db": energies,
            "chorus_minus_verse_db": contrast,
            "guidance": "preserve contrast during mastering; avoid corrective gain that flattens sections",
        }

    def _validation_ratio_for_issue(self, issue: str) -> str:
        if "mud" in issue or "boxiness" in issue or "low_mid" in issue:
            return "low_mid"
        if "harsh" in issue or "fatigue" in issue:
            return "harsh"
        if "presence" in issue or "melody" in issue:
            return "presence"
        return "body"

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _stem_metrics(self, y: np.ndarray, sr: int, name: str) -> Dict:
        m = self._audio_metrics(y, sr, name)
        m["role"] = detect_role(name)
        return m

    def _audio_metrics(self, y: np.ndarray, sr: int, name: str) -> Dict:
        y = _ensure_stereo(y)
        mono = np.mean(y, axis=1)
        bands = {band: self._band_energy_db(mono, sr, lo, hi)
                 for band, (lo, hi) in self.BANDS.items()}
        lin = {k: 10.0 ** (v / 20.0) for k, v in bands.items()}
        total = sum(lin.values()) or 1e-12
        ratios = {k: v / total for k, v in lin.items()}
        peak = float(np.max(np.abs(y)))
        rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-12)
        return {
            "name": name,
            "bands": bands,
            "ratios": ratios,
            "rms": rms,
            "summary": {
                "peak_db": 20.0 * np.log10(max(peak, 1e-12)),
                "rms_db": 20.0 * np.log10(max(rms, 1e-12)),
                "crest_db": 20.0 * np.log10(max(peak, 1e-12) / rms),
            },
            "per_band_corr": self._per_band_correlation(y, sr),
        }

    def _sum_rms_from_paths(self, paths: List[str]) -> float:
        values = []
        for path in paths:
            if not path or not os.path.exists(path):
                continue
            try:
                y, _ = sf.read(path, always_2d=True)
                values.append(float(np.sqrt(np.mean(np.asarray(y, dtype=np.float64) ** 2)) + 1e-12))
            except Exception:
                continue
        return float(np.median(values)) if values else 1e-6

    def _band_energy_db(self, mono: np.ndarray, sr: int, low: float, high: float) -> float:
        if len(mono) < 32:
            return -90.0
        nyq = sr / 2.0
        lo = max(low / nyq, 0.001)
        hi = min(high / nyq, 0.999)
        if lo >= hi:
            return -90.0
        sos = butter(2, [lo, hi], btype="band", output="sos")
        filtered = sosfilt(sos, mono.astype(np.float64))
        rms = np.sqrt(np.mean(filtered ** 2) + 1e-12)
        return float(20.0 * np.log10(max(rms, 1e-12)))

    def _per_band_correlation(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        if y.ndim < 2 or y.shape[1] < 2:
            return {"sub": 1.0, "low": 1.0, "mid": 1.0, "high": 1.0}
        bands = {"sub": (25, 110), "low": (110, 350), "mid": (350, 2500), "high": (2500, 12000)}
        result = {}
        for name, (lo_hz, hi_hz) in bands.items():
            nyq = sr / 2.0
            lo = max(lo_hz / nyq, 0.001)
            hi = min(hi_hz / nyq, 0.999)
            if lo >= hi:
                result[name] = 1.0
                continue
            sos = butter(2, [lo, hi], btype="band", output="sos")
            left = sosfilt(sos, y[:, 0].astype(np.float64))
            right = sosfilt(sos, y[:, 1].astype(np.float64))
            denom = float(np.std(left) * np.std(right))
            corr = float(np.corrcoef(left, right)[0, 1]) if denom > 1e-12 else 1.0
            result[name] = corr if np.isfinite(corr) else 1.0
        return result


def detect_role(name: str) -> str:
    n = name.lower()
    if "kick" in n:
        return "kick"
    if "snare" in n:
        return "snare"
    if "hat" in n:
        return "hat"
    if "clap" in n:
        return "clap"
    if "bass" in n:
        return "bass"
    if "pad" in n:
        return "pad"
    if "chord" in n:
        return "chord"
    if "counter" in n:
        return "counter"
    if "chorus" in n:
        return "chorus"
    if "melody" in n or "lead" in n:
        return "melody"
    if "perc" in n or "tambourine" in n or "maracas" in n:
        return "perc"
    if "fx" in n:
        return "fx"
    return "default"


def _ensure_stereo(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return np.stack([y, y], axis=1)
    if y.shape[1] == 1:
        return np.repeat(y, 2, axis=1)
    return y[:, :2]


def _scale01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _db(value: float) -> float:
    return float(20.0 * np.log10(max(value, 1e-12)))


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.3g}"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}:{_fmt(v)}" for k, v in value.items()) + "}"
    return str(value)


def _deep_merge(base: Dict, override: Dict) -> Dict:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
