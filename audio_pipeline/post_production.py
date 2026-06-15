import os
import random
import shutil
import gc
import json
from gain_staging import GainStager
import subprocess
import numpy as np
import soundfile as sf
import librosa
from typing import Dict
from spectral_processing import SpectralAnalyzer, AlgorithmicEQ

try:
    from song_structure import get_bar_type, get_phrase_position
except ImportError:
    import sys
    _V10_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "v10_refactored")
    if _V10_DIR not in sys.path:
        sys.path.append(_V10_DIR)
    from song_structure import get_bar_type, get_phrase_position

class ProductionEngine:
    """
    Advanced Autonomous Production Engine.
    Handles mixing, objective EQ, panning, and mastering.
    """
    
    def __init__(self, output_dir: str = "output/mastered"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.stager = GainStager(output_dir=os.path.join(output_dir, "staged"))
        self.eq = AlgorithmicEQ()
        self.analyzer = SpectralAnalyzer()
        self.preview_mode = False  # Set to True to enable iterative preview
        self.mix_headroom_peak = 10 ** (-6.0 / 20.0)
        self.bus_headroom_peak = 10 ** (-4.0 / 20.0)
        self.reference_track_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "output", "reference", "TheAlchemist_Tight_812814020637_1_5.mp3"
        )
        self.reference_analysis = None
        self.current_report = None

    def _bump_report_counter(self, key: str, amount: int = 1):
        if self.current_report is None:
            return
        counters = self.current_report.setdefault("counters", {})
        counters[key] = counters.get(key, 0) + amount

    def analyze_audio_characteristics(self, path: str) -> dict:
        """Analyze audio file and return comprehensive characteristics."""
        import pyloudnorm as pyln
        y, sr = sf.read(path)
        meter = pyln.Meter(sr)

        # Loudness
        lufs = meter.integrated_loudness(y)

        # Spectral balance (lows, mids, highs)
        lows = SpectralAnalyzer.detect_buildup(y, sr, (20, 150))
        mids = SpectralAnalyzer.detect_buildup(y, sr, (150, 2500))
        highs = SpectralAnalyzer.detect_buildup(y, sr, (2500, 20000))

        # Dynamic range (simplified)
        peak = np.max(np.abs(y))
        rms = np.sqrt(np.mean(y**2))
        crest_factor = peak / (rms + 1e-10)

        return {
            'lufs': lufs,
            'peak': peak,
            'rms': rms,
            'crest_factor': crest_factor,
            'spectral': {'lows': lows, 'mids': mids, 'highs': highs}
        }

    def print_audio_analysis(self, analysis: dict, label: str = "Analysis"):
        """Pretty-print audio analysis results."""
        print(f"  [{label}]")
        print(f"    LUFS: {analysis['lufs']:+.1f}")
        print(f"    Peak: {analysis['peak']:.3f} ({20*np.log10(max(1e-10, analysis['peak'])):.1f} dB)")
        print(f"    RMS:  {analysis['rms']:.3f}")
        print(f"    Crest Factor: {analysis['crest_factor']:.2f}")
        s = analysis['spectral']
        print(f"    Spectral: Lows={s['lows']:.2f}, Mids={s['mids']:.2f}, Highs={s['highs']:.2f}")

    def _read_audio_any(self, path: str, always_2d: bool = True):
        """Read wav/mp3 input, falling back to librosa when libsndfile cannot decode."""
        try:
            return sf.read(path, always_2d=always_2d)
        except Exception:
            y, sr = librosa.load(path, sr=None, mono=False)
            if y.ndim == 1:
                data = y[:, np.newaxis] if always_2d else y
            else:
                data = y.T
            return data.astype(np.float32), sr

    def _analyze_array(self, y: np.ndarray, sr: int, label: str = "") -> dict:
        """Compact objective analysis for mix decisions and reports."""
        import pyloudnorm as pyln
        y = self._ensure_stereo(np.asarray(y, dtype=np.float32))
        mono = self._mono(y)
        meter = pyln.Meter(sr)
        try:
            lufs = float(meter.integrated_loudness(y))
        except Exception:
            lufs = None
        peak = float(np.max(np.abs(y)))
        rms = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
        crest_db = float(20.0 * np.log10(max(peak, 1e-12) / rms))
        sub = self._band_isolate(y, sr, 30.0, 150.0)
        sub_peak = float(np.max(np.abs(sub)))
        sub_rms = float(np.sqrt(np.mean(sub ** 2)) + 1e-12)
        low_crest_db = float(20.0 * np.log10(max(sub_peak, 1e-12) / sub_rms))
        corr = 1.0
        if y.ndim == 2 and y.shape[1] >= 2:
            left = y[:, 0]
            right = y[:, 1]
            denom = float(np.std(left) * np.std(right))
            if denom > 1e-12:
                corr = float(np.corrcoef(left, right)[0, 1])
        bands = SpectralAnalyzer.GRANULAR_BANDS
        profile = SpectralAnalyzer.get_spectral_profile(y, sr, bands=bands)
        band_profile = {f"{lo}-{hi}": float(v) for (lo, hi), v in profile.items()}
        return {
            "label": label,
            "lufs": lufs,
            "peak": peak,
            "peak_db": float(20.0 * np.log10(max(peak, 1e-12))),
            "rms": rms,
            "crest_db": crest_db,
            "low_crest_db": low_crest_db,
            "stereo_correlation": corr,
            "spectral_profile": band_profile,
        }

    def _debug_analyze_array(self, y: np.ndarray, sr: int, label: str = "") -> dict:
        """
        Fast before/after metrics for debug traces.
        Avoids full BS.1770 LUFS and granular spectral analysis so long stem runs
        do not stall while still capturing level, crest, and stereo changes.
        """
        y = self._ensure_stereo(np.asarray(y, dtype=np.float32))
        mono = self._mono(y)
        peak = float(np.max(np.abs(y)))
        rms = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
        active_rms_db = self._active_rms_db(y)
        corr = 1.0
        if y.ndim == 2 and y.shape[1] >= 2:
            left = y[:, 0]
            right = y[:, 1]
            denom = float(np.std(left) * np.std(right))
            if denom > 1e-12:
                corr = float(np.corrcoef(left, right)[0, 1])
        return {
            "label": label,
            "lufs": None,
            "peak": peak,
            "peak_db": float(20.0 * np.log10(max(peak, 1e-12))),
            "rms": rms,
            "rms_db": float(20.0 * np.log10(max(rms, 1e-12))),
            "active_rms_db": active_rms_db,
            "crest_db": float(20.0 * np.log10(max(peak, 1e-12) / rms)),
            "low_crest_db": None,
            "stereo_correlation": corr,
            "duration_sec": float(len(mono) / max(sr, 1)),
        }

    def _analyze_file(self, path: str, label: str = "") -> dict:
        y, sr = self._read_audio_any(path, always_2d=True)
        data = self._analyze_array(y, sr, label or os.path.basename(path))
        data["path"] = path
        data["sample_rate"] = sr
        return data

    def _load_reference_analysis(self):
        if self.reference_analysis is not None:
            return self.reference_analysis
        if not os.path.exists(self.reference_track_path):
            print(f"  [Reference] Missing: {self.reference_track_path}")
            self.reference_analysis = None
            return None
        try:
            self.reference_analysis = self._analyze_file(self.reference_track_path, "reference")
            print(f"  [Reference] Loaded: {os.path.basename(self.reference_track_path)}")
        except Exception as e:
            print(f"  [Reference] Analysis failed: {e}")
            self.reference_analysis = None
        return self.reference_analysis

    def _start_mix_report(self, song_name: str, bpm: float, stem_paths: list):
        self.current_report = {
            "song_name": song_name,
            "bpm": bpm,
            "reference_path": self.reference_track_path if os.path.exists(self.reference_track_path) else None,
            "reference": self._load_reference_analysis(),
            "stems": [{"path": p, "name": os.path.basename(p)} for p in stem_paths],
            "buses": {},
            "decisions": [],
            "debug_events": [],
            "masters": {},
            "counters": {},
        }

    def _json_safe(self, value):
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return self._json_safe(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _report_decision(self, stage: str, message: str, data: dict = None):
        if self.current_report is None:
            return
        self.current_report["decisions"].append({
            "stage": stage,
            "message": message,
            "data": self._json_safe(data or {}),
        })

    def _report_debug_event(self, stage: str, subject: str, params: dict = None,
                            before: dict = None, after: dict = None, notes: list = None):
        if self.current_report is None:
            return
        self.current_report["debug_events"].append({
            "stage": stage,
            "subject": subject,
            "params": self._json_safe(params or {}),
            "before": self._json_safe(before or {}),
            "after": self._json_safe(after or {}),
            "notes": self._json_safe(notes or []),
        })

    def _write_mix_report(self, song_name: str):
        if not self.current_report:
            return
        json_path = os.path.join(self.output_dir, f"{song_name}_mix_report.json")
        md_path = os.path.join(self.output_dir, f"{song_name}_mix_report.md")
        debug_json_path = os.path.join(self.output_dir, f"{song_name}_debug_report.json")
        debug_md_path = os.path.join(self.output_dir, f"{song_name}_debug_report.md")
        try:
            with open(json_path, "w") as f:
                json.dump(self._json_safe(self.current_report), f, indent=2)
            lines = [
                f"# Mix Report: {song_name}",
                "",
                f"- Reference: {self.current_report.get('reference_path') or 'None'}",
                f"- Decisions: {len(self.current_report.get('decisions', []))}",
                f"- Debug Events: {len(self.current_report.get('debug_events', []))}",
                "",
                "## Counters",
            ]
            for key, value in self.current_report.get("counters", {}).items():
                lines.append(f"- {key}: {value}")
            lines += [
                "",
                "## Bus Metrics",
            ]
            for bus, data in self.current_report.get("buses", {}).items():
                lines.append(
                    f"- {bus}: LUFS={data.get('lufs')} peak={data.get('peak_db', 0):+.1f}dB "
                    f"crest={data.get('crest_db', 0):.1f}dB low_crest={data.get('low_crest_db', 0):.1f}dB"
                )
            lines += ["", "## Decisions"]
            for d in self.current_report.get("decisions", []):
                lines.append(f"- [{d['stage']}] {d['message']}")
            with open(md_path, "w") as f:
                f.write("\n".join(lines))

            debug_payload = {
                "song_name": song_name,
                "bpm": self.current_report.get("bpm"),
                "reference_path": self.current_report.get("reference_path"),
                "reference": self.current_report.get("reference"),
                "debug_events": self.current_report.get("debug_events", []),
                "masters": self.current_report.get("masters", {}),
                "buses": self.current_report.get("buses", {}),
                "counters": self.current_report.get("counters", {}),
            }
            with open(debug_json_path, "w") as f:
                json.dump(self._json_safe(debug_payload), f, indent=2)
            debug_lines = [
                f"# Debug Mix Report: {song_name}",
                "",
                f"- Reference: {self.current_report.get('reference_path') or 'None'}",
                f"- Events: {len(self.current_report.get('debug_events', []))}",
                "",
                "## Processing Events",
            ]
            for event in self.current_report.get("debug_events", []):
                before = event.get("before", {})
                after = event.get("after", {})
                params = event.get("params", {})
                before_summary = (
                    f"LUFS={before.get('lufs')} peak={before.get('peak_db', 0):+.1f}dB "
                    f"crest={before.get('crest_db', 0):.1f}dB"
                ) if before else "n/a"
                after_summary = (
                    f"LUFS={after.get('lufs')} peak={after.get('peak_db', 0):+.1f}dB "
                    f"crest={after.get('crest_db', 0):.1f}dB"
                ) if after else "n/a"
                debug_lines.append(f"- [{event.get('stage')}] {event.get('subject')}")
                debug_lines.append(f"  - before: {before_summary}")
                debug_lines.append(f"  - after: {after_summary}")
                if params:
                    param_text = ", ".join(f"{k}={v}" for k, v in params.items())
                    debug_lines.append(f"  - params: {param_text}")
                if event.get("notes"):
                    debug_lines.append(f"  - notes: {'; '.join(str(n) for n in event.get('notes', []))}")
            with open(debug_md_path, "w") as f:
                f.write("\n".join(debug_lines))
            print(f"  ✓ Mix report: {json_path}")
            print(f"  ✓ Debug report: {debug_json_path}")
        except Exception as e:
            print(f"  Warning: could not write mix report: {e}")

    def _safe_lufs(self, y: np.ndarray, sr: int):
        import pyloudnorm as pyln
        try:
            value = float(pyln.Meter(sr).integrated_loudness(self._ensure_stereo(y)))
            if np.isfinite(value) and value > -80.0:
                return value
        except Exception:
            pass
        return None

    def _active_rms_db(self, y: np.ndarray, floor_db: float = -70.0) -> float:
        y = self._ensure_stereo(np.asarray(y, dtype=np.float32))
        frame = 2048
        if len(y) < frame:
            rms = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
            return 20.0 * np.log10(rms)
        mono = self._mono(y)
        usable = len(mono) - (len(mono) % frame)
        frames = mono[:usable].reshape(-1, frame)
        rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
        active = rms[rms > 10.0 ** (floor_db / 20.0)]
        if len(active) == 0:
            active = rms
        return float(20.0 * np.log10(np.median(active) + 1e-12))

    def _peak_protect_array(self, y: np.ndarray, ceiling_db: float = -3.0):
        peak = float(np.max(np.abs(y)))
        ceiling = 10.0 ** (ceiling_db / 20.0)
        if peak > ceiling:
            gain = ceiling / max(peak, 1e-12)
            return (y * gain).astype(np.float32), 20.0 * np.log10(gain)
        return y.astype(np.float32), 0.0

    def apply_professional_eq_shaping(self, path: str, track_type: str, max_passes: int = 8) -> str:
        """
        Apply professional mixing standards iteratively to shape the sound.
        Uses in-memory NumPy/SciPy optimization to hit spectral target profiles.
        """
        y, sr = sf.read(path)
        
        # Ensure array is in the right orientation for SciPy filters (samples, channels)
        y = self._ensure_stereo(np.asarray(y, dtype=np.float32))

        # Run the in-memory frequency-aware optimizer
        y_opt = self.eq.optimize_to_target(y, sr, track_type, max_passes=max_passes)
        
        # Save optimized audio
        sf.write(path, y_opt, sr, subtype='FLOAT')
        
        # Cleanup
        del y
        del y_opt
        gc.collect()
        
        return path

    def process_full_mix(self, stems: Dict[str, str], song_name: str, bpm: float = 90.0):
        """
        Orchestrate the full mix with a consolidated high-integrity pipeline.
        Includes iterative preview/adjustment at each stage.
        """
        print(f"\n--- PRODUCING: {song_name} (BPM: {bpm}) ---")

        stem_paths = [p for p in stems.values() if p and os.path.exists(p)]
        if not stem_paths:
            print("  No stems found — skipping production.")
            return None, None
        self._start_mix_report(song_name, bpm, stem_paths)

        # 1. Pre-Analysis Pass
        print("[Step 1] Pre-Analysis (Loudness & Spectral)...")
        # Load pink noise reference once — used for frequency-band gain staging
        # Pink noise has equal energy per octave, matching human loudness perception.
        pink_y, pink_sr = self.stager.load_pink_noise()

        # Build processing map
        stem_names = [os.path.basename(p) for p in stem_paths]
        pan_map = self.compute_pan_positions(stem_names)

        # 2. Consolidated Processing Pass (Serial, 32-bit Float)
        print("[Step 2] Consolidated Processing Pass (Gain + Compress + EQ + FX)...")
        processed_dir = os.path.join(self.output_dir, "processed")
        automation_dir = os.path.join(self.output_dir, "automated")
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(automation_dir, exist_ok=True)

        final_processed_paths = []
        skip_processing = True

        # Check if we can resume from automated files
        for path in stem_paths:
            name = os.path.basename(path)
            auto_path = os.path.join(automation_dir, "proc_" + name.replace(".wav", "_io_auto.wav"))
            if not os.path.exists(auto_path):
                skip_processing = False
                break
            final_processed_paths.append(auto_path)

        if skip_processing and final_processed_paths:
            print("  [Resume] Found existing automated stems — skipping Step 2 and Step 3 processing.")
        else:
            final_processed_paths = []
            processed_paths = []
            for path in stem_paths:
                name = os.path.basename(path)

                # A. Load audio and check for silence
                y_raw, sr = sf.read(path, always_2d=True)
                if np.max(np.abs(y_raw)) < 1e-6:
                    print(f"  {name:48s}  [SILENT] skipping")
                    continue
                raw_analysis = self._debug_analyze_array(y_raw, sr, f"{name}:raw")

                # B. Pink noise gain staging — bring stem just audible above pink noise
                #    in its dominant frequency band (not flat LUFS; respects spectral energy)
                raw_gain_db = self._get_pink_noise_gain(y_raw, sr, name, pink_y, pink_sr)
                gain_db = self._clamp_gain_db(raw_gain_db, -12.0, 12.0)
                y_staged = y_raw * (10.0 ** (gain_db / 20.0))
                staged_analysis = self._debug_analyze_array(y_staged, sr, f"{name}:pink_staged")
                self._report_debug_event(
                    "stem_gain_staging",
                    name,
                    params={"raw_gain_db": raw_gain_db, "applied_gain_db": gain_db},
                    before=raw_analysis,
                    after=staged_analysis,
                    notes=["pink-noise band match"],
                )
                del y_raw

                # C. Adaptive compression — binary-search threshold for 2–4 dB mean GR.
                #    Makeup gain is recalculated from the adaptive threshold inside _numpy_compress.
                y_comp, mean_gr, comp_debug = self._adaptive_compress_stem(y_staged, sr, name, bpm)
                comp_analysis = self._debug_analyze_array(y_comp, sr, f"{name}:compressed")
                self._report_debug_event(
                    "stem_compression",
                    name,
                    params=comp_debug,
                    before=staged_analysis,
                    after=comp_analysis,
                )
                del y_staged

                # D. Post-compression pink noise re-check — correct residual level shift
                #    introduced by compression + makeup gain
                raw_trim_db = self._get_pink_noise_gain(y_comp, sr, name, pink_y, pink_sr)
                trim_db = self._clamp_gain_db(raw_trim_db, -4.0, 4.0)
                layer_trim_db = self._get_layer_balance_trim_db(name)
                trim_db += layer_trim_db
                y_final = y_comp * (10.0 ** (trim_db / 20.0))
                final_analysis = self._debug_analyze_array(y_final, sr, f"{name}:post_trim")
                self._report_debug_event(
                    "stem_post_compression_trim",
                    name,
                    params={
                        "raw_trim_db": raw_trim_db,
                        "clamped_trim_db": trim_db - layer_trim_db,
                        "layer_trim_db": layer_trim_db,
                        "applied_trim_db": trim_db,
                    },
                    before=comp_analysis,
                    after=final_analysis,
                )
                del y_comp

                gain_note = f" raw:{raw_gain_db:+.1f}" if abs(raw_gain_db - gain_db) > 0.05 else ""
                trim_note = f" raw:{raw_trim_db:+.1f}" if abs(raw_trim_db - (trim_db - layer_trim_db)) > 0.05 else ""
                layer_note = f" layer:{layer_trim_db:+.1f}" if abs(layer_trim_db) > 0.05 else ""
                print(
                    f"  {name:48s}  gain:{gain_db:+.1f}dB{gain_note} | "
                    f"comp={mean_gr:.1f}dB-GR | trim:{trim_db:+.1f}dB{trim_note}{layer_note}"
                )

                # Write gain-staged + compressed temp file for the ffmpeg chain
                comp_path = os.path.join(processed_dir, "comp_" + name)
                sf.write(comp_path, y_final, sr, subtype='FLOAT')
                del y_final
                gc.collect()

                # E. FFmpeg chain — harmonic saturation + creative filters only
                #    (volume and acompressor have already been applied above)
                filters, log = [], []
                hrm_filt, hrm_log = self._get_harmonic_filters(name)
                if hrm_filt:
                    filters.append(hrm_filt)
                    log.append(hrm_log)
                crt_filters, crt_log = self._get_creative_filters(name, bpm)
                filters.extend(crt_filters)
                log.extend(crt_log)

                out_path = os.path.join(processed_dir, "proc_" + name)
                if filters:
                    cmd = ["ffmpeg", "-y", "-i", comp_path, "-af", ",".join(filters),
                           "-c:a", "pcm_f32le", out_path]
                    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    if res.returncode != 0:
                        print(f"    ! FFmpeg failed: {res.stderr.decode()[:100]}")
                        shutil.copy2(comp_path, out_path)
                else:
                    shutil.copy2(comp_path, out_path)

                if os.path.exists(comp_path):
                    os.remove(comp_path)

                # F. Panning & Final Leveling (via NumPy to ensure precision)
                y, sr = sf.read(out_path)
                y = self.apply_panning(y, name, pan_map[name])
                post_fx_pan_analysis = self._debug_analyze_array(y, sr, f"{name}:post_fx_pan")
                sf.write(out_path, y, sr, subtype='FLOAT')
                del y
                gc.collect()

                # G. Apply Professional EQ Shaping (iterative, frequency-aware)
                if self._preserve_intentional_layer_eq(name):
                    print(f"  Preserving intentional layer EQ for {name}...")
                    self._report_debug_event(
                        "stem_objective_eq",
                        name,
                        params={"action": "preserved_intentional_layer_eq"},
                        before=post_fx_pan_analysis,
                        after=post_fx_pan_analysis,
                    )
                else:
                    print(f"  Applying professional EQ shaping to {name}...")
                    out_path = self.apply_professional_eq_shaping(out_path, name)
                    eq_y, eq_sr = sf.read(out_path, always_2d=True)
                    eq_analysis = self._debug_analyze_array(eq_y, eq_sr, f"{name}:post_objective_eq")
                    self._report_debug_event(
                        "stem_objective_eq",
                        name,
                        params={"target": name, "max_passes": 8},
                        before=post_fx_pan_analysis,
                        after=eq_analysis,
                    )

                processed_paths.append(out_path)
                gc.collect()

            # 3. Automation Pass (Serial, NumPy, 32-bit Float)
            print("[Step 3] Intro/Outro & Phrase Automations...")
            # We process each stem for intro/outro and phrase FX
            automated_paths = self.apply_intro_outro_automation(processed_paths, bpm, automation_dir)
            final_processed_paths = self.apply_phrase_automation_fx(automated_paths, bpm, automation_dir)

        sanity_dir = os.path.join(self.output_dir, "sanity")
        print("[Step 3.5] Processed Stem Sanity Check...")
        final_processed_paths = self.sanity_check_processed_stems(final_processed_paths, stem_paths, sanity_dir)

        # 4. Bus Processing & Summing
        print("[Step 4] Bus Processing & Intelligent EQ...")
        bus_paths = self.apply_bus_processing(final_processed_paths, song_name)

        # 4.5 Global FX Sends (Shared Reverb/Delay Buses)
        fx_bus_dir = os.path.join(self.output_dir, "buses")
        fx_bus_paths = self.apply_global_fx_sends(final_processed_paths, bpm, fx_bus_dir)
        bus_paths.extend(fx_bus_paths)

        # 5. Sum & Master
        print("[Step 5] Summing & Mastering...")
        mix_path = self.sum_stems(bus_paths, song_name)
        master_path = self.apply_mastering(mix_path, song_name)
        
        # Clean up main mix
        if mix_path and os.path.exists(mix_path):
            os.remove(mix_path)

        # 5.5 Create Mix Variants (Minimal, Bass1, Bass2, DnB)
        print("[Step 5.5] Creating Mix Variants...")
        
        variants = [
            {"suffix": "bass1",        "omit": ["_Harmonic_Bass"],                      "label": "Bass 1 Mix (No Harmonic Bass)"},
            {"suffix": "bass2",        "omit": ["_Bass"], "omit_not": ["_Harmonic_Bass"],"label": "Bass 2 Mix (No Bass)"},
            {"suffix": "minimal-bass1","omit": ["_Main_Melody", "_Harmonic_Bass"],       "label": "Minimal-Bass1 Mix (No Melody + No Harmonic Bass)"},
            {"suffix": "minimal-bass2","omit": ["_Main_Melody", "_Bass"], "omit_not": ["_Harmonic_Bass"], "label": "Minimal-Bass2 Mix (No Melody + No Bass)"},
            {"suffix": "dnb-mix-1",    "only": ["_Bass", "drum"], "not": ["_Harmonic_Bass"], "label": "DnB Mix 1 (Bass + Drums Only)"},
            {"suffix": "dnb-mix-2",    "only": ["_Harmonic_Bass", "drum"], "label": "DnB Mix 2 (Harmonic Bass + Drums Only)"},
        ]

        for var in variants:
            if "only" in var:
                # Include ONLY stems matching these patterns
                v_stems = [p for p in final_processed_paths
                           if any(x.lower() in os.path.basename(p).lower() for x in var["only"])]
                # Exclude stems matching 'not' patterns (e.g. keep _Bass but drop _Harmonic_Bass)
                if "not" in var:
                    v_stems = [p for p in v_stems
                               if not any(x.lower() in os.path.basename(p).lower() for x in var["not"])]
                v_fx = []
            else:
                # Omit stems matching omit patterns, but preserve stems matching omit_not patterns
                omit_list = var['omit']
                omit_not_list = var.get('omit_not', [])
                v_stems = [p for p in final_processed_paths
                           if not (
                               any(o.lower() in os.path.basename(p).lower() for o in omit_list)
                               and not any(n.lower() in os.path.basename(p).lower() for n in omit_not_list)
                           )]
                v_fx = fx_bus_paths
            
            # Combine processed stems and optional FX returns
            v_final_paths = v_stems + v_fx
            
            print(f"  Generating {var['label']}...")
            v_mix = self.sum_stems(v_final_paths, song_name, suffix=var['suffix'])
            if v_mix:
                self.apply_mastering(v_mix, song_name, suffix=var['suffix'])
                # Clean up variant mix
                if os.path.exists(v_mix):
                    os.remove(v_mix)

        self._write_mix_report(song_name)
        return master_path

    def _get_stem_lufs_target(self, name: str) -> float:
        n = name.lower()
        if any(x in n for x in ['kick','snare','hat','drum','bongo','conga','tambourine','maracas','perc','side_stick']):
            return -18.0
        elif 'bass' in n:
            return -22.0
        elif any(x in n for x in ['pad','chord']):
            return -27.0
        elif any(x in n for x in ['melody','chorus','counter']):
            return -24.0
        return -24.0

    def _get_dynamic_filters(self, name: str, bpm: float):
        name = name.lower()
        q_ms = round(60000.0 / bpm)
        if any(x in name for x in ['kick', 'snare', 'hat', 'clap', 'drum']):
            ratio = 6; threshold_db = -12
            makeup = round(10 ** ((-threshold_db * (1 - 1/ratio)) / 20), 2)
            rel = 80
            return f"acompressor=attack=5:release={rel}:ratio={ratio}:threshold=0.251:makeup={makeup}", f"drum-comp-{ratio}:1 makeup={makeup}"
        elif 'bass' in name or 'harmonic_bass' in name:
            ratio = 4; threshold_db = -16
            makeup = round(10 ** ((-threshold_db * (1 - 1/ratio)) / 20), 2)
            rel = round(q_ms * 2)
            return f"acompressor=attack=20:release={rel}:ratio={ratio}:threshold=0.159:makeup={makeup}", f"bass-comp-{ratio}:1 makeup={makeup}"
        elif any(x in name for x in ['pad', 'chord']):
            ratio = 2; threshold_db = -20
            makeup = round(10 ** ((-threshold_db * (1 - 1/ratio)) / 20), 2)
            rel = round(q_ms * 2)
            return f"acompressor=attack=30:release={rel}:ratio={ratio}:threshold=0.1:makeup={makeup}", f"pad-comp-{ratio}:1 makeup={makeup}"
        else:
            # melody/counter/chorus and default
            ratio = 3; threshold_db = -18
            makeup = round(10 ** ((-threshold_db * (1 - 1/ratio)) / 20), 2)
            return f"acompressor=attack=15:release={q_ms}:ratio={ratio}:threshold=0.126:makeup={makeup}", f"melody-comp-{ratio}:1 makeup={makeup}"

    # ------------------------------------------------------------------
    # Adaptive gain staging + compression helpers
    # ------------------------------------------------------------------

    def _get_stem_freq_band(self, name: str) -> tuple:
        """Return (low_hz, high_hz) of the stem's dominant frequency range."""
        n = name.lower()
        if 'kick' in n:                                    return (40,   4000)
        if any(x in n for x in ['snare', 'hat', 'clap',
                                  'drum', 'perc', 'bongo',
                                  'conga', 'tamb', 'marac']): return (150, 10000)
        if 'bass' in n:                                    return (40,    300)
        if any(x in n for x in ['pad', 'chord']):          return (200,  5000)
        if any(x in n for x in ['melody', 'counter',
                                  'chorus', 'lead']):       return (500,  8000)
        return (20, 20000)

    def _bandpass_rms(self, y: np.ndarray, sr: int, low: float, high: float) -> float:
        """RMS energy of y filtered to [low, high] Hz via scipy butterworth."""
        from scipy.signal import butter, sosfilt
        nyq = sr / 2.0
        lo = max(low / nyq, 1e-4)
        hi = min(high / nyq, 0.9999)
        sos = butter(4, [lo, hi], btype='band', output='sos')
        mono = y[:, 0] if y.ndim == 2 else y
        filtered = sosfilt(sos, mono.astype(np.float64))
        rms = np.sqrt(np.mean(filtered ** 2))
        return float(rms) if rms > 0 else 1e-10

    def _get_pink_noise_gain(self, y: np.ndarray, sr: int, name: str,
                              pink_y: np.ndarray, pink_sr: int) -> float:
        """Gain (dB) to bring the stem just audible above pink noise in its frequency band.
        Offsets represent 'just barely audible above the noise floor' per stem type."""
        OFFSETS = {
            'kick':    4.0,
            'drum':    4.0,
            'bass':    3.0,
            'pad':     1.0,
            'chord':   1.0,
            'melody':  3.0,
            'counter': 3.0,
            'chorus':  3.0,
            'lead':    3.0,
        }
        n = name.lower()
        offset = next((v for k, v in OFFSETS.items() if k in n), 2.0)
        low, high = self._get_stem_freq_band(name)

        # Resample pink noise to stem sr if needed
        if pink_sr != sr:
            import librosa
            pink_mono = pink_y[:, 0] if pink_y.ndim == 2 else pink_y
            pink_mono = librosa.resample(pink_mono.astype(np.float32), orig_sr=pink_sr, target_sr=sr)
            pink_y = pink_mono[:, np.newaxis]

        stem_rms = self._bandpass_rms(y, sr, low, high)
        pink_rms = self._bandpass_rms(pink_y, sr, low, high)
        gain_db  = 20.0 * np.log10(pink_rms / stem_rms) + offset
        return float(gain_db)

    def _clamp_gain_db(self, gain_db: float, min_db: float, max_db: float) -> float:
        """Keep pink-noise matching from making extreme corrective moves."""
        return float(np.clip(gain_db, min_db, max_db))

    def _get_layer_balance_trim_db(self, name: str) -> float:
        """Musical trims for stacked layers that should not all land equally loud."""
        n = name.lower()
        if 'kick' in n:
            if '_sub' in n:
                return -4.0
            if '_click' in n:
                return -5.0
            if '_punch' in n:
                return 0.0
        if 'snare' in n:
            if '_body' in n:
                return 0.0
            if '_snap' in n:
                return -3.0
            if '_air' in n:
                return -6.0
        if any(x in n for x in ['pad', 'chord']):
            if 'layer1' in n:
                return -2.0
            if 'layer2' in n:
                return -4.0
            if 'layer3' in n:
                return -5.0
        return 0.0

    def _preserve_intentional_layer_eq(self, name: str) -> bool:
        """Return True for layers whose separation is created by explicit EQ roles."""
        n = name.lower()
        if 'kick' in n and any(x in n for x in ['_sub', '_punch', '_click']):
            return True
        if 'snare' in n and any(x in n for x in ['_body', '_snap', '_air']):
            return True
        return False

    def _get_stem_comp_params(self, name: str, bpm: float) -> dict:
        """Fixed musical compression parameters per stem type.
        Ratio / attack / release stay constant; only threshold is found adaptively."""
        n = name.lower()
        q_ms = round(60000.0 / bpm)
        if any(x in n for x in ['kick', 'snare', 'hat', 'clap', 'drum']):
            return {'ratio': 6.0, 'attack_ms': 25.0, 'release_ms': 80.0}
        elif 'bass' in n:
            return {'ratio': 4.0, 'attack_ms': 20.0, 'release_ms': float(min(q_ms * 2, 400))}
        elif any(x in n for x in ['pad', 'chord']):
            return {'ratio': 2.0, 'attack_ms': 30.0, 'release_ms': float(min(q_ms * 2, 600))}
        else:
            return {'ratio': 3.0, 'attack_ms': 15.0, 'release_ms': float(min(q_ms, 300))}

    def _measure_gr_fast(self, y: np.ndarray, sr: int, threshold: float,
                         ratio: float, attack_ms: float, release_ms: float) -> float:
        """Estimate mean active gain reduction (dB, positive) on a 4× decimated envelope.
        Uses scipy lfilter for speed — accurate enough for threshold searching."""
        from scipy.signal import lfilter
        decimate = 4
        sr_d = max(sr // decimate, 1)
        env = np.abs(y if y.ndim == 1 else np.max(np.abs(y), axis=1))
        env_d = env[::decimate].astype(np.float64)
        # Use release coefficient (slower) for a conservative estimate with no branching
        rc = np.exp(-1.0 / (sr_d * max(release_ms, 1.0) / 1000.0))
        smoothed = lfilter([1.0 - rc], [1.0, -rc], env_d)
        threshold_db = 20.0 * np.log10(max(threshold, 1e-10))
        smoothed_db  = 20.0 * np.log10(np.maximum(smoothed, 1e-10))
        knee_width   = 6.0
        gain_db      = np.zeros_like(smoothed_db)
        above = smoothed_db > (threshold_db + knee_width / 2)
        gain_db[above] = (threshold_db + (smoothed_db[above] - threshold_db) / ratio
                          - smoothed_db[above])
        active = gain_db < -0.1
        return float(-np.mean(gain_db[active])) if active.any() else 0.0

    def _find_adaptive_threshold(self, y: np.ndarray, sr: int,
                                  ratio: float, attack_ms: float, release_ms: float,
                                  target_min: float = 2.0, target_max: float = 4.0) -> tuple:
        """Binary-search for the linear threshold that yields target_min–target_max dB
        mean active gain reduction. Returns (threshold_linear, gr_achieved_db)."""
        lo, hi = 0.02, 0.98
        target_mid = (target_min + target_max) / 2.0
        best = {'thr': (lo + hi) / 2.0,
                'gr':  self._measure_gr_fast(y, sr, (lo + hi) / 2.0, ratio, attack_ms, release_ms)}
        for _ in range(10):
            mid = (lo + hi) / 2.0
            gr  = self._measure_gr_fast(y, sr, mid, ratio, attack_ms, release_ms)
            if target_min <= gr <= target_max:
                return mid, gr
            if abs(gr - target_mid) < abs(best['gr'] - target_mid):
                best = {'thr': mid, 'gr': gr}
            if gr < target_min:
                hi = mid   # too little compression → lower threshold
            else:
                lo = mid   # too much → raise threshold
        return best['thr'], best['gr']

    def _adaptive_compress_stem(self, y: np.ndarray, sr: int,
                                 name: str, bpm: float) -> tuple:
        """Compress with a threshold tuned to achieve 2–4 dB mean active GR.
        Returns (y_compressed, gr_db). Makeup gain is calculated from the adaptive threshold."""
        p = self._get_stem_comp_params(name, bpm)
        threshold, gr = self._find_adaptive_threshold(
            y, sr, p['ratio'], p['attack_ms'], p['release_ms']
        )
        thr_db = 20.0 * np.log10(max(threshold, 1e-10))
        print(f"    Adaptive comp: thr={thr_db:.1f}dB  {p['ratio']:.0f}:1  "
              f"att={p['attack_ms']:.0f}ms  rel={p['release_ms']:.0f}ms  GR={gr:.1f}dB")
        y_out = self._numpy_compress(y, threshold=threshold, ratio=p['ratio'],
                                     attack_ms=p['attack_ms'], release_ms=p['release_ms'], sr=sr)
        debug = {
            "threshold_linear": threshold,
            "threshold_db": thr_db,
            "ratio": p["ratio"],
            "attack_ms": p["attack_ms"],
            "release_ms": p["release_ms"],
            "mean_gain_reduction_db": gr,
            "target_gain_reduction_db": "2.0-4.0",
        }
        return y_out, gr, debug

    def _get_harmonic_filters(self, name: str):
        name = name.lower()
        if any(x in name for x in ['kick', 'snare', 'hat', 'clap', 'drum']):
            # Removed bitcrush, replaced with ultra-subtle analog warmth
            return "aeval='tanh(val(0)*1.05)/tanh(1.05)|tanh(val(1)*1.05)/tanh(1.05)'", "drum-warmth-1.05"
        elif 'bass' in name:
            # Subtler bass saturation (was 2.5x, now 1.1x)
            return "aeval='tanh(val(0)*1.1)/tanh(1.1)|tanh(val(1)*1.1)/tanh(1.1)'", "bass-warmth-1.1"
        elif any(x in name for x in ['pad', 'chord']):
            # Subtle tape saturation (was 1.8x, now 1.05x)
            return "aeval='tanh(val(0)*1.05)/tanh(1.05)|tanh(val(1)*1.05)/tanh(1.05)'", "pad-warmth-1.05"
        return None, None

    def _get_spatial_filters(self, name: str, bpm: float):
        name = name.lower()
        q_ms = round(60000.0 / bpm)
        e_ms = q_ms // 2
        if any(x in name for x in ['melody', 'lead', 'counter', 'chorus']):
            dms = random.choice([e_ms, q_ms])
            return f"aecho=0.9:0.3:{dms}:0.25", f"delay-{dms}ms"
        return None, None

    def _get_creative_filters(self, name: str, bpm: float):
        """
        Extract the randomized creative filter logic.
        Now completely stripped of track-level Reverb and Delay (moved to Global FX Sends).
        Returns: (filters: list, log: list)
        """
        name = name.lower()
        filters, log = [], []

        if 'kick' in name and '_layer' in name:
            if '_sub' in name:
                filters += [
                    "highpass=f=28",
                    "lowpass=f=105",
                    "equalizer=f=58:width_type=o:width=1.1:gain=2.0",
                    "equalizer=f=240:width_type=o:width=1.3:gain=-3.0",
                ]
                log.append("kick-layer:sub")
            elif '_punch' in name:
                filters += [
                    "highpass=f=35",
                    "lowpass=f=4200",
                    "equalizer=f=85:width_type=o:width=1.0:gain=2.0",
                    "equalizer=f=280:width_type=o:width=1.3:gain=-3.0",
                    "equalizer=f=2200:width_type=o:width=1.2:gain=1.5",
                ]
                log.append("kick-layer:punch")
            elif '_click' in name:
                filters += [
                    "highpass=f=850",
                    "lowpass=f=9000",
                    "equalizer=f=3500:width_type=o:width=1.1:gain=3.0",
                    "equalizer=f=550:width_type=o:width=1.4:gain=-2.5",
                    "volume=-2dB",
                ]
                log.append("kick-layer:click")
        elif 'snare' in name and '_layer' in name:
            if '_body' in name:
                filters += [
                    "lowpass=f=5200",
                    "equalizer=f=220:width_type=o:width=1.2:gain=2.0",
                    "equalizer=f=3500:width_type=o:width=1.5:gain=-2.0",
                ]
                log.append("snare-layer:body")
            elif '_snap' in name:
                filters += [
                    "highpass=f=180",
                    "equalizer=f=2200:width_type=o:width=1.1:gain=2.5",
                    "equalizer=f=450:width_type=o:width=1.2:gain=-2.0",
                ]
                log.append("snare-layer:snap")
            elif '_air' in name:
                filters += [
                    "highpass=f=4500",
                    "lowpass=f=14000",
                    "volume=-4dB",
                ]
                log.append("snare-layer:air")
        elif self._is_drum_name(name):
            pass # Removed all track-level drum reverb
        elif 'bass' in name:
            sub_g = round(random.uniform(1.0, 2.5), 1) # Lowered
            filters += [f"equalizer=f=55:width_type=o:width=2:gain={sub_g}", "highpass=f=20"]
            log.append(f"sub+{sub_g}dB")
        elif any(x in name for x in ['pad','chord']):
            if random.random() < 0.4:
                tf = round(bpm / 60.0 * random.choice([0.5, 1]), 3)
                filters.append(f"tremolo=f={tf}:d=0.3") # Lowered depth
                log.append(f"tremolo@{tf}Hz")
        elif 'counter' in name:
            pass # Delay moved to Global FX
        elif any(x in name for x in ['melody','chorus']):
            pass # Reverb moved to Global FX

        return filters, log

    def group_stems_into_buses(self, stem_paths: list) -> Dict[str, list]:
        """Categorize stems into logical buses for group processing."""
        buses = {
            "drums": [],
            "bass": [],
            "melody": [],
            "fx": []
        }
        for path in stem_paths:
            name = os.path.basename(path).lower()
            if any(x in name for x in ['kick', 'snare', 'hat', 'clap', 'drum', 'bongo', 'conga', 'tambourine', 'maracas', 'perc', 'instr', 'side_stick']):
                buses["drums"].append(path)
            elif 'bass' in name:
                buses["bass"].append(path)
            elif any(x in name for x in ['melody', 'chorus', 'counter', 'pad', 'chord']):
                buses["melody"].append(path)
            else:
                buses["fx"].append(path)
        return buses

    def _skip_bus_stage_stem_optimization(self, name: str) -> bool:
        """Avoid re-EQing already produced musical stems during bus assembly."""
        n = name.lower()
        if self._preserve_intentional_layer_eq(n):
            return False
        skip_tokens = [
            'bass', 'harmonic_bass', 'melody', 'lead', 'counter', 'chorus',
            'pad', 'chord', 'bell', 'brass', 'strings', 'poly', 'fx'
        ]
        return any(token in n for token in skip_tokens)

    def _allow_bus_stage_light_optimization(self, name: str) -> bool:
        n = name.lower()
        return any(token in n for token in [
            'hat', 'tambourine', 'maracas', 'clap', 'bongo', 'conga', 'perc', 'side_stick'
        ])

    def _maybe_gentle_carve(self, target_y, ref_y, sr, freq_range, depth_db, label):
        """Apply a small static carve only when both signals show actual overlap."""
        target_score = SpectralAnalyzer.detect_buildup(target_y, sr, freq_range)
        ref_score = SpectralAnalyzer.detect_buildup(ref_y, sr, freq_range)
        target_score = float(target_score)
        ref_score = float(ref_score)
        if target_score > 2.5 and ref_score > 2.5:
            low, high = freq_range
            center = (low + high) / 2.0
            self.eq.sr = sr
            self._report_decision(
                "unmask",
                f"{label}: gentle {depth_db:+.1f}dB carve at {center:.0f}Hz",
                {"target_score": target_score, "reference_score": ref_score, "range": freq_range},
            )
            print(f"  [Unmask] {label}: {depth_db:+.1f}dB @ {center:.0f}Hz")
            return self.eq.peaking_filter(target_y, center, depth_db, Q=1.1)
        self._report_decision(
            "unmask",
            f"{label}: skipped, masking below threshold",
            {"target_score": target_score, "reference_score": ref_score, "range": freq_range},
        )
        return target_y

    def _maybe_gentle_kick_bass_slotting(self, bass_y, drum_y, sr):
        """Narrow, conditional kick/bass slotting from measured low-end overlap."""
        kick_freq = SpectralAnalyzer.get_fundamental_frequency(drum_y, sr)
        kick_freq = float(kick_freq)
        if not np.isfinite(kick_freq) or kick_freq < 35.0 or kick_freq > 120.0:
            self._report_decision("low_end", "kick/bass slotting skipped: unreliable kick fundamental")
            return bass_y
        band = (max(35.0, kick_freq * 0.82), min(140.0, kick_freq * 1.18))
        bass_score = SpectralAnalyzer.detect_buildup(bass_y, sr, band)
        drum_score = SpectralAnalyzer.detect_buildup(drum_y, sr, band)
        bass_score = float(bass_score)
        drum_score = float(drum_score)
        if bass_score <= 2.0 or drum_score <= 2.0:
            self._report_decision(
                "low_end",
                "kick/bass slotting skipped: low overlap",
                {"kick_freq": kick_freq, "bass_score": bass_score, "drum_score": drum_score},
            )
            return bass_y
        self.eq.sr = sr
        self._report_decision(
            "low_end",
            f"kick/bass slotting: -1.5dB at {kick_freq:.1f}Hz",
            {"kick_freq": kick_freq, "bass_score": bass_score, "drum_score": drum_score},
        )
        print(f"  [Low End] Kick fundamental {kick_freq:.1f}Hz; gentle bass slot -1.5dB")
        return self.eq.peaking_filter(bass_y, kick_freq, -1.5, Q=5.0)

    def _maybe_gain_match(self, before, after, max_gain_db: float = 1.0):
        """Compensate processing loudness drift without hiding bad decisions."""
        before_rms = float(np.sqrt(np.mean(before ** 2)) + 1e-12)
        after_rms = float(np.sqrt(np.mean(after ** 2)) + 1e-12)
        gain_db = 20.0 * np.log10(before_rms / after_rms)
        gain_db = float(np.clip(gain_db, -max_gain_db, max_gain_db))
        return (after * (10.0 ** (gain_db / 20.0))).astype(np.float32), gain_db

    def _bus_target_lufs(self, bus_name: str) -> float:
        targets = {
            "drums": -25.0,
            "bass": -27.0,
            "melody": -23.0,
            "fx": -34.0,
        }
        return targets.get(bus_name, -30.0)

    def _balance_bus_contributions(self, bus_sums: dict):
        """Keep bus loudness relationships musical before FX/mastering."""
        for bus_name, path in bus_sums.items():
            if not os.path.exists(path):
                continue
            y, sr = sf.read(path, always_2d=True)
            y = self._ensure_stereo(np.asarray(y, dtype=np.float32))
            before_analysis = self._debug_analyze_array(y, sr, f"{bus_name}:pre_balance")
            current_lufs = self._safe_lufs(y, sr)
            if current_lufs is None:
                continue
            target = self._bus_target_lufs(bus_name)
            raw_gain = target - current_lufs
            max_up = 9.0 if bus_name in ["drums", "bass"] else 4.0
            max_down = -6.0 if bus_name == "melody" else -4.0
            gain_db = float(np.clip(raw_gain, max_down, max_up))
            if abs(gain_db) < 0.5:
                continue
            y = y * (10.0 ** (gain_db / 20.0))
            y, protect_db = self._peak_protect_array(y, ceiling_db=-4.0)
            after_analysis = self._debug_analyze_array(y, sr, f"{bus_name}:post_balance")
            sf.write(path, y, sr, subtype="FLOAT")
            self._bump_report_counter("bus_contribution_balanced")
            self._report_decision(
                "bus_balance",
                f"{bus_name}: {gain_db:+.1f}dB toward {target:.1f}LUFS",
                {"current_lufs": current_lufs, "target_lufs": target, "raw_gain_db": raw_gain, "peak_protect_db": protect_db},
            )
            self._report_debug_event(
                "bus_balance",
                bus_name,
                params={
                    "target_lufs": target,
                    "raw_gain_db": raw_gain,
                    "applied_gain_db": gain_db,
                    "peak_protect_db": protect_db,
                },
                before=before_analysis,
                after=after_analysis,
            )
            print(f"  [Bus Balance] {bus_name}: {current_lufs:.1f} → target {target:.1f} LUFS ({gain_db:+.1f}dB)")

    def _find_source_for_processed_stem(self, processed_path: str, source_paths: list):
        stem_name = os.path.basename(processed_path).lower()
        stem_name = stem_name.replace("proc_", "")
        stem_name = stem_name.replace("_io_auto.wav", ".wav").replace("_io.wav", ".wav")
        stem_name = stem_name.replace("_auto.wav", ".wav")
        for src in source_paths:
            if os.path.basename(src).lower() == stem_name:
                return src
        key_parts = [p for p in stem_name.replace(".wav", "").split("_") if p]
        best = None
        best_score = 0
        for src in source_paths:
            src_name = os.path.basename(src).lower()
            score = sum(1 for p in key_parts if p in src_name)
            if score > best_score:
                best = src
                best_score = score
        return best if best_score >= 4 else None

    def sanity_check_processed_stems(self, processed_paths: list, source_paths: list, output_dir: str) -> list:
        """
        Repair pathological processed-stem levels before bus summing.
        Uses source stems as a reference, but keeps corrections conservative.
        """
        os.makedirs(output_dir, exist_ok=True)
        corrected = []
        for path in processed_paths:
            if not path or not os.path.exists(path):
                continue
            name = os.path.basename(path).lower()
            out_path = os.path.join(output_dir, os.path.basename(path))
            try:
                y, sr = sf.read(path, always_2d=True)
                y = self._ensure_stereo(np.asarray(y, dtype=np.float32))
            except Exception:
                shutil.copy2(path, out_path)
                corrected.append(out_path)
                continue

            src_path = self._find_source_for_processed_stem(path, source_paths)
            active_db = self._active_rms_db(y)
            proc_lufs = self._safe_lufs(y, sr)
            before_analysis = self._debug_analyze_array(y, sr, f"{os.path.basename(path)}:pre_sanity")
            gain_db = 0.0
            reason = []

            if src_path and os.path.exists(src_path):
                src_y, src_sr = sf.read(src_path, always_2d=True)
                src_y = self._ensure_stereo(np.asarray(src_y, dtype=np.float32))
                src_active = self._active_rms_db(src_y)
                active_loss = src_active - active_db
                if active_loss > 8.0:
                    gain_db += min(active_loss - 6.0, 8.0)
                    reason.append(f"restored active level loss {active_loss:.1f}dB")

            target_lufs = self._get_stem_lufs_target(name)
            if proc_lufs is not None and proc_lufs < target_lufs - 16.0:
                gain_db += min((target_lufs - 14.0) - proc_lufs, 6.0)
                reason.append(f"raised very quiet stem {proc_lufs:.1f}LUFS")

            if any(x in name for x in ["tambourine", "maracas", "hat", "clap", "perc"]):
                gain_db = min(gain_db, 3.0)
            else:
                gain_db = min(gain_db, 8.0)

            if gain_db > 0.05:
                y = y * (10.0 ** (gain_db / 20.0))
                self._bump_report_counter("processed_stem_level_restored")
                self._report_decision("stem_sanity", f"{os.path.basename(path)} +{gain_db:.1f}dB", {"reason": reason})
                print(f"  [Stem Sanity] {os.path.basename(path)} +{gain_db:.1f}dB ({'; '.join(reason)})")

            ceiling = -6.0 if any(x in name for x in ["tambourine", "maracas", "hat", "clap", "perc"]) else -3.0
            y, protect_db = self._peak_protect_array(y, ceiling_db=ceiling)
            after_analysis = self._debug_analyze_array(y, sr, f"{os.path.basename(path)}:post_sanity")
            if protect_db < -0.05:
                self._bump_report_counter("processed_stem_peak_protected")
                self._report_decision("stem_sanity", f"{os.path.basename(path)} peak protected {protect_db:.1f}dB", {"ceiling_db": ceiling})
                print(f"  [Stem Sanity] {os.path.basename(path)} peak {protect_db:.1f}dB to {ceiling:.1f}dBFS")

            if gain_db > 0.05 or protect_db < -0.05:
                self._report_debug_event(
                    "stem_sanity",
                    os.path.basename(path),
                    params={
                        "source_path": src_path,
                        "source_active_rms_db": src_active if src_path and os.path.exists(src_path) else None,
                        "processed_active_rms_db": active_db,
                        "processed_lufs": proc_lufs,
                        "applied_gain_db": gain_db,
                        "peak_protect_db": protect_db,
                        "ceiling_db": ceiling,
                    },
                    before=before_analysis,
                    after=after_analysis,
                    notes=reason,
                )

            sf.write(out_path, y, sr, subtype="FLOAT")
            corrected.append(out_path)
        return corrected

    def apply_bus_processing(self, stem_paths: list, song_name: str) -> list:
        """
        1. Apply stem-specific EQ optimizations.
        2. Sum stems into buses.
        3. Apply intelligent EQ carving between buses.
        4. Apply professional bus EQ shaping iteratively.
        """
        bus_groups = self.group_stems_into_buses(stem_paths)
        bus_dir = os.path.join(self.output_dir, "buses")
        os.makedirs(bus_dir, exist_ok=True)

        # 1. Stem Optimization
        optimized_paths = []
        for path in stem_paths:
            name = os.path.basename(path).lower()
            opt_path = os.path.join(bus_dir, "opt_" + os.path.basename(path))
            if self._preserve_intentional_layer_eq(name):
                print(f"  Preserving layer EQ through stem optimization: {name}")
                self._bump_report_counter("layer_eq_preserved")
                shutil.copy2(path, opt_path)
                optimized_paths.append(opt_path)
                continue
            if self._skip_bus_stage_stem_optimization(name):
                print(f"  Skipping bus-stage stem EQ: {name}")
                self._bump_report_counter("bus_stage_stem_eq_skipped")
                shutil.copy2(path, opt_path)
                optimized_paths.append(opt_path)
                continue
            if not self._allow_bus_stage_light_optimization(name):
                print(f"  Copying stem without bus-stage EQ: {name}")
                self._bump_report_counter("bus_stage_stem_eq_copied")
                shutil.copy2(path, opt_path)
                optimized_paths.append(opt_path)
                continue
            info = sf.info(path)
            if info.frames < 2048:
                print(f"  Skipping short stem: {name} ({info.frames} frames)")
                shutil.copy2(path, opt_path)
                optimized_paths.append(opt_path)
                continue
            y, sr = sf.read(path)
            y_before = self._ensure_stereo(np.asarray(y, dtype=np.float32))
            self.eq.sr = sr
            # Use light optimization only for auxiliary percussion.
            y = self.eq.optimize_to_target(y_before, sr, name, max_passes=2)
            y, gain_match_db = self._maybe_gain_match(y_before, self._ensure_stereo(y), max_gain_db=0.75)
            if abs(gain_match_db) > 0.05:
                self._report_decision("stem_eq", f"{name}: gain matched objective EQ {gain_match_db:+.2f}dB")
            self._bump_report_counter("bus_stage_stem_eq_applied")

            sf.write(opt_path, y, sr, subtype='FLOAT')
            optimized_paths.append(opt_path)

        # Re-group optimized stems
        bus_groups = self.group_stems_into_buses(optimized_paths)

        # 2. Sum into Buses (true additive mix — stems are gain staged so no averaging)
        bus_sums = {}
        for bus_name, paths in bus_groups.items():
            if not paths: continue
            out_path = os.path.join(bus_dir, f"bus_{bus_name}.wav")
            mix, sr = sf.read(paths[0], always_2d=True)
            mix = mix.astype(np.float64)
            for p in paths[1:]:
                y, _ = sf.read(p, always_2d=True)
                y = y.astype(np.float64)
                mix = self._pad_and_add(mix, y)
            peak = np.max(np.abs(mix))
            if peak > self.bus_headroom_peak:
                mix *= self.bus_headroom_peak / peak
            sf.write(out_path, mix.astype(np.float32), sr, subtype='FLOAT')
            bus_sums[bus_name] = out_path
            if self.current_report is not None:
                self.current_report["buses"][bus_name] = self._analyze_array(mix.astype(np.float32), sr, bus_name)

        self._balance_bus_contributions(bus_sums)

        # Parallel drum compression (NY style) only when the drum bus needs density.
        drum_transient_shape_applied = False
        if "drums" in bus_sums:
            d_y, d_sr = sf.read(bus_sums["drums"], always_2d=True)
            d_metrics = self._analyze_array(d_y, d_sr, "drums_pre_parallel")
            if d_metrics["crest_db"] > 10.5:
                d_y, parallel_debug = self._build_parallel_drum_bus(d_y, d_sr, blend=0.20)
                d_after = self._debug_analyze_array(d_y, d_sr, "drums_post_parallel")
                sf.write(bus_sums["drums"], d_y, d_sr, subtype='FLOAT')
                self._report_decision("drums", "NY parallel compression applied at 20%", d_metrics)
                self._report_debug_event(
                    "bus_parallel_compression",
                    "drums",
                    params=parallel_debug,
                    before=d_metrics,
                    after=d_after,
                )
                self._bump_report_counter("drum_parallel_applied")
                print("  [Parallel Drums] NY-style compression applied (20% blend)")
            else:
                self._report_decision("drums", "NY parallel compression skipped: density already controlled", d_metrics)
                self._report_debug_event(
                    "bus_parallel_compression",
                    "drums",
                    params={"action": "skipped", "reason": "crest already controlled", "crest_threshold_db": 10.5},
                    before=d_metrics,
                    after=d_metrics,
                )
                self._bump_report_counter("drum_parallel_skipped")
                print("  [Parallel Drums] skipped (density already controlled)")

        # Sub-bass harmonic enhancement (small-speaker translation)
        for sub_bus in ["drums", "bass"]:
            if sub_bus in bus_sums:
                s_y, s_sr = sf.read(bus_sums[sub_bus])
                s_y = self._apply_sub_harmonic_enhancement(s_y, s_sr)
                sf.write(bus_sums[sub_bus], s_y, s_sr, subtype='FLOAT')
        print("  [Sub Enhancement] 2nd harmonic generated for small-speaker translation")

        # 3. Intelligent Carving & Monitoring
        print("  [Monitoring] Analyzing Bus Spectral Health:")
        for bus_name, path in bus_sums.items():
            y, sr = sf.read(path)
            # Low (20-150), Mid (150-2.5k), High (2.5k-20k)
            lows = SpectralAnalyzer.detect_buildup(y, sr, (20, 150))
            mids = SpectralAnalyzer.detect_buildup(y, sr, (150, 2500))
            his  = SpectralAnalyzer.detect_buildup(y, sr, (2500, 20000))
            print(f"    - {bus_name:8s}: Lows={lows:.2f}, Mids={mids:.2f}, Highs={his:.2f}")

        # Melody vs Drums (gentle measured unmasking only)
        if "melody" in bus_sums and "drums" in bus_sums:
            m_y, sr = sf.read(bus_sums["melody"])
            d_y, _ = sf.read(bus_sums["drums"])
            m_y = self._maybe_gentle_carve(m_y, d_y, sr, (2200, 3200), -0.75, "melody behind snare presence")
            m_y = self._maybe_gentle_carve(m_y, d_y, sr, (250, 380), -0.75, "melody low-mid drum pocket")
            sf.write(bus_sums["melody"], m_y, sr, subtype='FLOAT')

        # Melody vs Bass (protect warmth; carve only the lower low-mid edge)
        if "melody" in bus_sums and "bass" in bus_sums:
            m_y, sr = sf.read(bus_sums["melody"])
            b_y, _ = sf.read(bus_sums["bass"])
            m_y = self._maybe_gentle_carve(m_y, b_y, sr, (120, 240), -0.75, "melody low edge behind bass")
            sf.write(bus_sums["melody"], m_y, sr, subtype='FLOAT')

        # Bass vs Drums (Kick/Bass Slotting)
        if "bass" in bus_sums and "drums" in bus_sums:
            b_y, sr = sf.read(bus_sums["bass"])
            d_y, _ = sf.read(bus_sums["drums"])
            b_y = self._maybe_gentle_kick_bass_slotting(b_y, d_y, sr)
            sf.write(bus_sums["bass"], b_y, sr, subtype='FLOAT')

        # Subtle sidechain: kick ducks bass gently
        if "drums" in bus_sums and "bass" in bus_sums:
            b_y, b_sr = sf.read(bus_sums["bass"], always_2d=True)
            d_y, _    = sf.read(bus_sums["drums"], always_2d=True)
            if len(d_y) < len(b_y):
                d_y = np.pad(d_y, ((0, len(b_y) - len(d_y)), (0,0)))
            b_y = self._apply_sidechain_ducking(b_y, d_y, b_sr, duck_depth=0.06)
            sf.write(bus_sums["bass"], b_y, b_sr, subtype='FLOAT')
            print("  [Sidechain] Kick→Bass ducking: -0.5dB peak, 80ms release")

        # Phase alignment check: try polarity flip on bass, keep whichever sums better with kick
        if "drums" in bus_sums and "bass" in bus_sums:
            b_y, b_sr = sf.read(bus_sums["bass"], always_2d=True)
            d_y, _    = sf.read(bus_sums["drums"], always_2d=True)
            min_len = min(len(b_y), len(d_y))
            rms_normal  = np.sqrt(np.mean((b_y[:min_len] + d_y[:min_len])**2))
            rms_flipped = np.sqrt(np.mean((-b_y[:min_len] + d_y[:min_len])**2))
            if rms_flipped > rms_normal:
                b_y = -b_y
                sf.write(bus_sums["bass"], b_y, b_sr, subtype='FLOAT')
                print("  [Phase] Bass polarity flipped for better kick alignment")
            else:
                print("  [Phase] Bass polarity OK (no flip needed)")

        # 4. Apply Professional Bus EQ Shaping (iterative, frequency-aware)
        print("  [Professional Bus EQ Shaping]...")
        for bus_name, path in bus_sums.items():
            if bus_name == "drums":
                print("    Preserving drums bus EQ to protect kick/snare layer roles.")
                self._bump_report_counter("bus_eq_skipped_drums")
                continue
            if bus_name == "melody":
                print("    Skipping melody bus objective EQ after measured unmasking.")
                self._report_decision("bus_eq", "melody bus objective EQ skipped after measured unmasking")
                self._bump_report_counter("bus_eq_skipped_melody")
                continue
            print(f"    Optimizing {bus_name} bus spectral profile...")
            y, sr = sf.read(path)
            y_before = self._ensure_stereo(np.asarray(y, dtype=np.float32))
            self.eq.sr = sr
            
            # Use the in-memory optimizer
            y_opt = self.eq.optimize_to_target(y_before, sr, bus_name, max_passes=3)
            y_opt, gain_match_db = self._maybe_gain_match(y_before, self._ensure_stereo(y_opt), max_gain_db=0.75)
            if abs(gain_match_db) > 0.05:
                self._report_decision("bus_eq", f"{bus_name}: gain matched bus EQ {gain_match_db:+.2f}dB")
            self._bump_report_counter("bus_eq_applied")
            
            sf.write(path, y_opt, sr, subtype='FLOAT')
            del y_before
            del y_opt
            gc.collect()

        # Transient shaping on drums bus
        if "drums" in bus_sums:
            d_y, d_sr = sf.read(bus_sums["drums"])
            d_metrics = self._analyze_array(d_y, d_sr, "drums_pre_transient")
            if d_metrics["crest_db"] < 8.0:
                d_y = self.eq.apply_transient_shaping(d_y, d_sr, attack_gain_db=1.25, sustain_gain_db=-0.75)
                sf.write(bus_sums["drums"], d_y, d_sr, subtype='FLOAT')
                drum_transient_shape_applied = True
                self._report_decision("drums", "transient shaping applied +1.25dB/-0.75dB", d_metrics)
                self._bump_report_counter("drum_transient_applied")
                print("  [Transient Shaping] Drums: +1.25dB attack, -0.75dB sustain")
            else:
                self._report_decision("drums", "transient shaping skipped: crest already healthy", d_metrics)
                self._bump_report_counter("drum_transient_skipped")
                print("  [Transient Shaping] skipped (crest already healthy)")

        # Harmonic exciter on melody bus (adds air above 8kHz)
        if "melody" in bus_sums:
            m_y, m_sr = sf.read(bus_sums["melody"])
            before = self._ensure_stereo(np.asarray(m_y, dtype=np.float32))
            m_y = self.eq.apply_exciter(before, m_sr, amount=0.08)
            m_y, gain_match_db = self._maybe_gain_match(before, self._ensure_stereo(m_y), max_gain_db=0.5)
            sf.write(bus_sums["melody"], m_y, m_sr, subtype='FLOAT')
            self._report_decision("melody", f"exciter applied 8%, gain_match={gain_match_db:+.2f}dB")
            print("  [Exciter] Melody: +8% upper harmonic blend above 8kHz")

        # M/S width expansion on melodic buses
        if "melody" in bus_sums:
            m_y, m_sr = sf.read(bus_sums["melody"])
            before = self._ensure_stereo(np.asarray(m_y, dtype=np.float32))
            m_y = self._apply_ms_width(before, m_sr, width_multiplier=1.22)
            m_y, gain_match_db = self._maybe_gain_match(before, self._ensure_stereo(m_y), max_gain_db=0.5)
            sf.write(bus_sums["melody"], m_y, m_sr, subtype='FLOAT')
            self._report_decision("melody", f"width applied x1.22, gain_match={gain_match_db:+.2f}dB")
            print("  [Width] Melody bus: ×1.22 stereo expansion (above 200Hz)")

        if self.current_report is not None:
            for bus_name, path in bus_sums.items():
                try:
                    self.current_report["buses"][bus_name] = self._analyze_file(path, bus_name)
                except Exception:
                    pass

        return list(bus_sums.values())

    def _apply_gentle_reference_master_eq(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Tiny master EQ moves only when the mix is clearly outside the reference shape."""
        ref = self._load_reference_analysis()
        if not ref:
            self._report_decision("master_eq", "reference EQ skipped: no reference analysis")
            return y
        y = self._ensure_stereo(y)
        current = self._analyze_array(y, sr, "pre_master_eq")["spectral_profile"]
        ref_profile = ref.get("spectral_profile", {})
        bands = SpectralAnalyzer.GRANULAR_BANDS
        moves = []
        for low, high in bands:
            key = f"{low}-{high}"
            c = current.get(key, 0.0)
            r = ref_profile.get(key, 0.0)
            if r <= 1e-6:
                continue
            delta = r - c
            if abs(delta) < 0.22:
                continue
            moves.append((abs(delta), low, high, float(np.clip(delta * 2.0, -0.75, 0.75))))
        if not moves:
            self._report_decision("master_eq", "reference EQ skipped: inside tolerance")
            print("  Reference master EQ: within tolerance")
            return y
        self.eq.sr = sr
        out = y.copy()
        for _, low, high, gain_db in sorted(moves, reverse=True)[:3]:
            center = (low + high) / 2.0
            if low <= 30:
                out = self.eq.low_shelf(out, high, gain_db)
            elif high >= 20000:
                out = self.eq.high_shelf(out, low, gain_db)
            else:
                out = self.eq.peaking_filter(out, center, gain_db, Q=0.8)
            self._report_decision(
                "master_eq",
                f"reference-guided {gain_db:+.2f}dB at {center:.0f}Hz",
                {"range": [low, high]},
            )
            print(f"  Reference master EQ: {gain_db:+.2f}dB @ {center:.0f}Hz")
        matched, gain_match_db = self._maybe_gain_match(y, out, max_gain_db=0.75)
        if abs(gain_match_db) > 0.05:
            self._report_decision("master_eq", f"gain matched reference EQ {gain_match_db:+.2f}dB")
        return matched

    def _apply_gentle_master_multiband(self, y: np.ndarray, sr: int):
        """Gentle, no-makeup multiband compression for transparent master control."""
        y = self._ensure_stereo(y)
        out = y.astype(np.float64).copy()
        bands = [
            {"name": "sub", "range": (30.0, 120.0), "ratio": 1.35, "attack": 35.0, "release": 180.0, "max_gr": 1.5, "crest": 9.0},
            {"name": "low_mid", "range": (120.0, 500.0), "ratio": 1.25, "attack": 25.0, "release": 160.0, "max_gr": 1.0, "crest": 8.0},
            {"name": "mid", "range": (500.0, 5000.0), "ratio": 1.18, "attack": 15.0, "release": 120.0, "max_gr": 1.0, "crest": 8.5},
            {"name": "air", "range": (5000.0, 16000.0), "ratio": 1.12, "attack": 8.0, "release": 90.0, "max_gr": 0.75, "crest": 7.5},
        ]
        report = []
        for spec in bands:
            low, high = spec["range"]
            band = self._band_isolate(y, sr, low, high).astype(np.float32)
            peak = float(np.max(np.abs(band)))
            rms = float(np.sqrt(np.mean(band ** 2)) + 1e-12)
            if peak < 1e-5:
                report.append({"band": spec["name"], "gr_db": 0.0, "status": "silent"})
                continue
            crest = 20.0 * np.log10(max(peak, 1e-12) / rms)
            if crest < spec["crest"]:
                report.append({"band": spec["name"], "gr_db": 0.0, "crest_db": crest, "status": "skipped"})
                continue
            threshold, gr = self._find_adaptive_threshold(
                band, sr, spec["ratio"], spec["attack"], spec["release"],
                target_min=0.25, target_max=spec["max_gr"]
            )
            if gr <= 0.2:
                report.append({"band": spec["name"], "gr_db": gr, "crest_db": crest, "status": "skipped"})
                continue
            compressed = self._numpy_compress(
                band, threshold=threshold, ratio=spec["ratio"],
                attack_ms=spec["attack"], release_ms=spec["release"], sr=sr,
                apply_makeup=False
            ).astype(np.float64)
            blend = min(1.0, spec["max_gr"] / max(gr, 1e-6))
            out += (compressed - band.astype(np.float64)) * blend
            report.append({"band": spec["name"], "gr_db": float(min(gr, spec["max_gr"])), "crest_db": crest, "status": "applied"})
        matched, gain_match_db = self._maybe_gain_match(y, out.astype(np.float32), max_gain_db=1.0)
        self._report_decision("master_multiband", "gentle multiband compression", {"bands": report, "gain_match_db": gain_match_db})
        print("  Master multiband:", ", ".join(f"{r['band']}={r.get('gr_db', 0):.1f}dB" for r in report))
        return matched, report

    def apply_mastering(self, input_wav, song_name, suffix: str = ""):
        """Mastering chain including Mid/Side mono-maker, Objective EQ, and PLR-targeted limiting."""
        s_part = f"_{suffix}" if suffix else ""
        output_path = os.path.join(self.output_dir, f"{song_name}{s_part}_master.wav")
        
        y, sr = sf.read(input_wav)
        y = self._ensure_stereo(np.asarray(y, dtype=np.float32))
        pre_master_analysis = self._analyze_array(y, sr, f"{song_name}{s_part}_premaster")

        # Glue compression — only a light cohesion move, not an automatic squeeze.
        if pre_master_analysis["crest_db"] > 9.0:
            print("  Applying light master bus glue compression...")
            glue_thr, glue_gr = self._find_adaptive_threshold(
                y, sr, ratio=1.5, attack_ms=10, release_ms=180,
                target_min=0.6, target_max=1.6
            )
            print(f"    Glue: thr={20*np.log10(max(glue_thr,1e-10)):.1f}dB  1.5:1  GR={glue_gr:.1f}dB")
            y_before = y.copy()
            y = self._numpy_compress(y, threshold=glue_thr, ratio=1.5, attack_ms=10, release_ms=180, sr=sr)
            y, gain_match_db = self._maybe_gain_match(y_before, y, max_gain_db=0.75)
            post_glue_analysis = self._debug_analyze_array(y, sr, f"{song_name}{s_part}_post_glue")
            self._report_decision("master_glue", f"applied light glue GR={glue_gr:.1f}dB gain_match={gain_match_db:+.2f}dB")
            self._report_debug_event(
                "master_glue",
                suffix or "main",
                params={
                    "threshold_linear": glue_thr,
                    "threshold_db": 20.0 * np.log10(max(glue_thr, 1e-10)),
                    "ratio": 1.5,
                    "attack_ms": 10,
                    "release_ms": 180,
                    "mean_gain_reduction_db": glue_gr,
                    "gain_match_db": gain_match_db,
                    "target_gain_reduction_db": "0.6-1.6",
                },
                before=pre_master_analysis,
                after=post_glue_analysis,
            )
        else:
            print("  Skipping master bus glue compression (crest already controlled).")
            self._report_decision("master_glue", "skipped: crest already controlled", pre_master_analysis)
            self._report_debug_event(
                "master_glue",
                suffix or "main",
                params={"action": "skipped", "reason": "crest already controlled", "crest_threshold_db": 9.0},
                before=pre_master_analysis,
                after=pre_master_analysis,
            )

        # 1. Mid/Side Mono-Maker (150Hz)
        print("  Focusing low-end mono...")
        mid = (y[:, 0] + y[:, 1]) * 0.5
        side = (y[:, 0] - y[:, 1]) * 0.5
        # Side HPF = side - side_lpf
        side_hpf = side - self._lowpass_numpy(side, sr, 90.0)
        y[:, 0] = mid + side_hpf
        y[:, 1] = mid - side_hpf

        # 2. Gentle multiband control, then reference-aware tonal polish.
        print("  Applying gentle master multiband compression...")
        pre_mb_analysis = self._debug_analyze_array(y, sr, f"{song_name}{s_part}_pre_multiband")
        y, mb_report = self._apply_gentle_master_multiband(y, sr)
        post_mb_analysis = self._debug_analyze_array(y, sr, f"{song_name}{s_part}_post_multiband")
        self._report_debug_event(
            "master_multiband",
            suffix or "main",
            params={"bands": mb_report},
            before=pre_mb_analysis,
            after=post_mb_analysis,
        )

        print("  Applying reference-aware master EQ...")
        pre_ref_eq_analysis = self._debug_analyze_array(y, sr, f"{song_name}{s_part}_pre_reference_eq")
        y = self._apply_gentle_reference_master_eq(y, sr)
        post_ref_eq_analysis = self._debug_analyze_array(y, sr, f"{song_name}{s_part}_post_reference_eq")
        self._report_debug_event(
            "master_reference_eq",
            suffix or "main",
            params={"reference_path": self.reference_track_path},
            before=pre_ref_eq_analysis,
            after=post_ref_eq_analysis,
        )
        
        tmp_master = input_wav + ".master_eq.wav"
        sf.write(tmp_master, y, sr, subtype='FLOAT')
        del y
        gc.collect()

        # 3. Professional Master Dynamics. Loudnorm includes true-peak limiting;
        # avoid pre-limiter auto makeup so the master keeps transient headroom.
        print("  Applying final polish & loudness normalization...")
        filter_str = "loudnorm=I=-16:TP=-1.5:LRA=11"
        
        # Read input SR so loudnorm doesn't upsample to its internal 192kHz analysis rate
        _info = sf.info(tmp_master)
        cmd = [
            "ffmpeg", "-y",
            "-i", tmp_master,
            "-af", filter_str,
            "-ar", str(_info.samplerate),
            "-c:a", "pcm_s24le",
            output_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if self.current_report is not None and os.path.exists(output_path):
            try:
                final_analysis = self._analyze_file(output_path, f"{song_name}{s_part}_master")
                self.current_report["masters"][suffix or "main"] = {
                    "pre": pre_master_analysis,
                    "multiband": mb_report,
                    "final": final_analysis,
                }
                self._report_debug_event(
                    "master_loudness_normalization",
                    suffix or "main",
                    params={"ffmpeg_filter": filter_str, "target_lufs": -16, "true_peak_db": -1.5, "lra": 11},
                    before=post_ref_eq_analysis,
                    after=final_analysis,
                )
            except Exception:
                pass

        if os.path.exists(tmp_master):
            os.remove(tmp_master)

        return output_path

    def _apply_reverb(self, src_path: str, out_path: str, reverb_filter: str, wet: float) -> bool:
        """
        Parallel reverb blend via filter_complex.
        reverb_filter uses in_gain=0 so only echo tails go into the wet path.
        wet=0.30 → 30% reverb tails + 70% dry original, mixed with amix weights.
        """
        dry = round(1.0 - wet, 2)
        fc = (
            f"[0:a]asplit=2[dry][fxin];"
            f"[fxin]{reverb_filter}[wet];"
            f"[dry][wet]amix=inputs=2:weights={dry} {wet}"
        )
        cmd = ["ffmpeg", "-y", "-i", src_path, "-filter_complex", fc, "-c:a", "pcm_f32le", out_path]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return res.returncode == 0

    def apply_creative_processing(self, stem_paths: list, bpm: float, output_dir: str) -> list:
        """
        Randomised creative per-stem processing.
        Every parameter and effect combination is re-rolled each run for unique productions.
        After all stems processed, sidechain ducking applied (70% of runs).
        """
        os.makedirs(output_dir, exist_ok=True)

        q_ms  = round(60000.0 / bpm)
        e_ms  = round(q_ms / 2)
        dq_ms = round(q_ms * 1.5)    # dotted quarter

        # Reverb type presets — in_gain=0 so only reverb tails enter the wet path
        drum_reverbs = {
            'ambience': 'aecho=0:0.9:4|8|14:0.45|0.28|0.12',
            'room':     'aecho=0:0.9:10|18|30:0.45|0.28|0.14',
            'plate':    'aecho=0:0.9:5|9|15|22|32:0.38|0.3|0.22|0.15|0.08',
            'spring':   'aecho=0:0.9:8|16|28:0.42|0.28|0.14',
        }
        melody_reverbs = {
            'small_hall': 'aecho=0:0.9:15|30|55|85:0.45|0.32|0.2|0.1',
            'chamber':    'aecho=0:0.9:12|25|45|70:0.43|0.3|0.19|0.09',
            'hall':       'aecho=0:0.9:20|45|80|130:0.45|0.33|0.22|0.11',
            'cathedral':  'aecho=0:0.9:30|80|160|280:0.43|0.32|0.2|0.1',
        }

        kick_ref = next(
            (p for p in stem_paths if 'kick' in os.path.basename(p).lower()), None
        )

        output_paths = []

        for path in stem_paths:
            name    = os.path.basename(path).lower()
            out_name = os.path.basename(path).replace("gs_hrm_", "crt_")
            if out_name == os.path.basename(path):
                base, ext = os.path.splitext(os.path.basename(path))
                out_name = base + "_crt" + ext
            out_path = os.path.join(output_dir, out_name)

            filters, log = [], []
            reverb_filter = None
            reverb_wet    = 0.28
            reverb_label  = ""

            # ── DRUMS ──────────────────────────────────────────────
            if any(x in name for x in ['kick','snare','hat','clap','drum',
                                        'bongo','conga','tambourine','maracas',
                                        'perc','instr','side_stick']):
                # Transient shaping — ALWAYS
                hi = round(random.uniform(2.0, 5.0), 1)
                filters += [f"equalizer=f=5000:width_type=o:width=2:gain={hi}",
                            "acompressor=attack=2:release=40:ratio=2:threshold=0.2:makeup=1.2"]
                log.append(f"transient +{hi}dB@5k")

                # Reverb — ALWAYS, random type
                rev_type = random.choice(list(drum_reverbs.keys()))
                reverb_filter = drum_reverbs[rev_type]
                reverb_wet    = round(random.uniform(0.28, 0.42), 2)
                reverb_label  = rev_type

                # Gated reverb on snare/clap (40%) — on top of the base reverb
                if any(x in name for x in ['snare','clap']) and random.random() < 0.4:
                    filters += ["aecho=0.8:0.4:30|60:0.35|0.18",
                                "agate=threshold=0.01:attack=5:release=80"]
                    log.append("gated-reverb")

                if random.random() < 0.3:
                    bits = random.choice([6, 7, 8, 10])
                    filters.append(f"acrusher=level_in=1:level_out=1:bits={bits}:mode=log:aa=1")
                    log.append(f"crush-{bits}bit")

                # Flanger — any drum stem
                if random.random() < 0.3:
                    spd = round(random.uniform(0.2, 1.2), 2)
                    dep = random.randint(3, 8)
                    filters.append(f"flanger=delay=5:depth={dep}:regen=20:width=90:speed={spd}")
                    log.append(f"flanger@{spd}Hz d={dep}")

                # Phaser
                if random.random() < 0.35:
                    spd = round(random.uniform(0.3, 1.5), 2)
                    filters.append(f"aphaser=in_gain=0.4:out_gain=0.74:delay=3:decay=0.4:speed={spd}:type=t")
                    log.append(f"phaser@{spd}Hz")

                # Chorus
                if random.random() < 0.25:
                    filters.append("chorus=0.7:0.9:40|45:0.3|0.25:0.4|0.35:2|1.8")
                    log.append("chorus")

                # Tremolo rhythmic gate
                if random.random() < 0.25:
                    tf = round(bpm / 60.0, 3)
                    dep = round(random.uniform(0.4, 0.8), 2)
                    filters.append(f"tremolo=f={tf}:d={dep}")
                    log.append(f"tremolo-gate@{tf}Hz")

                # Pitch shimmer on cymbals/hats
                if any(x in name for x in ['hat','cymbal','ride','crash','tambourine']) and random.random() < 0.3:
                    wd = round(random.uniform(0.01, 0.03), 3)
                    filters.append(f"vibrato=f=4:d={wd}")
                    log.append(f"shimmer-vib d={wd}")

                # Auto-pan LFO — ALWAYS on non-kick drums (apulsator)
                # Off-grid half-note or whole-note rate; unique phase per stem keeps each
                # drum element in a slightly different spatial position throughout the track.
                if 'kick' not in name:
                    base_hz  = random.choice([bpm / 120.0, bpm / 240.0])
                    drift    = random.uniform(-0.06, 0.06)
                    pan_hz   = round(max(0.08, base_hz + drift), 3)
                    amount   = round(random.uniform(0.05, 0.12), 2)
                    off_l    = round(random.uniform(0.0, 1.0), 2)
                    off_r    = round((off_l + 0.5) % 1.0, 2)
                    filters.append(
                        f"apulsator=mode=sine:hz={pan_hz}:amount={amount}"
                        f":offset_l={off_l}:offset_r={off_r}:width=1"
                    )
                    log.append(f"autopan@{pan_hz}Hz amt={amount}")

                # Contrast/punch enhancer (50%)
                if random.random() < 0.5:
                    contrast = random.randint(50, 80)
                    filters.append(f"acontrast=contrast={contrast}")
                    log.append(f"contrast={contrast}")

                # Vinyl de-emphasis — FM warmth (30%)
                if random.random() < 0.3:
                    filters.append("aemphasis=level_in=1:level_out=1:mode=reproduction:type=75fm")
                    log.append("vinyl-deemph")

            # ── BASS ───────────────────────────────────────────────
            elif any(x in name for x in ['bass']):
                sub_g = round(random.uniform(3.0, 6.0), 1)
                filters += [f"equalizer=f=55:width_type=o:width=2:gain={sub_g}",
                            "highpass=f=20"]
                log.append(f"sub+{sub_g}dB@55Hz")

                if random.random() < 0.35:
                    filters.append("chorus=0.7:0.9:50|55:0.3|0.25:0.5|0.4:2|1.6")
                    log.append("chorus-double")

                if random.random() < 0.30:
                    filters.append("equalizer=f=800:width_type=o:width=2:gain=2")
                    log.append("excite+2dB@800")

                # Virtual sub-harmonic synthesis (40%)
                if random.random() < 0.4:
                    strength = round(random.uniform(0.8, 2.0), 1)
                    filters.append(f"virtualbass=cutoff=250:strength={strength}")
                    log.append(f"virtualbass str={strength}")

                # Vintage compander for warmth (35%)
                if random.random() < 0.35:
                    filters.append(
                        "compand=attacks=0.1:decays=0.5"
                        ":points=-80/-80|-40/-35|-25/-15|-10/-6|0/0"
                    )
                    log.append("compand-vintage")

            # ── PADS / CHORDS ──────────────────────────────────────
            elif any(x in name for x in ['pad','chord']):
                # Reverb — always, hall or chamber
                rev_type = random.choice(['small_hall', 'chamber', 'hall'])
                reverb_filter = melody_reverbs[rev_type]
                reverb_wet    = round(random.uniform(0.35, 0.49), 2)
                reverb_label  = rev_type

                if random.random() < 0.6:
                    spd = round(random.uniform(0.2, 0.8), 2)
                    filters.append(f"aphaser=in_gain=0.4:out_gain=0.74:delay=3:decay=0.4:speed={spd}:type=t")
                    log.append(f"phaser@{spd}Hz")

                if random.random() < 0.4:
                    div  = random.choice([1, 2, 0.5])
                    tf   = round(bpm / 60.0 * div, 3)
                    dep  = round(random.uniform(0.2, 0.5), 2)
                    filters.append(f"tremolo=f={tf}:d={dep}")
                    log.append(f"tremolo@{tf}Hz")

                if random.random() < 0.3:
                    co  = random.randint(9000, 13000)
                    wr  = round(random.uniform(0.3, 0.7), 2)
                    wd  = round(random.uniform(0.01, 0.03), 3)
                    filters.append(f"lowpass=f={co},vibrato=f={wr}:d={wd}")
                    log.append(f"vinyl-LP{co}Hz")

                if random.random() < 0.35:
                    filters.append("chorus=0.6:0.85:45|50|55:0.35|0.3|0.25:0.3|0.4|0.35:2|1.8|1.6")
                    log.append("ensemble-chorus")

                # Dynamic EQ — reactive mud cut around 300Hz (30%)
                if random.random() < 0.3:
                    filters.append(
                        "adynamicequalizer=threshold=20:dfrequency=300:dqfactor=2"
                        ":tfrequency=300:tqfactor=2:range=6:mode=cutabove:tftype=bell"
                    )
                    log.append("dyn-EQ mud cut")

                # MS stereo widening via stereotools (40%)
                if random.random() < 0.4:
                    slev = round(random.uniform(1.2, 1.6), 1)
                    filters.append(f"stereotools=slev={slev}:mlev=1:mode=3")
                    log.append(f"stereo-MS slev={slev}")

                # Flanger — slow sweep for movement (25%)
                if random.random() < 0.25:
                    spd = round(random.uniform(0.1, 0.5), 2)
                    dep = random.randint(4, 9)
                    filters.append(f"flanger=delay=5:depth={dep}:regen=25:width=90:speed={spd}")
                    log.append(f"flanger d={dep}@{spd}Hz")

                # Tape de-emphasis warmth (25%)
                if random.random() < 0.25:
                    filters.append("aemphasis=level_in=1:level_out=1:mode=reproduction:type=75fm")
                    log.append("tape-deemph")

            # ── COUNTER MELODY — heavy processing ALWAYS ───────────
            elif 'counter' in name:
                # Cathedral reverb ALWAYS (via parallel blend, replaces old inline aecho reverb)
                reverb_filter = melody_reverbs['cathedral']
                reverb_wet    = 0.50
                reverb_label  = "cathedral"

                # Dotted-quarter delay ALWAYS
                d_wet = round(random.uniform(0.57, 0.85), 2)
                d_fb  = round(random.uniform(0.3, 0.50), 2)
                filters.append(f"aecho=1.0:{d_wet}:{dq_ms}:{d_fb}")
                log.append(f"dotted-¼delay {dq_ms}ms w={d_wet}")

                if random.random() < 0.5:
                    filters.append("chorus=0.8:0.9:25|50:0.5|0.4:0.8|0.6:2|1.6")
                    log.append("shimmer-chorus")
                if random.random() < 0.3:
                    filters.append("flanger=delay=5:depth=5:regen=30:width=90:speed=0.4")
                    log.append("flanger")

                # Harmonic exciter — presence in the upper-mids (40%)
                if random.random() < 0.4:
                    freq   = random.randint(4000, 7000)
                    amount = round(random.uniform(1.5, 3.5), 1)
                    filters.append(f"aexciter=freq={freq}:amount={amount}:blend=5")
                    log.append(f"exciter@{freq}Hz amt={amount}")

                # Phase shift for subtle harmonic timbral change (30%)
                if random.random() < 0.3:
                    shift = round(random.uniform(0.2, 0.5), 2)
                    filters.append(f"aphaseshift=shift={shift}:level=1")
                    log.append(f"phaseshift={shift}")

            # ── MAIN / CHORUS MELODY ───────────────────────────────
            elif any(x in name for x in ['melody','chorus']):
                # Reverb — ALWAYS, random hall type
                rev_type = random.choice(list(melody_reverbs.keys()))
                reverb_filter = melody_reverbs[rev_type]
                reverb_wet    = round(random.uniform(0.35, 0.49), 2)
                reverb_label  = rev_type

                if random.random() < 0.3:
                    co = random.randint(9000, 13000)
                    wr = round(random.uniform(0.3, 0.7), 2)
                    wd = round(random.uniform(0.01, 0.03), 3)
                    filters.append(f"lowpass=f={co},vibrato=f={wr}:d={wd}")
                    log.append(f"vinyl-LP{co}Hz wow@{wr}Hz")

                # Delay — ALWAYS (BPM-synced, randomised note value + wet/fb)
                dms = random.choice([e_ms, q_ms, dq_ms])
                wet = round(random.uniform(0.2, 0.40), 2)
                fb  = round(random.uniform(0.15, 0.35), 2)
                filters.append(f"aecho=0.9:{wet}:{dms}:{fb}")
                log.append(f"delay@{dms}ms w={wet}")

                if random.random() < 0.4:
                    filters.append("chorus=0.7:0.9:45|55:0.4|0.3:0.35|0.4:2|1.6")
                    log.append("chorus")

                if random.random() < 0.30:
                    spd = round(random.uniform(0.3, 1.2), 2)
                    filters.append(f"aphaser=in_gain=0.4:out_gain=0.74:delay=3:decay=0.4:speed={spd}:type=t")
                    log.append(f"phaser@{spd}Hz")

                if random.random() < 0.20:
                    dep = random.randint(3, 7)
                    spd = round(random.uniform(0.2, 0.8), 2)
                    filters.append(f"flanger=delay=5:depth={dep}:regen=20:width=90:speed={spd}")
                    log.append(f"flanger d={dep}@{spd}Hz")

                if random.random() < 0.25:
                    hms = random.randint(15, 35)
                    filters.append(f"adelay=0|{hms}")
                    log.append(f"haas-{hms}ms")

                # Harmonic exciter — air and presence (50%)
                if random.random() < 0.5:
                    freq   = random.randint(5000, 9000)
                    amount = round(random.uniform(1.0, 3.0), 1)
                    filters.append(f"aexciter=freq={freq}:amount={amount}:blend=4")
                    log.append(f"exciter@{freq}Hz amt={amount}")

                # Spectral tilt — subtle brightness or warmth shift (35%)
                if random.random() < 0.35:
                    slope = round(random.uniform(-0.8, 0.8), 1)
                    filters.append(f"atilt=freq=1000:slope={slope}:width=1000:order=5")
                    log.append(f"tilt slope={slope}")

                # Dynamic EQ — reactive harshness cut around 3kHz (30%)
                if random.random() < 0.3:
                    filters.append(
                        "adynamicequalizer=threshold=18:dfrequency=3000:dqfactor=2"
                        ":tfrequency=3000:tqfactor=2:range=4:mode=cutabove:tftype=bell"
                    )
                    log.append("dyn-EQ harsh cut")

            # ── FX STEMS ───────────────────────────────────────────
            elif any(x in name for x in ['fx','melody_fx']):
                if random.random() < 0.35:
                    filters.append("highpass=f=300,lowpass=f=3400")
                    log.append("telephone")
                if random.random() < 0.4:
                    bits = random.choice([4, 6, 8])
                    filters.append(f"acrusher=level_in=1:level_out=1:bits={bits}:mode=log:aa=1")
                    log.append(f"crush-{bits}bit")
                if random.random() < 0.45:
                    tf = round(bpm / 60.0, 3)
                    filters.append(f"tremolo=f={tf}:d=0.8")
                    log.append(f"rhythmic-gate@{tf}Hz")
                if random.random() < 0.2:
                    filters.append("chorus=0.6:0.9:7|8:0.7|0.6:0.9|0.8:2|1.8")
                    log.append("ring-mod-chorus")

                # Contrast punch (40%)
                if random.random() < 0.4:
                    contrast = random.randint(45, 75)
                    filters.append(f"acontrast=contrast={contrast}")
                    log.append(f"contrast={contrast}")

                # Stereo widening (30%)
                if random.random() < 0.3:
                    filters.append("stereowiden")
                    log.append("stereowiden")

            # ── Apply filter chain ──────────────────────────────────
            short = os.path.basename(path)[:48]
            if filters:
                print(f"  {short:48s}  {' | '.join(log)}")
                cmd = ["ffmpeg", "-y", "-i", path, "-af", ",".join(filters), "-c:a", "pcm_f32le", out_path]
                res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                if res.returncode != 0:
                    print(f"    ! FFmpeg failed ({res.stderr.decode()[:80]}), copying dry")
                    shutil.copy2(path, out_path)
            else:
                shutil.copy2(path, out_path)
                print(f"  {short:48s}  [bypass]")

            # ── Apply reverb (parallel wet/dry, second pass) ────────
            if reverb_filter:
                tmp = out_path + ".rev.wav"
                if self._apply_reverb(out_path, tmp, reverb_filter, reverb_wet):
                    os.replace(tmp, out_path)
                    print(f"    + reverb:{reverb_label} wet={reverb_wet}")
                else:
                    if os.path.exists(tmp):
                        os.remove(tmp)

            output_paths.append(out_path)

        # ── SIDECHAIN DUCKING — 70% of runs ───────────────────────
        if kick_ref and random.random() < 0.7:
            print(f"\n  Sidechain kick→bass/pads (release={q_ms}ms):")
            targets = [p for p in output_paths
                       if any(x in os.path.basename(p).lower() for x in ['bass','pad','chord'])]
            for tgt in targets:
                tmp = tgt + ".sc.wav"
                cmd = [
                    "ffmpeg", "-y", "-i", tgt, "-i", kick_ref,
                    "-filter_complex",
                    f"[0:a][1:a]sidechaincompress=threshold=0.05:ratio=8:attack=5:release={q_ms}:level_sc=0.5",
                    tmp
                ]
                res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                if res.returncode == 0:
                    os.replace(tmp, tgt)
                    print(f"    ✓ {os.path.basename(tgt)}")
                else:
                    if os.path.exists(tmp): os.remove(tmp)

        return output_paths

    def apply_intro_outro_automation(self, stem_paths: list, bpm: float, output_dir: str) -> list:
        """
        Apply long-form intro/outro arrangement moves to the elements that carry
        those sections: bass, pads/chords, verse melody, and drum1.
        One eligible group is left dry from bar 0 as the intro anchor; the rest
        receive per-song 4- or 8-bar fade/filter/wet-dry moves.
        """
        os.makedirs(output_dir, exist_ok=True)
        groups_present = sorted({
            group for path in stem_paths
            for group in [self._intro_outro_group(os.path.basename(path).lower())]
            if group
        })

        if not groups_present:
            output_paths = []
            for path in stem_paths:
                out_path = self._automation_out_path(path, output_dir, "_io")
                shutil.copy2(path, out_path)
                output_paths.append(out_path)
            return output_paths

        plan = self._build_intro_outro_plan(groups_present)
        print(
            "  Intro/outro plan: "
            f"intro={plan['intro_bars']} bars, outro={plan['outro_bars']} bars, "
            f"dry intro anchor={plan['anchor']}"
        )

        output_paths = []
        for path in stem_paths:
            name = os.path.basename(path).lower()
            out_path = self._automation_out_path(path, output_dir, "_io")
            group = self._intro_outro_group(name)

            if not group:
                shutil.copy2(path, out_path)
                output_paths.append(out_path)
                continue

            try:
                y, sr = sf.read(path)
            except Exception as e:
                print(f"  {os.path.basename(path)[:48]:48s}  intro/outro read failed: {e}")
                shutil.copy2(path, out_path)
                output_paths.append(out_path)
                continue

            y = self._ensure_stereo(np.asarray(y, dtype=np.float32))
            y_auto = y.copy()
            log = []

            if group != plan['anchor']:
                intro_start_bar = 8 - plan['intro_bars']
                intro_start = max(0, int(self._bars_to_seconds(intro_start_bar, bpm) * sr))
                intro_end = min(len(y_auto), int(self._bars_to_seconds(8, bpm) * sr))
                if intro_start > 0:
                    y_auto[:intro_start] = 0.0
                if intro_end > intro_start + sr // 4:
                    y_auto = self._apply_intro_transition(
                        y_auto, sr, intro_start, intro_end, plan['intro_effects'][group]
                    )
                    log.append(f"intro {plan['intro_bars']}bar {plan['intro_effects'][group]}->dry@bar8")
            else:
                log.append("intro dry-anchor")

            outro_start_bar = 72 - plan['outro_bars']
            outro_start = int(self._bars_to_seconds(outro_start_bar, bpm) * sr)
            outro_end = min(len(y_auto), int(self._bars_to_seconds(72, bpm) * sr))
            if outro_start < len(y_auto) - sr // 4 and outro_end > outro_start:
                y_auto = self._apply_outro_transition(
                    y_auto, sr, outro_start, outro_end, plan['outro_effects'][group]
                )
                log.append(f"outro {plan['outro_effects'][group]}")

            sf.write(out_path, y_auto, sr, subtype='FLOAT')
            print(f"  {os.path.basename(path)[:48]:48s}  {' | '.join(log)}")
            output_paths.append(out_path)

        return output_paths

    def _automation_out_path(self, path: str, output_dir: str, suffix: str) -> str:
        base_name = os.path.basename(path)
        for prefix in ["crt_", "auto_", "io_"]:
            if prefix in base_name:
                base_name = base_name.replace(prefix, "")
        base, ext = os.path.splitext(base_name)
        return os.path.join(output_dir, f"{base}{suffix}{ext}")

    def _intro_outro_group(self, name: str):
        if 'drum1_' in name:
            return 'drum1'
        if 'bass' in name:
            return 'bass'
        if any(x in name for x in ['pad', 'chord']):
            return 'pad'
        if 'main_melody' in name or ('melody' in name and not any(x in name for x in ['chorus', 'counter', 'fx'])):
            return 'verse_melody'
        return None

    def _build_intro_outro_plan(self, groups_present: list) -> dict:
        intro_palette = ['gain_fade', 'lowpass_open', 'highpass_restore', 'mid_iso_release', 'reverb_dry_in']
        outro_palette = ['gain_fade', 'lowpass_close', 'highpass_thin', 'mid_iso_fade', 'reverb_wash_out']
        anchor = random.choice(groups_present)
        return {
            'intro_bars': random.choice([4, 8]),
            'outro_bars': random.choice([4, 8]),
            'anchor': anchor,
            'intro_effects': {group: random.choice(intro_palette) for group in groups_present},
            'outro_effects': {group: random.choice(outro_palette) for group in groups_present},
        }

    def _bars_to_seconds(self, bars: float, bpm: float) -> float:
        return bars * (60.0 / bpm) * 4.0

    def _apply_intro_transition(self, y: np.ndarray, sr: int, start: int, end: int, effect: str) -> np.ndarray:
        out = y.copy()
        seg = out[start:end]
        if len(seg) <= 1:
            return out
        ramp = np.linspace(0.0, 1.0, len(seg), dtype=np.float32)[:, None]
        min_gain = 0.0 if effect == 'gain_fade' else 0.18

        if effect == 'gain_fade':
            out[start:end] = seg * ramp
        elif effect == 'lowpass_open':
            filtered = self._variable_lowpass(seg, sr, 450.0, 18000.0)
            out[start:end] = (filtered * (1.0 - ramp) + seg * ramp) * (min_gain + (1.0 - min_gain) * ramp)
        elif effect == 'highpass_restore':
            filtered = self._variable_highpass(seg, sr, 5000.0, 45.0)
            out[start:end] = (filtered * (1.0 - ramp) + seg * ramp) * (min_gain + (1.0 - min_gain) * ramp)
        elif effect == 'mid_iso_release':
            mid = self._band_isolate(seg, sr, 300.0, 3400.0)
            out[start:end] = (mid * (1.0 - ramp) + seg * ramp) * (0.25 + 0.75 * ramp)
        elif effect == 'reverb_dry_in':
            wet = self._simple_echo_wash(seg, sr, decay=0.42)
            wet_mix = 1.0 - ramp
            out[start:end] = (wet * wet_mix + seg * (1.0 - wet_mix)) * (0.22 + 0.78 * ramp)
        return out

    def _apply_outro_transition(self, y: np.ndarray, sr: int, start: int, end: int, effect: str) -> np.ndarray:
        out = y.copy()
        seg = out[start:end]
        if len(seg) <= 1:
            return out
        fade = np.linspace(1.0, 0.0, len(seg), dtype=np.float32)[:, None]
        wet_rise = 1.0 - fade

        if effect == 'gain_fade':
            out[start:end] = seg * fade
        elif effect == 'lowpass_close':
            filtered = self._variable_lowpass(seg, sr, 18000.0, 450.0)
            out[start:end] = (seg * fade + filtered * wet_rise) * (0.15 + 0.85 * fade)
        elif effect == 'highpass_thin':
            filtered = self._variable_highpass(seg, sr, 45.0, 5200.0)
            out[start:end] = (seg * fade + filtered * wet_rise) * (0.12 + 0.88 * fade)
        elif effect == 'mid_iso_fade':
            mid = self._band_isolate(seg, sr, 300.0, 3400.0)
            out[start:end] = (seg * fade + mid * wet_rise) * (0.10 + 0.90 * fade)
        elif effect == 'reverb_wash_out':
            wet = self._simple_echo_wash(seg, sr, decay=0.55)
            out[start:end] = seg * fade + wet * wet_rise * 0.55
        return out

    def _variable_lowpass(self, y: np.ndarray, sr: int, start_cutoff: float, end_cutoff: float) -> np.ndarray:
        cutoffs = np.linspace(start_cutoff, end_cutoff, len(y), dtype=np.float32)
        out = np.zeros_like(y)
        state = y[0].astype(np.float32)
        out[0] = state
        for i in range(1, len(y)):
            cutoff = max(20.0, min(float(cutoffs[i]), sr * 0.45))
            alpha = (2.0 * np.pi * cutoff) / (2.0 * np.pi * cutoff + sr)
            state = state + alpha * (y[i] - state)
            out[i] = state
        return out

    def _variable_highpass(self, y: np.ndarray, sr: int, start_cutoff: float, end_cutoff: float) -> np.ndarray:
        return y - self._variable_lowpass(y, sr, start_cutoff, end_cutoff)

    def _band_isolate(self, y: np.ndarray, sr: int, low_cutoff: float, high_cutoff: float) -> np.ndarray:
        lowpassed = self._lowpass_numpy(y, sr, high_cutoff)
        return lowpassed - self._lowpass_numpy(lowpassed, sr, low_cutoff)

    def _simple_echo_wash(self, y: np.ndarray, sr: int, decay: float = 0.45) -> np.ndarray:
        wet = y.copy() * 0.55
        for delay_sec, gain in [(0.18, decay), (0.37, decay * 0.55), (0.74, decay * 0.32)]:
            delay = int(delay_sec * sr)
            if delay < len(y):
                wet[delay:] += y[:-delay] * gain
        return self._lowpass_numpy(wet, sr, 6500.0)

    def apply_phrase_automation_fx(self, stem_paths: list, bpm: float, output_dir: str) -> list:
        """
        Apply unorthodox effects as short automations at phrase endings.
        The gestures are intentionally sparse and category-aware so the full mix keeps
        its arrangement while phrase endings get occasional hardware-performance drama.
        """
        os.makedirs(output_dir, exist_ok=True)
        drum_env = self._build_drum_modulator(stem_paths)
        output_paths = []

        for path in stem_paths:
            name = os.path.basename(path).lower()
            out_name = os.path.basename(path).replace("crt_", "auto_")
            if out_name == os.path.basename(path):
                base, ext = os.path.splitext(os.path.basename(path))
                out_name = base + "_auto" + ext
            out_path = os.path.join(output_dir, out_name)

            try:
                y, sr = sf.read(path)
            except Exception as e:
                print(f"  {os.path.basename(path)[:48]:48s}  automation read failed: {e}")
                shutil.copy2(path, out_path)
                output_paths.append(out_path)
                continue

            y = self._ensure_stereo(np.asarray(y, dtype=np.float32))
            automation_events = self._song_structure_events(len(y), sr, bpm)
            if not automation_events:
                shutil.copy2(path, out_path)
                output_paths.append(out_path)
                continue

            y_auto = y.copy()
            applied = []

            is_drum = self._is_drum_name(name)
            is_bass = 'bass' in name
            is_pad = any(x in name for x in ['pad', 'chord', 'string', 'choir'])
            is_melody = any(x in name for x in ['melody', 'lead', 'counter', 'chorus'])
            is_fx = any(x in name for x in ['fx', 'texture'])

            selected = self._select_automation_events(automation_events, 0.08) if is_bass else []
            if selected:
                y_auto = self._apply_octave_sub_bursts(y_auto, sr, selected, bpm)
                applied.append("octave-sub bursts")

            selected = self._select_automation_events(automation_events, 0.03) if (is_bass or is_drum or is_fx) else []
            if selected:
                y_auto = self._apply_auto_wah_bursts(y_auto, sr, selected, bpm)
                applied.append("auto-wah endings")

            selected = self._select_automation_events(automation_events, 0.05) if (is_pad or is_melody) else []
            if selected:
                y_auto = self._apply_pitch_drift_sections(y_auto, sr, selected, bpm)
                applied.append("section pitch drift")

            selected = self._select_automation_events(automation_events, 0.02) if (is_drum or is_melody or is_fx) else []
            if selected:
                y_auto = self._apply_stutter_gate(y_auto, sr, selected, bpm)
                applied.append("1/16 stutter gate")

            selected = self._select_automation_events(automation_events, 0.01) if (is_drum or is_melody or is_fx) else []
            if selected:
                y_auto = self._apply_tape_stop_events(y_auto, sr, selected, bpm)
                applied.append("tape stop")

            selected = self._select_automation_events(automation_events, 0.04) if (is_pad or is_melody) else []
            if selected:
                y_auto = self._apply_spectral_freeze(y_auto, sr, selected, bpm)
                applied.append("spectral freeze")

            selected = self._select_automation_events(automation_events, 0.10) if (is_pad or is_melody or is_drum) else []
            if selected:
                y_auto = self._apply_reverse_reverb_prehits(y_auto, sr, selected, bpm)
                applied.append("reverse reverb pre-hit")

            selected = self._select_automation_events(automation_events, 0.05) if (is_pad and drum_env) else []
            if selected:
                y_auto = self._apply_vocoder_texture(y_auto, sr, selected, bpm, drum_env)
                applied.append("drum vocoder texture")

            selected = self._select_automation_events(automation_events, 0.06) if (is_pad or is_melody or is_fx) else []
            if selected:
                y_auto = self._apply_granular_shimmer(y_auto, sr, selected, bpm)
                applied.append("granular shimmer")

            selected = self._select_automation_events(automation_events, 0.05) if (is_pad or is_melody or is_drum or is_fx) else []
            if selected:
                y_auto = self._apply_resonant_sweep(y_auto, sr, selected, bpm)
                applied.append("resonant sweep")

            sf.write(out_path, y_auto, sr, subtype='FLOAT')

            # Sectional energy scaling: chorus gets +1dB gain and ×1.1 stereo width
            try:
                from song_structure import get_bar_type
                y_se, sr_se = sf.read(out_path)
                y_se = self._ensure_stereo(np.asarray(y_se, dtype=np.float32))
                total_samples = len(y_se)
                samples_per_bar = int(sr_se * 60.0 / bpm * 4)  # 4 beats per bar
                chorus_gain = 10.0 ** (1.0 / 20.0)  # +1dB
                fade_samples = int(sr_se * 0.020)   # 20ms crossfade
                gain_curve = np.ones(total_samples, dtype=np.float32)
                bar = 0
                sample_pos = 0
                while sample_pos < total_samples:
                    bar_type = get_bar_type(bar)
                    is_chorus = (bar_type == 'C')
                    end_pos = min(sample_pos + samples_per_bar, total_samples)
                    target_gain = chorus_gain if is_chorus else 1.0
                    # Smooth transition
                    if sample_pos > 0:
                        fade_end = min(sample_pos + fade_samples, end_pos)
                        gain_curve[sample_pos:fade_end] = np.linspace(gain_curve[sample_pos-1], target_gain, fade_end - sample_pos)
                    gain_curve[sample_pos:end_pos] = target_gain
                    sample_pos = end_pos
                    bar += 1
                y_se = y_se * gain_curve[:, np.newaxis]
                # Also boost side channel during chorus by 10%
                # (re-use _apply_ms_width logic inline)
                # For simplicity just apply gain scaling (width can be a follow-up)
                peak = np.max(np.abs(y_se))
                if peak > 1.0:
                    y_se /= peak
                sf.write(out_path, y_se, sr_se, subtype='FLOAT')
            except Exception:
                pass  # song_structure unavailable or error — skip silently

            short = os.path.basename(path)[:48]
            if applied:
                print(f"  {short:48s}  {' | '.join(applied)}")
            else:
                print(f"  {short:48s}  [automation bypass]")
            output_paths.append(out_path)

        return output_paths

    def _song_structure_events(self, n_samples: int, sr: int, bpm: float) -> list:
        """
        Build automation targets from the v10 song structure.
        Stems are exported with the count-in removed, so bar 0 maps to sample 0.
        Events are limited to section transitions and mid-verse moments:
          8-bar intro → 16-bar verse → 8-bar chorus → 4-bar fill →
          16-bar verse2 → 8-bar chorus2 → 4-bar fill2 → 8-bar outro.
        """
        bar_dur = (60.0 / bpm) * 4.0
        duration = n_samples / float(sr)
        total_bars = max(1, int(np.ceil(duration / bar_dur)))
        events = []

        for bar in range(1, total_bars):
            t = bar * bar_dur
            if t >= duration - 0.25:
                break

            prev_section = get_bar_type(bar - 1)
            section = get_bar_type(bar)
            if section != prev_section:
                weight = 2.4
                if self._is_feature_transition(prev_section, section):
                    weight = 3.2
                events.append({
                    'time': t,
                    'kind': 'section',
                    'from': prev_section,
                    'to': section,
                    'weight': weight,
                })
            elif section.startswith('verse') and get_phrase_position(bar) == 0:
                events.append({
                    'time': t,
                    'kind': 'verse_midpoint',
                    'from': section,
                    'to': section,
                    'weight': 1.6,
                })

        return events

    def _is_feature_transition(self, from_section: str, to_section: str) -> bool:
        if from_section.startswith('verse') and to_section.startswith('chorus'):
            return True
        if from_section.startswith('fill') and (to_section.startswith('verse') or to_section == 'outro'):
            return True
        if from_section == 'intro' and to_section.startswith('verse'):
            return True
        return False

    def _select_automation_events(self, events: list, base_probability: float) -> list:
        selected = []
        for event in events:
            probability = min(0.95, base_probability * event.get('weight', 1.0))
            if random.random() < probability:
                selected.append(event)
        return selected

    def _event_time(self, event) -> float:
        return event.get('time', 0.0) if isinstance(event, dict) else float(event)

    def _build_drum_modulator(self, stem_paths: list):
        drum_paths = [p for p in stem_paths if self._is_drum_name(os.path.basename(p).lower())]
        if not drum_paths:
            return None
        try:
            y, sr = sf.read(drum_paths[0])
        except Exception:
            return None
        mono = self._mono(np.asarray(y, dtype=np.float32))
        env = self._smooth_envelope(np.abs(mono), max(64, int(sr * 0.015)))
        peak = np.max(env)
        if peak <= 1e-6:
            return None
        return sr, env / peak

    def _is_drum_name(self, name: str) -> bool:
        return any(x in name for x in [
            'kick', 'snare', 'hat', 'clap', 'drum', 'bongo', 'conga',
            'tambourine', 'maracas', 'perc', 'instr', 'side_stick',
            'ride', 'crash', 'cymbal'
        ])

    def _ensure_stereo(self, y: np.ndarray) -> np.ndarray:
        if y.ndim == 1:
            return np.stack([y, y], axis=1)
        if y.shape[1] == 1:
            return np.repeat(y, 2, axis=1)
        return y[:, :2]

    def _mono(self, y: np.ndarray) -> np.ndarray:
        return np.mean(y, axis=1) if y.ndim > 1 else y

    def _event_window(self, event_sec: float, sr: int, pre_sec: float, post_sec: float, n_samples: int):
        event_sec = self._event_time(event_sec)
        start = max(0, int((event_sec - pre_sec) * sr))
        end = min(n_samples, int((event_sec + post_sec) * sr))
        return start, end

    def _fade_window(self, length: int, fade_len: int) -> np.ndarray:
        win = np.ones(length, dtype=np.float32)
        fade_len = min(fade_len, length // 2)
        if fade_len > 1:
            fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            win[:fade_len] *= fade_in
            win[-fade_len:] *= fade_out
        return win

    def _smooth_envelope(self, x: np.ndarray, win_len: int) -> np.ndarray:
        if win_len <= 1:
            return x
        kernel = np.ones(win_len, dtype=np.float32) / float(win_len)
        return np.convolve(x, kernel, mode='same')

    def _peak_limit(self, y: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
        peak = np.max(np.abs(y))
        if peak > ceiling:
            y = y * (ceiling / peak)
        return y

    def _lowpass_numpy(self, y: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
        cutoff = max(20.0, min(cutoff, sr * 0.45))
        alpha = (2.0 * np.pi * cutoff) / (2.0 * np.pi * cutoff + sr)
        out = np.zeros_like(y)
        out[0] = y[0]
        for i in range(1, len(y)):
            out[i] = out[i - 1] + alpha * (y[i] - out[i - 1])
        return out

    def _numpy_compress(self, y: np.ndarray, threshold: float, ratio: float,
                        attack_ms: float, release_ms: float, sr: int,
                        apply_makeup: bool = True) -> np.ndarray:
        """Soft-knee compressor in numpy. threshold is linear (0-1 scale)."""
        y64 = y.astype(np.float64)
        # Envelope follower (full-wave rectified, IIR smoothing)
        env = np.abs(y64)
        if env.ndim == 2:
            env = np.max(env, axis=1)  # peak across channels
        attack_coef  = np.exp(-1.0 / (sr * attack_ms  / 1000.0))
        release_coef = np.exp(-1.0 / (sr * release_ms / 1000.0))
        smoothed = np.zeros_like(env)
        prev = 0.0
        for i in range(len(env)):
            coef = attack_coef if env[i] > prev else release_coef
            smoothed[i] = (1 - coef) * env[i] + coef * prev
            prev = smoothed[i]
        # Gain reduction curve (soft knee, width = 6dB around threshold)
        knee_width = 6.0
        threshold_db = 20.0 * np.log10(np.maximum(threshold, 1e-10))
        smoothed_db  = 20.0 * np.log10(np.maximum(smoothed, 1e-10))
        gain_db = np.zeros_like(smoothed_db)
        # Below knee: no gain reduction
        below_knee = smoothed_db < (threshold_db - knee_width / 2)
        # Above knee: full compression
        above_knee = smoothed_db > (threshold_db + knee_width / 2)
        # In knee: quadratic transition
        in_knee = ~below_knee & ~above_knee
        gain_db[above_knee] = threshold_db + (smoothed_db[above_knee] - threshold_db) / ratio - smoothed_db[above_knee]
        knee_excess = smoothed_db[in_knee] - threshold_db + knee_width / 2
        gain_db[in_knee] = (1.0 / ratio - 1.0) * (knee_excess ** 2) / (2.0 * knee_width)
        gain_lin = 10.0 ** (gain_db / 20.0)
        # Makeup gain
        makeup_db = threshold_db * (1.0 - 1.0 / ratio) if apply_makeup else 0.0
        makeup_lin = 10.0 ** (makeup_db / 20.0)
        if y64.ndim == 2:
            gain_lin = gain_lin[:, np.newaxis]
        return (y64 * gain_lin * makeup_lin).astype(np.float32)

    def _build_parallel_drum_bus(self, y: np.ndarray, sr: int, blend: float = 0.20) -> tuple:
        """NY-style parallel drum compression: heavy compress+saturate, blended back."""
        y64 = y.astype(np.float64)
        crush_thr, crush_gr = self._find_adaptive_threshold(y64, sr, ratio=10.0, attack_ms=2, release_ms=40)
        print(f"    Drum crush: thr={20*np.log10(max(crush_thr,1e-10)):.1f}dB  10:1  GR={crush_gr:.1f}dB")
        crush = self._numpy_compress(y64, threshold=crush_thr, ratio=10.0,
                                      attack_ms=2, release_ms=40, sr=sr).astype(np.float64)
        crush = np.tanh(crush * 2.5) / np.tanh(2.5)
        blend = float(np.clip(blend, 0.0, 0.35))
        out = y64 * (1.0 - blend) + crush * blend
        peak = np.max(np.abs(out))
        if peak > 1.0:
            out /= peak
        debug = {
            "threshold_linear": crush_thr,
            "threshold_db": 20.0 * np.log10(max(crush_thr, 1e-10)),
            "ratio": 10.0,
            "attack_ms": 2,
            "release_ms": 40,
            "mean_gain_reduction_db": crush_gr,
            "blend": blend,
            "saturation": "tanh_x2.5",
        }
        return out.astype(np.float32), debug

    def _apply_sidechain_ducking(self, bass_y: np.ndarray, drums_y: np.ndarray,
                                  sr: int, duck_depth: float = 0.06) -> np.ndarray:
        """Subtle kick-triggered gain reduction on bass bus. duck_depth=0.06 ≈ -0.5dB."""
        bass64 = bass_y.astype(np.float64)
        # Extract sidechain envelope from drums
        sc = np.abs(drums_y).astype(np.float64)
        if sc.ndim == 2:
            sc = np.max(sc, axis=1)
        # IIR envelope: fast attack (2ms), slow release (80ms)
        a_c = np.exp(-1.0 / (sr * 2.0  / 1000.0))
        r_c = np.exp(-1.0 / (sr * 80.0 / 1000.0))
        env, prev = np.zeros(len(sc)), 0.0
        for i in range(len(sc)):
            c = a_c if sc[i] > prev else r_c
            env[i] = (1 - c) * sc[i] + c * prev
            prev = env[i]
        # Normalise sidechain
        max_env = env.max()
        if max_env > 1e-10:
            env /= max_env
        # Gain reduction curve
        gain = 1.0 - env * duck_depth  # max reduction ≈ -1.5dB
        if bass64.ndim == 2:
            # Pad/trim to match bass length
            if len(gain) < bass64.shape[0]:
                gain = np.pad(gain, (0, bass64.shape[0] - len(gain)), constant_values=1.0)
            else:
                gain = gain[:bass64.shape[0]]
            gain = gain[:, np.newaxis]
        return (bass64 * gain).astype(np.float32)

    def _apply_ms_width(self, y: np.ndarray, sr: int, width_multiplier: float = 1.35) -> np.ndarray:
        """Expand stereo width of mid-high content (above 200Hz only)."""
        if y.ndim == 1:
            return y  # mono, skip
        y64 = y.astype(np.float64)
        mid  = (y64[:, 0] + y64[:, 1]) * 0.5
        side = (y64[:, 0] - y64[:, 1]) * 0.5
        # HPF the side channel above 200Hz (no width expansion in lows)
        side_hpf = side - self._lowpass_numpy(side, sr, 200.0)
        side_widened = side_hpf * width_multiplier
        y_out = y64.copy()
        y_out[:, 0] = mid + side_widened
        y_out[:, 1] = mid - side_widened
        peak = np.max(np.abs(y_out))
        if peak > 1.0:
            y_out /= peak
        return y_out.astype(np.float32)

    def _apply_sub_harmonic_enhancement(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Gentle saturation of sub-bass (<80Hz) to generate 2nd harmonic for small speakers."""
        y64 = y.astype(np.float64)
        # Low-pass the sub content
        sub = self._lowpass_numpy(y64 if y64.ndim == 1 else y64[:, 0], sr, 80.0)
        if y64.ndim == 2:
            sub_l = self._lowpass_numpy(y64[:, 0], sr, 80.0)
            sub_r = self._lowpass_numpy(y64[:, 1], sr, 80.0)
            sub = np.column_stack([sub_l, sub_r])
        # Mild saturation: generates 2nd harmonic (octave up = 2× fundamental)
        enhanced = np.tanh(sub * 1.3) / np.tanh(1.3)
        # Add back at 20% level
        result = y64 + enhanced * 0.20
        peak = np.max(np.abs(result))
        if peak > 1.0:
            result /= peak
        return result.astype(np.float32)

    def _apply_tpdf_dither(self, y: np.ndarray, bit_depth: int = 24) -> np.ndarray:
        """TPDF dither: two uniform random values summed = triangular noise distribution."""
        lsb = 2.0 / (2 ** bit_depth)
        dither = (np.random.uniform(-lsb, lsb, y.shape) +
                  np.random.uniform(-lsb, lsb, y.shape))
        return (y + dither.astype(y.dtype))

    def _apply_octave_sub_bursts(self, y: np.ndarray, sr: int, events: list, bpm: float) -> np.ndarray:
        out = y.copy()
        burst = (60.0 / bpm) * 2.0
        for event in events:
            start, end = self._event_window(event, sr, burst, 0.05, len(out))
            seg = out[start:end]
            if len(seg) < sr // 10:
                continue
            mono = self._mono(seg)
            rectified = np.abs(mono) - np.mean(np.abs(mono))
            sub = librosa.effects.pitch_shift(rectified.astype(np.float32), sr=sr, n_steps=-12)
            sub = sub[:len(seg)]
            sub = self._lowpass_numpy(sub, sr, 120.0)
            overlay = np.stack([sub, sub], axis=1) * 0.22
            win = self._fade_window(len(seg), int(sr * 0.08))[:, None]
            out[start:end] += overlay * win
        return out

    def _apply_auto_wah_bursts(self, y: np.ndarray, sr: int, events: list, bpm: float) -> np.ndarray:
        out = y.copy()
        burst = (60.0 / bpm) * 1.5
        for event in events:
            start, end = self._event_window(event, sr, burst, 0.10, len(out))
            seg = out[start:end]
            if len(seg) < sr // 8:
                continue
            mono = np.abs(self._mono(seg))
            env = self._smooth_envelope(mono, max(32, int(sr * 0.02)))
            if np.max(env) > 1e-6:
                env = env / np.max(env)
            filtered = np.zeros_like(seg)
            state = np.zeros(seg.shape[1], dtype=np.float32)
            for i, sample in enumerate(seg):
                cutoff = 350.0 + 2800.0 * env[i]
                alpha = (2.0 * np.pi * cutoff) / (2.0 * np.pi * cutoff + sr)
                state = state + alpha * (sample - state)
                filtered[i] = state
            win = self._fade_window(len(seg), int(sr * 0.05))[:, None]
            out[start:end] = seg * (1.0 - 0.55 * win) + filtered * (0.95 * win)
        return out

    def _apply_pitch_drift_sections(self, y: np.ndarray, sr: int, events: list, bpm: float) -> np.ndarray:
        out = y.copy()
        section_len = (60.0 / bpm) * random.choice([32, 64])
        for event in events:
            start, end = self._event_window(event, sr, section_len, 0.0, len(out))
            seg = out[start:end]
            if len(seg) < sr:
                continue
            steps = random.choice([-0.18, -0.12, 0.12, 0.18])
            drifted = []
            for ch in range(seg.shape[1]):
                drifted.append(librosa.effects.pitch_shift(seg[:, ch], sr=sr, n_steps=steps))
            drifted = np.stack(drifted, axis=1)[:len(seg)]
            ramp = np.linspace(0.0, 0.45, len(seg), dtype=np.float32)[:, None]
            out[start:end] = seg * (1.0 - ramp) + drifted * ramp
        return out

    def _apply_stutter_gate(self, y: np.ndarray, sr: int, events: list, bpm: float) -> np.ndarray:
        out = y.copy()
        sixteenth = (60.0 / bpm) / 4.0
        burst = sixteenth * random.choice([8, 12, 16])
        for event in events:
            start, end = self._event_window(event, sr, burst, 0.0, len(out))
            length = end - start
            if length <= 0:
                continue
            idx = np.arange(length) / float(sr)
            phase = np.mod(idx, sixteenth) / sixteenth
            gate = np.where(phase < random.uniform(0.38, 0.55), 1.0, random.uniform(0.02, 0.18))
            win = self._fade_window(length, int(sr * 0.02))
            out[start:end] *= (gate * win)[:, None]
        return out

    def _apply_tape_stop_events(self, y: np.ndarray, sr: int, events: list, bpm: float) -> np.ndarray:
        out = y.copy()
        stop_len = min(1.2, (60.0 / bpm) * 2.0)
        for event in events:
            start, end = self._event_window(event, sr, stop_len, 0.0, len(out))
            seg = out[start:end]
            if len(seg) < sr // 8:
                continue
            mono_len = len(seg)
            src_idx = np.linspace(0, mono_len - 1, mono_len)
            curve = np.cumsum(np.linspace(1.0, 0.12, mono_len))
            curve = curve / curve[-1] * (mono_len - 1)
            stopped = np.zeros_like(seg)
            for ch in range(seg.shape[1]):
                stopped[:, ch] = np.interp(src_idx, curve, seg[:, ch], left=seg[0, ch], right=0.0)
            amp = np.linspace(1.0, 0.0, mono_len, dtype=np.float32)[:, None]
            out[start:end] = stopped * amp
        return out

    def _apply_spectral_freeze(self, y: np.ndarray, sr: int, events: list, bpm: float) -> np.ndarray:
        out = y.copy()
        freeze_len = (60.0 / bpm) * random.choice([4, 8])
        grain_len = int(sr * 0.18)
        for event in events:
            center = int(max(0, (self._event_time(event) - (60.0 / bpm) * 2.0) * sr))
            source = y[center:center + grain_len]
            if len(source) < grain_len // 2:
                continue
            source = source * self._fade_window(len(source), int(sr * 0.02))[:, None]
            start, end = self._event_window(event, sr, 0.0, freeze_len, len(out))
            length = end - start
            tiled = np.resize(source, (length, source.shape[1]))
            pad = self._lowpass_numpy(tiled, sr, 4500.0) * 0.24
            env = self._fade_window(length, int(sr * 0.35))[:, None]
            out[start:end] += pad * env
        return out

    def _apply_reverse_reverb_prehits(self, y: np.ndarray, sr: int, events: list, bpm: float) -> np.ndarray:
        out = y.copy()
        tail_len = int(min(1.5, (60.0 / bpm) * 2.0) * sr)
        for event in events:
            hit = int(self._event_time(event) * sr)
            source = y[hit:min(len(y), hit + max(tail_len // 2, int(sr * 0.25)))]
            if len(source) < sr // 10:
                continue
            wet = np.flip(source, axis=0)
            wet = np.resize(wet, (tail_len, y.shape[1]))
            wet = self._lowpass_numpy(wet, sr, 6000.0)
            ramp = np.linspace(0.0, 1.0, tail_len, dtype=np.float32)[:, None]
            wet *= ramp * 0.20
            start = max(0, hit - tail_len)
            wet = wet[-(hit - start):]
            out[start:hit] += wet
        return out

    def _apply_vocoder_texture(self, y: np.ndarray, sr: int, events: list, bpm: float, drum_env) -> np.ndarray:
        env_sr, env = drum_env
        if env_sr != sr:
            x_old = np.linspace(0.0, 1.0, len(env))
            x_new = np.linspace(0.0, 1.0, int(len(env) * sr / env_sr))
            env = np.interp(x_new, x_old, env).astype(np.float32)
        out = y.copy()
        burst = (60.0 / bpm) * 4.0
        for event in events:
            start, end = self._event_window(event, sr, burst, 0.0, len(out))
            if end <= start:
                continue
            seg = out[start:end]
            mod = env[start:min(end, len(env))]
            if len(mod) < len(seg):
                mod = np.pad(mod, (0, len(seg) - len(mod)))
            mod = 0.35 + 0.85 * mod[:len(seg)]
            win = self._fade_window(len(seg), int(sr * 0.08))
            out[start:end] = seg * (1.0 - 0.45 * win[:, None]) + seg * mod[:, None] * (0.45 * win[:, None])
        return out

    def _apply_granular_shimmer(self, y: np.ndarray, sr: int, events: list, bpm: float) -> np.ndarray:
        out = y.copy()
        grain_len = max(2048, int(sr * random.uniform(0.07, 0.14)))
        for event in events:
            start, end = self._event_window(event, sr, (60.0 / bpm) * 0.5, (60.0 / bpm) * 3.0, len(out))
            seg = y[start:end]
            if len(seg) < grain_len * 2:
                continue
            shimmer = np.zeros_like(seg)
            for _ in range(random.randint(6, 10)):
                pos = random.randint(0, max(1, len(seg) - grain_len))
                grain = seg[pos:pos + grain_len]
                shifted = []
                for ch in range(grain.shape[1]):
                    shifted.append(librosa.effects.pitch_shift(grain[:, ch], sr=sr, n_steps=random.choice([7, 12])))
                shifted = np.stack(shifted, axis=1)[:grain_len]
                shifted *= self._fade_window(len(shifted), max(8, grain_len // 4))[:, None]
                dst = min(len(seg) - len(shifted), pos + random.randint(0, grain_len * 3))
                shimmer[dst:dst + len(shifted)] += shifted
            shimmer = self._lowpass_numpy(shimmer, sr, 9000.0)
            out[start:end] += shimmer * 0.16
        return out

    def _apply_resonant_sweep(self, y: np.ndarray, sr: int, events: list, bpm: float) -> np.ndarray:
        out = y.copy()
        sweep_len = (60.0 / bpm) * random.choice([4, 8])
        for event in events:
            start, end = self._event_window(event, sr, sweep_len, 0.0, len(out))
            seg = out[start:end]
            if len(seg) < sr // 4:
                continue
            low = self._lowpass_numpy(seg, sr, random.uniform(700.0, 1200.0))
            high = seg - self._lowpass_numpy(seg, sr, random.uniform(3500.0, 5500.0))
            sweep = np.linspace(0.0, 1.0, len(seg), dtype=np.float32)[:, None]
            resonant = low * (1.0 - sweep) + high * sweep
            win = self._fade_window(len(seg), int(sr * 0.08))[:, None]
            out[start:end] = seg + resonant * win * 0.22
        return out

    def apply_phase3_processing(self, stem_paths: list, output_dir: str) -> list:
        """
        Phase 3: harmonic enhancement + frequency slotting on all stems.
        1. Bitcrushing on drums, tanh saturation on bass/pads, bypass on rest.
        2. Detect kick fundamental → notch in bass at that frequency.
        3. Clarity EQ (mud + harshness) on melodies and pads.
        Returns list of output paths (hrm_ prefix).
        """
        # --- Step 1: per-stem harmonic enhancement ---
        harmonic_paths = []
        for path in stem_paths:
            name = os.path.basename(path).lower()
            out_path = os.path.join(output_dir, os.path.basename(path).replace("cmp_", "hrm_"))
            if out_path == path:
                base, ext = os.path.splitext(path)
                out_path = base + "_hrm" + ext

            if any(x in name for x in ['kick', 'snare', 'hat', 'clap', 'drum',
                                        'bongo', 'conga', 'tambourine', 'maracas',
                                        'perc', 'instr', 'side_stick']):
                # Bitcrushing: 10-bit depth — grit and punch
                filt = "acrusher=level_in=1:level_out=1:bits=10:mode=log:aa=1"
                label = "bitcrush 10-bit"

            elif any(x in name for x in ['bass']):
                # Tanh soft saturation — harmonic warmth
                filt = "aeval='tanh(val(0)*2.5)/tanh(2.5)|tanh(val(1)*2.5)/tanh(2.5)'"
                label = "tanh saturation ×2.5"

            elif any(x in name for x in ['pad', 'chord']):
                # Gentle tape saturation — analogue glue
                filt = "aeval='tanh(val(0)*1.8)/tanh(1.8)|tanh(val(1)*1.8)/tanh(1.8)'"
                label = "tape saturation ×1.8"

            else:
                shutil.copy2(path, out_path)
                harmonic_paths.append(out_path)
                continue

            print(f"  {os.path.basename(path)[:55]:55s}  [{label}]")
            cmd = ["ffmpeg", "-y", "-i", path, "-af", filt, "-c:a", "pcm_f32le", out_path]
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if result.returncode != 0:
                shutil.copy2(path, out_path)
            harmonic_paths.append(out_path)

        # --- Step 2: frequency slotting — detect kick fundamental, notch bass ---
        kick_paths = [p for p in harmonic_paths if 'kick' in os.path.basename(p).lower()]
        bass_paths = [p for p in harmonic_paths if 'bass' in os.path.basename(p).lower()]

        kick_freq = None
        if kick_paths:
            try:
                y_kick, sr_kick = sf.read(kick_paths[0])
                mono = np.mean(y_kick, axis=1) if y_kick.ndim > 1 else y_kick
                kick_freq = self.analyzer.get_fundamental_frequency(mono, sr_kick)
                print(f"  Kick fundamental detected: {kick_freq:.1f}Hz")
            except Exception as e:
                print(f"  Kick fundamental detection failed: {e}")

        if kick_freq and bass_paths:
            for bass_path in bass_paths:
                notched = bass_path.replace("hrm_", "hrm_notch_")
                filt = f"equalizer=f={kick_freq:.1f}:width_type=o:width=2:gain=-4"
                cmd = ["ffmpeg", "-y", "-i", bass_path, "-af", filt, "-c:a", "pcm_f32le", notched]
                result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                if result.returncode == 0:
                    os.replace(notched, bass_path)
                    print(f"  Bass notch applied at {kick_freq:.1f}Hz (-4dB)")

        # --- Step 3: clarity EQ on melodies and pads ---
        for i, path in enumerate(harmonic_paths):
            name = os.path.basename(path).lower()
            if not any(x in name for x in ['melody', 'lead', 'counter', 'chorus', 'pad', 'chord', 'fx_']):
                continue
            clarity_path = path + ".clarity.wav"
            # Bell cut 350Hz (-3dB mud), bell cut 3200Hz (-2.5dB harshness)
            filt = "equalizer=f=350:width_type=o:width=1:gain=-3,equalizer=f=3200:width_type=o:width=1.5:gain=-2.5"
            cmd = ["ffmpeg", "-y", "-i", path, "-af", filt, "-c:a", "pcm_f32le", clarity_path]
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if result.returncode == 0:
                os.replace(clarity_path, path)
                print(f"  Clarity EQ applied: {os.path.basename(path)[:50]}")

        return harmonic_paths

    def apply_dynamic_processing(self, stem_path: str, bpm: float, output_dir: str) -> str:
        """
        Per-category dynamic compression via FFmpeg acompressor.
        - Drums:    30ms attack, 80ms release, 3:1, -20dBFS threshold, +3dB makeup
        - Melodies: 5ms attack, 1/4-note release (BPM-synced), 3:1, -18dBFS threshold
        - Bass/Pad: 20ms attack, 1/2-note release (BPM-synced), 3:1, -18dBFS threshold
        Master bus limiting is deferred until after spatial FX.
        """
        name = os.path.basename(stem_path).lower()
        out_path = os.path.join(output_dir, os.path.basename(stem_path).replace("gs_pan_", "cmp_"))
        if out_path == stem_path:
            base, ext = os.path.splitext(stem_path)
            out_path = base + "_cmp" + ext

        q_ms  = round(60000.0 / bpm)   # quarter note in ms
        hn_ms = q_ms * 2               # half note in ms

        # FFmpeg acompressor: threshold is linear (0-1). -20dBFS = 0.1, -18dBFS = 0.126
        if any(x in name for x in ['kick', 'snare', 'hat', 'clap', 'drum',
                                    'bongo', 'conga', 'tambourine', 'maracas',
                                    'perc', 'instr', 'side_stick']):
            # Drums: punch through with fast attack/release, moderate gain reduction
            filt = "acompressor=threshold=0.1:ratio=3:attack=30:release=80:makeup=1.413:knee=2"
            label = "drums  (30ms att / 80ms rel / 3:1 / +3dB makeup)"

        elif any(x in name for x in ['bass']):
            # Bass: slow breathing release synced to half note
            filt = f"acompressor=threshold=0.126:ratio=3:attack=20:release={hn_ms}:makeup=1.259:knee=3"
            label = f"bass   (20ms att / {hn_ms}ms rel=½note / 3:1)"

        elif any(x in name for x in ['pad', 'chord']):
            # Pads: same slow release as bass for harmonic glue
            filt = f"acompressor=threshold=0.126:ratio=3:attack=20:release={hn_ms}:makeup=1.259:knee=3"
            label = f"pad    (20ms att / {hn_ms}ms rel=½note / 3:1)"

        elif any(x in name for x in ['melody', 'lead', 'counter', 'chorus', 'fx_']):
            # Melodies: fast attack, BPM-synced quarter-note release
            filt = f"acompressor=threshold=0.126:ratio=3:attack=5:release={q_ms}:makeup=1.259:knee=2"
            label = f"melody (5ms att / {q_ms}ms rel=¼note / 3:1)"

        else:
            filt = f"acompressor=threshold=0.126:ratio=3:attack=20:release={q_ms}:makeup=1.259:knee=2"
            label = f"default (20ms att / {q_ms}ms rel / 3:1)"

        print(f"  {os.path.basename(stem_path)[:55]:55s}  [{label}]")
        cmd = ["ffmpeg", "-y", "-i", stem_path, "-af", filt, "-c:a", "pcm_f32le", out_path]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"    Warning: compression failed — copying dry stem")
            shutil.copy2(stem_path, out_path)
        return out_path

    def apply_spatial_fx(self, stem_path: str, bpm: float) -> str:
        """
        Apply BPM-synced reverb + delay per stem category using FFmpeg aecho.
        Compensates for Fantom's MFX/Reverb being absent on individual USB outputs.
        """
        name = os.path.basename(stem_path).lower()
        output_path = stem_path.replace("pan_", "fx_")

        # Avoid collision if pan_ not in path
        if output_path == stem_path:
            base, ext = os.path.splitext(stem_path)
            output_path = base + "_fx" + ext

        # BPM-synced note values in ms: 1/4, 1/8, 1/4T, 1/8T
        q_ms  = round(60000.0 / bpm)
        e_ms  = round(q_ms / 2)
        qt_ms = round(q_ms * 2 / 3)   # quarter triplet
        et_ms = round(e_ms * 2 / 3)   # eighth triplet

        # Bass and kick: keep dry and tight — no reverb
        if any(x in name for x in ['bass', 'kick']):
            shutil.copy2(stem_path, output_path)
            return output_path

        # Percussion (snare, hat, etc.): small room reverb only, no delay
        if any(x in name for x in ['drum', 'snare', 'hat', 'perc', 'clap',
                                     'bongo', 'conga', 'tambourine', 'maracas']):
            fx = "aecho=0.8:0.3:25|55:0.15|0.07"

        # Pads and chords: large hall reverb only (pads have inherent sustain)
        elif any(x in name for x in ['pad', 'chord', 'string', 'choir']):
            fx = "aecho=0.8:0.55:20|60|120|200:0.5|0.35|0.2|0.08"

        # Melodies and leads: reverb + randomised BPM-synced delay
        elif any(x in name for x in ['melody', 'lead', 'chorus', 'counter', 'fx_']):
            delay_ms = random.choice([q_ms, e_ms, qt_ms, et_ms])
            wet      = round(random.uniform(0.10, 0.40), 2)
            feedback = round(random.uniform(0.07, 0.20), 2)
            note_label = {q_ms: '1/4', e_ms: '1/8', qt_ms: '1/4T', et_ms: '1/8T'}[delay_ms]
            print(f"    {os.path.basename(stem_path)}: delay={note_label} ({delay_ms}ms) wet={wet} fb={feedback}")
            reverb_fx = "aecho=0.8:0.5:15|45|85:0.4|0.25|0.12"
            delay_fx  = f"aecho=1.0:{wet}:{delay_ms}:{feedback}"
            fx = f"{reverb_fx},{delay_fx}"

        # Default: light reverb
        else:
            fx = "aecho=0.8:0.45:25|65:0.35|0.18"

        cmd = ["ffmpeg", "-y", "-i", stem_path, "-af", fx, "-c:a", "pcm_f32le", output_path]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"  Warning: spatial FX failed for {name}, using dry stem")
            shutil.copy2(stem_path, output_path)
        return output_path

    def compute_pan_positions(self, stem_names, mode='standard'):
        """
        Distribute stems across the full stereo field with no overlaps.
        - Bass / Kick: always centre (0.0)
        - Pads / Chorus: M/S widening (full-width, no pan offset)
        - Everything else: evenly spaced across -1.0 → +1.0, shuffled randomly
        
        mode='wall': wider distribution, more tracks set to 'wide'
        Returns dict {stem_name: ('center'|'wide'|'pan', value)}
        """
        center_names, wide_names, pan_names = [], [], []
        for name in stem_names:
            n = name.lower()
            if any(x in n for x in ['bass', 'kick']):
                center_names.append(name)
            elif any(x in n for x in ['pad', 'chord', 'chorus']) or (mode == 'wall' and any(x in n for x in ['melod', 'counter'])):
                wide_names.append(name)
            else:
                pan_names.append(name)

        # Evenly space across full -1.0 to +1.0, nudge any exact-zero away from centre
        count = len(pan_names)
        if count == 1:
            positions = [random.choice([-1.0, 1.0])]
        elif count > 1:
            positions = [-1.0 + 2.0 * i / (count - 1) for i in range(count)]
            positions = [p + 0.08 if abs(p) < 0.05 else p for p in positions]
        else:
            positions = []

        random.shuffle(pan_names)  # randomise which stem gets which slot

        pan_map = {}
        for name in center_names:
            pan_map[name] = ('center', 0.0)
        for name in wide_names:
            width = random.uniform(1.8, 2.2) if mode == 'wall' else random.uniform(1.7, 2.0)
            pan_map[name] = ('wide', width)
        for name, pos in zip(pan_names, positions):
            pan_map[name] = ('pan', round(pos, 3))
        return pan_map

    def process_pristine_mix(self, stems: Dict[str, str], song_name: str, bpm: float = 90.0):
        """
        Pristine Mix Pathway:
        - Original recordings (Gain Staged)
        - Wall of Sound Panning
        - Smart Dynamic EQ (Iterative)
        - Glue Compression & Master EQ
        - LUFS Limiting
        """
        print(f"\n--- PRODUCING PRISTINE MIX: {song_name} ---")

        stem_paths = [p for p in stems.values() if p and os.path.exists(p)]
        if not stem_paths:
            print("  No stems found — skipping pristine mix.")
            return None

        # 1. Gain Staging & Wall of Sound Panning
        print("[Pristine Step 1] Gain Staging & Wall Panning...")
        processed_dir = os.path.join(self.output_dir, "pristine_processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        stem_names = [os.path.basename(p) for p in stem_paths]
        pan_map = self.compute_pan_positions(stem_names, mode='wall')
        
        processed_paths = []
        for path in stem_paths:
            name = os.path.basename(path)
            out_path = os.path.join(processed_dir, "pristine_" + name)

            # A. Load original recordings
            y, sr = sf.read(path)
            y = self._ensure_stereo(np.asarray(y, dtype=np.float32))

            # B. Wall of Sound Panning FIRST
            # (Panning can change perceived and actual loudness)
            y = self.apply_panning(y, name, pan_map[name])

            # C. Gain Stage AFTER Panning
            import pyloudnorm as pyln
            meter = pyln.Meter(sr)
            current_lufs = meter.integrated_loudness(y)
            
            if current_lufs <= -70.0: continue
            
            target_lufs = self._get_stem_lufs_target(name)
            gain_db = target_lufs - current_lufs
            
            # Apply gain
            print(f"    Gain Staging: {current_lufs:+.1f} LUFS -> {target_lufs:+.1f} LUFS ({gain_db:+.1f} dB)")
            y *= (10 ** (gain_db / 20))
            
            # Save panned and gain-staged audio
            sf.write(out_path, y, sr, subtype='FLOAT')
            
            # D. Smart Dynamic EQ (Iterative)
            print(f"    Smart EQ shaping: {name}")
            self.apply_professional_eq_shaping(out_path, name)
            
            processed_paths.append(out_path)
            gc.collect()

        # 2. Summing
        print("[Pristine Step 2] Summing Stems...")
        sum_wav = self.sum_stems(processed_paths, song_name, suffix="pristine_sum")
        
        # 3. Mastering Chain (Glue + Master EQ + Limiter)
        print("[Pristine Step 3] Pristine Mastering...")
        output_path = os.path.join(self.output_dir, f"{song_name}_pristine-mix.wav")
        
        y, sr = sf.read(sum_wav)
        y = self._ensure_stereo(np.asarray(y, dtype=np.float32))

        # Glue compression (mild cohesion) — threshold adapts for 2–4 dB GR
        print("  Applying mild glue compression...")
        glue_thr, glue_gr = self._find_adaptive_threshold(y, sr, ratio=1.5, attack_ms=10, release_ms=150)
        print(f"    Glue: thr={20*np.log10(max(glue_thr,1e-10)):.1f}dB  1.5:1  GR={glue_gr:.1f}dB")
        y = self._numpy_compress(y, threshold=glue_thr, ratio=1.5, attack_ms=10, release_ms=150, sr=sr)

        # Smart Dynamic EQ (Master pass)
        print("  Applying Master Smart EQ...")
        y = self.eq.optimize_to_target(y, sr, "master", max_passes=12)
        
        tmp_master = sum_wav + ".pristine_master.wav"
        sf.write(tmp_master, y, sr, subtype='FLOAT')
        del y
        gc.collect()

        # LUFS limiting with more crest-factor room than the main production master.
        print("  Applying LUFS limiting (-16 LUFS)...")
        filter_str = "loudnorm=I=-16:TP=-1.5:LRA=11"
        _info = sf.info(tmp_master)
        cmd = [
            "ffmpeg", "-y", "-i", tmp_master, "-af", filter_str,
            "-ar", str(_info.samplerate), "-c:a", "pcm_s24le", output_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if os.path.exists(tmp_master): os.remove(tmp_master)
        if os.path.exists(sum_wav): os.remove(sum_wav)
        
        print(f"✓ Pristine Mix Complete: {os.path.basename(output_path)}")
        return output_path

    def apply_panning(self, y, name, pan_entry):
        """
        Apply a pre-computed pan position from compute_pan_positions.
        pan_entry: ('center'|'wide'|'pan', value)
        """
        mode, value = pan_entry

        if y.ndim == 1:
            y = np.stack([y, y], axis=1)

        if mode == 'center':
            mono = np.mean(y, axis=1, keepdims=True)
            return np.hstack([mono, mono])

        if mode == 'wide':
            print(f"    {name}: M/S width ×{value:.2f}")
            return self._widen_stereo(y, value)

        # mode == 'pan'
        pan = value
        print(f"    {name}: pan {pan:+.3f}")
        left_gain  = np.sqrt(0.5 * (1.0 - pan))
        right_gain = np.sqrt(0.5 * (1.0 + pan))
        out = y.copy()
        out[:, 0] *= left_gain
        out[:, 1] *= right_gain
        return out

    def _widen_stereo(self, y, width):
        """Mid/Side stereo widening. width > 1 = wider, < 1 = narrower."""
        mid  = (y[:, 0] + y[:, 1]) * 0.5
        side = (y[:, 0] - y[:, 1]) * 0.5 * width
        out  = np.stack([mid + side, mid - side], axis=1)
        peak = np.max(np.abs(out))
        if peak > 0.99:
            out = out * (0.95 / peak)
        return out

    def apply_global_fx_sends(self, stem_paths: list, bpm: float, output_dir: str) -> list:
        """
        Create shared Master Reverb and Delay buses by summing stem sends.
        Mimics professional DAW Send/Return architecture.
        """
        os.makedirs(output_dir, exist_ok=True)
        q_ms = round(60000.0 / bpm)
        
        # 1. Calculate Send Levels and Sum in Memory
        reverb_sum = None
        delay_sum = None
        sr = 48000
        
        print("  [Global FX] Generating shared Reverb & Delay sends...")
        for path in stem_paths:
            name = os.path.basename(path).lower()
            try:
                y, sr = sf.read(path)
            except Exception: continue
            
            # Professional Send Mapping
            rev_level = 0.0
            dly_level = 0.0
            
            if any(x in name for x in ['kick', 'bass', 'sub']):
                pass # Dry
            elif any(x in name for x in ['snare', 'clap']):
                rev_level = 0.28
            elif 'hat' in name:
                rev_level = 0.12
            elif any(x in name for x in ['pad', 'chord']):
                rev_level = 0.40
            elif any(x in name for x in ['melody', 'lead', 'chorus']):
                rev_level = 0.32
                dly_level = 0.25
            elif 'counter' in name:
                rev_level = 0.38
                dly_level = 0.40
            elif 'fx' in name:
                rev_level = 0.25
                dly_level = 0.18
            else:
                rev_level = 0.15
            
            if rev_level > 0:
                y_rev = y * rev_level
                reverb_sum = y_rev if reverb_sum is None else self._pad_and_add(reverb_sum, y_rev)
            if dly_level > 0:
                y_dly = y * dly_level
                delay_sum = y_dly if delay_sum is None else self._pad_and_add(delay_sum, y_dly)
        
        REVERB_RETURN_GAIN = 10 ** (-10.0 / 20)
        DELAY_RETURN_GAIN  = 10 ** (-12.0 / 20)

        fx_paths = []

        # 2. Process Global Reverb Bus (100% Wet, "Abbey Road" Style)
        if reverb_sum is not None:
            rev_send_path = os.path.join(output_dir, "bus_reverb_send.wav")
            sf.write(rev_send_path, reverb_sum, sr, subtype='FLOAT')

            rev_out_path = os.path.join(output_dir, "bus_reverb_wet.wav")
            # User requested: 1/16 note predelay, 1/2 note tail, 500Hz HPF, High boost
            pre_ms = round(q_ms / 4)
            tail_ms = round(q_ms * 2)

            # Construct dense reverb via cascaded aecho (mimics plate/hall)
            # HPF @ 500Hz, High shelf boost @ 6kHz
            filters = [
                f"adelay={pre_ms}|{pre_ms}", # Pure pre-delay
                f"aecho=0.9:0.8:{tail_ms//4}|{tail_ms//2}|{tail_ms}:0.4|0.3|0.2", # Dense tail
                "highpass=f=500", # Mud cut
                "equalizer=f=8000:width_type=s:width=1:gain=3.5", # High boost
                "stereowiden=delay=20:feedback=0.25:crossfeed=0.25:drymix=0.85" # Extra width
            ]

            cmd = ["ffmpeg", "-y", "-i", rev_send_path, "-af", ",".join(filters), "-c:a", "pcm_f32le", rev_out_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Apply return gain attenuation
            rev_ret_y, rev_ret_sr = sf.read(rev_out_path)
            sf.write(rev_out_path, (rev_ret_y * REVERB_RETURN_GAIN).astype(np.float32), rev_ret_sr, subtype='FLOAT')
            del rev_ret_y
            fx_paths.append(rev_out_path)
            print(f"    ✓ Master Reverb Bus created (Predelay={pre_ms}ms, Tail={tail_ms}ms, return={20*np.log10(REVERB_RETURN_GAIN):.0f}dB)")

        # 3. Process Global Delay Bus (100% Wet)
        if delay_sum is not None:
            dly_send_path = os.path.join(output_dir, "bus_delay_send.wav")
            sf.write(dly_send_path, delay_sum, sr, subtype='FLOAT')

            dly_out_path = os.path.join(output_dir, "bus_delay_wet.wav")
            # Dotted-eighth delay
            d_ms = round(q_ms * 0.75)
            filters = [
                f"aecho=0.8:0.7:{d_ms}:0.4",
                "highpass=f=400",
                "lowpass=f=8000"
            ]
            cmd = ["ffmpeg", "-y", "-i", dly_send_path, "-af", ",".join(filters), "-c:a", "pcm_f32le", dly_out_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Apply return gain attenuation
            dly_ret_y, dly_ret_sr = sf.read(dly_out_path)
            sf.write(dly_out_path, (dly_ret_y * DELAY_RETURN_GAIN).astype(np.float32), dly_ret_sr, subtype='FLOAT')
            del dly_ret_y
            fx_paths.append(dly_out_path)
            print(f"    ✓ Master Delay Bus created ({d_ms}ms, return={20*np.log10(DELAY_RETURN_GAIN):.0f}dB)")

        return fx_paths

    def _pad_and_add(self, a, b):
        """Add two NumPy arrays of potentially different lengths."""
        if len(a) < len(b):
            res = np.zeros_like(b)
            res[:len(a)] = a
            res += b
        else:
            res = np.zeros_like(a)
            res[:len(b)] = b
            res += a
        return res

    def sum_stems(self, stem_paths, song_name, suffix: str = ""):
        """Mix all stems into a single stereo file."""
        s_part = f"_{suffix}" if suffix else ""
        output_path = os.path.join(self.output_dir, f"{song_name}{s_part}_mix.wav")
        
        # Filter out any non-existent paths
        valid_paths = [p for p in stem_paths if os.path.exists(p)]
        if not valid_paths:
            print("  ! ERROR: No valid stems to sum!")
            return None

        print(f"  Summing {len(valid_paths)} stems into final mix...")
        mix, sr = sf.read(valid_paths[0], always_2d=True)
        mix = mix.astype(np.float64)
        for p in valid_paths[1:]:
            y, _ = sf.read(p, always_2d=True)
            y = y.astype(np.float64)
            mix = self._pad_and_add(mix, y)
        peak = np.max(np.abs(mix))
        if peak > self.mix_headroom_peak:
            mix *= self.mix_headroom_peak / peak
        sf.write(output_path, mix.astype(np.float32), sr, subtype='FLOAT')
        return output_path


if __name__ == "__main__":
    pass
