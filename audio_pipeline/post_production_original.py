import os
import random
import shutil
import gc
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

        # 1. Pre-Analysis Pass
        print("[Step 1] Pre-Analysis (Loudness & Spectral)...")
        pink_ref = self.stager.generate_pink_noise()
        target_lufs = self.stager.get_integrated_lufs(pink_ref)
        
        # Build processing map
        stem_names = [os.path.basename(p) for p in stem_paths]
        pan_map = self.compute_pan_positions(stem_names)
        
        # 2. Consolidated Processing Pass (Serial, 32-bit Float)
        print("[Step 2] Consolidated Processing Pass (Gain + Pan + Dynamics + EQ + FX)...")
        processed_dir = os.path.join(self.output_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        processed_paths = []
        for path in stem_paths:
            name = os.path.basename(path)

            # A. Gain Staging (Initial match to target)
            current_lufs = self.stager.get_integrated_lufs(path)
            if current_lufs <= -70.0:
                print(f"  {name:48s}  [SILENT] skipping")
                continue
            gain_db = target_lufs - current_lufs

            # B. Build the filter chain
            filters = []
            log = []

            # Initial Volume Adjustment
            filters.append(f"volume={gain_db}dB")
            log.append(f"gain:{gain_db:+.1f}dB")

            # Dynamic Processing (Compression)
            comp_filt, comp_log = self._get_dynamic_filters(name, bpm)
            filters.append(comp_filt)
            log.append(comp_log)

            # Harmonic Enhancement & Phase 3 EQ
            hrm_filt, hrm_log = self._get_harmonic_filters(name)
            if hrm_filt:
                filters.append(hrm_filt)
                log.append(hrm_log)

            # Creative Processing
            crt_filters, crt_log = self._get_creative_filters(name, bpm)
            filters.extend(crt_filters)
            log.extend(crt_log)

            # Spatial FX: DEPRECATED track-level (Moved to Step 4.5 Global Sends)
            spat_filt, spat_log = None, None

            # Output setup
            out_path = os.path.join(processed_dir, "proc_" + name)

            # C. Execute Single-Pass FFmpeg (Dry only)
            cmd = ["ffmpeg", "-y", "-i", path, "-af", ",".join(filters), "-c:a", "pcm_f32le", out_path]

            print(f"  {name:48s}  {' | '.join(log)}")
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if res.returncode != 0:
                print(f"    ! FFmpeg failed, copying dry: {res.stderr.decode()[:100]}")
                shutil.copy2(path, out_path)

            # D. Panning & Final Leveling (via NumPy to ensure precision)
            y, sr = sf.read(out_path)
            y = self.apply_panning(y, name, pan_map[name])
            sf.write(out_path, y, sr, subtype='FLOAT')
            del y
            gc.collect()

            # E. Apply Professional EQ Shaping (iterative, frequency-aware)
            print(f"  Applying professional EQ shaping to {name}...")
            out_path = self.apply_professional_eq_shaping(out_path, name)

            processed_paths.append(out_path)
            gc.collect()

        # 3. Automation Pass (Serial, NumPy, 32-bit Float)
        print("[Step 3] Intro/Outro & Phrase Automations...")
        automation_dir = os.path.join(self.output_dir, "automated")
        os.makedirs(automation_dir, exist_ok=True)
        
        # We process each stem for intro/outro and phrase FX
        automated_paths = self.apply_intro_outro_automation(processed_paths, bpm, automation_dir)
        final_processed_paths = self.apply_phrase_automation_fx(automated_paths, bpm, automation_dir)

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

        # 5.5 Create Mix Variants (Minimal, Bass1, Bass2)
        print("[Step 5.5] Creating Mix Variants...")
        
        variants = [
            {"suffix": "minimal", "omit": "_Main_Melody.wav",    "label": "Minimal Mix (No Verse Melody)"},
            {"suffix": "bass1",    "omit": "_Harmonic_Bass.wav", "label": "Bass 1 Mix (No Bass 2)"},
            {"suffix": "bass2",    "omit": "_Bass.wav",          "label": "Bass 2 Mix (No Bass 1)"}
        ]

        for var in variants:
            # Filter stems for this variant
            # We use endswith to ensure exact track type matching
            v_stems = [p for p in final_processed_paths if not p.endswith(var['omit'])]
            v_stems.extend(fx_bus_paths)
            
            print(f"  Generating {var['label']}...")
            v_mix = self.sum_stems(v_stems, song_name, suffix=var['suffix'])
            if v_mix:
                self.apply_mastering(v_mix, song_name, suffix=var['suffix'])

        # 6. SP-404 Remix
        print("[Step 6] Creating SP-404 Performance Remix...")
        remix_path = self.apply_404_remix(master_path, song_name, bpm)

        return master_path, remix_path

    def _get_dynamic_filters(self, name: str, bpm: float):
        name = name.lower()
        q_ms = round(60000.0 / bpm)
        if any(x in name for x in ['kick', 'snare', 'hat', 'clap', 'drum']):
            # Subtle drum glue (Ratio 2:1, higher threshold -12dB)
            return "acompressor=attack=15:release=80:ratio=2:threshold=0.25:makeup=1.15", "drum-comp-2:1"
        elif any(x in name for x in ['bass', 'pad', 'chord']):
            rel = round(q_ms * 2)
            # Gentle leveling for bass/pads (Ratio 1.5:1)
            return f"acompressor=attack=30:release={rel}:ratio=1.5:threshold=0.3:makeup=1.1", "bass/pad-comp"
        else:
            # Transparent catch for melodies (Ratio 1.5:1)
            return f"acompressor=attack=10:release={q_ms}:ratio=1.5:threshold=0.25:makeup=1.1", "melody-comp"

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

        if self._is_drum_name(name):
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
            info = sf.info(path)
            if info.frames < 2048:
                print(f"  Skipping short stem: {name} ({info.frames} frames)")
                opt_path = os.path.join(bus_dir, "opt_" + os.path.basename(path))
                shutil.copy2(path, opt_path)
                optimized_paths.append(opt_path)
                continue
            y, sr = sf.read(path)
            # Use the new objective optimization system for all stems
            y = self.eq.optimize_to_target(y, sr, name, max_passes=8)

            opt_path = os.path.join(bus_dir, "opt_" + os.path.basename(path))
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
            # Soft peak limiter: only attenuate if true clipping would occur
            peak = np.max(np.abs(mix))
            if peak > 1.0:
                mix /= peak
            sf.write(out_path, mix.astype(np.float32), sr, subtype='FLOAT')
            bus_sums[bus_name] = out_path

        # 3. Intelligent Carving & Monitoring
        print("  [Monitoring] Analyzing Bus Spectral Health:")
        for bus_name, path in bus_sums.items():
            y, sr = sf.read(path)
            # Low (20-150), Mid (150-2.5k), High (2.5k-20k)
            lows = SpectralAnalyzer.detect_buildup(y, sr, (20, 150))
            mids = SpectralAnalyzer.detect_buildup(y, sr, (150, 2500))
            his  = SpectralAnalyzer.detect_buildup(y, sr, (2500, 20000))
            print(f"    - {bus_name:8s}: Lows={lows:.2f}, Mids={mids:.2f}, Highs={his:.2f}")

        # Melody vs Drums (clear space for snare/clarity)
        if "melody" in bus_sums and "drums" in bus_sums:
            m_y, sr = sf.read(bus_sums["melody"])
            d_y, _ = sf.read(bus_sums["drums"])
            # Carve snare crack (2.5kHz) from melody
            m_y = self.eq.intelligent_carve(m_y, d_y, sr, (2000, 3000), depth=-2.5)
            # Carve mud (300Hz) from melody if drums are heavy there
            m_y = self.eq.intelligent_carve(m_y, d_y, sr, (200, 450), depth=-3.0)
            sf.write(bus_sums["melody"], m_y, sr, subtype='FLOAT')

        # Melody vs Bass
        if "melody" in bus_sums and "bass" in bus_sums:
            m_y, sr = sf.read(bus_sums["melody"])
            b_y, _ = sf.read(bus_sums["bass"])
            # Carve bass fundamental range from melody to avoid mud
            m_y = self.eq.intelligent_carve(m_y, b_y, sr, (60, 250), depth=-4.0)
            sf.write(bus_sums["melody"], m_y, sr, subtype='FLOAT')

        # Bass vs Drums (Kick/Bass Slotting)
        if "bass" in bus_sums and "drums" in bus_sums:
            b_y, sr = sf.read(bus_sums["bass"])
            d_y, _ = sf.read(bus_sums["drums"])
            b_y = self.eq.apply_frequency_slotting(b_y, d_y, sr)
            sf.write(bus_sums["bass"], b_y, sr, subtype='FLOAT')

        # 4. Apply Professional Bus EQ Shaping (iterative, frequency-aware)
        print("  [Professional Bus EQ Shaping]...")
        for bus_name, path in bus_sums.items():
            print(f"    Optimizing {bus_name} bus spectral profile...")
            y, sr = sf.read(path)
            y = self._ensure_stereo(np.asarray(y, dtype=np.float32))
            
            # Use the in-memory optimizer
            y_opt = self.eq.optimize_to_target(y, sr, bus_name, max_passes=8)
            
            sf.write(path, y_opt, sr, subtype='FLOAT')
            del y
            del y_opt
            gc.collect()

        return list(bus_sums.values())

    def apply_mastering(self, input_wav, song_name, suffix: str = ""):
        """Mastering chain including Mid/Side mono-maker, Objective EQ, and PLR-targeted limiting."""
        s_part = f"_{suffix}" if suffix else ""
        output_path = os.path.join(self.output_dir, f"{song_name}{s_part}_master.wav")
        
        y, sr = sf.read(input_wav)
        y = self._ensure_stereo(np.asarray(y, dtype=np.float32))

        # 1. Mid/Side Mono-Maker (150Hz)
        print("  Focusing low-end mono...")
        mid = (y[:, 0] + y[:, 1]) * 0.5
        side = (y[:, 0] - y[:, 1]) * 0.5
        # Side HPF = side - side_lpf
        side_hpf = side - self._lowpass_numpy(side, sr, 150.0)
        y[:, 0] = mid + side_hpf
        y[:, 1] = mid - side_hpf

        # 2. Objective Mastering EQ (in-memory)
        print("  Applying Objective Mastering EQ...")
        y = self.eq.optimize_to_target(y, sr, "master", max_passes=12)
        
        tmp_master = input_wav + ".master_eq.wav"
        sf.write(tmp_master, y, sr, subtype='FLOAT')
        del y
        gc.collect()

        # 3. Professional Master Dynamics (-14 LUFS, PLR 10-12)
        # Using a soft clipper and limiter chain
        print("  Applying final polish & limiter...")
        filter_str = (
            "asmooth=size=3," # Gentle smoothing
            "acolorspace=all=bt709," # Not relevant for audio but placeholder for color logic if needed
            "alimiter=limit=0.95:level=1:attack=5:release=50,"
            "loudnorm=I=-14:TP=-1.0:LRA=7"
        )
        # Cleaned up filter string
        filter_str = "alimiter=limit=0.9:level=1:attack=5:release=50:level_in=1, loudnorm=I=-14:TP=-1.0:LRA=7"
        
        cmd = [
            "ffmpeg", "-y",
            "-i", tmp_master,
            "-af", filter_str,
            "-c:a", "pcm_f32le",
            output_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
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
                reverb_wet    = round(random.uniform(0.20, 0.30), 2)
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
                reverb_wet    = round(random.uniform(0.25, 0.35), 2)
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

                # Tape de-emphasis warmth (25%)
                if random.random() < 0.25:
                    filters.append("aemphasis=level_in=1:level_out=1:mode=reproduction:type=75fm")
                    log.append("tape-deemph")

            # ── COUNTER MELODY — heavy processing ALWAYS ───────────
            elif 'counter' in name:
                # Cathedral reverb ALWAYS (via parallel blend, replaces old inline aecho reverb)
                reverb_filter = melody_reverbs['cathedral']
                reverb_wet    = 0.35
                reverb_label  = "cathedral"

                # Dotted-quarter delay ALWAYS
                d_wet = round(random.uniform(0.4, 0.65), 2)
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
                reverb_wet    = round(random.uniform(0.25, 0.35), 2)
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

    def apply_404_remix(self, input_wav, song_name, bpm):
        """Apply aggressive master FX every 4th bar (SP-404 style)."""
        output_path = os.path.join(self.output_dir, f"{song_name}_404_remix.wav")
        
        bar_dur = (60.0 / bpm) * 4
        four_bars = bar_dur * 4
        
        # Define 4-bar effect sequence
        # We apply the effect for the duration of the 4th bar of each cycle
        
        # Effect 1: Bitcrush (Bar 4)
        fx1 = f"acrusher=level_in=1:level_out=1:bits=8:mode=log:aa=1:enable='between(mod(t,{four_bars}),{bar_dur*3},{bar_dur*4})'"
        
        # Effect 2: Resonant LP hit (Bar 8). Keep this expression-free for
        # FFmpeg 8.x compatibility; lowpass supports timeline enable but not
        # arbitrary per-sample cutoff expressions in this build.
        t1 = bar_dur * 7
        t2 = bar_dur * 8
        fx2 = f"lowpass=f=2400:width_type=q:width=3:enable='between(mod(t,{four_bars*2}),{t1},{t2})'"
        
        # Effect 3: Flanger / Phased Stutter (Bar 12)
        fx3 = f"vibrato=f=10:d=0.5:enable='between(mod(t,{four_bars*3}),{bar_dur*11},{bar_dur*12})'"
        
        # Effect 4: Vinyl Sim / Extreme Pumping (Bar 16)
        # acompressor does not support timeline enable on FFmpeg 8.x, so use
        # timeline-capable tremolo for the performance-style pump.
        pump_hz = round(max(1.0, bpm / 30.0), 3)
        fx4 = f"tremolo=f={pump_hz}:d=0.75:enable='between(mod(t,{four_bars*4}),{bar_dur*15},{bar_dur*16})'"

        filter_str = f"{fx1},{fx2},{fx3},{fx4}"
        
        cmd = [
            "ffmpeg", "-y",
            "-i", input_wav,
            "-af", filter_str,
            output_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"  Warning: 404 remix failed — {result.stderr.decode(errors='replace')[-300:]}")
            return None
        return output_path

    def compute_pan_positions(self, stem_names):
        """
        Distribute stems across the full stereo field with no overlaps.
        - Bass / Kick: always centre (0.0)
        - Pads / Chorus: M/S widening (full-width, no pan offset)
        - Everything else: evenly spaced across -1.0 → +1.0, shuffled randomly
        Returns dict {stem_name: ('center'|'wide'|'pan', value)}
        """
        center_names, wide_names, pan_names = [], [], []
        for name in stem_names:
            n = name.lower()
            if any(x in n for x in ['bass', 'kick']):
                center_names.append(name)
            elif any(x in n for x in ['pad', 'chord', 'chorus']):
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
            pan_map[name] = ('wide', random.uniform(1.7, 2.0))
        for name, pos in zip(pan_names, positions):
            pan_map[name] = ('pan', round(pos, 3))
        return pan_map

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
            fx_paths.append(rev_out_path)
            print(f"    ✓ Master Reverb Bus created (Predelay={pre_ms}ms, Tail={tail_ms}ms)")

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
            fx_paths.append(dly_out_path)
            print(f"    ✓ Master Delay Bus created ({d_ms}ms)")

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
        if peak > 1.0:
            mix /= peak
        sf.write(output_path, mix.astype(np.float32), sr, subtype='FLOAT')
        return output_path


if __name__ == "__main__":
    pass
