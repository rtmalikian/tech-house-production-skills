"""
Bus Automation Engine — Intro/outro, verse transitions, chorus, fill.

Applies arrangement-aware automation to bus sums (drums, bass, melody).
1 effect per bus at a time. At verse transitions, only 1 bus gets an effect.
"""

import os
import sys
import json
import random
import itertools
import re
import numpy as np
import soundfile as sf

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

import config
from dsp_engine import DSPEngine

DRAMATIC_EFFECTS = ['impact', 'riser', 'beat_repeat', 'stutter', 'silence_drop',
                    'tape_stop', 'vinyl_stop', 'distortion_build', 'bitcrush_sweep',
                    'filter_drop', 'reverb_freeze', 'hpf_sweep', 'lpf_sweep',
                    'hpf_on_off', 'lpf_on_off']


class BusAutomationEngine:
    """Bus-level arrangement automation."""

    def __init__(self, dsp: DSPEngine = None):
        self.dsp = dsp or DSPEngine()
        self.sr = None

    def _apply_transition_gain_bump(self, y, start, end, bump_db=None):
        """Add a gain bump at transition start, fading back over 500ms."""
        if bump_db is None:
            bump_db = config.EFFECT_DEPTH.get('transition_gain_bump_db', 3.0)
        sr = self.sr or 44100
        bump_len = min(int(sr * 0.5), (end - start) // 4)
        if bump_len <= 0:
            return y
        bump_curve = np.ones(end - start, dtype=np.float32)
        bump_lin = 10.0 ** (bump_db / 20.0)
        bump_curve[:bump_len] = np.linspace(bump_lin, 1.0, bump_len)
        y[start:end] *= bump_curve[:, np.newaxis]
        return y

    def apply_bus_automation(self, bus_paths: dict, sr: int, bpm: float,
                             output_dir: str, song_name: str = None) -> dict:
        """
        Apply automation to bus sums.

        Args:
            bus_paths: dict of {bus_name: path} e.g. {'drums': 'bus_drums.wav', ...}
            sr: sample rate
            bpm: beats per minute
            output_dir: directory for automated output

        Returns:
            dict of {bus_name: automated_path}
        """
        os.makedirs(output_dir, exist_ok=True)
        beats_per_bar = _beats_per_bar_from_song_name(song_name)
        samples_per_bar = int(sr * 60.0 / bpm * beats_per_bar)

        # Load all buses
        buses = {}
        for bus_name, path in bus_paths.items():
            if path and os.path.exists(path):
                y, file_sr = sf.read(path, always_2d=True)
                buses[bus_name] = {
                    'y': _ensure_stereo(np.asarray(y, dtype=np.float32)),
                    'sr': file_sr,
                    'path': path,
                }

        if not buses:
            return bus_paths

        sr = buses[list(buses.keys())[0]]['sr']
        self.sr = sr

        # Detect song sections
        total_samples = max(len(b['y']) for b in buses.values())
        total_bars = max(1, int(np.ceil(total_samples / samples_per_bar)))

        # Build automation plan
        plan = self._build_automation_plan(total_bars, samples_per_bar, bpm, buses.keys())

        # Log plan to song's log/ directory
        log_dir = os.path.join(os.path.dirname(output_dir), "log")
        self._log_plan(plan, log_dir, bpm, total_bars, samples_per_bar)

        # Apply automation to each bus
        output_paths = {}
        for bus_name, bus_data in buses.items():
            y = bus_data['y'].copy()
            applied = []

            for event in plan:
                effect = event.get('effects', {}).get(bus_name)
                if not effect:
                    continue

                start = event.get('start_sample', 0)
                end = event.get('end_sample', len(y))
                onset = event.get('onset', 'instant')

                if start >= len(y):
                    continue
                end = min(end, len(y))

                y = self._apply_transition_gain_bump(y, start, end)
                y = self._apply_effect(y, sr, bpm, effect, start, end, bus_name, onset=onset)
                applied.append(f"{event['name']}:{effect}")

            # Save
            # Apply drum bus processing after automation
            if bus_name == 'drums':
                y = self.dsp.parallel_compress(y, sr,
                    ratio=config.DRUM_PARALLEL_COMP['ratio'],
                    attack_ms=config.DRUM_PARALLEL_COMP['attack_ms'],
                    release_ms=config.DRUM_PARALLEL_COMP['release_ms'],
                    blend=config.DRUM_PARALLEL_COMP['blend'])

                # Transient shaping — boost kick/snare attack
                y_transient, y_sustain = self.dsp.transient_sustain_split(y, sr)
                transient_boost_db = 4.0
                y = y + y_transient * (10.0 ** (transient_boost_db / 20.0) - 1.0)

                # Presence EQ — snare/hat attack + kick thump
                y = self.dsp.peaking_filter(y, sr, center_freq=3500, gain_db=2.5, q=1.0)
                y = self.dsp.peaking_filter(y, sr, center_freq=70, gain_db=1.5, q=1.0)

                # Dynamic soft clip — adapts to local peak level
                clip_cfg = config.DRUM_DYNAMIC_SOFT_CLIP
                y = self.dsp.dynamic_soft_clip(y, sr,
                    headroom_db=clip_cfg['bus_headroom_db'],
                    block_ms=clip_cfg['block_ms'])

            out_path = os.path.join(output_dir, f"auto_{bus_name}.wav")
            sf.write(out_path, y, sr, subtype='FLOAT')
            output_paths[bus_name] = out_path

            short = bus_name[:20]
            print(f"  {short:20s}  {' | '.join(applied) if applied else '[bypass]'}")

        return output_paths

    # ── Intensity Envelope System ────────────────────────────────────

    RAMPING_EFFECTS = {'reverb_wash', 'reverb_throw', 'delay_feedback_swell',
                       'chorus_swell', 'riser', 'distortion_build', 'bitcrush_sweep',
                       'stereo_widen', 'stereo_narrow', 'gain_fade', 'resonance_sweep'}

    FIXED_EFFECTS = {'tape_wobble', 'vinyl_crackle', 'sidechain_pump', 'hpf_sweep',
                     'lpf_sweep', 'hpf_on_off', 'lpf_on_off', 'filter_drop',
                     'tape_stop', 'vinyl_stop', 'tape_start', 'beat_repeat',
                     'stutter', 'silence_drop', 'reverb_freeze', 'impact',
                     'tremolo', 'phaser_sweep', 'isolate_low', 'isolate_mid',
                     'isolate_high'}

    def _log_plan(self, plan: list, log_dir: str, bpm: float,
                  total_bars: int, samples_per_bar: int):
        """Write automation plan to log/ as JSON and readable text."""
        os.makedirs(log_dir, exist_ok=True)

        # Convert sample positions to bar numbers for readability
        def sample_to_bar(s):
            return round(s / samples_per_bar, 1)

        # JSON log
        json_events = []
        for event in plan:
            json_events.append({
                'name': event['name'],
                'section': event.get('section', ''),
                'start_bar': sample_to_bar(event.get('start_sample', 0)),
                'end_bar': sample_to_bar(event.get('end_sample', 0)),
                'effects': event.get('effects', {}),
                'onset': event.get('onset', 'instant'),
                'anchor_bus': event.get('anchor_bus'),
            })
        json_data = {
            'bpm': bpm,
            'total_bars': total_bars,
            'samples_per_bar': samples_per_bar,
            'events': json_events,
        }
        json_path = os.path.join(log_dir, "automation_plan.json")
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        # Readable text log
        txt_path = os.path.join(log_dir, "automation_plan.txt")
        with open(txt_path, 'w') as f:
            f.write(f"=== AUTOMATION PLAN ===\n")
            f.write(f"BPM: {bpm} | Total Bars: {total_bars}\n\n")
            for event in plan:
                name = event['name']
                section = event.get('section', '')
                start = sample_to_bar(event.get('start_sample', 0))
                end = sample_to_bar(event.get('end_sample', 0))
                onset = event.get('onset', 'instant')
                effects = event.get('effects', {})
                f.write(f"{name.upper()} (bars {start}-{end}) [{onset}]\n")
                if event.get('anchor_bus'):
                    f.write(f"  anchor: {event['anchor_bus']} [protected]\n")
                for bus, effect in effects.items():
                    f.write(f"  {bus}: {effect}\n")
                f.write("\n")

        print(f"  Automation plan logged: {json_path}")

    def _build_intensity_envelopes(self, seg_len: int, sr: int, bpm: float,
                                    onset: str) -> tuple:
        """Build depth and mix envelopes for effect ramping."""
        if onset == 'instant':
            depth = np.ones(seg_len, dtype=np.float32)
            mix = np.ones(seg_len, dtype=np.float32)

        elif onset == 'quarter_note':
            quarter_samples = int(sr * 60.0 / bpm)
            ramp_len = min(quarter_samples, seg_len)
            depth = np.ones(seg_len, dtype=np.float32)
            mix = np.ones(seg_len, dtype=np.float32)
            depth[:ramp_len] = np.linspace(0.0, 1.0, ramp_len)
            mix[:ramp_len] = np.linspace(0.0, 1.0, ramp_len)

        elif onset == 'rapid':
            peak_pos = max(1, int(seg_len * 0.2))
            depth = np.ones(seg_len, dtype=np.float32)
            mix = np.ones(seg_len, dtype=np.float32)
            depth[:peak_pos] = np.linspace(0.0, 1.0, peak_pos) ** 0.3
            mix[:peak_pos] = np.linspace(0.0, 1.0, peak_pos) ** 0.3

        elif onset == 'two_note':
            two_beat_samples = int(sr * 60.0 / bpm * 2)
            ramp_len = min(two_beat_samples, seg_len)
            depth = np.ones(seg_len, dtype=np.float32)
            mix = np.ones(seg_len, dtype=np.float32)
            depth[:ramp_len] = np.linspace(0.0, 1.0, ramp_len)
            mix[:ramp_len] = np.linspace(0.0, 1.0, ramp_len)

        else:
            depth = np.ones(seg_len, dtype=np.float32)
            mix = np.ones(seg_len, dtype=np.float32)

        return depth, mix

    def _build_automation_plan(self, total_bars: int, samples_per_bar: int,
                                bpm: float, available_buses=None) -> list:
        """Build the full automation plan with exact bar positions.
        
        One bus per transition, rotating through available musical buses.
        Other buses stay dry (no automation effects, reverb/delay sends still active).
        """
        plan = []
        available = list(available_buses or ['drums', 'bass', 'pads', 'melody'])
        cycle_buses = [b for b in ['drums', 'bass', 'pads', 'melody'] if b in available]
        if not cycle_buses:
            cycle_buses = ['drums', 'bass', 'melody']
        bus_cycle = itertools.cycle(cycle_buses)
        intro_anchor = self._choose_anchor_bus(available)
        outro_anchor = self._choose_anchor_bus(available)
        if intro_anchor is None or outro_anchor is None:
            print("  [Automation] WARNING: no pads/melody bus found for intro/outro anchor protection")

        # INTRO — anchor stays stable while rhythm/low-end rise toward verse start.
        intro_end = min(8, total_bars)
        effects = {}
        for bus in cycle_buses:
            if bus == intro_anchor:
                continue
            if bus in {'drums', 'bass'}:
                effects[bus] = 'gain_rise'
            else:
                effects[bus] = random.choice(['gain_rise', 'hpf_sweep', 'reverb_wash'])
        plan.append({
            'name': 'intro',
            'section': 'intro',
            'start_sample': 0,
            'end_sample': intro_end * samples_per_bar,
            'effects': effects,
            'onset': 'instant',
            'anchor_bus': intro_anchor,
        })

        # VERSE SECTIONS — 1 bus per transition, rotating
        for section_name in ['verse1', 'verse2']:
            if section_name not in config.SONG_SECTIONS:
                continue
            verse_start_bar, verse_end_bar = config.SONG_SECTIONS[section_name]
            if verse_start_bar >= total_bars:
                continue

            for trans_bar in config.VERSE_TRANSITION_BARS:
                abs_bar = verse_start_bar + trans_bar
                if abs_bar >= total_bars or abs_bar >= verse_end_bar:
                    continue

                active_bus = self._next_automated_bus(bus_cycle, available)
                effects = {}
                palette = config.BUS_EFFECT_PALETTES.get(active_bus, {}).get('verse', [])
                if palette:
                    effects[active_bus] = random.choice(palette)

                if not effects:
                    continue

                start_sample = abs_bar * samples_per_bar
                duration_bars = random.choice([1, 2])
                end_sample = min((abs_bar + duration_bars) * samples_per_bar,
                                 verse_end_bar * samples_per_bar)

                plan.append({
                    'name': f'{section_name}_bar{trans_bar}',
                    'section': 'verse',
                    'start_sample': start_sample,
                    'end_sample': end_sample,
                    'effects': effects,
                    'onset': random.choices(
                        ['quarter_note', 'rapid', 'instant'],
                        weights=[50, 30, 20], k=1)[0],
                })

        # PRE-CHORUS BUILDS — 1 bus per build, rotating
        for section_name in ['chorus1', 'chorus2']:
            if section_name not in config.SONG_SECTIONS:
                continue
            chorus_start_bar = config.SONG_SECTIONS[section_name][0]
            build_start = chorus_start_bar - 4
            if build_start < 0 or build_start >= total_bars:
                continue

            active_bus = self._next_automated_bus(bus_cycle, available)
            effects = {}
            palette = config.BUILD_EFFECTS.get(active_bus, [])
            if palette:
                effects[active_bus] = random.choice(palette)

            plan.append({
                'name': f'pre_{section_name}_build',
                'section': 'build',
                'start_sample': build_start * samples_per_bar,
                'end_sample': chorus_start_bar * samples_per_bar,
                'effects': effects,
                'onset': 'two_note',
            })

        # CHORUS SECTIONS — 1 bus per chorus, rotating
        for section_name in ['chorus1', 'chorus2']:
            if section_name not in config.SONG_SECTIONS:
                continue
            chorus_start_bar, chorus_end_bar = config.SONG_SECTIONS[section_name]
            if chorus_start_bar >= total_bars:
                continue

            active_bus = self._next_automated_bus(bus_cycle, available)
            effects = {}
            palette = config.BUS_EFFECT_PALETTES.get(active_bus, {}).get('chorus', [])
            if palette:
                effects[active_bus] = random.choice(palette)

            plan.append({
                'name': section_name,
                'section': 'chorus',
                'start_sample': chorus_start_bar * samples_per_bar,
                'end_sample': chorus_end_bar * samples_per_bar,
                'effects': effects,
                'onset': random.choices(
                    ['quarter_note', 'rapid', 'instant'],
                    weights=[50, 30, 20], k=1)[0],
            })

        # FILL SECTIONS — 1 bus per fill, rotating
        for section_name in ['fill1', 'fill2']:
            if section_name not in config.SONG_SECTIONS:
                continue
            fill_start_bar, fill_end_bar = config.SONG_SECTIONS[section_name]
            if fill_start_bar >= total_bars:
                continue

            active_bus = self._next_automated_bus(bus_cycle, available)
            effects = {}
            palette = config.BUS_EFFECT_PALETTES.get(active_bus, {}).get('fill', [])
            if palette:
                effects[active_bus] = random.choice(palette)

            fade_end = min(fill_start_bar + 3, fill_end_bar)
            plan.append({
                'name': section_name,
                'section': 'fill',
                'start_sample': fill_start_bar * samples_per_bar,
                'end_sample': fade_end * samples_per_bar,
                'effects': effects,
                'onset': random.choices(
                    ['quarter_note', 'rapid', 'instant'],
                    weights=[50, 30, 20], k=1)[0],
            })

        # OUTRO — anchor remains as the stable tail while other buses wash/fade out.
        if 'outro' in config.SONG_SECTIONS:
            cfg_outro_start, cfg_outro_end = config.SONG_SECTIONS['outro']
            outro_start_bar = cfg_outro_start if cfg_outro_start < total_bars else max(0, total_bars - 8)
            outro_end_bar = min(max(cfg_outro_end, outro_start_bar + 1), total_bars)
            if outro_start_bar < total_bars:
                effects = {}
                for bus in cycle_buses:
                    if bus == outro_anchor:
                        continue
                    if bus == 'drums':
                        effects[bus] = random.choice(['outro_fade', 'lpf_sweep', 'tape_stop', 'vinyl_stop'])
                    elif bus == 'bass':
                        effects[bus] = random.choice(['outro_fade', 'hpf_sweep', 'lpf_sweep'])
                    else:
                        effects[bus] = random.choice(['outro_fade', 'reverb_wash', 'lpf_sweep'])
                plan.append({
                    'name': 'outro',
                    'section': 'outro',
                    'start_sample': outro_start_bar * samples_per_bar,
                    'end_sample': outro_end_bar * samples_per_bar,
                    'effects': effects,
                    'onset': random.choices(
                        ['quarter_note', 'rapid', 'instant'],
                        weights=[50, 30, 20], k=1)[0],
                    'anchor_bus': outro_anchor,
                })

        return plan

    def _choose_anchor_bus(self, available_buses) -> str:
        candidates = [bus for bus in ['pads', 'melody'] if bus in available_buses]
        if len(candidates) == 2:
            return random.choice(candidates)
        if candidates:
            return candidates[0]
        return None

    def _next_automated_bus(self, bus_cycle, available_buses) -> str:
        for _ in range(max(1, len(available_buses) + 4)):
            bus = next(bus_cycle)
            if bus in available_buses:
                return bus
        return next(bus_cycle)

    def _apply_effect(self, y: np.ndarray, sr: int, bpm: float,
                      effect: str, start: int, end: int,
                      bus_name: str, onset: str = 'instant') -> np.ndarray:
        """Apply a single effect to a segment of audio."""
        out = y.copy()
        seg_len = end - start
        if seg_len < sr // 10:
            return out

        seg = out[start:end].copy()
        orig_seg = seg.copy()  # keep dry signal for envelope blending

        # ── Tape/Vinyl ───────────────────────────────────────────
        if effect == 'tape_stop':
            dur = min(1.5, seg_len / sr * 0.8)
            seg = self.dsp.tape_stop(seg, sr, dur)

        elif effect == 'vinyl_stop':
            dur = min(2.0, seg_len / sr * 0.8)
            seg = self.dsp.vinyl_stop(seg, sr, dur)

        elif effect == 'tape_start':
            dur = min(1.5, seg_len / sr * 0.8)
            seg = self.dsp.tape_start(seg, sr, dur)

        elif effect == 'tape_wobble':
            seg = self.dsp.tape_wobble(seg, sr, depth=config.EFFECT_DEPTH.get('tape_wobble_depth', 0.012))

        elif effect == 'vinyl_crackle':
            seg = self.dsp.vinyl_crackle(seg, sr, amount=config.EFFECT_DEPTH.get('vinyl_crackle_amount', 0.08))

        # ── Rhythmic ─────────────────────────────────────────────
        elif effect == 'sidechain_pump':
            seg = self.dsp.sidechain_pump(seg, sr, bpm, depth=config.EFFECT_DEPTH.get('sidechain_pump_depth', 0.95))

        elif effect == 'beat_repeat':
            mid = len(seg) // 2
            seg = self.dsp.beat_repeat(seg, sr, bpm, mid, repeats=4, division=0.25)

        elif effect == 'stutter':
            mid = len(seg) // 2
            seg = self.dsp.stutter(seg, sr, mid, stutter_len=2048, repeats=6)

        elif effect == 'silence_drop':
            drop_start = max(0, seg_len - int(sr * 60.0 / bpm))
            seg = self.dsp.silence_drop(seg, sr, drop_start, int(sr * 0.1))

        # ── Filter ───────────────────────────────────────────────
        elif effect == 'hpf_sweep':
            if bus_name == 'bass':
                seg = self.dsp.ladder_filter_sweep(seg, sr, config.EFFECT_DEPTH.get('hpf_sweep_bass_cutoff', 4000.0), 30.0, 'hpf', 0.5)
            else:
                seg = self.dsp.ladder_filter_sweep(seg, sr, 8000.0, 200.0, 'hpf', 0.5)

        elif effect == 'lpf_sweep':
            seg = self.dsp.ladder_filter_sweep(seg, sr, 18000.0, config.EFFECT_DEPTH.get('lpf_sweep_end_cutoff', 200.0), 'lpf', 0.5)

        elif effect == 'lpf_on_off':
            # Sudden LPF at 1kHz for the segment
            lf = LadderFilter(mode=LadderFilter.Mode.LPF24, cutoff_hz=1000, resonance=0.5, drive=1.0)
            seg = lf(seg, sr)

        elif effect == 'hpf_on_off':
            # Sudden HPF at 500Hz for the segment
            lf = LadderFilter(mode=LadderFilter.Mode.HPF24, cutoff_hz=500, resonance=0.5, drive=1.0)
            seg = lf(seg, sr)

        elif effect == 'filter_drop':
            mid = len(seg) // 2
            seg = self.dsp.filter_drop(seg, sr, mid, drop_hz=200.0, hold_sec=0.5)

        elif effect == 'riser':
            seg = self.dsp.ladder_filter_sweep(seg, sr, 400.0, 18000.0, 'lpf', 0.5)
            peak_db = config.EFFECT_DEPTH.get('riser_gain_peak_db', 8.0)
            gain_curve = np.linspace(0.0, peak_db, len(seg), dtype=np.float32)
            seg = seg * (10.0 ** (gain_curve[:, np.newaxis] / 20.0))

        elif effect == 'resonance_sweep':
            # Sweep resonance from 0 to 0.9 using overlap-add
            block_size = 4096
            n_blocks = max(1, len(seg) // block_size)
            resonances = np.linspace(0.0, 0.9, n_blocks)

            def process_res_block(block, sr, idx=0):
                i = min(idx, n_blocks - 1)
                lf = LadderFilter(mode=LadderFilter.Mode.LPF24, cutoff_hz=2000,
                                  resonance=float(resonances[i]), drive=1.0)
                return lf(block, sr)

            hop = int(block_size * 0.5)
            n_samples = len(seg)
            n_channels = seg.shape[1] if seg.ndim > 1 else 1
            n_blocks_oa = max(1, (n_samples - block_size) // hop + 1)
            out_seg = np.zeros((n_samples + block_size, n_channels), dtype=np.float64)
            norm = np.zeros((n_samples + block_size, 1), dtype=np.float64)
            window = np.hanning(block_size).astype(np.float64).reshape(-1, 1)
            for i in range(n_blocks_oa):
                s = i * hop
                e = min(s + block_size, n_samples)
                actual_len = e - s
                block = seg[s:e]
                idx = int(i * n_blocks / max(n_blocks_oa, 1))
                processed = process_res_block(block, sr, idx)
                out_seg[s:e] += processed[:actual_len].astype(np.float64) * window[:actual_len]
                norm[s:e] += window[:actual_len]
            norm = np.maximum(norm, 1e-10)
            seg = (out_seg[:n_samples] / norm[:n_samples]).astype(np.float32)

        # ── Spatial ──────────────────────────────────────────────
        elif effect == 'reverb_wash':
            if bus_name == 'melody':
                seg = self.dsp.reverb_wash(seg, sr, start_wet=0.6, end_wet=0.0, room_size=0.7)
            else:
                seg = self.dsp.reverb_wash(seg, sr, start_wet=0.0,
                                            end_wet=config.EFFECT_DEPTH.get('reverb_wash_drums_wet', 0.8), room_size=0.6)

        elif effect == 'reverb_throw':
            # Reverb wet spike then back
            seg = self.dsp.reverb_wash(seg, sr, start_wet=0.0, end_wet=0.5, room_size=0.8)

        elif effect == 'reverb_freeze':
            seg = self.dsp.reverb_freeze(seg, sr, wet_level=0.5)

        elif effect == 'delay_feedback_swell':
            delay_sec = 60.0 / bpm * 0.75  # Dotted eighth
            seg = self.dsp.delay_feedback_swell(seg, sr, delay_sec=delay_sec,
                                                 start_feedback=0.1,
                                                 end_feedback=config.EFFECT_DEPTH.get('delay_feedback_end_feedback', 0.9))

        elif effect == 'chorus_swell':
            seg = self.dsp.chorus_swell(seg, sr, start_depth=0.0,
                                         end_depth=config.EFFECT_DEPTH.get('chorus_swell_end_depth', 0.7))

        elif effect == 'phaser_sweep':
            block_size = 4096
            n_blocks = max(1, len(seg) // block_size)
            freqs = np.linspace(500, 3000, n_blocks)
            out_seg = np.zeros_like(seg)
            for i in range(n_blocks):
                s = i * block_size
                e = min((i + 1) * block_size, len(seg))
                block = seg[s:e]
                ph = Phaser(rate_hz=0.5, depth=0.5, centre_frequency_hz=float(freqs[i]),
                            feedback=0.0, mix=0.5)
                out_seg[s:e] = ph(block, sr)
            seg = out_seg

        # ── Stereo ───────────────────────────────────────────────
        elif effect == 'stereo_widen':
            width_curve = np.linspace(1.0, config.EFFECT_DEPTH.get('stereo_widen_max', 2.0), len(seg), dtype=np.float32)
            seg = self.dsp.stereo_width_automation(seg, width_curve)

        elif effect == 'stereo_narrow':
            width_curve = np.linspace(1.0, 0.3, len(seg), dtype=np.float32)
            seg = self.dsp.stereo_width_automation(seg, width_curve)

        # ── Distortion/Bitcrush ──────────────────────────────────
        elif effect == 'distortion_build':
            seg = self.dsp.distortion_build(seg, sr, peak_drive_db=config.EFFECT_DEPTH.get('distortion_build_peak_db', 25.0))

        elif effect == 'bitcrush_sweep':
            seg = self.dsp.bitcrush_sweep(seg, sr, start_bits=12, end_bits=config.EFFECT_DEPTH.get('bitcrush_sweep_end_bits', 2))

        # ── Frequency Isolation ──────────────────────────────────
        elif effect == 'isolate_low':
            seg = self.dsp.isolate_low(seg, sr)

        elif effect == 'isolate_mid':
            seg = self.dsp.isolate_mid(seg, sr)

        elif effect == 'isolate_high':
            seg = self.dsp.isolate_high(seg, sr)

        # ── Tremolo ──────────────────────────────────────────────
        elif effect == 'tremolo':
            freq = bpm / 60.0
            t = np.linspace(0, len(seg) / sr, len(seg), dtype=np.float32)
            mod = 0.5 + 0.5 * np.sin(2 * np.pi * freq * t)
            seg = seg * mod[:, np.newaxis]

        # ── Impact ───────────────────────────────────────────────
        elif effect == 'impact':
            impact_pos = 0
            seg = self.dsp.impact(seg, sr, impact_pos, duration_sec=0.5)

        # ── Gain Fade ────────────────────────────────────────────
        elif effect == 'gain_rise':
            gain_curve = np.linspace(0.04, 1.15, len(seg), dtype=np.float32)
            seg = seg * gain_curve[:, np.newaxis]

        elif effect == 'outro_fade':
            gain_curve = np.linspace(1.0, 0.04, len(seg), dtype=np.float32)
            seg = seg * gain_curve[:, np.newaxis]

        elif effect == 'gain_fade':
            if bus_name == 'melody':
                # Intro: fade in, Outro: fade out
                gain_curve = np.linspace(0.2, 1.0, len(seg), dtype=np.float32)
            else:
                gain_curve = np.linspace(1.0, 0.0, len(seg), dtype=np.float32)
            seg = seg * gain_curve[:, np.newaxis]

        # Apply intensity envelope — ramp depth and mix for fixed effects
        if effect in self.FIXED_EFFECTS and onset != 'instant':
            depth_env, mix_env = self._build_intensity_envelopes(
                len(seg), sr, bpm, onset)
            if mix_env.ndim == 1 and seg.ndim == 2:
                mix_env = mix_env[:, np.newaxis]
            seg = orig_seg * (1.0 - mix_env) + seg * mix_env

        # Apply crossfade at effect boundaries to prevent clicks
        # Shorter crossfade for dramatic effects to preserve impact
        if effect in DRAMATIC_EFFECTS:
            crossfade_len = min(int(sr * 0.02), seg_len // 3)  # 20ms for dramatic
        else:
            crossfade_len = min(int(sr * 0.1), seg_len // 3)   # 100ms for subtle
        if crossfade_len > 0 and effect not in ['gain_fade', 'gain_rise', 'outro_fade', 'tape_stop', 'vinyl_stop',
                                                  'tape_start', 'silence_drop']:
            orig_seg = y[start:end].copy()
            fade_in = np.linspace(0, 1, crossfade_len, dtype=np.float32)[:, None]
            fade_out = np.linspace(1, 0, crossfade_len, dtype=np.float32)[:, None]
            seg[:crossfade_len] = orig_seg[:crossfade_len] * fade_out + seg[:crossfade_len] * fade_in
            seg[-crossfade_len:] = orig_seg[-crossfade_len:] * fade_in + seg[-crossfade_len:] * fade_out

        # Peak protection after effect — higher ceiling for dramatic effects
        peak = np.max(np.abs(seg))
        ceiling = 0.97 if effect in DRAMATIC_EFFECTS else 0.89  # -0.3 vs -1.0 dBFS
        if peak > ceiling:
            seg = seg * (ceiling / peak)

        # Section-specific compression — SKIP for dramatic effects to preserve dynamics
        if effect not in ['gain_fade', 'gain_rise', 'outro_fade', 'silence_drop', 'stereo_widen', 'stereo_narrow',
                          'vinyl_crackle', 'sidechain_pump', 'tremolo'] and effect not in DRAMATIC_EFFECTS:
            from pedalboard import Compressor as _Comp
            comp = _Comp(threshold_db=-15.0,
                         ratio=config.SECTION_COMP['ratio'],
                         attack_ms=config.SECTION_COMP['attack_ms'],
                         release_ms=config.SECTION_COMP['release_ms'])
            seg = comp(seg, sr)
            seg = self.dsp._peak_protect(seg, ceiling_db=-1.0)

        # DC offset removal
        seg = seg - np.mean(seg, axis=0)

        out[start:end] = seg
        return out.astype(np.float32)


# Import pedalboard effects needed by _apply_effect
from pedalboard import LadderFilter, Phaser


def _ensure_stereo(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return np.stack([y, y], axis=1)
    if y.shape[1] == 1:
        return np.repeat(y, 2, axis=1)
    return y[:, :2]


def _beats_per_bar_from_song_name(song_name: str = None) -> float:
    """Return quarter-note beats per bar from names like *_5-8_* or *_3-4_*."""
    if not song_name:
        return 4.0
    match = re.search(r'(?<!\d)(\d+)-(\d+)(?!\d)', str(song_name))
    if not match:
        return 4.0
    numerator = int(match.group(1))
    denominator = int(match.group(2))
    if numerator <= 0 or denominator <= 0:
        return 4.0
    return numerator * (4.0 / denominator)
