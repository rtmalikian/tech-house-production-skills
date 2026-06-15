import mido
import time
import json
import random
import os
import re
import subprocess
from typing import List, Dict, Optional

# Roland Fantom Constants
FANTOM_MODEL_ID = [0x00, 0x00, 0x00, 0x5B]
ROLAND_ID = 0x41
DEVICE_ID = 0x10  # Default Device ID
_TONE_PORTAMENTO_SYSEX_ENABLED = os.environ.get("FANTOM_TONE_PORTAMENTO_SYSEX", "0").strip().lower() in {"1", "true", "yes", "on"}
_DEFAULT_FANTOM_OUTPUT_PORTS = [
    "FANTOM-6 7 8",
    "FANTOM-6 7 8 MIDI OUT 1",
    "FANTOM-6 7 8 MIDI OUT 2",
    "FANTOM-6 7 8 DAW CTRL",
]

def calculate_checksum(data: List[int]) -> int:
    """Calculate Roland checksum."""
    sum_data = sum(data) % 128
    return (128 - sum_data) % 128

def create_roland_sysex(address: List[int], data: List[int]) -> mido.Message:
    """
    Create a Roland DT1 (Data Set) SysEx message.
    Format: F0 41 dev model 12 addr data sum F7
    """
    body = address + data
    checksum = calculate_checksum(body)
    full_data = [ROLAND_ID, DEVICE_ID] + FANTOM_MODEL_ID + [0x12] + body + [checksum]
    return mido.Message('sysex', data=full_data)

def list_midi_output_ports() -> List[str]:
    """Return safe Fantom output candidates without python-rtmidi enumeration."""
    ports = []
    env_port = os.environ.get("FANTOM_MIDI_PORT", "").strip()
    if env_port:
        ports.append(env_port)
    env_ports = os.environ.get("FANTOM_MIDI_PORTS", "").strip()
    if env_ports:
        ports.extend(port.strip() for port in env_ports.split(",") if port.strip())

    try:
        result = subprocess.run(
            ["system_profiler", "SPMIDIDataType"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if "FANTOM" in result.stdout.upper():
            ports.extend(_DEFAULT_FANTOM_OUTPUT_PORTS)
    except Exception:
        pass

    ports.extend(_DEFAULT_FANTOM_OUTPUT_PORTS)
    deduped = []
    for port in ports:
        if port not in deduped:
            deduped.append(port)
    return deduped

def _addr_add(address: List[int], offset: List[int]) -> List[int]:
    """Add Roland 7-bit address bytes, aligning shorter offsets to the right."""
    addr = address[:]
    off = [0] * (len(addr) - len(offset)) + offset
    carry = 0
    out = [0] * len(addr)
    for i in range(len(addr) - 1, -1, -1):
        value = addr[i] + off[i] + carry
        out[i] = value & 0x7F
        carry = value >> 7
    return out

def _nibbles(value: int, width: int = 4) -> List[int]:
    return [(value >> (4 * i)) & 0x0F for i in range(width - 1, -1, -1)]

def _percent_to_127(percent: float) -> int:
    return max(0, min(127, int(round(127 * percent / 100.0))))

def _percent_to_1023(percent: float) -> int:
    return max(0, min(1023, int(round(1023 * percent / 100.0))))

def _vary_percent(percent: float, spread: float = 0.15) -> float:
    return max(0.0, min(100.0, percent * random.uniform(1.0 - spread, 1.0 + spread)))

def _signed_63(value: int) -> int:
    return max(1, min(127, 64 + value))

def _clamp_signed_depth(value: int, limit: int) -> int:
    limit = abs(int(limit))
    return max(-limit, min(limit, int(round(value))))

def _signed_100(value: int) -> int:
    return max(28, min(228, 128 + value))

def _scale_tune_value(cents: int) -> int:
    """Fantom Zone Scale Tune data: -64..+63 cents encoded as 0..127."""
    return max(0, min(127, 64 + max(-64, min(63, int(round(cents))))))

def _mfx_param_value(value: int) -> List[int]:
    # MFX parameter data is transmitted as a 4-nibble value centered at 32768.
    return _nibbles(max(12768, min(52768, 32768 + value)), 4)

class FantomController:
    def __init__(self, port_name: Optional[str] = None):
        candidates = [port_name] if port_name else self.detect_fantom_ports()
        self.port_name = None
        self.output = None
        self.sound_db = {}
        self.load_sound_db()
        for candidate in candidates:
            if not candidate:
                continue
            try:
                self.output = mido.open_output(candidate)
                self.port_name = candidate
                print(f"Connected to Roland Fantom on port: {self.port_name}")
                break
            except Exception as e:
                print(f"Error opening MIDI port {candidate}: {e}")
        if not self.output:
            print("Roland Fantom MIDI port not found.")

    def load_sound_db(self):
        db_path = os.path.join(os.path.dirname(__file__), "fantom_sounds.json")
        try:
            with open(db_path, "r") as f:
                self.sound_db = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load sound database: {e}")

    def select_random_patches(self) -> List[Dict]:
        """
        Randomly select appropriate patches for the standard tracks.
        """
        if not self.sound_db: return []
        
        assignments = []
        def pick(cat):
            opts = self.sound_db.get(cat, [])
            if not opts: return {'name': 'Default', 'msb': 87, 'lsb': 64, 'pc': 0}
            return random.choice(opts)

        # Main Melody (Ch 0)
        p = pick('melody'); p['channel'] = 0; assignments.append(p)
        # Counter Melody (Ch 1)
        p = pick('melody'); p['channel'] = 1; assignments.append(p)
        # Chorus Melody (Ch 2)
        p = pick('melody'); p['channel'] = 2; assignments.append(p)
        # Pad (Ch 3)
        p = pick('pad'); p['channel'] = 3; assignments.append(p)
        # Bass (Ch 4)
        p = pick('bass'); p['channel'] = 4; assignments.append(p)
        # Harmonic Bass (Ch 5)
        p = pick('bass'); p['channel'] = 5; assignments.append(p)
        # Drums (Ch 9)
        p = pick('drums'); p['channel'] = 9; assignments.append(p)
        
        return assignments

    def detect_fantom_ports(self) -> List[str]:
        return [name for name in list_midi_output_ports() if "FANTOM" in name.upper()]

    def detect_fantom_port(self) -> Optional[str]:
        for name in list_midi_output_ports():
            if "FANTOM" in name.upper():
                return name
        return None

    def select_patch(self, channel: int, msb: int, lsb: int, pc: int):
        if not self.output: return
        
        # 1. Kill any hanging notes before changing patch
        self.output.send(mido.Message('control_change', channel=channel, control=123, value=0))
        time.sleep(0.02)
        
        # 2. Bank Select MSB/LSB
        self.output.send(mido.Message('control_change', channel=channel, control=0, value=msb))
        self.output.send(mido.Message('control_change', channel=channel, control=32, value=lsb))
        
        # 3. Hardware Sync Delay (Crucial for mode switching Tones <-> Drum Kits)
        time.sleep(0.1)
        
        # 4. Program Change (Subtract 1 to convert from 1-based Sound List to 0-based MIDI)
        pc_zero_based = max(0, min(127, pc - 1))
        self.output.send(mido.Message('program_change', channel=channel, program=pc_zero_based))
        
        # 5. Final settle delay
        time.sleep(0.1)

    def _send_dt1(self, address: List[int], data: List[int], delay: float = 0.012):
        if not self.output:
            return
        self.output.send(create_roland_sysex(address, data))
        time.sleep(delay)

    def _zcore_base(self, part_idx: int) -> List[int]:
        return [0x02, 0x10 + part_idx, 0x00, 0x00]

    def _drum_kit_base(self, part_idx: int) -> List[int]:
        return [0x02, 0x30 + (part_idx * 2), 0x00, 0x00]

    def _drum_inst_set_base(self, part_idx: int) -> List[int]:
        return [0x03, part_idx * 4, 0x00, 0x00]

    def _set_mfx_type_and_params(self, mfx_base: List[int], mfx_type: int,
                                 params: Dict[int, int], switch: bool = True):
        self._send_dt1(_addr_add(mfx_base, [0x00, 0x00]), [int(mfx_type)])
        self._send_dt1(_addr_add(mfx_base, [0x00, 0x01]), [1 if switch else 0])
        for param_idx, value in sorted(params.items(), key=lambda kv: int(kv[0])):
            param_idx = int(param_idx)
            value = int(value)
            param_start = 0x10 + ((param_idx - 1) * 4)
            self._send_dt1(_addr_add(mfx_base, [param_start >> 7, param_start & 0x7F]),
                           _mfx_param_value(value))

    def _tone_mfx_base(self, part_idx: int) -> List[int]:
        return _addr_add(self._zcore_base(part_idx), [0x00, 0x01, 0x00])

    def _drum_mfx_base(self, part_idx: int) -> List[int]:
        return _addr_add(self._drum_kit_base(part_idx), [0x00, 0x01, 0x00])

    def apply_track_sound_design(self, part_idx: int, track_name: str) -> Dict:
        """
        Apply temporary Fantom tone/drum-kit edits after patch selection.
        part_idx is 0-based. These edits affect the temporary part for recording,
        not saved user tones.

        Returns a dict with human-readable strings AND a ``_raw`` key containing
        the exact SysEx parameters so they can be replayed later (e.g. sample pack).
        """
        n = track_name.lower()
        applied = {"lfo_matrix": False, "mfx": None, "drum_fxm": None}
        raw = {"mfx": None, "lfo_matrix": None, "drum_fxm": None}

        if self._is_eastern_percussion_tone(n):
            applied["mfx"], raw["mfx"] = self.apply_tone_phonograph_mfx(part_idx)
            applied["_raw"] = raw
            return applied

        if self._is_drum_track_name(n):
            applied["mfx"], raw["mfx"] = self.apply_random_drum_mfx(part_idx, n)
            note = self._extract_drum_note(n)
            if note is not None:
                applied["drum_fxm"], raw["drum_fxm"] = self.enable_drum_note_fxm(part_idx, note)
            applied["_raw"] = raw
            return applied

        if any(x in n for x in ['bass', 'melody', 'lead', 'counter', 'chorus', 'pad', 'chord']):
            applied["lfo_matrix"], raw["lfo_matrix"] = self.apply_zcore_lfo_matrix(part_idx, n)

        if self._is_melody_track_name(n):
            applied["mfx"], raw["mfx"] = self.apply_random_melody_mfx(part_idx, n)

        applied["_raw"] = raw
        return applied

    def apply_sound_design_from_manifest(self, part_idx: int, raw: Dict) -> Dict:
        """
        Replay sound design from a previously recorded manifest's ``_raw`` data.
        Applies the exact same MFX, LFO matrix, and drum FXM parameters.
        """
        if not raw:
            return {}

        mfx = raw.get("mfx")
        if mfx:
            base = self._drum_mfx_base(part_idx) if mfx.get("base") == "drum" else self._tone_mfx_base(part_idx)
            self._set_mfx_type_and_params(base, mfx["type"], mfx["params"])

        lfo = raw.get("lfo_matrix")
        if lfo:
            self._replay_lfo_matrix(part_idx, lfo)

        fxm = raw.get("drum_fxm")
        if fxm:
            self._replay_drum_fxm(part_idx, fxm["note"], fxm["color"], fxm["depth"])

        return raw

    def _replay_lfo_matrix(self, part_idx: int, lfo: Dict) -> None:
        """Replay stored LFO matrix SysEx parameters."""
        for partial_idx, partial in enumerate(lfo.get("partials", [])):
            block = _addr_add(self._zcore_base(part_idx), [0x00, 0x30 + (partial_idx * 2), 0x00])
            # Cast all values to int (JSON deserializes numbers as int/float, but
            # dict keys and some nested values may arrive as strings after round-trip).
            lfo1_wave = int(partial["lfo1_wave"])
            lfo1_note = int(partial["lfo1_note"])
            lfo1_tvf = int(partial["lfo1_tvf"])
            lfo1_tva = int(partial["lfo1_tva"])
            lfo2_wave = int(partial["lfo2_wave"])
            lfo2_note = int(partial["lfo2_note"])
            lfo2_tvf = int(partial["lfo2_tvf"])
            lfo2_tva = int(partial["lfo2_tva"])
            # LFO1
            self._send_dt1(_addr_add(block, [0x00, 0x00]), [lfo1_wave])
            self._send_dt1(_addr_add(block, [0x00, 0x01]), [1])
            self._send_dt1(_addr_add(block, [0x00, 0x02]), [lfo1_note])
            self._send_dt1(_addr_add(block, [0x00, 0x19]), _nibbles(_signed_100(lfo1_tvf), 2))
            self._send_dt1(_addr_add(block, [0x00, 0x1B]), _nibbles(_signed_100(lfo1_tva), 2))
            # LFO2
            self._send_dt1(_addr_add(block, [0x00, 0x4F]), [lfo2_wave])
            self._send_dt1(_addr_add(block, [0x00, 0x50]), [1])
            self._send_dt1(_addr_add(block, [0x00, 0x51]), [lfo2_note])
            self._send_dt1(_addr_add(block, [0x00, 0x68]), _nibbles(_signed_100(lfo2_tvf), 2))
            self._send_dt1(_addr_add(block, [0x00, 0x6A]), _nibbles(_signed_100(lfo2_tva), 2))
            # Matrix routing
            m = partial["matrix"]
            m1_dest1 = int(m["m1_dest1"])
            m1_sens1 = int(m["m1_sens1"])
            m1_dest2 = int(m["m1_dest2"])
            m1_sens2 = int(m["m1_sens2"])
            m2_dest1 = int(m["m2_dest1"])
            m2_sens1 = int(m["m2_sens1"])
            m2_dest2 = int(m["m2_dest2"])
            m2_sens2 = int(m["m2_sens2"])
            self._send_dt1(_addr_add(block, [0x00, 0x56]), [104])
            self._send_dt1(_addr_add(block, [0x00, 0x57]), [m1_dest1])
            self._send_dt1(_addr_add(block, [0x00, 0x58]), [_signed_63(m1_sens1)])
            self._send_dt1(_addr_add(block, [0x00, 0x59]), [m1_dest2])
            self._send_dt1(_addr_add(block, [0x00, 0x5A]), [_signed_63(m1_sens2)])
            self._send_dt1(_addr_add(block, [0x00, 0x5F]), [105])
            self._send_dt1(_addr_add(block, [0x00, 0x60]), [m2_dest1])
            self._send_dt1(_addr_add(block, [0x00, 0x61]), [_signed_63(m2_sens1)])
            self._send_dt1(_addr_add(block, [0x00, 0x62]), [m2_dest2])
            self._send_dt1(_addr_add(block, [0x00, 0x63]), [_signed_63(m2_sens2)])
            # FXM (melody only)
            if partial.get("fxm_enabled"):
                fxm_color = int(partial["fxm_color"])
                fxm_depth = int(partial["fxm_depth"])
                self._send_dt1(_addr_add(block, [0x00, 0x06]), [1])
                self._send_dt1(_addr_add(block, [0x00, 0x07]), [fxm_color])
                self._send_dt1(_addr_add(block, [0x00, 0x08]), [fxm_depth])
                self._send_dt1(_addr_add(block, [0x00, 0x09]), [int(partial.get("osc_fat", 8))])
                self._send_dt1(_addr_add(block, [0x00, 0x0A]), [int(partial.get("osc_delay_mode", 0))])
                self._send_dt1(_addr_add(block, [0x00, 0x0B]), [int(partial.get("osc_delay_time", 0))])

    def _replay_drum_fxm(self, part_idx: int, note: int, color: int, depth: int) -> None:
        """Replay stored drum FXM SysEx parameters."""
        note = int(note)
        color = int(color)
        depth = int(depth)
        if note < 21 or note > 108:
            return
        inst_base = self._drum_inst_set_base(part_idx)
        key_step = (note - 21) * 5
        key_base = _addr_add(inst_base, [(key_step >> 7) & 0x7F, key_step & 0x7F, 0x00])
        for offset in [[0x00, 0x2C], [0x00, 0x4F], [0x00, 0x72], [0x01, 0x15]]:
            self._send_dt1(_addr_add(key_base, offset), [1])
            self._send_dt1(_addr_add(key_base, [offset[0], offset[1] + 1]), [color])
            self._send_dt1(_addr_add(key_base, [offset[0], offset[1] + 2]), [depth])

    def _is_drum_track_name(self, name: str) -> bool:
        return any(x in name for x in [
            'drum1_', 'drum2_', 'drum_aux_', 'kick', 'snare', 'hat',
            'clap', 'bongo', 'conga', 'tambourine', 'maracas'
        ])

    def _extract_drum_note(self, name: str) -> Optional[int]:
        match = re.search(r"_n(\d+)", name)
        return int(match.group(1)) if match else None

    def _is_eastern_percussion_tone(self, name: str) -> bool:
        note = self._extract_drum_note(name)
        return 'drum_aux_' in name and note in {54, 60, 61, 62, 70}

    def _is_melody_track_name(self, name: str) -> bool:
        return any(x in name for x in [
            'main_melody', 'main melody', 'counter_melody', 'counter melody',
            'chorus_melody', 'chorus melody', 'lead'
        ])

    def apply_random_drum_mfx(self, part_idx: int, track_name: str = "") -> tuple:
        chooser = random.random()
        if chooser < 0.45:
            return self.apply_drum_bitcrusher_mfx(part_idx, track_name)
        if chooser < 0.80:
            return self.apply_drum_lofi_compress_mfx(part_idx)
        return self.apply_drum_phonograph_mfx(part_idx)

    def apply_drum_bitcrusher_mfx(self, part_idx: int, track_name: str = "") -> tuple:
        # MFX 46 Bit Crusher: moderate crush with tight +/-5% variation.
        sample_rate_pct = random.uniform(55, 65)
        bit_down_pct = random.uniform(45, 55)
        filter_pct = random.uniform(45, 55)
        level = random.randint(88, 112)
        params = {
            1: _percent_to_127(sample_rate_pct),    # Sample Rate
            2: int(round(20 * bit_down_pct / 100)), # Bit Down, range 0-20
            3: _percent_to_127(filter_pct),         # Filter
            4: random.randint(15, 18),              # Low Gain, center to slight lift
            5: random.randint(14, 16),              # High Gain, near center
            6: level,                               # Output Level
        }
        self._set_mfx_type_and_params(self._drum_mfx_base(part_idx), 46, params)
        label = f"MFX46 Bit Crusher sr={sample_rate_pct:.0f} bit={bit_down_pct:.0f} filter={filter_pct:.0f}"
        raw = {"base": "drum", "type": 46, "params": params}
        return label, raw

    def apply_drum_lofi_compress_mfx(self, part_idx: int) -> tuple:
        # MFX 45 LOFI Compress: compressor pre-filter into Lo-Fi, post LPF around 2 kHz.
        pre_filter = random.randint(2, 6)
        lofi_type = random.randint(3, 9)
        balance = random.randint(60, 85)
        low_gain = random.choice([16, 17, 18])  # roughly +1..+3 dB in local encoding convention
        params = {
            1: pre_filter,          # Pre Filter Type 1-6
            2: lofi_type,           # LoFi Type 1-9
            3: 1,                   # Post Filter Type: LPF
            4: 2000,                # Post Filter Cutoff: 2000 Hz
            5: low_gain,            # Low Gain, slight lift
            6: 15,                  # High Gain, center
            7: balance,             # Balance D100:0W-D0:100W
            8: 100,                 # Output Level
        }
        self._set_mfx_type_and_params(self._drum_mfx_base(part_idx), 45, params)
        label = f"MFX45 LOFI Compress pre={pre_filter} lofi={lofi_type} LPF=2000 balance={balance}"
        raw = {"base": "drum", "type": 45, "params": params}
        return label, raw

    def apply_drum_phonograph_mfx(self, part_idx: int) -> tuple:
        # MFX 91 Phonograph is available on Fantom v3+. Keep it subtle for drums.
        signal_dist = _percent_to_127(random.randint(10, 20))
        freq_range_pct = random.randint(40, 60)
        total_noise = _percent_to_127(10)
        params = {
            1: signal_dist,                         # Signal destruction/distortion
            2: _percent_to_127(freq_range_pct),     # Frequency Range
            3: random.choice([0, 1, 2]),            # Disc Type: LP/EP/SP
            4: random.randint(2, 8),                # Scratch noise
            5: random.randint(2, 8),                # Dust noise
            6: random.randint(2, 8),                # Hiss noise
            7: total_noise,                         # Total noise level
            8: random.randint(3, 10),               # Wow
            9: random.randint(3, 10),               # Flutter
            10: random.randint(2, 8),               # Random spin variation
            11: random.randint(3, 10),              # Total wow/flutter
            12: random.randint(15, 28),             # Balance, subtle wet
            13: 100,                                # Output Level
        }
        self._set_mfx_type_and_params(self._drum_mfx_base(part_idx), 91, params)
        label = f"MFX91 Phonograph noise=10 freq={freq_range_pct} dist={round(signal_dist / 127 * 100)}"
        raw = {"base": "drum", "type": 91, "params": params}
        return label, raw

    def apply_tone_bitcrusher_mfx(self, part_idx: int) -> tuple:
        # Same creative settings as the drum Bit Crusher, but against a Z-Core tone.
        params = {
            1: _percent_to_127(75),  # Sample Rate
            2: int(round(20 * 0.85)), # Bit Down, range 0-20
            3: _percent_to_127(30),  # Filter
            4: 15,                   # Low Gain center for -15..+15 dB
            5: 15,                   # High Gain center for -15..+15 dB
            6: 100,                  # Output Level
        }
        self._set_mfx_type_and_params(self._tone_mfx_base(part_idx), 46, params)
        raw = {"base": "tone", "type": 46, "params": params}
        return "Tone MFX46 Bit Crusher sr=75 bit=85 filter=30", raw

    def apply_tone_phonograph_mfx(self, part_idx: int) -> tuple:
        # Phonograph-only treatment for eastern percussion tones; no LoFi/Bit Crush.
        signal_dist = _percent_to_127(random.randint(10, 18))
        freq_range_pct = random.randint(45, 60)
        total_noise = _percent_to_127(10)
        params = {
            1: signal_dist,
            2: _percent_to_127(freq_range_pct),
            3: random.choice([0, 1, 2]),
            4: random.randint(1, 6),
            5: random.randint(1, 6),
            6: random.randint(1, 6),
            7: total_noise,
            8: random.randint(2, 8),
            9: random.randint(2, 8),
            10: random.randint(1, 6),
            11: random.randint(2, 8),
            12: random.randint(12, 24),
            13: 100,
        }
        self._set_mfx_type_and_params(self._tone_mfx_base(part_idx), 91, params)
        label = f"Tone MFX91 Phonograph noise=10 freq={freq_range_pct} dist={round(signal_dist / 127 * 100)}"
        raw = {"base": "tone", "type": 91, "params": params}
        return label, raw

    def apply_bpm_looper_mfx(self, part_idx: int) -> tuple:
        # MFX 83 BPM Looper: Auto mode, Length 50, Rate 1/2, On Timing 1, On Length 2.
        params = {
            1: 50,   # Length
            2: 1,    # Rate sync ON
            4: 15,   # Rate Note: 1/2 using the Fantom note-value ordering
            5: 1,    # On Timing
            6: 2,    # On Length
            7: 1,    # Loop Mode AUTO (OFF, AUTO, ON)
            8: 100,  # Output Level
        }
        self._set_mfx_type_and_params(self._tone_mfx_base(part_idx), 83, params)
        raw = {"base": "tone", "type": 83, "params": params}
        return "MFX83 BPM Looper auto len=50 rate=1/2 timing=1 length=2", raw

    def apply_random_melody_mfx(self, part_idx: int, track_name: str) -> tuple:
        """
        Pick a temporary tone MFX for melody-like parts. Presets are conservative
        because this is inserted before audio recording; post-production still
        does the heavier mix work later.
        """
        palette = [
            self._melody_mfx_chorus,
            self._melody_mfx_hexa_chorus,
            self._melody_mfx_flanger,
            self._melody_mfx_enhancer,
            self._melody_mfx_super_filter,
            self._melody_mfx_auto_wah,
            self._melody_mfx_phaser,
        ]
        if any(x in track_name for x in ['counter', 'chorus']) and random.random() < 0.35:
            return self.apply_bpm_looper_mfx(part_idx)
        return random.choice(palette)(part_idx)

    def _melody_mfx_chorus(self, part_idx: int) -> tuple:
        params = {
            1: random.choice([0, 1]),           # filter OFF/LPF
            2: random.randint(8, 13),           # cutoff index, mid/high
            3: random.randint(8, 22),           # pre-delay
            4: 1,                               # rate sync ON
            6: random.choice([9, 12, 15]),      # 1/8, 1/4, 1/2
            7: random.randint(22, 42),          # depth
            8: random.choice([90, 120, 150]),   # phase
            9: 15,                              # low gain center
            10: 16,                             # high gain slight lift
            11: random.randint(20, 32),         # approx 20-32% wet balance
            12: 100,                            # level
        }
        self._set_mfx_type_and_params(self._tone_mfx_base(part_idx), 23, params)
        raw = {"base": "tone", "type": 23, "params": params}
        return "MFX23 Chorus wet~25", raw

    def _melody_mfx_hexa_chorus(self, part_idx: int) -> tuple:
        params = {
            1: random.randint(8, 18),           # pre-delay
            2: 1,                               # rate sync ON
            4: random.choice([9, 12, 15]),      # synced note
            5: random.randint(18, 36),          # depth
            6: random.randint(3, 9),            # pre-delay deviation
            7: random.randint(4, 10),           # depth deviation
            8: random.randint(10, 18),          # pan deviation
            9: random.randint(20, 30),          # approx wet balance
            10: 96,                             # level
        }
        self._set_mfx_type_and_params(self._tone_mfx_base(part_idx), 26, params)
        raw = {"base": "tone", "type": 26, "params": params}
        return "MFX26 Hexa-Chorus wet~25", raw

    def _melody_mfx_flanger(self, part_idx: int) -> tuple:
        params = {
            1: random.choice([0, 1, 2]),        # filter OFF/LPF/HPF
            2: random.randint(7, 12),           # cutoff index
            3: random.randint(2, 10),           # pre-delay
            4: 1,                               # rate sync ON
            6: random.choice([9, 12, 15]),      # synced note
            7: random.randint(10, 24),          # depth
            8: random.choice([90, 120, 150]),   # phase
            9: random.randint(45, 55),          # feedback center-ish
            10: 15,                             # low gain center
            11: 15,                             # high gain center
            12: random.randint(14, 24),         # wet balance
            13: 92,                             # level
        }
        self._set_mfx_type_and_params(self._tone_mfx_base(part_idx), 24, params)
        raw = {"base": "tone", "type": 24, "params": params}
        return "MFX24 Flanger wet~20", raw

    def _melody_mfx_enhancer(self, part_idx: int) -> tuple:
        params = {
            1: random.randint(18, 36),          # sensitivity
            2: random.randint(18, 34),          # mix
            3: 15,                              # low gain center
            4: random.randint(16, 18),          # slight high lift
            5: 100,                             # level
        }
        self._set_mfx_type_and_params(self._tone_mfx_base(part_idx), 7, params)
        raw = {"base": "tone", "type": 7, "params": params}
        return "MFX07 Enhancer mix~25", raw

    def _melody_mfx_super_filter(self, part_idx: int) -> tuple:
        params = {
            1: random.choice([0, 1, 2]),        # LPF/BPF/HPF
            2: random.choice([0, 1]),           # slope
            3: random.randint(58, 92),          # cutoff
            4: random.randint(10, 28),          # resonance
            5: random.randint(0, 3),            # gain
            6: 1,                               # modulation ON
            7: random.choice([0, 1, 2, 3, 4]),  # modulation wave
            8: 1,                               # sync ON
            10: random.choice([9, 12, 15]),     # note
            11: random.randint(8, 22),          # depth
            12: random.randint(18, 42),         # attack
            13: 100,                            # level
        }
        self._set_mfx_type_and_params(self._tone_mfx_base(part_idx), 5, params)
        raw = {"base": "tone", "type": 5, "params": params}
        return "MFX05 Super Filter synced", raw

    def _melody_mfx_auto_wah(self, part_idx: int) -> tuple:
        params = {
            1: random.choice([0, 1]),           # LPF/BPF
            2: random.randint(42, 74),          # manual
            3: random.randint(18, 36),          # peak
            4: random.randint(16, 34),          # sensitivity
            5: random.choice([0, 1]),           # polarity
            6: 1,                               # sync ON
            8: random.choice([9, 12, 15]),      # note
            9: random.randint(12, 28),          # depth
            10: random.choice([90, 120, 150]),  # phase
            11: 15,                             # low gain center
            12: 16,                             # high gain slight lift
            13: 96,                             # level
        }
        self._set_mfx_type_and_params(self._tone_mfx_base(part_idx), 8, params)
        raw = {"base": "tone", "type": 8, "params": params}
        return "MFX08 Auto Wah synced", raw

    def _melody_mfx_phaser(self, part_idx: int) -> tuple:
        params = {
            1: random.choice([0, 1, 2]),        # stages
            2: random.randint(18, 38),          # speed
            3: random.randint(18, 38),          # depth
            4: 15,                              # low gain center
            5: 16,                              # high gain slight lift
            6: random.randint(20, 30),          # approx wet/mix where available
            7: 96,                              # level
        }
        self._set_mfx_type_and_params(self._tone_mfx_base(part_idx), 11, params)
        raw = {"base": "tone", "type": 11, "params": params}
        return "MFX11 Phaser wet~25", raw

    def enable_drum_note_fxm(self, part_idx: int, note: int) -> tuple:
        if note < 21 or note > 108:
            return "drum FXM skipped: note out of range", None

        inst_base = self._drum_inst_set_base(part_idx)
        key_step = (note - 21) * 5
        key_base = _addr_add(inst_base, [(key_step >> 7) & 0x7F, key_step & 0x7F, 0x00])
        color = random.randint(2, 4)
        depth = random.randint(8, 16)
        for offset in [[0x00, 0x2C], [0x00, 0x4F], [0x00, 0x72], [0x01, 0x15]]:
            self._send_dt1(_addr_add(key_base, offset), [1])
            self._send_dt1(_addr_add(key_base, [offset[0], offset[1] + 1]), [color])
            self._send_dt1(_addr_add(key_base, [offset[0], offset[1] + 2]), [depth])
        label = f"note {note} FXM color={color} depth={depth}"
        raw = {"note": note, "color": color, "depth": depth}
        return label, raw

    def apply_zcore_lfo_matrix(self, part_idx: int, track_name: str) -> tuple:
        role_depth = 12 if 'bass' in track_name else 18 if any(x in track_name for x in ['pad', 'chord']) else 14
        note_choices = [9, 12, 15, 18]  # 1/8, 1/4, 1/2, 1 in Fantom note ordering.
        wave_choices = [0, 1, 2, 3, 4, 6, 7, 8]  # SIN, TRI, SAW, SQR, TRP, S&H, CHS.
        wave_names = {
            0: "SIN", 1: "TRI", 2: "SAW", 3: "SQR", 4: "TRP", 6: "S&H", 7: "CHS", 8: "VSIN"
        }
        note_names = {9: "1/8", 12: "1/4", 15: "1/2", 18: "1"}
        is_melody = self._is_melody_track_name(track_name)
        # Safe destinations only: no pitch or pitch-LFO modulation. CHO/REV sends are excluded.
        safe_destinations = [
            (2, "CUT"), (3, "RES"), (4, "LEV"), (5, "PAN"),
            (10, "TVF-LFO1"), (11, "TVF-LFO2"), (12, "TVA-LFO1"), (13, "TVA-LFO2"),
            (14, "PAN-LFO1"), (15, "PAN-LFO2"), (16, "LFO1-RATE"), (17, "LFO2-RATE"),
            (34, "PW"), (35, "PWM"), (36, "FAT"), (37, "XMOD"), (39, "SSAW-DETN"),
        ]
        if is_melody:
            safe_destinations.extend([
                (27, "TVA-DCY"),    # light melodic amp-envelope decay modulation
                (28, "TVA-REL"),    # light melodic amp-envelope release modulation
                (42, "TVA-DEPTH"),  # light sustain/amp-envelope depth style movement
            ])
        modulation_routes = []
        partials_raw = []

        for partial in range(4):
            block = _addr_add(self._zcore_base(part_idx), [0x00, 0x30 + (partial * 2), 0x00])
            lfo1_wave = random.choice(wave_choices)
            lfo2_wave = random.choice([w for w in wave_choices if w != lfo1_wave])
            lfo1_note = random.choice(note_choices)
            lfo2_note = random.choice(note_choices)
            lfo1_tvf = random.randint(max(4, role_depth - 10), role_depth)
            lfo1_tva = random.randint(max(3, role_depth - 14), max(5, role_depth - 4))
            lfo2_tvf = random.randint(max(4, role_depth - 8), role_depth)
            lfo2_tva = random.randint(max(3, role_depth - 14), max(5, role_depth - 5))

            # LFO1
            self._send_dt1(_addr_add(block, [0x00, 0x00]), [lfo1_wave])
            self._send_dt1(_addr_add(block, [0x00, 0x01]), [1])
            self._send_dt1(_addr_add(block, [0x00, 0x02]), [lfo1_note])
            self._send_dt1(_addr_add(block, [0x00, 0x19]), _nibbles(_signed_100(lfo1_tvf), 2))
            self._send_dt1(_addr_add(block, [0x00, 0x1B]), _nibbles(_signed_100(lfo1_tva), 2))

            # LFO2
            self._send_dt1(_addr_add(block, [0x00, 0x4F]), [lfo2_wave])
            self._send_dt1(_addr_add(block, [0x00, 0x50]), [1])
            self._send_dt1(_addr_add(block, [0x00, 0x51]), [lfo2_note])
            self._send_dt1(_addr_add(block, [0x00, 0x68]), _nibbles(_signed_100(lfo2_tvf), 2))
            self._send_dt1(_addr_add(block, [0x00, 0x6A]), _nibbles(_signed_100(lfo2_tva), 2))

            # Matrix 1: LFO1 modulates LFO2 rate plus one safe destination.
            extra1 = random.choice([d for d in safe_destinations if d[0] != 17])
            sens1 = random.randint(6, 14)
            def matrix_sens(dest_code: int, lo: int, hi: int) -> int:
                if dest_code in {27, 28, 42}:
                    return random.randint(2, 5)
                if dest_code in {37, 39}:
                    return random.randint(2, 6)
                return random.randint(lo, hi)

            sens1b = matrix_sens(extra1[0], 3, 9)
            self._send_dt1(_addr_add(block, [0x00, 0x56]), [104])  # Source LFO1
            self._send_dt1(_addr_add(block, [0x00, 0x57]), [17])   # Destination LFO2-RATE
            self._send_dt1(_addr_add(block, [0x00, 0x58]), [_signed_63(sens1)])
            self._send_dt1(_addr_add(block, [0x00, 0x59]), [extra1[0]])
            self._send_dt1(_addr_add(block, [0x00, 0x5A]), [_signed_63(sens1b)])

            # Matrix 2: LFO2 modulates two safe tone-shaping destinations.
            dest2 = random.sample(safe_destinations, 2)
            sens2a = matrix_sens(dest2[0][0], 5, 12)
            sens2b = matrix_sens(dest2[1][0], 3, 10)
            self._send_dt1(_addr_add(block, [0x00, 0x5F]), [105])  # Source LFO2
            self._send_dt1(_addr_add(block, [0x00, 0x60]), [dest2[0][0]])
            self._send_dt1(_addr_add(block, [0x00, 0x61]), [_signed_63(sens2a)])
            self._send_dt1(_addr_add(block, [0x00, 0x62]), [dest2[1][0]])
            self._send_dt1(_addr_add(block, [0x00, 0x63]), [_signed_63(sens2b)])

            partial_raw = {
                "lfo1_wave": lfo1_wave, "lfo1_note": lfo1_note,
                "lfo1_tvf": lfo1_tvf, "lfo1_tva": lfo1_tva,
                "lfo2_wave": lfo2_wave, "lfo2_note": lfo2_note,
                "lfo2_tvf": lfo2_tvf, "lfo2_tva": lfo2_tva,
                "matrix": {
                    "m1_dest1": 17, "m1_sens1": sens1,
                    "m1_dest2": extra1[0], "m1_sens2": sens1b,
                    "m2_dest1": dest2[0][0], "m2_sens1": sens2a,
                    "m2_dest2": dest2[1][0], "m2_sens2": sens2b,
                },
                "fxm_enabled": False,
            }

            # Oscillator FXM for melody patches — enable on all partials
            if is_melody:
                fxm_color = random.randint(2, 8)    # max ~6% of 127
                fxm_depth = random.randint(4, 31)   # max ~24% of 127
                self._send_dt1(_addr_add(block, [0x00, 0x06]), [1])       # FXM Switch ON
                self._send_dt1(_addr_add(block, [0x00, 0x07]), [fxm_color])
                self._send_dt1(_addr_add(block, [0x00, 0x08]), [fxm_depth])

                # Oscillator Delay Mode, Delay Time, and FAT
                osc_delay_mode = random.choice([0, 1, 2])     # OFF, Normal, Random
                osc_delay_time = random.randint(0, 40)         # moderate delay
                osc_fat = random.randint(2, 16)                # subtle to moderate detune
                self._send_dt1(_addr_add(block, [0x00, 0x09]), [osc_fat])
                self._send_dt1(_addr_add(block, [0x00, 0x0A]), [osc_delay_mode])
                self._send_dt1(_addr_add(block, [0x00, 0x0B]), [osc_delay_time])

                partial_raw["fxm_enabled"] = True
                partial_raw["fxm_color"] = fxm_color
                partial_raw["fxm_depth"] = fxm_depth
                partial_raw["osc_fat"] = osc_fat
                partial_raw["osc_delay_mode"] = osc_delay_mode
                partial_raw["osc_delay_time"] = osc_delay_time

            partials_raw.append(partial_raw)

            if partial == 0:
                modulation_routes = [
                    f"LFO1 {wave_names.get(lfo1_wave, lfo1_wave)}@{note_names.get(lfo1_note, lfo1_note)} TVF={lfo1_tvf} TVA={lfo1_tva}",
                    f"LFO2 {wave_names.get(lfo2_wave, lfo2_wave)}@{note_names.get(lfo2_note, lfo2_note)} TVF={lfo2_tvf} TVA={lfo2_tva}",
                    f"LFO1->LFO2-RATE sens=+{sens1}",
                    f"LFO1->{extra1[1]} sens=+{sens1b}",
                    f"LFO2->{dest2[0][1]} sens=+{sens2a}",
                    f"LFO2->{dest2[1][1]} sens=+{sens2b}",
                ]
                if is_melody:
                    modulation_routes.append(
                        f"FXM ON color={fxm_color} depth={fxm_depth} FAT={osc_fat} delay_mode={osc_delay_mode} delay_time={osc_delay_time}"
                    )

        label = "LFO matrix: " + "; ".join(modulation_routes)
        raw = {"partials": partials_raw}
        return label, raw

    def build_melody_sysex_automation_track(self, part_idx: int, track_name: str,
                                            ticks_per_beat: int, bar_ticks: int,
                                            count_in_ticks: int,
                                            total_bars: int = 72) -> tuple:
        """
        Build timed Fantom SysEx automation for melody-like Z-Core parts.

        The generated MIDI track is intended to be baked into the exact
        recording-pass MIDI. It modulates safe matrix destinations rather than
        changing patch programs during playback.
        """
        if bar_ticks <= 0:
            return None, {"enabled": False, "skip_reason": "invalid_bar_ticks"}
        role = self._melody_automation_role(track_name)
        portamento_metadata = self._build_portamento_profile(track_name)
        if role is None and not portamento_metadata.get("enabled"):
            return None, portamento_metadata

        track = mido.MidiTrack()
        safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", track_name).strip("_") or "melody"
        track.name = f"automation_{part_idx + 1}_{safe_label}"
        events = []
        manifest_events = []

        def abs_tick_for_bar(bar_float: float) -> int:
            return int(round(count_in_ticks + max(0.0, bar_float) * bar_ticks))

        def add_portamento_event(bar_float: float, label: str, profile: Dict):
            tick = abs_tick_for_bar(bar_float)
            serial = 0
            for address, data, desc in self._portamento_sysex_messages(part_idx, profile):
                events.append((tick + serial, create_roland_sysex(address, data)))
                serial += 1
            manifest_events.append({
                "label": label,
                "bar": round(float(bar_float), 3),
                "intensity": "portamento",
                **profile,
            })

        def add_matrix_event(bar_float: float, label: str, cut: int, res: int,
                             rel: int, dcy: int, delay_time: int,
                             fat: int, intensity: str):
            tick = abs_tick_for_bar(bar_float)
            values = {
                "cutoff_depth": _clamp_signed_depth(cut, 18),
                "resonance_depth": _clamp_signed_depth(res, 4),
                "amp_release_depth": _clamp_signed_depth(rel, 7),
                "amp_decay_depth": _clamp_signed_depth(dcy, 7),
                "osc_delay_time": max(0, min(48, int(delay_time))),
                "osc_fat": max(0, min(18, int(fat))),
            }
            serial = 0
            for partial in range(2):
                block = _addr_add(self._zcore_base(part_idx), [0x00, 0x30 + (partial * 2), 0x00])
                # Matrix 1: LFO1 -> cutoff and amp release.
                for offset, data in [
                    ([0x00, 0x56], [104]),
                    ([0x00, 0x57], [2]),   # CUT
                    ([0x00, 0x58], [_signed_63(values["cutoff_depth"])]),
                    ([0x00, 0x59], [28]),  # TVA-REL
                    ([0x00, 0x5A], [_signed_63(values["amp_release_depth"])]),
                    ([0x00, 0x5F], [105]),
                    ([0x00, 0x60], [3]),   # RES
                    ([0x00, 0x61], [_signed_63(values["resonance_depth"])]),
                    ([0x00, 0x62], [27]),  # TVA-DCY
                    ([0x00, 0x63], [_signed_63(values["amp_decay_depth"])]),
                    ([0x00, 0x09], [values["osc_fat"]]),
                    ([0x00, 0x0B], [values["osc_delay_time"]]),
                ]:
                    events.append((tick + serial, create_roland_sysex(_addr_add(block, offset), data)))
                    serial += 1
            manifest_events.append({
                "label": label,
                "bar": round(float(bar_float), 3),
                "intensity": intensity,
                **values,
            })

        if portamento_metadata.get("enabled"):
            add_portamento_event(0, "initial_portamento_setup", portamento_metadata)

        if role is None:
            if not events:
                return None, portamento_metadata
            events.sort(key=lambda item: item[0])
            last_tick = 0
            track.append(mido.MetaMessage('track_name', name=track.name, time=0))
            for tick, msg in events:
                delta = max(0, tick - last_tick)
                track.append(msg.copy(time=delta))
                last_tick = tick
            end_tick = count_in_ticks + (total_bars * bar_ticks)
            track.append(mido.MetaMessage('end_of_track', time=max(0, end_tick - last_tick)))
            return track, {
                **portamento_metadata,
                "track_name": track_name,
                "automation_track": track.name,
                "part": part_idx + 1,
                "role": "portamento_only",
                "bar_ticks": bar_ticks,
                "count_in_ticks": count_in_ticks,
                "event_count": len(events),
                "portamento_automation": portamento_metadata,
            }

        add_matrix_event(0, "initial_safe_baseline", 3, 0, 1, 1, 6, 3, "baseline")

        for section, start, end in self._melody_automation_sections(role, total_bars):
            span = max(1, end - start)
            if section.startswith("verse"):
                add_matrix_event(start, f"{section}_start", 4, 0, 1, 1, 7, 3, "subtle")
                for bar in range(start + 4, end, 4):
                    is_eight = (bar - start) % 8 == 0
                    add_matrix_event(
                        bar - 0.15,
                        f"{section}_{'8bar' if is_eight else '4bar'}_pulse_open",
                        8 if is_eight else 6,
                        2 if is_eight else 1,
                        3 if is_eight else 2,
                        2,
                        10 if is_eight else 8,
                        5 if is_eight else 4,
                        "subtle_8bar" if is_eight else "subtle_4bar",
                    )
                    add_matrix_event(
                        bar + 0.55,
                        f"{section}_{'8bar' if is_eight else '4bar'}_pulse_settle",
                        4,
                        0,
                        1,
                        1,
                        7,
                        3,
                        "settle",
                    )
                add_matrix_event(end - 0.5, f"{section}_energy_top", 9, 2, 3, 3, 11, 5, "medium")
            elif section.startswith("chorus"):
                add_matrix_event(start, f"{section}_open", 8, 1, 2, 2, 10, 5, "medium")
                add_matrix_event(start + span * 0.50, f"{section}_mid_lift", 12, 2, 4, 3, 14, 7, "medium")
                add_matrix_event(end - 0.5, f"{section}_peak", 14 if section == "chorus2" else 12, 3, 5, 4, 16, 8, "strong")
            elif section == "outro":
                add_matrix_event(start, "outro_warm_entry", 6, 1, 3, 2, 10, 4, "subtle")
                add_matrix_event(end - 0.75, "outro_close", 1, 0, 2, 1, 4, 2, "closing")

        for label, start, end, direction, strength in self._transition_sweeps(role, total_bars):
            if direction == "down":
                add_matrix_event(start, f"{label}_close_start", 8, 1, 3, 2, 10, 5, "transition")
                add_matrix_event((start + end) / 2.0, f"{label}_close_mid", 4, 1, 2, 1, 7, 3, "transition")
                add_matrix_event(end, f"{label}_close_resolve", 1, 0, 1, 1, 4, 2, "transition_resolve")
            else:
                peak_cut = 16 if strength == "strong" else 12
                add_matrix_event(start, f"{label}_sweep_start", 4, 0, 1, 1, 7, 3, "transition")
                add_matrix_event((start + end) / 2.0, f"{label}_sweep_mid", peak_cut - 3, 2, 3, 3, 12, 6, "transition")
                add_matrix_event(end, f"{label}_sweep_resolve", peak_cut, 3 if strength == "strong" else 2, 4, 3, 15, 7, "transition_resolve")

        if not events:
            return None, {"enabled": False, "skip_reason": "no_automation_events"}

        events.sort(key=lambda item: item[0])
        last_tick = 0
        track.append(mido.MetaMessage('track_name', name=track.name, time=0))
        for tick, msg in events:
            delta = max(0, tick - last_tick)
            track.append(msg.copy(time=delta))
            last_tick = tick
        end_tick = count_in_ticks + (total_bars * bar_ticks)
        track.append(mido.MetaMessage('end_of_track', time=max(0, end_tick - last_tick)))
        metadata = {
            "enabled": True,
            "track_name": track_name,
            "automation_track": track.name,
            "part": part_idx + 1,
            "role": role,
            "bar_ticks": bar_ticks,
            "count_in_ticks": count_in_ticks,
            "event_count": len(events),
            "filter_sweep_automation": [e for e in manifest_events if "sweep" in e["label"] or "transition" in e["intensity"]],
            "filter_resonance_automation": {
                "max_depth": max(
                    abs(e["resonance_depth"])
                    for e in manifest_events
                    if "resonance_depth" in e
                ),
                "policy": "very_subtle_clamped",
            },
            "envelope_automation": [e for e in manifest_events if "amp_release_depth" in e],
            "portamento_automation": portamento_metadata,
        }
        return track, metadata

    def _is_portamento_target_name(self, track_name: str) -> bool:
        n = (track_name or "").lower().replace("_", " ")
        return any(x in n for x in [
            "main melody", "counter melody", "chorus melody", "melody",
            "lead", "bass", "harmonic bass",
        ])

    def _build_portamento_profile(self, track_name: str) -> Dict:
        if not self._is_portamento_target_name(track_name):
            return {"enabled": False, "skip_reason": "not_portamento_target"}
        if random.random() >= 0.20:
            return {"enabled": False, "skip_reason": "probability_gate", "probability": 0.20}

        time_percent = random.uniform(8.0, 25.0)
        mode = random.choice(["normal", "legato"])
        porta_type = random.choice(["rate", "time"])
        start = random.choice(["pitch", "note"])
        curve_type = random.choice([1, 2, 3])
        return {
            "enabled": True,
            "probability": 0.20,
            "target": "bass" if "bass" in (track_name or "").lower() else "melody",
            "mode": mode,
            "type": porta_type,
            "start": start,
            "curve_type": curve_type,
            "time_percent": round(time_percent, 2),
            "zone_time": _percent_to_127(time_percent),
            "tone_time": _percent_to_1023(time_percent),
            "sysex_scope": "scene_zone_and_tone_common" if _TONE_PORTAMENTO_SYSEX_ENABLED else "scene_zone",
            "tone_common_sysex_enabled": _TONE_PORTAMENTO_SYSEX_ENABLED,
        }

    def _portamento_sysex_messages(self, part_idx: int, profile: Dict) -> List[tuple]:
        zone = part_idx + 1
        zone_base = self._scene_zone_base(zone)
        mode_value = 0 if profile.get("mode") == "normal" else 1
        type_value = 0 if profile.get("type") == "rate" else 1
        start_value = 0 if profile.get("start") == "pitch" else 1
        curve_value = max(0, min(2, int(profile.get("curve_type", 1)) - 1))
        zone_time = max(0, min(127, int(profile.get("zone_time", 16))))
        tone_time = max(0, min(1023, int(profile.get("tone_time", 160))))

        # Scene Zone portamento switch/time: keeps the recording part explicit.
        messages = [
            (_addr_add(zone_base, [0x00, 0x26]), [1], "zone_portamento_switch_on"),
            (_addr_add(zone_base, [0x00, 0x27]), [zone_time], "zone_portamento_time"),
        ]

        if not _TONE_PORTAMENTO_SYSEX_ENABLED:
            return messages

        tone_base = self._zcore_base(part_idx)
        # Z-Core common portamento parameters. Values follow the Fantom
        # parameter guide: switch, mode NORMAL/LEGATO, type RATE/TIME,
        # start PITCH/NOTE, time 0-1023, curve 1-3.
        for offset, data, desc in [
            ([0x00, 0x07], [1], "tone_portamento_switch_on"),
            ([0x00, 0x08], [mode_value], "tone_portamento_mode"),
            ([0x00, 0x09], [type_value], "tone_portamento_type"),
            ([0x00, 0x0A], [start_value], "tone_portamento_start"),
            ([0x00, 0x0B], _nibbles(tone_time, 4), "tone_portamento_time"),
            ([0x00, 0x0F], [curve_value], "tone_portamento_curve"),
        ]:
            messages.append((_addr_add(tone_base, offset), data, desc))
        return messages

    def reset_zone_portamento(self, zone: int, delay: float = 0.006):
        """Disable Scene Zone portamento and reset its time for calibration."""
        if not self.output:
            return
        zone_base = self._scene_zone_base(zone)
        self._send_dt1(_addr_add(zone_base, [0x00, 0x26]), [0], delay=delay)
        self._send_dt1(_addr_add(zone_base, [0x00, 0x27]), [0], delay=delay)

    def _melody_automation_role(self, track_name: str) -> Optional[str]:
        n = (track_name or "").lower()
        if "counter" in n or "fx" in n:
            return None
        if "chorus_melody" in n or "chorus melody" in n:
            return "chorus"
        if "main_melody" in n or "main melody" in n:
            return "main"
        return None

    def _melody_automation_sections(self, role: str, total_bars: int) -> List[tuple]:
        sections = {
            "main": [("verse1", 8, 24), ("verse2", 36, 52), ("outro", 64, min(72, total_bars))],
            "chorus": [("chorus1", 24, 32), ("chorus2", 52, 60)],
        }
        return [(name, start, min(end, total_bars)) for name, start, end in sections.get(role, []) if start < total_bars]

    def _transition_sweeps(self, role: str, total_bars: int) -> List[tuple]:
        all_sweeps = [
            ("intro_to_verse", 6.0, 8.0, "up", "medium", {"main"}),
            ("verse1_to_chorus1", 22.0, 24.0, "up", "strong", {"chorus"}),
            ("chorus1_to_fill1", 30.5, 32.0, "up", "medium", {"chorus"}),
            ("fill1_to_verse2", 34.5, 36.0, "up", "medium", {"main"}),
            ("verse2_to_chorus2", 50.0, 52.0, "up", "strong", {"chorus"}),
            ("chorus2_to_fill2", 58.5, 60.0, "up", "medium", {"chorus"}),
            ("fill2_to_outro", 62.0, 64.0, "down", "medium", {"main"}),
        ]
        return [
            (label, start, min(end, float(total_bars)), direction, strength)
            for label, start, end, direction, strength, roles in all_sweeps
            if role in roles and start < total_bars
        ]

    def set_part_level(self, channel: int, level: int):
        if not self.output: return
        self.output.send(mido.Message('control_change', channel=channel, control=7, value=level))

    def _scene_zone_base(self, zone: int) -> List[int]:
        """Scene Zone base address for 1-based Fantom Zone numbers."""
        zone = max(1, min(16, int(zone)))
        return [0x02, 0x00, 0x10 + (zone - 1), 0x00]

    def set_zone_receive_pitch_bend(self, zone: int, enabled: bool = True):
        """Enable/disable Pitch Bend reception for a Scene Zone."""
        if not self.output: return
        self._send_dt1(_addr_add(self._scene_zone_base(zone), [0x00, 0x39]),
                       [1 if enabled else 0])

    def set_zone_scale_tune(self, zone: int, key_pc: int, cent_offsets: Dict[int, int]) -> Dict:
        """
        Apply a CUSTOM Scene Zone Scale Tune.

        zone is 1-based. cent_offsets maps absolute pitch classes 0=C..11=B
        to cent offsets. The Fantom stores C-B offsets as -64..+63 encoded
        0..127, and uses a separate scale-tune key parameter.
        """
        if not self.output:
            return {}
        key_pc = max(0, min(11, int(key_pc)))
        normalized = {pc % 12: max(-64, min(63, int(round(cents))))
                      for pc, cents in (cent_offsets or {}).items()
                      if int(round(cents)) != 0}
        base = self._scene_zone_base(zone)
        self._send_dt1(_addr_add(base, [0x00, 0x29]), [0])       # CUSTOM
        self._send_dt1(_addr_add(base, [0x00, 0x2A]), [key_pc])  # C..B
        for pc in range(12):
            self._send_dt1(_addr_add(base, [0x00, 0x2B + pc]),
                           [_scale_tune_value(normalized.get(pc, 0))])
        self.set_zone_receive_pitch_bend(zone, True)
        return {
            "type": "custom_scale_tune",
            "zone": zone,
            "key_pc": key_pc,
            "cent_offsets": {str(pc): cents for pc, cents in sorted(normalized.items())},
        }

    def reset_zone_scale_tune(self, zone: int) -> Dict:
        """Return a Scene Zone to equal temperament scale tune."""
        if not self.output:
            return {}
        base = self._scene_zone_base(zone)
        self._send_dt1(_addr_add(base, [0x00, 0x29]), [1])  # EQUAL
        for pc in range(12):
            self._send_dt1(_addr_add(base, [0x00, 0x2B + pc]), [_scale_tune_value(0)])
        self.set_zone_receive_pitch_bend(zone, True)
        return {"type": "equal_scale_tune", "zone": zone}

    def setup_scene(self, patch_assignments: List[Dict]):
        print("Configuring Roland Fantom Scene with Random Tones...")
        for p in patch_assignments:
            print(f"  Channel {p['channel']}: {p['name']} (MSB:{p['msb']} LSB:{p['lsb']} PC:{p['pc']})")
            self.select_patch(p['channel'], p['msb'], p['lsb'], p['pc'])
            self.set_part_level(p['channel'], 100)
        print("Fantom configuration complete.")

    def set_zone_eq_switch(self, zone: int, on: bool):
        """
        Enable/Disable EQ for a specific Zone (1-16).
        Address: 02 00 2(zone-1) 08
        """
        if not self.output: return
        addr = [0x02, 0x00, 0x20 + (zone - 1), 0x08]
        data = [1 if on else 0]
        self.output.send(create_roland_sysex(addr, data))

    def set_zone_eq_gain(self, zone: int, band: str, gain_db: float):
        """
        Set EQ gain for a specific band and zone.
        bands: 'input' (00), 'low' (01), 'mid' (02), 'high' (03)
        gain_db: -24.0 to +24.0 (maps to 40-88)
        """
        if not self.output: return
        band_map = {'input': 0x00, 'low': 0x01, 'mid': 0x02, 'high': 0x03}
        if band not in band_map: return
        
        offset = band_map[band]
        addr = [0x02, 0x00, 0x20 + (zone - 1), offset]
        
        # Map -24..+24 to 40..88 (0.5dB steps?)
        # Guide says 40-88, so 48 units for 48dB -> 1 unit = 1dB. 64 = 0dB.
        val = int(max(40, min(88, 64 + gain_db)))
        self.output.send(create_roland_sysex(addr, [val]))
