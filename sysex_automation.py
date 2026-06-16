"""
SYSEX AUTOMATION ENGINE — Real-time parameter modulation during recording.
Sends SysEx messages to the Fantom at specific bar positions to create
dynamic movement in LFO rate, filter cutoff, MFX, and modulation matrix.

This is what makes tech house sound alive — parameters changing during
buildups, breakdowns, and drops.
"""
import mido
import time
import random

# ============================================================================
# FANTOM SYSEX BASE ADDRESSES (from skill 09)
# ============================================================================
# Zone parameters (relative to zone base for part N)
ZONE_BASE = {
    'tva_level':    [0x00, 0x00, 0x06],  # Part volume
    'tvf_cutoff':   [0x00, 0x00, 0x00],  # Filter cutoff
    'tvf_reso':     [0x00, 0x00, 0x01],  # Filter resonance
    'lfo1_rate':    [0x00, 0x00, 0x18],  # LFO1 speed
    'lfo1_depth':   [0x00, 0x00, 0x1A],  # LFO1 depth (to TVF cutoff)
    'lfo1_wave':    [0x00, 0x00, 0x10],  # LFO1 waveform
    'lfo2_rate':    [0x00, 0x30, 0x18],  # LFO2 speed
    'lfo2_depth':   [0x00, 0x30, 0x1A],  # LFO2 depth
    'lfo2_dest':    [0x00, 0x30, 0x0C],  # LFO2 destination
    'pan':          [0x00, 0x00, 0x04],  # Pan
    'pitch':        [0x00, 0x00, 0x02],  # Pitch coarse
}

# LFO waveform types
LFO_WAVES = {
    'sin': 1, 'tri': 2, 'saw': 3, 'sqr': 4,
    'trp': 5, 'sah': 6, 'rnd': 7,
}

# LFO destinations
LFO_DESTS = {
    'cutoff': 1, 'level': 2, 'lfo2_rate': 3, 'pan': 4,
    'pitch': 5, 'cutoff2': 6, 'reso': 7, 'fat': 8,
    'pwm': 9, 'xmod': 10, 'mfx': 11,
}

# ============================================================================
# AUTOMATION CURVE DEFINITIONS
# ============================================================================
def _zcore_base(part_idx):
    """Calculate zone core base address for a part (0-indexed)."""
    return [0x10, 0x00, 0x00, 0x00 + part_idx * 0x10]

def _addr_add(base, offset):
    """Add offset to base address."""
    return [base[i] + offset[i] for i in range(min(len(base), len(offset)))]

def build_buildup_automation(part_idx, role, total_bars, bpm):
    """
    Build automation events for buildups and drops.
    Returns list of (bar, label, sysex_bytes) tuples.
    """
    events = []
    base = _zcore_base(part_idx)
    
    # Bar timings for 128-bar arrangement
    drop1_start = 16
    breakdown_start = 48
    drop2_start = 80
    outro_start = 112
    
    if role in ('acid', 'bass', 'sub_bass', 'stab', 'pad'):
        # ============================================================
        # A-A-A-B SWITCH-UP AUTOMATION — Aggressive modulation on B bars
        # ONLY during drops (not intro/outro/breakdown)
        # ============================================================
        
        for bar in range(drop1_start, drop2_start + 32):  # Drop1 + Drop2 only
            bar_in_phrase = bar % 4
            if bar_in_phrase == 3:  # B bar — switch-up (subtle, not aggressive)
                # Filter cutoff: gentle increase (not spike)
                cutoff_val = random.randint(80, 95)
                addr = _addr_add(base, ZONE_BASE['tvf_cutoff'])
                events.append((bar, 'switchup_cutoff',
                              _sysex_dt1(addr, [cutoff_val])))
                
                # LFO depth: subtle increase (not extreme)
                depth_val = random.randint(25, 35)
                addr = _addr_add(base, ZONE_BASE['lfo1_depth'])
                events.append((bar, 'switchup_lfo',
                              _sysex_dt1(addr, [depth_val])))
            
            elif bar_in_phrase == 0:  # Back to A — reset to normal
                addr = _addr_add(base, ZONE_BASE['tvf_cutoff'])
                events.append((bar, 'reset_cutoff',
                              _sysex_dt1(addr, [random.randint(70, 80)])))
                
                addr = _addr_add(base, ZONE_BASE['lfo1_depth'])
                events.append((bar, 'reset_lfo',
                              _sysex_dt1(addr, [random.randint(15, 20)])))
        
        # ============================================================
        # FILTER CUTOFF SWEEP — Opening filter during builds
        # ============================================================
        
        # Breakdown: start with closed filter, gradually open
        # Bars 48-63: filter slowly opening (60 → 100)
        for bar in range(breakdown_start, breakdown_start + 16):
            progress = (bar - breakdown_start) / 15.0
            cutoff = int(60 + progress * 40)  # 60 → 100
            addr = _addr_add(base, ZONE_BASE['tvf_cutoff'])
            events.append((bar, 'filter_open', 
                          _sysex_dt1(addr, [cutoff])))
        
        # Bars 64-75: filter fully open, resonance increasing
        for bar in range(breakdown_start + 16, breakdown_start + 28):
            progress = (bar - breakdown_start - 16) / 11.0
            reso = int(20 + progress * 40)  # 20 → 60
            addr = _addr_add(base, ZONE_BASE['tvf_reso'])
            events.append((bar, 'reso_build', 
                          _sysex_dt1(addr, [reso])))
        
        # Bar 76: filter snap closed before drop (tension!)
        addr = _addr_add(base, ZONE_BASE['tvf_cutoff'])
        events.append((breakdown_start + 28, 'filter_snap_closed',
                      _sysex_dt1(addr, [40])))
        
        # Bar 80 (drop2): filter blasts open
        addr = _addr_add(base, ZONE_BASE['tvf_cutoff'])
        events.append((drop2_start, 'filter_drop_open',
                      _sysex_dt1(addr, [110])))
        
        # Reset resonance at drop
        addr = _addr_add(base, ZONE_BASE['tvf_reso'])
        events.append((drop2_start, 'reso_reset',
                      _sysex_dt1(addr, [20])))
        
        # ============================================================
        # LFO RATE MODULATION — Speed up during builds
        # ============================================================
        
        if role == 'acid':
            # Acid: LFO rate speeds up dramatically during breakdown
            for bar in range(breakdown_start + 8, breakdown_start + 28):
                progress = (bar - breakdown_start - 8) / 19.0
                lfo_rate = int(4 + progress * 8)  # 4 → 12 (slow → fast)
                addr = _addr_add(base, ZONE_BASE['lfo1_rate'])
                events.append((bar, 'acid_lfo_speedup',
                              _sysex_dt1(addr, [lfo_rate])))
            
            # Reset LFO rate at drop
            addr = _addr_add(base, ZONE_BASE['lfo1_rate'])
            events.append((drop2_start, 'acid_lfo_reset',
                          _sysex_dt1(addr, [6])))
        
        # ============================================================
        # LFO DEPTH MODULATION — Increase modulation intensity
        # ============================================================
        
        # Breakdown: LFO depth increases gradually (subtle wobble)
        for bar in range(breakdown_start + 4, breakdown_start + 28):
            progress = (bar - breakdown_start - 4) / 23.0
            depth = int(15 + progress * 20)  # 15 → 35 (subtle, not extreme)
            addr = _addr_add(base, ZONE_BASE['lfo1_depth'])
            events.append((bar, 'lfo_depth_build',
                          _sysex_dt1(addr, [depth])))
        
        # Reset at drop
        addr = _addr_add(base, ZONE_BASE['lfo1_depth'])
        events.append((drop2_start, 'lfo_depth_reset',
                      _sysex_dt1(addr, [20])))
        
        # ============================================================
        # PAN MODULATION — Stereo movement during builds
        # ============================================================
        
        if role in ('stab', 'pad'):
            # Stabs/pads: pan slowly during breakdown
            for bar in range(breakdown_start + 8, breakdown_start + 24, 2):
                progress = (bar - breakdown_start - 8) / 15.0
                # Oscillate pan: center → left → right → center
                import math
                pan_val = int(64 + 30 * math.sin(progress * math.pi * 2))
                addr = _addr_add(base, ZONE_BASE['pan'])
                events.append((bar, 'pan_sweep',
                              _sysex_dt1(addr, [pan_val])))
            
            # Reset to center at drop
            addr = _addr_add(base, ZONE_BASE['pan'])
            events.append((drop2_start, 'pan_reset',
                          _sysex_dt1(addr, [64])))
    
    # ============================================================
    # DRUM AUTOMATION — Open hat filter, ride filter
    # ============================================================
    if role in ('open_hat', 'closed_hat', 'ride', 'crash'):
        # Buildup before drop1: hats get brighter
        for bar in range(8, drop1_start):
            progress = (bar - 8) / (drop1_start - 8 - 1)
            cutoff = int(80 + progress * 30)  # 80 → 110
            addr = _addr_add(base, ZONE_BASE['tvf_cutoff'])
            events.append((bar, 'hat_brighten',
                          _sysex_dt1(addr, [cutoff])))
        
        # Buildup before drop2: hats get brighter again
        for bar in range(breakdown_start + 20, drop2_start):
            progress = (bar - breakdown_start - 20) / (drop2_start - breakdown_start - 20 - 1)
            cutoff = int(70 + progress * 40)  # 70 → 110
            addr = _addr_add(base, ZONE_BASE['tvf_cutoff'])
            events.append((bar, 'hat_brighten_drop2',
                          _sysex_dt1(addr, [cutoff])))
    
    return events


def _sysex_dt1(address, data):
    """Build a Roland SysEx DT1 message."""
    msg = [0x41, 0x10, 0x00, 0x00, 0x7F]  # Roland header
    msg.extend(address)
    msg.extend(data)
    # Calculate checksum
    chk = 0
    for b in address + data:
        chk = (chk + b) & 0x7F
    chk = (128 - chk) & 0x7F
    msg.append(chk)
    return mido.Message('sysex', data=msg)


# ============================================================================
# AUTOMATION SCHEDULER
# ============================================================================
class AutomationScheduler:
    """Schedules and fires SysEx automation events during playback."""
    
    def __init__(self, outport, automation_events, bpm, tpb=480):
        self.outport = outport
        self.events = sorted(automation_events, key=lambda e: e[0])
        self.bpm = bpm
        self.tpb = tpb
        self.bar_duration = 60.0 / bpm * 4  # seconds per bar
        self.event_idx = 0
        self.fired = set()
    
    def check_and_fire(self, current_time_sec, count_in_sec=0):
        """Check if any automation events should fire at the current time."""
        # Current bar position (after count-in)
        elapsed = current_time_sec - count_in_sec
        if elapsed < 0:
            return
        
        current_bar = elapsed / self.bar_duration
        
        # Fire all events up to current bar
        while self.event_idx < len(self.events):
            bar, label, sysex = self.events[self.event_idx]
            if bar <= current_bar:
                event_id = f"{bar}_{label}"
                if event_id not in self.fired:
                    try:
                        self.outport.send(sysex)
                        self.fired.add(event_id)
                    except:
                        pass
                self.event_idx += 1
            else:
                break
    
    def get_stats(self):
        """Return stats about automation events."""
        return {
            'total_events': len(self.events),
            'fired_events': len(self.fired),
            'labels': list(set(e[1] for e in self.events)),
        }


# ============================================================================
# MAIN: BUILD ALL AUTOMATION FOR A SONG
# ============================================================================
def build_song_automation(track_info_list, total_bars, bpm):
    """
    Build automation events for all tracks in a song.
    
    track_info_list: list of dicts with 'part_idx', 'role', 'category'
    Returns: list of (bar, label, sysex_bytes) for all tracks
    """
    all_events = []
    
    for info in track_info_list:
        part_idx = info.get('part_idx', 0)
        role = info.get('category', info.get('role', 'other'))
        
        events = build_buildup_automation(part_idx, role, total_bars, bpm)
        all_events.extend(events)
    
    # Sort by bar number
    all_events.sort(key=lambda e: e[0])
    
    return all_events


if __name__ == '__main__':
    # Test: print automation events for a typical track
    events = build_buildup_automation(0, 'acid', 128, 128)
    print(f"Generated {len(events)} automation events for acid track:")
    for bar, label, _ in events[:20]:
        print(f"  Bar {bar:3d}: {label}")
    if len(events) > 20:
        print(f"  ... and {len(events) - 20} more")
