
import json
import os
import re
import numpy as np

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

def apply_arrangement_envelope(y, sr, bpm, stem_name, role, layer):
    y = ensure_stereo(np.asarray(y, dtype=np.float32))
    seed = int(os.environ.get("ARRANGEMENT_DENSITY_SEED", "0") or 0)
    beats_per_bar = beats_per_bar_from_name(stem_name)
    samples_per_bar = max(1, int(sr * (60.0 / max(float(bpm), 1.0)) * beats_per_bar))
    policy = choose_policy(seed)
    gain_db = np.zeros(len(y), dtype=np.float32)
    events = []
    pink_decisions = []
    for section, (bar_start, bar_end) in SECTION_BARS.items():
        start = min(len(y), bar_start * samples_per_bar)
        end = min(len(y), bar_end * samples_per_bar)
        if end <= start:
            continue
        sec_gain, decision = section_gain_db(
            stem_name, role, layer, section, policy, bar_start, bar_end, y[start:end], sr
        )
        gain_db[start:end] += float(sec_gain)
        events.append({"section": section, "bars": [bar_start, bar_end], "gain_db": float(sec_gain)})
        pink_decisions.append(decision)
    curve = np.power(10.0, gain_db / 20.0).astype(np.float32)
    curve = smooth_edges(curve, max(32, int(sr * 0.035)))
    out = y * curve[:, None]
    report = {
        "stem_name": stem_name,
        "role": role,
        "layer": layer,
        "seed": seed,
        "beats_per_bar": beats_per_bar,
        "policy": policy,
        "events": events,
        "pink_decisions": pink_decisions,
        "section_metrics": section_metrics(out, sr, bpm, beats_per_bar),
    }
    log_path = os.environ.get("ARRANGEMENT_DENSITY_LOG")
    if log_path:
        with open(log_path, "a") as f:
            f.write(json.dumps(report, default=str) + "\n")
    return out.astype(np.float32), report

def choose_policy(seed):
    rng = np.random.default_rng(seed)
    bass1_groups = ["intro", "verse", "pre_chorus", "outro"]
    bass2_groups = ["chorus", "fill"]
    if bool(rng.integers(0, 2)):
        bass1_groups, bass2_groups = bass2_groups, bass1_groups
    layer2_verse_start = int(rng.choice([12, 14, 16]))
    layer3_build_start = int(rng.choice([20, 22, 48, 50]))
    fx_fill_gain_db = float(rng.choice([-4.5, -3.5, -2.5]))
    return {
        "bass1_groups": bass1_groups,
        "bass2_groups": bass2_groups,
        "bass_owners": {
            "intro": "bass1" if "intro" in bass1_groups else "bass2",
            "verse": "bass1" if "verse" in bass1_groups else "bass2",
            "pre_chorus": "bass1" if "pre_chorus" in bass1_groups else "bass2",
            "chorus": "bass1" if "chorus" in bass1_groups else "bass2",
            "fill": "bass1" if "fill" in bass1_groups else "bass2",
            "outro": "bass1" if "outro" in bass1_groups else "bass2",
        },
        "layer2_verse_start_bar": layer2_verse_start,
        "layer3_build_start_bar": layer3_build_start,
        "fx_fill_gain_db": fx_fill_gain_db,
        "supporting_layer_foreground_margin_db": 4.0,
        "supporting_layer_trim_limit_db": -10.0,
    }

def section_gain_db(name, role, layer, section, policy, bar_start, bar_end, y_section, sr):
    n = (name or "").lower()
    r = (role or "").lower()
    l = (layer or "").lower()

    is_bass2 = "harmonic_bass" in n
    is_bass1 = (not is_bass2) and ("_bass.wav" in n or "usb01-02_bass" in n)
    section_group = section_owner_group(section)
    if is_bass1:
        base_gain = 0.0 if section_group in policy["bass1_groups"] else -90.0
        return base_gain, pink_decision(name, section, "bass_owner", base_gain, base_gain, y_section, sr, False)
    if is_bass2:
        base_gain = 0.0 if section_group in policy["bass2_groups"] else -90.0
        return base_gain, pink_decision(name, section, "bass_owner", base_gain, base_gain, y_section, sr, False)

    is_chorus_or_fill = "chorus" in section or "fill" in section
    is_verse = section.startswith("verse")
    is_build = "pre_chorus" in section
    is_outro = section == "outro"

    if "layer3" in n or l.endswith("layer3"):
        if is_chorus_or_fill:
            return pink_support_gain(name, section, "layer3", -1.5, y_section, sr, policy)
        if is_build and bar_end >= policy["layer3_build_start_bar"]:
            return pink_support_gain(name, section, "layer3", -3.0, y_section, sr, policy)
        base_gain = -90.0
        return base_gain, pink_decision(name, section, "layer3", base_gain, base_gain, y_section, sr, False)

    if "layer2" in n or l in {"melody_poly", "counter_bell", "chorus_brass", "pad_layer2"}:
        if is_verse:
            base_gain = -2.5 if bar_start >= policy["layer2_verse_start_bar"] else -90.0
            if base_gain > -80.0:
                return pink_support_gain(name, section, "layer2", base_gain, y_section, sr, policy)
            return base_gain, pink_decision(name, section, "layer2", base_gain, base_gain, y_section, sr, False)
        if is_chorus_or_fill or is_build:
            return pink_support_gain(name, section, "layer2", -1.0, y_section, sr, policy)
        if is_outro:
            return pink_support_gain(name, section, "layer2", -6.0, y_section, sr, policy)
        return pink_support_gain(name, section, "layer2", -4.0, y_section, sr, policy)

    if "melody_fx" in n or r == "fx":
        if "fill" in section:
            return pink_support_gain(name, section, "melody_fx", policy["fx_fill_gain_db"], y_section, sr, policy)
        if "chorus" in section:
            return pink_support_gain(name, section, "melody_fx", -5.0, y_section, sr, policy)
        if is_build:
            return pink_support_gain(name, section, "melody_fx", -4.0, y_section, sr, policy)
        return pink_support_gain(name, section, "melody_fx", -8.0, y_section, sr, policy)

    if "counter" in n:
        if is_verse:
            return pink_support_gain(name, section, "counter", -5.0 if bar_start < policy["layer2_verse_start_bar"] else -2.5, y_section, sr, policy)
        if is_chorus_or_fill:
            return pink_support_gain(name, section, "counter", -4.0, y_section, sr, policy)
        return pink_support_gain(name, section, "counter", -6.0, y_section, sr, policy)

    if "pad" in n or "chord" in n:
        if is_chorus_or_fill:
            return pink_support_gain(name, section, "pad", -3.0, y_section, sr, policy)
        if is_verse:
            return pink_support_gain(name, section, "pad", -5.0 if bar_start < policy["layer2_verse_start_bar"] else -3.5, y_section, sr, policy)
        if is_outro:
            return pink_support_gain(name, section, "pad", -6.0, y_section, sr, policy)
        return pink_support_gain(name, section, "pad", -4.5, y_section, sr, policy)

    return 0.0, pink_decision(name, section, "anchor_or_percussion", 0.0, 0.0, y_section, sr, False)

def pink_support_gain(name, section, layer_kind, base_gain, y_section, sr, policy):
    trim, before_score, after_score = pink_section_trim_db(
        y_section, sr, base_gain,
        foreground_margin_db=float(policy.get("supporting_layer_foreground_margin_db", 4.0)),
        trim_limit_db=float(policy.get("supporting_layer_trim_limit_db", -10.0)),
    )
    final_gain = float(base_gain + trim)
    return final_gain, pink_decision(name, section, layer_kind, base_gain, final_gain, y_section, sr, bool(trim < -0.01), before_score, after_score)

def pink_decision(name, section, layer_kind, base_gain, final_gain, y_section, sr, trimmed, before_score=None, after_score=None):
    if before_score is None:
        before_score = pink_weighted_section_score_db(y_section, sr, base_gain)
    if after_score is None:
        after_score = pink_weighted_section_score_db(y_section, sr, final_gain)
    return {
        "stem_name": name,
        "section": section,
        "layer_kind": layer_kind,
        "base_gain_db": float(base_gain),
        "final_gain_db": float(final_gain),
        "pink_score_before_db": float(before_score),
        "pink_score_after_db": float(after_score),
        "pink_trimmed": bool(trimmed),
    }

def pink_section_trim_db(y_section, sr, base_gain_db, foreground_margin_db=4.0, trim_limit_db=-10.0):
    before_score = pink_weighted_section_score_db(y_section, sr, base_gain_db)
    if not np.isfinite(before_score) or before_score <= foreground_margin_db:
        return 0.0, before_score, before_score
    trim = -min(float(before_score - foreground_margin_db), abs(float(trim_limit_db)))
    after_score = pink_weighted_section_score_db(y_section, sr, base_gain_db + trim)
    return float(trim), before_score, after_score

def pink_weighted_section_score_db(y_section, sr, gain_db):
    y_section = ensure_stereo(np.asarray(y_section, dtype=np.float32))
    if len(y_section) < 32:
        return -120.0
    y_adj = y_section * (10.0 ** (float(gain_db) / 20.0))
    mono = np.mean(y_adj, axis=1).astype(np.float64)
    mono = mono - float(np.mean(mono))
    rms = float(np.sqrt(np.mean(mono ** 2)) + 1e-12)
    if rms < 1e-8:
        return -120.0
    # Pink compensation: a source that is too foreground will show a high
    # compensated section score across its audible octave bands.
    win = np.hanning(len(mono))
    spec = np.abs(np.fft.rfft(mono * win)) ** 2
    freqs = np.fft.rfftfreq(len(mono), 1.0 / float(sr))
    bands = [(80, 160), (160, 320), (320, 640), (640, 1280), (1280, 2560), (2560, 5120), (5120, 10240)]
    vals = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < min(hi, sr * 0.45))
        if not np.any(mask):
            continue
        band_rms = float(np.sqrt(np.mean(spec[mask])) + 1e-12)
        band_db = 20.0 * np.log10(max(band_rms, 1e-12))
        center = np.sqrt(lo * hi)
        pink_comp = 3.0103 * np.log2(center / 1000.0)
        vals.append(band_db + pink_comp)
    if not vals:
        return 20.0 * np.log10(rms)
    # Normalize to the section RMS so this behaves as audibility balance, not a
    # raw level duplicate of ordinary RMS.
    return float(np.percentile(vals, 70) - (20.0 * np.log10(rms) + 62.0))

def section_owner_group(section):
    if section.startswith("verse") or "pre_chorus" in section:
        return "verse"
    if section.startswith("chorus"):
        return "chorus"
    if section.startswith("fill"):
        return "fill"
    if section == "intro":
        return "intro"
    if section == "outro":
        return "outro"
    return section

def beats_per_bar_from_name(name):
    m = re.search(r"_(\d+)-(\d+)_", name or "")
    if not m:
        return 4.0
    return float(m.group(1))

def section_metrics(y, sr, bpm, beats_per_bar):
    out = {}
    spb = max(1, int(sr * (60.0 / max(float(bpm), 1.0)) * beats_per_bar))
    mono = np.mean(np.abs(y), axis=1)
    for section, (bar_start, bar_end) in SECTION_BARS.items():
        start = min(len(mono), bar_start * spb)
        end = min(len(mono), bar_end * spb)
        if end <= start:
            continue
        seg = mono[start:end]
        rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)) + 1e-12)
        rms_db = float(20.0 * np.log10(max(rms, 1e-12)))
        active = bool(rms_db > -70.0)
        out[section] = {"rms_db": rms_db, "active": active}
    return out

def smooth_edges(curve, fade_len):
    if len(curve) < fade_len * 2:
        return curve
    out = curve.copy()
    changes = np.where(np.abs(np.diff(out)) > 1e-6)[0] + 1
    for idx in changes:
        lo = max(0, idx - fade_len)
        hi = min(len(out), idx + fade_len)
        if hi <= lo:
            continue
        start = out[lo]
        stop = out[hi - 1]
        out[lo:hi] = np.linspace(start, stop, hi - lo, dtype=np.float32)
    return out

def ensure_stereo(y):
    if y.ndim == 1:
        return np.column_stack([y, y])
    if y.shape[1] == 1:
        return np.repeat(y, 2, axis=1)
    return y[:, :2]
