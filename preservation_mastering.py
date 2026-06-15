"""
Preservation mastering chain for the portable 2026-05-10 full pipeline.

The goal is control, not tonal remaking: no reference EQ matching, no stereo
widening, and no limiter except final downward peak safety if LUFS normalization
would otherwise exceed the ceiling.
"""

import json
import os

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import butter, lfilter, sosfilt


def ensure_stereo(y):
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = np.column_stack([y, y])
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    return y[:, :2].astype(np.float32)


def db_to_amp(db):
    return float(10.0 ** (float(db) / 20.0))


def amp_to_db(x):
    return 20.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float64), 1e-12))


def biquad_peaking(sr, freq, q, gain_db):
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / sr
    alpha = np.sin(w0) / (2.0 * q)
    b0 = 1.0 + alpha * a
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / a
    return np.array([b0 / a0, b1 / a0, b2 / a0]), np.array([1.0, a1 / a0, a2 / a0])


def biquad_high_shelf(sr, freq, q, gain_db):
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / sr
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)
    sqrt_a = np.sqrt(a)
    b0 = a * ((a + 1) + (a - 1) * cos_w0 + 2 * sqrt_a * alpha)
    b1 = -2 * a * ((a - 1) + (a + 1) * cos_w0)
    b2 = a * ((a + 1) + (a - 1) * cos_w0 - 2 * sqrt_a * alpha)
    a0 = (a + 1) - (a - 1) * cos_w0 + 2 * sqrt_a * alpha
    a1 = 2 * ((a - 1) - (a + 1) * cos_w0)
    a2 = (a + 1) - (a - 1) * cos_w0 - 2 * sqrt_a * alpha
    return np.array([b0 / a0, b1 / a0, b2 / a0]), np.array([1.0, a1 / a0, a2 / a0])


def apply_biquad(y, b, a):
    return np.column_stack([lfilter(b, a, y[:, ch]) for ch in range(y.shape[1])]).astype(np.float32)


def apply_generic_mastering_eq(y, sr):
    y = ensure_stereo(y)
    nyq = sr * 0.5
    hp = butter(2, 20.0 / nyq, btype="highpass", output="sos")
    lp_freq = min(20000.0, nyq * 0.98)
    lp = butter(2, lp_freq / nyq, btype="lowpass", output="sos")
    y = np.column_stack([sosfilt(hp, y[:, ch]) for ch in range(2)]).astype(np.float32)
    y = np.column_stack([sosfilt(lp, y[:, ch]) for ch in range(2)]).astype(np.float32)

    for freq, q, gain in ((250.0, 0.8, -1.0), (3500.0, 2.5, -1.5)):
        b, a = biquad_peaking(sr, freq, q, gain)
        y = apply_biquad(y, b, a)
    b, a = biquad_high_shelf(sr, 12000.0, 0.6, 1.0)
    return apply_biquad(y, b, a)


def smooth_peak_envelope(x, sr, attack_ms, release_ms):
    mono = np.max(np.abs(x), axis=1)
    attack = np.exp(-1.0 / max(1.0, sr * attack_ms / 1000.0))
    release = np.exp(-1.0 / max(1.0, sr * release_ms / 1000.0))
    env = np.zeros_like(mono, dtype=np.float64)
    for i, value in enumerate(mono):
        coeff = attack if value > (env[i - 1] if i else 0.0) else release
        prev = env[i - 1] if i else value
        env[i] = coeff * prev + (1.0 - coeff) * value
    return env


def compressor_gain_db(env_db, threshold_db, ratio=2.0, knee_db=0.0):
    over = env_db - threshold_db
    if knee_db <= 0:
        compressed_over = np.maximum(over, 0.0) / ratio
        return np.minimum(0.0, compressed_over - np.maximum(over, 0.0))

    half = knee_db * 0.5
    gain = np.zeros_like(env_db)
    below = over <= -half
    above = over >= half
    middle = ~(below | above)
    gain[above] = (over[above] / ratio) - over[above]
    x = over[middle] + half
    gain[middle] = ((1.0 / ratio - 1.0) * x * x) / (2.0 * knee_db)
    return np.minimum(0.0, gain)


def compress_band(band, sr, bpm, attack_ms, release_note_beats, ratio, knee, target_gr_db=1.5):
    release_ms = (60.0 / max(float(bpm), 1.0)) * float(release_note_beats) * 1000.0
    env = smooth_peak_envelope(band, sr, attack_ms, release_ms)
    env_db = amp_to_db(env)
    active = env_db[np.isfinite(env_db) & (env_db > -90.0)]
    if active.size == 0:
        return band, {"threshold_db": -90.0, "avg_gr_db": 0.0, "max_gr_db": 0.0, "makeup_db": 0.0}

    threshold = float(np.percentile(active, 99.2) - target_gr_db)
    knee_db = 0.0 if knee == "hard" else 6.0
    gr_db = compressor_gain_db(env_db, threshold, ratio=ratio, knee_db=knee_db)
    gain = np.power(10.0, gr_db / 20.0).astype(np.float32)
    compressed = band * gain[:, None]

    reduction = -gr_db[gr_db < -0.01]
    avg_gr = float(np.mean(reduction)) if reduction.size else 0.0
    max_gr = float(np.max(reduction)) if reduction.size else 0.0
    makeup_db = min(avg_gr, 2.0)
    compressed *= db_to_amp(makeup_db)
    return compressed.astype(np.float32), {
        "threshold_db": threshold,
        "avg_gr_db": avg_gr,
        "max_gr_db": max_gr,
        "makeup_db": makeup_db,
        "attack_ms": float(attack_ms),
        "release_ms": float(release_ms),
        "ratio": float(ratio),
        "knee": knee,
    }


def objective_multiband_peak_control(y, sr, bpm):
    y = ensure_stereo(y)
    nyq = sr * 0.5
    low_sos = butter(4, 200.0 / nyq, btype="lowpass", output="sos")
    mid_sos = butter(4, [200.0 / nyq, 2000.0 / nyq], btype="bandpass", output="sos")
    high_sos = butter(4, 2000.0 / nyq, btype="highpass", output="sos")

    bands = {
        "low": np.column_stack([sosfilt(low_sos, y[:, ch]) for ch in range(2)]).astype(np.float32),
        "mid": np.column_stack([sosfilt(mid_sos, y[:, ch]) for ch in range(2)]).astype(np.float32),
        "high": np.column_stack([sosfilt(high_sos, y[:, ch]) for ch in range(2)]).astype(np.float32),
    }
    settings = {
        "low": {"attack_ms": 25.0, "release_note_beats": 2.0, "ratio": 2.0, "knee": "hard"},
        "mid": {"attack_ms": 20.0, "release_note_beats": 1.0, "ratio": 2.0, "knee": "soft"},
        "high": {"attack_ms": 10.0, "release_note_beats": 0.5, "ratio": 2.0, "knee": "soft"},
    }
    out = np.zeros_like(y, dtype=np.float32)
    report = {}
    for name, band in bands.items():
        processed, stats = compress_band(band, sr, bpm=bpm, **settings[name])
        out += processed
        report[name] = stats
    return out.astype(np.float32), report


def integrated_lufs(y, sr):
    meter = pyln.Meter(sr)
    try:
        return float(meter.integrated_loudness(ensure_stereo(y)))
    except Exception:
        return float("nan")


def peak_db(y):
    return float(amp_to_db(np.max(np.abs(y))))


def normalize_to_lufs_without_limiting(y, sr, target_lufs=-14.0, peak_ceiling_db=-1.0):
    y = ensure_stereo(y)
    in_lufs = integrated_lufs(y, sr)
    gain_db = 0.0 if not np.isfinite(in_lufs) else float(target_lufs - in_lufs)
    out = y * db_to_amp(gain_db)
    pk = peak_db(out)
    peak_safety_db = 0.0
    if pk > peak_ceiling_db:
        peak_safety_db = float(peak_ceiling_db - pk)
        out *= db_to_amp(peak_safety_db)
    return out.astype(np.float32), {
        "input_lufs": in_lufs,
        "target_lufs": float(target_lufs),
        "lufs_gain_db": gain_db,
        "peak_safety_db": peak_safety_db,
        "output_lufs": integrated_lufs(out, sr),
        "output_peak_db": peak_db(out),
    }


def render_preservation_master_file(
    source_path,
    master_path,
    bpm=90.0,
    streaming_path=None,
    premaster_path=None,
    report_path=None,
):
    y, sr = sf.read(source_path, always_2d=True)
    y = ensure_stereo(y)

    if premaster_path:
        sf.write(premaster_path, y, sr, subtype="PCM_24")

    eq = apply_generic_mastering_eq(y, sr)
    controlled, band_report = objective_multiband_peak_control(eq, sr, bpm)

    master = controlled
    master_peak = peak_db(master)
    peak_gain_db = 0.0
    if master_peak > -1.0:
        peak_gain_db = -1.0 - master_peak
        master = master * db_to_amp(peak_gain_db)
    if master_path:
        sf.write(master_path, master.astype(np.float32), sr, subtype="PCM_24")

    streaming_report = None
    if streaming_path:
        streaming, streaming_report = normalize_to_lufs_without_limiting(master, sr)
        sf.write(streaming_path, streaming, sr, subtype="PCM_24")

    report = {
        "source_path": source_path,
        "premaster_path": premaster_path,
        "master_path": master_path,
        "streaming_path": streaming_path,
        "bpm": float(bpm),
        "generic_eq": {
            "highpass_hz": 20.0,
            "lowpass_hz": 20000.0,
            "mud": {"freq_hz": 250.0, "q": 0.8, "gain_db": -1.0},
            "harshness": {"freq_hz": 3500.0, "q": 2.5, "gain_db": -1.5},
            "shimmer": {"freq_hz": 12000.0, "q": 0.6, "gain_db": 1.0, "type": "high_shelf"},
        },
        "multiband_peak_control": band_report,
        "master_lufs": integrated_lufs(master, sr),
        "master_peak_db": peak_db(master),
        "master_peak_safety_gain_db": peak_gain_db,
        "streaming": streaming_report,
    }
    if report_path:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
    return master_path or streaming_path, report
