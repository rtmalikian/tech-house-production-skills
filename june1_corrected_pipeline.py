#!/usr/bin/env python3
"""June 1 corrected mix/master renderer.

This script implements the fixed sequence requested in prompt_june1.md:
load the two explicit stem folders, classify stems by filename, apply
role-bounded randomized panning, pink-noise-informed gain staging, render a
headroom-safe stereo mix, and apply a simple streaming master chain.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pyloudnorm as pyln
import scipy.ndimage
import scipy.signal
import soundfile as sf


ROOT = Path("/Users/raphael/Coding/crate-dig/skillbased_pipeline")
TARGETS = (
    {
        "name": "05302026_100525_75bpm_G#_G#_Major_4-4_lofi-broken_beat",
        "bpm": 75.0,
        "recordings": ROOT
        / "output/05302026_100525_75bpm_G#_G#_Major_4-4_lofi-broken_beat/recordings",
    },
    {
        "name": "05272026_025553_88bpm_C_C_Major_4-4_chopped_break-lofi",
        "bpm": 88.0,
        "recordings": ROOT
        / "output/05272026_025553_88bpm_C_C_Major_4-4_chopped_break-lofi/recordings",
    },
)

TARGET_LUFS = -14.0
TRUE_PEAK_CEILING_DBTP = -1.0
PREMASTER_HEADROOM_DB = -6.0
SR_TARGET = 48_000
SATURATION_WET = 0.055
EPS = 1.0e-12
BEATS_PER_BAR = 4
TRANSITION_REVERSAL_PROBABILITY = 0.10
FX_CROSSFADE_MS = 15.0
SIDECHAIN_TRIGGER_THRESHOLD = 0.15    # Only trigger on strong kick transients (-16 dBFS)
SIDECHAIN_DUCK_DB = -3.0             # 3 dB ducking — audible pump without crushing
SIDECHAIN_ATTACK_MS = 5.0            # Fast grab on kick transient
SIDECHAIN_RELEASE_MS = 150.0         # Slow release — stay ducked through kick body
REVERB_RETURN_HIGHPASS_HZ = 200.0

BUS_NAMES = (
    "bass_bus",
    "melody_bus",
    "pad_bus",
    "kick_bus",
    "snare_bus",
    "perc_bus",
    "fx_bus",
)

ARRANGEMENT = (
    ("intro", 8),
    ("verse1", 16),
    ("chorus1", 8),
    ("fill1", 4),
    ("verse2", 16),
    ("chorus2", 8),
    ("fill2", 4),
    ("outro", 8),
)


@dataclass(frozen=True)
class Stem:
    path: Path
    category: str
    pan: float
    source_sr: int
    frames: int


CATEGORY_TARGET_OFFSETS_DB = {
    "kick": 1.0,
    "snare": -1.0,
    "hats": -9.0,
    "bass": -2.5,
    "melody": -6.5,
    "pads": -8.0,
    "texture": -12.0,
    "FX": -12.5,
}

BAND_LIMITS = {
    "kick": (40.0, 5_000.0),
    "snare": (180.0, 5_000.0),
    "hats": (4_000.0, 14_000.0),
    "bass": (35.0, 220.0),
    "melody": (250.0, 6_000.0),
    "pads": (180.0, 5_000.0),
    "texture": (500.0, 12_000.0),
    "FX": (500.0, 12_000.0),
}


def db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def amp_to_db(value: float) -> float:
    return float(20.0 * math.log10(max(float(value), EPS)))


def ensure_stereo(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return np.column_stack([y, y]).astype(np.float32, copy=False)
    if y.shape[1] == 1:
        return np.repeat(y, 2, axis=1).astype(np.float32, copy=False)
    return y[:, :2].astype(np.float32, copy=False)


def read_audio(path: Path, sr: int) -> np.ndarray:
    y, file_sr = sf.read(path, dtype="float32", always_2d=True)
    y = ensure_stereo(y)
    if file_sr == sr:
        return y
    gcd = math.gcd(int(file_sr), int(sr))
    return scipy.signal.resample_poly(y, sr // gcd, file_sr // gcd, axis=0).astype(np.float32)


def write_audio(path: Path, y: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(y, dtype=np.float32), sr, subtype="FLOAT")


def classify_stem(path: Path) -> str:
    name = path.name.lower()
    if "kick" in name:
        return "kick"
    if any(token in name for token in ("snare", "clap", "sidestick", "side_stick")):
        return "snare"
    if any(token in name for token in ("hat", "ride", "crash")):
        return "hats"
    if "bass" in name:
        return "bass"
    if "pad" in name or "chord" in name:
        return "pads"
    if "fx" in name:
        return "FX"
    if "melody" in name:
        return "melody"
    return "texture"


def bus_for_category(category: str) -> str:
    return {
        "bass": "bass_bus",
        "melody": "melody_bus",
        "pads": "pad_bus",
        "kick": "kick_bus",
        "snare": "snare_bus",
        "hats": "perc_bus",
        "texture": "fx_bus",
        "FX": "fx_bus",
    }[category]


def unique_pan(rng: random.Random, low: float, high: float, used: set[float]) -> float:
    for _ in range(128):
        pan = round(rng.uniform(low, high), 4)
        if pan not in used:
            used.add(pan)
            return pan
    step = 0.0001 if high >= low else -0.0001
    pan = low
    while round(pan, 4) in used and ((step > 0 and pan <= high) or (step < 0 and pan >= high)):
        pan += step
    pan = round(max(min(pan, high), low), 4)
    used.add(pan)
    return pan


def assign_pans(paths: Sequence[Path], seed: int) -> List[Stem]:
    rng = random.Random(seed)
    used: set[float] = set()
    wide_side = -1
    stems: List[Stem] = []
    for path in sorted(paths):
        info = sf.info(path)
        category = classify_stem(path)
        if category in {"kick", "bass"}:
            pan = unique_pan(rng, -0.05, 0.0, used)
        elif category == "snare":
            pan = unique_pan(rng, -0.6, -0.3, used)
        elif category == "hats":
            pan = unique_pan(rng, 0.3, 0.6, used)
        else:
            if wide_side < 0:
                pan = unique_pan(rng, -0.95, -0.4, used)
            else:
                pan = unique_pan(rng, 0.4, 0.95, used)
            wide_side *= -1
        stems.append(Stem(path=path, category=category, pan=pan, source_sr=info.samplerate, frames=info.frames))
    return stems


def mono_projection(y: np.ndarray) -> np.ndarray:
    y = ensure_stereo(y)
    return np.mean(y, axis=1, dtype=np.float32)


def equal_power_pan(y: np.ndarray, pan: float) -> np.ndarray:
    mono = mono_projection(y)
    angle = (float(pan) + 1.0) * math.pi * 0.25
    left = math.cos(angle)
    right = math.sin(angle)
    return np.column_stack([mono * left, mono * right]).astype(np.float32)


def butter_bandpass(y: np.ndarray, sr: int, low: float, high: float, order: int = 4) -> np.ndarray:
    nyq = sr * 0.5
    low = max(low, 20.0) / nyq
    high = min(high, nyq * 0.95) / nyq
    if high <= low:
        return y
    sos = scipy.signal.butter(order, [low, high], btype="bandpass", output="sos")
    return scipy.signal.sosfilt(sos, y, axis=0).astype(np.float32)


def frame_rms_envelope(y: np.ndarray, frame: int = 2048, hop: int = 512) -> Tuple[np.ndarray, int, int]:
    mono = np.abs(mono_projection(y))
    if mono.size < frame:
        return np.array([float(np.sqrt(np.mean(mono * mono) + EPS))], dtype=np.float32), frame, hop
    frame_count = 1 + (mono.size - frame) // hop
    envelope = np.empty(frame_count, dtype=np.float32)
    for idx in range(frame_count):
        chunk = mono[idx * hop : idx * hop + frame]
        envelope[idx] = float(np.sqrt(np.mean(chunk * chunk) + EPS))
    return envelope, frame, hop


def frames_to_sample_mask(active_frames: np.ndarray, length: int, frame: int, hop: int) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for idx, active in enumerate(active_frames):
        if active:
            mask[idx * hop : idx * hop + frame] = True
    return mask


def active_region_mask(y: np.ndarray, frame: int = 2048, hop: int = 512) -> Tuple[np.ndarray, Dict]:
    envelope, frame, hop = frame_rms_envelope(y, frame=frame, hop=hop)
    length = y.shape[0]
    floor = float(np.percentile(envelope, 20))
    body = float(np.percentile(envelope, 75))
    threshold = max(floor + 0.30 * max(body - floor, 0.0), body * 0.18, db_to_amp(-72.0))
    active_frames = envelope > threshold
    if active_frames.size > 3:
        active_frames = scipy.ndimage.binary_closing(active_frames, structure=np.ones(3, dtype=bool))
    mask = frames_to_sample_mask(active_frames, length, frame, hop)
    if not np.any(mask):
        mask[:] = True
    return mask, {
        "mask_mode": "active_note_or_sustain_regions",
        "metered_sample_fraction": float(np.mean(mask)),
        "threshold_dbfs": amp_to_db(threshold),
        "noise_floor_dbfs": amp_to_db(floor),
        "body_reference_dbfs": amp_to_db(body),
    }


def hit_window_mask(y: np.ndarray, sr: int, category: str) -> Tuple[np.ndarray, Dict]:
    mono = np.abs(mono_projection(y))
    if mono.size == 0:
        return np.ones(0, dtype=bool), {"mask_mode": "hit_windows", "hit_window_count": 0}
    envelope = mono
    smooth_samples = max(1, int(round(0.004 * sr)))
    envelope = scipy.ndimage.maximum_filter1d(envelope.astype(np.float32), size=smooth_samples, mode="nearest")
    floor = float(np.percentile(envelope, 35))
    attack_ref = float(np.percentile(envelope, 96))
    threshold = max(floor + 0.45 * max(attack_ref - floor, 0.0), attack_ref * 0.22, db_to_amp(-70.0))
    distance_seconds = {"kick": 0.18, "snare": 0.16, "hats": 0.045, "texture": 0.06}.get(category, 0.10)
    peaks, _ = scipy.signal.find_peaks(envelope, height=threshold, distance=max(1, int(distance_seconds * sr)))
    if peaks.size == 0:
        return active_region_mask(y)
    pre_seconds = 0.0
    post_seconds = {"kick": 0.150, "snare": 0.150}.get(category, 0.100)
    post = int(round(post_seconds * sr))
    loudest_peak = int(peaks[np.argmax(envelope[peaks])])
    mask = np.zeros(mono.shape[0], dtype=bool)
    start = max(0, loudest_peak)
    end = min(mask.size, loudest_peak + post)
    mask[start:end] = True
    return mask, {
        "mask_mode": "loudest_drum_hit_window",
        "hit_window_count": int(peaks.size),
        "metered_sample_fraction": float(np.mean(mask)),
        "threshold_dbfs": amp_to_db(threshold),
        "noise_floor_dbfs": amp_to_db(floor),
        "attack_reference_dbfs": amp_to_db(attack_ref),
        "pre_window_ms": 0.0,
        "post_window_ms": post_seconds * 1000.0,
    }


def loudest_bar_mask(y: np.ndarray, sr: int, bpm: float, beats_per_bar: int = 4) -> Tuple[np.ndarray, Dict]:
    mono = np.abs(mono_projection(y))
    if mono.size == 0:
        return np.ones(0, dtype=bool), {"mask_mode": "loudest_bar", "selected_bar_samples": 0}
    bar_samples = int(round(sr * (60.0 / bpm) * beats_per_bar))
    bar_samples = max(1, min(bar_samples, mono.size))
    power = mono.astype(np.float64) ** 2
    csum = np.concatenate(([0.0], np.cumsum(power)))
    max_start = mono.size - bar_samples
    hop = max(1, bar_samples // 16)
    starts = np.arange(0, max(1, max_start + 1), hop, dtype=np.int64)
    energies = (csum[starts + bar_samples] - csum[starts]) / float(bar_samples)
    best_idx = int(np.argmax(energies))
    best_start = int(starts[best_idx])
    best_end = min(mono.size, best_start + bar_samples)
    mask = np.zeros(mono.size, dtype=bool)
    mask[best_start:best_end] = True
    return mask, {
        "mask_mode": "loudest_bar",
        "selected_bar_samples": best_end - best_start,
        "selected_bar_start_sec": best_start / float(sr),
        "selected_bar_end_sec": best_end / float(sr),
        "candidate_bar_count": int(len(starts)),
        "metered_sample_fraction": float(np.mean(mask)),
    }


def metering_mask(y: np.ndarray, sr: int, category: str, bpm: float = 120.0) -> Tuple[np.ndarray, Dict]:
    if category in {"kick", "snare", "hats", "texture"}:
        return hit_window_mask(y, sr, category)
    return loudest_bar_mask(y, sr, bpm)


def rms(y: np.ndarray, mask: np.ndarray | None = None) -> float:
    values = ensure_stereo(y)
    if mask is not None:
        values = values[mask]
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64)) + EPS))


def make_pink_noise(length: int, sr: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(length)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(length, 1.0 / sr)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    pink = np.fft.irfft(spectrum * scale, n=length)
    pink = pink / max(float(np.sqrt(np.mean(pink * pink))), EPS)
    pink = pink * db_to_amp(-20.0)
    return np.column_stack([pink, pink]).astype(np.float32)


def gain_stage_stem(y: np.ndarray, category: str, pink_signal: np.ndarray, pink_rms_full: float, sr: int, bpm: float = 120.0) -> Tuple[np.ndarray, Dict]:
    low, high = BAND_LIMITS[category]
    band_y = butter_bandpass(y, sr, low, high)
    mask, mask_report = metering_mask(band_y, sr, category, bpm=bpm)
    stem_rms = rms(band_y, mask)
    band_pink = butter_bandpass(pink_signal, sr, low, high)
    pink_rms = rms(band_pink, mask)
    if pink_rms < EPS:
        pink_rms = pink_rms_full
    target_rms = pink_rms * db_to_amp(CATEGORY_TARGET_OFFSETS_DB[category])
    gain = target_rms / max(stem_rms, EPS)
    staged = y * np.float32(gain)
    return staged.astype(np.float32), {
        "category": category,
        "band_hz": [low, high],
        **mask_report,
        "active_band_rms_dbfs": amp_to_db(stem_rms),
        "pink_band_rms_dbfs": amp_to_db(pink_rms),
        "target_offset_db": CATEGORY_TARGET_OFFSETS_DB[category],
        "applied_gain_db": amp_to_db(gain),
        "post_gain_fullband_rms_dbfs": amp_to_db(rms(staged)),
    }


def highpass(y: np.ndarray, sr: int, cutoff: float = 20.0) -> np.ndarray:
    sos = scipy.signal.butter(2, cutoff / (sr * 0.5), btype="highpass", output="sos")
    return scipy.signal.sosfilt(sos, y, axis=0).astype(np.float32)


def lowpass(y: np.ndarray, sr: int, cutoff: float = 14_000.0) -> np.ndarray:
    sos = scipy.signal.butter(2, cutoff / (sr * 0.5), btype="lowpass", output="sos")
    return scipy.signal.sosfilt(sos, y, axis=0).astype(np.float32)


def peaking_eq(y: np.ndarray, sr: int, center_hz: float, gain_db: float, q: float) -> np.ndarray:
    a = db_to_amp(gain_db)
    omega = 2.0 * math.pi * center_hz / sr
    alpha = math.sin(omega) / (2.0 * q)
    cosw = math.cos(omega)
    b0 = 1.0 + alpha * a
    b1 = -2.0 * cosw
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * cosw
    a2 = 1.0 - alpha / a
    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    a_coeff = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return scipy.signal.lfilter(b, a_coeff, y, axis=0).astype(np.float32)


def bar_to_sample(bar_number: float, bpm: float, sr: int) -> int:
    seconds = ((float(bar_number) - 1.0) * BEATS_PER_BAR * 60.0) / float(bpm)
    return int(round(seconds * sr))


def arrangement_grid(bpm: float, sr: int, length: int) -> Dict:
    sections = []
    start_bar = 1
    for name, bars in ARRANGEMENT:
        end_bar = start_bar + bars - 1
        start = min(length, bar_to_sample(start_bar, bpm, sr))
        end = min(length, bar_to_sample(end_bar + 1, bpm, sr))
        sections.append(
            {
                "name": name,
                "start_bar": start_bar,
                "end_bar": end_bar,
                "bars": bars,
                "start_sample": start,
                "end_sample": end,
            }
        )
        start_bar = end_bar + 1
    return {
        "bpm": bpm,
        "beats_per_bar": BEATS_PER_BAR,
        "samples_per_bar": float(sr * 60.0 * BEATS_PER_BAR / float(bpm)),
        "total_bars": sum(bars for _, bars in ARRANGEMENT),
        "sections": sections,
        "explicit_transition_fill_bars": [[33, 36], [65, 68]],
    }


def sample_range_for_bars(start_bar: int, end_bar: int, bpm: float, sr: int, length: int) -> Tuple[int, int]:
    start = min(length, bar_to_sample(start_bar, bpm, sr))
    end = min(length, bar_to_sample(end_bar + 1, bpm, sr))
    return start, max(start, end)


def fade_curve(length: int, start_amp: float, end_amp: float) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.linspace(float(start_amp), float(end_amp), length, dtype=np.float32)


def apply_linear_gain(y: np.ndarray, start: int, end: int, start_db: float, end_db: float) -> None:
    if end <= start:
        return
    curve = fade_curve(end - start, db_to_amp(start_db), db_to_amp(end_db))
    y[start:end] *= curve[:, None]


def splice_processed_segment(y: np.ndarray, start: int, end: int, processed: np.ndarray, sr: int, crossfade_ms: float = FX_CROSSFADE_MS) -> None:
    if end <= start:
        return
    processed = ensure_stereo(processed).astype(np.float32)
    span = end - start
    if processed.shape[0] < span:
        processed = np.pad(processed, ((0, span - processed.shape[0]), (0, 0)))
    elif processed.shape[0] > span:
        processed = processed[:span]
    original = y[start:end].copy()
    fade_len = min(int(round((crossfade_ms / 1000.0) * sr)), max(1, span // 2))
    if fade_len > 1:
        fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)[:, None]
        fade_out = 1.0 - fade_in
        processed[:fade_len] = original[:fade_len] * fade_out + processed[:fade_len] * fade_in
        processed[-fade_len:] = processed[-fade_len:] * fade_out + original[-fade_len:] * fade_in
    y[start:end] = processed


def replace_segment_gain(y: np.ndarray, start: int, end: int, gain_db: float, sr: int) -> None:
    if end <= start:
        return
    processed = y[start:end] * np.float32(db_to_amp(gain_db))
    splice_processed_segment(y, start, end, processed, sr)


def mute_segment(y: np.ndarray, start: int, end: int, sr: int) -> None:
    if end <= start:
        return
    splice_processed_segment(y, start, end, np.zeros((end - start, 2), dtype=np.float32), sr)


def apply_smooth_mask(y: np.ndarray, mask: np.ndarray, sr: int, crossfade_ms: float = FX_CROSSFADE_MS) -> np.ndarray:
    mask = mask.astype(np.float32)
    fade_len = max(1, int(round((crossfade_ms / 1000.0) * sr)))
    kernel = np.ones(fade_len, dtype=np.float32) / float(fade_len)
    smooth = np.convolve(mask, kernel, mode="same")
    smooth = np.clip(smooth, 0.0, 1.0)
    return (y * smooth[:, None]).astype(np.float32)


def sidechain_ducking_curve(
    trigger_bus: np.ndarray,
    sr: int,
    threshold: float = SIDECHAIN_TRIGGER_THRESHOLD,
    duck_db: float = SIDECHAIN_DUCK_DB,
    attack_ms: float = SIDECHAIN_ATTACK_MS,
    release_ms: float = SIDECHAIN_RELEASE_MS,
) -> Tuple[np.ndarray, Dict]:
    trigger = np.abs(mono_projection(trigger_bus))
    active = (trigger >= float(threshold)).astype(np.float32)
    attack_coeff = 1.0 - math.exp(-1.0 / max(1.0, sr * attack_ms / 1000.0))
    release_coeff = 1.0 - math.exp(-1.0 / max(1.0, sr * release_ms / 1000.0))
    env = np.zeros_like(trigger, dtype=np.float32)
    prev = 0.0
    for i in range(len(active)):
        target = float(active[i])
        coeff = attack_coeff if target > prev else release_coeff
        prev = coeff * target + (1.0 - coeff) * prev
        env[i] = prev
    duck_gain = db_to_amp(duck_db)
    curve = (1.0 + (duck_gain - 1.0) * env).astype(np.float32)
    return curve, {
        "trigger_threshold": threshold,
        "duck_db": duck_db,
        "duck_gain": duck_gain,
        "attack_ms": attack_ms,
        "release_ms": release_ms,
        "trigger_sample_count": int(np.count_nonzero(active)),
        "ducked_sample_count": int(np.count_nonzero(env > 0.01)),
        "ducked_sample_fraction": float(np.mean(env > 0.01)) if env.size else 0.0,
        "max_trigger_envelope": float(np.max(trigger)) if trigger.size else 0.0,
    }


def apply_dual_bus_sidechain(automated: Dict[str, np.ndarray], sr: int) -> Tuple[Dict[str, np.ndarray], Dict]:
    processed = {name: automated[name].copy() for name in BUS_NAMES}
    kick_curve, kick_report = sidechain_ducking_curve(processed["kick_bus"], sr)
    snare_curve, snare_report = sidechain_ducking_curve(processed["snare_bus"], sr)
    processed["bass_bus"] *= kick_curve[:, None]
    for bus_name in ("melody_bus", "pad_bus"):
        processed[bus_name] *= snare_curve[:, None]
    report = {
        "operation": "dual_bus_sidechain_separation_transient_safety_matrix",
        "processing_stage": "pre_master_bus_summation",
        "low_end_separation": {
            **kick_report,
            "trigger_bus": "kick_bus",
            "destination_buses": ["bass_bus"],
            "destination_channels": "both",
            "policy": "Kick envelope ducks bass only for low-end definition.",
        },
        "mid_range_pocket": {
            **snare_report,
            "trigger_bus": "snare_bus",
            "destination_buses": ["melody_bus", "pad_bus"],
            "bypassed_buses": ["fx_bus"],
            "destination_channels": "both",
            "policy": "Snare envelope ducks melody and pad buses while fx_bus remains a continuous background anchor.",
        },
        "parameters": {
            "trigger_threshold": SIDECHAIN_TRIGGER_THRESHOLD,
            "duck_db": SIDECHAIN_DUCK_DB,
            "attack_ms": SIDECHAIN_ATTACK_MS,
            "release_ms": SIDECHAIN_RELEASE_MS,
        },
    }
    return processed, report


def multitap_reverb_return(
    source: np.ndarray,
    sr: int,
    decay_seconds: float,
    wet_gain: float,
    predelay_ms: float,
    rng: np.random.Generator,
    tap_spacing_ms: float,
    width: float,
) -> np.ndarray:
    y = ensure_stereo(source).astype(np.float32)
    out = np.zeros_like(y)
    predelay = int(round((predelay_ms / 1000.0) * sr))
    spacing = max(1, int(round((tap_spacing_ms / 1000.0) * sr)))
    max_delay = min(y.shape[0] - 1, predelay + int(round(decay_seconds * sr)))
    if max_delay <= predelay:
        return out
    tap_count = max(4, min(48, int((max_delay - predelay) // spacing)))
    for idx in range(tap_count):
        delay = predelay + (idx + 1) * spacing
        if delay >= y.shape[0]:
            break
        t = delay / float(sr)
        decay = math.exp(-3.0 * max(0.0, t - predelay / float(sr)) / max(decay_seconds, 1.0e-3))
        jitter = float(rng.uniform(0.72, 1.08))
        gain = wet_gain * decay * jitter / math.sqrt(idx + 1.0)
        if gain <= 0.0:
            continue
        src = y[:-delay]
        if idx % 2 == 0:
            out[delay:, 0] += src[:, 1] * gain * width
            out[delay:, 1] += src[:, 0] * gain
        else:
            out[delay:, 0] += src[:, 0] * gain
            out[delay:, 1] += src[:, 1] * gain * width
    return highpass(out, sr, REVERB_RETURN_HIGHPASS_HZ)


def apply_spatial_reverb_matrix(automated: Dict[str, np.ndarray], sr: int, rng: np.random.Generator) -> Tuple[Dict[str, np.ndarray], Dict]:
    specs = {
        "melody_reverb_return": {
            "target_bus": "melody_bus",
            "algorithm": "small_room_short_plate",
            "decay_seconds": float(rng.uniform(0.6, 1.0)),
            "wet_gain": float(rng.uniform(0.05, 0.10)),
            "wet_gain_bounds": [0.05, 0.10],
            "predelay_ms": 0.0,
            "tap_spacing_ms": 19.0,
            "width": 1.18,
            "purpose": "Subtle early reflections widen primary melodic chops without pushing them backward.",
        },
        "pad_reverb_return": {
            "target_bus": "pad_bus",
            "algorithm": "large_hall_vintage_plate",
            "decay_seconds": float(rng.uniform(1.8, 2.5)),
            "wet_gain": float(rng.uniform(0.15, 0.25)),
            "wet_gain_bounds": [0.15, 0.25],
            "predelay_ms": 0.0,
            "tap_spacing_ms": 37.0,
            "width": 1.45,
            "purpose": "Low-level wide atmospheric wash for sustained chords.",
        },
        "snare_reverb_return": {
            "target_bus": "snare_bus",
            "algorithm": "vintage_plate_discrete_room",
            "decay_seconds": float(rng.uniform(0.5, 0.8)),
            "wet_gain": float(rng.uniform(0.06, 0.12)),
            "wet_gain_bounds": [0.06, 0.12],
            "predelay_ms": float(rng.uniform(15.0, 25.0)),
            "tap_spacing_ms": 17.0,
            "width": 1.22,
            "purpose": "Quiet brief cloud around snare while predelay preserves transient crack.",
        },
    }
    returns: Dict[str, np.ndarray] = {}
    return_reports: Dict[str, Dict] = {}
    for return_name, spec in specs.items():
        target_bus = spec["target_bus"]
        wet = multitap_reverb_return(
            automated[target_bus],
            sr,
            decay_seconds=spec["decay_seconds"],
            wet_gain=spec["wet_gain"],
            predelay_ms=spec["predelay_ms"],
            rng=rng,
            tap_spacing_ms=spec["tap_spacing_ms"],
            width=spec["width"],
        ).astype(np.float32)
        returns[return_name] = wet
        return_reports[return_name] = {
            **{k: v for k, v in spec.items() if k not in {"tap_spacing_ms", "width"}},
            "return_highpass_hz": REVERB_RETURN_HIGHPASS_HZ,
            "peak_dbfs": amp_to_db(float(np.max(np.abs(wet)))) if wet.size else -240.0,
            "rms_dbfs": amp_to_db(rms(wet)) if wet.size else -240.0,
            "tap_spacing_ms": spec["tap_spacing_ms"],
            "stereo_width_multiplier": spec["width"],
        }
    report = {
        "operation": "low_gain_localized_spatial_reverb_matrix",
        "processing_stage": "pre_master_bus_summation",
        "return_highpass_hz": REVERB_RETURN_HIGHPASS_HZ,
        "returns": return_reports,
        "dry_core_bypass": {
            "bypassed_buses": ["kick_bus", "perc_bus", "bass_bus"],
            "wet_percent": 0.0,
            "policy": "Kick, percussion, and bass remain dry to anchor groove and low end.",
        },
        "transparent_wet_ceiling_policy": "All return wet gains are randomized inside strict low-gain ceilings and high-passed at 200 Hz.",
    }
    return returns, report


def apply_automated_lowpass_segment(y: np.ndarray, sr: int, start: int, end: int, start_hz: float, end_hz: float) -> None:
    if end <= start:
        return
    block = 4096
    original = y[start:end].copy()
    total = original.shape[0]
    out = np.zeros_like(original)
    for offset in range(0, total, block):
        chunk_end = min(total, offset + block)
        position = (offset + chunk_end) * 0.5 / max(total, 1)
        cutoff = start_hz + (end_hz - start_hz) * position
        out[offset:chunk_end] = lowpass(original[offset:chunk_end], sr, cutoff=float(cutoff))
    y[start:end] = out


def apply_automated_highpass_segment(y: np.ndarray, sr: int, start: int, end: int, start_hz: float, end_hz: float) -> None:
    if end <= start:
        return
    block = 4096
    original = y[start:end].copy()
    total = original.shape[0]
    out = np.zeros_like(original)
    for offset in range(0, total, block):
        chunk_end = min(total, offset + block)
        position = (offset + chunk_end) * 0.5 / max(total, 1)
        cutoff = start_hz + (end_hz - start_hz) * position
        out[offset:chunk_end] = highpass(original[offset:chunk_end], sr, cutoff=float(cutoff))
    y[start:end] = out


def delay_reverb_wet(segment: np.ndarray, sr: int, bpm: float) -> np.ndarray:
    delay_samples = max(1, int(round((60.0 / float(bpm)) * 0.75 * sr)))
    out = np.zeros_like(segment)
    if delay_samples < segment.shape[0]:
        out[delay_samples:] += segment[:-delay_samples] * 0.62
    short_delay = max(1, int(round(0.087 * sr)))
    if short_delay < segment.shape[0]:
        out[short_delay:] += segment[:-short_delay] * 0.22
    tail_delay = max(1, int(round(0.143 * sr)))
    if tail_delay < segment.shape[0]:
        out[tail_delay:] += segment[:-tail_delay] * 0.14
    return lowpass(out, sr, 8500.0)


def widen_stereo(y: np.ndarray, amount: float) -> np.ndarray:
    y = ensure_stereo(y).astype(np.float32)
    mid = np.mean(y, axis=1, keepdims=True)
    side = (y[:, :1] - y[:, 1:2]) * 0.5
    left = mid + side * float(amount)
    right = mid - side * float(amount)
    return np.concatenate([left, right], axis=1).astype(np.float32)


def stereo_delay_mix(segment: np.ndarray, sr: int, bpm: float, wet: float, feedback: float = 0.45) -> np.ndarray:
    dry = ensure_stereo(segment)
    delay = max(1, int(round((60.0 / float(bpm)) * 0.50 * sr)))
    wet_sig = np.zeros_like(dry)
    if delay < dry.shape[0]:
        wet_sig[delay:, 0] += dry[:-delay, 1] * feedback
        wet_sig[delay:, 1] += dry[:-delay, 0] * feedback
    return (dry * (1.0 - wet) + wet_sig * wet).astype(np.float32)


def bitcrush(segment: np.ndarray, bits: int, target_sr: float, sr: int) -> np.ndarray:
    y = ensure_stereo(segment).astype(np.float32)
    decim = max(1, int(round(sr / max(float(target_sr), 1.0))))
    held = y[::decim]
    if held.shape[0] == 0:
        return y
    crushed = np.repeat(held, decim, axis=0)[: y.shape[0]]
    levels = max(2, int(2 ** int(bits)))
    crushed = np.round(np.clip(crushed, -1.0, 1.0) * (levels / 2.0)) / (levels / 2.0)
    return crushed.astype(np.float32)


def ring_modulate(segment: np.ndarray, sr: int, carrier_hz: float) -> np.ndarray:
    y = ensure_stereo(segment)
    t = np.arange(y.shape[0], dtype=np.float32) / float(sr)
    carrier = np.sin(2.0 * np.pi * float(carrier_hz) * t).astype(np.float32)
    return (y * carrier[:, None]).astype(np.float32)


def auto_pan(segment: np.ndarray, sr: int, rate_hz: float) -> np.ndarray:
    y = mono_projection(segment)
    t = np.arange(y.shape[0], dtype=np.float32) / float(sr)
    pan = np.sin(2.0 * np.pi * float(rate_hz) * t)
    left = y * np.sqrt((1.0 - pan) * 0.5)
    right = y * np.sqrt((1.0 + pan) * 0.5)
    return np.column_stack([left, right]).astype(np.float32)


def phaser_sweep(segment: np.ndarray, sr: int, rate_hz: float) -> np.ndarray:
    y = ensure_stereo(segment).astype(np.float32)
    out = y.copy()
    block = 2048
    for offset in range(0, y.shape[0], block):
        end = min(y.shape[0], offset + block)
        pos = (offset + end) * 0.5 / float(sr)
        center = 450.0 + 1050.0 * (0.5 + 0.5 * math.sin(2.0 * math.pi * float(rate_hz) * pos))
        out[offset:end] = peaking_eq(y[offset:end], sr, center_hz=center, gain_db=-5.0, q=2.5)
    return out


def tape_brake(segment: np.ndarray) -> np.ndarray:
    y = ensure_stereo(segment)
    curve = np.exp(-np.linspace(0.0, 5.5, y.shape[0], dtype=np.float32))
    return (y * curve[:, None]).astype(np.float32)


def pitch_grime_shift(segment: np.ndarray, semitones: float) -> np.ndarray:
    y = ensure_stereo(segment)
    factor = 2.0 ** (float(semitones) / 12.0)
    short = scipy.signal.resample_poly(y, 1, max(1, int(round(factor * 8))), axis=0)
    restored = scipy.signal.resample(short, y.shape[0], axis=0)
    return restored.astype(np.float32)


def linear_wet_delay(segment: np.ndarray, sr: int, bpm: float, wet_start: float, wet_end: float, feedback: float) -> np.ndarray:
    dry = ensure_stereo(segment)
    wet_sig = stereo_delay_mix(dry, sr, bpm, wet=1.0, feedback=feedback)
    wet_curve = fade_curve(dry.shape[0], wet_start, wet_end)
    return (dry * (1.0 - wet_curve[:, None]) + wet_sig * wet_curve[:, None]).astype(np.float32)


def reverse_with_crossfade(y: np.ndarray, start: int, end: int, sr: int, crossfade_ms: float = 10.0) -> None:
    if end <= start:
        return
    original = y[start:end].copy()
    reversed_seg = original[::-1].copy()
    fade_len = min(int(round((crossfade_ms / 1000.0) * sr)), max(1, (end - start) // 2))
    if fade_len > 1:
        fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)[:, None]
        fade_out = 1.0 - fade_in
        reversed_seg[:fade_len] = original[:fade_len] * fade_out + reversed_seg[:fade_len] * fade_in
        reversed_seg[-fade_len:] = reversed_seg[-fade_len:] * fade_out + original[-fade_len:] * fade_in
    y[start:end] = reversed_seg


def transition_windows(grid: Dict, bpm: float, sr: int, length: int) -> List[Dict]:
    sections = grid.get("sections", [])
    windows: List[Dict] = []
    for idx, section in enumerate(sections[:-1]):
        if not (str(section["name"]).startswith("verse") or str(section["name"]).startswith("chorus")):
            continue
        end_bar = int(section["end_bar"])
        start_bar = end_bar
        start, end = sample_range_for_bars(start_bar, end_bar, bpm, sr, length)
        if end > start:
            windows.append(
                {
                    "section": section["name"],
                    "next_section": sections[idx + 1]["name"],
                    "start_bar": start_bar,
                    "end_bar": end_bar,
                    "start_sample": start,
                    "end_sample": end,
                }
            )
    return windows


def apply_transition_fx(automated: Dict[str, np.ndarray], grid: Dict, bpm: float, sr: int, rng: np.random.Generator) -> List[Dict]:
    length = max(bus.shape[0] for bus in automated.values())
    option_names = (
        "reversal_swell",
        "isolator_bandpass_break",
        "chorus_washout",
        "phaser_sweep",
        "phase_inverted_widener",
        "bit_crushed_downscale",
    )
    ops: List[Dict] = []
    for window in transition_windows(grid, bpm, sr, length):
        start = int(window["start_sample"])
        end = int(window["end_sample"])
        option = option_names[int(rng.integers(0, len(option_names)))]
        params: Dict = {}
        segment = automated["melody_bus"][start:end].copy()
        if option == "reversal_swell":
            processed = segment[::-1].copy()
            params["array_reversal"] = "[::-1]"
        elif option == "isolator_bandpass_break":
            low = float(rng.uniform(450.0, 600.0))
            high = float(rng.uniform(1800.0, 2400.0))
            processed = butter_bandpass(segment, sr, low, high, order=4)
            params.update({"low_cut_hz": low, "high_cut_hz": high})
        elif option == "chorus_washout":
            wet_end = float(rng.uniform(0.50, 0.75))
            duck_db = float(rng.uniform(-4.5, -2.5))
            washed = linear_wet_delay(segment, sr, bpm, 0.10, wet_end, feedback=0.52)
            processed = washed * np.float32(db_to_amp(duck_db))
            params.update({"wet_start_percent": 10.0, "wet_end_percent": wet_end * 100.0, "duck_db": duck_db})
        elif option == "phaser_sweep":
            rate = float(rng.uniform(0.5, 2.5))
            processed = phaser_sweep(segment, sr, rate)
            params["lfo_rate_hz"] = rate
        elif option == "phase_inverted_widener":
            center_atten_db = float(rng.uniform(-8.0, -5.0))
            mid = np.mean(segment, axis=1, keepdims=True) * db_to_amp(center_atten_db)
            side = np.concatenate([segment[:, :1], -segment[:, 1:2]], axis=1)
            processed = (mid + side).astype(np.float32)
            params.update({"right_channel_phase_inverted": True, "center_atten_db": center_atten_db, "side_energy_percent": 100.0})
        else:
            target_sr = float(rng.uniform(6000.0, 9000.0))
            bits = int(rng.integers(6, 11))
            processed = bitcrush(segment, bits=bits, target_sr=target_sr, sr=sr)
            params.update({"target_sample_rate_hz": target_sr, "bits": bits})
        splice_processed_segment(automated["melody_bus"], start, end, processed, sr)
        ops.append(
            window
            | {
                "operation": "transition_point_fx",
                "technique": option,
                "target_bus": "melody_bus",
                "anchor_rule": "single_bus_only_other_six_buses_bypassed",
                "unaffected_anchor_buses": [bus for bus in BUS_NAMES if bus != "melody_bus"],
                "crossfade_ms": FX_CROSSFADE_MS,
                "parameters": params,
            }
        )
    return ops


def apply_stutter_roll(kick_bus: np.ndarray, start: int, end: int, sr: int, slice_ms: float) -> None:
    if end <= start:
        return
    chunk_len = min(max(1, int(round((slice_ms / 1000.0) * sr))), end - start)
    chunk = kick_bus[start : start + chunk_len].copy()
    if chunk.shape[0] == 0:
        return
    repeats = int(math.ceil((end - start) / chunk.shape[0]))
    rolled = np.tile(chunk, (repeats, 1))[: end - start]
    splice_processed_segment(kick_bus, start, end, rolled, sr)


def mini_fill_windows(grid: Dict, bpm: float, sr: int, length: int) -> List[Dict]:
    windows = []
    for section in grid["sections"]:
        if not any(token in section["name"] for token in ("verse", "chorus", "fill")):
            continue
        section_bar_count = int(section["bars"])
        for local_bar in range(4, section_bar_count + 1, 4):
            absolute_bar = int(section["start_bar"]) + local_bar - 1
            bar_start = bar_to_sample(absolute_bar, bpm, sr)
            two_beat_samples = int(round((2.0 * 60.0 / float(bpm)) * sr))
            end = min(length, bar_to_sample(absolute_bar + 1, bpm, sr))
            start = max(0, min(length, end - two_beat_samples))
            if end > start:
                windows.append(
                    {
                        "section": section["name"],
                        "bar": absolute_bar,
                        "start_sample": start,
                        "end_sample": end,
                    }
                )
    return windows


def section_activity_mask_for_stem(stem_name: str, grid: Dict, length: int, sr: int) -> Tuple[np.ndarray | None, Dict | None]:
    lower = stem_name.lower()
    if "verse" not in lower and "chorus" not in lower:
        return None, None
    mask = np.zeros(length, dtype=np.float32)
    if "verse" in lower:
        targets = ("verse1", "verse2")
        label = "verse"
    else:
        targets = ("chorus1", "chorus2")
        label = "chorus"
    for section in grid.get("sections", []):
        if section.get("name") in targets:
            mask[int(section["start_sample"]) : int(section["end_sample"])] = 1.0
    return mask, {"section_label": label, "active_sections": list(targets), "policy": "labeled_stem_only_active_in_matching_sections"}


def apply_section_energy_flips(automated: Dict[str, np.ndarray], grid: Dict, sr: int, rng: np.random.Generator) -> List[Dict]:
    ops = []
    for section in grid["sections"]:
        name = section["name"]
        start = int(section["start_sample"])
        end = int(section["end_sample"])
        if name.startswith("chorus"):
            replace_segment_gain(automated["perc_bus"], start, end, 1.5, sr)
            processed_fx = highpass(automated["fx_bus"][start:end], sr, cutoff=80.0)
            splice_processed_segment(automated["fx_bus"], start, end, processed_fx, sr)
            processed_pad = widen_stereo(automated["pad_bus"][start:end], amount=1.25)
            splice_processed_segment(automated["pad_bus"], start, end, processed_pad, sr)
            ops.append({"operation": "chorus_energy_lift", "section": name, "perc_gain_db": 1.5, "fx_highpass_hz": 80.0, "pad_width_multiplier": 1.25, "crossfade_ms": FX_CROSSFADE_MS})
        elif name.startswith("verse"):
            replace_segment_gain(automated["perc_bus"], start, end, -1.0, sr)
            processed_pad = widen_stereo(automated["pad_bus"][start:end], amount=0.75)
            splice_processed_segment(automated["pad_bus"], start, end, processed_pad, sr)
            ops.append({"operation": "verse_energy_containment", "section": name, "perc_gain_db": -1.0, "pad_width_multiplier": 0.75, "crossfade_ms": FX_CROSSFADE_MS})
    return ops


def apply_intro_profile(automated: Dict[str, np.ndarray], bpm: float, sr: int, length: int, rng: np.random.Generator) -> Dict:
    profile = int(rng.integers(1, 7))
    b1_4 = sample_range_for_bars(1, 4, bpm, sr, length)
    intro = sample_range_for_bars(1, 8, bpm, sr, length)
    params: Dict = {}
    if profile == 1:
        for bus in ("kick_bus", "perc_bus"):
            mute_segment(automated[bus], *b1_4, sr)
        low = float(rng.uniform(200.0, 450.0))
        high = float(rng.uniform(12_000.0, 15_000.0))
        for bus in ("melody_bus", "pad_bus"):
            apply_automated_lowpass_segment(automated[bus], sr, intro[0], intro[1], low, high)
        params.update({"profile_name": "filter_build", "muted_buses": ["kick_bus", "perc_bus"], "cutoff_start_hz": low, "cutoff_end_hz": high})
    elif profile == 2:
        for bus in ("kick_bus", "snare_bus", "bass_bus"):
            mute_segment(automated[bus], *intro, sr)
        wet = float(rng.uniform(0.40, 0.65))
        for bus in ("pad_bus", "fx_bus"):
            processed = stereo_delay_mix(automated[bus][intro[0] : intro[1]], sr, bpm, wet=wet, feedback=0.58)
            splice_processed_segment(automated[bus], intro[0], intro[1], processed, sr)
        params.update({"profile_name": "ambient_float", "muted_buses": ["kick_bus", "snare_bus", "bass_bus"], "wet_percent": wet * 100.0})
    elif profile == 3:
        for bus in BUS_NAMES:
            if bus != "melody_bus":
                mute_segment(automated[bus], *b1_4, sr)
        mute_segment(automated["kick_bus"], *intro, sr)
        params.update({"profile_name": "isolated_chop", "bars_1_4_only_bus": "melody_bus", "kick_drop_bar": 9})
    elif profile == 4:
        b1_6 = sample_range_for_bars(1, 6, bpm, sr, length)
        b7_8 = sample_range_for_bars(7, 8, bpm, sr, length)
        low = float(rng.uniform(350.0, 500.0))
        high = float(rng.uniform(2000.0, 3000.0))
        for bus in ("melody_bus", "pad_bus"):
            processed = butter_bandpass(automated[bus][b1_6[0] : b1_6[1]], sr, low, high, order=4)
            splice_processed_segment(automated[bus], b1_6[0], b1_6[1], processed, sr)
        processed_fx = automated["fx_bus"][b7_8[0] : b7_8[1]][::-1].copy() * np.linspace(0.25, 1.15, b7_8[1] - b7_8[0], dtype=np.float32)[:, None]
        splice_processed_segment(automated["fx_bus"], b7_8[0], b7_8[1], processed_fx, sr)
        params.update({"profile_name": "vinyl_radio_edit", "bandpass_low_hz": low, "bandpass_high_hz": high, "scratch_riser_bars": [7, 8]})
    elif profile == 5:
        beat_samples = int(round((60.0 / float(bpm)) * sr))
        rest_start = min(length, bar_to_sample(1, bpm, sr) + beat_samples)
        rest_end = b1_4[1]
        for bus in ("kick_bus", "snare_bus", "bass_bus"):
            mute_segment(automated[bus], rest_start, rest_end, sr)
        feedback = float(rng.uniform(0.45, 0.75))
        processed = stereo_delay_mix(automated["melody_bus"][b1_4[0] : b1_4[1]], sr, bpm, wet=0.65, feedback=feedback)
        splice_processed_segment(automated["melody_bus"], b1_4[0], b1_4[1], processed, sr)
        params.update({"profile_name": "echoing_downbeat", "drum_bass_mute_after_first_beat": True, "delay_feedback": feedback})
    else:
        cutoff = float(rng.uniform(380.0, 500.0))
        q = float(rng.uniform(2.0, 4.5))
        for bus in ("melody_bus", "pad_bus", "bass_bus"):
            apply_automated_lowpass_segment(automated[bus], sr, intro[0], intro[1], cutoff, 16_000.0)
        params.update({"profile_name": "underwater_submerge", "cutoff_start_hz": cutoff, "cutoff_end_hz": 16_000.0, "resonance_q": q})
    return {"operation": "randomized_intro_profile", "profile": profile, "start_bar": 1, "end_bar": 8, "crossfade_ms": FX_CROSSFADE_MS, **params}


def apply_outro_profile(automated: Dict[str, np.ndarray], bpm: float, sr: int, length: int, rng: np.random.Generator) -> Dict:
    profile = int(rng.integers(1, 7))
    outro = sample_range_for_bars(65, 72, bpm, sr, length)
    params: Dict = {}
    if profile == 1:
        for bus in ("kick_bus", "bass_bus"):
            mute_segment(automated[bus], *outro, sr)
        target = float(rng.uniform(800.0, 1500.0))
        for bus in ("melody_bus", "pad_bus"):
            apply_linear_gain(automated[bus], outro[0], outro[1], 0.0, -18.0)
            apply_automated_highpass_segment(automated[bus], sr, outro[0], outro[1], 20.0, target)
        params.update({"profile_name": "low_pass_dissolve", "muted_buses": ["kick_bus", "bass_bus"], "highpass_target_hz": target})
    elif profile == 2:
        start = max(outro[0], min(outro[1], outro[1] - int(round(6.0 * sr))))
        for bus in ("melody_bus", "pad_bus", "bass_bus"):
            processed = tape_brake(automated[bus][start:outro[1]])
            splice_processed_segment(automated[bus], start, outro[1], processed, sr)
        params.update({"profile_name": "sudden_tape_stop", "tape_stop_seconds": 6.0, "fx_bus_left_running": True})
    elif profile == 3:
        for bus in ("kick_bus", "snare_bus", "bass_bus"):
            mute_segment(automated[bus], *outro, sr)
        wet_max = float(rng.uniform(0.75, 0.95))
        for bus in ("melody_bus", "pad_bus"):
            processed = linear_wet_delay(automated[bus][outro[0] : outro[1]], sr, bpm, 0.0, wet_max, feedback=0.68)
            splice_processed_segment(automated[bus], outro[0], outro[1], processed, sr)
        params.update({"profile_name": "ambient_washout", "muted_buses": ["kick_bus", "snare_bus", "bass_bus"], "wet_end_percent": wet_max * 100.0})
    elif profile == 4:
        fx_gain = float(rng.uniform(5.0, 8.0))
        floor = float(rng.uniform(400.0, 600.0))
        apply_linear_gain(automated["fx_bus"], outro[0], outro[1], 0.0, fx_gain)
        for bus in ("melody_bus", "pad_bus"):
            apply_automated_lowpass_segment(automated[bus], sr, outro[0], outro[1], 14_000.0, floor)
        params.update({"profile_name": "vinyl_degradation", "fx_gain_end_db": fx_gain, "lowpass_floor_hz": floor})
    elif profile == 5:
        for bars, bus in (((65, 66), "bass_bus"), ((67, 68), "kick_bus"), ((69, 70), "snare_bus")):
            start, end = sample_range_for_bars(bars[0], bars[1], bpm, sr, length)
            mute_segment(automated[bus], start, end, sr)
        start, end = sample_range_for_bars(71, 72, bpm, sr, length)
        for bus in ("perc_bus", "pad_bus"):
            apply_linear_gain(automated[bus], start, end, 0.0, -80.0)
        params.update({"profile_name": "structural_stripper", "drop_schedule": {"65-66": "bass_bus", "67-68": "kick_bus", "69-70": "snare_bus", "71-72": "perc_bus_pad_bus_fade"}})
    else:
        for idx, bar in enumerate((69, 70, 71, 72)):
            start, end = sample_range_for_bars(bar, bar, bpm, sr, length)
            bits = int(max(1, 5 - idx + int(rng.integers(-1, 2))))
            target_sr = float(rng.uniform(4000.0, 9000.0))
            for bus in ("melody_bus", "pad_bus"):
                processed = bitcrush(automated[bus][start:end], bits=bits, target_sr=target_sr, sr=sr)
                splice_processed_segment(automated[bus], start, end, processed, sr)
        params.update({"profile_name": "broken_bit_stream", "final_four_bars_bitcrushed": True})
    return {"operation": "randomized_outro_profile", "profile": profile, "start_bar": 65, "end_bar": 72, "crossfade_ms": FX_CROSSFADE_MS, **params}


def apply_mini_fill_fx(automated: Dict[str, np.ndarray], grid: Dict, bpm: float, sr: int, rng: np.random.Generator) -> List[Dict]:
    ops = []
    technique_names = (
        "djfx_stutter_roll",
        "vinyl_tape_brake",
        "bass_vacuum_drop",
        "pitch_grime_shift",
        "pan_rotator_tremolo",
        "ring_modulated_metallic_ring",
        "stem_kill_drop",
        "mid_eq_isolation_break",
        "low_eq_muffle_isolation",
        "high_eq_sizzle_isolation",
    )
    for window in mini_fill_windows(grid, bpm, sr, max(bus.shape[0] for bus in automated.values())):
        start = int(window["start_sample"])
        end = int(window["end_sample"])
        technique = technique_names[int(rng.integers(0, len(technique_names)))]
        target_bus = "melody_bus"
        params: Dict = {}
        if technique == "djfx_stutter_roll":
            target_bus = "kick_bus"
            slice_ms = float(rng.uniform(40.0, 90.0))
            apply_stutter_roll(automated[target_bus], start, end, sr, slice_ms=slice_ms)
            params["slice_ms"] = slice_ms
        elif technique == "vinyl_tape_brake":
            processed = tape_brake(automated[target_bus][start:end])
            splice_processed_segment(automated[target_bus], start, end, processed, sr)
        elif technique == "bass_vacuum_drop":
            target_bus = "bass_bus"
            mute_segment(automated[target_bus], start, end, sr)
        elif technique == "pitch_grime_shift":
            semitones = float(rng.uniform(3.0, 6.0))
            processed = pitch_grime_shift(automated[target_bus][start:end], semitones)
            splice_processed_segment(automated[target_bus], start, end, processed, sr)
            params["semitones_down"] = semitones
        elif technique == "pan_rotator_tremolo":
            target_bus = "perc_bus"
            rate = float(rng.uniform(8.0, 16.0))
            processed = auto_pan(automated[target_bus][start:end], sr, rate)
            splice_processed_segment(automated[target_bus], start, end, processed, sr)
            params["lfo_rate_hz"] = rate
        elif technique == "ring_modulated_metallic_ring":
            carrier = float(rng.uniform(250.0, 600.0))
            processed = ring_modulate(automated[target_bus][start:end], sr, carrier)
            splice_processed_segment(automated[target_bus], start, end, processed, sr)
            params["carrier_hz"] = carrier
        elif technique == "stem_kill_drop":
            target_bus = str(rng.choice(["melody_bus", "pad_bus"]))
            mute_segment(automated[target_bus], start, end, sr)
        elif technique == "mid_eq_isolation_break":
            low = float(rng.uniform(250.0, 400.0))
            high = float(rng.uniform(2000.0, 3500.0))
            processed = butter_bandpass(automated[target_bus][start:end], sr, low, high, order=4)
            splice_processed_segment(automated[target_bus], start, end, processed, sr)
            params.update({"low_cut_hz": low, "high_cut_hz": high})
        elif technique == "low_eq_muffle_isolation":
            cutoff = float(rng.uniform(120.0, 220.0))
            processed = lowpass(automated[target_bus][start:end], sr, cutoff)
            splice_processed_segment(automated[target_bus], start, end, processed, sr)
            params["cutoff_hz"] = cutoff
        else:
            target_bus = "perc_bus"
            cutoff = float(rng.uniform(3000.0, 5500.0))
            processed = highpass(automated[target_bus][start:end], sr, cutoff)
            splice_processed_segment(automated[target_bus], start, end, processed, sr)
            params["cutoff_hz"] = cutoff
        ops.append(
            window
            | {
                "operation": "sp404_dj_mini_fill",
                "technique": technique,
                "target_bus": target_bus,
                "anchor_rule": "single_bus_only_other_six_buses_bypassed",
                "unaffected_anchor_buses": [bus for bus in BUS_NAMES if bus != target_bus],
                "crossfade_ms": FX_CROSSFADE_MS,
                "parameters": params,
            }
        )
    return ops


BUS_STAGE_OFFSETS = {
    "kick_bus": 3.0,
    "snare_bus": 2.0,
    "bass_bus": -2.0,
    "melody_bus": -2.0,
    "pad_bus": -4.0,
    "perc_bus": -4.0,
    "fx_bus": -6.0,
}

BUS_STAGE_CATEGORIES = {
    "kick_bus": "kick",
    "snare_bus": "snare",
    "perc_bus": "hats",
    "bass_bus": "bass",
    "melody_bus": "melody",
    "pad_bus": "pads",
    "fx_bus": "FX",
}


def stage_buses(buses: Dict[str, np.ndarray], pink_signal: np.ndarray, sr: int, bpm: float) -> Tuple[Dict[str, np.ndarray], Dict]:
    report = {}
    for bus_name, offset in BUS_STAGE_OFFSETS.items():
        if bus_name not in buses:
            continue
        bus = buses[bus_name]
        if float(np.max(np.abs(bus))) < EPS:
            continue
        category = BUS_STAGE_CATEGORIES[bus_name]
        low, high = BAND_LIMITS[category]
        band_bus = butter_bandpass(bus, sr, low, high)
        mask, _ = metering_mask(band_bus, sr, category, bpm=bpm)
        bus_rms = rms(band_bus, mask)
        band_pink = butter_bandpass(pink_signal, sr, low, high)
        pink_rms = rms(band_pink, mask)
        if pink_rms < EPS or bus_rms < EPS:
            continue
        target_rms = pink_rms * db_to_amp(offset)
        gain = target_rms / max(bus_rms, EPS)
        buses[bus_name] = (bus * np.float32(gain)).astype(np.float32)
        report[bus_name] = {
            "offset_db": offset,
            "bus_rms_dbfs": amp_to_db(bus_rms),
            "pink_rms_dbfs": amp_to_db(pink_rms),
            "applied_gain_db": amp_to_db(gain),
            "post_stage_rms_dbfs": amp_to_db(rms(buses[bus_name])),
        }
    return buses, report


def apply_fx_automation(buses: Dict[str, np.ndarray], bpm: float, sr: int, seed: int) -> Tuple[Dict[str, np.ndarray], Dict]:
    length = max(bus.shape[0] for bus in buses.values())
    automated = {name: buses[name].copy() for name in BUS_NAMES}
    grid = arrangement_grid(bpm, sr, length)
    operations: List[Dict] = []
    rng = np.random.default_rng(seed + 90_001)

    section_energy_ops = apply_section_energy_flips(automated, grid, sr, rng)
    operations.extend(section_energy_ops)

    intro_op = apply_intro_profile(automated, bpm, sr, length, rng)
    operations.append(intro_op)

    transition_ops = apply_transition_fx(automated, grid, bpm, sr, rng)
    operations.extend(transition_ops)

    mini_ops = apply_mini_fill_fx(automated, grid, bpm, sr, rng)
    operations.extend(mini_ops)

    outro_op = apply_outro_profile(automated, bpm, sr, length, rng)
    operations.append(outro_op)

    automated, sidechain_report = apply_dual_bus_sidechain(automated, sr)
    operations.append(sidechain_report)

    for drum_bus_name in ("kick_bus", "snare_bus", "perc_bus"):
        if drum_bus_name in automated:
            automated[drum_bus_name] = np.tanh(automated[drum_bus_name] * 1.2) / math.tanh(1.2)

    for bus_name in ("kick_bus", "snare_bus"):
        if bus_name in automated and float(np.max(np.abs(automated[bus_name]))) > EPS:
            compressed, comp_report = bus_compress(automated[bus_name], sr)
            automated[bus_name] = compressed
            operations.append({"operation": "bus_compression", "target_bus": bus_name, **comp_report})

    reverb_returns, reverb_report = apply_spatial_reverb_matrix(automated, sr, rng)
    operations.append(reverb_report)

    premaster = sum(automated.values(), np.zeros((length, 2), dtype=np.float32))
    premaster += sum(reverb_returns.values(), np.zeros((length, 2), dtype=np.float32))

    report = {
        "source": "fx-improved.mnd",
        "arrangement_grid": grid,
        "bus_names": list(BUS_NAMES),
        "operations": operations,
        "section_energy_flip_count": len(section_energy_ops),
        "intro_profile": intro_op,
        "outro_profile": outro_op,
        "transition_fx_count": len(transition_ops),
        "transition_fx_techniques": {op["technique"]: sum(1 for row in transition_ops if row["technique"] == op["technique"]) for op in transition_ops},
        "mini_fill_count": len(mini_ops),
        "mini_fill_techniques": {op["technique"]: sum(1 for row in mini_ops if row["technique"] == op["technique"]) for op in mini_ops},
        "single_bus_anchor_rule": {
            "applies_to": ["transition_point_fx", "sp404_dj_mini_fill"],
            "policy": "Each localized transition or mini-fill effect targets exactly one bus; all other six buses are bypassed and continue as anchors.",
            "localized_operation_count": len(transition_ops) + len(mini_ops),
        },
        "dual_bus_sidechain_report": sidechain_report,
        "spatial_reverb_report": reverb_report,
        "crossfade_ms": FX_CROSSFADE_MS,
        "policy": "fx-improved.mnd automation, dual-bus sidechain ducking, and localized reverb returns are applied after panning and gain staging, before final June 1 master-bus processing.",
    }
    return {"premaster": premaster, **automated, **reverb_returns}, report


def bus_compress(y: np.ndarray, sr: int, ratio: float = 3.0, attack_ms: float = 25.0,
                 release_ms: float = 150.0, target_gr_min: float = 2.0,
                 target_gr_max: float = 3.0, max_gr_cap: float = 3.0) -> Tuple[np.ndarray, Dict]:
    mono = np.sqrt(np.mean(np.square(y, dtype=np.float64), axis=1) + EPS).astype(np.float32)
    attack_s = attack_ms / 1000.0
    release_s = release_ms / 1000.0
    attack_samples = max(1, int(round(attack_s * sr)))
    release_samples = max(1, int(round(release_s * sr)))
    attack_env = scipy.ndimage.maximum_filter1d(mono, size=attack_samples, mode="nearest")
    release_kernel = np.exp(-np.arange(release_samples, dtype=np.float32) / max(float(release_samples), 1.0))
    release_kernel /= max(float(np.sum(release_kernel)), EPS)
    env = np.maximum(attack_env, scipy.signal.fftconvolve(attack_env, release_kernel, mode="same"))
    env_db = 20.0 * np.log10(np.maximum(env, EPS))
    level_db = 20.0 * np.log10(np.maximum(mono, EPS)).astype(np.float32)
    active = level_db[level_db > np.percentile(level_db, 40)]
    base = float(np.percentile(active, 85)) if active.size else -24.0
    thresholds = [base + off for off in (4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0)]
    best_y = y
    best_report: Dict = {}
    best_error = float("inf")
    target_mid = (target_gr_min + target_gr_max) / 2.0
    for threshold in thresholds:
        over = np.maximum(env_db - threshold, 0.0)
        reduction_db = -over * (1.0 - 1.0 / ratio)
        active_red = -reduction_db[over > 0.1]
        avg_gr = float(np.mean(active_red)) if active_red.size else 0.0
        max_gr = float(np.max(active_red)) if active_red.size else 0.0
        if max_gr > max_gr_cap:
            continue
        if avg_gr < target_gr_min * 0.5:
            continue
        error = abs(avg_gr - target_mid)
        if error < best_error:
            best_error = error
            gain = np.power(10.0, reduction_db / 20.0).astype(np.float32)
            best_y = (y * gain[:, None]).astype(np.float32)
            best_report = {
                "threshold_dbfs": float(threshold),
                "attack_ms": attack_ms,
                "release_ms": release_ms,
                "ratio": f"{ratio}:1",
                "average_active_gain_reduction_db": avg_gr,
                "max_gain_reduction_db": max_gr,
            }
    if best_report:
        makeup_db = best_report.get("average_active_gain_reduction_db", 0.0)
        best_y = best_y * db_to_amp(float(makeup_db))
        best_report["linear_makeup_gain_db"] = float(makeup_db)
    return best_y.astype(np.float32), best_report


def compressor_gain_curve(y: np.ndarray, sr: int, threshold_db: float) -> Tuple[np.ndarray, Dict]:
    ratio = 1.5
    attack = 0.030
    release = 0.200
    mono = np.sqrt(np.mean(np.square(y, dtype=np.float64), axis=1) + EPS).astype(np.float32)
    attack_samples = max(1, int(round(attack * sr)))
    release_samples = max(1, int(round(release * sr)))
    attack_env = scipy.ndimage.maximum_filter1d(mono, size=attack_samples, mode="nearest")
    release_kernel = np.exp(-np.arange(release_samples, dtype=np.float32) / max(float(release_samples), 1.0))
    release_kernel /= max(float(np.sum(release_kernel)), EPS)
    env = np.maximum(attack_env, scipy.signal.fftconvolve(attack_env, release_kernel, mode="same"))
    env_db = 20.0 * np.log10(np.maximum(env, EPS))
    over = np.maximum(env_db - threshold_db, 0.0)
    reduction_db = -over * (1.0 - 1.0 / ratio)
    gain = np.power(10.0, reduction_db / 20.0).astype(np.float32)
    compressed = y * gain[:, None]
    active_reduction = -reduction_db[over > 0.1]
    avg_gr = float(np.mean(active_reduction)) if active_reduction.size else 0.0
    max_gr = float(np.max(active_reduction)) if active_reduction.size else 0.0
    return compressed.astype(np.float32), {
        "threshold_dbfs": float(threshold_db),
        "attack_ms": 30.0,
        "release_ms": 200.0,
        "ratio": "1.5:1",
        "average_active_gain_reduction_db": avg_gr,
        "max_gain_reduction_db": max_gr,
    }


def glue_compress(y: np.ndarray, sr: int) -> Tuple[np.ndarray, Dict]:
    mono = np.sqrt(np.mean(np.square(y, dtype=np.float64), axis=1) + EPS)
    level_db = 20.0 * np.log10(np.maximum(mono, EPS)).astype(np.float32)
    active = level_db[level_db > np.percentile(level_db, 40)]
    base = float(np.percentile(active, 85)) if active.size else -24.0
    thresholds = [base + offset for offset in (2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0)]
    best_y = y
    best_report: Dict = {}
    best_error = float("inf")
    max_gr_cap = 5.0
    for threshold in thresholds:
        candidate, report = compressor_gain_curve(y, sr, threshold)
        max_gr = float(report.get("max_gain_reduction_db", 0.0))
        if max_gr > max_gr_cap:
            continue
        error = abs(float(report["average_active_gain_reduction_db"]) - 2.5)
        if error < best_error:
            best_error = error
            best_y = candidate
            best_report = report
    if not best_report:
        for threshold in thresholds:
            candidate, report = compressor_gain_curve(y, sr, threshold)
            max_gr = float(report.get("max_gain_reduction_db", 0.0))
            if max_gr <= max_gr_cap * 1.5:
                best_y = candidate
                best_report = report
                break
    if not best_report:
        best_y, best_report = compressor_gain_curve(y, sr, thresholds[-1])
    makeup_db = best_report.get("average_active_gain_reduction_db", 0.0)
    best_y = best_y * db_to_amp(float(makeup_db))
    best_report["linear_makeup_gain_db"] = float(makeup_db)
    best_report["target_gain_reduction_db"] = [2.0, 3.0]
    return best_y.astype(np.float32), best_report


def saturate(y: np.ndarray, wet: float = SATURATION_WET) -> np.ndarray:
    drive = 1.6
    saturated = np.tanh(y * drive) / math.tanh(drive)
    return (y * (1.0 - wet) + saturated * wet).astype(np.float32)


def true_peak_db(y: np.ndarray, sr: int) -> float:
    oversampled = scipy.signal.resample_poly(y, 4, 1, axis=0)
    return amp_to_db(float(np.max(np.abs(oversampled))))


def lookahead_limiter(y: np.ndarray, sr: int, ceiling_db: float, measure_true_peak: bool = True) -> Tuple[np.ndarray, Dict]:
    ceiling = db_to_amp(ceiling_db)
    lookahead_samples = max(1, int(round(0.003 * sr)))
    peak = np.max(np.abs(y), axis=1)
    future_peak = scipy.ndimage.maximum_filter1d(
        peak,
        size=lookahead_samples + 1,
        mode="nearest",
        origin=-(lookahead_samples // 2),
    )
    gain = np.minimum(1.0, ceiling / np.maximum(future_peak, EPS)).astype(np.float32)
    limited = y * gain[:, None]
    sample_peak = float(np.max(np.abs(limited)))
    if sample_peak > ceiling:
        limited *= np.float32(ceiling / sample_peak)
    measured_true_peak = true_peak_db(limited, sr) if measure_true_peak else amp_to_db(float(np.max(np.abs(limited))))
    if measure_true_peak and measured_true_peak > ceiling_db:
        limited *= np.float32(db_to_amp(ceiling_db - measured_true_peak))
        measured_true_peak = true_peak_db(limited, sr)
    attenuation = -20.0 * np.log10(np.maximum(gain, EPS))
    active = attenuation[attenuation > 0.01]
    return limited.astype(np.float32), {
        "ceiling_dbtp": ceiling_db,
        "lookahead_ms": 3.0,
        "max_limiter_attenuation_db": float(np.max(active)) if active.size else 0.0,
        "average_active_limiter_attenuation_db": float(np.mean(active)) if active.size else 0.0,
        "final_true_peak_dbtp": measured_true_peak,
    }


def integrated_lufs(y: np.ndarray, sr: int, meter: pyln.Meter | None = None) -> float:
    if meter is None:
        meter = pyln.Meter(sr, block_size=0.400)
    return float(meter.integrated_loudness(y.astype(np.float64)))


def loudness_limited_master(y: np.ndarray, sr: int) -> Tuple[np.ndarray, Dict]:
    meter = pyln.Meter(sr, block_size=0.400)
    unscaled_lufs = integrated_lufs(y, sr, meter)
    initial_gain = TARGET_LUFS - unscaled_lufs
    low, high = initial_gain - 6.0, initial_gain + 24.0
    best_y = y
    best_report: Dict = {}
    best_error = float("inf")
    for _ in range(8):
        mid = (low + high) * 0.5
        candidate, limiter_report = lookahead_limiter(y * db_to_amp(mid), sr, TRUE_PEAK_CEILING_DBTP, measure_true_peak=True)
        lufs = integrated_lufs(candidate, sr, meter)
        error = abs(lufs - TARGET_LUFS)
        if error < best_error:
            best_error = error
            best_y = candidate
            best_report = {
                **limiter_report,
                "input_gain_db": float(mid),
                "integrated_lufs": lufs,
                "target_lufs": TARGET_LUFS,
                "loudness_error_db": float(lufs - TARGET_LUFS),
            }
        if lufs < TARGET_LUFS:
            low = mid
        else:
            high = mid
    final_y, final_limiter_report = lookahead_limiter(best_y, sr, TRUE_PEAK_CEILING_DBTP, measure_true_peak=True)
    final_lufs = integrated_lufs(final_y, sr, meter)
    refinement_passes = 0
    while final_lufs < TARGET_LUFS - 0.15 and refinement_passes < 4:
        makeup_db = min(TARGET_LUFS - final_lufs, 3.0)
        final_y, final_limiter_report = lookahead_limiter(
            final_y * db_to_amp(makeup_db),
            sr,
            TRUE_PEAK_CEILING_DBTP,
            measure_true_peak=True,
        )
        final_lufs = integrated_lufs(final_y, sr, meter)
        refinement_passes += 1
    best_report.update(final_limiter_report)
    best_report["integrated_lufs"] = final_lufs
    best_report["loudness_error_db"] = float(final_lufs - TARGET_LUFS)
    best_report["unscaled_integrated_lufs"] = unscaled_lufs
    best_report["search_iterations"] = 8
    best_report["final_refinement_passes"] = refinement_passes
    return final_y.astype(np.float32), best_report


def apply_master_chain(mix: np.ndarray, sr: int) -> Tuple[np.ndarray, Dict]:
    stages: List[Dict] = []
    y = highpass(mix, sr, 20.0)
    stages.append({"stage": "high_pass_filter", "cutoff_hz": 20.0})
    y = peaking_eq(y, sr, center_hz=250.0, gain_db=-1.5, q=1.5)
    stages.append({"stage": "surgical_eq_cut", "center_hz": 250.0, "gain_db": -1.5})
    y = peaking_eq(y, sr, center_hz=3_200.0, gain_db=-1.5, q=1.0)
    stages.append({"stage": "surgical_eq_cut", "center_hz": 3200.0, "gain_db": -1.5})
    y = lowpass(y, sr, cutoff=14_000.0)
    stages.append({"stage": "low_pass_filter", "rolloff_start_hz": 14_000.0, "order": 2})
    y, compression_report = glue_compress(y, sr)
    stages.append({"stage": "glue_compression", **compression_report})
    y = saturate(y, SATURATION_WET)
    stages.append({"stage": "lofi_saturation", "wet_percent": SATURATION_WET * 100.0})
    y, limiter_report = loudness_limited_master(y, sr)
    stages.append({"stage": "true_peak_limiter", **limiter_report})
    return y, {"stages": stages, "limiter": limiter_report, "compression": compression_report}


def pink_reference_rms_by_category(length: int, sr: int, seed: int) -> Tuple[Dict[str, float], np.ndarray]:
    pink = make_pink_noise(length, sr, seed)
    refs: Dict[str, float] = {}
    for category, (low, high) in BAND_LIMITS.items():
        refs[category] = rms(butter_bandpass(pink, sr, low, high), None)
    return refs, pink


def render_stem(stem: Stem, length: int, pink_refs: Dict[str, float], pink_signal: np.ndarray, sr: int, bpm: float) -> Tuple[np.ndarray, Dict]:
    y = read_audio(stem.path, sr)
    if y.shape[0] < length:
        y = np.pad(y, ((0, length - y.shape[0]), (0, 0)))
    elif y.shape[0] > length:
        y = y[:length]
    panned = equal_power_pan(y, stem.pan)
    staged, staging_report = gain_stage_stem(panned, stem.category, pink_signal, pink_refs[stem.category], sr, bpm=bpm)
    grid = arrangement_grid(bpm, sr, length)
    section_mask, section_report = section_activity_mask_for_stem(stem.path.name, grid, length, sr)
    if section_mask is not None:
        staged = apply_smooth_mask(staged, section_mask, sr)
    row = {
        "file": stem.path.name,
        "category": stem.category,
        "bus": bus_for_category(stem.category),
        "pan": stem.pan,
        "source_sr": stem.source_sr,
        "processing_order": ["read_audio", "equal_power_pan", "gain_stage_stem", "section_label_mask", "bus_route"],
        "panning_before_gain_staging": True,
        "section_activity": section_report or {"section_label": None, "policy": "always_active"},
        **staging_report,
    }
    return staged, row


def empty_buses(length: int) -> Dict[str, np.ndarray]:
    return {name: np.zeros((length, 2), dtype=np.float32) for name in BUS_NAMES}


def render_stems_parallel(stems: Sequence[Stem], length: int, pink_refs: Dict[str, float], pink_signal: np.ndarray, sr: int, bpm: float, workers: int) -> Tuple[Dict[str, np.ndarray], List[Dict]]:
    buses = empty_buses(length)
    stem_rows: List[Dict] = []
    if workers <= 1:
        for stem in stems:
            staged, row = render_stem(stem, length, pink_refs, pink_signal, sr, bpm)
            buses[row["bus"]] += staged
            stem_rows.append(row)
        return buses, stem_rows

    pending: set[concurrent.futures.Future[Tuple[np.ndarray, Dict]]] = set()
    stem_iter = iter(stems)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in range(min(workers, len(stems))):
            pending.add(executor.submit(render_stem, next(stem_iter), length, pink_refs, pink_signal, sr, bpm))
        while pending:
            done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                staged, row = future.result()
                buses[row["bus"]] += staged
                stem_rows.append(row)
                try:
                    pending.add(executor.submit(render_stem, next(stem_iter), length, pink_refs, pink_signal, sr, bpm))
                except StopIteration:
                    pass
    stem_rows.sort(key=lambda row: row["file"])
    return buses, stem_rows


def render_target(target: Dict, seed: int, sr: int, workers: int) -> Dict:
    recordings = Path(target["recordings"])
    if not recordings.exists():
        raise FileNotFoundError(f"Missing recordings directory: {recordings}")
    wavs = sorted(recordings.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No wav files found in {recordings}")

    stems = assign_pans(wavs, seed)
    length = max(
        int(math.ceil(stem.frames * (sr / stem.source_sr))) if stem.source_sr != sr else stem.frames
        for stem in stems
    )
    pink_refs, pink_signal = pink_reference_rms_by_category(length, sr, seed + 10_000)
    print(f"rendering {target['name']} with {workers} stem workers", flush=True)
    buses, stem_rows = render_stems_parallel(stems, length, pink_refs, pink_signal, sr, float(target["bpm"]), workers)
    buses, bus_stage_report = stage_buses(buses, pink_signal, sr, float(target["bpm"]))
    automated_buses, fx_report = apply_fx_automation(buses, float(target["bpm"]), sr, seed)
    mix = automated_buses["premaster"]

    pre_peak_dbfs = amp_to_db(float(np.max(np.abs(mix))))
    headroom_gain_db = min(0.0, PREMASTER_HEADROOM_DB - pre_peak_dbfs)
    mix *= np.float32(db_to_amp(headroom_gain_db))
    post_peak_dbfs = amp_to_db(float(np.max(np.abs(mix))))
    master, master_report = apply_master_chain(mix, sr)

    song_root = recordings.parent
    out_dir = song_root / "june1_corrected"
    mix_path = out_dir / "corrected_mix.wav"
    master_path = out_dir / "corrected_master.wav"
    analysis_path = out_dir / "corrected_analysis.json"
    write_audio(mix_path, mix, sr)
    write_audio(master_path, master, sr)

    final_lufs = master_report["limiter"]["integrated_lufs"]
    final_true_peak = master_report["limiter"]["final_true_peak_dbtp"]
    analysis = {
        "track": target["name"],
        "recordings": str(recordings),
        "output_dir": str(out_dir),
        "sample_rate": sr,
        "bpm": target["bpm"],
        "seed": seed,
        "workers": workers,
        "processing_order": ["read_audio", "equal_power_pan", "gain_stage_stem", "section_label_mask", "bus_route", "fx_improved_bus_automation", "dual_bus_sidechain", "localized_spatial_reverb_returns", "master_chain"],
        "panning_before_gain_staging": True,
        "stem_count": len(stem_rows),
        "pan_positions": {row["file"]: row["pan"] for row in stem_rows},
        "stem_rms_values": {
            row["file"]: {
                "category": row["category"],
                "bus": row["bus"],
                "active_band_rms_dbfs": row["active_band_rms_dbfs"],
                "post_gain_fullband_rms_dbfs": row["post_gain_fullband_rms_dbfs"],
                "applied_gain_db": row["applied_gain_db"],
                "mask_mode": row.get("mask_mode"),
                "metered_sample_fraction": row.get("metered_sample_fraction"),
                "hit_window_count": row.get("hit_window_count"),
                "processing_order": row.get("processing_order"),
                "panning_before_gain_staging": row.get("panning_before_gain_staging"),
                "section_activity": row.get("section_activity"),
            }
            for row in stem_rows
        },
        "mix_metrics": {
            "pre_headroom_peak_dbfs": pre_peak_dbfs,
            "headroom_gain_db": headroom_gain_db,
            "corrected_mix_peak_dbfs": post_peak_dbfs,
            "required_headroom_db": PREMASTER_HEADROOM_DB,
        },
        "submix_bus_architecture": {
            "bus_names": list(BUS_NAMES),
            "routing": {
                "bass": "bass_bus",
                "melody": "melody_bus",
                "pads": "pad_bus",
                "kick": "kick_bus",
                "snare": "snare_bus",
                "hats": "perc_bus",
                "texture": "fx_bus",
                "FX": "fx_bus",
            },
            "bus_peak_dbfs_after_fx": {
                name: amp_to_db(float(np.max(np.abs(audio)))) for name, audio in automated_buses.items() if name in BUS_NAMES
            },
            "reverb_return_peak_dbfs": {
                name: amp_to_db(float(np.max(np.abs(audio))))
                for name, audio in automated_buses.items()
                if name.endswith("_reverb_return")
            },
        },
        "fx_automation_report": fx_report,
        "final_lufs_metrics": {
            "integrated_lufs": final_lufs,
            "target_lufs": TARGET_LUFS,
            "loudness_error_db": final_lufs - TARGET_LUFS,
            "true_peak_dbtp": final_true_peak,
            "true_peak_ceiling_dbtp": TRUE_PEAK_CEILING_DBTP,
        },
        "master_chain": master_report,
        "stems": stem_rows,
        "deliverables": {
            "corrected_mix": str(mix_path),
            "corrected_master": str(master_path),
            "corrected_analysis": str(analysis_path),
        },
    }
    analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    return analysis


def short_summary(results: Iterable[Dict]) -> Dict:
    tracks = {}
    for row in results:
        tracks[row["track"]] = {
            "output_dir": row["output_dir"],
            "stem_count": row["stem_count"],
            "submix_bus_count": len(row.get("submix_bus_architecture", {}).get("bus_names", [])),
            "mini_fill_count": row.get("fx_automation_report", {}).get("mini_fill_count"),
            "transition_fx_count": row.get("fx_automation_report", {}).get("transition_fx_count"),
            "intro_profile": row.get("fx_automation_report", {}).get("intro_profile", {}).get("profile_name"),
            "outro_profile": row.get("fx_automation_report", {}).get("outro_profile", {}).get("profile_name"),
            "corrected_mix_peak_dbfs": row["mix_metrics"]["corrected_mix_peak_dbfs"],
            "integrated_lufs": row["final_lufs_metrics"]["integrated_lufs"],
            "true_peak_dbtp": row["final_lufs_metrics"]["true_peak_dbtp"],
            "deliverables": row["deliverables"],
        }
    return {
        "prompt": ["prompt_june1.md", "fx.md", "fx-improved.mnd"],
        "target_lufs": TARGET_LUFS,
        "true_peak_ceiling_dbtp": TRUE_PEAK_CEILING_DBTP,
        "tracks": tracks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducible bounded pan positions.")
    parser.add_argument("--sr", type=int, default=SR_TARGET, help="Render sample rate.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent stem workers per target.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workers = max(1, int(args.workers))
    seed = int(args.seed) if args.seed is not None else random.SystemRandom().randint(1, 2_147_483_647)
    results = [render_target(target, seed=seed + idx, sr=args.sr, workers=workers) for idx, target in enumerate(TARGETS)]
    aggregate = short_summary(results)
    aggregate_path = ROOT / "output" / "june1_corrected_analysis.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
