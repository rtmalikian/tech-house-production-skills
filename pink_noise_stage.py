
import json, os
import numpy as np
import pyloudnorm as pyln
from scipy.signal import butter, sosfilt

PINK_REFERENCE_LUFS = -18.0
BANDS = [(31.5,22.3,44.5),(63,44.5,89.1),(125,89.1,178.2),(250,178.2,356.4),
         (500,356.4,712.7),(1000,712.7,1425.4),(2000,1425.4,2850.9),
         (4000,2850.9,5701.8),(8000,5701.8,11403.6),(16000,11403.6,20000)]

def pink_noise_stage_processed_stem(y, sr, role="default", layer=None, stem_name=""):
    y = _ensure_stereo(np.asarray(y, dtype=np.float32))
    pink = _set_lufs(_pink_noise(len(y), sr), sr, PINK_REFERENCE_LUFS)
    stem_bands = _band_rms_db(y, sr)
    pink_bands = _band_rms_db(pink, sr)
    active = _active_band_indices(stem_bands)
    target = _target_margin(role, layer, stem_name)
    if active:
        margins = [stem_bands[i]["rms_db"] - pink_bands[i]["rms_db"] for i in active]
        weights = [max(stem_bands[i]["rms_db"] + 90.0, 1.0) for i in active]
        current = float(np.average(margins, weights=weights))
        gain_db = float(np.clip(target - current, -18.0, 12.0))
    else:
        current, gain_db = 0.0, 0.0
    before = _metrics(y, sr)
    y_out = y * (10.0 ** (gain_db / 20.0))
    after = _metrics(y_out, sr)
    path = os.environ.get("GAIN_FLOW_TEST_LOG")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps({
                "stem_name": stem_name, "role": role, "layer": layer,
                "mode": "pink_noise_staged_flow", "reference": "pink_noise_octave_bands",
                "pink_reference_lufs": PINK_REFERENCE_LUFS,
                "target_margin_db": target, "current_margin_db": current,
                "final_gain_db": gain_db,
                "active_band_centers_hz": [BANDS[i][0] for i in active],
                "processed_before_final_gain": before,
                "processed_after_final_gain": after,
                "stem_bands": stem_bands, "pink_bands": pink_bands,
            }) + "\n")
    return y_out.astype(np.float32), gain_db

def _target_margin(role, layer, stem_name):
    name, role, layer = (stem_name or "").lower(), (role or "default").lower(), (layer or "").lower()
    if role in ("kick", "snare", "bass") or "kick" in name or "snare" in name:
        return 2.0
    if role in ("hat", "clap") or any(t in name for t in ("hat", "ride", "crash", "tambourine", "maracas")):
        return -1.0
    if role in ("melody", "chorus") or layer in ("melody_lead", "chorus_poly", "chorus_brass"):
        return 1.0
    return 0.0

def _pink_noise(n, sr):
    rng = np.random.default_rng(424242)
    n_fft = int(2 ** np.ceil(np.log2(max(n, 2))))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    scale = np.ones_like(freqs); scale[1:] = 1.0 / np.sqrt(freqs[1:])
    phases = rng.uniform(0.0, 2.0 * np.pi, len(freqs))
    noise = np.fft.irfft(scale * np.exp(1j * phases), n_fft)[:n]
    noise = noise / max(np.max(np.abs(noise)), 1e-12)
    return np.column_stack([noise, noise]).astype(np.float32)

def _set_lufs(y, sr, target):
    meter = pyln.Meter(sr)
    try:
        current = float(meter.integrated_loudness(y))
    except Exception:
        current = -70.0
    if current <= -70.0 or not np.isfinite(current):
        return y
    return (y * (10.0 ** ((target - current) / 20.0))).astype(np.float32)

def _band_rms_db(y, sr):
    mono = np.mean(_ensure_stereo(y), axis=1)
    out, nyq = [], sr / 2.0
    for center, low, high in BANDS:
        lo, hi = max(low, 20.0), min(high, nyq * 0.98)
        if lo >= hi:
            rms_db = -120.0
        else:
            sos = butter(3, [lo / nyq, hi / nyq], btype="band", output="sos")
            band = sosfilt(sos, mono.astype(np.float64))
            rms_db = 20.0 * np.log10(float(np.sqrt(np.mean(band ** 2)) + 1e-12))
        out.append({"center_hz": center, "low_hz": lo, "high_hz": hi, "rms_db": float(rms_db)})
    return out

def _active_band_indices(stem_bands):
    max_db = max(b["rms_db"] for b in stem_bands)
    return [i for i, b in enumerate(stem_bands) if b["rms_db"] >= max_db - 24.0 and b["rms_db"] > -85.0]

def _metrics(y, sr):
    y = _ensure_stereo(np.asarray(y, dtype=np.float32))
    meter = pyln.Meter(sr)
    try:
        lufs = float(meter.integrated_loudness(y))
    except Exception:
        lufs = -70.0
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-12)
    return {"lufs": lufs, "peak_db": 20.0 * np.log10(max(peak, 1e-12)),
            "crest_db": 20.0 * np.log10(max(peak, 1e-12) / rms)}

def _ensure_stereo(y):
    if y.ndim == 1:
        return np.column_stack([y, y])
    if y.shape[1] == 1:
        return np.repeat(y, 2, axis=1)
    return y[:, :2]
