#!/usr/bin/env python3
"""
Tech House Audio Quality Assessment
====================================
Objective metrics for evaluating tech house audio at every stage of
the production pipeline. Each metric maps to a real-world listening
skill, with measurable thresholds instead of arbitrary limits.

Usage:
    python assessment.py <audio_file.wav>
    python assessment.py --compare <mix.wav> <reference.wav>
    python assessment.py --stage <audio.wav> --stage-name mastering
"""
import sys
import json
import argparse
import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import librosa
except ImportError:
    librosa = None

try:
    import pyloudnorm as pyln
except ImportError:
    pyln = None


# ============================================================================
# CORE METRICS
# ============================================================================

def crest_factor_db(signal):
    """Peak-to-RMS ratio in dB. Higher = more punchy/dynamic.
    Maps to: Producer's sense of 'punch' and 'impact'."""
    peak = np.max(np.abs(signal))
    rms = np.sqrt(np.mean(signal**2))
    return round(20 * np.log10(peak / (rms + 1e-10)), 1)


def dynamic_range_db(signal, frame_size=2048, hop=512):
    """Difference between 95th and 5th percentile of frame RMS.
    Maps to: Producer's sense of 'dynamics' vs 'crushed'."""
    rms_frames = []
    for i in range(0, len(signal) - frame_size, hop):
        frame = signal[i:i+frame_size]
        rms_frames.append(np.sqrt(np.mean(frame**2)))
    rms_db = 20 * np.log10(np.array(rms_frames) + 1e-10)
    return round(float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5)), 1)


def spectral_balance(signal, sr):
    """Energy distribution across 5 frequency bands.
    Maps to: Producer's sense of 'tonal balance'."""
    fft = np.fft.rfft(signal)
    mag = np.abs(fft)**2
    freqs = np.fft.rfftfreq(len(signal), 1/sr)

    bands = {
        'sub_20_60': (20, 60),
        'bass_60_250': (60, 250),
        'low_mid_250_2k': (250, 2000),
        'presence_2k_6k': (2000, 6000),
        'air_6k_20k': (6000, 20000),
    }

    energies = {}
    total = 0
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        e = float(np.sum(mag[mask]))
        energies[name] = e
        total += e

    # Convert to percentages and dB
    result = {}
    for name, e in energies.items():
        pct = (e / (total + 1e-10)) * 100
        result[name] = {
            'percent': round(pct, 1),
            'db_relative': round(10 * np.log10(e / (total + 1e-10) + 1e-10), 1),
        }
    return result


def stereo_correlation(left, right):
    """Pearson correlation between L and R channels.
    +1 = mono, 0 = uncorrelated stereo, -1 = phase inverted.
    Maps to: Producer's sense of 'width'."""
    return round(float(np.corrcoef(left, right)[0, 1]), 3)


def side_to_mid_ratio(left, right):
    """Ratio of side (L-R) energy to mid (L+R) energy.
    0 = mono, 0.5 = moderate, 1.0 = very wide.
    Maps to: Producer's sense of 'width'."""
    mid = (left + right) / 2
    side = (left - right) / 2
    mid_rms = np.sqrt(np.mean(mid**2))
    side_rms = np.sqrt(np.mean(side**2))
    return round(float(side_rms / (mid_rms + 1e-10)), 3)


def warmth_ratio(signal, sr):
    """Ratio of warm frequencies (60-250 Hz) to presence (2-6 kHz).
    Positive = warm, negative = bright.
    Maps to: Producer's sense of 'warmth' vs 'brightness'."""
    fft = np.fft.rfft(signal)
    mag = np.abs(fft)**2
    freqs = np.fft.rfftfreq(len(signal), 1/sr)

    warm = float(np.sum(mag[(freqs >= 60) & (freqs < 250)]))
    presence = float(np.sum(mag[(freqs >= 2000) & (freqs < 6000)]))
    return round(10 * np.log10(warm / (presence + 1e-10)), 1)


def spectral_centroid_hz(signal, sr):
    """Center of spectral mass in Hz.
    Higher = brighter, lower = darker.
    Maps to: Producer's sense of 'presence'."""
    if librosa:
        return round(float(np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr))), 0)
    # Fallback
    fft = np.fft.rfft(signal)
    mag = np.abs(fft)
    freqs = np.fft.rfftfreq(len(signal), 1/sr)
    return round(float(np.sum(freqs * mag) / (np.sum(mag) + 1e-10)), 0)


def transient_density(signal, sr):
    """Transients per second. Higher = more percussive.
    Maps to: Producer's sense of 'energy' and 'drive'."""
    if librosa:
        onsets = librosa.onset.onset_detect(y=signal, sr=sr)
        return round(len(onsets) / (len(signal) / sr), 1)
    # Fallback: count peaks above threshold
    envelope = np.abs(signal)
    threshold = np.max(envelope) * 0.3
    peaks = np.where((envelope[:-2] < envelope[1:-1]) &
                     (envelope[1:-1] > envelope[2:]) &
                     (envelope[1:-1] > threshold))[0]
    return round(len(peaks) / (len(signal) / sr), 1)


def lufs_integrated(signal, sr):
    """Integrated loudness in LUFS (ITU-R BS.1770-4).
    Maps to: Producer's sense of 'loudness'."""
    if pyln:
        meter = pyln.Meter(sr)
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        return round(float(meter.integrated_loudness(signal)), 1)
    return None


def loudness_range(signal, sr):
    """Loudness range in LU. Higher = more dynamic.
    Maps to: Producer's sense of 'dynamic range'."""
    if pyln:
        meter = pyln.Meter(sr)
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        return round(float(meter.loudness_range(signal)), 1)
    return None


def peak_dbfs(signal):
    """Peak level in dBFS."""
    return round(20 * np.log10(np.max(np.abs(signal)) + 1e-10), 1)


# ============================================================================
# STAGE-SPECIFIC ASSESSMENTS
# ============================================================================

def assess_sound_selection(kick_path):
    """Assess a kick sample for tech house suitability."""
    if not librosa:
        return {'error': 'librosa not installed'}
    y, sr = librosa.load(kick_path, sr=44100)

    # Fundamental
    try:
        f0 = librosa.yin(y, fmin=30, fmax=100, sr=sr)
        f0_clean = f0[f0 > 0]
        fundamental = float(np.median(f0_clean)) if len(f0_clean) > 0 else 0
    except:
        fundamental = 0

    # Crest factor (punch)
    crest = crest_factor_db(y)

    # Tail length
    envelope = np.abs(y)
    peak = np.max(envelope)
    threshold = peak * 0.1
    above = np.where(envelope > threshold)[0]
    tail_ms = float((above[-1] - above[0]) / sr * 1000) if len(above) > 0 else 0

    # Click ratio (2-5 kHz in attack)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    click = np.where((freqs >= 2000) & (freqs <= 5000))[0]
    click_ratio = float(np.mean(S[click, :5]) / (np.mean(S[:, :5]) + 1e-10))

    score = 0
    checks = []

    # Fundamental in useful range?
    if 35 <= fundamental <= 65:
        score += 1
        checks.append(('fundamental', f'{fundamental:.0f} Hz', 'PASS'))
    else:
        checks.append(('fundamental', f'{fundamental:.0f} Hz', 'FAIL (want 35-65 Hz)'))

    # Punchy enough?
    if crest >= 10:
        score += 1
        checks.append(('crest_factor', f'{crest} dB', 'PASS'))
    else:
        checks.append(('crest_factor', f'{crest} dB', 'FAIL (want >= 10 dB)'))

    # Tail clear before next beat at 126 BPM?
    beat_ms = 60000 / 126  # 476ms
    if tail_ms < beat_ms * 0.8:
        score += 1
        checks.append(('tail_length', f'{tail_ms:.0f} ms', 'PASS'))
    else:
        checks.append(('tail_length', f'{tail_ms:.0f} ms', f'FAIL (must clear in {beat_ms*0.8:.0f} ms)'))

    # Has click?
    if click_ratio > 0.1:
        score += 1
        checks.append(('click_ratio', f'{click_ratio:.3}', 'PASS'))
    else:
        checks.append(('click_ratio', f'{click_ratio:.3}', 'FAIL (want > 0.1)'))

    return {
        'score': f'{score}/4',
        'checks': checks,
        'fundamental_hz': round(fundamental, 1),
        'crest_db': crest,
        'tail_ms': round(tail_ms, 1),
        'click_ratio': round(click_ratio, 3),
    }


def assess_mix(audio_path, sr=44100):
    """Full mix quality assessment."""
    data, sr = sf.read(audio_path)
    if data.ndim == 1:
        mono = data
        left = right = data
    else:
        left, right = data[:, 0], data[:, 1]
        mono = (left + right) / 2

    result = {
        'peak_dbfs': peak_dbfs(mono),
        'crest_factor_db': crest_factor_db(mono),
        'dynamic_range_db': dynamic_range_db(mono),
        'spectral_centroid_hz': spectral_centroid_hz(mono, sr),
        'warmth_ratio_db': warmth_ratio(mono, sr),
        'transient_density': transient_density(mono, sr),
        'spectral_balance': spectral_balance(mono, sr),
    }

    if lufs_integrated:
        result['integrated_lufs'] = lufs_integrated(data, sr)
    if loudness_range:
        result['loudness_range_lu'] = loudness_range(data, sr)

    if data.ndim == 2:
        result['stereo'] = {
            'correlation': stereo_correlation(left, right),
            'side_to_mid_ratio': side_to_mid_ratio(left, right),
        }

    # Issues
    issues = []
    sb = result['spectral_balance']

    if sb['sub_20_60']['percent'] > 25:
        issues.append('Excessive sub energy — tighten sidechain or HPF bass')
    if sb['low_mid_250_2k']['percent'] > 30:
        issues.append('Muddy low-mids — cut 200-400 Hz on multiple elements')
    if sb['air_6k_20k']['percent'] < 5:
        issues.append('Lacking air — boost 10-12 kHz shelf or brighten hats')
    if result['crest_factor_db'] < 6:
        issues.append('Over-compressed — reduce compression or limiter GR')
    if result['crest_factor_db'] > 16:
        issues.append('Under-compressed — may lack energy for club play')

    if data.ndim == 2:
        if result['stereo']['correlation'] < 0:
            issues.append('PHASE ISSUE — stereo correlation below 0, check for phase cancellation')

    result['issues'] = issues
    result['passes'] = len(issues) == 0

    return result


def assess_mastering(audio_path, target_lufs=-8, sr=44100):
    """Mastering quality assessment with tech house targets."""
    data, sr = sf.read(audio_path)
    if data.ndim == 1:
        mono = data
        left = right = data
    else:
        left, right = data[:, 0], data[:, 1]
        mono = (left + right) / 2

    result = {
        'peak_dbfs': peak_dbfs(mono),
        'true_peak_ok': peak_dbfs(mono) <= -0.3,
    }

    if lufs_integrated:
        lufs = lufs_integrated(data, sr)
        result['integrated_lufs'] = lufs
        result['lufs_ok'] = abs(lufs - target_lufs) <= 1.5

    result['crest_factor_db'] = crest_factor_db(mono)
    result['crest_ok'] = 4 <= result['crest_factor_db'] <= 10

    sb = spectral_balance(mono, sr)
    result['spectral_balance'] = sb

    # Spectral balance checks
    issues = []
    if sb['sub_20_60']['percent'] > 25:
        issues.append('Too much sub — club systems will distort')
    if sb['sub_20_60']['percent'] < 10:
        issues.append('Thin low end — lacks club impact')
    if sb['low_mid_250_2k']['percent'] > 30:
        issues.append('Muddy low-mids — cut 200-400 Hz')
    if sb['presence_2k_6k']['percent'] > 25:
        issues.append('Harsh presence — cut 3-5 kHz')
    if sb['air_6k_20k']['percent'] < 5:
        issues.append('Dull highs — add air')

    if data.ndim == 2:
        corr = stereo_correlation(left, right)
        result['stereo_correlation'] = corr
        if corr < 0.2:
            issues.append('Too wide — may have phase issues on mono systems')
        if corr > 0.9:
            issues.append('Too narrow — no stereo interest')

    result['issues'] = issues
    result['passes'] = len(issues) == 0

    return result


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Tech House Audio Quality Assessment')
    parser.add_argument('audio', help='Audio file to assess')
    parser.add_argument('--stage', choices=['sound', 'mix', 'master'],
                        default='mix', help='Production stage to assess')
    parser.add_argument('--target-lufs', type=float, default=-8,
                        help='Target LUFS for mastering (default: -8)')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()

    if not sf:
        print("ERROR: soundfile not installed. Run: pip install soundfile")
        sys.exit(1)

    if args.stage == 'sound':
        result = assess_sound_selection(args.audio)
    elif args.stage == 'master':
        result = assess_mastering(args.audio, target_lufs=args.target_lufs)
    else:
        result = assess_mix(args.audio)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"TECH HOUSE QUALITY ASSESSMENT — {args.stage.upper()}")
        print(f"File: {args.audio}")
        print(f"{'='*60}")

        for key, value in result.items():
            if key == 'issues':
                continue
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    if isinstance(v, dict):
                        print(f"  {k}: {v}")
                    else:
                        print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")

        if 'issues' in result:
            print(f"\n{'='*60}")
            if result['issues']:
                print("ISSUES:")
                for issue in result['issues']:
                    print(f"  ⚠ {issue}")
            else:
                print("✓ No issues detected")

        if 'passes' in result:
            status = "✓ PASS" if result['passes'] else "✗ FAIL"
            print(f"\nOverall: {status}")


if __name__ == '__main__':
    main()
