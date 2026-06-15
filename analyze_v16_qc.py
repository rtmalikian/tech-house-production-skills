import numpy as np
import soundfile as sf
import os

SONG = "TH_0614_1659_124_Bmin"
data, sr = sf.read(f"output/{SONG}/mastered_qc/{SONG}_qc_master_eq.wav")
mono = data.mean(axis=1) if data.ndim > 1 else data
duration = len(mono) / sr

print("=" * 60)
print(f"V16 QC ANALYSIS — {SONG}")
print(f"Duration: {duration:.1f}s")
print("=" * 60)

# 1. First 10 seconds
print("\n1. FIRST 10 SECONDS (checking for silence):")
for sec in range(10):
    start = int(sec * sr)
    end = int((sec + 1) * sr)
    if end > len(mono):
        break
    chunk = mono[start:end]
    rms = 20 * np.log10(np.sqrt(np.mean(chunk**2)) + 1e-10)
    peak = 20 * np.log10(np.max(np.abs(chunk)) + 1e-10)
    bar = '█' * max(0, int((rms + 60) / 3))
    status = 'SILENT' if rms < -50 else 'quiet' if rms < -30 else 'OK'
    print(f"   {sec}s-{sec+1}s: rms={rms:6.1f} peak={peak:6.1f}  {bar}  [{status}]")

# 2. Seconds 25-40
print("\n2. SECONDS 25-40 (checking for loud element at 31s):")
for t in np.arange(25, 40, 0.5):
    start = int(t * sr)
    end = int((t + 0.5) * sr)
    if end > len(mono):
        break
    chunk = mono[start:end]
    rms = 20 * np.log10(np.sqrt(np.mean(chunk**2)) + 1e-10)
    peak = 20 * np.log10(np.max(np.abs(chunk)) + 1e-10)
    bar = '█' * max(0, int((rms + 60) / 2))
    marker = ' ← LOUD!' if rms > -10 else ''
    print(f"   {t:5.1f}s: rms={rms:6.1f} peak={peak:6.1f}  {bar}{marker}")

# 3. Stem levels at 31 seconds
print("\n3. STEM LEVELS AT 30-32s (identifying the loud element):")
stems_dir = f"output/{SONG}/recordings_balanced"
for fname in sorted(os.listdir(stems_dir)):
    if not fname.endswith('.wav') or fname.startswith('pass'):
        continue
    path = os.path.join(stems_dir, fname)
    try:
        sdata, ssr = sf.read(path)
        smono = sdata.mean(axis=1) if sdata.ndim > 1 else sdata
        start = int(30 * ssr)
        end = int(32 * ssr)
        if end > len(smono):
            continue
        chunk = smono[start:end]
        rms = 20 * np.log10(np.sqrt(np.mean(chunk**2)) + 1e-10)
        peak = 20 * np.log10(np.max(np.abs(chunk)) + 1e-10)
        marker = ' ← LOUD!' if rms > -12 else ''
        print(f"   {fname[:35]:35s} rms={rms:6.1f} peak={peak:6.1f}{marker}")
    except:
        pass

# 4. Section at 31s
bpm = 124
bar_duration = 4 * 60.0 / bpm
bar_at_31 = 31 / bar_duration
print(f"\n4. SECTION AT 31s: bar {bar_at_31:.1f}")
if bar_at_31 < 16:
    print(f"   Section: INTRO (bars 0-15)")
elif bar_at_31 < 48:
    print(f"   Section: DROP 1 (bars 16-47)")
elif bar_at_31 < 80:
    print(f"   Section: BREAKDOWN (bars 48-79)")
else:
    print(f"   Section: DROP 2 (bars 80-111)")

# 5. Check what's on the original (pre-QC) stems at 31s
print("\n5. ORIGINAL (PRE-QC) STEM LEVELS AT 30-32s:")
orig_dir = f"output/{SONG}/recordings"
for fname in sorted(os.listdir(orig_dir)):
    if not fname.endswith('.wav') or fname.startswith('pass'):
        continue
    path = os.path.join(orig_dir, fname)
    try:
        sdata, ssr = sf.read(path)
        smono = sdata.mean(axis=1) if sdata.ndim > 1 else sdata
        start = int(30 * ssr)
        end = int(32 * ssr)
        if end > len(smono):
            continue
        chunk = smono[start:end]
        rms = 20 * np.log10(np.sqrt(np.mean(chunk**2)) + 1e-10)
        peak = 20 * np.log10(np.max(np.abs(chunk)) + 1e-10)
        marker = ' ← LOUD!' if rms > -12 else ''
        print(f"   {fname[:35]:35s} rms={rms:6.1f} peak={peak:6.1f}{marker}")
    except:
        pass
