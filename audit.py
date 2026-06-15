import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import os
from scipy.signal import find_peaks

data, sr = sf.read("output/TH_0613_1618_127_Csmin/mastered/corrected_master.wav")
meter = pyln.Meter(sr)
mono = data.mean(axis=1) if data.ndim > 1 else data

print("=" * 70)
print("QUALITY CONTROL AUDIT — TH_0613_1618_127_Csmin")
print("=" * 70)

# 1. Loudness
lufs = meter.integrated_loudness(data)
lra = meter.loudness_range(data)
peak_db = 20 * np.log10(np.max(np.abs(data)) + 1e-10)
rms = np.sqrt(np.mean(mono**2))
crest = peak_db - 20 * np.log10(rms + 1e-10)

print(f"\n1. LOUDNESS")
print(f"   LUFS:         {lufs:.1f}  (club: -8 to -6, streaming: -14)")
print(f"   LRA:          {lra:.1f} LU  (target: 4-8)")
print(f"   Peak:         {peak_db:.1f} dBFS")
print(f"   Crest Factor: {crest:.1f} dB  (target: 4-8 dB)")

# 2. Spectral Balance
fft = np.fft.rfft(mono)
mag = np.abs(fft)**2
freqs = np.fft.rfftfreq(len(mono), 1/sr)
bands = {
    'Sub 20-60': (20, 60),
    'Bass 60-250': (60, 250),
    'Low-Mid 250-2k': (250, 2000),
    'Presence 2k-6k': (2000, 6000),
    'Air 6k-20k': (6000, 20000),
}
total = sum(np.sum(mag[(freqs >= lo) & (freqs < hi)]) for lo, hi in bands.values())
print(f"\n2. SPECTRAL BALANCE")
issues = []
for name, (lo, hi) in bands.items():
    pct = np.sum(mag[(freqs >= lo) & (freqs < hi)]) / total * 100
    target = {
        'Sub 20-60': (15, 20), 'Bass 60-250': (25, 35),
        'Low-Mid 250-2k': (15, 25), 'Presence 2k-6k': (10, 20),
        'Air 6k-20k': (5, 15)
    }[name]
    status = '✓' if target[0] <= pct <= target[1] else '✗'
    if status == '✗':
        if pct > target[1]:
            issues.append(f"Too much {name}: {pct:.0f}% (target {target[0]}-{target[1]}%)")
        else:
            issues.append(f"Too little {name}: {pct:.0f}% (target {target[0]}-{target[1]}%)")
    print(f"   {name}: {pct:5.1f}%  target {target[0]}-{target[1]}%  {status}")

# 3. Stereo
if data.ndim == 2:
    L, R = data[:, 0], data[:, 1]
    corr = np.corrcoef(L, R)[0, 1]
    mid = (L + R) / 2
    side = (L - R) / 2
    smr = np.sqrt(np.mean(side**2)) / (np.sqrt(np.mean(mid**2)) + 1e-10)
    print(f"\n3. STEREO")
    print(f"   Correlation: {corr:.3f}  (target: 0.3-0.7)")
    print(f"   Side/Mid:    {smr:.3f}  (target: 0.1-0.4)")
    if corr > 0.8:
        issues.append("Too narrow — lacks stereo interest")
    if corr < 0.2:
        issues.append("Too wide — phase issues on mono systems")

# 4. Kick detection
print(f"\n4. KICK/BPM")
env_smooth = np.convolve(np.abs(mono), np.ones(int(sr*0.01))/int(sr*0.01), mode='same')
peaks, _ = find_peaks(env_smooth, height=np.max(env_smooth)*0.3, distance=int(sr*0.3))
if len(peaks) > 1:
    intervals = np.diff(peaks) / sr
    detected_bpm = 60 / np.mean(intervals)
    print(f"   Transients: {len(peaks)}")
    print(f"   Implied BPM: {detected_bpm:.0f}  (expected: 127)")

# 5. Duration
duration = len(mono) / sr
bars = duration / (4 * 60 / 127)
print(f"\n5. ARRANGEMENT")
print(f"   Duration: {duration:.1f}s ({duration/60:.1f} min)")
print(f"   Bars: ~{bars:.0f}  (expected: 88)")

# 6. Per-stem levels
print(f"\n6. STEM LEVELS")
stems_dir = "output/TH_0613_1618_127_Csmin/recordings"
stem_files = sorted([f for f in os.listdir(stems_dir) if f.endswith('.wav') and not f.startswith('pass')])
for f in stem_files:
    path = os.path.join(stems_dir, f)
    try:
        sdata, _ = sf.read(path)
        smono = sdata.mean(axis=1) if sdata.ndim > 1 else sdata
        speak = 20 * np.log10(np.max(np.abs(smono)) + 1e-10)
        srms = 20 * np.log10(np.sqrt(np.mean(smono**2)) + 1e-10)
        print(f"   {f:35s} peak={speak:6.1f} rms={srms:6.1f}")
    except:
        pass

# 7. Issues summary
print(f"\n{'='*70}")
print("ISSUES FOUND:")
if issues:
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
else:
    print("   None")
print(f"{'='*70}")
