import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, iirfilter, sosfilt
import pyloudnorm as pyln

data, sr = sf.read("output/TH_0613_1738_126_Dsmin/mastered/corrected_master.wav")
meter = pyln.Meter(sr)

# 1. HPF at 35 Hz (sub cleanup)
sos_hp = butter(4, 35, btype='high', fs=sr, output='sos')

# 2. Low shelf cut at 150 Hz (-6 dB) using a peaking filter
# Peaking EQ at 100 Hz, Q=0.7, -6 dB
from scipy.signal import iirpeak
# iirpeak doesn't support gain, so we use manual biquad
def peaking_eq(signal, freq, gain_db, q, sr):
    w0 = 2 * np.pi * freq / sr
    A = 10**(gain_db / 40)
    alpha = np.sin(w0) / (2 * q)
    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A
    b = np.array([b0/a0, b1/a0, b2/a0])
    a = np.array([1, a1/a0, a2/a0])
    from scipy.signal import lfilter
    return lfilter(b, a, signal)

# Apply corrections
if data.ndim == 2:
    corrected = np.column_stack([
        sosfiltfilt(sos_hp, data[:, 0]),
        sosfiltfilt(sos_hp, data[:, 1])
    ])
    # Low shelf cut at 100 Hz
    corrected[:, 0] = peaking_eq(corrected[:, 0], 100, -6, 0.7, sr)
    corrected[:, 1] = peaking_eq(corrected[:, 1], 100, -6, 0.7, sr)
    # Presence boost at 3 kHz
    corrected[:, 0] = peaking_eq(corrected[:, 0], 3000, +3, 1.0, sr)
    corrected[:, 1] = peaking_eq(corrected[:, 1], 3000, +3, 1.0, sr)
    # Air boost at 10 kHz
    corrected[:, 0] = peaking_eq(corrected[:, 0], 10000, +2, 0.7, sr)
    corrected[:, 1] = peaking_eq(corrected[:, 1], 10000, +2, 0.7, sr)
else:
    corrected = sosfiltfilt(sos_hp, data)
    corrected = peaking_eq(corrected, 100, -6, 0.7, sr)
    corrected = peaking_eq(corrected, 3000, +3, 1.0, sr)
    corrected = peaking_eq(corrected, 10000, +2, 0.7, sr)

# Match LUFS
lufs_before = meter.integrated_loudness(data)
lufs_after = meter.integrated_loudness(corrected)
corrected = corrected * 10**((lufs_before - lufs_after)/20)

sf.write("output/TH_0613_1738_126_Dsmin/mastered/TH_0613_1738_126_Dsmin_master_eq.wav",
         corrected, sr, subtype='PCM_24')

# Audit
mono_c = corrected.mean(axis=1) if corrected.ndim > 1 else corrected
fft = np.fft.rfft(mono_c)
mag = np.abs(fft)**2
freqs = np.fft.rfftfreq(len(mono_c), 1/sr)
bands = {'Sub': (20,60), 'Bass': (60,250), 'LoMid': (250,2000), 'Pres': (2000,6000), 'Air': (6000,20000)}
total = sum(np.sum(mag[(freqs >= lo) & (freqs < hi)]) for lo, hi in bands.values())
print("After corrective EQ:")
for name, (lo, hi) in bands.items():
    pct = np.sum(mag[(freqs >= lo) & (freqs < hi)]) / total * 100
    target = {'Sub': (15,20), 'Bass': (25,35), 'LoMid': (15,25), 'Pres': (10,20), 'Air': (5,15)}[name]
    status = '✓' if target[0] <= pct <= target[1] else '✗'
    print(f"  {name}: {pct:.0f}%  target {target[0]}-{target[1]}%  {status}")
print(f"  LUFS: {meter.integrated_loudness(corrected):.1f}")
