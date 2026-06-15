import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt
import pyloudnorm as pyln

data, sr = sf.read("output/TH_0613_1738_126_Dsmin/mastered/corrected_master.wav")
meter = pyln.Meter(sr)

# HPF at 40 Hz
sos_hp = butter(4, 40, btype='high', fs=sr, output='sos')
if data.ndim == 2:
    corrected = np.column_stack([
        sosfiltfilt(sos_hp, data[:, 0]),
        sosfiltfilt(sos_hp, data[:, 1])
    ])
else:
    corrected = sosfiltfilt(sos_hp, data)

# Match LUFS
lufs_before = meter.integrated_loudness(data)
lufs_after = meter.integrated_loudness(corrected)
corrected = corrected * 10**((lufs_before - lufs_after)/20)

sf.write("output/TH_0613_1738_126_Dsmin/mastered/corrected_master_eq.wav",
         corrected, sr, subtype='PCM_24')

mono_c = corrected.mean(axis=1) if corrected.ndim > 1 else corrected
fft = np.fft.rfft(mono_c)
mag = np.abs(fft)**2
freqs = np.fft.rfftfreq(len(mono_c), 1/sr)
bands = {'Sub': (20,60), 'Bass': (60,250), 'LoMid': (250,2000), 'Pres': (2000,6000), 'Air': (6000,20000)}
total = sum(np.sum(mag[(freqs >= lo) & (freqs < hi)]) for lo, hi in bands.values())
print("After corrective EQ (HPF 40 Hz):")
for name, (lo, hi) in bands.items():
    pct = np.sum(mag[(freqs >= lo) & (freqs < hi)]) / total * 100
    print(f"  {name}: {pct:.0f}%")
print(f"  LUFS: {meter.integrated_loudness(corrected):.1f}")
