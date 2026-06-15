import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import os

REF_DIR = "/Users/raphael/Coding/techhouse/reference"
reference = {
    'Sub 20-60': (15, 20), 'Bass 60-250': (25, 35),
    'Low-Mid 250-2k': (15, 25), 'Presence 2k-6k': (10, 20), 'Air 6k-20k': (5, 15),
}
bands = {
    'Sub 20-60': (20, 60), 'Bass 60-250': (60, 250),
    'Low-Mid 250-2k': (250, 2000), 'Presence 2k-6k': (2000, 6000), 'Air 6k-20k': (6000, 20000),
}

print('=' * 70)
print('REFERENCE TRACK ANALYSIS')
print('=' * 70)

results = []
for fname in sorted(os.listdir(REF_DIR)):
    if not fname.endswith('.mp3'):
        continue
    path = os.path.join(REF_DIR, fname)
    try:
        data, sr = sf.read(path)
        meter = pyln.Meter(sr)
        mono = data.mean(axis=1) if data.ndim > 1 else data
        lufs = meter.integrated_loudness(data)
        lra = meter.loudness_range(data)
        peak_db = 20 * np.log10(np.max(np.abs(data)) + 1e-10)
        rms = np.sqrt(np.mean(mono ** 2))
        crest = peak_db - 20 * np.log10(rms + 1e-10)
        corr = np.corrcoef(data[:, 0], data[:, 1])[0, 1] if data.ndim == 2 else 1.0
        fft = np.fft.rfft(mono); mag = np.abs(fft) ** 2; freqs = np.fft.rfftfreq(len(mono), 1 / sr)
        total = sum(np.sum(mag[(freqs >= lo) & (freqs < hi)]) for lo, hi in bands.values())
        duration = len(mono) / sr
        spectral = {}
        for name, (lo, hi) in bands.items():
            spectral[name] = float(np.sum(mag[(freqs >= lo) & (freqs < hi)]) / total * 100)
        
        print(f'\n--- {fname[:60]} ---')
        print(f'  Duration: {duration:.0f}s ({duration/60:.1f} min)')
        print(f'  LUFS: {lufs:.1f}  LRA: {lra:.1f}  Peak: {peak_db:.1f} dBFS')
        print(f'  Crest: {crest:.1f} dB  Stereo: {corr:.3f}')
        for name, (lo, hi) in bands.items():
            pct = spectral[name]
            tlo, thi = reference[name]
            status = '✓' if tlo <= pct <= thi else '✗'
            print(f'    {name:20s}: {pct:5.1f}%  {status}')
        
        results.append({
            'name': fname, 'lufs': lufs, 'lra': lra, 'crest': crest,
            'stereo': corr, 'spectral': spectral
        })
    except Exception as e:
        print(f'  ERROR: {e}')

# Averages
if results:
    print(f'\n{"="*70}')
    print(f'REFERENCE AVERAGES (n={len(results)})')
    print(f'{"="*70}')
    avg_lufs = np.mean([r['lufs'] for r in results])
    avg_crest = np.mean([r['crest'] for r in results])
    avg_stereo = np.mean([r['stereo'] for r in results])
    print(f'  LUFS: {avg_lufs:.1f}  Crest: {avg_crest:.1f} dB  Stereo: {avg_stereo:.3f}')
    for name in bands:
        avg_pct = np.mean([r['spectral'][name] for r in results])
        tlo, thi = reference[name]
        status = '✓' if tlo <= avg_pct <= thi else '✗'
        print(f'  {name:20s}: {avg_pct:5.1f}%  target {tlo}-{thi}%  {status}')

# Our track
print(f'\n{"="*70}')
print(f'OUR TRACK vs REFERENCE AVERAGE')
print(f'{"="*70}')
data, sr = sf.read('output/TH_0614_1130_128_Dsmin/mastered/TH_0614_1130_128_Dsmin_professional.wav')
meter = pyln.Meter(sr); mono = data.mean(axis=1) if data.ndim > 1 else data
lufs = meter.integrated_loudness(data)
peak_db = 20 * np.log10(np.max(np.abs(data)) + 1e-10)
rms = np.sqrt(np.mean(mono ** 2))
crest = peak_db - 20 * np.log10(rms + 1e-10)
corr = np.corrcoef(data[:, 0], data[:, 1])[0, 1] if data.ndim == 2 else 1.0
fft = np.fft.rfft(mono); mag = np.abs(fft) ** 2; freqs = np.fft.rfftfreq(len(mono), 1 / sr)
total = sum(np.sum(mag[(freqs >= lo) & (freqs < hi)]) for lo, hi in bands.values())

print(f'  Metric           : Ours     Ref Avg   Gap')
print(f'  LUFS             : {lufs:6.1f}   {avg_lufs:6.1f}   {lufs - avg_lufs:+.1f}')
print(f'  Crest Factor     : {crest:6.1f}   {avg_crest:6.1f}   {crest - avg_crest:+.1f}')
print(f'  Stereo           : {corr:6.3f}   {avg_stereo:6.3f}   {corr - avg_stereo:+.3f}')
for name, (lo, hi) in bands.items():
    our_pct = float(np.sum(mag[(freqs >= lo) & (freqs < hi)]) / total * 100)
    ref_pct = np.mean([r['spectral'][name] for r in results])
    gap = our_pct - ref_pct
    status = '✓' if abs(gap) < 5 else '⚠'
    print(f'  {name:20s}: {our_pct:5.1f}%  {ref_pct:5.1f}%  {gap:+.1f}%  {status}')
