"""
Test every FFmpeg audio filter used in post_production.py.
Generates a 2-second stereo test tone then applies each filter.
Reports PASS / FAIL with the error message on failure.
"""
import subprocess
import tempfile
import os
import sys

FFMPEG = "ffmpeg"

def make_tone(path, duration=2):
    cmd = [
        FFMPEG, "-y",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
        "-ac", "2", "-ar", "48000", path
    ]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0

def test_af(label, filter_str, src):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        dst = f.name
    cmd = [FFMPEG, "-y", "-i", src, "-af", filter_str, dst]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    ok = r.returncode == 0 and os.path.getsize(dst) > 1000
    if os.path.exists(dst):
        os.remove(dst)
    if not ok:
        err = r.stderr.decode(errors="replace")
        # grab the last meaningful line
        lines = [l for l in err.splitlines() if l.strip()]
        hint = lines[-1][:120] if lines else "unknown error"
        return False, hint
    return True, ""

def test_fc(label, fc_str, src):
    """Test a filter_complex string (used for parallel reverb)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        dst = f.name
    cmd = [FFMPEG, "-y", "-i", src, "-filter_complex", fc_str, dst]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    ok = r.returncode == 0 and os.path.getsize(dst) > 1000
    if os.path.exists(dst):
        os.remove(dst)
    if not ok:
        err = r.stderr.decode(errors="replace")
        lines = [l for l in err.splitlines() if l.strip()]
        hint = lines[-1][:120] if lines else "unknown error"
        return False, hint
    return True, ""

# Representative BPM=90 values
BPM = 90
q_ms  = round(60000 / BPM)      # 667
e_ms  = round(q_ms / 2)         # 333
dq_ms = round(q_ms * 1.5)       # 1000

FILTERS = [
    # ── Drums ──
    ("drums: transient EQ",
     "equalizer=f=5000:width_type=o:width=2:gain=3"),
    ("drums: fast compressor",
     "acompressor=attack=2:release=40:ratio=2:threshold=0.2:makeup=1.2"),
    ("drums: gated reverb (aecho)",
     "aecho=0.8:0.4:30|60:0.35|0.18"),
    ("drums: noise gate",
     "agate=threshold=0.01:attack=5:release=80"),
    ("drums: bitcrush 8-bit",
     "acrusher=level_in=1:level_out=1:bits=8:mode=log:aa=1"),
    ("drums: bitcrush 6-bit",
     "acrusher=level_in=1:level_out=1:bits=6:mode=log:aa=1"),
    ("drums: flanger",
     "flanger=delay=5:depth=5:regen=20:width=90:speed=0.6"),
    ("drums: phaser",
     "aphaser=in_gain=0.4:out_gain=0.74:delay=3:decay=0.4:speed=0.8:type=t"),
    ("drums: chorus",
     "chorus=0.7:0.9:40|45:0.3|0.25:0.4|0.35:2|1.8"),
    ("drums: tremolo gate",
     f"tremolo=f={round(BPM/60.0,3)}:d=0.6"),
    ("drums: shimmer vibrato",
     "vibrato=f=4:d=0.02"),
    ("drums: apulsator auto-pan",
     "apulsator=mode=sine:hz=0.37:amount=0.08:offset_l=0.25:offset_r=0.75:width=1"),
    ("drums: acontrast punch",
     "acontrast=contrast=65"),
    ("drums: aemphasis vinyl deemph",
     "aemphasis=level_in=1:level_out=1:mode=reproduction:type=75fm"),

    # ── Bass ──
    ("bass: sub EQ boost",
     "equalizer=f=55:width_type=o:width=2:gain=4"),
    ("bass: highpass 20Hz",
     "highpass=f=20"),
    ("bass: chorus doubling",
     "chorus=0.7:0.9:50|55:0.3|0.25:0.5|0.4:2|1.6"),
    ("bass: harmonic exciter EQ",
     "equalizer=f=800:width_type=o:width=2:gain=2"),
    ("bass: virtualbass",
     "virtualbass=cutoff=250:strength=1.5"),
    ("bass: compand vintage",
     "compand=attacks=0.1:decays=0.5:points=-80/-80|-40/-35|-25/-15|-10/-6|0/0"),

    # ── Pads ──
    ("pads: phaser",
     "aphaser=in_gain=0.4:out_gain=0.74:delay=3:decay=0.4:speed=0.4:type=t"),
    ("pads: tremolo",
     f"tremolo=f={round(BPM/60.0,3)}:d=0.35"),
    ("pads: vinyl lowpass + vibrato",
     "lowpass=f=11000,vibrato=f=0.5:d=0.02"),
    ("pads: ensemble chorus",
     "chorus=0.6:0.85:45|50|55:0.35|0.3|0.25:0.3|0.4|0.35:2|1.8|1.6"),
    ("pads: adynamicequalizer mud cut",
     "adynamicequalizer=threshold=20:dfrequency=300:dqfactor=2:tfrequency=300:tqfactor=2:range=6:mode=cutabove:tftype=bell"),
    ("pads: stereotools MS widen",
     "stereotools=slev=1.4:mlev=1:mode=3"),
    ("pads: aemphasis tape deemph",
     "aemphasis=level_in=1:level_out=1:mode=reproduction:type=75fm"),

    # ── Counter melody ──
    ("counter: dotted-quarter delay",
     f"aecho=1.0:0.5:{dq_ms}:0.4"),
    ("counter: shimmer chorus",
     "chorus=0.8:0.9:25|50:0.5|0.4:0.8|0.6:2|1.6"),
    ("counter: flanger",
     "flanger=delay=5:depth=5:regen=30:width=90:speed=0.4"),
    ("counter: aexciter",
     "aexciter=freq=5500:amount=2.5:blend=5"),
    ("counter: aphaseshift",
     "aphaseshift=shift=0.35:level=1"),

    # ── Melodies ──
    ("melody: vinyl lowpass + vibrato",
     "lowpass=f=10000,vibrato=f=0.5:d=0.02"),
    ("melody: BPM delay (quarter)",
     f"aecho=0.9:0.3:{q_ms}:0.25"),
    ("melody: BPM delay (eighth)",
     f"aecho=0.9:0.3:{e_ms}:0.25"),
    ("melody: chorus",
     "chorus=0.7:0.9:45|55:0.4|0.3:0.35|0.4:2|1.6"),
    ("melody: Haas widening",
     "adelay=0|25"),
    ("melody: aexciter air",
     "aexciter=freq=7000:amount=2.0:blend=4"),
    ("melody: atilt bright",
     "atilt=freq=1000:slope=0.5:width=1000:order=5"),
    ("melody: atilt dark",
     "atilt=freq=1000:slope=-0.5:width=1000:order=5"),
    ("melody: adynamicequalizer harsh cut",
     "adynamicequalizer=threshold=18:dfrequency=3000:dqfactor=2:tfrequency=3000:tqfactor=2:range=4:mode=cutabove:tftype=bell"),

    # ── FX stems ──
    ("fx: telephone bandpass",
     "highpass=f=300,lowpass=f=3400"),
    ("fx: heavy bitcrush 4-bit",
     "acrusher=level_in=1:level_out=1:bits=4:mode=log:aa=1"),
    ("fx: rhythmic tremolo gate",
     f"tremolo=f={round(BPM/60.0,3)}:d=0.8"),
    ("fx: ring-mod chorus",
     "chorus=0.6:0.9:7|8:0.7|0.6:0.9|0.8:2|1.8"),
    ("fx: acontrast",
     "acontrast=contrast=60"),
    ("fx: stereowiden",
     "stereowiden"),

    # ── Spatial FX ──
    ("spatial: drum room echo",
     "aecho=0.8:0.3:25|55:0.15|0.07"),
    ("spatial: pad hall echo",
     "aecho=0.8:0.55:20|60|120|200:0.5|0.35|0.2|0.08"),
    ("spatial: melody reverb+delay",
     f"aecho=0.8:0.5:15|45|85:0.4|0.25|0.12,aecho=1.0:0.25:{q_ms}:0.15"),

    # ── Harmonic enhancement (phase 3) ──
    ("phase3: drum bitcrush 10-bit",
     "acrusher=level_in=1:level_out=1:bits=10:mode=log:aa=1"),
    ("phase3: bass tanh saturation",
     "aeval='tanh(val(0)*2.5)/tanh(2.5)|tanh(val(1)*2.5)/tanh(2.5)'"),
    ("phase3: pad tape saturation",
     "aeval='tanh(val(0)*1.8)/tanh(1.8)|tanh(val(1)*1.8)/tanh(1.8)'"),
    ("phase3: clarity EQ",
     "equalizer=f=350:width_type=o:width=1:gain=-3,equalizer=f=3200:width_type=o:width=1.5:gain=-2.5"),
    ("phase3: kick notch in bass",
     "equalizer=f=47:width_type=o:width=2:gain=-4"),

    # ── Dynamic compression (phase 2) ──
    ("compress: drums",
     "acompressor=threshold=0.1:ratio=3:attack=30:release=80:makeup=1.413:knee=2"),
    ("compress: bass half-note release",
     f"acompressor=threshold=0.126:ratio=3:attack=20:release={q_ms*2}:makeup=1.259:knee=3"),
    ("compress: melody quarter-note release",
     f"acompressor=threshold=0.126:ratio=3:attack=5:release={q_ms}:makeup=1.259:knee=2"),

    # ── Master bus ──
    ("master: alimiter",
     "alimiter=limit=0.9:level=1:attack=5:release=50:level_in=1"),
    ("master: loudnorm",
     "loudnorm=I=-14:TP=-1.0:LRA=7"),
]

# Parallel reverb filter_complex tests
FC_TESTS = [
    ("reverb FC: drum ambience",
     "[0:a]asplit=2[dry][fxin];[fxin]aecho=0:0.9:4|8|14:0.45|0.28|0.12[wet];[dry][wet]amix=inputs=2:weights=0.75 0.25"),
    ("reverb FC: drum room",
     "[0:a]asplit=2[dry][fxin];[fxin]aecho=0:0.9:10|18|30:0.45|0.28|0.14[wet];[dry][wet]amix=inputs=2:weights=0.75 0.25"),
    ("reverb FC: drum plate",
     "[0:a]asplit=2[dry][fxin];[fxin]aecho=0:0.9:5|9|15|22|32:0.38|0.3|0.22|0.15|0.08[wet];[dry][wet]amix=inputs=2:weights=0.75 0.25"),
    ("reverb FC: drum spring",
     "[0:a]asplit=2[dry][fxin];[fxin]aecho=0:0.9:8|16|28:0.42|0.28|0.14[wet];[dry][wet]amix=inputs=2:weights=0.75 0.25"),
    ("reverb FC: melody small_hall",
     "[0:a]asplit=2[dry][fxin];[fxin]aecho=0:0.9:15|30|55|85:0.45|0.32|0.2|0.1[wet];[dry][wet]amix=inputs=2:weights=0.7 0.3"),
    ("reverb FC: melody chamber",
     "[0:a]asplit=2[dry][fxin];[fxin]aecho=0:0.9:12|25|45|70:0.43|0.3|0.19|0.09[wet];[dry][wet]amix=inputs=2:weights=0.7 0.3"),
    ("reverb FC: melody hall",
     "[0:a]asplit=2[dry][fxin];[fxin]aecho=0:0.9:20|45|80|130:0.45|0.33|0.22|0.11[wet];[dry][wet]amix=inputs=2:weights=0.7 0.3"),
    ("reverb FC: melody cathedral",
     "[0:a]asplit=2[dry][fxin];[fxin]aecho=0:0.9:30|80|160|280:0.43|0.32|0.2|0.1[wet];[dry][wet]amix=inputs=2:weights=0.65 0.35"),
]

def main():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tone = f.name

    if not make_tone(tone):
        print("ERROR: could not generate test tone — is ffmpeg installed?")
        sys.exit(1)

    print(f"Test tone: {tone}\n")
    print(f"{'Filter':<50}  {'Result'}")
    print("-" * 70)

    fails = []

    for label, fstr in FILTERS:
        ok, err = test_af(label, fstr, tone)
        status = "PASS" if ok else f"FAIL  ← {err}"
        print(f"  {label:<50}  {status}")
        if not ok:
            fails.append((label, err))

    print()
    print("── filter_complex (parallel reverb) ──")
    for label, fcstr in FC_TESTS:
        ok, err = test_fc(label, fcstr, tone)
        status = "PASS" if ok else f"FAIL  ← {err}"
        print(f"  {label:<50}  {status}")
        if not ok:
            fails.append((label, err))

    os.remove(tone)

    print()
    if fails:
        print(f"FAILURES ({len(fails)}):")
        for label, err in fails:
            print(f"  ✗ {label}")
            print(f"    {err}")
        sys.exit(1)
    else:
        print(f"All {len(FILTERS) + len(FC_TESTS)} filters passed.")

if __name__ == "__main__":
    main()
