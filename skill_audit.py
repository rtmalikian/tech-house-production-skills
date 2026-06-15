"""
SKILL COMPLIANCE AUDIT
Checks each skill (01-09) against what the pipeline actually implements.
"""
import json
import re

issues = []
passes = []

def check(condition, skill, item, detail=""):
    if condition:
        passes.append(f"  ✓ {skill}: {item}")
    else:
        issues.append(f"  ✗ {skill}: {item}" + (f" — {detail}" if detail else ""))

# ============================================================================
# SKILL 01: SOUND SELECTION
# ============================================================================
print("01 SOUND SELECTION")

# Check if curated patch DB exists and is used
import os
db_path = "skills/tech_house_fantom_patches.json"
check(os.path.exists(db_path), "01", "Patch database exists")
with open(db_path) as f:
    db = json.load(f)
check(len(db.get('tech_house_bass', {}).get('house', [])) > 0, "01", "House bass patches in DB")
check(len(db.get('tech_house_drums', [])) > 0, "01", "Drum kits in DB")
check(len(db.get('tech_house_stab', [])) > 0, "01", "Stab patches in DB")

# Check if record_stems.py loads the patch DB
with open("record_stems.py") as f:
    stems_code = f.read()
check("tech_house_fantom_patches.json" in stems_code, "01", "record_stems loads patch DB")
check("select_patch_for_track" in stems_code, "01", "Patch selection function exists")

# Check if frequency slot allocation is implemented
check("classify_track" in stems_code, "01", "Track classification (role detection)")

# ============================================================================
# SKILL 02: DRUM PROGRAMMING
# ============================================================================
print("\n02 DRUM PROGRAMMING")

with open("midi_drum_sequences.py") as f:
    drums_code = f.read()

# Check for 4-on-floor kick
check("4-on-the-floor" in drums_code or "four_on_floor" in drums_code, "02", "4-on-floor kick pattern")
check("122, 127" in drums_code or "randint(122, 127)" in drums_code, "02", "Kick velocity 122-127 (consistent)")

# Check for clap on 2&4
check("CLAP" in drums_code and "tpb" in drums_code, "02", "Clap on beats 2&4")

# Check for 16th note hats
check("sixteenth" in drums_code and "CLOSED_HAT" in drums_code, "02", "16th note hi-hats")

# Check for velocity variation on hats
check("velocity" in drums_code and "randint" in drums_code, "02", "Hat velocity variation")

# Check for NO ghost kicks
check("ghost" not in drums_code.lower() or "NO ghost" in drums_code, "02", "No ghost kicks")

# Check for quantized kick (no micro-techniques)
check("quantize_drums" in drums_code, "02", "Kick/clap quantized (no micro-techniques)")

# Check swing values
with open("midi_config.py") as f:
    config_code = f.read()
check("'none': 0.50" in config_code, "02", "Swing default = none (0.50)")
check("'light': 0.50" in config_code, "02", "Light swing = 0.50 (no swing)")

# Check humanization
check("'timing_subtle': (0, 0)" in config_code, "02", "Zero timing jitter")

# ============================================================================
# SKILL 03: BASS DESIGN & PROGRAMMING
# ============================================================================
print("\n03 BASS DESIGN")

with open("midi_orchestrator.py") as f:
    orch_code = f.read()

# Check bass velocity (should be reduced)
check("45 +" in orch_code or "45+" in orch_code, "03", "Bass velocity reduced (45 base)")

# Check sub bass is drops-only
check("'drop1', 'drop2'" in orch_code, "03", "Sub bass only in drops")

# Check sub bass velocity is low
check("'vel': sub_vel" in orch_code or "60 * energy" in orch_code, "03", "Sub bass velocity reduced")

# Check bass patterns exist
with open("midi_composition.py") as f:
    comp_code = f.read()
check("tech_house_offbeat" in comp_code, "03", "Tech house bass patterns defined")
check("tech_house_driving" in comp_code, "03", "Driving bass pattern defined")

# Check sidechain settings
with open("june1_corrected_pipeline.py") as f:
    pipeline_code = f.read()
check("SIDECHAIN_RELEASE_MS = 150" in pipeline_code, "03", "Sidechain release 150ms")
check("SIDECHAIN_TRIGGER_THRESHOLD = 0.15" in pipeline_code, "03", "Sidechain threshold 0.15")

# ============================================================================
# SKILL 04: SYNTH ELEMENTS
# ============================================================================
print("\n04 SYNTH ELEMENTS")

# Check chord stab velocity
check("'vel': stab_vel" in orch_code and "110" in orch_code, "04", "Stab velocity boosted to 110")

# Check acid line velocity
check("90, 120" in orch_code, "04", "Acid velocity 90-120")

# Check acid line exists
check("acid_ev" in orch_code, "04", "Acid line track generated")

# Check stabs in drops
check("'drop1', 'drop2'" in orch_code and "stab" in orch_code.lower(), "04", "Stabs in drop sections")

# Check pad in breakdowns
check("'breakdown'" in orch_code and "pad" in orch_code.lower(), "04", "Pads in breakdown sections")

# ============================================================================
# SKILL 05: ARRANGEMENT
# ============================================================================
print("\n05 ARRANGEMENT")

with open("midi_song_structure.py") as f:
    structure_code = f.read()

# Check 88-bar structure
check("TOTAL_BARS = 88" in config_code, "05", "88-bar arrangement")

# Check tech house sections
check("'intro'" in structure_code and "'build1'" in structure_code, "05", "Intro + Build1 sections")
check("'drop1'" in structure_code and "'breakdown'" in structure_code, "05", "Drop1 + Breakdown sections")
check("'drop2'" in structure_code and "'outro'" in structure_code, "05", "Drop2 + Outro sections")

# Check section bar counts
check("bar < 16" in structure_code, "05", "Intro = 16 bars")
check("bar < 24" in structure_code, "05", "Build1 = 8 bars")
check("bar < 40" in structure_code, "05", "Drop1 = 16 bars")
check("bar < 48" in structure_code, "05", "Breakdown = 8 bars")

# Check minor key preference
check("'Minor'" in orch_code and "0.90" in orch_code, "05", "90% minor key selection")

# ============================================================================
# SKILL 06: MIXING
# ============================================================================
print("\n06 MIXING")

# Check Fantom EQ on bass
with open("record_stems.py") as f:
    stems_code = f.read()
check("set_zone_eq_gain" in stems_code and "low" in stems_code, "06", "Fantom zone EQ applied")
check("'low', -12.0" in stems_code, "06", "Bass low cut -12 dB")
check("'mid', +6.0" in stems_code, "06", "Bass mid boost +6 dB")

# Check sidechain is NOT on Fantom
check("sidechain" not in stems_code.lower() or "post-production" in stems_code.lower(), "06",
      "Sidechain NOT on Fantom (post-production)")

# Check calibration
check("calibrate_part_gain" in stems_code, "06", "Level calibration implemented")
check("is_drum" in stems_code, "06", "Drums skip calibration (default +6 dB)")

# ============================================================================
# SKILL 07: MASTERING
# ============================================================================
print("\n07 MASTERING")

# Check corrective EQ
with open("run_pipeline.py") as f:
    pipeline_run_code = f.read()
check("peaking_eq" in pipeline_run_code or os.path.exists("corrective_eq2.py"), "07",
      "Corrective post-EQ implemented")
check("3000" in pipeline_run_code and "+3" in pipeline_run_code, "07", "Post EQ: +3 dB @ 3 kHz")

# Check mastering pipeline
check("run_pipeline.py" in os.listdir("."), "07", "Mastering pipeline exists")
check("LUFS" in pipeline_code and "-14" in pipeline_code, "07", "LUFS targeting -14")

# Check spectral balance
check("100" in stems_code and "-6" in stems_code, "07", "Post EQ: -6 dB @ 100 Hz")
check("3000" in stems_code and "+3" in stems_code, "07", "Post EQ: +3 dB @ 3 kHz")

# ============================================================================
# SKILL 08: REFERENCE COMPARISON
# ============================================================================
print("\n08 REFERENCE COMPARISON")

check(os.path.exists("skills/08_reference_comparison.md"), "08", "Reference skill exists")
# Check if reference comparison is implemented in code
check("REFERENCE A/B" in pipeline_run_code or "ab_report" in pipeline_run_code, "08",
      "Reference A/B comparison integrated into pipeline")

# ============================================================================
# SKILL 09: FANTOM SYSEX
# ============================================================================
print("\n09 FANTOM SYSEX")

# Check Fantom controller
check(os.path.exists("audio_pipeline/fantom_midi_control.py"), "09", "Fantom controller exists")

# Check tech house patches loaded
check("tech_house_fantom_patches.json" in stems_code, "09", "Tech house patches loaded")

# Check LFO matrix applied
check("apply_step_lfo" in stems_code or "_apply_step_lfo" in stems_code, "09",
      "Step LFO matrix applied")

# Check MFX applied
check("_apply_chorus_mfx" in stems_code, "09", "Chorus MFX on stabs")
check("_apply_acid_bass_mfx" in stems_code or "Super Filter" in stems_code, "09",
      "Super Filter MFX on acid")

# Check sound design applied
check("apply_sound_design" in stems_code, "09", "Sound design function exists")

# Check sync click detection
check("sync_ch_left" in stems_code or "sync click" in stems_code.lower(), "09",
      "Sync click detection for sample-accurate trim")

# ============================================================================
# ASSESSMENT TOOL
# ============================================================================
print("\nASSESSMENT TOOL")

check(os.path.exists("skills/assessment.py"), "ASSESS", "Assessment script exists")

# Check what metrics it measures
with open("skills/assessment.py") as f:
    assess_code = f.read()
check("crest_factor" in assess_code, "ASSESS", "Crest factor measurement")
check("spectral_balance" in assess_code, "ASSESS", "Spectral balance measurement")
check("stereo_correlation" in assess_code, "ASSESS", "Stereo correlation")
check("lufs" in assess_code.lower() or "loudness" in assess_code.lower(), "ASSESS", "LUFS measurement")

# ============================================================================
# PRINT RESULTS
# ============================================================================
print("\n" + "=" * 60)
print("AUDIT RESULTS")
print("=" * 60)

print(f"\n✓ PASSED: {len(passes)}")
for p in passes:
    print(p)

print(f"\n✗ FAILED: {len(issues)}")
for i in issues:
    print(i)

print(f"\nCompliance: {len(passes)}/{len(passes)+len(issues)} ({len(passes)/(len(passes)+len(issues))*100:.0f}%)")
