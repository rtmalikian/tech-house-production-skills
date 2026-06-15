#!/usr/bin/env python3
"""
Record short, reusable sample stems from generated MIDI.

Sample sets:
  - verse 8 bars: drums (bus + stems + one-shots), melody/bass/pad (stems, 10s tail)
  - chorus 8 bars: drums (bus + stems + one-shots), chorus melody/bass/pad (stems, 10s tail)

Melodic stems use a long tail (10s) to capture reverb/delay tails and long releases.
Each clip includes a 4-beat count-in for alignment.

Usage:
    music_venv/bin/python scripts/audio_pipeline/generate_sample_stems.py path/to/song.mid
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import mido
import numpy as np
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.append(str(SCRIPT_DIR))
sys.path.append(str(PROJECT_ROOT))

from audio_recorder import AudioRecorder, MultiPassOrchestrator, _COUNT_IN_BEATS, _SYNC_CH, _SYNC_NOTE, _SYNC_SOUND  # noqa: E402
from fantom_midi_control import FantomController  # noqa: E402


def load_recording_patch_map(recordings_dir: str) -> Dict[str, Dict]:
    """Load recording manifests and build a track-name → {patch, sound_design_raw} map.

    The first manifest that contains a given track name wins. This allows the
    sample pack generator to replay the exact same Fantom patches and SysEx
    parameters that were used during the main recording pass.
    """
    patch_map: Dict[str, Dict] = {}
    rec_dir = Path(recordings_dir)
    if not rec_dir.is_dir():
        return patch_map
    manifests = sorted(rec_dir.glob("*_manifest.json"))
    for mf in manifests:
        try:
            data = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            continue
        for part in data.get("parts", []):
            track_name = part.get("source_track_name", "")
            if not track_name or track_name.lower() in patch_map:
                continue
            patch_map[track_name.lower()] = {
                "patch": part.get("patch"),
                "sound_design_raw": part.get("sound_design_raw"),
                "sound_design": part.get("sound_design"),
            }
    if patch_map:
        print(f"  Loaded {len(patch_map)} recorded patch assignments from {recordings_dir}")
    return patch_map


DEFAULT_SAMPLE_SETS = [
    {
        "label": "verse_8bars_drums",
        "kind": "drums",
        "start_bar": 8,
        "bars": 8,
        "track_prefixes": ["drum1_", "drum_aux_"],
        "make_bus": True,
        "tail_seconds": 4.5,
    },
    {
        "label": "verse_8bars_melodic",
        "kind": "melodic",
        "start_bar": 8,
        "bars": 8,
        "tracks": ["Bass", "Pad (Chords)"],
        "track_name_patterns": ["Main Melody"],
        "make_bus": False,
        "tail_seconds": 10.0,
    },
    {
        "label": "chorus_8bars_drums",
        "kind": "drums",
        "start_bar": 24,
        "bars": 8,
        "track_prefixes": ["drum2_", "drum_aux_"],
        "make_bus": True,
        "tail_seconds": 4.5,
    },
    {
        "label": "chorus_8bars_melodic",
        "kind": "melodic",
        "start_bar": 24,
        "bars": 8,
        "tracks": ["Bass", "Pad (Chords)"],
        "track_name_patterns": ["Chorus Melody"],
        "make_bus": False,
        "tail_seconds": 10.0,
    },
]


def safe_label(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name).strip("_")


def get_tempo_us(mid: mido.MidiFile) -> int:
    for track in mid.tracks:
        for msg in track:
            if msg.is_meta and msg.type == "set_tempo":
                return msg.tempo
    return mido.bpm2tempo(120)


def get_time_signature(mid: mido.MidiFile) -> Tuple[int, int]:
    for track in mid.tracks:
        for msg in track:
            if msg.is_meta and msg.type == "time_signature":
                return msg.numerator, msg.denominator
    return 4, 4


def ticks_per_bar(mid: mido.MidiFile) -> int:
    numerator, denominator = get_time_signature(mid)
    return int(mid.ticks_per_beat * numerator * (4 / denominator))


def track_name(track: mido.MidiTrack) -> str:
    if getattr(track, "name", ""):
        return track.name
    for msg in track:
        if msg.is_meta and msg.type == "track_name":
            return msg.name
    return ""


def has_notes(track: mido.MidiTrack) -> bool:
    return any(msg.type == "note_on" and getattr(msg, "velocity", 0) > 0 for msg in track if not msg.is_meta)


def find_tracks(mid: mido.MidiFile, spec: Dict) -> List[Tuple[int, mido.MidiTrack]]:
    selected = []
    wanted_names = {name.lower() for name in spec.get("tracks", [])}
    prefixes = tuple(prefix.lower() for prefix in spec.get("track_prefixes", []))
    name_patterns = tuple(p.lower() for p in spec.get("track_name_patterns", []))

    for idx, track in enumerate(mid.tracks):
        name = track_name(track)
        lname = name.lower()
        if not has_notes(track):
            continue
        if wanted_names and lname in wanted_names:
            selected.append((idx, track))
        elif prefixes and lname.startswith(prefixes):
            selected.append((idx, track))
        elif name_patterns and any(p in lname for p in name_patterns):
            selected.append((idx, track))

    return selected


def slice_track(track: mido.MidiTrack, start_tick: int, end_tick: int, shift_tick: int) -> mido.MidiTrack:
    events = []
    active_kept = {}
    abs_tick = 0

    for msg in track:
        abs_tick += msg.time
        if msg.is_meta:
            continue

        is_note_on = msg.type == "note_on" and getattr(msg, "velocity", 0) > 0
        is_note_end = msg.type == "note_off" or (msg.type == "note_on" and getattr(msg, "velocity", 0) == 0)

        if is_note_on and start_tick <= abs_tick < end_tick:
            active_kept[msg.note] = active_kept.get(msg.note, 0) + 1
            events.append((abs_tick - start_tick + shift_tick, msg.copy()))
        elif is_note_end and active_kept.get(msg.note, 0) > 0:
            active_kept[msg.note] -= 1
            events.append((min(abs_tick, end_tick) - start_tick + shift_tick, msg.copy()))
        elif msg.type in ("control_change", "program_change", "pitchwheel") and abs_tick <= start_tick:
            events.append((shift_tick, msg.copy()))

    out = mido.MidiTrack()
    out.name = track_name(track)
    out.append(mido.MetaMessage("track_name", name=out.name, time=0))
    events.sort(key=lambda item: item[0])

    last = 0
    for abs_out, msg in events:
        abs_out = max(0, abs_out)
        out.append(msg.copy(time=max(0, abs_out - last)))
        last = abs_out
    return out


def make_sync_track(count_in_ticks: int, tpb: int) -> mido.MidiTrack:
    sync_track = mido.MidiTrack()
    sync_track.name = "__sync__"
    sync_track.append(mido.MetaMessage("track_name", name="__sync__", time=0))
    note_dur = max(12, tpb // 12)
    for beat in range(_COUNT_IN_BEATS):
        delta_on = beat * tpb if beat == 0 else tpb - note_dur
        vel = 118 if beat == 0 else 92
        sync_track.append(mido.Message("note_on", channel=_SYNC_CH, note=_SYNC_NOTE, velocity=vel, time=delta_on))
        sync_track.append(mido.Message("note_off", channel=_SYNC_CH, note=_SYNC_NOTE, velocity=0, time=note_dur))
    return sync_track


def apply_patch_and_design(orchestrator: MultiPassOrchestrator, part_idx: int,
                           track_name_value: str, patch_info: Dict,
                           recorded_patches: Dict = None) -> Dict:
    controller = orchestrator.controller

    # Check if we have a recorded patch for this track — if so, replay it exactly.
    recorded = None
    if recorded_patches:
        recorded = recorded_patches.get(track_name_value.lower())

    if recorded and recorded.get("patch"):
        rp = recorded["patch"]
        controller.select_patch(part_idx, rp["msb"], rp["lsb"], rp["pc"])
        controller.set_part_level(part_idx, 115)
        raw = recorded.get("sound_design_raw")
        if raw:
            controller.apply_sound_design_from_manifest(part_idx, raw)
        design = recorded.get("sound_design", {})
        design["_replayed_from_manifest"] = True
        return design

    # Fallback: use the randomly selected patch.
    controller.select_patch(part_idx, patch_info["msb"], patch_info["lsb"], patch_info["pc"])
    controller.set_part_level(part_idx, 115)

    if patch_info.get("msb") == 97:
        return {"model_expansion": patch_info.get("name"), "zcore_edits": "skipped"}
    if hasattr(controller, "apply_track_sound_design"):
        return controller.apply_track_sound_design(part_idx, track_name_value) or {}
    return {}


def build_clip_mid(source_mid: mido.MidiFile, selected_tracks: List[Tuple[int, mido.MidiTrack]],
                   sample_spec: Dict, orchestrator: MultiPassOrchestrator,
                   tempo_us: int, count_in_ticks: int,
                   recorded_patches: Dict = None) -> Tuple[mido.MidiFile, List[str], List[Dict]]:
    tpb = source_mid.ticks_per_beat
    bar_ticks = ticks_per_bar(source_mid)
    start_tick = sample_spec["start_bar"] * bar_ticks
    end_tick = start_tick + (sample_spec["bars"] * bar_ticks)

    clip_mid = mido.MidiFile()
    clip_mid.ticks_per_beat = tpb
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
    numerator, denominator = get_time_signature(source_mid)
    meta.append(mido.MetaMessage("time_signature", numerator=numerator, denominator=denominator, time=0))
    clip_mid.tracks.append(meta)

    recorded_names = []
    manifest_parts = []

    for part_idx, (source_idx, source_track) in enumerate(selected_tracks[:15]):
        source_name = track_name(source_track)
        clipped = slice_track(source_track, start_tick, end_tick, count_in_ticks)
        if not has_notes(clipped):
            continue

        clipped.name = source_name
        patch_info = orchestrator.get_patch_for_track_name(source_name)
        print(f"  {sample_spec['label']}: Part {part_idx + 1} -> {source_name} [{patch_info['name']}]")
        sound_design = apply_patch_and_design(orchestrator, part_idx, source_name, patch_info,
                                              recorded_patches=recorded_patches)

        for msg in clipped:
            if not msg.is_meta and hasattr(msg, "channel"):
                msg.channel = part_idx

        clip_mid.tracks.append(clipped)
        recorded_names.append(source_name)
        manifest_parts.append({
            "source_track_index": source_idx,
            "source_track_name": source_name,
            "recorded_track_name": source_name,
            "part": part_idx + 1,
            "midi_channel": part_idx + 1,
            "usb_pair": f"{part_idx * 2 + 1}/{part_idx * 2 + 2}",
            "patch": patch_info,
            "sound_design": sound_design,
        })

    clip_mid.tracks.append(make_sync_track(count_in_ticks, tpb))
    orchestrator.controller.select_patch(_SYNC_CH, _SYNC_SOUND["msb"], _SYNC_SOUND["lsb"], _SYNC_SOUND["pc"])
    orchestrator.controller.set_part_level(_SYNC_CH, 127)
    return clip_mid, recorded_names, manifest_parts


def save_user_midi_excerpt(clip_mid: mido.MidiFile, output_path: Path) -> Path:
    """Save the playable sample MIDI without the internal sync click track."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    excerpt_mid = mido.MidiFile(type=clip_mid.type)
    excerpt_mid.ticks_per_beat = clip_mid.ticks_per_beat

    for track in clip_mid.tracks:
        name = track_name(track)
        if name == "__sync__":
            continue
        out_track = mido.MidiTrack()
        out_track.name = name
        for msg in track:
            out_track.append(msg.copy())
        excerpt_mid.tracks.append(out_track)

    excerpt_mid.save(output_path)
    return output_path


def sum_stems_to_bus(stem_paths: Iterable[str], output_path: Path) -> Optional[Path]:
    arrays = []
    samplerate = None
    min_len = None
    for stem_path in stem_paths:
        data, sr = sf.read(stem_path, always_2d=True)
        if samplerate is None:
            samplerate = sr
        if sr != samplerate:
            print(f"  Skipping bus sum due to sample-rate mismatch: {stem_path}")
            return None
        arrays.append(data[:, :2])
        min_len = len(data) if min_len is None else min(min_len, len(data))

    if not arrays or samplerate is None or min_len is None:
        return None

    bus = np.sum([arr[:min_len] for arr in arrays], axis=0)
    peak = float(np.max(np.abs(bus))) if bus.size else 0.0
    if peak > 0.95:
        bus = bus * (0.95 / peak)
    sf.write(output_path, bus.astype(np.float32), samplerate, subtype="FLOAT")
    return output_path


def first_transient_sample(data: np.ndarray, samplerate: int, threshold_db: float = -42.0) -> Optional[int]:
    mono = np.max(np.abs(data), axis=1)
    if mono.size == 0:
        return None

    peak = float(np.max(mono))
    if peak < 1e-6:
        return None

    threshold = max(10 ** (threshold_db / 20.0), peak * 0.06)
    hits = np.where(mono >= threshold)[0]
    if len(hits) == 0:
        return None

    # Step back to the nearest local quiet point to preserve transient onset.
    idx = int(hits[0])
    quiet = max(peak * 0.015, 10 ** (-60 / 20.0))
    search_start = max(0, idx - int(0.050 * samplerate))
    for pos in range(idx, search_start, -1):
        if mono[pos] <= quiet:
            return pos
    return idx


def transient_onsets(data: np.ndarray, samplerate: int, threshold_db: float = -42.0,
                     min_gap_ms: float = 70.0) -> List[int]:
    mono = np.max(np.abs(data), axis=1)
    if mono.size == 0:
        return []

    peak = float(np.max(mono))
    if peak < 1e-6:
        return []

    threshold = max(10 ** (threshold_db / 20.0), peak * 0.06)
    quiet = max(peak * 0.015, 10 ** (-60 / 20.0))
    min_gap = int(samplerate * min_gap_ms / 1000.0)
    hits = np.where(mono >= threshold)[0]
    if len(hits) == 0:
        return []

    onsets = []
    last_onset = -min_gap
    in_hit = False
    for idx in hits:
        idx = int(idx)
        if in_hit and idx - last_onset < min_gap:
            continue

        search_start = max(0, idx - int(0.050 * samplerate))
        onset = idx
        for pos in range(idx, search_start, -1):
            if mono[pos] <= quiet:
                onset = pos
                break

        if onset - last_onset >= min_gap:
            onsets.append(onset)
            last_onset = onset
        in_hit = True

    return onsets


def find_clean_one_shot_end(data: np.ndarray, samplerate: int, hit: int,
                            max_end: int, min_tail_ms: float = 120.0,
                            guard_ms: float = 8.0) -> Tuple[int, Optional[int]]:
    onsets = transient_onsets(data, samplerate)
    min_end = hit + int(samplerate * min_tail_ms / 1000.0)
    guard = int(samplerate * guard_ms / 1000.0)
    for onset in onsets:
        if onset <= hit + guard:
            continue
        candidate = max(hit, onset - guard)
        if candidate >= min_end:
            return min(candidate, max_end), onset
    return max_end, None


def extract_normalized_one_shot(stem_path: str, output_path: Path,
                                pre_roll_ms: float = 12.0, tail_seconds: float = 2.0,
                                peak_target: float = 0.90) -> Optional[Dict]:
    data, sr = sf.read(stem_path, always_2d=True)
    hit = first_transient_sample(data, sr)
    if hit is None:
        print(f"  One-shot skipped, no transient: {stem_path}")
        return None

    start = max(0, hit - int(sr * pre_roll_ms / 1000.0))
    max_end = min(len(data), hit + int(sr * tail_seconds))
    end, next_transient = find_clean_one_shot_end(data, sr, hit, max_end)
    sample = data[start:end, :2].astype(np.float32)
    if sample.size == 0:
        return None

    peak = float(np.max(np.abs(sample)))
    if peak < 1e-6:
        return None
    sample *= peak_target / peak

    fade_len = min(len(sample), int(sr * 0.010))
    if fade_len > 1:
        fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
        sample[-fade_len:] *= fade[:, None]

    sf.write(output_path, sample, sr, subtype="FLOAT")
    return {
        "path": str(output_path),
        "source_stem": stem_path,
        "first_transient_sample": hit,
        "next_transient_sample": next_transient,
        "start_sample": start,
        "end_sample": end,
        "cropped_before_next_hit": next_transient is not None,
        "duration_seconds": len(sample) / float(sr),
        "peak_target": peak_target,
    }


def create_drum_one_shots(stem_paths: Dict[str, str], output_dir: Path, sample_label: str,
                          tail_seconds: float, peak_target: float) -> Dict[str, Dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    one_shots = {}

    for stem_key, stem_path in stem_paths.items():
        stem = Path(stem_path)
        label = stem.stem
        marker = "_usb"
        if marker in label:
            label = label.split(marker, 1)[-1]
            label = label.split("_", 1)[-1] if "_" in label else label
        label = safe_label(label)
        output_path = output_dir / f"{safe_label(sample_label)}_one_shot_{label}.wav"
        result = extract_normalized_one_shot(
            stem_path,
            output_path,
            tail_seconds=tail_seconds,
            peak_target=peak_target,
        )
        if result:
            one_shots[stem_key] = result
            print(f"  One-shot: {output_path}")

    return one_shots


def record_sample_set(source_mid: mido.MidiFile, midi_path: Path, sample_spec: Dict,
                      root_output: Path, device_index: int,
                      controller: FantomController, one_shot_tail: float,
                      one_shot_peak: float, recorded_patches: Dict = None) -> Dict:
    selected = find_tracks(source_mid, sample_spec)
    if not selected:
        print(f"Skipping {sample_spec['label']}: no matching tracks found")
        return {}

    tail_seconds = sample_spec.get("tail_seconds", 4.5)

    sample_dir = root_output / sample_spec["label"]
    stems_dir = sample_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)

    recorder = AudioRecorder(device_index=device_index, output_dir=str(stems_dir))
    orchestrator = MultiPassOrchestrator(recorder, controller)

    tempo_us = get_tempo_us(source_mid)
    tpb = source_mid.ticks_per_beat
    count_in_ticks = _COUNT_IN_BEATS * tpb
    count_in_seconds = mido.tick2second(count_in_ticks, tpb, tempo_us)
    clip_seconds = mido.tick2second(sample_spec["bars"] * ticks_per_bar(source_mid), tpb, tempo_us)

    print(f"\n=== SAMPLE: {sample_spec['label']} ({sample_spec['start_bar']}-{sample_spec['start_bar'] + sample_spec['bars'] - 1}) ===")
    clip_mid, recorded_names, manifest_parts = build_clip_mid(
        source_mid, selected, sample_spec, orchestrator, tempo_us, count_in_ticks,
        recorded_patches=recorded_patches
    )
    if not recorded_names:
        print(f"Skipping {sample_spec['label']}: no notes in selected range")
        return {}

    clip_name = f"{safe_label(midi_path.stem)}_{sample_spec['label']}"
    midi_excerpt_dir = sample_dir / "midi"
    midi_excerpt_path = save_user_midi_excerpt(
        clip_mid,
        midi_excerpt_dir / f"{clip_name}.mid",
    )
    print(f"  MIDI excerpt: {midi_excerpt_path}")

    with tempfile.NamedTemporaryFile(prefix=clip_name + "_", suffix=".mid", delete=False) as tmp:
        temp_midi = tmp.name
    clip_mid.save(temp_midi)

    output_filename = f"{clip_name}_pass.wav"
    # AudioRecorder.play_midi_and_record adds a fixed 2s release buffer internally.
    # Subtract that here so the requested tail is the actual recorded tail.
    record_duration = count_in_seconds + clip_seconds + max(0.0, tail_seconds - 2.0)
    actual_pre_roll = recorder.play_midi_and_record(
        temp_midi, output_filename, controller.port_name, record_duration
    )
    Path(temp_midi).unlink(missing_ok=True)

    stem_paths, sync_info = recorder.split_stems(
        output_filename,
        recorded_names,
        sample_spec["label"],
        pre_roll_seconds=actual_pre_roll,
        count_in_seconds=count_in_seconds,
        return_sync_info=True,
    )

    if sync_info.get("fallback_used"):
        print(f"  NOTE: Sync click detection used fallback — stems may be slightly misaligned")

    # Normalize extracted stems and clean up the large pass file
    print("  Normalizing stems...")
    normalized_stems = {}
    for key, path in stem_paths.items():
        data, sr = sf.read(path)
        peak = np.max(np.abs(data))
        if peak > 1e-6:
            data = data * (0.99 / peak) # Normalize to -0.1 dB
            sf.write(path, data.astype(np.float32), sr, subtype="FLOAT")
        normalized_stems[key] = path
    
    # Delete the large raw recording pass file to save disk space
    pass_file_path = Path(recorder.output_dir) / output_filename
    if pass_file_path.exists():
        pass_file_path.unlink()
        print(f"  Deleted multi-stem pass file: {output_filename}")

    bus_path = None
    one_shots = {}
    if sample_spec.get("make_bus"):
        bus_dir = sample_dir / "bus"
        bus_dir.mkdir(exist_ok=True)
        bus_path = sum_stems_to_bus(normalized_stems.values(), bus_dir / f"{clip_name}_drum_bus.wav")
        if bus_path:
            # Normalize the drum bus
            data, sr = sf.read(bus_path)
            peak = np.max(np.abs(data))
            if peak > 1e-6:
                data = data * (0.99 / peak)
                sf.write(bus_path, data, sr, subtype="FLOAT")
            print(f"  Drum bus: {bus_path} (normalized)")
            
        one_shots = create_drum_one_shots(
            normalized_stems,
            sample_dir / "oneshots",
            sample_spec["label"],
            one_shot_tail,
            one_shot_peak,
        )

    manifest = {
        "source_midi": str(midi_path),
        "sample_label": sample_spec["label"],
        "kind": sample_spec["kind"],
        "start_bar": sample_spec["start_bar"],
        "bars": sample_spec["bars"],
        "tail_seconds": tail_seconds,
        "count_in_seconds": count_in_seconds,
        "record_duration_seconds": record_duration,
        "midi_excerpt": str(midi_excerpt_path),
        "stems": normalized_stems,
        "bus": str(bus_path) if bus_path else None,
        "one_shots": one_shots,
        "parts": manifest_parts,
    }
    manifest_path = sample_dir / f"{clip_name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  Manifest: {manifest_path}")
    return manifest


def generate_sample_pack(midi_file: str, output_root: str = "output/samples", 
                         device_index: int = 9, verse_start: int = 8, 
                         chorus_start: int = 24, bars: int = 8,
                         recordings_dir: str = None) -> str:
    """High-level entry point for programmatic sample pack generation."""
    midi_path = Path(midi_file).resolve()
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    specs = []
    for spec in DEFAULT_SAMPLE_SETS:
        spec = dict(spec)
        spec["bars"] = bars
        if spec["label"].startswith("verse"):
            spec["start_bar"] = verse_start
        elif spec["label"].startswith("chorus"):
            spec["start_bar"] = chorus_start
        specs.append(spec)

    root_output = Path(output_root).resolve() / safe_label(midi_path.stem)
    root_output.mkdir(parents=True, exist_ok=True)

    source_mid = mido.MidiFile(str(midi_path))
    controller = FantomController()
    if not controller.output or not controller.port_name:
        raise RuntimeError("Roland Fantom MIDI output port not found.")

    recorded_patches = load_recording_patch_map(recordings_dir) if recordings_dir else {}

    run_manifest = {}
    for spec in specs:
        run_manifest[spec["label"]] = record_sample_set(
            source_mid, midi_path, spec, root_output, device_index,
            controller, 2.0, 0.90, recorded_patches=recorded_patches
        )

    run_manifest["_midi_excerpts"] = {
        label: result.get("midi_excerpt")
        for label, result in run_manifest.items()
        if isinstance(result, dict) and result.get("midi_excerpt")
    }

    run_manifest_path = root_output / f"{safe_label(midi_path.stem)}_sample_run_manifest.json"
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    
    return str(root_output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate 8-bar melodic and drum sample stems from a MIDI song.")
    parser.add_argument("midi_file", help="Generated MIDI file to sample.")
    parser.add_argument("--output-root", default="output/samples", help="Root folder for sample output.")
    parser.add_argument("--device-index", type=int, default=9, help="sounddevice input device index for Fantom USB audio.")
    parser.add_argument("--verse-start", type=int, default=8, help="0-based start bar for verse samples.")
    parser.add_argument("--chorus-start", type=int, default=24, help="0-based start bar for chorus samples.")
    parser.add_argument("--bars", type=int, default=8, help="Number of bars per sample clip.")
    parser.add_argument("--one-shot-tail", type=float, default=2.0, help="Tail length for extracted drum one-shots, seconds.")
    parser.add_argument("--one-shot-peak", type=float, default=0.90, help="Peak normalization target for drum one-shots.")
    parser.add_argument("--recordings-dir", type=str, default=None, help="Path to recordings dir with *_manifest.json files for patch replay.")
    args = parser.parse_args()

    midi_path = Path(args.midi_file).resolve()
    if not midi_path.exists():
        print(f"MIDI file not found: {midi_path}", file=sys.stderr)
        return 1

    specs = []
    for spec in DEFAULT_SAMPLE_SETS:
        spec = dict(spec)
        spec["bars"] = args.bars
        if spec["label"].startswith("verse"):
            spec["start_bar"] = args.verse_start
        elif spec["label"].startswith("chorus"):
            spec["start_bar"] = args.chorus_start
        specs.append(spec)

    root_output = Path(args.output_root).resolve() / safe_label(midi_path.stem)
    root_output.mkdir(parents=True, exist_ok=True)

    source_mid = mido.MidiFile(str(midi_path))
    controller = FantomController()
    if not controller.output or not controller.port_name:
        print("Roland Fantom MIDI output port not found.", file=sys.stderr)
        return 1

    recorded_patches = load_recording_patch_map(args.recordings_dir) if args.recordings_dir else {}

    run_manifest = {}
    for spec in specs:
        run_manifest[spec["label"]] = record_sample_set(
            source_mid, midi_path, spec, root_output, args.device_index,
            controller, args.one_shot_tail, args.one_shot_peak,
            recorded_patches=recorded_patches
        )

    run_manifest["_midi_excerpts"] = {
        label: result.get("midi_excerpt")
        for label, result in run_manifest.items()
        if isinstance(result, dict) and result.get("midi_excerpt")
    }

    run_manifest_path = root_output / f"{safe_label(midi_path.stem)}_sample_run_manifest.json"
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(f"\nSample generation complete: {root_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
