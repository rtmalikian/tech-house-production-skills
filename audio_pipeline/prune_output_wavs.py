#!/usr/bin/env python3
"""
Prune WAV render files from an output directory.

Dry run by default:
    music_venv/bin/python scripts/audio_pipeline/prune_output_wavs.py

Delete files:
    music_venv/bin/python scripts/audio_pipeline/prune_output_wavs.py --delete

Custom output directory:
    music_venv/bin/python scripts/audio_pipeline/prune_output_wavs.py --output-dir output/recordings --delete
"""

import argparse
from pathlib import Path


def format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024.0
    return f"{size}B"


def find_wavs(output_dir: Path) -> list[Path]:
    return sorted(p for p in output_dir.rglob("*.wav") if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune .wav files under an output directory.")
    parser.add_argument("--output-dir", default="output", help="Directory to prune. Defaults to output.")
    parser.add_argument("--delete", action="store_true", help="Actually delete files. Default is dry run.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists() or not output_dir.is_dir():
        print(f"Output directory not found: {output_dir}")
        return 1

    wavs = find_wavs(output_dir)
    total_bytes = sum(p.stat().st_size for p in wavs)

    action = "Deleting" if args.delete else "Dry run"
    print(f"{action}: {len(wavs)} WAV files under {output_dir}")
    print(f"Total size: {format_bytes(total_bytes)}")

    if not wavs:
        return 0

    if not args.delete:
        print("No files deleted. Re-run with --delete to prune.")
        return 0

    deleted = 0
    for path in wavs:
        path.unlink()
        deleted += 1

    print(f"Deleted {deleted} WAV files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
