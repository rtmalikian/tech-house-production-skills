#!/usr/bin/env bash
set -euo pipefail

TARGET_LUFS="${1:--20}"
DURATION_SECONDS="${2:-300}"
OUTPUT="${3:-pink_noise_5min_${TARGET_LUFS}LUFS.wav}"
SAMPLE_RATE="${SAMPLE_RATE:-48000}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required. Install it with: brew install ffmpeg" >&2
  exit 1
fi

ffmpeg -hide_banner -y \
  -f lavfi \
  -i "anoisesrc=color=pink:duration=${DURATION_SECONDS}:sample_rate=${SAMPLE_RATE}:amplitude=0.25" \
  -af "loudnorm=I=${TARGET_LUFS}:TP=-3:LRA=1:print_format=summary,aresample=${SAMPLE_RATE}" \
  -ar "${SAMPLE_RATE}" \
  -c:a pcm_s24le \
  "${OUTPUT}"

