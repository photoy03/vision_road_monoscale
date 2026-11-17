#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
display_index="${DISPLAY_INDEX:-0}"

cd "$script_dir"
exec python3 prius_wheel_based_driver_eye.py \
  --preview-eye-camera \
  --clean-preview \
  --fullscreen \
  --display-index "$display_index" \
  --preview-width 1920 \
  --preview-height 1080 \
  "$@"
