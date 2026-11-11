#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec /usr/bin/python3 tools/view_lane_results.py
