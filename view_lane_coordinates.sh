#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash
ROAD_ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$ROAD_ROOT/install/setup.bash"
exec /usr/bin/python3 "$ROAD_ROOT/tools/view_lane_coordinates.py" "$@"
