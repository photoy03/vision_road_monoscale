#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash
ROAD_ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$ROAD_ROOT/install/setup.bash"
exec ros2 launch vision_road maploop.launch.py "$@"
