#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /opt/ros/humble/setup.bash
source install/setup.bash
exec ros2 launch vision_road carla.launch.py "$@"
