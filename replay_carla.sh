#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /opt/ros/humble/setup.bash
source install/setup.bash
exec /usr/bin/python3 tools/replay_carla_video.py "$@"
