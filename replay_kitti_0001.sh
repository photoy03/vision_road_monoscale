#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /opt/ros/humble/setup.bash
source install/setup.bash
exec ros2 launch vision_road kitti.launch.py \
  sequence:="$PWD/data/kitti/2011_09_26/2011_09_26_drive_0001_sync" \
  odom_source:="${1:-oxts}" rate:="${2:-0.5}"
