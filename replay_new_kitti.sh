#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
route="${1:-0015}"
case "$route" in
  0001|0015|0028|0018|0056) ;;
  *) echo 'Usage: bash replay_new_kitti.sh [0001|0015|0028|0018|0056] [oxts|monoscale] [rate]'; exit 2 ;;
esac
source /opt/ros/humble/setup.bash
source install/setup.bash
exec ros2 launch vision_road kitti.launch.py \
  sequence:="$PWD/data/kitti/2011_09_26/2011_09_26_drive_${route}_sync" \
  odom_source:="${2:-oxts}" rate:="${3:-0.5}"
