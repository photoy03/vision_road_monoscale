#!/usr/bin/env bash
# Lane detection -> display filter -> HERO HUD in one command.
#
#   ./run_lane_hud.sh maploop|town04|town05   replay that recording at 30 Hz (header-paced), then stop
#   ./run_lane_hud.sh live                    no replay: a running CARLA (or bag player) publishes
#                                             /sensing/camera/dashcam/image_raw and /carla/ground_truth/pose
# options:
#   --no-hud        detector and display filter only
#   --loop          replay again and again until Ctrl+C
#   --frames N      replay only N frames
#   --learned       YOLOPv2 lane mask gate (GPU; off by default)
#   --no-guidance   no lane return / lane change guidance on the HUD (hero_coach lane_guidance)
# The HUD is this workspace's copy (src/hero_hud), built into install/.
set -euo pipefail
ROAD="$(cd "$(dirname "$0")" && pwd)"
HUD_CONFIG=$ROAD/src/hero_hud/config/hud.yaml
DISK=/media/jaehun/38F7-677B
VIDEO=${1:-}; shift || true
HUD=1; LOOP=0; FRAMES=0; LEARNED=""; GUIDANCE=true
while [ $# -gt 0 ]; do
  case "$1" in
    --no-hud) HUD=0;; --loop) LOOP=1;; --frames) FRAMES=$2; shift;;
    --learned) LEARNED="learned_weights:=$ROAD/data/models/yolopv2/yolopv2.pt";;
    --no-guidance) GUIDANCE=false;;
    *) echo "unknown option $1"; exit 2;;
  esac; shift
done
case "$VIDEO" in
  maploop) BAG=$DISK/rosbag/maploop_30hz; CACHE=$ROAD/data/maploop30hz_native.cdrz;;
  town04)  BAG=$DISK/rosbag/town04_30hz;  CACHE=$DISK/vision-road-cache/town04_30hz_native.cdrz;;
  town05)  BAG=$DISK/rosbag/town05_30hz;  CACHE=$DISK/vision-road-cache/town05_30hz_native.cdrz;;
  live)    BAG=;;
  *) sed -n '2,15p' "$0"; exit 2;;
esac
set +u; source /opt/ros/humble/setup.bash; source "$ROAD/install/setup.bash"; set -u
PIDS=()
# Background jobs of a script start with SIGINT ignored, and ros2 launch stops its nodes with SIGINT:
# start them with SIGINT restored, then SIGINT stops everything in order.
SIGINT_EXEC=(/usr/bin/python3 -c 'import os,signal,sys;signal.signal(signal.SIGINT,signal.SIG_DFL);os.execvp(sys.argv[1],sys.argv[1:])')
cleanup() { for p in "${PIDS[@]}"; do kill -INT "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# Detector with the display filter inside it (carla.yaml settings), reference pose adapter, TF.
( exec "${SIGINT_EXEC[@]}" ros2 launch vision_road maploop.launch.py display_filter_inline:=true lane_guidance:=$GUIDANCE $LEARNED ) &
PIDS+=($!)
/usr/bin/python3 - <<'PY'
import time,rclpy
rclpy.init();n=rclpy.create_node('lane_hud_wait');end=time.monotonic()+40
while n.count_publishers('/road/lane_display_markers')<1 and time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.1)
rclpy.shutdown()
PY

if [ "$HUD" = 1 ]; then
  ( set +u; unset QT_QPA_PLATFORM_PLUGIN_PATH QT_QPA_FONTDIR
    exec "${SIGINT_EXEC[@]}" "$ROAD/install/hero_hud/lib/hero_hud/hud_node" --ros-args --params-file "$HUD_CONFIG" \
      -p display.mirror_horizontal:=false -p road.lane_only:=true ) &
  PIDS+=($!)
  sleep 2
fi

if [ -z "$BAG" ]; then
  echo "live: waiting for /sensing/camera/dashcam/image_raw and /carla/ground_truth/pose (Ctrl+C to stop)"
  wait
else
  while true; do
    /usr/bin/python3 "$ROAD/tools/replay_maploop.py" --bag "$BAG" --cache "$CACHE" --max-frames "$FRAMES" --rate 1
    [ "$LOOP" = 1 ] || break
  done
fi
