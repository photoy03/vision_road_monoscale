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
#   --fake-vehicle-state  made-up /hero/vehicle_state and /hero/steering_wheel from the pose
#                   (tools/fake_vehicle_state.py), until the driving node sends them: the HUD then
#                   draws the current path and the guidance takes its speed
#   --crop-rows LO:HI   the replay sends only image rows LO..HI-1 (e.g. 578:984, the rows the lane
#                   BEV reads; CameraInfo roi says which), a third of the image bytes
#   --signs         traffic lights and speed signs (YOLO, src/yolo_signs, GPU) on the HUD: the detector
#                   runs with the YOLO environment's Python (YOLO_PYTHON, default ~/yolo_env/bin/python)
#                   and lane_guidance puts what it confirms into the HUD scene.  SIGNS_FULL_FRAME=true|false
#                   (whole image or the original upper-centre tile), SIGNS_FPS (max inferences per second, 0 = all),
#                   SIGNS_IMGSZ (inference size, default 1920); the detector takes the images serialized (raw_cdr)
# The HUD is this workspace's copy (src/hero_hud), built into install/.
set -euo pipefail
ROAD="$(cd "$(dirname "$0")" && pwd)"
HUD_CONFIG=$ROAD/src/hero_hud/config/hud.yaml
DISK=/media/jaehun/38F7-677B
VIDEO=${1:-}; shift || true
HUD=1; LOOP=0; FRAMES=0; LEARNED=""; GUIDANCE=true; FAKE_VS=false; CROP=(); SIGNS=false
YOLO_PYTHON=${YOLO_PYTHON:-$HOME/yolo_env/bin/python}; SIGNS_FULL_FRAME=${SIGNS_FULL_FRAME:-true}; SIGNS_FPS=${SIGNS_FPS:-0.0}
# Plan 30 (D1): with --signs the perception worker hands every 2nd image to the sign detector through shared memory
# (the camera is not subscribed twice: lane latency p95 27.7-28.2 -> 24.8-25.6 ms on town05).
SIGNS_SHM=${SIGNS_SHM:-vision_road_dashcam}
SIGNS_IMGSZ=${SIGNS_IMGSZ:-1920}  # plan 30: the full frame at 1920 finds far lights (s7 confirmed 14 -> 80 %)
while [ $# -gt 0 ]; do
  case "$1" in
    --no-hud) HUD=0;; --loop) LOOP=1;; --frames) FRAMES=$2; shift;;
    --learned) LEARNED="learned_weights:=$ROAD/data/models/yolopv2/yolopv2.pt";;
    --no-guidance) GUIDANCE=false;;
    --fake-vehicle-state) FAKE_VS=true;;
    --crop-rows) CROP=(--crop-rows "$2"); shift;;
    --signs) SIGNS=true;;
    *) echo "unknown option $1"; exit 2;;
  esac; shift
done
case "$VIDEO" in
  maploop) BAG=$DISK/rosbag/maploop_30hz; CACHE=$ROAD/data/maploop30hz_native.cdrz;;
  town04)  BAG=$DISK/rosbag/town04_30hz;  CACHE=$DISK/vision-road-cache/town04_30hz_native.cdrz;;
  town05)  BAG=$DISK/rosbag/town05_30hz;  CACHE=$DISK/vision-road-cache/town05_30hz_native.cdrz;;
  town05_signs) BAG=$DISK/rosbag/town05_signs; CACHE=$DISK/vision-road-cache/town05_signs_native.cdrz;;
  live)    BAG=;;
  *) sed -n '2,22p' "$0"; exit 2;;
esac
set +u; source /opt/ros/humble/setup.bash; source "$ROAD/install/setup.bash"; set -u
PIDS=()
# Background jobs of a script start with SIGINT ignored, and ros2 launch stops its nodes with SIGINT:
# start them with SIGINT restored, then SIGINT stops everything in order.
SIGINT_EXEC=(/usr/bin/python3 -c 'import os,signal,sys;signal.signal(signal.SIGINT,signal.SIG_DFL);os.execvp(sys.argv[1],sys.argv[1:])')
cleanup() { for p in "${PIDS[@]}"; do kill -INT "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# Detector with the display filter inside it (carla.yaml settings), reference pose adapter, TF.
SHM="";[ "$SIGNS" = true ] && [ -n "$SIGNS_SHM" ] && SHM="image_shm:=$SIGNS_SHM"
( exec "${SIGINT_EXEC[@]}" ros2 launch vision_road maploop.launch.py display_filter_inline:=true lane_guidance:=$GUIDANCE fake_vehicle_state:=$FAKE_VS signs:=$SIGNS $SHM $LEARNED ) &
PIDS+=($!)
if [ "$SIGNS" = true ]; then
  # The YOLO environment (ultralytics for YOLO26) with this ROS environment sourced: it imports rclpy.
  ( exec "${SIGINT_EXEC[@]}" "$YOLO_PYTHON" "$ROAD/src/yolo_signs/town05_yolo26m_detector.py" --ros-args \
      -p full_frame:=$SIGNS_FULL_FRAME -p max_inference_fps:=$SIGNS_FPS -p imgsz:=$SIGNS_IMGSZ -p raw_cdr:=true -p image_shm:=$SIGNS_SHM ) &
  PIDS+=($!)
fi
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
    /usr/bin/python3 "$ROAD/tools/replay_maploop.py" --bag "$BAG" --cache "$CACHE" --max-frames "$FRAMES" --rate 1 ${CROP[@]+"${CROP[@]}"}
    [ "$LOOP" = 1 ] || break
  done
fi
