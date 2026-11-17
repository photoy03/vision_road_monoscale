#!/bin/bash
# Lane-side real-time latency on THIS computer (image published -> display line received), Town05, 30 Hz,
# with the demo setting: outer lines, centrelines, lane guidance, fake vehicle state, display filter in the
# perception worker, HUD copy.  Writes the computer's description and one folder per run, then SUMMARY.txt.
#
#   tools/measure_latency.sh                 3 runs with full images, 3 with the AI filter (full images; if
#                                            data/models/yolopv2/yolopv2.pt is there)
#   RUNS=1 FRAMES=600 tools/measure_latency.sh   a short check (600 frames = 20 s)
#   CROP=1 tools/measure_latency.sh          also 3 runs with row-cropped images (off by default: full images are
#                                            the standard)
#
# Needs: this workspace built (colcon build --symlink-install, see BUILD_AND_TEST.md), ROS 2 Humble, the HUD copy
# built, and the Town05 recording: VISION_ROAD_DISK/rosbag/town05_30hz/ (27 GB) and
# VISION_ROAD_DISK/vision-road-cache/town05_30hz_native.cdrz(.json) (5 GB; copy it with the bag keeping file times,
# e.g. rsync -a, or the replay reads the slower bag).  VISION_ROAD_DISK defaults to /media/jaehun/38F7-677B.
# Changes nothing on the computer (no power, CPU or priority settings); close other heavy programs first.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
OUT=${1:-results/latency_$(hostname)_$(date +%Y%m%d_%H%M)};RUNS=${RUNS:-3};FRAMES=${FRAMES:-0}
mkdir -p "$OUT"
{
  echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "os: $(uname -srm)"
  echo "--- CPU"; lscpu | grep -E '^(Model name|CPU\(s\)|Thread|Core|Socket|CPU max MHz|CPU min MHz|L3)' || true
  echo "--- power and CPU frequency policy (read only)"
  echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
  echo "energy_performance_preference: $(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null || echo unknown)"
  echo "power profile: $(powerprofilesctl get 2>/dev/null || echo unknown)"
  for s in /sys/class/power_supply/*/online; do [ -e "$s" ] && echo "AC $(basename "$(dirname "$s")"): $(cat "$s") (1 = plugged in)"; done
  echo "--- GPU"; nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,pstate --format=csv 2>/dev/null || echo "no nvidia-smi"
  echo "--- memory"; free -g | head -2
  echo "--- load before the runs"; uptime
  echo "--- software"; python3 -c "import numpy,cv2,sys;print('python',sys.version.split()[0],'numpy',numpy.__version__,'opencv',cv2.__version__)"
} > "$OUT/system.txt" 2>&1
set +u;source /opt/ros/humble/setup.bash;source install/setup.bash;set -u
# The C++ parts (src/vision_road/vision_road/native.cpp) are built by colcon build; without them the Python code runs (slower).
python3 -c "
try:
    import vision_road._native as n;ok=all(hasattr(n,f) for f in ('color_planes','ridges','edge_like','track_match','corridor_error','reproject'))
except ImportError:ok=False
print('native fast path (vision_road._native):','built' if ok else 'MISSING - rebuild: colcon build --symlink-install --packages-select vision_road')" >> "$OUT/system.txt" 2>&1
cat "$OUT/system.txt"
A="display_filter_inline:=true lane_guidance:=true fake_vehicle_state:=true"
W=$PWD/data/models/yolopv2/yolopv2.pt
run(){ local n=$1;shift;echo "run $n ...";python3 tools/realtime_eval.py --video town05 --frames "$FRAMES" --output "$OUT/$n" "$@" > "$OUT/$n.log" 2>&1 && echo "  done" || echo "  FAILED (see $OUT/$n.log)"; }
for i in $(seq 1 "$RUNS");do run full$i --launch-arg $A;done
if [ "${CROP:-0}" = 1 ];then for i in $(seq 1 "$RUNS");do run crop$i --crop-rows 578:984 --launch-arg $A;done;fi
if [ -f "$W" ];then for i in $(seq 1 "$RUNS");do run ai$i --launch-arg $A learned_weights:=$W;done;fi
python3 tools/latency_summary.py "$OUT" | tee "$OUT/SUMMARY.txt"
echo "results: $OUT (SUMMARY.txt, system.txt)"
