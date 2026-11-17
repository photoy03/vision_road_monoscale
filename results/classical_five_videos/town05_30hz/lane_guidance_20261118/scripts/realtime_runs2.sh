#!/bin/bash
# Plan 21 real-time runs, round 2 (Town05): a same-day baseline without the guidance node, then three judge runs
# with it and no test listener (the listener is a test-only process; compute time comes from the recorded run).
cd /home/jaehun/Desktop/vision-road;source /opt/ros/humble/setup.bash;source install/setup.bash
R=results/classical_five_videos/town05_30hz/lane_guidance_20261118/runs/realtime;P=${1:-b2_};BASE=${2:-}
[ -n "$BASE" ] && { python3 tools/realtime_eval.py --video town05 --output $R/${P}noguidance_town05 --launch-arg display_filter_inline:=true lane_guidance:=false > $R/${P}noguidance_town05.log 2>&1;echo baseline $?; }
for i in 1 2 3;do python3 tools/realtime_eval.py --video town05 --output $R/${P}judge${i}_town05 --launch-arg display_filter_inline:=true lane_guidance:=true > $R/${P}judge${i}_town05.log 2>&1;echo judge$i $?;done
