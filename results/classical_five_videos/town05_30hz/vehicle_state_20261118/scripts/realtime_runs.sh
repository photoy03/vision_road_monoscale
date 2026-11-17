#!/bin/bash
# Plan 23 real-time runs (Town05): detector + display filter + lane guidance + fake vehicle state + the HUD copy
# (lane-only, drawing the current path); three judge runs, then one recorded run.
cd /home/jaehun/Desktop/vision-road;source /opt/ros/humble/setup.bash;source install/setup.bash
R=results/classical_five_videos/town05_30hz/vehicle_state_20261118/runs/realtime;P=${1:-}
A="display_filter_inline:=true lane_guidance:=true fake_vehicle_state:=true"
for i in 1 2 3;do python3 tools/realtime_eval.py --video town05 --output $R/${P}judge${i}_town05 --launch-arg $A > $R/${P}judge${i}_town05.log 2>&1;echo judge$i $?;done
python3 tools/realtime_eval.py --video town05 --record --output $R/${P}record_town05 --launch-arg $A > $R/${P}record_town05.log 2>&1;echo record $?
