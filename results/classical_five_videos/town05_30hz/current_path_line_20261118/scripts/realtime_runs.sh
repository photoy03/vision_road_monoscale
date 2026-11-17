#!/bin/bash
# Plan 24 real-time runs (Town05): as plan 23 (detector + display filter + lane guidance + fake vehicle state +
# the HUD copy, lane-only), the HUD now drawing the current path as config/hud.yaml says (dots, 8 m). Three runs.
cd /home/jaehun/Desktop/vision-road;source /opt/ros/humble/setup.bash;source install/setup.bash
R=results/classical_five_videos/town05_30hz/current_path_line_20261118/runs/realtime
A="display_filter_inline:=true lane_guidance:=true fake_vehicle_state:=true"
for i in 1 2 3;do python3 tools/realtime_eval.py --video town05 --output $R/judge${i}_town05 --launch-arg $A > $R/judge${i}_town05.log 2>&1;echo judge$i $?;done
