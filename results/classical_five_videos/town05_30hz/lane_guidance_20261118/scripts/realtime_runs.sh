#!/bin/bash
# Plan 21 real-time runs (Town05): detector + display filter + lane guidance node + the HUD copy; three judge runs,
# then one recorded run.  The guidance listener runs during every run.
cd /home/jaehun/Desktop/vision-road;source /opt/ros/humble/setup.bash;source install/setup.bash
L=results/classical_five_videos/town05_30hz/lane_guidance_20261118;R=$L/runs/realtime;P=${1:-}
for i in 1 2 3;do
  python3 $L/scripts/guidance_listener.py $R/${P}judge${i}_guidance.json 105 > $R/${P}judge${i}_guidance.log 2>&1 &
  python3 tools/realtime_eval.py --video town05 --output $R/${P}judge${i}_town05 --launch-arg display_filter_inline:=true lane_guidance:=true > $R/${P}judge${i}_town05.log 2>&1;echo judge$i $?;wait
done
python3 $L/scripts/guidance_listener.py $R/${P}record_guidance.json 110 > $R/${P}record_guidance.log 2>&1 &
python3 tools/realtime_eval.py --video town05 --record --output $R/${P}record_town05 --launch-arg display_filter_inline:=true lane_guidance:=true > $R/${P}record_town05.log 2>&1;echo record $?;wait
