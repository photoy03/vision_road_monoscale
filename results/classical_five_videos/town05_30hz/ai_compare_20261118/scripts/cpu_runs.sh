#!/bin/bash
# AI filter off / on, Town05 real time (full images, demo setting), with per-process CPU and GPU use (cpu_sample.py, 50 s).
cd /home/jaehun/Desktop/vision-road;source /opt/ros/humble/setup.bash;source install/setup.bash
N=results/classical_five_videos/town05_30hz/ai_compare_20261118;R=$N/runs/realtime;mkdir -p $R;W=$PWD/data/models/yolopv2/yolopv2.pt
A="display_filter_inline:=true lane_guidance:=true fake_vehicle_state:=true"
run(){ n=$1;shift;python3 $N/scripts/cpu_sample.py $R/${n}_cpu.json 50 gpu > $R/${n}_cpu.log 2>&1 &
  python3 tools/realtime_eval.py --video town05 --output $R/$n --launch-arg $A "$@" > $R/$n.log 2>&1;echo $n $?;wait; }
run off1;run on1 learned_weights:=$W
