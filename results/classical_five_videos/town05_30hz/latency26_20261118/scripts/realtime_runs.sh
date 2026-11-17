#!/bin/bash
# Plan 26 real-time runs (Town05, this laptop): the demo setting (outer lines, centrelines, guidance, fake vehicle
# state, display filter in the worker, HUD copy), three runs each:
#   crop  : the replay sends image rows 578-983 (CameraInfo roi)
#   full  : full images (as before)
#   ai    : + AI filter (YOLOPv2 learned gate), full images, image_in_worker with the model fed from the worker
#   aicrop: + AI filter with row-cropped images (the model sees the image black outside the rows), one run
cd /home/jaehun/Desktop/vision-road;source /opt/ros/humble/setup.bash;source install/setup.bash
R=results/classical_five_videos/town05_30hz/latency26_20261118/runs/realtime;W=$PWD/data/models/yolopv2/yolopv2.pt;P=${1:-}
A="display_filter_inline:=true lane_guidance:=true fake_vehicle_state:=true"
run(){ n=$1;shift;python3 tools/realtime_eval.py --video town05 --output $R/$P$n "$@" > $R/$P$n.log 2>&1;echo $n $?; }
for i in 1 2 3;do run crop$i --crop-rows 578:984 --launch-arg $A;done
for i in 1 2 3;do run full$i --launch-arg $A;done
for i in 1 2 3;do run ai$i --launch-arg $A learned_weights:=$W;done
run aicrop1 --crop-rows 578:984 --launch-arg $A learned_weights:=$W
