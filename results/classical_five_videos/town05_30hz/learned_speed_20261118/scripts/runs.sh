#!/bin/bash
# AI filter (YOLOPv2 learned gate) off vs on, Town05 real time with HUD, at commit ffb2339 (outer lines on).
# The code under src/ must be the committed state when this runs (the caller swaps plan 20 work out and back).
# off: no weights (image_in_worker: the worker subscribes the camera itself)
# on: learned_weights:=yolopv2.pt (camera through road_node, compute worker + model process)
# structure: learned_weights:=<missing file> (the 'on' structure without the model: it fails to load and stays off)
cd /home/jaehun/Desktop/vision-road;source /opt/ros/humble/setup.bash;source install/setup.bash
L=results/classical_five_videos/town05_30hz/learned_speed_20261118;S=$L/scripts;R=$L/runs
W=/home/jaehun/Desktop/vision-road/data/models/yolopv2/yolopv2.pt;M=/home/jaehun/Desktop/vision-road/data/models/yolopv2/missing_on_purpose.pt
run(){ # name gpu launch-args...
  n=$1;g=$2;shift 2
  python3 $S/cpu_sample.py $R/${n}_cpu.json 50 $g > $R/${n}_cpu.log 2>&1 &
  python3 tools/realtime_eval.py --video town05 --output $R/$n --launch-arg display_filter_inline:=true "$@" > $R/$n.log 2>&1;echo $n $?;wait
}
for i in 1 2 3;do run off$i nogpu;done
for i in 1 2 3;do run on$i gpu learned_weights:=$W;done
for i in 1 2;do run structure$i nogpu learned_weights:=$M;done
