#!/bin/bash
# Plan 30: the painted-line truth (paint_stops) for the given scenes, one after another.
cd /home/jaehun/Desktop/vision-road;source /opt/ros/humble/setup.bash;source install/setup.bash
P=results/town05_30hz/gt_loop_20261119/paint_stops
for s in "$@";do python3 tools/gt_eval.py paint_stops $s $P/$s > $P/$s.log 2>&1;echo "$s $?";done
