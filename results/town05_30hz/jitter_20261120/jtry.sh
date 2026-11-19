#!/bin/bash
# Plan 32: predict (with the display line) on the 14 recordings, three at a time, then the jitter table.
# usage: jtry.sh OUT_DIR "detector set" "display set"
cd /home/jaehun/Desktop/vision-road;source /opt/ros/humble/setup.bash;source install/setup.bash
OUT=$1;SET=$2;DISP=$3;GU="offset_at_m=1.4096 change_without_neighbour=True trust_single_line=True on_m=0.25 off_m=0.20 keep_target_far=True";mkdir -p $OUT
run(){ for s in "$@";do python3 tools/gt_eval.py predict $s $OUT/$s ${SET:+--set $SET} ${DISP:+--display $DISP} --guide $GU > $OUT/$s.predict.log 2>&1;done; }
run town05_30hz s1_lanefollow s2_wobble s3a_zigzag_cross s3b_zigzag_inlane & run maploop_30hz s4_straddle s5_lanechange s6_yellow s7_lights & run s8a_lanefollow_20kmh s8b_lanefollow_50kmh s8c_lanechange_20kmh s8d_lanechange_50kmh & wait
python3 results/town05_30hz/jitter_20261120/jtable.py $OUT > $OUT/jitter.md;cat $OUT/jitter.md
