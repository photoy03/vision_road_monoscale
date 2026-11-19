#!/bin/bash
# Plan 31: one try over the 12 GT scenes, three at a time.  usage: try.sh OUT_DIR "set overrides" ["guide overrides"]
cd /home/jaehun/Desktop/vision-road;source /opt/ros/humble/setup.bash;source install/setup.bash
OUT=$1;SET=$2;GUIDE=${3-offset_at_m=1.4096 change_without_neighbour=True trust_single_line=True on_m=0.25 off_m=0.20}
G=results/town05_30hz/gt_loop_20261119;mkdir -p $OUT
run(){ for s in "$@";do
  python3 tools/gt_eval.py predict $s $OUT/$s ${SET:+--set $SET} --guide $GUIDE > $OUT/$s.predict.log 2>&1
  for f in $G/round_00/$s/yolo*.jsonl.gz;do [ -s $f ] && ln -sf $(realpath $f) $OUT/$s/;done
  ln -sf $(realpath $G/paint_stops/$s/paint_stops.jsonl.gz) $OUT/$s/paint_stops.jsonl.gz
  python3 tools/gt_eval.py metrics $s $OUT/$s --tag _1920 --sign sign_min_height_px=50 light_max_offcentre=0.3 > $OUT/$s.metrics.log 2>&1;done; }
run s1_lanefollow s2_wobble s3a_zigzag_cross s3b_zigzag_inlane & run s4_straddle s5_lanechange s6_yellow s7_lights & run s8a_lanefollow_20kmh s8b_lanefollow_50kmh s8c_lanechange_20kmh s8d_lanechange_50kmh & wait
python3 $G/scripts/summary.py $OUT --tag _1920 > $OUT/table.md;echo done
