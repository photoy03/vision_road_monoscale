#!/bin/bash
# Plan 30: the metrics of one round again (after a metric or paint_stops change), every scene, and its table.
# usage: recompute.sh ROUND_DIR [TAG [SIGN settings...]]   e.g. recompute.sh round_00 ""   /   recompute.sh round_01 _1920 sign_min_height_px=50 light_max_offcentre=0.3
cd /home/jaehun/Desktop/vision-road;source /opt/ros/humble/setup.bash;source install/setup.bash
R=$1;TAG=${2-};shift;shift;G=results/town05_30hz/gt_loop_20261119
for s in s1_lanefollow s2_wobble s3a_zigzag_cross s3b_zigzag_inlane s4_straddle s5_lanechange s6_yellow s7_lights s8a_lanefollow_20kmh s8b_lanefollow_50kmh s8c_lanechange_20kmh s8d_lanechange_50kmh;do
  [ -f $R/$s/predictions.jsonl.gz ] || continue
  ln -sf $(realpath $G/paint_stops/$s/paint_stops.jsonl.gz) $R/$s/paint_stops.jsonl.gz
  python3 tools/gt_eval.py metrics $s $R/$s ${TAG:+--tag $TAG} ${1:+--sign "$@"} > $R/$s.metrics.log 2>&1 || echo "$s metrics failed"
done
python3 $G/scripts/summary.py $R ${TAG:+--tag $TAG} > $R/table.md
