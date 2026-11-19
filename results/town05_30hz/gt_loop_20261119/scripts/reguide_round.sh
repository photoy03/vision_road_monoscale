#!/bin/bash
# Plan 30: guidance-only round: re-run LaneGuidance (overrides after --) on SOURCE's centres, link its yolo files and
# the painted stop lines, then the metrics.  usage: reguide_round.sh SOURCE_ROUND OUT_ROUND "scenes" [guide overrides ...]
cd /home/jaehun/Desktop/vision-road;source /opt/ros/humble/setup.bash;source install/setup.bash
SRC=$1;OUT=$2;SC=$3;shift 3;mkdir -p $OUT
for s in $SC;do
  python3 tools/gt_eval.py reguide $s $OUT/$s --source $SRC/$s --guide "$@" > $OUT/$s.reguide.log 2>&1 || { echo "$s reguide failed";continue; }
  for f in $SRC/$s/yolo*.jsonl.gz;do [ -e "$f" ] && ln -sf $(realpath $f) $OUT/$s/;done
  P=results/town05_30hz/gt_loop_20261119/paint_stops/$s/paint_stops.jsonl.gz;[ -f $P ] && ln -sf $(realpath $P) $OUT/$s/
  python3 tools/gt_eval.py metrics $s $OUT/$s > $OUT/$s.metrics.log 2>&1;echo "$s done"
done
