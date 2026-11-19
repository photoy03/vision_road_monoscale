#!/bin/bash
# Plan 30: one evaluation round over the GT scenes: predict (detector + markings + guidance, offline), yolo (signs,
# YOLO environment), metrics.  usage: eval_round.sh ROUND_DIR [--no-yolo] [scene ...]   (waits for each scene's cache)
# Settings: SET="name=value ..." (detector), GUIDE="name=value ..." (LaneGuidance, default: the launch's values);
# YOLO_FROM=DIR links that round's yolo files; TAG (yolo file for the metrics, default _1920) and SIGN (SignSettings,
# default: the launch's bridge values) for the metrics.  The detector settings come from carla.yaml.
cd /home/jaehun/Desktop/vision-road;source /opt/ros/humble/setup.bash;source install/setup.bash
GUIDE=${GUIDE-offset_at_m=1.4096 change_without_neighbour=True trust_single_line=True on_m=0.25 off_m=0.20}
TAG=${TAG-_1920};SIGN=${SIGN-sign_min_height_px=50 light_max_offcentre=0.3}
OUT=$1;shift;YOLO=1;[ "${1:-}" = --no-yolo ] && { YOLO=0;shift; }
C=/media/jaehun/38F7-677B/vision-road-cache;Y=${YOLO_PYTHON:-$HOME/yolo_env/bin/python};mkdir -p $OUT
for s in ${@:-s1_lanefollow s2_wobble s3a_zigzag_cross s3b_zigzag_inlane s4_straddle s5_lanechange s6_yellow s7_lights s8a_lanefollow_20kmh s8b_lanefollow_50kmh s8c_lanechange_20kmh s8d_lanechange_50kmh};do
  n=town05_${s}_20261119;until [ -f $C/${n}_offline/frames.json ] && [ -f $C/${n}_native.cdrz.json ];do sleep 20;done
  python3 tools/gt_eval.py predict $s $OUT/$s ${SET:+--set $SET} ${GUIDE:+--guide $GUIDE} > $OUT/$s.predict.log 2>&1
  if [ -n "${YOLO_FROM:-}" ];then for f in $YOLO_FROM/$s/yolo*.jsonl.gz;do [ -e "$f" ] && ln -sf $(realpath $f) $OUT/$s/;done;fi
  if [ $YOLO = 1 ] && [ ! -f $OUT/$s/yolo.jsonl.gz ];then $Y tools/gt_eval.py yolo $s $OUT/$s > $OUT/$s.yolo.log 2>&1;fi
  # The painted stop lines (semantic camera) once per scene: shared by all rounds.
  P=results/town05_30hz/gt_loop_20261119/paint_stops/$s;[ -f $P/paint_stops.jsonl.gz ] || python3 tools/gt_eval.py paint_stops $s $P > $P.log 2>&1
  ln -sf $(realpath $P/paint_stops.jsonl.gz) $OUT/$s/paint_stops.jsonl.gz
  python3 tools/gt_eval.py metrics $s $OUT/$s ${TAG:+--tag $TAG} ${SIGN:+--sign $SIGN} > $OUT/$s.metrics.log 2>&1;echo "$s done"
done
