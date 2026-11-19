#!/bin/bash
# Plan 30 (D4): YOLO on the top-centre 640 tile for every scene, next to round_00's full-frame results (yolo_tile.jsonl.gz).
cd /home/jaehun/Desktop/vision-road;R=results/town05_30hz/gt_loop_20261119/round_00;Y=$HOME/yolo_env/bin/python
until [ -f $R/s8d_lanechange_50kmh/metrics.json ];do sleep 20;done
for s in s1_lanefollow s2_wobble s3a_zigzag_cross s3b_zigzag_inlane s4_straddle s5_lanechange s6_yellow s7_lights s8a_lanefollow_20kmh s8b_lanefollow_50kmh s8c_lanechange_20kmh s8d_lanechange_50kmh;do
  [ -f $R/$s/yolo_tile.jsonl.gz ] || $Y tools/gt_eval.py yolo $s $R/$s --full-frame 0 --imgsz 640 --tag _tile > $R/$s.yolo_tile.log 2>&1;echo "$s $(tail -1 $R/$s.yolo_tile.log)"
done
