#!/bin/bash
# Plan 30 (D4): YOLO on the full frame at imgsz 1920 for every scene (yolo_1920.jsonl.gz in round_00).
cd /home/jaehun/Desktop/vision-road;R=results/town05_30hz/gt_loop_20261119/round_00;Y=$HOME/yolo_env/bin/python
for s in s1_lanefollow s7_lights s8b_lanefollow_50kmh s8a_lanefollow_20kmh s2_wobble s3a_zigzag_cross s3b_zigzag_inlane s4_straddle s5_lanechange s6_yellow s8c_lanechange_20kmh s8d_lanechange_50kmh;do
  [ -f $R/$s/yolo_1920.jsonl.gz ] || $Y tools/gt_eval.py yolo $s $R/$s --full-frame 1 --imgsz 1920 --tag _1920 > $R/$s.yolo_1920.log 2>&1;echo "$s $(tail -1 $R/$s.yolo_1920.log)"
done
