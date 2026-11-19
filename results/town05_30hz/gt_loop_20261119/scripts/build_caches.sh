#!/bin/bash
# Plan 30: lossless image caches (replay) and offline BEV caches (evaluation) for the new Town05 scenes, on the SSD's
# vision-road-cache/.  The bags and sidecars are only read.
cd /home/jaehun/Desktop/vision-road;source /opt/ros/humble/setup.bash;source install/setup.bash
D=/media/jaehun/38F7-677B;C=$D/vision-road-cache;L=results/town05_30hz/gt_loop_20261119/logs
for s in ${@:-s1_lanefollow s2_wobble s3a_zigzag_cross s3b_zigzag_inlane s4_straddle s5_lanechange s6_yellow s7_lights s8a_lanefollow_20kmh s8b_lanefollow_50kmh s8c_lanechange_20kmh s8d_lanechange_50kmh};do
  n=town05_${s}_20261119;bag=$(ls $D/rosbag/$n/*.db3)
  [ -f $C/${n}_native.cdrz.json ] || python3 tools/cache_maploop_cdr.py --bag $bag --output $C/${n}_native.cdrz > $L/cdrz_$s.log 2>&1
  [ -f $C/${n}_offline/frames.json ] || python3 tools/maploop_offline_compare.py cache --cache $C/${n}_offline --bag $bag --cache-file $C/${n}_native.cdrz > $L/offline_$s.log 2>&1
  echo "$s done $(date +%T)"
done
[ -f $C/town05_signs_native.cdrz.json ] || python3 tools/cache_maploop_cdr.py --bag $D/rosbag/town05_signs/town05_signs_0.db3 --output $C/town05_signs_native.cdrz > $L/cdrz_signs.log 2>&1
echo all done
