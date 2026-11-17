#!/bin/bash
# Plan 20 real-time runs (Town05): three judge runs, then one recorded run with the topic listener.
cd /home/jaehun/Desktop/vision-road;source /opt/ros/humble/setup.bash;source install/setup.bash
R=results/classical_five_videos/town05_30hz/centrelines_20261118/runs/realtime;P=${1:-}
for i in 1 2 3;do python3 tools/realtime_eval.py --video town05 --output $R/${P}judge${i}_town05 --launch-arg display_filter_inline:=true > $R/${P}judge${i}_town05.log 2>&1;echo judge$i $?;done
python3 results/classical_five_videos/town05_30hz/centrelines_20261118/scripts/outer_listener.py $R/${P}record_town05_topics.json 110 > $R/${P}record_town05_listener.log 2>&1 &
python3 tools/realtime_eval.py --video town05 --record --output $R/${P}record_town05 --launch-arg display_filter_inline:=true > $R/${P}record_town05.log 2>&1;echo record $?
wait;cat $R/${P}record_town05_listener.log
