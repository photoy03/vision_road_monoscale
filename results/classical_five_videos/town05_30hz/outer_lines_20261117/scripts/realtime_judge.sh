#!/bin/bash
# Plan 19 real-time runs (Town05): a same-day baseline with outer lines off (carla.yaml switched for that
# run only, then restored), three judge runs with them on, one recorded run with the topic listener.
cd /home/jaehun/Desktop/vision-road;source /opt/ros/humble/setup.bash;source install/setup.bash
R=results/classical_five_videos/town05_30hz/outer_lines_20261117/runs/realtime;Y=src/vision_road/config/carla.yaml
for i in 1 2 3;do python3 tools/realtime_eval.py --video town05 --output $R/judge${i}_town05 --launch-arg display_filter_inline:=true > $R/judge${i}_town05.log 2>&1;echo judge$i $?;done
python3 results/classical_five_videos/town05_30hz/outer_lines_20261117/scripts/outer_listener.py $R/record_town05_topics.json 110 > $R/record_town05_listener.log 2>&1 &
python3 tools/realtime_eval.py --video town05 --record --output $R/record_town05 --launch-arg display_filter_inline:=true > $R/record_town05.log 2>&1;echo record $?
wait;cat $R/record_town05_listener.log
