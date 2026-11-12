#!/usr/bin/env bash
# usage: learned_run.sh OUTDIR [launch args...] — one 1x replay, performance report, lane and marking topics recorded.
cd /home/jaehun/Desktop/vision-road
source /opt/ros/humble/setup.bash; source install/setup.bash
out="$1"; shift; mkdir -p "$out"
setsid ./run_maploop.sh performance_report:="$out/performance.json" "$@" > "$out/launch.log" 2>&1 < /dev/null &
launch=$!
/usr/bin/python3 /tmp/claude-1000/-home-jaehun-Desktop-vision-road/29db546f-08e5-4fc6-9d69-1be1602904b8/scratchpad/topic_rec3.py "$out/topics.json" > "$out/recorder.log" 2>&1 &
rec=$!
sleep 15
./replay_maploop.sh --rate 1 > "$out/replay.log" 2>&1
sleep 3
kill -INT "$rec"; wait "$rec"
children=$(ps -o pid= --ppid "$launch")
for c in $children; do kill -TERM "$c" 2>/dev/null; done
for i in $(seq 1 40); do [ -f "$out/performance.json" ] && break; sleep 1; done
kill -TERM "$launch" 2>/dev/null; sleep 2
cat "$out/recorder.log"; grep -i "learned lane\|error" "$out/launch.log" | grep -v Traceback | head -5
