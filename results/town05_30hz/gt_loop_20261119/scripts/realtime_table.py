"""Plan 30 D: one line per real-time run (lane latency, YOLO rate and image->detection time).  usage: realtime_table.py DIR..."""
import glob, json, os, sys
import numpy as np
for base in sys.argv[1:]:
    for d in sorted(glob.glob(base.rstrip('/')+'/*_[123]')):
        if not os.path.isdir(d):continue
        try:s = json.load(open(d+'/summary.json'))
        except Exception as e:print(d, 'no summary', e);continue
        L = s['latency_ms'];row = [d, 'p2d med/p95', L['publish_to_display']['median'], L['publish_to_display']['p95'], 'p2r p95', L['publish_to_receive']['p95'],
                                    's2f p95', L['start_to_finish']['p95'], 'proc', s['processed_ratio'], 'gaps>100', s['display_gaps_over_100ms']]
        f = d+'/signs_stats.json'
        if os.path.exists(f):
            y = json.load(open(f))['frames'];st = np.array([x['start'] for x in y]);en = np.array([x['end'] for x in y]);rc = np.array([x['received'] for x in y])
            row += ['yolo fps', round(len(y)/(en[-1]-st[0]), 2), 'infer p95', round(float(np.percentile((en-st)*1e3, 95)), 1), 'recv->det p95', round(float(np.percentile((en-rc)*1e3, 95)), 1)]
        print(*row)
