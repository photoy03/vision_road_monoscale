#!/usr/bin/env python3
"""One line per real-time run (runs/realtime/*/summary.json): processed share, p95/mean/max publish->display,
drift, display gaps > 100 ms, worker.detect mean/p95, out.markers mean/p95 (the outer topics go out there)."""
import json,sys
from pathlib import Path
R=Path(__file__).resolve().parent.parent/'runs'/'realtime'
for d in sorted(p for p in R.iterdir() if (p/'summary.json').exists()):
    s=json.load(open(d/'summary.json'));l=s['latency_ms']['publish_to_display'];ph=s['phases_ms']
    print(f"{d.name:16s} processed {s['processed_ratio']} p95 {l['p95']} mean {l['mean']} max {l['max']} drift {s['latency_drift_ms_last_vs_first_tenth']} "
          f"gaps>100 {s['display_gaps_over_100ms']} detect {ph['worker.detect']['mean']}/{ph['worker.detect']['p95']} markers {ph['out.markers']['mean']}/{ph['out.markers']['p95']}")
