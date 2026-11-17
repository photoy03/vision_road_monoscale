#!/usr/bin/env python3
"""One line per real-time run (tools/realtime_eval.py output folders under DIR) and the pass/fail against a p95
limit: processed share, publish->display p95/mean/max, display gaps over 100 ms, and the stage means/p95
(image delivery publish->receive, BEV warp, detection, display filter, learned mask).
usage: latency_summary.py DIR [--limit MS]"""
import argparse,json,re
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('dir',type=Path);ap.add_argument('--limit',type=float,default=25.);a=ap.parse_args()
runs=sorted(p.parent for p in a.dir.glob('*/summary.json'))
if not runs:raise SystemExit(f'no runs (*/summary.json) under {a.dir}')
print(f"{'run':20s} {'processed':>9s} {'p95':>6s} {'mean':>6s} {'max':>6s} {'gaps':>4s} | {'delivery':>11s} {'warp':>9s} {'detect':>11s} {'display':>9s} {'learned':>9s}  pass(p95<={a.limit:g}, 100%, no gaps)")
groups={}
for d in runs:
    s=json.loads((d/'summary.json').read_text());l=s['latency_ms']['publish_to_display'];r=s['latency_ms']['publish_to_receive'];ph=s['phases_ms']
    f=lambda k:f"{ph[k]['mean']:4.1f}/{ph[k]['p95']:4.1f}" if k in ph else '    -    '
    ok=l['p95']<=a.limit and s['processed_ratio']>=1. and s['display_gaps_over_100ms']==0
    print(f"{d.name:20s} {100*s['processed_ratio']:8.2f}% {l['p95']:6.2f} {l['mean']:6.2f} {l['max']:6.1f} {s['display_gaps_over_100ms']:4d} | {r['mean']:4.1f}/{r['p95']:5.1f} {f('worker.warp')} {f('worker.detect'):>11s} {f('worker.display')} {f('worker.learned')}  {'pass' if ok else 'FAIL'}")
    groups.setdefault(re.sub(r'\d+','#',d.name),[]).append(ok)
print('stages: mean/p95 ms; delivery = replay publish -> perception worker receives the image')
for g,v in groups.items():print(f'{g}: {sum(v)}/{len(v)} runs pass')
