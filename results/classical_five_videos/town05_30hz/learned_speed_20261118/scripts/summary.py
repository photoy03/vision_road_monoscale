#!/usr/bin/env python3
"""One table of the off / on / structure runs: publish -> display line p95 / mean / max, processed share, frames over
33 ms, phase means (the worker's warp / detect / display, road_node's learned mask hand-off and image hand-over to the
worker when the camera goes through road_node), image transfer (publish -> receive), and CPU per role (cores)."""
import json,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent.parent/'runs';rows=[]
for d in sorted(p for p in R.iterdir() if (p/'summary.json').exists()):
    s=json.load(open(d/'summary.json'));f=json.load(open(d/'frames.json'))
    lat=np.array([(r['display']-r['publish'])/1e6 for r in f if r.get('display') and r.get('publish')])
    ph=s['phases_ms'];l=s['latency_ms']
    cpu=json.load(open(R/f'{d.name}_cpu.json')) if (R/f'{d.name}_cpu.json').exists() else {}
    g=cpu.get('groups',{})
    row=dict(run=d.name,p95=l['publish_to_display']['p95'],mean=l['publish_to_display']['mean'],max=l['publish_to_display']['max'],processed=s['processed_ratio'],
             over33=round(float(np.mean(lat>33.)),4),transfer=l['publish_to_receive']['mean'],
             phases={k:round(v['mean'],2) for k,v in ph.items()},cpu_cores={k:round(v/100,2) for k,v in sorted(g.items())},
             pipeline_cores=cpu.get('pipeline_threads'),system_cores=cpu.get('system_threads_used'),gpu=cpu.get('gpu'))
    rows.append(row)
json.dump(rows,open(R.parent/'metrics'/'runs.json','w'),indent=1)
for r in rows:
    print(f"{r['run']:11s} p95 {r['p95']:6.2f} mean {r['mean']:6.2f} max {r['max']:6.2f} proc {r['processed']} >33ms {r['over33']:.4f} transfer {r['transfer']:.2f} | cores {r['cpu_cores']} pipeline {r['pipeline_cores']} system {r['system_cores']}")
    print(' '*12+'phases '+' '.join(f'{k}:{v}' for k,v in r['phases'].items())+(f" | gpu util {r['gpu']['util_mean']}% max {r['gpu']['util_max']} mem {r['gpu']['mem_used_mib_mean']}/{r['gpu']['mem_total_mib']} MiB apps {r['gpu']['apps']}" if r['gpu'] else ''))
