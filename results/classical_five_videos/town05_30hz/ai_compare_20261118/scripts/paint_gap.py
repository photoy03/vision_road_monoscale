#!/usr/bin/env python3
"""Own boundary vs its paint (dump_lines.py records): per frame and side, the median lateral distance from the
side's measured paint points (lanes, not carried over) to the boundary, where the boundary covers them; also the
share of frames with both own boundaries, and the per-10 m wobble is left to outer_eval.py.
usage: paint_gap.py VIDEO NAME...   prints mean / p95 (cm) per side"""
import sys,json,gzip
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent.parent/'runs'
for name in sys.argv[2:]:
    rows=[json.loads(l) for l in gzip.open(D/f'{name}_{sys.argv[1]}.jsonl.gz','rt')];out={}
    both=np.mean([len({l['side'] for l in r['bounds']})==2 for r in rows]);anyl=np.mean([len(r['bounds'])>0 for r in rows])
    outer=np.mean([len(r.get('outer',[]))>0 for r in rows]);centre=np.mean([any((c.get('name')=='ego') for c in r.get('centres',[])) for r in rows])
    for side in ('left','right'):
        g=[]
        for r in rows:
            b=next((np.array(l['points']) for l in r['bounds'] if l['side']==side and not l['rep']),None)
            if b is None:continue
            b=b[np.argsort(b[:,0])];pts=[np.array(l['points']) for l in r['lanes'] if l['side']==side and not l['rep']]
            if not pts:continue
            p=np.vstack(pts);s=(p[:,0]>=b[0,0])&(p[:,0]<=b[-1,0])
            if s.sum()>=3:g.append(float(np.median(np.abs(p[s,1]-np.interp(p[s,0],b[:,0],b[:,1])))))
        g=np.array(g)*100;out[side]=f'n {len(g)} mean {g.mean():.2f} p95 {np.percentile(g,95):.2f} cm'
    print(f'{sys.argv[1]} {name}: both own lines {100*both:.1f}% any own line {100*anyl:.1f}% outer line {100*outer:.1f}% ego centreline {100*centre:.1f}% | paint-boundary',out)
