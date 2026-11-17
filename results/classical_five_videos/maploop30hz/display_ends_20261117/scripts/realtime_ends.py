#!/usr/bin/env python3
"""The ends metric on live runs (realtime_eval.py frames.json display_lines): frame to frame with the vehicle
motion removed (rear-axle poses of the offline cache), counts of 0 m > 5 cm, drawn far end > 20 cm, 10 m > 10 cm.
usage: realtime_ends.py RUN_DIR ..."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from ends_eval import CACHE,pose_at,relative,carry,at
for run in sys.argv[1:]:
    video='town05' if 'town05' in run else 'maploop'
    poses=np.array(json.load(open(CACHE[video]))['poses']);poses[:,3]=np.unwrap(poses[:,3])
    stamps=sorted(r['stamp'] for r in json.load(open(run+'/source_timing.json'))['records'])
    fr=json.load(open(run+'/frames.json'));P=[pose_at(poses,t) for t in stamps]
    line=f'{run.rstrip("/").split("/")[-1]:24s}'
    for side in ('left','right'):
        c0=cf=c10=n=0;shown=0;from0=0
        L=[{s:np.asarray(p) for s,_,p in f.get('display_lines',[])} for f in fr]
        for i in range(len(L)):
            if side in L[i]:shown+=1;from0+=L[i][side][0,0]<=.05
            if i==0 or side not in L[i] or side not in L[i-1]:continue
            cur=L[i][side];pc=carry(L[i-1][side],relative(P[i-1],P[i]));n+=1
            a,b=at(cur,0.),at(pc,0.);c0+=a is not None and b is not None and abs(a-b)>.05
            xf=min(cur[-1,0],pc[-1,0]);a,b=at(cur,xf),at(pc,xf);cf+=a is not None and b is not None and abs(a-b)>.2
            a,b=at(cur,10.),at(pc,10.);c10+=a is not None and b is not None and abs(a-b)>.1
        line+=f' {side[0]}: y0>5 {c0:3d} far>20 {cf:2d} y10>10 {c10} shown {shown} from0 {from0/max(1,shown):.3f} |'
    print(line)
