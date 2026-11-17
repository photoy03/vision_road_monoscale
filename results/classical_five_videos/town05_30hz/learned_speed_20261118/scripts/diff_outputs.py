#!/usr/bin/env python3
"""How much the learned gate changes the outputs (offline, Town05): per kind (own boundaries, outer lines),
frames where the set of shown sides differs, and frames where a side shown in both moved more than 5 / 20 cm
(lateral, compared at 8 m, or at the nearest shared distance)."""
import gzip,json,sys
import numpy as np
from pathlib import Path
D=Path(__file__).resolve().parent.parent/'offline'
a=[json.loads(l) for l in gzip.open(D/'off_town05.jsonl.gz','rt')];b=[json.loads(l) for l in gzip.open(D/'on_town05.jsonl.gz','rt')]
res={}
for kind in ('bounds','outer'):
    sets=0;moved5=0;moved20=0;exact=0;both=0;n=len(a)
    for x,y in zip(a,b):
        X={l['side']:np.array(l['points']) for l in x[kind]};Y={l['side']:np.array(l['points']) for l in y[kind]}
        if X.keys()!=Y.keys():sets+=1
        same=X.keys()==Y.keys() and all(np.array_equal(X[s],Y[s]) for s in X);exact+=same
        worst=0.
        for s in X.keys()&Y.keys():
            lo=max(X[s][0,0],Y[s][0,0]);hi=min(X[s][-1,0],Y[s][-1,0])
            if hi<lo:worst=max(worst,9.);continue
            q=min(max(8.,lo),hi);worst=max(worst,abs(np.interp(q,X[s][:,0],X[s][:,1])-np.interp(q,Y[s][:,0],Y[s][:,1])))
        if X.keys()&Y.keys():both+=1;moved5+=worst>.05;moved20+=worst>.2
    res[kind]=dict(frames=n,identical=round(exact/n,4),side_set_differs=round(sets/n,4),moved_over_5cm=round(moved5/n,4),moved_over_20cm=round(moved20/n,4))
print(json.dumps(res,indent=1));json.dump(res,open(D.parent/'metrics'/'output_diff.json','w'),indent=1)
