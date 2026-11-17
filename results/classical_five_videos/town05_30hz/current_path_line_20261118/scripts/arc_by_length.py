#!/usr/bin/env python3
"""Plan 24: how far ahead the current-path arc (fake steering, plan 23) still matches where the car went.
Same measurement as plan 23 evaluate23.py (lateral miss of the constant-steering arc against the car's pose
after travelling L metres), for L = 2..24 m.  Writes ../metrics/arc_by_length.json."""
import sys,json,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=Path('/home/jaehun/Desktop/vision-road')
P23=ROOT/'results/classical_five_videos/town05_30hz/vehicle_state_20261118/scripts'
sys.path.insert(0,str(P23));sys.path.insert(0,str(ROOT/'tools'))
import scenarios23 as S  # noqa: E402
from fake_vehicle_state import VehicleStateEstimator,WHEELBASE_M  # noqa: E402
LS=[2.,4.,6.,8.,10.,12.,16.,20.,24.]
def to_base(p,pose):
    c,s=math.cos(pose[2]),math.sin(pose[2]);dx,dy=p[0]-pose[0],p[1]-pose[1];return c*dx+s*dy,-s*dx+c*dy
def arc_y(x,k):
    if abs(k)<1e-6:return 0.
    r=1/k;q=r*r-x*x
    return r-math.copysign(math.sqrt(q),r) if q>0 else r
rows=S.load();out={}
for name in ('wobble','yellow','dashed','change','none'):
    fr=S.frames(name,rows);est=VehicleStateEstimator();st=[est.update(t,*pose) for t,pose,*_ in fr]
    poses=np.array([f[1] for f in fr]);dist=np.r_[0.,np.cumsum(np.hypot(np.diff(poses[:,0]),np.diff(poses[:,1])))]
    miss={L:[] for L in LS}
    for i,e in enumerate(st):
        if i<10 or abs(e.speed_mps)<2.:continue
        k=math.tan(e.steering_rad)/WHEELBASE_M
        for L in LS:
            j=int(np.searchsorted(dist,dist[i]+L))
            if j>=len(fr):continue
            xa,ya=to_base(poses[j],poses[i]);miss[L].append(ya-arc_y(xa,k))
    out[name]={str(L):dict(n=len(v),rms_cm=round(float(np.sqrt(np.mean(np.square(v))))*100,1),p95_cm=round(float(np.percentile(np.abs(v),95))*100,1)) for L,v in miss.items() if v}
    print(name,' '.join(f"{L:g}m {out[name][str(L)]['rms_cm']}/{out[name][str(L)]['p95_cm']}" for L in LS if str(L) in out[name]),flush=True)
json.dump(out,open(HERE.parent/'metrics'/'arc_by_length.json','w'),indent=1)
