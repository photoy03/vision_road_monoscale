"""Boundary kinks: an observed boundary polyline whose direction turns by more than ANGLE degrees
between neighbouring stretches (each at least 0.5 m of polyline, measured over 1 m on each side),
beyond 8 m.  Lists frames, side, where, angle.
usage: kinks.py RUN_DIR [ANGLE]"""
import sys,gzip,json,math,numpy as np
angle=float(sys.argv[2]) if len(sys.argv)>2 else 25.
rows=[json.loads(l) for l in gzip.open(sys.argv[1]+'/predictions.jsonl.gz','rt')];t0=rows[0]['stamp'];found=[]
for r in rows:
    for b in r['boundaries']:
        if b['reprojected']:continue
        p=np.asarray(b['points']);p=p[np.argsort(p[:,0])]
        if np.ptp(p[:,0])<3:continue
        s=np.r_[0,np.cumsum(np.hypot(np.diff(p[:,0]),np.diff(p[:,1])))];L=s[-1]
        worst=None
        for k in np.arange(1.,L-1.,.25):
            a=np.array([np.interp(k-1,s,p[:,0]),np.interp(k-1,s,p[:,1])]);m=np.array([np.interp(k,s,p[:,0]),np.interp(k,s,p[:,1])]);c=np.array([np.interp(k+1,s,p[:,0]),np.interp(k+1,s,p[:,1])])
            if m[0]<8:continue
            d1=m-a;d2=c-m;turn=abs(math.degrees(math.atan2(d1[0]*d2[1]-d1[1]*d2[0],d1@d2)))
            if worst is None or turn>worst[0]:worst=(turn,float(m[0]),float(m[1]))
        if worst and worst[0]>angle:found.append((r['frame'],round(r['stamp']-t0,2),b['side'],round(worst[1],1),round(worst[2],2),round(worst[0])))
runs=[]
for f in found:
    if runs and f[0]-runs[-1][-1][0]<=3 and f[2]==runs[-1][-1][2]:runs[-1].append(f)
    else:runs.append([f])
print(f'{sys.argv[1].split("/")[-1]}: {len(found)} frames, {len(runs)} events (angle>{angle:.0f} deg, x>=8 m)')
for g in runs:print(f'  t {g[0][1]:5.2f}-{g[-1][1]:5.2f} frames {g[0][0]}-{g[-1][0]} {g[0][2]:5s} x {g[0][3]} y {g[0][4]} max angle {max(f[5] for f in g)}')
