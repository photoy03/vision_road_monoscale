"""Half-resolution camera (960x540) against full resolution, same code and settings, per side:
boundary covering x (observed) at 10/20/25 m, median far end of observed paint, frame-to-frame std at 10/20 m
(cm, changes <=0.3 m), and |half - full| boundary at 10/20/25 m (p50/p95 cm, frames where both cover x).
usage: half_quality.py FULL_RUN HALF_RUN"""
import sys,gzip,json,numpy as np
F,H=[[json.loads(l) for l in gzip.open(p+'/predictions.jsonl.gz','rt')] for p in sys.argv[1:3]]
def line(r,side):
    b=[np.asarray(q['points']) for q in r['boundaries'] if q['side']==side and not q['reprojected']]
    return max(b,key=lambda p:np.ptp(p[:,0])) if b else None
def at(p,x):
    if p is None:return None
    p=p[np.argsort(p[:,0])];return float(np.interp(x,p[:,0],p[:,1])) if p[0,0]<=x<=p[-1,0] else None
def reach(r,side):
    v=[np.asarray(l['points'])[:,0].max() for l in r['lanes'] if l['side']==side and not l['reprojected']];return max(v) if v else None
for side in ('left','right'):
    for name,R in (('full',F),('half',H)):
        cov={x:np.mean([at(line(r,side),x) is not None for r in R]) for x in (10,20,25)}
        rch=[v for v in (reach(r,side) for r in R) if v is not None]
        std={}
        for x in (10,20):
            y=[at(line(r,side),x) for r in R];d=np.array([b-a for a,b in zip(y[:-1],y[1:]) if a is not None and b is not None]);d=d[np.abs(d)<=.3]
            std[x]=round(float(np.std(d))*100,2)
        print(f'{side:5s} {name}: covered 10/20/25 m {cov[10]:.3f}/{cov[20]:.3f}/{cov[25]:.3f}  paint reach median {np.median(rch):.1f} m  std 10/20 m {std[10]}/{std[20]} cm')
    for x in (10,20,25):
        d=[abs(a-b) for a,b in ((at(line(f,side),x),at(line(h,side),x)) for f,h in zip(F,H)) if a is not None and b is not None]
        print(f'      |half-full| at {x} m: p50 {np.percentile(d,50)*100:.1f} p95 {np.percentile(d,95)*100:.1f} cm ({len(d)} frames)')
