"""Display filter stages vs paint (Town04 curves): the display fit of this frame's boundary alone
(no smoothing, quadratic in 5/10/15 m stations, optional near weighting) vs the paint, p95 cm at 6/8/10/15/20 m.
usage: display_fit_gap.py RUN_DIR [fit_near_m]"""
import sys,gzip,json,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT/'src/vision_road'))
from vision_road.display_filter import fit,evaluate
near=float(sys.argv[2]) if len(sys.argv)>2 else 0.
rows=[json.loads(l) for l in gzip.open(sys.argv[1]+'/predictions.jsonl.gz','rt')];t0=rows[0]['stamp'];XS=(6,8,10,15,20)
def at(p,x):p=np.asarray(p);return float(np.interp(x,p[:,0],p[:,1])) if p[0,0]<=x<=p[-1,0] else None
for a,b in ((15,20),(20,25),(50,55),(30,35)):
    line=f'{a}-{b}'
    for side in ('left','right'):
        v={x:[] for x in XS}
        for r in rows:
            if not a<=r['stamp']-t0<b:continue
            bs=[np.asarray(q['points']) for q in r['boundaries'] if q['side']==side and not q['reprojected']]
            if not bs:continue
            p=max(bs,key=lambda p:np.ptp(p[:,0]))
            if np.ptp(p[:,0])<3:continue
            st,lo,hi=fit(p,near=near)
            lanes=[l['points'] for l in r['lanes'] if l['side']==side and not l['reprojected']]
            for x in XS:
                if not lo<=x<=hi:continue
                yd=float(evaluate(st,np.array([x]))[0]);ys=[y for y in (at(l,x) for l in lanes) if y is not None]
                if ys:v[x].append(min(abs(y-yd) for y in ys))
        line+=f' {side[0].upper()} '+'/'.join(f'{np.percentile(v[x],95)*100:.0f}' if v[x] else '-' for x in XS)
    print(line)
