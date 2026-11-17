#!/usr/bin/env python3
"""Detector boundaries (offline predictions): direction change against the previous frame's boundary carried by the
poses, over the part both cover beyond 8 m (at least 1 m) (the slope of each over that part, mrad), and the largest lateral
offset of a 1-hit paint fragment from the boundary built through it.  Straight = |yaw rate| < 2 deg/s.
usage: bnd_heading.py maploop|town05"""
import sys,gzip,json,math
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from ends_eval import PRED,CACHE,pose_at,relative,carry,at
video=sys.argv[1]
rows=[json.loads(l) for l in gzip.open(PRED[video],'rt')];t0=rows[0]['stamp']
poses=np.array(json.load(open(CACHE[video]))['poses']);poses[:,3]=np.unwrap(poses[:,3])
P=[pose_at(poses,r['stamp']) for r in rows]
def best(r,side):
    b=[np.asarray(q['points'],float) for q in r['boundaries'] if q['side']==side and not q['reprojected']]
    b=max(b,key=lambda p:np.ptp(p[:,0]),default=None)
    return None if b is None else b[np.argsort(b[:,0])]
for side in ('left','right'):
    rec=[]
    for i in range(1,len(rows)):
        c,p=best(rows[i],side),best(rows[i-1],side)
        if c is None or p is None:continue
        m=relative(P[i-1],P[i]);pc=carry(p,m);pc=pc[np.argsort(pc[:,0])]
        lo,hi=max(8.,c[0,0],pc[0,0]),min(c[-1,0],pc[-1,0])
        if hi-lo<1:continue
        sc=(at(c,hi)-at(c,lo))/(hi-lo);sp=(at(pc,hi)-at(pc,lo))/(hi-lo)
        yaw=abs(m[2])/(rows[i]['stamp']-rows[i-1]['stamp'])
        rec.append((rows[i]['stamp']-t0,abs(math.atan(sc)-math.atan(sp))*1e3,math.degrees(yaw)<2,hi-lo))
    a=np.array([r[1] for r in rec]);st=np.array([r[2] for r in rec])
    print(f'{video} {side}: frames {len(a)} straight {st.sum()} | dir change p50 {np.median(a):.1f} p95 {np.percentile(a,95):.1f} mrad;'
          f' straight p95 {np.percentile(a[st],95):.1f}, >50 mrad {(a[st]>50).sum()}, >100 {(a[st]>100).sum()}, >200 {(a[st]>200).sum()}')
    top=sorted([r for r in rec if r[2]],key=lambda r:-r[1])[:8]
    print('   largest (straight):',[(round(t,2),round(v)) for t,v,_,_ in top])
    # 1-hit fragments joined into the boundary far from the line through the rest
    odd=[]
    for r in rows:
        b=best(r,side)
        if b is None:continue
        frags=[(np.asarray(l['points'],float),l['hits']) for l in r['lanes'] if l['side']==side and not l['reprojected']]
        solid=[f for f,h in frags if h>=3]
        for f,h in frags:
            if h>1 or not solid:continue
            xm=float(f[:,0].mean())
            if not b[0,0]<=xm<=b[-1,0]:continue
            if abs(at(b,xm)-float(f[:,1].mean()))>.3:continue  # not on the boundary
            ref=np.vstack(solid);ref=ref[np.argsort(ref[:,0])]
            if ref[-1,0]-ref[0,0]<1.5:continue
            c=np.polyfit(ref[:,0],ref[:,1],1 if ref[-1,0]-ref[0,0]<8 else 2);off=abs(np.polyval(c,xm)-float(f[:,1].mean()))
            if off>.5:odd.append((round(r['stamp']-t0,2),round(xm,1),round(off,2)))
    print(f'   1-hit fragments on the boundary > 0.5 m off the line through fragments with >= 3 hits: {len(odd)} frames',odd[:12])
