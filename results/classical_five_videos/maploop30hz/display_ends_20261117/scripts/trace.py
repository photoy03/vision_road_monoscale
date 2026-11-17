#!/usr/bin/env python3
"""Per-frame state of one side's display track in a time window.  usage: trace.py VIDEO SIDE T0 T1 [name=value ...]"""
import sys,gzip,json
import numpy as np
sys.argv=[sys.argv[0]]+sys.argv[1:]
from ends_eval import PRED,CACHE,settings,at
from vision_road.display_filter import DisplayFilter,relative,evaluate
from vision_road.core import Timeline
video,side,a,b=sys.argv[1],sys.argv[2],float(sys.argv[3]),float(sys.argv[4]);s=settings(sys.argv[5:])
rows=[json.loads(l) for l in gzip.open(PRED[video],'rt')];t0=rows[0]['stamp']
odom=json.load(open(CACHE[video]))['poses'];f=DisplayFilter(s);tl=Timeline(.25);k=0;last=None
for r in rows:
    t=r['stamp']
    while k<len(odom) and odom[k][0]<=t+.1:tl.add(*odom[k]);k+=1
    pose=tl.at(t);m=(0.,0.,0.) if last is None else relative(last[1],pose);last=(t,pose)
    paint=[(l['side'],l['points']) for l in r['lanes'] if not l['reprojected']]
    f.update(t,[(q['side'],q['points'],q['reprojected']) for q in r['boundaries']],m,paint)
    if not a<=t-t0<=b:continue
    tr=f.sides[side];out={sd:p for sd,p,_,_ in f.output()}
    obs=[np.asarray(q['points']) for q in r['boundaries'] if q['side']==side and not q['reprojected']]
    o=max(obs,key=lambda p:np.ptp(p[:,0]),default=None)
    pn=sorted(round(float(np.asarray(l['points'])[:,0].min()),1) for l in r['lanes'] if l['side']==side and not l['reprojected'])
    d=out.get(side)
    print(f"{t-t0:6.3f} obs {'-' if o is None else f'{o[:,0].min():5.1f}-{o[:,0].max():5.1f} y@lo {o[np.argmin(o[:,0]),1]:6.2f}'} paint_near {pn} "
          f"acc {tr.history[-1] if tr.history else None} pred {tr.predicted} span {tuple(round(v,1) for v in tr.span()) if tr.span() else None} "
          f"mem {len(tr.memory)}{'' if not len(tr.memory) else f' x {tr.memory[:,0].min():.1f}-{tr.memory[:,0].max():.1f}'} "
          f"disp {'-' if d is None else f'{d[0,0]:.1f}-{d[-1,0]:.1f} y0 {d[0,1]:.2f} y5 {at(d,5.) if at(d,5.) is None else round(at(d,5.),2)} y10 {at(d,10.) if at(d,10.) is None else round(at(d,10.),2)}'}")
