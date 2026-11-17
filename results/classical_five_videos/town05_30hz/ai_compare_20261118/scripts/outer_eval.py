#!/usr/bin/env python3
"""Plan 19 metrics from outer_run.py records.  usage: outer_eval.py VIDEO NAME

Per side (left/right):
  visible   frames with outer paint visible (outer_run.py definition)
  caught    of those, frames with an outer line shown (measured / any incl. carried over)
  extra     frames with a measured outer line but no visible paint by that definition
  gapvar    measured outer lines whose distance to the shown own line varies > 0.3 m over the
            outer line's observed part (max-min of the normal distance), share
  wobble    frame-to-frame lateral change at 10 m with the vehicle motion removed (rear-axle poses):
            outer vs own line of the same side, mean abs / p95 / rms (cm), both frames measured
  style     outer style shares (solid/dashed/unknown), yellow share, opposite share
per 5 s: visible %, caught %."""
import sys,json,gzip,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
video,name=sys.argv[1],sys.argv[2]
rows=[json.loads(l) for l in gzip.open(HERE.parent/'runs'/f'{name}_{video}.jsonl.gz','rt')]
poses=np.array(json.load(open(f'/media/jaehun/38F7-677B/vision-road-cache/{video}_offline/frames.json'))['poses']);poses[:,3]=np.unwrap(poses[:,3])


def pose(t):return np.array([np.interp(t,poses[:,0],poses[:,i]) for i in (1,2,3)])


def relative(a,b):
    c,s=math.cos(a[2]),math.sin(a[2]);wx,wy=b[0]-a[0],b[1]-a[1];return c*wx+s*wy,-s*wx+c*wy,b[2]-a[2]


def carry(p,m):
    dx,dy,dyaw=m;c,s=math.cos(dyaw),math.sin(dyaw);px=p[:,0]-dx;py=p[:,1]-dy
    return np.column_stack([c*px+s*py,-s*px+c*py])


def at(p,x):return float(np.interp(x,p[:,0],p[:,1])) if p[0,0]<=x<=p[-1,0] else None


def normal_gap(outer,own):
    """normal distance of each outer point to the own polyline (densified), ignoring points past its ends"""
    x=np.arange(own[0,0],own[-1,0],.1);Q=np.column_stack([x,np.interp(x,own[:,0],own[:,1])])
    if len(Q)<3:return np.array([])
    D=outer[:,None,:]-Q[None,:,:];n=np.argmin((D**2).sum(2),1);T=np.gradient(Q,axis=0);T/=np.linalg.norm(T,axis=1)[:,None]
    d=D[np.arange(len(outer)),n];along=np.abs((d*T[n]).sum(1));N=np.column_stack([-T[:,1],T[:,0]])
    return np.abs((d*N[n]).sum(1))[along<=.15]


P=[pose(r['stamp']) for r in rows];out={}
for side in ('left','right'):
    o=side+'_outer';vis=[];cm=[];ca=[];extra=0;gapvar=[];wob={'outer':[],'own':[]};styles=[];yel=[];opp=[]
    seg={}
    for i,r in enumerate(rows):
        own=r['own'].get(side);ou=r['outer'].get(o)
        visible=own is not None and not own['reprojected'] and r['paint'].get(side,0)>=2.
        measured=ou is not None and not ou['reprojected'];shown=ou is not None
        b=int(r['t']//5);s=seg.setdefault(b,[0,0,0])
        if visible:
            vis.append(i);s[0]+=1;s[1]+=measured;s[2]+=shown;cm.append(measured);ca.append(shown)
        elif measured:extra+=1
        if measured and own is not None:
            pts=np.array(ou['points']);lo,hi=ou['observed'];pts=pts[(pts[:,0]>=lo)&(pts[:,0]<=hi)]
            g=normal_gap(pts,np.array(own['points'])) if len(pts)>=2 else np.array([])
            if len(g)>=3:gapvar.append(float(g.max()-g.min()))
        if measured:styles.append(ou['style']);yel.append(ou['yellow']);opp.append(ou['opposite'])
        if i:
            m=relative(P[i-1],P[i])
            for kind,cur,prv in (('outer',ou,rows[i-1]['outer'].get(o)),('own',own,rows[i-1]['own'].get(side))):
                if cur is None or prv is None or cur['reprojected'] or prv['reprojected']:continue
                if kind=='outer' and (own is None):continue
                a=at(np.array(cur['points']),10.);bb=at(carry(np.array(prv['points']),m),10.)
                if a is not None and bb is not None:wob[kind].append(a-bb)
    w={k:dict(n=len(v),mean_cm=round(float(np.mean(np.abs(v)))*100,2) if v else None,p95_cm=round(float(np.percentile(np.abs(v),95))*100,2) if v else None,
              rms_cm=round(float(np.sqrt(np.mean(np.square(v))))*100,2) if v else None) for k,v in wob.items()}
    gv=np.array(gapvar)
    out[side]=dict(frames=len(rows),visible=len(vis),caught_measured=round(float(np.mean(cm)),4) if cm else None,caught_shown=round(float(np.mean(ca)),4) if ca else None,
                   extra_measured=extra,gapvar_n=len(gv),gapvar_over_0_3=round(float(np.mean(gv>.3)),4) if len(gv) else None,
                   gapvar_p95_m=round(float(np.percentile(gv,95)),3) if len(gv) else None,wobble=w,
                   style={k:round(styles.count(k)/max(1,len(styles)),3) for k in ('solid','dashed','unknown')},
                   yellow=round(float(np.mean(yel)),3) if yel else None,opposite=round(float(np.mean(opp)),3) if opp else None,
                   per5s={f'{b*5}-{b*5+5}':dict(visible=v[0],measured=round(v[1]/v[0],2) if v[0] else None,shown=round(v[2]/v[0],2) if v[0] else None) for b,v in sorted(seg.items())})
(HERE.parent/'metrics').mkdir(exist_ok=True)
(HERE.parent/'metrics'/f'{name}_{video}.json').write_text(json.dumps(out,indent=1))
for side,e in out.items():
    w=e['wobble']
    print(f"{video} {name} {side:5s} visible {e['visible']:4d}/{e['frames']} caught {e['caught_measured']} (shown {e['caught_shown']}) extra {e['extra_measured']} "
          f"gapvar>0.3 {e['gapvar_over_0_3']} (p95 {e['gapvar_p95_m']}) wobble10 outer {w['outer']['rms_cm']}/{w['outer']['p95_cm']} own {w['own']['rms_cm']}/{w['own']['p95_cm']} cm "
          f"style {e['style']} yellow {e['yellow']} opp {e['opposite']}")
    print('   per5s visible/measured: '+' '.join(f"{k}:{v['visible']}/{v['measured']}" for k,v in e['per5s'].items()))
