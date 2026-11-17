#!/usr/bin/env python3
"""HUD display line ends (plan 18): the display filter re-run offline on the detector's predictions.

usage: ends_eval.py maploop|town05 NAME [name=value ...]   (FilterSettings overrides on top of carla.yaml)

Input: the current detector predictions (realtime_latency_20261117/offline/<video>_step4) and the
rear-axle poses of the offline cache.  Settings: carla.yaml lane_display_filter + overrides.
Per side, frame to frame with the previous drawn line carried into the current base_link
(vehicle motion removed):
  y0     lateral change at 0 m (count > 5 cm)
  far    lateral change at the drawn far end (the nearer of the two ends; count > 20 cm)
  y10    lateral change at 10 m (count > 10 cm)
plus the same at fixed 0/10/20 m without removing the motion (as in the plan's baseline),
tools/display_filter_eval.py metrics (10 m jitter/shake, toggles, shown frames, curve lag),
display vs paint p95 in the curve, near-end (3-6 m) direction change in mrad around paint
fragments leaving the near range (dashed-gap kink), and the share of drawn lines starting at 0 m.
Writes metrics/NAME_<video>.json and metrics/NAME_<video>_display.jsonl.gz."""
import sys,gzip,json,math
from dataclasses import replace
from pathlib import Path
import numpy as np, yaml
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4]
sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
from vision_road.display_filter import settings_from  # noqa: E402
import display_filter_eval as dfe  # noqa: E402
PRED={v:ROOT/f'results/classical_five_videos/realtime_latency_20261117/offline/{v}_step4/predictions.jsonl.gz' for v in ('maploop','town05')}
CACHE={v:Path(f'/media/jaehun/38F7-677B/vision-road-cache/{v}_offline/frames.json') for v in ('maploop','town05')}
CURVE=dict(maploop=(22.,35.),town05=(15.,30.))
SEGS=dict(maploop=[('straight',5,20),('curve',22,35),('straight2',35,42),('crosswalk',42,48),('island',48,53),('junction2',53,61)],
          town05=[('straight',0,9),('turn',10,15),('curve',15,30),('turn2',35,40),('straight2',40,45),('junction',45,50),('straight3',55,60)])
XS=(6,8,10,15,20)


def at(p,x):
    return float(np.interp(x,p[:,0],p[:,1])) if p[0,0]-1e-6<=x<=p[-1,0]+1e-6 else None


def carry(p,m):
    dx,dy,dyaw=m;c,s=math.cos(dyaw),math.sin(dyaw);px=p[:,0]-dx;py=p[:,1]-dy
    return np.column_stack([c*px+s*py,-s*px+c*py])


def pose_at(poses,t):
    return np.array([np.interp(t,poses[:,0],poses[:,1]),np.interp(t,poses[:,0],poses[:,2]),np.interp(t,poses[:,0],poses[:,3])])


def relative(a,b):
    c,s=math.cos(a[2]),math.sin(a[2]);wx,wy=b[0]-a[0],b[1]-a[1]
    return c*wx+s*wy,-s*wx+c*wy,b[2]-a[2]


def settings(over):
    values=yaml.safe_load(open(ROOT/'src/vision_road/config/carla.yaml'))['lane_display_filter']['ros__parameters']
    s=settings_from(values)
    for kv in over:
        k,v=kv.split('=',1);s=replace(s,**{k:type(getattr(s,k))(eval(v))})
    return s


def stats(v,limit):
    v=np.abs(np.array(v,float))
    return dict(n=len(v),over=int((v>limit).sum()),p95_cm=round(float(np.percentile(v,95))*100,2) if len(v) else None,
                max_cm=round(float(v.max())*100,1) if len(v) else None)


def main():
    video,name=sys.argv[1],sys.argv[2];s=settings(sys.argv[3:])
    rows=[json.loads(l) for l in gzip.open(PRED[video],'rt')];t0=rows[0]['stamp']
    poses=np.array(json.load(open(CACHE[video]))['poses']);poses[:,3]=np.unwrap(poses[:,3])
    odom=[tuple(p) for p in poses.tolist()]
    filtered=dfe.run(rows,s,odom)
    P=[pose_at(poses,r['stamp']) for r in rows];T=[r['stamp']-t0 for r in rows]
    out=dict(video=video,name=name,settings=vars(s),frames=len(rows),sides={})
    for side in ('left','right'):
        y0=[];far=[];y10=[];raw0=[];raw10=[];raw20=[];big=[]
        shown=0;from0=0;head=[]
        for i in range(len(rows)):
            c=filtered[i].get(side)
            if c is not None:
                shown+=1;from0+=float(np.min(c[0][:,0]))<=.05
            if i==0 or c is None or filtered[i-1].get(side) is None:continue
            cur=np.asarray(c[0]);prv=np.asarray(filtered[i-1][side][0]);m=relative(P[i-1],P[i]);pc=carry(prv,m)
            a,b=at(cur,0.),at(pc,0.)
            if a is not None and b is not None:
                y0.append(a-b)
                if abs(a-b)>.05:big.append((round(T[i],3),'y0',round(a-b,3)))
            xf=min(cur[-1,0],pc[-1,0]);a,b=at(cur,xf),at(pc,xf)
            if a is not None and b is not None:
                far.append(a-b)
                if abs(a-b)>.2:big.append((round(T[i],3),'far',round(a-b,3),round(float(xf),1)))
            a,b=at(cur,10.),at(pc,10.)
            if a is not None and b is not None:y10.append(a-b)
            for x,lst in ((0.,raw0),(10.,raw10),(20.,raw20)):
                a,b=at(cur,x),at(prv,x)
                if a is not None and b is not None:lst.append(a-b)
        ends=dict(y0=stats(y0,.05),far=stats(far,.2),y10=stats(y10,.1),
                  raw_y0=stats(raw0,.05),raw_y10=stats(raw10,.1),raw_y20=stats(raw20,.2),
                  shown_frames=shown,from_0m_share=round(from0/max(1,shown),4),big=big)
        out['sides'][side]=ends
    out['display_filter_eval']=dfe.metrics(rows,filtered,CURVE[video])
    # Display vs paint p95 (observed display frames) per segment, cm at XS.
    gap={}
    for g,a,b in SEGS[video]:
        for side in ('left','right'):
            v={x:[] for x in XS}
            for r,f,t in zip(rows,filtered,T):
                if not a<=t<b or side not in f or f[side][1]:continue
                lanes=[np.asarray(l['points']) for l in r['lanes'] if l['side']==side and not l['reprojected']]
                d=np.asarray(f[side][0])
                for x in XS:
                    yd=at(d,x);ys=[y for y in (at(l[np.argsort(l[:,0])],x) for l in lanes) if y is not None]
                    ys=ys+[(p+q)/2 for k,p in enumerate(ys) for q in ys[k+1:] if .15<=abs(p-q)<=.40]
                    if yd is not None and ys:v[x].append(min(abs(y-yd) for y in ys))
            gap[f'{g} {a}-{b}/{side}']='/'.join(f'{np.percentile(v[x],95)*100:.0f}' if v[x] else '-' for x in XS)
    out['paint_gap_p95_cm_6_8_10_15_20']=gap
    # Near-end direction change (3-6 m), mrad, on frames where the nearest paint jumps > 1.5 m farther.
    kink={}
    for g,a,b in SEGS[video]:
        for side in ('left','right'):
            ev=[];oth=[];prev_near=None
            for i,(r,f,t) in enumerate(zip(rows,filtered,T)):
                paint=[np.asarray(l['points']) for l in r['lanes'] if l['side']==side and not l['reprojected']]
                near=min((float(p[:,0].min()) for p in paint),default=None)
                e=prev_near is not None and near is not None and near-prev_near>1.5;prev_near=near
                if not a<=t<b or i==0 or side not in f or side not in filtered[i-1]:continue
                pc=carry(np.asarray(filtered[i-1][side][0]),relative(P[i-1],P[i]));cc=np.asarray(f[side][0])
                v=[at(pc,3.),at(pc,6.),at(cc,3.),at(cc,6.)]
                if None in v:continue
                h=((v[3]-v[2])-(v[1]-v[0]))/3.*1e3;(ev if e else oth).append(h)
            r_=lambda q:round(float(np.sqrt(np.mean(np.square(q)))),2) if q else None
            kink[f'{g} {a}-{b}/{side}']=dict(event_rms=r_(ev),events=len(ev),other_rms=r_(oth))
    out['near_head_mrad']=kink
    # Near 0-5 m vs paint seen earlier (observed paint of the last 2 s carried into this frame by the poses): p95 cm.
    def world(i,p):
        x,y,a=P[i];c,s_=math.cos(a),math.sin(a);return np.column_stack([x+c*p[:,0]-s_*p[:,1],y+s_*p[:,0]+c*p[:,1]])
    def base(i,w):
        x,y,a=P[i];c,s_=math.cos(a),math.sin(a);d=w-[x,y];return np.column_stack([c*d[:,0]+s_*d[:,1],-s_*d[:,0]+c*d[:,1]])
    seen={side:[] for side in ('left','right')};ng={}
    acc={}
    for i,(r,f,t) in enumerate(zip(rows,filtered,T)):
        for side in ('left','right'):
            pts=[np.asarray(l['points'],float) for l in r['lanes'] if l['side']==side and not l['reprojected']]
            seen[side].append(world(i,np.vstack(pts)) if pts else np.empty((0,2)))
            if side not in f or f[side][1]:continue
            old=np.vstack(seen[side][max(0,i-60):i]) if i else np.empty((0,2))
            if not len(old):continue
            q=base(i,old);q=q[(q[:,0]>=0)&(q[:,0]<=5)]
            if not len(q):continue
            d=np.asarray(f[side][0])
            if d[0,0]>q[:,0].min():q=q[q[:,0]>=d[0,0]]
            if not len(q):continue
            yd=np.interp(q[:,0],d[:,0],d[:,1]);err=np.abs(q[:,1]-yd)
            # per 1 m bin the paint closest to the line (double lines: two strands), then the bin values
            for b0 in range(5):
                m=(q[:,0]>=b0)&(q[:,0]<b0+1)
                if m.any():
                    for g,a,b in SEGS[video]:
                        if a<=t<b:acc.setdefault(f'{g} {a}-{b}/{side}',[]).append(float(err[m].min()))
    out['near_gap_p95_cm_0_5m']={k:round(float(np.percentile(v,95))*100,1) for k,v in acc.items()}
    D=HERE.parent/'metrics';D.mkdir(exist_ok=True)
    (D/f'{name}_{video}.json').write_text(json.dumps(out,indent=1))
    with gzip.open(D/f'{name}_{video}_display.jsonl.gz','wt') as fo:
        for r,f,t in zip(rows,filtered,T):
            fo.write(json.dumps(dict(frame=r['frame'],t=round(t,3),sides={k:dict(points=np.round(v[0],3).tolist(),predicted=v[1]) for k,v in f.items()}))+'\n')
    for side,e in out['sides'].items():
        print(f"{video} {name} {side:5s} y0>5cm {e['y0']['over']:3d}/{e['y0']['n']} far>20cm {e['far']['over']:3d} y10>10cm {e['y10']['over']:2d} "
              f"| raw y0>5 {e['raw_y0']['over']:3d} y20>20 {e['raw_y20']['over']:3d} | shown {e['shown_frames']} from0 {e['from_0m_share']}")


if __name__=='__main__':
    main()
