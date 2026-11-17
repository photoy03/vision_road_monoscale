#!/usr/bin/env python3
"""Plan 19: run the detector (current carla.yaml + overrides) over an offline BEV cache and record,
per frame, the own boundary lines, the outer lines with their attributes, and the leftover paint
that could be an outer line (the denominator of the detection rate).

usage: outer_run.py VIDEO NAME [name=value ...]     (Settings overrides, e.g. carla_outer_parallel_m=0.4)
writes ../runs/NAME_VIDEO.jsonl.gz and prints the time spent in outer_lines.

Paint visible on a side (frame): the own line of that side is confirmed and measured this frame, and
the paint not used by the own lines (fragments, near ones farther out included) has at least 2 m of
fragments whose median distance from the own line (along its normal, own line extended straight
6 m past its ends) is 2.3-4.7 m outward.  No parallel test: any paint there counts.
"""
import sys,json,gzip,time,math,os
from dataclasses import replace
from pathlib import Path
import cv2,numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4]
sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import maploop_offline_compare as M  # noqa: E402
from vision_road.core import Timeline  # noqa: E402
from vision_road.temporal import TemporalDetector  # noqa: E402
from vision_road.carla import CarlaDetector  # noqa: E402

video,name=sys.argv[1],sys.argv[2]
cache=Path('/media/jaehun/38F7-677B/vision-road-cache')/f'{video}_offline'
pk=ROOT/'src/vision_road';cfg=M.settings(pk)
for kv in sys.argv[3:]:
    k,v=kv.split('=',1);cfg=replace(cfg,**{k:type(getattr(cfg,k))(eval(v))})
meta=json.loads((cache/'frames.json').read_text())
valid=cv2.imread(str(cache/'valid.png'),0);valid[M.projector(pk,cfg).my>=1080*M.LAUNCH['image_roi_bottom']]=0
# LEARNED_FILE=<learned.npz>: those BEV masks (loaded once), one frame late as in the node, gate 'remove' within 15 m.
masks=None
if os.environ.get('LEARNED_FILE'):
    _d=np.load(os.environ['LEARNED_FILE']);_w=int(_d['shape'][1]);masks=_d['masks']
    cfg=replace(cfg,carla_learned_gate='remove',carla_learned_gate_far_m=15.)
det=TemporalDetector(cfg);det.detector=CarlaDetector(cfg);tl=Timeline(.25);k=0
grab={};spent=[]
orig=CarlaDetector.outer_lines


def visible_paint(fragments,own):
    """side -> metres of leftover paint 2.3-4.7 m outside the own line (median normal distance)."""
    out={}
    for side,(sample,smooth) in own.items():
        if len(sample)<2 or sample[-1]-sample[0]<1.:continue
        sign=1. if side=='left' else -1.
        x=np.arange(sample[0]-6.,sample[-1]+6.+1e-9,.25);y=np.interp(x,sample,smooth)
        a,b=np.polyfit(sample[:min(len(sample),6)],smooth[:min(len(sample),6)],1);y[x<sample[0]]=a*x[x<sample[0]]+b
        a,b=np.polyfit(sample[-min(len(sample),6):],smooth[-min(len(sample),6):],1);y[x>sample[-1]]=a*x[x>sample[-1]]+b
        Q=np.column_stack([x,y]);T=np.gradient(Q,axis=0);T/=np.linalg.norm(T,axis=1)[:,None];N=np.column_stack([-T[:,1],T[:,0]])
        total=0.
        for f in fragments:
            P=f['lane'].points[::3];D=P[:,None,:]-Q[None,:,:];n=np.argmin((D**2).sum(2),1);d=D[np.arange(len(P)),n]
            ok=np.abs((d*T[n]).sum(1))<=.3
            if ok.sum()<3:continue
            g=sign*(d*N[n]).sum(1)[ok]
            if 2.3<=np.median(g)<=4.7:total+=float(np.ptp(f['lane'].points[:,0]))
        out[side]=total
    return out


def wrapped(self,fragments,own,own_yellow):
    start=time.perf_counter();out=orig(self,fragments,own,own_yellow);spent.append((time.perf_counter()-start)*1e3)
    if not os.environ.get('TIMING'):grab['paint']=visible_paint(fragments,own)  # TIMING=1: skip (it runs inside the timed call)
    grab['own']={s:np.column_stack(v).round(3).tolist() for s,v in own.items()}
    grab['cand']=[dict(side=l.side,gap=round(l.gap_m,3),paint=round(l.paint_m,2),observed=[round(float(v),2) for v in l.observed],yellow=l.yellow) for l in out]
    return out


CarlaDetector.outer_lines=wrapped
rows=[];n=len(meta['frames']);t0=meta['frames'][0]['stamp'];clock=[]
for i in range(n):
    t=meta['frames'][i]['stamp']
    while k<len(meta['poses']) and meta['poses'][k][0]<=t+.05:tl.add(*meta['poses'][k]);k+=1
    grab.clear();s0=time.perf_counter()
    if masks is not None:det.detector.learned=(np.unpackbits(masks[i-1],axis=1)[:,:_w]*255).astype(np.uint8) if i>=1 else None
    det.process(cv2.imread(str(cache/'bev'/f'{i:05d}.png')),valid,t,tl.at(t))
    clock.append((time.perf_counter()-s0)*1e3)
    rows.append(dict(frame=i,t=round(t-t0,3),stamp=t,
        own={l.side:dict(points=np.round(l.points,3).tolist(),reprojected=bool(l.reprojected),yellow=None) for l in det.boundary_lanes},
        outer={l.side:dict(points=np.round(l.points,3).tolist(),reprojected=bool(l.reprojected),style=l.style,**l.outer) for l in det.outer_lanes},
        paint=grab.get('paint',{}),cand=grab.get('cand',[]),spine=grab.get('own',{})))
D=HERE.parent/'runs';D.mkdir(exist_ok=True)
with gzip.open(D/f'{name}_{video}.jsonl.gz','wt') as fo:
    for r in rows:fo.write(json.dumps(r)+'\n')
s=np.array(spent) if spent else np.zeros(1)
print(f'{video} {name}: frames {n}, outer_lines mean {s.mean():.2f} p95 {np.percentile(s,95):.2f} ms; process mean {np.mean(clock):.1f} ms')
