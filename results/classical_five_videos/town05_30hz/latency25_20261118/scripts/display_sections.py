#!/usr/bin/env python3
"""Plan 25 step 3: inclusive wall time of the display filter's functions per frame (perf_counter wrappers),
replaying the detector outputs as the perception worker feeds them.  usage: display_sections.py"""
import sys,json,time,functools,collections
from pathlib import Path
import cv2,numpy as np,yaml
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import maploop_offline_compare as M
from vision_road.core import Timeline
from vision_road.temporal import TemporalDetector
from vision_road.carla import CarlaDetector,clone
from vision_road import display_filter as D
from builtin_interfaces.msg import Time
acc=collections.defaultdict(float);depth=collections.Counter()
def wrap(owner,name,label):
    f=getattr(owner,name)
    @functools.wraps(f)
    def w(*a,**k):
        if depth[label]:return f(*a,**k)
        depth[label]+=1;s=time.perf_counter()
        try:return f(*a,**k)
        finally:acc[label]+=time.perf_counter()-s;depth[label]-=1
    setattr(owner,name,w)
for n in ('fit','evaluate','outside','shift','relative','simplify'):wrap(D,n,n)
for n in ('move','store','recall','take','update','points','span_ok','span'):wrap(D.SideTrack,n,'track.'+n)
for n in ('update','output'):wrap(D.DisplayFilter,n,'filter.'+n)
wrap(D.DisplayStage,'messages','stage.messages');wrap(D.OneEuro,'__call__','oneeuro')
cache=Path('/media/jaehun/38F7-677B/vision-road-cache')/'town05_offline';pk=ROOT/'src/vision_road';cfg=M.settings(pk)
meta=json.loads((cache/'frames.json').read_text());valid=cv2.imread(str(cache/'valid.png'),0);valid[M.projector(pk,cfg).my>=1080*M.LAUNCH['image_roi_bottom']]=0
cv2.setNumThreads(2);err=M.LAUNCH['display_simplify_error_m']
section=yaml.safe_load(open(pk/'config/carla.yaml'));fs=D.settings_from((section.get('lane_display_filter') or {}).get('ros__parameters',{}))
core=D.DisplayCore(fs);stage=D.DisplayStage(fs,'base_link');det=TemporalDetector(cfg);det.detector=CarlaDetector(cfg);tl=Timeline(.25);k=0;rows=[]
for i in range(len(meta['frames'])):
    t=meta['frames'][i]['stamp']
    while k<len(meta['poses']) and meta['poses'][k][0]<=t+.05:tl.add(*meta['poses'][k]);k+=1
    pose=tl.at(t);lanes,_=det.process(cv2.imread(str(cache/'bev'/f'{i:05d}.png')),valid,t,pose,defer=True);det.finish()
    acc.clear();s=time.perf_counter()
    paint=[clone(l,points=D.simplify(l.points,err)) for l in lanes];bounds=[clone(l,points=D.simplify(l.points,err)) for l in det.boundary_lanes]
    core.styles(t,D.lane_styles(lanes))
    shown=core.step(t,[(l.side,l.points,l.reprojected) for l in bounds if len(l.points)>=2],pose,[(l.side,l.points) for l in paint if not l.reprojected and len(l.points)>=2])
    ns=round(t*1e9);stage.messages(Time(sec=ns//1000000000,nanosec=ns%1000000000),shown)
    rows.append(dict(acc)|{'total':time.perf_counter()-s})
for key in sorted({k for r in rows for k in r},key=lambda k:-np.mean([r.get(k,0) for r in rows])):
    v=np.array([r.get(key,0) for r in rows])*1e3;print(f'{key:16s} mean {v.mean():.3f} p95 {np.percentile(v,95):.3f}')
