#!/usr/bin/env python3
"""Plan 25 step 3: the display filter path of the perception worker (rt_worker.Perception), step by step, on the
cached Town05 BEVs: simplify + clone, lane_styles, DisplayCore.step, DisplayStage.messages.  Optionally writes
every frame's display messages (points) for an identity check.  usage: display_times.py [OUT.jsonl.gz]"""
import sys,json,time,gzip
from pathlib import Path
import cv2,numpy as np,yaml
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import maploop_offline_compare as M
from vision_road.core import Timeline
from vision_road.temporal import TemporalDetector
from vision_road.carla import CarlaDetector,clone
from vision_road.display_filter import DisplayCore,DisplayStage,simplify,lane_styles,settings_from
from builtin_interfaces.msg import Time
cache=Path('/media/jaehun/38F7-677B/vision-road-cache')/'town05_offline';pk=ROOT/'src/vision_road';cfg=M.settings(pk)
meta=json.loads((cache/'frames.json').read_text());valid=cv2.imread(str(cache/'valid.png'),0);valid[M.projector(pk,cfg).my>=1080*M.LAUNCH['image_roi_bottom']]=0
cv2.setNumThreads(2);err=M.LAUNCH['display_simplify_error_m']
section=yaml.safe_load(open(pk/'config/carla.yaml'));fs=settings_from((section.get('lane_display_filter') or {}).get('ros__parameters',{}))
core=DisplayCore(fs);stage=DisplayStage(fs,'base_link')
det=TemporalDetector(cfg);det.detector=CarlaDetector(cfg);tl=Timeline(.25);k=0;T={'simplify':[],'styles':[],'step':[],'messages':[],'total':[]}
out=gzip.open(sys.argv[1],'wt') if len(sys.argv)>1 else None
for i in range(len(meta['frames'])):
    t=meta['frames'][i]['stamp']
    while k<len(meta['poses']) and meta['poses'][k][0]<=t+.05:tl.add(*meta['poses'][k]);k+=1
    pose=tl.at(t);lanes,_=det.process(cv2.imread(str(cache/'bev'/f'{i:05d}.png')),valid,t,pose,defer=True)
    a=time.perf_counter()
    paint=[clone(l,points=simplify(l.points,err)) for l in lanes];bounds=[clone(l,points=simplify(l.points,err)) for l in det.boundary_lanes]
    b=time.perf_counter();core.styles(t,lane_styles(lanes));c=time.perf_counter()
    shown=core.step(t,[(l.side,l.points,l.reprojected) for l in bounds if len(l.points)>=2],pose,[(l.side,l.points) for l in paint if not l.reprojected and len(l.points)>=2])
    d=time.perf_counter();ns=round(t*1e9);msgs=stage.messages(Time(sec=ns//1000000000,nanosec=ns%1000000000),shown);e=time.perf_counter()
    for key,v in zip(T,(b-a,c-b,d-c,e-d,e-a)):T[key].append(v*1e3)
    det.finish()
    if out:out.write(json.dumps(dict(frame=i,markers=[(m.ns,m.id,m.action,round(m.color.r,4),[(p.x,p.y) for p in m.points]) for m in msgs[0].markers],
                                     styles=[(s.side,s.style,round(s.confidence,6)) for s in msgs[1].lanes] if hasattr(msgs[1],'lanes') else str(msgs[1])))+'\n')
for key,v in T.items():print(f'{key:9s} mean {np.mean(v):.3f}  p95 {np.percentile(v,95):.3f} ms')
