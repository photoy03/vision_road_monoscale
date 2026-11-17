#!/usr/bin/env python3
"""Exact detector outputs for identity checks: per frame the paint lanes, own boundaries and outer lines
(points at full float precision, style, flags).  usage: dump_lines.py VIDEO OUT.jsonl.gz [name=value ...]
Also prints the mean / p95 time of TemporalDetector.process (with DEFER=1: without finish())."""
import sys,json,gzip,time,os
from dataclasses import replace
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import maploop_offline_compare as M
from vision_road.core import Timeline
from vision_road.temporal import TemporalDetector
from vision_road.carla import CarlaDetector
video,out=sys.argv[1],sys.argv[2]
if os.environ.get('CV_THREADS'):cv2.setNumThreads(int(os.environ['CV_THREADS']))  # as the worker's opencv_threads
cache=Path('/media/jaehun/38F7-677B/vision-road-cache')/f'{video}_offline';pk=ROOT/'src/vision_road';cfg=M.settings(pk)
for kv in sys.argv[3:]:
    k,v=kv.split('=',1);cfg=replace(cfg,**{k:type(getattr(cfg,k))(eval(v))})
meta=json.loads((cache/'frames.json').read_text());valid=cv2.imread(str(cache/'valid.png'),0);valid[M.projector(pk,cfg).my>=1080*M.LAUNCH['image_roi_bottom']]=0
# LEARNED_FILE=<learned.npz>: those BEV masks, one frame late as in the node, gate 'remove' within 15 m (the launch settings).
mask_at=None
if os.environ.get('LEARNED_FILE'):
    cfg=replace(cfg,carla_learned_gate='remove',carla_learned_gate_far_m=15.);mask_at=M.learned_masks(os.environ['LEARNED_FILE'])
det=TemporalDetector(cfg);det.detector=CarlaDetector(cfg);tl=Timeline(.25);k=0;clock=[]
def L(l):return dict(side=l.side,points=l.points.tolist(),style=l.style,conf=l.style_confidence,rep=bool(l.reprojected),hits=int(l.hits),outer=getattr(l,'outer',None))
with gzip.open(out,'wt') as fo:
    for i in range(len(meta['frames'])):
        t=meta['frames'][i]['stamp']
        while k<len(meta['poses']) and meta['poses'][k][0]<=t+.05:tl.add(*meta['poses'][k]);k+=1
        im=cv2.imread(str(cache/'bev'/f'{i:05d}.png'));s=time.perf_counter()
        if mask_at is not None:det.detector.learned=mask_at(i-1) if i>=1 else None
        # DEFER=1: as the perception worker calls it (outer lines after the own lines are out).
        if os.environ.get('DEFER'):lanes,_=det.process(im,valid,t,tl.at(t),defer=True);clock.append((time.perf_counter()-s)*1e3);det.finish()
        else:lanes,_=det.process(im,valid,t,tl.at(t));clock.append((time.perf_counter()-s)*1e3)
        fo.write(json.dumps(dict(frame=i,lanes=[L(l) for l in lanes],bounds=[L(l) for l in det.boundary_lanes],outer=[L(l) for l in getattr(det,'outer_lanes',[])],
                                 centres=[dict(c,points=np.asarray(c['points']).tolist()) for c in getattr(det,'centre_lanes',[])],stamp=t))+'\n')
print(video,'process mean %.2f p95 %.2f ms'%(np.mean(clock),np.percentile(clock,95)))
