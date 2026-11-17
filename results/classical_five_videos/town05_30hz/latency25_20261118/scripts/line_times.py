#!/usr/bin/env python3
"""Plan 25 step 2: time per source line inside chosen functions (a settrace line timer: inflated by its own
overhead, so only the ranking counts), on every 3rd cached Town05 BEV after a warm-up run.
usage: line_times.py FUNC [FUNC...]  (e.g. carla.CarlaDetector.binary temporal.TemporalDetector.process)"""
import sys,json,time,collections,linecache
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import maploop_offline_compare as M
from vision_road.core import Timeline
from vision_road import temporal,carla,display_filter
from vision_road.temporal import TemporalDetector
from vision_road.carla import CarlaDetector
cache=Path('/media/jaehun/38F7-677B/vision-road-cache')/'town05_offline';pk=ROOT/'src/vision_road';cfg=M.settings(pk)
meta=json.loads((cache/'frames.json').read_text());valid=cv2.imread(str(cache/'valid.png'),0);valid[M.projector(pk,cfg).my>=1080*M.LAUNCH['image_roi_bottom']]=0
cv2.setNumThreads(2)
codes={}
for spec in sys.argv[1:]:
    mod,*path=spec.split('.');obj={'carla':carla,'temporal':temporal,'display_filter':display_filter}[mod]
    for p in path:obj=getattr(obj,p)
    codes[obj.__code__]=spec
times=collections.Counter();last={}
def tracer(frame,event,arg):
    if frame.f_code not in codes:return None
    def local(frame,event,arg):
        now=time.perf_counter();key=id(frame)
        if key in last:
            ln=last[key];times[(frame.f_code,ln)]+=now-last.get(('t',key),now)
        if event=='line':last[key]=frame.f_lineno;last[('t',key)]=time.perf_counter()
        elif event=='return':last.pop(key,None);last.pop(('t',key),None)
        return local
    return local
det=TemporalDetector(cfg);det.detector=CarlaDetector(cfg);tl=Timeline(.25);k=0;n=0
for i in range(len(meta['frames'])):
    t=meta['frames'][i]['stamp']
    while k<len(meta['poses']) and meta['poses'][k][0]<=t+.05:tl.add(*meta['poses'][k]);k+=1
    im=cv2.imread(str(cache/'bev'/f'{i:05d}.png'))
    on=i%3==0 and i>30
    if on:sys.settrace(tracer);n+=1
    det.process(im,valid,t,tl.at(t),defer=True);det.finish()
    if on:sys.settrace(None)
for code,spec in codes.items():
    rows=sorted(((v,ln) for (c,ln),v in times.items() if c is code),reverse=True)[:25];total=sum(v for (c,_),v in times.items() if c is code)
    print(f'== {spec}: {total/n*1e3:.2f} ms/frame (traced)')
    for v,ln in rows:print(f'{v/n*1e3:7.3f} ms  {ln:5d}  {linecache.getline(code.co_filename,ln).strip()[:110]}')
