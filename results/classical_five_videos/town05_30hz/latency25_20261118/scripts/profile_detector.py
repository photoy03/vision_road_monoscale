#!/usr/bin/env python3
"""Plan 25 step 2: where the detector's time goes (cProfile over the cached Town05 BEVs, as the worker calls it:
process(defer=True), finish() after).  usage: profile_detector.py VIDEO [N]   writes ../metrics/profile_<video>.txt"""
import sys,json,time,cProfile,pstats,io
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import maploop_offline_compare as M
from vision_road.core import Timeline
from vision_road.temporal import TemporalDetector
from vision_road.carla import CarlaDetector
video=sys.argv[1];N=int(sys.argv[2]) if len(sys.argv)>2 else 0
cache=Path('/media/jaehun/38F7-677B/vision-road-cache')/f'{video}_offline';pk=ROOT/'src/vision_road';cfg=M.settings(pk)
meta=json.loads((cache/'frames.json').read_text());valid=cv2.imread(str(cache/'valid.png'),0);valid[M.projector(pk,cfg).my>=1080*M.LAUNCH['image_roi_bottom']]=0
cv2.setNumThreads(1)
det=TemporalDetector(cfg);det.detector=CarlaDetector(cfg);tl=Timeline(.25);k=0
frames=range(len(meta['frames']) if not N else N);ims=[cv2.imread(str(cache/'bev'/f'{i:05d}.png')) for i in frames]
pr=cProfile.Profile();proc=[];fin=[]
for i in frames:
    t=meta['frames'][i]['stamp']
    while k<len(meta['poses']) and meta['poses'][k][0]<=t+.05:tl.add(*meta['poses'][k]);k+=1
    pr.enable();s=time.perf_counter();det.process(ims[i],valid,t,tl.at(t),defer=True);m=time.perf_counter();det.finish();e=time.perf_counter();pr.disable()
    proc.append((m-s)*1e3);fin.append((e-m)*1e3)
out=io.StringIO();st=pstats.Stats(pr,stream=out);st.sort_stats('tottime').print_stats(45);st.sort_stats('cumulative').print_stats(60)
head=f'{video}: process mean {np.mean(proc):.2f} p95 {np.percentile(proc,95):.2f} ms (with profiler overhead); finish mean {np.mean(fin):.2f}\n'
(Path(__file__).resolve().parent.parent/'metrics'/f'profile_{video}.txt').write_text(head+out.getvalue());print(head)
