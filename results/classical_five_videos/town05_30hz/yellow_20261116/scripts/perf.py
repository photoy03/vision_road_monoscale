"""ms/frame of TemporalDetector.process over frames 450-1050 with the relative yellow test off/on (twice each).
usage: perf.py CACHE"""
import sys,json,time,cv2
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import maploop_offline_compare as M
from vision_road.core import Timeline
from vision_road.temporal import TemporalDetector
from vision_road.carla import CarlaDetector
cache=Path(sys.argv[1]);base=M.settings(ROOT/'src/vision_road');meta=json.loads((cache/'frames.json').read_text())
a,b=450,1050;bevs=[cv2.imread(str(cache/'bev'/f'{i:05d}.png')) for i in range(a,b)];cv2.setNumThreads(2)
valid=cv2.imread(str(cache/'valid.png'),0);valid[M.projector(ROOT/'src/vision_road',base).my>=1080*M.LAUNCH['image_roi_bottom']]=0
for name,w in (('off',0.),('on',6.),('off2',0.),('on2',6.)):
    cfg=replace(base,carla_yellow_relative_min=w);d=TemporalDetector(cfg);d.detector=CarlaDetector(cfg);tl=Timeline(.25);k=0;total=0.
    for j,i in enumerate(range(a,b)):
        t=meta['frames'][i]['stamp']
        while k<len(meta['poses']) and meta['poses'][k][0]<=t+.05:tl.add(*meta['poses'][k]);k+=1
        t0=time.perf_counter();d.process(bevs[j],valid,t,tl.at(t));total+=time.perf_counter()-t0
    print(name,'%.2f ms/frame'%(total/(b-a)*1e3),flush=True)
