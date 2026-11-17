"""Cost of sending one frame's worker result through the pipe (pickle + unpickle), with and without the
debug images (BEV, paint mask), on cached Town05 frames.  usage: pickle_cost.py"""
import sys,json,time,pickle,cv2,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import maploop_offline_compare as M
from vision_road.core import Timeline
from vision_road.temporal import TemporalDetector
from vision_road.carla import CarlaDetector
cache=Path('/media/jaehun/38F7-677B/vision-road-cache/town05_offline');cfg=M.settings(ROOT/'src/vision_road');meta=json.loads((cache/'frames.json').read_text())
valid=cv2.imread(str(cache/'valid.png'),0);d=TemporalDetector(cfg);d.detector=CarlaDetector(cfg);tl=Timeline(.25);k=0;full=[];lean=[]
for i in range(1500,1600):
    t=meta['frames'][i]['stamp']
    while k<len(meta['poses']) and meta['poses'][k][0]<=t+.05:tl.add(*meta['poses'][k]);k+=1
    bev=cv2.imread(str(cache/'bev'/f'{i:05d}.png'));lanes,binary=d.process(bev,valid,t,tl.at(t))
    r=dict(lanes=lanes,binary=binary,bev=bev,boundaries=d.boundary_lanes,markings=[],measured=1,predicted=0,registration='x',diagnostics=getattr(d.detector,'diagnostics',{}))
    for store,res in ((full,r),(lean,{**r,'binary':None,'bev':None})):
        a=time.perf_counter();b=pickle.dumps(res,protocol=pickle.HIGHEST_PROTOCOL);pickle.loads(b);store.append((time.perf_counter()-a)*1e3)
print('with images %.3f ms, without %.3f ms per frame (median)'%(np.median(full),np.median(lean)))
