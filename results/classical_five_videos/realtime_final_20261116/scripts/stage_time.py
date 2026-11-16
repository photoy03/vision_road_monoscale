"""Wall time per frame of the detector's main functions (exclusive of nested wrapped calls is not separated:
inner functions are listed on their own lines too).  usage: stage_time.py VIDEO FIRST LAST"""
import sys,json,time,cv2,numpy as np
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import maploop_offline_compare as M
from vision_road.core import Timeline
from vision_road import temporal as TM, carla as CA, marking_style as MS, path_tracking as PT
from vision_road.temporal import TemporalDetector
from vision_road.carla import CarlaDetector
video,a,b=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
cache=Path('/media/jaehun/38F7-677B/vision-road-cache')/f'{video}_offline';cv2.setNumThreads(2)
pk=ROOT/'src/vision_road';cfg=M.settings(pk);meta=json.loads((cache/'frames.json').read_text())
valid=cv2.imread(str(cache/'valid.png'),0);valid[M.projector(pk,cfg).my>=1080*M.LAUNCH['image_roi_bottom']]=0
spent=defaultdict(float)
def wrap(owner,name,label=None):
    f=getattr(owner,name)
    def g(*x,**y):
        t=time.perf_counter()
        try:return f(*x,**y)
        finally:spent[label or name]+=time.perf_counter()-t
    setattr(owner,name,g)
for n in ('binary','paint_fragments','coherent_paths','merge_strands','pair_vote','spine','parallel_support','share_shape','width_error','edge_like','cut_tail','learned_gate'):wrap(CarlaDetector,n)
for n in ('process',):wrap(TemporalDetector,n,'temporal.process')
for n in dir(TemporalDetector):
    if n.startswith('_') and not n.startswith('__') and callable(getattr(TemporalDetector,n)):wrap(TemporalDetector,n,'temporal.'+n)
wrap(MS,'classify_lane');wrap(TM,'classify_lane','temporal->classify_lane')
import cv2 as _cv
for fn in ('warpAffine','cvtColor','findTransformECC','estimateAffinePartial2D','dilate','erode','calcOpticalFlowPyrLK','goodFeaturesToTrack','phaseCorrelate','remap','bitwise_and'):
    if hasattr(_cv,fn):wrap(_cv,fn,'cv2.'+fn)
wrap(PT.PathTracking,'select','path_tracking.select')
det=TemporalDetector(cfg);det.detector=CarlaDetector(cfg);tl=Timeline(.25);k=0
bevs=[cv2.imread(str(cache/'bev'/f'{i:05d}.png')) for i in range(a,b)]
for j,i in enumerate(range(a,b)):
    t=meta['frames'][i]['stamp']
    while k<len(meta['poses']) and meta['poses'][k][0]<=t+.05:tl.add(*meta['poses'][k]);k+=1
    det.process(bevs[j],valid,t,tl.at(t))
n=b-a
for k_,v in sorted(spent.items(),key=lambda x:-x[1]):
    if v/n*1e3>=.05:print(f'{k_:28s} {v/n*1e3:6.2f} ms/frame')
