"""Run the detector over a frame range (carla.yaml settings) and print the width check per frame.
usage: width_debug.py CACHE FIRST LAST [warmup]"""
import sys,json,math,cv2,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import maploop_offline_compare as M
from vision_road.core import Timeline
from vision_road.temporal import TemporalDetector
from vision_road.carla import CarlaDetector
cache=Path(sys.argv[1]);a,b=int(sys.argv[2]),int(sys.argv[3]);w=int(sys.argv[4]) if len(sys.argv)>4 else 60
cfg=M.settings(ROOT/'src/vision_road');meta=json.loads((cache/'frames.json').read_text())
valid=cv2.imread(str(cache/'valid.png'),0);valid[M.projector(ROOT/'src/vision_road',cfg).my>=1080*M.LAUNCH['image_roi_bottom']]=0
det=TemporalDetector(cfg);det.detector=CarlaDetector(cfg);d=det.detector
tl=Timeline(.25);poses=meta['poses'];k=0
orig=d.width_error
def we(side,points):
    v=orig(side,points);other=d.tracking_priors.get('right' if side=='left' else 'left')
    if i>=a and v is not None:
        ref=other[0];print(f'  {i} {side} cand {points[0,0]:.1f}-{points[-1,0]:.1f} y {np.interp(8,points[:,0],points[:,1]):.2f}  other {ref[0,0]:.1f}-{ref[-1,0]:.1f} y8 {np.interp(8,ref[:,0],ref[:,1]):.2f} lane_width {d.lane_width:.2f} off {v:.2f}')
    return v
d.width_error=we
for row in meta['frames'][max(0,a-w):b]:
    i,t=row['frame'],row['stamp']
    while k<len(poses) and poses[k][0]<=t+.05:tl.add(*poses[k]);k+=1
    det.process(cv2.imread(str(cache/'bev'/f'{i:05d}.png')),valid,t,tl.at(t))
