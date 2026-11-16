"""Check the rewritten spine() with cubic/near off against the old np.polyfit quadratic on Town04 chains.
usage: cmp_spine.py CACHE"""
import sys,json,cv2,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import maploop_offline_compare as M
from vision_road.carla import CarlaDetector,grouped_median
from dataclasses import replace
cache=Path(sys.argv[1]);cfg=replace(M.settings(ROOT/'src/vision_road'),carla_spine_cubic_m=0.,carla_spine_near_m=0.)
valid=cv2.imread(str(cache/'valid.png'),0);d=CarlaDetector(cfg);worst=0.;n=0
orig=d.spine
def spine(group,forward=None,center=None):
    global worst,n
    r=orig(group,forward,center)
    f,cn=(forward,center) if forward is not None else grouped_median(np.vstack([a['lane'].points for a in group]))
    o=float(np.mean(f));x=f-o;w=np.ones(len(x))
    for _ in range(5):
        fit=np.polyfit(x,cn,2,w=w);res=np.abs(cn-np.polyval(fit,x));w=np.minimum(1.,.20/np.maximum(res,.001))
    worst=max(worst,float(np.abs(np.polyval(fit,r[0]-o)-r[1]).max()));n+=1;return r
d.spine=spine
for i in range(0,1800,15):d.detect(d.binary(cv2.imread(str(cache/'bev'/f'{i:05d}.png')),valid))
print('spines',n,'max |new-old| m',worst)
