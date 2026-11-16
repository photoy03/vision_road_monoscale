"""Time the steps of CarlaDetector.binary on cached BEVs.  usage: binary_parts.py VIDEO FIRST LAST"""
import sys,time,cv2,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import maploop_offline_compare as M
from vision_road.carla import shifted
video,a,b=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]);cv2.setNumThreads(2)
cache=Path('/media/jaehun/38F7-677B/vision-road-cache')/f'{video}_offline';c=M.settings(ROOT/'src/vision_road')
bevs=[cv2.imread(str(cache/'bev'/f'{i:05d}.png')) for i in range(a,b)];T={}
def tick(name,t):T[name]=T.get(name,0)+time.perf_counter()-t;return time.perf_counter()
for bev in bevs:
    t=time.perf_counter()
    gray=cv2.cvtColor(bev,cv2.COLOR_BGR2GRAY).astype(np.float32);gray=cv2.GaussianBlur(gray,(3,3),.55);t=tick('gray',t)
    h,l,s=cv2.split(cv2.cvtColor(bev,cv2.COLOR_BGR2HLS));t=tick('hls',t)
    b_,g,r=cv2.split(bev.astype(np.float32));white=(s<120)&(l>c.carla_white_light_min)
    yellow=(h>=16)&(h<=42)&(s>20)&((g-b_)>18)&((g-b_)>.09*np.maximum(g,1))&((g-b_)>1.25*(r-g));t=tick('color fixed',t)
    yl=(r+g)/2-b_;size=2*round(c.carla_yellow_relative_window_m/c.mpp/2)+1
    road=cv2.medianBlur(np.clip(yl+128,0,255).astype(np.uint8),size).astype(np.float32)-128;yellow|=(yl>=6)&(yl-road>=6);t=tick('yellow relative',t)
    color=white|yellow;response=np.zeros_like(gray);smooth=cv2.blur(gray,(3,3))
    for width in (.10,.16,.24,.34):
        radius=max(2,round((width/2+.07)/c.mpp));left=shifted(smooth,-radius);right=shifted(smooth,radius)
        ridge=gray-np.maximum(left,right);background=np.maximum(np.maximum(left,right),25.)
        eligible=(ridge>=c.carla_ridge_min)&(ridge/background>=c.carla_relative_min);response=np.maximum(response,np.where(eligible,ridge,0))
    t=tick('ridge',t)
    step=max(2,round(.25/c.mpp));horizontal=(gray-np.maximum(shifted(smooth,-step,0),shifted(smooth,step,0))>=4)&color
    long=cv2.morphologyEx(horizontal.astype(np.uint8),cv2.MORPH_OPEN,np.ones((1,max(3,round(1.4/c.mpp))),np.uint8));t=tick('grid',t)
    tr=cv2.morphologyEx(horizontal.astype(np.uint8),cv2.MORPH_OPEN,np.ones((1,max(3,round(1./c.mpp))),np.uint8));t=tick('transverse',t)
n=len(bevs);print(bevs[0].shape);[print(f'{k:16s} {v/n*1e3:.2f} ms') for k,v in T.items()]
