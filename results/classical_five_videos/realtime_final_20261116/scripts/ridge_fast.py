"""Check a faster multi-width ridge response (OpenCV ops on a replicate-padded image) against the original."""
import sys,time,cv2,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import maploop_offline_compare as M
from vision_road.carla import shifted
cv2.setNumThreads(2);c=M.settings(ROOT/'src/vision_road');cache=Path('/media/jaehun/38F7-677B/vision-road-cache')/f'{sys.argv[1]}_offline'
bevs=[cv2.imread(str(cache/'bev'/f'{i:05d}.png')) for i in range(0,1800,6)]
def old(gray,smooth):
    response=np.zeros_like(gray)
    for width in (.10,.16,.24,.34):
        radius=max(2,round((width/2+.07)/c.mpp));left=shifted(smooth,-radius);right=shifted(smooth,radius)
        ridge=gray-np.maximum(left,right);background=np.maximum(np.maximum(left,right),25.)
        eligible=(ridge>=c.carla_ridge_min)&(ridge/background>=c.carla_relative_min);response=np.maximum(response,np.where(eligible,ridge,0))
    return response
def new(gray,smooth):
    radii=[max(2,round((w/2+.07)/c.mpp)) for w in (.10,.16,.24,.34)];pad=max(radii);w=gray.shape[1]
    padded=cv2.copyMakeBorder(smooth,0,0,pad,pad,cv2.BORDER_REPLICATE);response=np.zeros_like(gray)
    for radius in radii:
        side=cv2.max(padded[:,pad-radius:pad-radius+w],padded[:,pad+radius:pad+radius+w])
        ridge=cv2.subtract(gray,side)
        eligible=(ridge>=c.carla_ridge_min)&(ridge>=c.carla_relative_min*cv2.max(side,25.))
        response=cv2.max(response,np.where(eligible,ridge,np.float32(0)))
    return response
same=True;to=tn=0
for bev in bevs:
    gray=cv2.GaussianBlur(cv2.cvtColor(bev,cv2.COLOR_BGR2GRAY).astype(np.float32),(3,3),.55);smooth=cv2.blur(gray,(3,3))
    t=time.perf_counter();a=old(gray,smooth);to+=time.perf_counter()-t
    t=time.perf_counter();b=new(gray,smooth);tn+=time.perf_counter()-t
    if not np.array_equal(a>0,b>0) or np.abs(a-b).max()>1e-4:same=False;print('diff',np.sum((a>0)!=(b>0)),np.abs(a-b).max())
print('same:',same,'old %.2f ms new %.2f ms'%(to/len(bevs)*1e3,tn/len(bevs)*1e3))
