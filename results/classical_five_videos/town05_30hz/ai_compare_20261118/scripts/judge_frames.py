#!/usr/bin/env python3
"""AI filter off vs on, one frame: the raw BEV (cache) enlarged x3, the learned mask (the one the gate used: previous
frame's) as blue tint, own boundaries off = red, on = green (thin, 1 px dots), paint lanes off = orange, on = cyan.
Beside it the camera preview.  usage: judge_frames.py VIDEO SECONDS...   writes ../captures/judge_<video>_t<s>.jpg"""
import sys,json,gzip
from pathlib import Path
import cv2,numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4];sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import maploop_offline_compare as M
video=sys.argv[1];cache=Path('/media/jaehun/38F7-677B/vision-road-cache')/f'{video}_offline';cfg=M.settings(ROOT/'src/vision_road')
A=[json.loads(l) for l in gzip.open(HERE.parent/'offline'/f'{video}_off.jsonl.gz','rt')];B=[json.loads(l) for l in gzip.open(HERE.parent/'offline'/f'{video}_on.jsonl.gz','rt')];t0=A[0]['stamp']
mask_at=M.learned_masks(cache/'learned.npz');S=3
def px(p):return np.column_stack([(-p[:,1]-cfg.lateral_min)/cfg.mpp,(cfg.forward_max-p[:,0])/cfg.mpp])*S
for sec in map(float,sys.argv[2:]):
    i=min(range(len(A)),key=lambda i:abs(A[i]['stamp']-t0-sec))
    bev=cv2.resize(cv2.imread(str(cache/'bev'/f'{i:05d}.png')),None,fx=S,fy=S,interpolation=cv2.INTER_NEAREST)
    m=cv2.resize(mask_at(i-1),None,fx=S,fy=S,interpolation=cv2.INTER_NEAREST)>0 if i else np.zeros(bev.shape[:2],bool)
    tint=bev.copy();tint[m]=(.6*tint[m]+.4*np.array([255,120,0])).astype(np.uint8)
    for rows,color,key in ((A,(0,0,255),'bounds'),(B,(0,200,0),'bounds'),(A,(0,140,255),'lanes'),(B,(255,255,0),'lanes')):
        for l in rows[i][key]:
            for x,y in px(np.asarray(l['points'])).astype(int):
                if key=='bounds':cv2.circle(tint,(x+(-2 if color[2] else 2),y),1,color,-1)
                else:tint[max(0,y-1):y+1,max(0,x-1):x+1]=color
    far=int((cfg.forward_max-15.)/cfg.mpp*S);cv2.line(tint,(0,far),(tint.shape[1],far),(255,0,255),1)
    cv2.putText(tint,f'{video} t={sec:.2f}s frame {i}',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),2)
    cv2.putText(tint,'red=off green=on (boundaries), orange/cyan paint off/on, blue=AI mask, magenta=15 m',(5,40),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),1)
    # zoom: +-1 m around each side's boundary (off, else on), 3-25 m ahead, enlarged x3 more
    zooms=[]
    for side in ('left','right'):
        line=next((np.asarray(l['points']) for rows in (A,B) for l in rows[i]['bounds'] if l['side']==side),None)
        if line is None:continue
        y=float(np.interp(10.,line[:,0],line[:,1]));c0=int(((-y-1.)-cfg.lateral_min)/cfg.mpp*S);c1=int(((-y+1.)-cfg.lateral_min)/cfg.mpp*S)
        r0=int((cfg.forward_max-25.)/cfg.mpp*S);r1=int((cfg.forward_max-3.)/cfg.mpp*S);z=tint[r0:r1,max(0,c0):c1]
        z=cv2.resize(z,None,fx=3,fy=1,interpolation=cv2.INTER_NEAREST);cv2.putText(z,side+' 3-25 m',(3,15),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1);zooms.append(z)
    if zooms:
        hz=max(z.shape[0] for z in zooms);cv2.imwrite(str(HERE.parent/'captures'/f'zoom_{video}_t{sec:05.2f}.jpg'),np.hstack([cv2.copyMakeBorder(z,0,hz-z.shape[0],0,6,cv2.BORDER_CONSTANT,value=(255,255,255)) for z in zooms]),[cv2.IMWRITE_JPEG_QUALITY,92])
    pv=sorted((cache/'preview').glob('*'));cam=cv2.imread(str(pv[i])) if i<len(pv) else None
    out=tint if cam is None else np.hstack([tint,cv2.copyMakeBorder(cv2.resize(cam,(int(cam.shape[1]*tint.shape[0]/cam.shape[0]/2),tint.shape[0]//2)),0,tint.shape[0]-tint.shape[0]//2,0,0,cv2.BORDER_CONSTANT)])
    cv2.imwrite(str(HERE.parent/'captures'/f'judge_{video}_t{sec:05.2f}.jpg'),out,[cv2.IMWRITE_JPEG_QUALITY,90])
    print(sec,i,out.shape)
