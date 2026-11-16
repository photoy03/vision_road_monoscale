"""Relative yellowness of ridge pixels: Y=(r+g)/2-b minus its local road level (median over a
square window).  Distributions within 0.35 m of each side's boundary, x 5-20 m, every 10th frame.
usage: rel_probe.py CACHE RUN_DIR [window_m]"""
import sys,gzip,json,cv2,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import maploop_offline_compare as M
from vision_road.carla import CarlaDetector
cache=Path(sys.argv[1]);win=float(sys.argv[3]) if len(sys.argv)>3 else 1.5
cfg=M.settings(ROOT/'src/vision_road');d=CarlaDetector(cfg);valid=cv2.imread(str(cache/'valid.png'),0)
rows=[json.loads(l) for l in gzip.open(sys.argv[2]+'/predictions.jsonl.gz','rt')]
F,L,mpp=cfg.forward_max,cfg.lateral_max,cfg.mpp;k=2*round(win/mpp/2)+1
out={'left':[],'right':[],'elsewhere':[]}
for r in rows[::10]:
    bev=cv2.imread(str(cache/'bev'/f"{r['frame']:05d}.png"));d.binary(bev,valid)
    B,G,R=cv2.split(bev.astype(np.float32));Yv=(R+G)/2-B
    bg=cv2.medianBlur(np.clip(Yv+128,0,255).astype(np.uint8),k).astype(np.float32)-128
    dY=Yv-bg;ridge=(d.response>0)&(d.valid_safe>0)
    H,W=valid.shape;yy,xx=np.mgrid[0:H,0:W];X=F-yy*mpp;Yl=L-xx*mpp;near=np.zeros_like(ridge)
    for side in ('left','right'):
        b=[np.asarray(x['points']) for x in r['boundaries'] if x['side']==side and not x['reprojected']]
        if not b:continue
        p=max(b,key=lambda p:np.ptp(p[:,0]))
        band=(X>=max(5,p[0,0]))&(X<=min(20,p[-1,0]))&(np.abs(Yl-np.interp(X,p[:,0],p[:,1]))<=.35)
        out[side]+=list(dY[band&ridge]);near|=cv2.dilate(band.astype(np.uint8),np.ones((1,15),np.uint8))>0
    out['elsewhere']+=list(dY[ridge&~near&(X>=5)&(X<=20)])
for k_,v in out.items():
    v=np.array(v)
    if len(v):print(f'{k_:9s} n {len(v):7d}  dY pct 10/50/90 {np.percentile(v,10):5.1f} {np.percentile(v,50):5.1f} {np.percentile(v,90):5.1f}  share>=4 {np.mean(v>=4):.2f} >=6 {np.mean(v>=6):.2f} >=8 {np.mean(v>=8):.2f}')
