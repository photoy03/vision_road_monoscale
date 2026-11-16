"""Where the left (yellow) line's paint is lost in CarlaDetector.binary: within 0.35 m of the run's
left boundary, x 5-20 m, every 10th frame.  ridge = multi-width ridge response > 0; then the colour
gates white / yellow / neither among ridge pixels; raw = kept in the paint mask; ymask = yellow paint.
usage: stage_probe.py CACHE RUN_DIR [side]"""
import sys,gzip,json,cv2,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import maploop_offline_compare as M
from vision_road.carla import CarlaDetector
cache=Path(sys.argv[1]);side=sys.argv[3] if len(sys.argv)>3 else 'left'
cfg=M.settings(ROOT/'src/vision_road');d=CarlaDetector(cfg)
valid=cv2.imread(str(cache/'valid.png'),0)
rows=[json.loads(l) for l in gzip.open(sys.argv[2]+'/predictions.jsonl.gz','rt')]
F,L=cfg.forward_max,cfg.lateral_max;mpp=cfg.mpp;tot={}
for r in rows[::10]:
    b=[np.asarray(x['points']) for x in r['boundaries'] if x['side']==side and not x['reprojected']]
    if not b:continue
    p=max(b,key=lambda p:np.ptp(p[:,0]));bev=cv2.imread(str(cache/'bev'/f"{r['frame']:05d}.png"))
    d.binary(bev,valid)
    H,W=valid.shape;yy,xx=np.mgrid[0:H,0:W];X=F-yy*mpp;Y=L-xx*mpp
    inx=(X>=max(5,p[0,0]))&(X<=min(20,p[-1,0]))
    band=inx&(np.abs(Y-np.interp(X,p[:,0],p[:,1]))<=.35)
    ridge=band&(d.response>0)
    s=d.saturation;l=d.light
    white=(s<120)&(l>cfg.carla_white_light_min)
    vals=dict(ridge=ridge.sum(),white=(ridge&white).sum(),yellow=(ridge&d.yellow).sum(),neither=(ridge&~white&~d.yellow).sum(),raw=(band&(d.raw_mask>0)).sum(),ymask=(band&(d.yellow_mask>0)).sum())
    B=bev.astype(np.float32);gb=(B[...,1]-B[...,0])[ridge]
    vals['gb_med']=float(np.median(gb)) if len(gb) else 0
    for k,v in vals.items():tot.setdefault(k,[]).append(float(v))
n=len(tot['ridge']);R=sum(tot['ridge'])
print(f"frames {n}  ridge px/frame {R/n:.0f}  white {sum(tot['white'])/R:.2f} yellow {sum(tot['yellow'])/R:.2f} neither {sum(tot['neither'])/R:.2f}  raw/ridge {sum(tot['raw'])/R:.2f}  ymask/raw {sum(tot['ymask'])/max(1,sum(tot['raw'])):.2f}  g-b median {np.median(tot['gb_med']):.1f}")
