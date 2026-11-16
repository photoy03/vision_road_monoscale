"""Fragments (paint_fragments nodes) within 0.35 m of the run's boundary on one side, x 5-20 m,
every 10th frame: length, share flagged yellow (>12% yellow pixels), share kept in that side's strands.
usage: fragment_probe.py CACHE RUN_DIR [side]"""
import sys,gzip,json,cv2,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import maploop_offline_compare as M
from vision_road.carla import CarlaDetector
cache=Path(sys.argv[1]);side=sys.argv[3] if len(sys.argv)>3 else 'left'
cfg=M.settings(ROOT/'src/vision_road');d=CarlaDetector(cfg);seen={}
orig=d.coherent_paths
def grab(nodes):seen['nodes']=list(nodes);return orig(nodes)
d.coherent_paths=grab
valid=cv2.imread(str(cache/'valid.png'),0)
rows=[json.loads(l) for l in gzip.open(sys.argv[2]+'/predictions.jsonl.gz','rt')]
L=[];Y=[];K=[]
for r in rows[::10]:
    b=[np.asarray(x['points']) for x in r['boundaries'] if x['side']==side and not x['reprojected']]
    if not b:continue
    p=max(b,key=lambda p:np.ptp(p[:,0]))
    lanes=d.detect(d.binary(cv2.imread(str(cache/'bev'/f"{r['frame']:05d}.png")),valid))
    kept=[l.points for l in lanes if l.side==side]
    for a in seen.get('nodes',[]):
        q=a['lane'].points;q=q[(q[:,0]>=5)&(q[:,0]<=20)]
        if len(q)<2 or np.median(np.abs(q[:,1]-np.interp(q[:,0],p[:,0],p[:,1])))>.35:continue
        L.append(np.ptp(q[:,0]));Y.append(a['yellow']);K.append(any(np.median(np.abs(q[:,1]-np.interp(q[:,0],k[:,0],k[:,1])))<.1 and k[0,0]-.2<=q[:,0].mean()<=k[-1,0]+.2 for k in kept))
L=np.array(L);Y=np.array(Y);K=np.array(K)
print(f"fragments {len(L)} median len {np.median(L):.2f} m  length-weighted: yellow {L[Y].sum()/L.sum():.2f} kept {L[K].sum()/L.sum():.2f}  kept|yellow {L[Y&K].sum()/max(1e-9,L[Y].sum()):.2f} kept|white {L[~Y&K].sum()/max(1e-9,L[~Y].sum()):.2f}")
