import sys,cv2,numpy as np
from pathlib import Path
sys.path.insert(0,'tools');sys.path.insert(0,'src/vision_road')
import maploop_offline_compare as M
from vision_road.carla import CarlaDetector
from vision_road import road_markings as RM
pk=Path('src/vision_road');cfg=M.settings(pk);proj=M.projector(pk,cfg);cache=Path(sys.argv[1])
valid=cv2.imread(str(cache/'valid.png'),0);valid[proj.my>=1080*M.LAUNCH['image_roi_bottom']]=0
L=M.learned_masks(cache/'learned.npz');tiles=[]
for i in map(int,sys.argv[3:]):
    bev=cv2.imread(str(cache/'bev'/f'{i:05d}.png'));d=CarlaDetector(cfg);d.binary(bev,valid)
    marks=RM.detect(d,bev,valid,[],L(i))
    v=bev.copy();lm=L(i)>0;v[lm]=(0.5*v[lm]+[0,120,0]).astype(np.uint8)
    for m in marks:
        if m['kind']=='stop_line':
            r=round((cfg.forward_max-m['x'])/cfg.mpp);cv2.line(v,(0,r),(v.shape[1],r),(0,0,255),1)
        else:
            p0=(round((cfg.lateral_max-m['y1'])/cfg.mpp),round((cfg.forward_max-m['x1'])/cfg.mpp));p1=(round((cfg.lateral_max-m['y0'])/cfg.mpp),round((cfg.forward_max-m['x0'])/cfg.mpp))
            cv2.rectangle(v,p0,p1,(255,0,255) if m['kind']=='crosswalk' else (0,200,255),1)
    tiles.append(cv2.resize(v,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST));print(i,[(m['kind'],m.get('x',m.get('x0')),m.get('area')) for m in marks])
cv2.imwrite(sys.argv[2],np.hstack(tiles))
