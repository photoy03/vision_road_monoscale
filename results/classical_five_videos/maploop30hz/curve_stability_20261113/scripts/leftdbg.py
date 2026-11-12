"""BEV view: raw paint (gray), detector mask after grid (white), fragments by side (green/blue), boundary (yellow)."""
import sys,json,cv2,numpy as np
from pathlib import Path
sys.path.insert(0,'/home/jaehun/Desktop/vision-road/tools');sys.path.insert(0,'/home/jaehun/Desktop/vision-road/src/vision_road')
import maploop_offline_compare as M
from vision_road.carla import CarlaDetector
from vision_road.temporal import TemporalDetector
from vision_road.core import Timeline
from dataclasses import replace
def main():
    pk=Path('/home/jaehun/Desktop/vision-road/src/vision_road');cfg=M.settings(pk);proj=M.projector(pk,cfg);cache=Path(sys.argv[1])
    sets=[kv.split('=') for kv in sys.argv[4:] if '=' in kv];frames=[int(v) for v in sys.argv[3].split(',')]
    for k,v in sets:cfg=replace(cfg,**{k:type(getattr(cfg,k))(v)})
    valid=cv2.imread(str(cache/'valid.png'),0);valid[proj.my>=1080*M.LAUNCH['image_roi_bottom']]=0
    meta=json.loads((cache/'frames.json').read_text());d=TemporalDetector(cfg);d.detector=CarlaDetector(cfg);tl=Timeline(.25);k=0;tiles=[]
    for i in range(min(frames)-60,max(frames)+1):
        t=meta['frames'][i]['stamp']
        while k<len(meta['poses']) and meta['poses'][k][0]<=t+.05:tl.add(*meta['poses'][k]);k+=1
        bev=cv2.imread(str(cache/'bev'/f'{i:05d}.png'));lanes,_=d.process(bev,valid,t,tl.at(t))
        if i not in frames:continue
        det=d.detector;v=(bev*.6).astype(np.uint8);v[det.raw_mask>0]=(150,150,150)
        def px(p):return np.column_stack([(cfg.lateral_max-p[:,1])/cfg.mpp,(cfg.forward_max-p[:,0])/cfg.mpp]).astype(np.int32)
        for l in lanes:
            if l.reprojected:continue
            cv2.polylines(v,[px(l.points)],False,(0,255,0) if l.side=='left' else (255,120,0),1)
        for b in d.boundary_lanes:cv2.polylines(v,[px(b.points)],False,(0,255,255),1)
        cv2.putText(v,str(i),(3,12),0,.4,(255,255,255),1)
        tiles.append(cv2.resize(v,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST))
        print(i,'left fragments',sum(1 for l in lanes if l.side=='left' and not l.reprojected),'reach',max([l.points[:,0].max() for l in lanes if l.side=='left' and not l.reprojected],default=0))
    cv2.imwrite(sys.argv[2],np.hstack(tiles))
if __name__=='__main__':main()
