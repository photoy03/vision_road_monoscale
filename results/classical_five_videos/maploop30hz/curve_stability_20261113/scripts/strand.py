import sys,json,cv2,numpy as np
from pathlib import Path
from dataclasses import replace
sys.path.insert(0,'/home/jaehun/Desktop/vision-road/tools');sys.path.insert(0,'/home/jaehun/Desktop/vision-road/src/vision_road')
import maploop_offline_compare as M
from vision_road.carla import CarlaDetector
from vision_road.temporal import TemporalDetector
from vision_road.core import Timeline
def main():
    pk=Path('/home/jaehun/Desktop/vision-road/src/vision_road');cfg=M.settings(pk);proj=M.projector(pk,cfg);cache=Path(sys.argv[1])
    valid=cv2.imread(str(cache/'valid.png'),0);valid[proj.my>=1080*M.LAUNCH['image_roi_bottom']]=0
    meta=json.loads((cache/'frames.json').read_text());d=TemporalDetector(cfg);d.detector=CarlaDetector(cfg);tl=Timeline(.25);k=0
    rows=[];det=d.detector;orig=det.merge_strands
    def spy(nodes,sample,smooth,yellow,side=None):
        r=orig(nodes,sample,smooth,yellow,side)
        if side=='left':det._last=(r[0],det.strand_gap.get('left'),len(nodes))
        return r
    det.merge_strands=spy
    for i in range(600,1050):
        t=meta['frames'][i]['stamp']
        while k<len(meta['poses']) and meta['poses'][k][0]<=t+.05:tl.add(*meta['poses'][k]);k+=1
        det._last=None;d.process(cv2.imread(str(cache/'bev'/f'{i:05d}.png')),valid,t,tl.at(t))
        cand=[b for b in det.boundary_candidates if b.side=='left'];out=[b for b in d.boundary_lanes if b.side=='left' and not b.reprojected]
        def at(p,x):return float(np.interp(x,p[:,0],p[:,1])) if len(p) and p[0,0]<=x<=p[-1,0] else None
        rows.append((i,at(cand[0].points,6) if cand else None,at(out[0].points,6) if out else None,det._last))
    jumps=[(a,b) for a,b in zip(rows[:-1],rows[1:]) if a[2] is not None and b[2] is not None and abs(b[2]-a[2])>.05 and b[0]>=660]
    print('output jumps>5cm',len(jumps))
    cj=[(a,b) for a,b in zip(rows[:-1],rows[1:]) if a[1] is not None and b[1] is not None and abs(b[1]-a[1])>.05 and b[0]>=660];print('candidate (pre-temporal) jumps>5cm',len(cj))
    for a,b in jumps[:12]:print(b[0],'cand %.3f->%.3f out %.3f->%.3f'%(a[1] or 0,b[1] or 0,a[2],b[2]),'offset/gap/nodes',a[3] and (round(a[3][0],3),a[3][1] and round(a[3][1][0],2),a[3][2]),'->',b[3] and (round(b[3][0],3),b[3][1] and round(b[3][1][0],2),b[3][2]))
if __name__=='__main__':main()
