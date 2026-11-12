import sys,json,cv2,numpy as np
from pathlib import Path
from dataclasses import replace
sys.path.insert(0,'/home/jaehun/Desktop/vision-road/tools');sys.path.insert(0,'/home/jaehun/Desktop/vision-road/src/vision_road')
import maploop_offline_compare as M
from vision_road.carla import CarlaDetector
from vision_road.temporal import TemporalDetector
from vision_road.core import Timeline
sets=dict(carla_learned_gate='remove',carla_learned_gate_far_m=15.,carla_strand_pair=True)
for kv in sys.argv[2:]:
    k,v=kv.split('=');sets[k]=type(getattr(M.settings(Path('/home/jaehun/Desktop/vision-road/src/vision_road')),k))(v) if v not in('true','false') else v=='true'
pk=Path('/home/jaehun/Desktop/vision-road/src/vision_road');cfg=replace(M.settings(pk),**sets);proj=M.projector(pk,cfg);cache=Path(sys.argv[1])
valid=cv2.imread(str(cache/'valid.png'),0);valid[proj.my>=1080*M.LAUNCH['image_roi_bottom']]=0
mask_at=M.learned_masks(cache/"learned.npz")
meta=json.loads((cache/'frames.json').read_text());d=TemporalDetector(cfg);d.detector=CarlaDetector(cfg);tl=Timeline(.25);k=0
det=d.detector;orig=det.pair_vote
def spy(nodes,sample,smooth,side):
    r=orig(nodes,sample,smooth,side)
    if side=='left':
        near=min(range(len(nodes)),key=lambda i:nodes[i]['lane'].points[0,0])
        det._v=dict(hold=det.strand_hold.get('left'),voted=r is not None,n=len(nodes),labels=det.diagnostics.get('strand_labels_left') if r is not None else None,
                    near=(round(float(nodes[near]['lane'].points[0,0]),1),round(float(nodes[near]['lane'].points[-1,0]),1)),gap=det.strand_gap.get('left'))
    return r
det.pair_vote=spy
rows=[]
for i in range(0,1050):
    t=meta['frames'][i]['stamp']
    while k<len(meta['poses']) and meta['poses'][k][0]<=t+.05:tl.add(*meta['poses'][k]);k+=1
    det._v=None;det.diagnostics.pop('strand_labels_left',None)
    det.learned=mask_at(i-1) if i>=1 else None
    d.process(cv2.imread(str(cache/'bev'/f'{i:05d}.png')),valid,t,tl.at(t))
    if i<640:continue
    cand=[b for b in det.boundary_candidates if b.side=='left'];out=[b for b in d.boundary_lanes if b.side=='left' and not b.reprojected]
    def at(p,x):return float(np.interp(x,p[:,0],p[:,1])) if len(p) and p[0,0]<=x<=p[-1,0] else None
    rows.append((i,at(cand[0].points,6) if cand else None,at(out[0].points,6) if out else None,det._v))
for a,b in zip(rows[:-1],rows[1:]):
    if b[0]<660:continue
    if a[2] is not None and b[2] is not None and abs(b[2]-a[2])>.05:
        print(b[0],'cand %s->%s out %.3f->%.3f'%(a[1] and round(a[1],3),b[1] and round(b[1],3),a[2],b[2]),'|',a[3] and (a[3]['hold'],a[3]['labels'][:6]),'->',b[3] and (b[3]['hold'],b[3]['labels'][:6]))
print('fallback frames',sum(1 for r in rows if r[3] and not r[3]['voted']),'of',len(rows))
