import sys,json,cv2,numpy as np
from pathlib import Path
from dataclasses import replace
sys.path.insert(0,'/home/jaehun/Desktop/vision-road/tools');sys.path.insert(0,'/home/jaehun/Desktop/vision-road/src/vision_road')
import maploop_offline_compare as M
from vision_road.carla import CarlaDetector
from vision_road.temporal import TemporalDetector
from vision_road.core import Timeline
pk=Path('/home/jaehun/Desktop/vision-road/src/vision_road');cache=Path(sys.argv[1])
cfg=replace(M.settings(pk),carla_learned_gate='remove',carla_learned_gate_far_m=15.);proj=M.projector(pk,cfg)
valid=cv2.imread(str(cache/'valid.png'),0);valid[proj.my>=1080*M.LAUNCH['image_roi_bottom']]=0
mask_at=M.learned_masks(cache/'learned.npz');meta=json.loads((cache/'frames.json').read_text())
d=TemporalDetector(cfg);d.detector=CarlaDetector(cfg);tl=Timeline(.25);k=0;det=d.detector
orig=det.pair_vote;origr=det.recall_pairs
def spy(nodes,sample,smooth,side):
    det._fixed=None;r=orig(nodes,sample,smooth,side)
    if side=='left' and r is not None:print('FRAME',det.frame_index,[(round(float(a['lane'].points[0,0]),1),round(float(a['lane'].points[-1,0]),1),round(float(a['lane'].points[:,1].mean()),2),a['yellow'],len(a['pixels'][0])) for a in nodes],det.diagnostics.get('strand_labels_left'))
    if side=='left':det._v=(r is not None,len(nodes),det.diagnostics.get('strand_labels_left'),det._fixed,det.strand_gap.get('left'),[round(float(np.ptp(a['lane'].points[:,0])),1) for a in nodes])
    return r
def spyr(*a):
    f=origr(*a);det._fixed=sorted(f);return f
det.pair_vote=spy;det.recall_pairs=spyr
for i in range(1400,1530):
    t=meta['frames'][i]['stamp']
    while k<len(meta['poses']) and meta['poses'][k][0]<=t+.05:tl.add(*meta['poses'][k]);k+=1
    det._v=None;det.diagnostics.pop('strand_labels_left',None);det.learned=mask_at(i-1)
    d.process(cv2.imread(str(cache/'bev'/f'{i:05d}.png')),valid,t,tl.at(t))

