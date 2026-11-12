import sys,json,time,cv2,numpy as np
from pathlib import Path
from dataclasses import replace
sys.path.insert(0,'/home/jaehun/Desktop/vision-road/tools');sys.path.insert(0,'/home/jaehun/Desktop/vision-road/src/vision_road')
import maploop_offline_compare as M
from vision_road.carla import CarlaDetector
from vision_road.temporal import TemporalDetector
from vision_road.core import Timeline
pk=Path('/home/jaehun/Desktop/vision-road/src/vision_road');cache=Path(sys.argv[1])
base=M.settings(pk);mask_at=M.learned_masks(cache/'learned.npz');meta=json.loads((cache/'frames.json').read_text())
bevs=[cv2.imread(str(cache/'bev'/f'{i:05d}.png')) for i in range(600,1260)]
for name,extra in (('off',dict(carla_strand_pair=False,carla_strand_memory_m=0.)),('on',{}),('off2',dict(carla_strand_pair=False,carla_strand_memory_m=0.)),('on2',{})):
    cfg=replace(base,carla_learned_gate='remove',carla_learned_gate_far_m=15.,**extra);proj=M.projector(pk,cfg)
    valid=cv2.imread(str(cache/'valid.png'),0);valid[proj.my>=1080*M.LAUNCH['image_roi_bottom']]=0
    d=TemporalDetector(cfg);d.detector=CarlaDetector(cfg);tl=Timeline(.25);k=0;det=d.detector
    spent=[0.];orig=det.merge_strands
    def timed(*a,**kw):
        t0=time.perf_counter();r=orig(*a,**kw);spent[0]+=time.perf_counter()-t0;return r
    det.merge_strands=timed;total=0.
    for j,i in enumerate(range(600,1260)):
        t=meta['frames'][i]['stamp']
        while k<len(meta['poses']) and meta['poses'][k][0]<=t+.05:tl.add(*meta['poses'][k]);k+=1
        det.learned=mask_at(i-1);t0=time.perf_counter();d.process(bevs[j],valid,t,tl.at(t));total+=time.perf_counter()-t0
    n=660;print(name,'process %.2f ms/frame, merge_strands %.3f ms/frame'%(total/n*1e3,spent[0]/n*1e3),flush=True)
