#!/usr/bin/env python3
"""YOLOPv2 mask time per frame, as the model process runs it (torch 1 thread, half precision, CUDA): resize to
640x360 + inference + BEV remap, on 300 cached Town05 previews (960x540, resized the same way).
Prints mean / p95 ms of the whole call and of the inference alone."""
import sys,time,json
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import torch
import maploop_offline_compare as M
from vision_road.learned_mask import LaneModel,INPUT
torch.set_num_threads(1);cv2.setNumThreads(1)
pk=ROOT/'src/vision_road';proj=M.projector(pk,M.settings(pk))
model=LaneModel(str(ROOT/'data/models/yolopv2/yolopv2.pt'),proj,'cuda')
cache=Path('/media/jaehun/38F7-677B/vision-road-cache/town05_offline')
frames=[cv2.imread(str(cache/'preview'/f'{i:05d}.png')) for i in range(0,1800,6)]
whole=[];infer=[]
for k,f in enumerate(frames):
    small=cv2.resize(f,INPUT);s=time.perf_counter();model.image_mask(small);m=time.perf_counter()
    model.bev(small,(1080,1920));e=time.perf_counter()
    if k>=10:infer.append((m-s)*1e3);whole.append((e-m)*1e3)
r=dict(frames=len(whole),bev_call_mean_ms=round(float(np.mean(whole)),2),bev_call_p95_ms=round(float(np.percentile(whole,95)),2),
       inference_mean_ms=round(float(np.mean(infer)),2),inference_p95_ms=round(float(np.percentile(infer,95)),2),gpu=torch.cuda.get_device_name(0),
       max_memory_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20))
print(json.dumps(r));json.dump(r,open(sys.argv[1],'w'),indent=1)
