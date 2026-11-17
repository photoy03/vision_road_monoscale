#!/usr/bin/env python3
"""Plan 26: the YOLOPv2 BEV lane mask from the full image against the mask from a row-cropped image (rows
578-983 only, black elsewhere: what the model sees with --crop-rows and the AI filter on), every Town05 frame.
Writes ../offline/learned_{full,crop}.npz (cache format of maploop_offline_compare.py learned) and per frame the
differing pixels, inside the gated range (0-15 m ahead, where the gate removes paint) and in all.
usage: ai_crop_masks.py [VIDEO]"""
import sys,json
from pathlib import Path
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import maploop_offline_compare as M
from realtime_eval import VIDEOS
from vision_road.zstd_cdr import cache_records
from vision_road.learned_mask import LaneModel
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
video=sys.argv[1] if len(sys.argv)>1 else 'town05';LO,HI=578,984
pk=ROOT/'src/vision_road';cfg=M.settings(pk);proj=M.projector(pk,cfg);model=LaneModel(str(ROOT/'data/models/yolopv2/yolopv2.pt'),proj)
bag,cache=VIDEOS[video];db=next(bag.glob('*.db3'));full=[];crop=[];rows=[];far=round((cfg.forward_max-15.)/cfg.mpp)
padded=None
for i,raw in enumerate(cache_records(cache,db)):
    m=deserialize_message(raw,Image);f=np.frombuffer(m.data,np.uint8).reshape(m.height,m.width,3)
    if padded is None:padded=np.zeros_like(f)
    padded[:]=0;padded[LO:HI]=f[LO:HI]
    a=model.bev(f)>0;b=model.bev(padded)>0;full.append(np.packbits(a,axis=1));crop.append(np.packbits(b,axis=1))
    g=slice(far,None);ua=a[g]|b[g];rows.append(dict(diff=int(np.count_nonzero(a^b)),diff_gate=int(np.count_nonzero(a[g]^b[g])),
        full=int(a.sum()),full_gate=int(a[g].sum()),iou_gate=float((a[g]&b[g]).sum()/max(1,ua.sum()))))
    if i%300==0:print(i,rows[-1],flush=True)
shape=np.array(proj.valid.shape);O=Path(__file__).resolve().parent.parent/'offline'
np.savez_compressed(O/f'learned_full_{video}.npz',masks=np.stack(full),shape=shape);np.savez_compressed(O/f'learned_crop_{video}.npz',masks=np.stack(crop),shape=shape)
d=np.array([r['diff_gate'] for r in rows]);n=np.array([r['full_gate'] for r in rows]);iou=np.array([r['iou_gate'] for r in rows])
summary=dict(frames=len(rows),gate_pixels_mean=float(n.mean()),diff_gate_mean=float(d.mean()),diff_gate_p95=float(np.percentile(d,95)),diff_gate_share_of_mask=float(d.sum()/max(1,n.sum())),
             iou_gate_median=float(np.median(iou)),iou_gate_p5=float(np.percentile(iou,5)),frames_any_gate_diff=int((d>0).sum()))
json.dump(dict(summary=summary,frames=rows),open(Path(__file__).resolve().parent.parent/'metrics'/f'ai_crop_masks_{video}.json','w'))
print(json.dumps(summary))
