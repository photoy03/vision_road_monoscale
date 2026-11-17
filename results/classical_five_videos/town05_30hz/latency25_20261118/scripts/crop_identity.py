#!/usr/bin/env python3
"""Plan 25 step 1: the BEV from a row-cropped image equals the BEV from the full image, on every frame.
For each recorded image (the replay cache): BEV the perception worker makes from the full image (Projector.warp,
then the valid area above image_roi_bottom) against the BEV from rows LO:HI only (warp(frame, top=LO)), as
tools/replay_maploop.py --crop-rows sends them.  Equal BEVs give equal detector outputs (same input).
usage: crop_identity.py VIDEO [LO:HI]    writes ../offline/crop_identity_<video>.txt"""
import sys,time
from pathlib import Path
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import maploop_offline_compare as M
from realtime_eval import VIDEOS
from vision_road.zstd_cdr import cache_records
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
video=sys.argv[1];lo,hi=(int(v) for v in (sys.argv[2] if len(sys.argv)>2 else '578:984').split(':'))
pk=ROOT/'src/vision_road';cfg=M.settings(pk);p=M.projector(pk,cfg);p.initialize(*p.size)
roi=M.LAUNCH['image_roi_bottom'];valid=p.valid.copy();valid[p.my>=p.source_size[1]*roi]=0;need=p.rows(roi)
bag,cache=VIDEOS[video];db=next(bag.glob('*.db3'));n=same=0;diff=[];t0=time.time()
for raw in cache_records(cache,db):
    m=deserialize_message(raw,Image);f=np.frombuffer(m.data,np.uint8).reshape(m.height,m.width,3)
    a=p.warp(f);a=cv2.bitwise_and(a,a,mask=valid)
    b=p.warp(np.ascontiguousarray(f[lo:hi]),lo);b=cv2.bitwise_and(b,b,mask=valid)
    n+=1;eq=np.array_equal(a,b);same+=eq
    if not eq:diff.append(n-1)
line=f'{video}: rows read {need[0]}-{need[1]-1}, sent {lo}-{hi-1} ({hi-lo} of {p.source_size[1]} rows, {100*(hi-lo)/p.source_size[1]:.1f}%); BEV identical {same}/{n} frames'+(f'; differ {diff[:10]}' if diff else '')
print(line,f'({time.time()-t0:.0f} s)');(Path(__file__).resolve().parent.parent/'offline'/f'crop_identity_{video}.txt').write_text(line+'\n')
