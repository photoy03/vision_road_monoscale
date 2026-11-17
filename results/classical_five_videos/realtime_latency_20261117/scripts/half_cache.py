"""Offline BEV cache from half-resolution camera images (1920x1080 -> 960x540, INTER_AREA, as a camera
streaming half size would): same calibration and BEV, the warp reads the smaller image.
usage: half_cache.py VIDEO OUT_CACHE   (frames.json and poses copied from the full-size cache)"""
import sys,json,shutil,cv2,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import maploop_offline_compare as M,realtime_eval as R
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from vision_road.zstd_cdr import cache_records
video,out=sys.argv[1],Path(sys.argv[2]);bag,cdr=R.VIDEOS[video];db=next(bag.glob('*.db3'))
full=Path('/media/jaehun/38F7-677B/vision-road-cache')/f'{video}_offline'
pk=ROOT/'src/vision_road';cfg=M.settings(pk);proj=M.projector(pk,cfg);proj.initialize(960,540)
(out/'bev').mkdir(parents=True,exist_ok=True)
for i,data in enumerate(cache_records(cdr,db)):
    m=deserialize_message(data,Image);frame=np.frombuffer(m.data,np.uint8).reshape(m.height,m.width,3)
    if m.encoding=='rgb8':frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    small=cv2.resize(frame,(960,540),interpolation=cv2.INTER_AREA)
    bev=proj.warp(small);valid=proj.valid.copy();valid[proj.my>=540*M.LAUNCH['image_roi_bottom']]=0
    cv2.imwrite(str(out/'bev'/f'{i:05d}.png'),cv2.bitwise_and(bev,bev,mask=valid),[cv2.IMWRITE_PNG_COMPRESSION,1])
    if i%300==0:print(video,i,flush=True)
cv2.imwrite(str(out/'valid.png'),proj.valid);shutil.copy(full/'frames.json',out/'frames.json')
