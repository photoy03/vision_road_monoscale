"""Per-frame cost outside ROS on cached frames: native image -> BEV warp (from the CDR cache) and
TemporalDetector.process (carla.yaml), opencv threads 2 as the node.  usage: offline_cost.py VIDEO FIRST LAST"""
import sys,json,time,cv2,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import maploop_offline_compare as M,realtime_eval as R
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from vision_road.zstd_cdr import cache_records
from vision_road.core import Timeline
from vision_road.temporal import TemporalDetector
from vision_road.carla import CarlaDetector
video,a,b=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]);bag,cache=R.VIDEOS[video];db=next(bag.glob('*.db3'))
cv2.setNumThreads(2);pk=ROOT/'src/vision_road';cfg=M.settings(pk);proj=M.projector(pk,cfg)
meta=json.loads((Path('/media/jaehun/38F7-677B/vision-road-cache')/f'{video}_offline'/'frames.json').read_text())
det=TemporalDetector(cfg);det.detector=CarlaDetector(cfg);tl=Timeline(.25);k=0;warp=[];proc=[];conv=[]
for i,data in enumerate(cache_records(cache,db,a,b-a)):
    t0=time.perf_counter();msg=deserialize_message(data,Image);t1=time.perf_counter()
    frame=np.frombuffer(msg.data,np.uint8).reshape(msg.height,msg.width,3)
    if msg.encoding=='rgb8':frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    t2=time.perf_counter();bev=proj.warp(frame);valid=proj.valid.copy();valid[proj.my>=frame.shape[0]*M.LAUNCH['image_roi_bottom']]=0;bev=cv2.bitwise_and(bev,bev,mask=valid);t3=time.perf_counter()
    t=meta['frames'][a+i]['stamp']
    while k<len(meta['poses']) and meta['poses'][k][0]<=t+.05:tl.add(*meta['poses'][k]);k+=1
    det.process(bev,valid,t,tl.at(t));t4=time.perf_counter()
    conv.append((t2-t1)*1e3);warp.append((t3-t2)*1e3);proc.append((t4-t3)*1e3)
s=lambda v:f'median {np.median(v):.2f} p95 {np.percentile(v,95):.2f} max {max(v):.2f}'
print(video,a,b,'encoding',msg.encoding,'| to-bgr',s(conv),'| warp',s(warp),'| detector',s(proc))
