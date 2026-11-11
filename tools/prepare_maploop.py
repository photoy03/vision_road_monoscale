#!/usr/bin/env python3
"""Read-only ROS sqlite extraction; cache calibrated BEV losslessly, not the bag."""
import argparse,json,sqlite3,sys,time,zipfile,hashlib
from pathlib import Path
import cv2,numpy as np,yaml
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo,Image
from geometry_msgs.msg import PoseWithCovarianceStamped
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src/vision_road'))
from vision_road.core import Settings,Projector,Timeline

def stamp(msg):return msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bag',type=Path,required=True);ap.add_argument('--handoff',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 cv2.setNumThreads(2);con=sqlite3.connect('file:'+str(a.bag.resolve())+'?mode=ro',uri=True)
 topics={name:(i,typ) for i,name,typ in con.execute('select id,name,type from topics')}
 def messages(topic):
  return con.execute('select timestamp,data from messages where topic_id=? order by timestamp',(topics[topic][0],))
 info=deserialize_message(next(messages('/sensing/camera/dashcam/camera_info'))[1],CameraInfo)
 with zipfile.ZipFile(a.handoff) as z:
  mount_data=z.read('handoff_maploop_30hz/calib/sensor_kit_calibration.yaml');(a.output/'sensor_kit_calibration.yaml').write_bytes(mount_data)
  (a.output/'HANDOFF_README.md').write_bytes(z.read('handoff_maploop_30hz/README.md'))
 kit=yaml.safe_load(mount_data);mount=kit['sensor_kit_base_link']['dashcam_link']
 assert mount['pitch']==0 and mount['roll']==0 and mount['yaw']==0 and mount['y']==0
 rig=dict(distortion_model='plumb_bob',camera_height_m=mount['z']+kit['actor_origin_height_m'],camera_forward_m=mount['x'],camera_left_m=0.,pitch_deg=0.,roll_deg=0.,yaw_deg=0.)
 np.savez(a.output/'dashcam.npz',K=np.array(info.k).reshape(3,3),D=np.array(info.d),image_size=np.array([info.width,info.height]))
 (a.output/'rig.yaml').write_text(yaml.safe_dump(rig));settings=yaml.safe_load((ROOT/'src/vision_road/config/carla.yaml').read_text())['vision_road']['ros__parameters'];cfg=Settings(**{k:v for k,v in settings.items() if k in Settings.__dataclass_fields__})
 projector=Projector(a.output/'dashcam.npz',rig,cfg);timeline=Timeline(.10);truths=[]
 for _,data in messages('/carla/ground_truth/pose'):
  msg=deserialize_message(data,PoseWithCovarianceStamped);p=msg.pose.pose.position;q=msg.pose.pose.orientation;yaw=np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
  # Supplied handoff: truth actor origin -> rear axle base_link.
  x=p.x-1.4096*np.cos(yaw);y=p.y-1.4096*np.sin(yaw);t=stamp(msg);timeline.add(t,x,y,yaw);truths.append([t,x,y,yaw])
 timeline.samples=truths
 (a.output/'truth_base.json').write_text(json.dumps(truths));frames=[];start=time.monotonic()
 for folder in ['bev','preview','ros_samples']:(a.output/folder).mkdir(exist_ok=True)
 # Iterate once. Images use header simulation time, not recording wall time.
 origin=None
 for i,(record_time,data) in enumerate(messages('/sensing/camera/dashcam/image_raw')):
  msg=deserialize_message(data,Image);assert msg.encoding=='bgr8'
  frame=np.frombuffer(msg.data,np.uint8).reshape(msg.height,msg.step)[:,:msg.width*3].reshape(msg.height,msg.width,3)
  t=stamp(msg)
  if origin is None:origin=t
  bev=projector.warp(frame);valid=projector.valid.copy();valid[projector.my>=.915*msg.height]=0;bev=cv2.bitwise_and(bev,bev,mask=valid)
  cv2.imwrite(str(a.output/'bev'/f'{i:05d}.png'),bev,[cv2.IMWRITE_PNG_COMPRESSION,2]);cv2.imwrite(str(a.output/'preview'/f'{i:05d}.jpg'),cv2.resize(frame,(960,540)),[cv2.IMWRITE_JPEG_QUALITY,94])
  if 300<=i<324:(a.output/'ros_samples'/f'{i:05d}.cdr').write_bytes(data)
  pose=timeline.at(t);assert pose is not None
  frames.append(dict(frame=i,stamp=t,seconds=t-origin,record_timestamp=record_time,pose=list(pose)))
  if i%100==0:
   print(i,round(t-origin,3),round(time.monotonic()-start,1),flush=True)
   cv2.imwrite(str(a.output/'valid.png'),valid)
 (a.output/'frames.json').write_text(json.dumps(frames));cv2.imwrite(str(a.output/'valid.png'),valid)
 (a.output/'config.json').write_text(json.dumps(dict(bag=str(a.bag),bag_untouched=True,image_topic='/sensing/camera/dashcam/image_raw',native_resolution=[info.width,info.height],header_frame=info.header.frame_id,distortion=info.distortion_model,K=list(info.k),D=list(info.d),rig=rig,hood_bottom_fraction=.915,truth_is_reference_not_estimated_odometry=True,truth_to_base_forward_offset_m=1.4096,frames=len(frames),simulation_duration=frames[-1]['seconds'],record_duration_sec=(frames[-1]['record_timestamp']-frames[0]['record_timestamp'])/1e9,elapsed_sec=time.monotonic()-start),indent=2))
 print('done',len(frames),flush=True)
if __name__=='__main__':main()
