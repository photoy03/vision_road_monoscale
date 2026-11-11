#!/usr/bin/env python3
"""Native bag CDR images -> actual launch -> existing and standard display topics."""
import argparse,json,math,os,signal,subprocess,time
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovarianceStamped
from vision_lane_interfaces.msg import MarkingSegmentArray,LaneStyleArray
from visualization_msgs.msg import MarkerArray,Marker

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 rclpy.init();node=Node('maploop_output_verifier');local=[];styles=[];paint=[];boundary=[];overlays=[];global_map=[]
 subs=[node.create_subscription(typ,topic,target.append,qos) for typ,topic,target,qos in [
  (MarkingSegmentArray,'/parking/cpp/marking_segments',local,10),(LaneStyleArray,'/road/lane_styles',styles,10),
  (MarkerArray,'/road/lane_markers',paint,10),(MarkerArray,'/road/lane_boundary_markers',boundary,10),
  (Image,'/road/overlay',overlays,qos_profile_sensor_data),(MarkingSegmentArray,'/parking/cpp/global_marking_segments_fused',global_map,10)]]
 image=node.create_publisher(Image,'/sensing/camera/dashcam/image_raw',10);truth=node.create_publisher(PoseWithCovarianceStamped,'/carla/ground_truth/pose',10)
 def pump(seconds):
  until=time.monotonic()+seconds
  while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.02)
 process=None
 try:
  with (a.output/'launch.log').open('w') as log:
   process=subprocess.Popen(['ros2','launch','vision_road','maploop.launch.py','use_sim_time:=false'],stdout=log,stderr=log,start_new_session=True)
   deadline=time.monotonic()+20
   while time.monotonic()<deadline and (image.get_subscription_count()<1 or truth.get_subscription_count()<1):
    if process.poll() is not None:raise RuntimeError('Launch exited')
    pump(.1)
   assert image.get_subscription_count() and truth.get_subscription_count();pump(1.)
   frames=json.loads((a.input/'frames.json').read_text());count=24
   for row in frames[300:300+count]:
    msg=deserialize_message((a.input/'ros_samples'/f'{row["frame"]:05d}.cdr').read_bytes(),Image)
    x,y,yaw=row['pose'];pose=PoseWithCovarianceStamped();pose.header.stamp=msg.header.stamp;pose.header.frame_id='map';pose.pose.pose.position.x=x+1.4096*math.cos(yaw);pose.pose.pose.position.y=y+1.4096*math.sin(yaw);pose.pose.pose.orientation.z=math.sin(yaw/2);pose.pose.pose.orientation.w=math.cos(yaw/2)
    truth.publish(pose);pump(.06);image.publish(msg);pump(.30)
   pump(1.)
   assert len(local)==len(styles)==len(paint)==len(boundary)==count,(len(local),len(styles),len(paint),len(boundary))
   assert all(m.header.frame_id=='base_link' and m.pose_valid for m in local)
   assert any(m.segments for m in local) and not global_map
   expected=frames[323]['pose'];observed=local[-1].observer_pose
   assert abs(observed.x-expected[0])<1e-5 and abs(observed.y-expected[1])<1e-5
   assert abs(math.atan2(math.sin(observed.theta-expected[2]),math.cos(observed.theta-expected[2])))<1e-5
   assert all(m.markers[0].action==Marker.DELETEALL and len(m.markers)<=3 for m in boundary)
   assert any(len(m.markers)>1 for m in boundary)
   assert overlays and overlays[-1].width==960 and overlays[-1].height==540 and overlays[-1].encoding=='bgr8'
   assert '/road/centerline' not in dict(node.get_topic_names_and_types())
   report=dict(input_frames=count,local_arrays=len(local),style_arrays=len(styles),paint_marker_arrays=len(paint),boundary_marker_arrays=len(boundary),overlays=len(overlays),pose_valid=sum(m.pose_valid for m in local),nonempty=sum(bool(m.segments) for m in local),frame='base_link',global_map_arrays=len(global_map),note='Recorded native CDR images; actor reference reconstructed from interpolated recorded truth poses. Actual ROS launch/serialization. No CARLA server or MonoScale accuracy measured.')
   (a.output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
 finally:
  if process is not None and process.poll() is None:
   process.send_signal(signal.SIGINT)
   try:process.wait(timeout=8)
   except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=5)
  node.destroy_node();rclpy.try_shutdown()
if __name__=='__main__':main()
