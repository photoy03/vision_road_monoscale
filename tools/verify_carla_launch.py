#!/usr/bin/env python3
"""Actual ROS launch/messages using supplied CARLA frames; no simulator server needed."""
import os,signal,subprocess,time,json,argparse
from pathlib import Path
import cv2,rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray,Marker
from vision_lane_interfaces.msg import MarkingSegmentArray,LaneStyleArray
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    process=None;rclpy.init();node=Node('carla_validation_reader');bridge=CvBridge();local=[];styles=[];global_map=[];binary=[];paint=[];boundary=[]
    subscriptions=[node.create_subscription(MarkerArray,'/road/lane_markers',paint.append,10),node.create_subscription(MarkerArray,'/road/lane_boundary_markers',boundary.append,10),node.create_subscription(Image,'/parking/cpp/binary',binary.append,qos_profile_sensor_data),node.create_subscription(MarkingSegmentArray,'/parking/cpp/marking_segments',local.append,10),node.create_subscription(LaneStyleArray,'/road/lane_styles',styles.append,10),node.create_subscription(MarkingSegmentArray,'/parking/cpp/global_marking_segments_fused',global_map.append,10)]
    image=node.create_publisher(Image,'/carla/ego_vehicle/rgb_front/image',10);odom=node.create_publisher(Odometry,'/carla/ego_vehicle/odometry',10)
    def pump(duration):
        end=time.monotonic()+duration
        while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.02)
    try:
        with (args.output/'launch.log').open('w') as log:
            process=subprocess.Popen(['ros2','launch','vision_road','carla.launch.py','use_sim_time:=false'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+15
            while image.get_subscription_count()<1 and time.monotonic()<deadline:
                if process.poll() is not None:raise RuntimeError('CARLA launch exited')
                pump(.1)
            assert image.get_subscription_count()>0
            # Wait for the output subscriber discovery too, before counting inputs.
            deadline=time.monotonic()+8
            while node.count_publishers('/parking/cpp/marking_segments')<1 and time.monotonic()<deadline:pump(.1)
            assert node.count_publishers('/parking/cpp/marking_segments')>0
            pump(1.)
            cap=cv2.VideoCapture(str(ROOT/'data/simulation/2026_11_11_16_21.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,200)
            for i in range(18):
                ok,frame=cap.read();assert ok
                stamp=node.get_clock().now().to_msg();pose=Odometry();pose.header.stamp=stamp;pose.header.frame_id='map';pose.child_frame_id='ego_vehicle';pose.pose.pose.orientation.w=1.;odom.publish(pose)
                msg=bridge.cv2_to_imgmsg(frame,'bgr8');msg.header.stamp=stamp;msg.header.frame_id='ego_vehicle/rgb_front';image.publish(msg);pump(.25)
            cap.release();pump(1.)
            assert len(local)==18 and len(styles)==18,(len(local),len(styles))
            assert all(m.header.frame_id=='ego_vehicle' for m in local)
            assert any(m.segments for m in local)
            assert any(m.pose_valid for m in local)
            assert len(paint)==18 and len(boundary)==18
            assert all(m.markers[0].action==Marker.DELETEALL for m in boundary)
            assert any(len(m.markers)>1 for m in boundary)
            assert all(len(m.markers)<=3 for m in boundary)
            assert binary and all(m.header.frame_id=='ego_vehicle' for m in binary)
            assert not global_map,'Mapping must be disabled for provisional camera profile'
            assert '/road/centerline' not in dict(node.get_topic_names_and_types())
            report=dict(paint_marker_arrays=len(paint),boundary_marker_arrays=len(boundary),input_frames=18,lane_arrays=len(local),style_arrays=len(styles),frames_with_segments=sum(bool(m.segments) for m in local),frames_with_pose=sum(m.pose_valid for m in local),global_arrays=len(global_map),output_frame='ego_vehicle',binary_frames=len(binary),note='CARLA-shaped image/odom topics through actual launch; supplied RGB frames and stationary synthetic odometry. No CARLA server, actual TF or metric calibration validated.')
            (args.output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:process.wait(timeout=8)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=5)
        node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
