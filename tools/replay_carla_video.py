#!/usr/bin/env python3
"""Replay the supplied CARLA clip with original capture intervals and /clock."""
import argparse,json,time,subprocess
from pathlib import Path
import cv2,rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,ReliabilityPolicy
from cv_bridge import CvBridge
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from builtin_interfaces.msg import Time
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--video',type=Path,default=ROOT/'data/simulation/2026_11_11_16_21.mp4');ap.add_argument('--rate',type=float,default=.5);ap.add_argument('--max-frames',type=int,default=0);ap.add_argument('--topic',default='/carla/ego_vehicle/rgb_front/image');a=ap.parse_args()
    if a.rate<=0:raise ValueError('rate must be positive')
    frames=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(a.video)]))['frames'];pts=[float(f['best_effort_timestamp_time']) for f in frames]
    rclpy.init();node=Node('carla_video_replay');pub=node.create_publisher(Image,a.topic,QoSProfile(depth=10,reliability=ReliabilityPolicy.RELIABLE));clock=node.create_publisher(Clock,'/clock',10);bridge=CvBridge();cap=cv2.VideoCapture(str(a.video));sent=0
    try:
        deadline=time.monotonic()+15
        while pub.get_subscription_count()<1 and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.1)
        if pub.get_subscription_count()<1:raise RuntimeError('Start carla.launch.py before replay')
        start=time.monotonic()+1.
        for i,t in enumerate(pts):
            if a.max_frames and i>=a.max_frames:break
            due=start+(t-pts[0])/a.rate
            while time.monotonic()<due:rclpy.spin_once(node,timeout_sec=min(.02,max(0,due-time.monotonic())))
            ok,im=cap.read()
            if not ok:raise RuntimeError('Incomplete video decode')
            ns=round((1.+t-pts[0])*1e9);stamp=Time(sec=ns//1000000000,nanosec=ns%1000000000)
            clock.publish(Clock(clock=stamp));msg=bridge.cv2_to_imgmsg(im,'bgr8');msg.header.stamp=stamp;msg.header.frame_id='ego_vehicle/rgb_front';pub.publish(msg);rclpy.spin_once(node,timeout_sec=0);sent+=1
        print('Published',sent,'frames; rate',a.rate,'No odometry synthesized.')
        end=time.monotonic()+1.
        while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.05)
    finally:cap.release();node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
