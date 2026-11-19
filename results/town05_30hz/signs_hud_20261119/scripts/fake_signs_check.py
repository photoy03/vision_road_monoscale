#!/usr/bin/env python3
"""Plan 27 check without a recording: lane_guidance (signs.enabled) and the HUD copy (lane-only, signals on) run,
this script publishes empty lane centres at 30 Hz and made-up /yolo/detections, and records the scenes.
Phases (s): 0-2 red light, 2-4 yellow (orange), 4-7 green + 60 sign, 7-9 nothing (sign held 3 s), 9-11 one-frame
flickers (a red every 5th frame).  usage: fake_signs_check.py OUT.json"""
import sys,json,time,rclpy
from rclpy.node import Node
from std_msgs.msg import String
from hero_hud.msg import SceneOverlay
rclpy.init();n=Node('fake_signs_check');scenes=[]
n.create_subscription(SceneOverlay,'/hero/scene_overlay',lambda m:scenes.append(dict(t=m.header.stamp.sec+m.header.stamp.nanosec*1e-9,
    signal=int(m.signal.state) if m.has_signal else None,sign=float(m.sign.value) if m.has_sign else None)),50)
cen=n.create_publisher(String,'/road/lane_centres',10);det=n.create_publisher(String,'/yolo/detections',10)
end=time.time()+15
while time.time()<end and n.count_subscribers('/yolo/detections')<1:rclpy.spin_once(n,timeout_sec=.1)
t0=100.;k=0
def D(name):return dict(class_id=0,class_name=name,confidence=.8,xyxy=[900,100,930,180])
while k<11*30:
    t=t0+k/30.;ph=k/30.;d=[]
    if ph<2:d=[D('traffic_light_red')]
    elif ph<4:d=[D('traffic_light_orange')]
    elif ph<7:d=[D('traffic_light_green'),D('traffic_sign_60')]
    elif ph<9:d=[]
    else:d=[D('traffic_light_red')] if k%5==0 else []
    sec=int(t);ns=int(round((t-sec)*1e9))
    det.publish(String(data=json.dumps(dict(stamp=dict(sec=sec,nanosec=ns),frame_id='x',image_width=1920,image_height=1080,detections=d))))
    rclpy.spin_once(n,timeout_sec=.005)
    cen.publish(String(data=json.dumps(dict(stamp=t,frame_id='base_link',lanes=[]))))
    k+=1;end=time.time()+1/30.
    while time.time()<end:rclpy.spin_once(n,timeout_sec=.005)
for _ in range(20):rclpy.spin_once(n,timeout_sec=.05)
json.dump(scenes,open(sys.argv[1],'w'));print('scenes',len(scenes))
