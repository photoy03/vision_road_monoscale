#!/usr/bin/env python3
"""Plan 28 check without a recording: lane_guidance (stop_line.enabled, signs.enabled) and the HUD copy (lane-only)
run; this script publishes at 30 Hz two straight display lines (+-1.75 m), empty lane centres, a stop line coming
closer (20 m -> 4 m over 4 s), then a crosswalk (4-6 s, 8-12 m ahead), an arrow marking (should not be drawn),
and a green light (4-6 s) via /yolo/detections; it records /hero/lane_guidance's stop_line_m.
usage: fake_markings_check.py OUT.json"""
import sys,json,time,rclpy
from rclpy.node import Node
from std_msgs.msg import String,Header
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker,MarkerArray
rclpy.init();n=Node('fake_markings_check');info=[]
n.create_subscription(String,'/hero/lane_guidance',lambda m:info.append(json.loads(m.data)),50)
disp=n.create_publisher(MarkerArray,'/road/lane_display_markers',10);mk=n.create_publisher(MarkerArray,'/road/road_markings',10)
cen=n.create_publisher(String,'/road/lane_centres',10);det=n.create_publisher(String,'/yolo/detections',10)
end=time.time()+15
while time.time()<end and n.count_subscribers('/road/road_markings')<2:rclpy.spin_once(n,timeout_sec=.1)
def strip(head,ns,i,pts,predicted=False,w=.05):
    m=Marker();m.header=head;m.ns=ns;m.id=i;m.type=Marker.LINE_STRIP;m.action=Marker.ADD;m.pose.orientation.w=1.;m.scale.x=w
    m.color.a=.9;m.color.g=1.;m.color.b=1.;m.color.r=1. if predicted else 0.;m.points=[Point(x=float(x),y=float(y),z=.02) for x,y in pts];return m
t0=200.
for k in range(7*30):
    t=t0+k/30.;ph=k/30.;sec=int(t);ns=int(round((t-sec)*1e9));head=Header(stamp=Time(sec=sec,nanosec=ns),frame_id='base_link')
    clear=Marker();clear.header=head;clear.action=Marker.DELETEALL
    d=MarkerArray(markers=[clear,strip(head,'display/left',0,[(x,1.75) for x in range(0,21)]),strip(head,'display/right',1,[(x,-1.75) for x in range(0,21)])])
    marks=[clear]
    if ph<4:x=20-4*ph;marks.append(strip(head,'stop_line',1,[(x,1.6),(x,-1.6)],w=.3))
    if 4<=ph<6:marks.append(strip(head,'crosswalk',2,[(8,1.6),(12,1.6),(12,-1.6),(8,-1.6),(8,1.6)],w=.1))
    marks.append(strip(head,'marking',3,[(6,.3),(7.5,.3),(7.5,-.3),(6,-.3),(6,.3)],w=.1))
    disp.publish(d);mk.publish(MarkerArray(markers=marks))
    dets=[dict(class_id=0,class_name='traffic_light_green',confidence=.8,xyxy=[900,100,930,180])] if 4<=ph<6 else []
    det.publish(String(data=json.dumps(dict(stamp=dict(sec=sec,nanosec=ns),frame_id='x',image_width=1920,image_height=1080,detections=dets))))
    cen.publish(String(data=json.dumps(dict(stamp=t,frame_id='base_link',lanes=[]))))
    e=time.time()+1/30.
    while time.time()<e:rclpy.spin_once(n,timeout_sec=.005)
for _ in range(20):rclpy.spin_once(n,timeout_sec=.05)
json.dump([dict(t=round(r['stamp']-t0,3),stop_line_m=r.get('stop_line_m'),light=r.get('light')) for r in info],open(sys.argv[1],'w'));print('guidance msgs',len(info))
