#!/usr/bin/env python3
"""Plan 29 check without a recording.  Runs against lane_guidance (signs, stop_line, traffic enabled), the
traffic_sign_bridge and the HUD copy (lane-only).  This script publishes at 30 Hz for 9 s, the car driving at
8 m/s along x (/road/reference_odom, /hero/vehicle_state):
  - lane centres: the own lane 0.6 m to the left of the car (the guidance shows a return ribbon);
  - display lines at +-1.75 m around that lane;
  - a stop line ahead, 30 m -> 0 m from the rear axle (moving with the car), from 1 s;
  - /yolo/detections: a red light 0-6 s, then green.
Records /hero/lane_guidance (state, shown, path end, traffic) and the scene's ideal end.
usage: fake_red_approach.py OUT.json"""
import sys,json,time,math,rclpy
from rclpy.node import Node
from std_msgs.msg import String,Header
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker,MarkerArray
from hero_hud.msg import SceneOverlay,VehicleState
rclpy.init();n=Node('fake_red_approach');info=[];scenes=[]
n.create_subscription(String,'/hero/lane_guidance',lambda m:info.append(json.loads(m.data)),50)
scenes_full=[]
from rosidl_runtime_py import message_to_ordereddict
n.create_subscription(SceneOverlay,'/hero/scene_overlay',lambda m:scenes_full.append(json.loads(json.dumps(message_to_ordereddict(m)))),50)
n.create_subscription(SceneOverlay,'/hero/scene_overlay',lambda m:scenes.append(dict(t=m.header.stamp.sec+m.header.stamp.nanosec*1e-9,
    ideal_end=max([p.x for p in m.ideal.points]) if m.has_ideal and m.ideal.points else None,signal=int(m.signal.state) if m.has_signal else None)),50)
cen=n.create_publisher(String,'/road/lane_centres',10);det=n.create_publisher(String,'/yolo/detections',10)
mk=n.create_publisher(MarkerArray,'/road/road_markings',10);disp=n.create_publisher(MarkerArray,'/road/lane_display_markers',10)
od=n.create_publisher(Odometry,'/road/reference_odom',10);vs=n.create_publisher(VehicleState,'/hero/vehicle_state',10)
end=time.time()+20
while time.time()<end and (n.count_subscribers('/road/lane_centres')<1 or n.count_subscribers('/yolo/detections')<1):rclpy.spin_once(n,timeout_sec=.1)
time.sleep(1.)
def strip(head,ns,i,pts,w=.05):
    m=Marker();m.header=head;m.ns=ns;m.id=i;m.type=Marker.LINE_STRIP;m.action=Marker.ADD;m.pose.orientation.w=1.;m.scale.x=w
    m.color.a=.9;m.color.g=1.;m.color.b=1.;m.points=[Point(x=float(x),y=float(y),z=.02) for x,y in pts];return m
t0=300.;v=8.;y0=.6
lane=lambda name,off,opp=False:dict(name=name,points=[[x,off] for x in range(0,31)],ref_x_m=5.,offset_m=off,heading_rad=0.,curvature_1pm=0.,width_m=3.5,
    opposite_direction=opp,lane_like=True,left_boundary=dict(style='dashed',yellow=False),right_boundary=dict(style='dashed',yellow=False),both_m=[0.,30.])
for k in range(9*30):
    t=t0+k/30.;ph=k/30.;sec=int(t);ns=int(round((t-sec)*1e9));head=Header(stamp=Time(sec=sec,nanosec=ns),frame_id='base_link')
    o=Odometry();o.header=Header(stamp=head.stamp,frame_id='map');o.child_frame_id='base_link';o.pose.pose.orientation.w=1.;o.pose.pose.position.x=v*ph;od.publish(o)
    s=VehicleState();s.header=head;s.speed_mps=v;s.gear=s.GEAR_DRIVE;vs.publish(s)
    clear=Marker();clear.header=head;clear.action=Marker.DELETEALL
    disp.publish(MarkerArray(markers=[clear,strip(head,'display/left',0,[(x,y0+1.75) for x in range(0,21)]),strip(head,'display/right',1,[(x,y0-1.75) for x in range(0,21)])]))
    marks=[clear];stop=30.-v*(ph-1.)
    if ph>=1. and stop>-1:marks.append(strip(head,'stop_line',1,[(stop,y0+1.6),(stop,y0-1.6)],w=.3))
    mk.publish(MarkerArray(markers=marks))
    d=[dict(class_id=2,class_name='traffic_light_red' if ph<6 else 'traffic_light_green',confidence=.8,xyxy=[940,80,980,170])]
    det.publish(String(data=json.dumps(dict(stamp=dict(sec=sec,nanosec=ns),frame_id='x',image_width=1920,image_height=1080,detections=d))))
    rclpy.spin_once(n,timeout_sec=.003)
    cen.publish(String(data=json.dumps(dict(stamp=t,frame_id='base_link',lanes=[lane('ego',y0),lane('left',y0+3.5,True),lane('right',y0-3.5)]))))
    e=time.time()+1/30.
    while time.time()<e:rclpy.spin_once(n,timeout_sec=.003)
for _ in range(20):rclpy.spin_once(n,timeout_sec=.05)
import os
if os.environ.get('FULL'):  # every field (but compute_ms), for an identity check between two versions
    json.dump(dict(info=[{k:v for k,v in r.items() if k!='compute_ms'} for r in info],scenes=scenes_full),open(sys.argv[1],'w'));sys.exit(0)
json.dump(dict(info=[dict(t=round(r['stamp']-t0,3),state=r['state'],shown=r['shown'],path_end=r['path'][-1][0] if r['path'] else None,traffic=r.get('traffic'),light=r.get('light')) for r in info],
               scenes=[dict(s,t=round(s['t']-t0,3)) for s in scenes]),open(sys.argv[1],'w'));print('guidance',len(info),'scenes',len(scenes))
