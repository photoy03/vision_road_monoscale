#!/usr/bin/env python3
"""Plan 23: plan 21's publisher on scenarios23.py (the car also turns along its path: every line moved by
-delta and turned by -psi).  The fake vehicle state is not made here: tools/fake_vehicle_state.py derives
it from the /road/reference_odom this publishes, as it will for a plain replay.

Play one fake-motion script (scenarios.py) as the HUD and the guidance node see live data, at 30 Hz:
/road/lane_display_markers + /road/lane_display_styles (vision_road's display filter run here on the shifted
boundaries and paint, as road_node does), /road/lane_centres (shifted, named by position), and
/road/reference_odom (the shifted pose).  Every line is moved by -delta; once the car is over a line the own
boundaries are taken as the nearest line on each side among the own lines and the outer lines, as the
detector would.  usage: scenario_publisher.py SCRIPT LOG.json   (LOG: wall time and script time per frame)"""
import sys,json,time,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));ROOT=HERE.parents[4]
sys.path.insert(0,str(ROOT/'results/classical_five_videos/town05_30hz/lane_guidance_20261118/scripts'))
sys.path.insert(0,str(ROOT/'src/vision_road'))
import yaml,rclpy
from rclpy.qos import QoSProfile,ReliabilityPolicy
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray
from vision_lane_interfaces.msg import LaneStyleArray
from builtin_interfaces.msg import Time
from vision_road.display_filter import DisplayCore,DisplayStage,settings_from,simplify
import scenarios23 as S
script,log_path=sys.argv[1],sys.argv[2]
settings=settings_from(yaml.safe_load(open(ROOT/'src/vision_road/config/carla.yaml'))['lane_display_filter']['ros__parameters'])
core=DisplayCore(settings);stage=DisplayStage(settings,'base_link')
rclpy.init();node=rclpy.create_node('scenario_publisher')
pub_m=node.create_publisher(MarkerArray,'/road/lane_display_markers',10);pub_s=node.create_publisher(LaneStyleArray,'/road/lane_display_styles',10)
pub_c=node.create_publisher(String,'/road/lane_centres',10);pub_o=node.create_publisher(Odometry,'/road/reference_odom',QoSProfile(depth=50,reliability=ReliabilityPolicy.BEST_EFFORT))
frames=S.frames(script);time.sleep(1.);log=[];start=time.monotonic();t0=frames[0][0]


def moved(points,delta,psi):
    return S.turn(points,delta,psi)


def boundaries(row,delta,psi=0.):
    """Own left / right after the move: the nearest shifted line each side (own lines and outer lines)."""
    lines=[]
    for b in row['bounds']:lines.append((b['points'],b['rep'],b['side']))
    for o in row['outer']:lines.append((o['points'],o['rep'],o['side']))
    out={}
    for pts,rep,name in lines:
        p=moved(pts,delta,psi)
        y=float(np.interp(8.,p[:,0],p[:,1])) if p[0,0]<=8.<=p[-1,0] else float(p[0,1])
        side='left' if y>0 else 'right'
        if side not in out or abs(y)<abs(out[side][2]):out[side]=(p,rep,y,name)
    return out


def styles(row,chosen):
    """(side, style, confidence) per new side, from the line it came from."""
    own={}
    for l in row['lanes']:
        if l['side'] not in own or l['conf']>own[l['side']][1]:own[l['side']]=(l['style'],l['conf'])
    outer={o['side']:(o['style'],o['conf']) for o in row['outer']}
    res=[]
    for side,(_,rep,_,name) in chosen.items():
        st,conf=own.get(name,outer.get(name,('unknown',0.)))
        res.append((side,st if st in ('solid','dashed') else None,float(np.float32(conf))))
    return res


for t,pose,lanes,delta,psi,row in frames:
    due=start+(t-t0)
    while time.monotonic()<due:time.sleep(.001)
    ns=round(t*1e9);stamp=Time(sec=ns//1000000000,nanosec=ns%1000000000)
    odom=Odometry();odom.header.stamp=stamp;odom.header.frame_id='map';odom.child_frame_id='base_link'
    odom.pose.pose.position.x=float(pose[0]);odom.pose.pose.position.y=float(pose[1])
    odom.pose.pose.orientation.z=math.sin(pose[2]/2);odom.pose.pose.orientation.w=math.cos(pose[2]/2);pub_o.publish(odom)
    chosen=boundaries(row,delta,psi)
    core.styles(t,styles(row,chosen))
    paint=[(l['side'],simplify(moved(l['points'],delta,psi),.025)) for l in row['lanes'] if not l['rep'] and len(l['points'])>=2]
    shown=core.step(t,[(side,simplify(p,.025),rep) for side,(p,rep,_,_) in chosen.items() if len(p)>=2],pose,paint)
    markers,style=stage.messages(stamp,shown);pub_m.publish(markers);pub_s.publish(style)
    pub_c.publish(String(data=json.dumps(dict(stamp=t,frame_id='base_link',lanes=lanes),separators=(',',':'))))
    rclpy.spin_once(node,timeout_sec=0.)
    log.append(dict(wall=time.time(),t=round(t-t0,3),delta=round(delta,3),psi=round(psi,4)))
json.dump(log,open(log_path,'w'));node.destroy_node();rclpy.shutdown()
