"""Record boundary, display, road marking markers, styles and odometry per stamp until SIGINT."""
import json,signal,sys,math,rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import Odometry
from vision_lane_interfaces.msg import LaneStyleArray
out=sys.argv[1];rows={};odom=[]
def sec(h):return round(h.stamp.sec+h.stamp.nanosec*1e-9,4)
def markers(name):
    def cb(msg):
        if not msg.markers:return
        rows.setdefault(sec(msg.markers[0].header),{})[name]=[dict(ns=m.ns,id=m.id,side='left' if m.ns.endswith('/left') else 'right',predicted=m.color.r>.5,
            points=[[round(p.x,3),round(p.y,3)] for p in m.points]) for m in msg.markers if m.action==0 and len(m.points)>=2]
    return cb
def styles(name):
    def cb(msg):rows.setdefault(sec(msg.header),{})[name]=[dict(side=l.side,style=int(l.style),confidence=float(l.confidence),reprojected=bool(l.reprojected)) for l in msg.lanes]
    return cb
def on_odom(msg):
    q=msg.pose.pose.orientation;p=msg.pose.pose.position
    odom.append([sec(msg.header),p.x,p.y,math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))])
rclpy.init();n=Node('topic_recorder')
for topic,name in (('/road/lane_boundary_markers','raw'),('/road/lane_display_markers','display'),('/road/road_markings','road_markings'),('/road/lane_markers','paint')):
    n.create_subscription(MarkerArray,topic,markers(name),50)
n.create_subscription(LaneStyleArray,'/road/lane_styles',styles('raw_styles'),50)
n.create_subscription(LaneStyleArray,'/road/lane_display_styles',styles('display_styles'),50)
n.create_subscription(Odometry,'/road/reference_odom',on_odom,QoSProfile(depth=200,reliability=ReliabilityPolicy.BEST_EFFORT))
stop=[False]
signal.signal(signal.SIGTERM,lambda *a:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *a:stop.__setitem__(0,True))
while not stop[0]:rclpy.spin_once(n,timeout_sec=.1)
json.dump(dict(rows=[dict(stamp=k,**v) for k,v in sorted(rows.items())],odom=odom),open(out,'w'))
print('rows',len(rows),'raw',sum('raw' in v for v in rows.values()),'markings',sum('road_markings' in v for v in rows.values()))
