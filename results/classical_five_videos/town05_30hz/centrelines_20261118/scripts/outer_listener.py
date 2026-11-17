#!/usr/bin/env python3
"""Count /road/lane_outer, /road/lane_centres and their marker topics; keep every 30th centre message.
usage: outer_listener.py OUT.json SECONDS"""
import sys,json,time,rclpy
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
rclpy.init();node=rclpy.create_node('outer_listener');state=dict(outer=0,centres=0,outer_markers=0,centre_markers=0,lanes={},samples=[])
def on_outer(m):state['outer']+=1
def on_centres(m):
    state['centres']+=1;d=json.loads(m.data)
    for l in d['lanes']:state['lanes'][l['name']]=state['lanes'].get(l['name'],0)+1
    if state['centres']%30==1:state['samples'].append(dict(d,lanes=[dict(l,points=l['points'][:2]+['...']) for l in d['lanes']]))
def counter(k):
    def f(m):state[k]+=1
    return f
node.create_subscription(String,'/road/lane_outer',on_outer,10);node.create_subscription(String,'/road/lane_centres',on_centres,10)
node.create_subscription(MarkerArray,'/road/lane_outer_markers',counter('outer_markers'),10);node.create_subscription(MarkerArray,'/road/lane_centre_markers',counter('centre_markers'),10)
end=time.time()+float(sys.argv[2])
while time.time()<end:rclpy.spin_once(node,timeout_sec=.1)
json.dump(state,open(sys.argv[1],'w'),indent=1);print({k:v for k,v in state.items() if k!='samples'})
