#!/usr/bin/env python3
"""Count /road/lane_outer and /road/lane_outer_markers messages; keep every 30th JSON message.
usage: outer_listener.py OUT.json SECONDS"""
import sys,json,time,rclpy
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
rclpy.init();node=rclpy.create_node('outer_listener');state=dict(json=0,markers=0,samples=[],sides={})
def on_json(m):
    state['json']+=1;d=json.loads(m.data)
    for l in d['lanes']:state['sides'][l['side']]=state['sides'].get(l['side'],0)+1
    if state['json']%30==1:state['samples'].append(dict(d,lanes=[dict(l,points=l['points'][:3]+['...']) for l in d['lanes']]))
def on_markers(m):state['markers']+=1
node.create_subscription(String,'/road/lane_outer',on_json,10);node.create_subscription(MarkerArray,'/road/lane_outer_markers',on_markers,10)
end=time.time()+float(sys.argv[2])
while time.time()<end:rclpy.spin_once(node,timeout_sec=.1)
json.dump(state,open(sys.argv[1],'w'),indent=1);print('json',state['json'],'markers',state['markers'],state['sides'])
