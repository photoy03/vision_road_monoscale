#!/usr/bin/env python3
"""Collect /hero/lane_guidance and count /hero/scene_overlay during a real-time run.
usage: guidance_listener.py OUT.json SECONDS"""
import sys,json,time,rclpy
import numpy as np
from std_msgs.msg import String
rclpy.init();node=rclpy.create_node('guidance_listener');rows=[];count=dict(scene=0,scene_ideal=0)
def on_guidance(m):d=json.loads(m.data);d.pop('path',None);rows.append(d)
node.create_subscription(String,'/hero/lane_guidance',on_guidance,50)
try:
    from hero_hud.msg import SceneOverlay
    def on_scene(m):count['scene']+=1;count['scene_ideal']+=bool(m.has_ideal)
    node.create_subscription(SceneOverlay,'/hero/scene_overlay',on_scene,50)
except ImportError:pass
end=time.time()+float(sys.argv[2])
while time.time()<end:rclpy.spin_once(node,timeout_sec=.1)
c=np.array([r['compute_ms'] for r in rows]) if rows else np.zeros(1)
res=dict(messages=len(rows),**count,compute_ms=dict(mean=round(float(c.mean()),3),p95=round(float(np.percentile(c,95)),3),max=round(float(c.max()),3)),
         shown=round(float(np.mean([r['shown'] for r in rows])),4) if rows else None,states={k:sum(1 for r in rows if r['state']==k) for k in ('none','keep','return','change','forbidden')},
         changes=rows[-1]['changes'] if rows else None)
json.dump(dict(summary=res,rows=rows),open(sys.argv[1],'w'));print(json.dumps(res))
