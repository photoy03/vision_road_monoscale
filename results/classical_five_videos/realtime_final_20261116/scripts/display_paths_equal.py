"""DisplayStage fed road_node's messages vs fed the same lanes directly (inline path): same display output?
usage: display_paths_equal.py RUN_DIR"""
import sys,gzip,json,yaml,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'src/vision_road'))
from builtin_interfaces.msg import Time
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker,MarkerArray
from vision_road import display_filter as D
from vision_road.core import Timeline
cfg=yaml.safe_load(open(ROOT/'src/vision_road/config/carla.yaml'))['lane_display_filter']['ros__parameters']
odom=json.load(open('/home/jaehun/.codex/.chatgpt-projects/g-p-6a551231636c819182badfc2f75ebf40/vision-road/results/maploop30hz/input/truth_base.json'))
tl=Timeline(.25);[tl.add(*p) for p in odom]
rows=[json.loads(l) for l in gzip.open(sys.argv[1]+'/predictions.jsonl.gz','rt')]
def msg(lanes,ns,stamp):
    out=MarkerArray();c=Marker();c.header=Header(stamp=stamp,frame_id='base_link');c.action=Marker.DELETEALL;out.markers.append(c)
    for i,l in enumerate(lanes):
        m=Marker();m.header=c.header;m.ns=ns+'/'+l['side'];m.id=i;m.type=Marker.LINE_STRIP;m.action=Marker.ADD;m.color.r=1. if l['reprojected'] else 0.
        m.points=[Point(x=float(x),y=float(y),z=.02) for x,y in l['points']];out.markers.append(m)
    return out
A=D.DisplayStage(D.settings_from(cfg));B=D.DisplayStage(D.settings_from(cfg));diff=0
for r in rows:
    ns=round(r['stamp']*1e9);stamp=Time(sec=ns//10**9,nanosec=ns%10**9)
    A.on_paint(msg(r['lanes'],'observed_paint',stamp));a=A.on_boundaries(msg(r['boundaries'],'boundary_fit',stamp),tl.at)
    b=B.on_lanes(stamp,[(l['side'],np.asarray(l['points'],float),l['reprojected']) for l in r['boundaries'] if len(l['points'])>=2],
                 [(l['side'],np.asarray(l['points'],float)) for l in r['lanes'] if not l['reprojected'] and len(l['points'])>=2],tl.at)
    pa=[(m.ns,m.color.r,[(p.x,p.y) for p in m.points]) for m in a[0].markers];pb=[(m.ns,m.color.r,[(p.x,p.y) for p in m.points]) for m in b[0].markers]
    diff+=pa!=pb
print('frames',len(rows),'different displays',diff)
