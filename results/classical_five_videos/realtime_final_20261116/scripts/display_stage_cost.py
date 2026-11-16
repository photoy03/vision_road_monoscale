"""Cost of DisplayStage (inline display filter) per frame on messages built like road_node's, from an offline
run's predictions; split into message parsing, filter update and message building.  usage: display_stage_cost.py RUN_DIR"""
import sys,gzip,json,time,yaml,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'src/vision_road'))
from builtin_interfaces.msg import Time
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker,MarkerArray
from vision_road import display_filter as D
cfg=yaml.safe_load(open(ROOT/'src/vision_road/config/carla.yaml'))['lane_display_filter']['ros__parameters']
odom=json.load(open('/home/jaehun/.codex/.chatgpt-projects/g-p-6a551231636c819182badfc2f75ebf40/vision-road/results/maploop30hz/input/truth_base.json'))
from vision_road.core import Timeline
tl=Timeline(.25);[tl.add(*p) for p in odom]
rows=[json.loads(l) for l in gzip.open(sys.argv[1]+'/predictions.jsonl.gz','rt')]
def msg(lanes,ns,stamp):
    out=MarkerArray();c=Marker();c.header=Header(stamp=stamp,frame_id='base_link');c.action=Marker.DELETEALL;out.markers.append(c)
    for i,l in enumerate(lanes):
        m=Marker();m.header=c.header;m.ns=ns+'/'+l['side'];m.id=i;m.type=Marker.LINE_STRIP;m.action=Marker.ADD;m.color.r=1. if l['reprojected'] else 0.
        m.points=[Point(x=float(x),y=float(y),z=.02) for x,y in l['points']];out.markers.append(m)
    return out
stage=D.DisplayStage(D.settings_from(cfg));T={'build_inputs':0,'on_paint':0,'on_boundaries':0}
orig=stage.filter.update;spent=[0.]
def upd(*a,**k):
    t=time.perf_counter();r=orig(*a,**k);spent[0]+=time.perf_counter()-t;return r
stage.filter.update=upd
for r in rows:
    ns=round(r['stamp']*1e9);stamp=Time(sec=ns//10**9,nanosec=ns%10**9)
    t=time.perf_counter();p=msg(r['lanes'],'observed_paint',stamp);b=msg(r['boundaries'],'boundary_fit',stamp);T['build_inputs']+=time.perf_counter()-t
    t=time.perf_counter();stage.on_paint(p);T['on_paint']+=time.perf_counter()-t
    t=time.perf_counter();stage.on_boundaries(b,tl.at);T['on_boundaries']+=time.perf_counter()-t
n=len(rows);print({k:round(v/n*1e3,2) for k,v in T.items()},'filter.update',round(spent[0]/n*1e3,2),'ms/frame')
