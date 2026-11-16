"""Per-frame trace of one side through the display filter (carla.yaml settings).
usage: dtrace.py REPO RUN_DIR FIRST-LAST [left|right]"""
import sys,gzip,json,numpy as np
sys.path.insert(0,sys.argv[1]+'/src/vision_road');sys.path.insert(0,sys.argv[1]+'/tools')
import display_filter_eval as E
from vision_road import display_filter as D
from vision_road.display_filter import FilterSettings
ODOM='/home/jaehun/.codex/.chatgpt-projects/g-p-6a551231636c819182badfc2f75ebf40/vision-road/results/maploop30hz/input/truth_base.json'
rows=[json.loads(l) for l in gzip.open(sys.argv[2]+'/predictions.jsonl.gz','rt')]
odom=json.load(open(ODOM))
side=sys.argv[4] if len(sys.argv)>4 else 'right'
a,b=map(int,sys.argv[3].split('-'))
log={}
orig=D.SideTrack.take
def take(self,stations,t,alone=None):
    before=None if self.value is None else self.value.copy();n=len(self.pending)
    ok=orig(self,stations,t,alone)
    kind='init' if before is None else ('gate' if ok and n==0 and abs(stations[1]-before[1])<=self.s.gate_m else ('SWITCH' if ok else 'reject'))
    log.setdefault(t,{})[id(self)]=kind;return ok
D.SideTrack.take=take
out=E.run(rows,FilterSettings(near_m=0.,memory_m=10.),odom)
for i in range(a,b):
    r=rows[i];bs=[np.asarray(x['points']) for x in r['boundaries'] if x['side']==side and not x['reprojected']]
    ob=max(bs,key=lambda p:np.ptp(p[:,0]),default=None)
    o='-' if ob is None else f'obs {ob[0,0]:5.1f}-{ob[-1,0]:5.1f} y {ob[0,1]:6.2f}..{ob[-1,1]:6.2f}'
    d=out[i].get(side)
    s='-' if d is None else f'disp y0 {np.interp(0,d[0][:,0],d[0][:,1]):6.2f} y10 {np.interp(10,d[0][:,0],d[0][:,1]):6.2f} pred {d[1]}'
    print(i,o.ljust(40),s,list(log.get(r['stamp'],{}).values()))
