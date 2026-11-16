"""Side moves over the whole run and display-filter switches.
move: a side's observed boundary (longest, span>=3 m) lies >1 m sideways from the same side's last
observed boundary of the past 10 frames, motion compensated (truth odometry), at x=clamp(8, overlap).
switch: the display filter accepts a sustained change (>0.30 m at 10 m).
coverage: per segment, fraction of frames with an observed boundary per side.
usage: [FRAMES=cache/frames.json (odometry from its poses; default maploop truth_base.json)] moves.py REPO OUT.json NAME=RUN_DIR ...
Segments are maploop's."""
import sys,gzip,json,numpy as np
sys.path.insert(0,sys.argv[1]+'/src/vision_road');sys.path.insert(0,sys.argv[1]+'/tools')
from vision_road.core import Timeline
from vision_road.display_filter import relative,FilterSettings
import vision_road.display_filter as D,display_filter_eval as E
SEG=[(0,5,'junction'),(5,20,'straight'),(20,22,'turn'),(22,35,'curve'),(35,42,'straight2'),(42,48,'crosswalk'),(48,53,'island'),(53,61,'junction2')]
def seg(t):return next((n for a,b,n in SEG if a<=t<b),'end')
import os
odom=json.load(open(os.environ['FRAMES']))['poses'] if os.environ.get('FRAMES') else json.load(open('/home/jaehun/.codex/.chatgpt-projects/g-p-6a551231636c819182badfc2f75ebf40/vision-road/results/maploop30hz/input/truth_base.json'))
def moves(rows):
    tl=Timeline(.25);k=0;poses=[]
    for r in rows:
        while k<len(odom) and odom[k][0]<=r['stamp']+.1:tl.add(*odom[k]);k+=1
        poses.append(tl.at(r['stamp']))
    t0=rows[0]['stamp'];out=[]
    for side in ('left','right'):
        last=None
        for i,r in enumerate(rows):
            bs=[np.asarray(b['points']) for b in r['boundaries'] if b['side']==side and not b['reprojected']]
            bs=[p for p in bs if np.ptp(p[:,0])>=3]
            if not bs:continue
            p=max(bs,key=lambda p:np.ptp(p[:,0]))
            if last is not None and i-last[0]<=10 and poses[i] is not None and poses[last[0]] is not None:
                dx,dy,dyaw=relative(poses[last[0]],poses[i]);q=last[1]-[dx,dy];c,s=np.cos(-dyaw),np.sin(-dyaw)
                q=q@np.array([[c,s],[-s,c]])
                lo=max(p[0,0],q[:,0].min());hi=min(p[-1,0],q[:,0].max())
                x=float(np.clip(8,lo,hi)) if lo<=hi else (p[0,0] if p[0,0]>q[:,0].max() else p[-1,0])
                o=np.argsort(q[:,0]);ya=float(np.interp(x,q[o,0],q[o,1]));yb=float(np.interp(x,p[:,0],p[:,1]))
                if abs(yb-ya)>1.:t=r['stamp']-t0;out.append(dict(side=side,frame=i,t=round(t,2),seg=seg(t),y_from=round(ya,2),y_to=round(yb,2),gap=i-last[0],x=round(x,1)))
            last=(i,p)
    return out
def switches(rows):
    ev=[];orig=D.SideTrack.take;t0=rows[0]['stamp'];names={}
    def take(self,st,t,alone=None):
        b=None if self.value is None else self.value[1];ok=orig(self,st,t,alone)
        if ok and b is not None and abs(self.value[1]-b)>self.s.gate_m:ev.append((round(t-t0,2),names.get(id(self)),round(b,2),round(float(self.value[1]),2)))
        return ok
    f0=D.DisplayFilter.reset
    def reset(self):
        f0(self);[names.__setitem__(id(v),k) for k,v in self.sides.items()]
    D.SideTrack.take=take;D.DisplayFilter.reset=reset
    E.run(rows,FilterSettings(near_m=0.,memory_m=10.),odom);D.SideTrack.take=orig;D.DisplayFilter.reset=f0
    return [dict(t=a,frame=round(a*30),side=s,seg=seg(a),y10_from=b,y10_to=c) for a,s,b,c in ev]
def coverage(rows):
    t0=rows[0]['stamp'];out={}
    for a,b,n in SEG:
        part=[r for r in rows if a<=r['stamp']-t0<b]
        out[n]={s:round(float(np.mean([any(x['side']==s and not x['reprojected'] for x in r['boundaries']) for r in part])),3) for s in ('left','right')}
    return out
res={}
for spec in sys.argv[3:]:
    name,path=spec.split('=');rows=[json.loads(l) for l in gzip.open(path+'/predictions.jsonl.gz','rt')]
    res[name]=dict(moves=moves(rows),display_switches=switches(rows),coverage=coverage(rows))
    print('==',name,'moves',len(res[name]['moves']),'display switches',len(res[name]['display_switches']))
    for m in res[name]['moves']:print('  move',m)
    for m in res[name]['display_switches']:print('  switch',m)
    print('  coverage',' '.join(f"{k}:L{v['left']:.2f}/R{v['right']:.2f}" for k,v in res[name]['coverage'].items()))
json.dump(res,open(sys.argv[2],'w'),indent=1)
