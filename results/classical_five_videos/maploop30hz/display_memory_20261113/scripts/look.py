import sys,gzip,json,numpy as np
sys.path.insert(0,'/home/jaehun/Desktop/vision-road/src/vision_road')
from vision_road.display_filter import DisplayFilter,FilterSettings,relative,evaluate,fit
from vision_road.core import Timeline
P,T=sys.argv[1],sys.argv[2];frames=[int(x) for x in sys.argv[3].split(',')];side=sys.argv[4]
rows=[json.loads(l) for l in gzip.open(P,'rt')];odom=json.load(open(T))
f=DisplayFilter(FilterSettings(near_m=0.,memory_m=10.));tl=Timeline(.25);k=0;last=None
for i,r in enumerate(rows[:max(frames)+1]):
    t=r['stamp']
    while k<len(odom) and odom[k][0]<=t+.1:tl.add(*odom[k]);k+=1
    pose=tl.at(t);motion=(0.,0.,0.) if last is None else relative(last[1],pose);last=(t,pose)
    tr=f.sides[side]
    obs=[np.asarray(b['points']) for b in r['boundaries'] if b['side']==side and not b['reprojected']]
    o=max(obs,key=lambda p:np.ptp(p[:,0]),default=None)
    f.update(t,[(b['side'],b['points'],b['reprojected']) for b in r['boundaries']],motion,[(l['side'],l['points']) for l in r['lanes'] if not l['reprojected']])
    if i in frames:
        print('frame',i,'obs',None if o is None else np.round(o[[0,len(o)//2,-1]],2).tolist())
        if o is not None:
            rc=tr.recall(o[:,0].min(),fit(o)[0]);print('  recall',None if rc is None else np.round(rc,2).tolist())
        print('  value at 0,3,6,10,15',None if tr.value is None else np.round(evaluate(tr.value,np.array([0,3,6,10,15.])),3).tolist(),'pending',len(tr.pending))
        print('  paint',[np.round(np.asarray(l['points'])[[0,-1]],2).tolist() for l in r['lanes'] if l['side']==side and not l['reprojected']])
