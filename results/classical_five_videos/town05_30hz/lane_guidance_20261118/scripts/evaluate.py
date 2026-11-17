#!/usr/bin/env python3
"""Run the guidance over the scripts (scenarios.py) and the plain recording.  usage: evaluate.py
Per script: target switches, frames whose target is the oncoming lane, state counts and timeline, the
switch time against the moment the car centre crossed the line; plain recording: shown share, path
wobble (frame to frame, vehicle motion removed, at 5 / 10 m and at the merge end), update time."""
import sys,json,math,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
sys.path.insert(0,str(Path('/home/jaehun/Desktop/vision-road/src/hero_coach')))
from hero_coach.lane_guidance import LaneGuidance,Lane,carry,relative  # noqa: E402
import scenarios as S  # noqa: E402
rows=S.load();out={}
for name in ('wobble','yellow','dashed','change','none'):
    g=LaneGuidance();frames=S.frames(name,rows);log=[];times=[]
    for t,pose,lanes,d,r in frames:
        s=time.perf_counter();o=g.update(t,pose,[Lane.from_message(l) for l in lanes]);times.append((time.perf_counter()-s)*1e3)
        origin=next((l['origin'] for l in lanes if l['name']==o.target),o.target)
        log.append(dict(t=round(t-frames[0][0],3),delta=round(d,3),state=o.state,target=o.target,origin=origin,offset=round(o.offset_m,3),
                        vlat=round(o.lateral_speed_mps,3),changes=o.changes,shown=o.shown,merge=round(o.merge_m,1),
                        path=None if o.path is None else np.round(o.path,3).tolist(),pose=pose,lanes=[l['name'] for l in lanes],reason=o.reason))
    origins=[x['origin'] for x in log];switch=[x for a,x in zip(log,log[1:]) if x['changes']!=a['changes']]
    states={k:sum(1 for x in log if x['state']==k) for k in ('keep','return','change','forbidden','none')}
    cross=next((x['t'] for x in log if x['delta']<=-1.76),None) if name=='change' else None
    res=dict(frames=len(log),changes=log[-1]['changes'] if log else 0,switch_t=[x['t'] for x in switch],switch_delta=[x['delta'] for x in switch],
             line_crossed_t=cross,oncoming_target_frames=sum(1 for o in origins if o=='left'),
             target_origins={o:origins.count(o) for o in set(origins)},states=states,shown=round(float(np.mean([x['shown'] for x in log])),4),
             update_ms=dict(mean=round(float(np.mean(times)),3),p95=round(float(np.percentile(times,95)),3)))
    # path wobble (frame to frame, motion removed) where the path is shown
    w={5.:[],10.:[]};wt=[]
    for a,b in zip(log,log[1:]):
        # Same target, no target move, both reliable; scripts: where shown (the plain recording: all reliable frames).
        if a['path'] is None or b['path'] is None or a['changes']!=b['changes'] or a['origin']!=b['origin'] or 'none' in (a['state'],b['state']):continue
        if name!='none' and not (a['shown'] and b['shown']):continue
        m=relative(a['pose'],b['pose']);pa=carry(np.array(a['path']),m);pb=np.array(b['path'])
        for x in (5.,10.):w[x].append(float(np.interp(x,pb[:,0],pb[:,1])-np.interp(x,pa[:,0],pa[:,1])))
        wt.append(b['t'])
    res['path_wobble_cm']={str(k):dict(n=len(v),rms=round(float(np.sqrt(np.mean(np.square(v))))*100,2),p95=round(float(np.percentile(np.abs(v),95))*100,2),max=round(float(np.max(np.abs(v)))*100,1)) for k,v in w.items() if v}
    res['worst_wobble_10m']=sorted(((round(abs(v)*100,1),t) for v,t in zip(w[10.],wt)),reverse=True)[:5]
    out[name]=res;json.dump(log,open(HERE.parent/'runs'/f'{name}.json','w'))
    print(name,json.dumps({k:v for k,v in res.items()}))
(HERE.parent/'metrics').mkdir(exist_ok=True);json.dump(out,open(HERE.parent/'metrics'/'scenarios.json','w'),indent=1)
