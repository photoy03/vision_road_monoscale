#!/usr/bin/env python3
"""Plan 23 offline checks.  usage: evaluate23.py

For each script (scenarios23.py: plan 21's scripts with the heading turned along the path) and the plain
recording ('none'):
  - the fake vehicle state (tools/fake_vehicle_state.py VehicleStateEstimator) from the fake rear-axle poses;
  - arc accuracy (also for an 'ideal' arc from the true curvature of the fake path at that moment, centred
    differences over +-0.1 s: what a constant-steering arc can do at best, without the estimator's lag);
    the plain recording also split into straight/curve and junction turns (heading change > 0.7 rad in 3 s);
  - the HUD's current-path arc (kinematic bicycle, wheelbase 2.70 m, from the base_link origin
    along +x with tan(steering)/wheelbase curvature) against where the car actually was after travelling
    5 and 10 m (that pose in this frame's base_link): lateral miss at the actual x, cm;
  - the guidance with speed from the pose (as plan 21) and with speed from the vehicle state: target moves,
    oncoming targets, switch time, states, shown share; and plan 21's own frames (no heading turn) for
    comparison.
Writes ../metrics/evaluate23.json."""
import sys,json,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=Path('/home/jaehun/Desktop/vision-road')
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'src/hero_coach'))
import scenarios23 as S  # noqa: E402
import scenarios as S21  # noqa: E402
from fake_vehicle_state import VehicleStateEstimator,WHEELBASE_M  # noqa: E402
from hero_coach.lane_guidance import LaneGuidance,Lane  # noqa: E402


def to_base(p,pose):
    c,s=math.cos(pose[2]),math.sin(pose[2]);dx,dy=p[0]-pose[0],p[1]-pose[1];return c*dx+s*dy,-s*dx+c*dy


def arc_y(x,k):
    """y of the arc of curvature k leaving the origin along +x, at forward distance x (inside a half turn)."""
    if abs(k)<1e-6:return 0.
    r=1/k;q=r*r-x*x
    return r-math.copysign(math.sqrt(q),r) if q>0 else r


def guidance(frames,speeds=None):
    g=LaneGuidance();log=[]
    for i,(t,pose,lanes,d,*_rest) in enumerate(frames):
        o=g.update(t,pose,[Lane.from_message(l) for l in lanes],speed=None if speeds is None else speeds[i])
        origin=next((l.get('origin') for l in lanes if l['name']==o.target),o.target)
        log.append((t,d,o.state,o.changes,origin,o.shown,o.merge_m))
    switch=[round(b[0]-frames[0][0],3) for a,b in zip(log,log[1:]) if b[3]!=a[3]]
    return dict(changes=log[-1][3] if log else 0,switch_t=switch,oncoming=sum(1 for x in log if x[4]=='left'),
                states={k:sum(1 for x in log if x[2]==k) for k in ('none','keep','return','change','forbidden')},
                shown=round(float(np.mean([x[5] for x in log])),4),merge_mean=round(float(np.mean([x[6] for x in log if x[5]] or [0])),1))


rows=S.load();out={}
for name in ('wobble','yellow','dashed','change','none'):
    frames=S.frames(name,rows);est=VehicleStateEstimator();states=[]
    for t,pose,*_ in frames:states.append(est.update(t,*pose))
    # arc accuracy
    poses=np.array([f[1] for f in frames]);ts=np.array([f[0] for f in frames])
    dist=np.r_[0.,np.cumsum(np.hypot(np.diff(poses[:,0]),np.diff(poses[:,1])))]
    miss={5.:[],10.:[]};ideal={5.:[],10.:[]};worst=[];parts={'road':[],'turn':[]}
    yaw=np.unwrap(poses[:,2])
    for i,e in enumerate(states):
        if i<10 or abs(e.speed_mps)<2.:continue  # past the low-pass start
        k=math.tan(e.steering_rad)/WHEELBASE_M
        a,b=max(0,i-3),min(len(frames)-1,i+3);kt=(yaw[b]-yaw[a])/max(dist[b]-dist[a],1e-3)
        w=(ts>=ts[i]-3.)&(ts<=ts[i]);turning=np.ptp(yaw[w])>.7
        for L in (5.,10.):
            j=int(np.searchsorted(dist,dist[i]+L))
            if j>=len(frames):continue
            xa,ya=to_base(poses[j],poses[i]);m=ya-arc_y(xa,k);miss[L].append(m);ideal[L].append(ya-arc_y(xa,kt))
            if L==10.:parts['turn' if turning else 'road'].append(m)
            if L==10.:worst.append((round(abs(m)*100,1),round(ts[i]-ts[0],2)))
    stat=lambda v:dict(n=len(v),rms_cm=round(float(np.sqrt(np.mean(np.square(v))))*100,1),p95_cm=round(float(np.percentile(np.abs(v),95))*100,1),max_cm=round(float(np.max(np.abs(v)))*100,1)) if v else None
    res=dict(frames=len(frames),arc_miss_5m=stat(miss[5.]),arc_miss_10m=stat(miss[10.]),ideal_arc_miss_5m=stat(ideal[5.]),ideal_arc_miss_10m=stat(ideal[10.]),
             arc_miss_10m_road=stat(parts['road']),arc_miss_10m_turn=stat(parts['turn']),worst_10m=sorted(worst,reverse=True)[:3],
             steering_deg=dict(max=round(float(np.degrees(max(abs(e.steering_rad) for e in states))),2) if states else None),
             speed_mps=dict(mean=round(float(np.mean([e.speed_mps for e in states])),2)),
             guidance_pose_speed=guidance(frames),guidance_vehicle_speed=guidance(frames,[e.speed_mps for e in states]))
    if name!='none':
        res['guidance_plan21_frames']=guidance([(f[0],f[1],f[2],f[3]) for f in S21.frames(name,rows)])
    out[name]=res
    g1,g2=res['guidance_pose_speed'],res['guidance_vehicle_speed']
    print(name,'arc10',res['arc_miss_10m'],'ideal10',res['ideal_arc_miss_10m'],'arc5',res['arc_miss_5m'],'ideal5',res['ideal_arc_miss_5m'],'road10',res['arc_miss_10m_road'],'turn10',res['arc_miss_10m_turn'],'| pose-speed',g1['changes'],g1['switch_t'],g1['oncoming'],g1['shown'],'| vehicle-speed',g2['changes'],g2['switch_t'],g2['oncoming'],g2['shown'],
          '| plan21 frames',res.get('guidance_plan21_frames',{}).get('changes'),res.get('guidance_plan21_frames',{}).get('switch_t'),flush=True)
(HERE.parent/'metrics').mkdir(exist_ok=True);json.dump(out,open(HERE.parent/'metrics'/'evaluate23.json','w'),indent=1)
