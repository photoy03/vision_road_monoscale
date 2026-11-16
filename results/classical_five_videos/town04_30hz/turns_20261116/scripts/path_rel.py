"""Near end of observed boundaries relative to the vehicle's own arc (curvature over its last metre of
travel, from the cache poses): lateral rel = y0 - k x0^2/2, heading rel = angle - k x0 (deg).
usage: path_rel.py CACHE RUN_DIR FIRST-LAST[,FIRST-LAST...] [side]"""
import sys,gzip,json,math,numpy as np
cache,run,rngs=sys.argv[1:4];side=sys.argv[4] if len(sys.argv)>4 else None
P=np.asarray(json.load(open(cache+'/frames.json'))['poses']);yaw=np.unwrap(P[:,3]);s=np.r_[0,np.cumsum(np.hypot(np.diff(P[:,1]),np.diff(P[:,2])))]
rows=[json.loads(l) for l in gzip.open(run+'/predictions.jsonl.gz','rt')]
for rng in rngs.split(','):
    a,b=map(int,rng.split('-'))
    for i in range(a,b):
        t=rows[i]['stamp'];sn=np.interp(t,P[:,0],s);yn=np.interp(t,P[:,0],yaw)
        tb=np.interp(sn-1.,s,P[:,0]);k=(yn-np.interp(tb,P[:,0],yaw))/1.
        for q in rows[i]['boundaries']:
            if q['reprojected'] or (side and q['side']!=side):continue
            p=np.asarray(q['points']);p=p[np.argsort(p[:,0])];x0=p[0,0]
            if x0>15:continue
            x1=min(p[-1,0],x0+2.)
            if x1-x0<.5:continue
            y0=float(np.interp(x0,p[:,0],p[:,1]));ang=math.degrees(math.atan2(float(np.interp(x1,p[:,0],p[:,1]))-y0,x1-x0))
            print(f'{i} {t-rows[0]["stamp"]:5.2f} {q["side"]:5s} x0 {x0:5.1f} k {k:6.3f} y0 {y0:6.2f} rel_y {y0-k*x0*x0/2:6.2f} ang {ang:6.1f} rel_ang {ang-math.degrees(k*x0):6.1f}')
