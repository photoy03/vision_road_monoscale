"""Near-end geometry of observed boundaries per frame: start x, y at start, angle of the first 2 m
(deg, against the heading), and the vehicle yaw rate (deg/s) from the cache poses.
usage: near_geom.py CACHE RUN_DIR FIRST-LAST [side]"""
import sys,gzip,json,math,numpy as np
cache,run,rng=sys.argv[1:4];side=sys.argv[4] if len(sys.argv)>4 else None;a,b=map(int,rng.split('-'))
poses=np.asarray(json.load(open(cache+'/frames.json'))['poses']);yaw=np.unwrap(poses[:,3])
rows=[json.loads(l) for l in gzip.open(run+'/predictions.jsonl.gz','rt')]
for i in range(a,b):
    r=rows[i];t=r['stamp'];rate=math.degrees((np.interp(t+.1,poses[:,0],yaw)-np.interp(t-.1,poses[:,0],yaw))/.2)
    for q in r['boundaries']:
        if q['reprojected'] or (side and q['side']!=side):continue
        p=np.asarray(q['points']);p=p[np.argsort(p[:,0])];x0=p[0,0];x1=min(p[-1,0],x0+2.)
        if x1-x0<.5:continue
        y0=float(np.interp(x0,p[:,0],p[:,1]));y1=float(np.interp(x1,p[:,0],p[:,1]))
        print(f'{i} {t-rows[0]["stamp"]:5.2f} {q["side"]:5s} x {x0:5.1f}-{p[-1,0]:5.1f} y0 {y0:6.2f} angle {math.degrees(math.atan2(y1-y0,x1-x0)):6.1f} yawrate {rate:6.1f}')
