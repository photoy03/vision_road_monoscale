"""Frame-to-frame boundary change at 10 m with and without vehicle motion compensation.

The previous frame's observed boundary is moved into the current base_link by the rear-axle
poses of the offline cache (frames.json), then compared with the current one at x = 10 m
(both must cover it).  Per segment and side: std of changes up to 0.30 m (cm), changes over 5 cm (j5)
and over 30 cm (j30); 'raw' is the same without the motion.
usage: mc_jitter.py maploop|town05|town04 CACHE OUT.json NAME=RUN_DIR ..."""
import sys,gzip,json,math,numpy as np
SEGS=dict(town04=[('junction',0,10),('curve',15,20),('hairpin',20,25),('junction',25,30),('straight',30,35),('turn',35,40),('straight',40,45),('junction',45,50),('curve',50,55),('curve',55,58)],
          town05=[('straight',0,9),('turn',10,15),('curve',15,30),('turn',35,40),('straight',40,45),('junction',45,50),('straight',55,60)],
          maploop=[('junction',0,5),('straight',5,20),('turn',20,22),('curve',22,35),('straight2',35,42),('crosswalk',42,48),('island',48,53),('junction2',53,61)])
def pose_at(poses,t):
    p=np.asarray(poses);yaw=np.unwrap(p[:,3])
    return np.array([np.interp(t,p[:,0],p[:,1]),np.interp(t,p[:,0],p[:,2]),np.interp(t,p[:,0],yaw)])
def relative(a,b):
    c,s=math.cos(a[2]),math.sin(a[2]);wx,wy=b[0]-a[0],b[1]-a[1]
    return c*wx+s*wy,-s*wx+c*wy,b[2]-a[2]
def at(p,x):return float(np.interp(x,p[:,0],p[:,1])) if p[0,0]<=x<=p[-1,0] else None
video,cache,out=sys.argv[1],sys.argv[2],sys.argv[3];poses=json.load(open(cache+'/frames.json'))['poses'];res={}
for spec in sys.argv[4:]:
    name,path=spec.split('=');rows=[json.loads(l) for l in gzip.open(path+'/predictions.jsonl.gz','rt')];t0=rows[0]['stamp']
    P=[pose_at(poses,r['stamp']) for r in rows];res[name]={}
    def line(r,side):
        b=[np.asarray(q['points']) for q in r['boundaries'] if q['side']==side and not q['reprojected']]
        return max(b,key=lambda p:np.ptp(p[:,0])) if b else None
    for g,a,b in SEGS[video]:
        key=f'{g} {a}-{b}';res[name][key]={}
        for side in ('left','right'):
            mc=[];raw=[]
            for i in range(1,len(rows)):
                if not a<=rows[i]['stamp']-t0<b:continue
                p,q=line(rows[i],side),line(rows[i-1],side)
                if p is None or q is None:continue
                y=at(p,10.)
                if y is None:continue
                yq=at(q,10.)
                if yq is not None:raw.append(y-yq)
                dx,dy,dyaw=relative(P[i-1],P[i]);c,s=math.cos(dyaw),math.sin(dyaw)
                qx,qy=q[:,0]-dx,q[:,1]-dy;m=np.column_stack([c*qx+s*qy,-s*qx+c*qy]);m=m[np.argsort(m[:,0])]
                ym=at(m,10.)
                if ym is not None:mc.append(y-ym)
            def stat(d):
                d=np.array(d)
                if not len(d):return None
                small=d[np.abs(d)<=.3]
                return dict(std_cm=round(float(np.std(small))*100,2) if len(small) else None,j5=int(np.sum(np.abs(d)>.05)),j30=int(np.sum(np.abs(d)>.3)),n=len(d))
            res[name][key][side]=dict(mc=stat(mc),raw=stat(raw))
        f=lambda s,k:'-' if res[name][key][s][k] is None else f"{res[name][key][s][k]['std_cm']}/{res[name][key][s][k]['j5']}/{res[name][key][s][k]['j30']}"
        print(f"{name:10s} {key:15s} L mc {f('left','mc'):14s} raw {f('left','raw'):14s} | R mc {f('right','mc'):14s} raw {f('right','raw')}")
json.dump(res,open(out,'w'),indent=1)
