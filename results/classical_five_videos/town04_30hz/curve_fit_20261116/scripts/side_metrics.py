"""Per segment and side, from predictions.jsonl.gz (observed boundary = longest non-reprojected):
  y10 std: std of frame-to-frame change of the boundary's lateral position at 10 m (|d|<=0.3 m), cm
  j5: frame-to-frame changes > 5 cm at 10 m;  obs: frames with an observed boundary
  near8: frames with observed paint starting before 8 m;  reach: median far end of observed paint, m
  styles: style label of the side's longest paint lane
usage: side_metrics.py town04|town05|maploop OUT.json NAME=RUN_DIR ..."""
import sys,gzip,json,numpy as np
SEGS=dict(town04=[('junction',0,10),('curve',15,20),('curve',20,25),('junction',25,30),('straight',30,35),('turn',35,40),('straight',40,45),('junction',45,50),('curve',50,55),('curve',55,58)],
          town05=[('straight',0,9),('turn1',10,15),('curve',15,30),('turn2',35,40),('straight',40,45),('junction',45,50),('straight',55,60)],
          maploop=[('junction',0,5),('straight',5,20),('curve',22,35),('straight2',35,42),('crosswalk',42,48),('island',48,53),('junction2',53,61)])
def at(p,x):return float(np.interp(x,p[:,0],p[:,1])) if p[0,0]<=x<=p[-1,0] else None
def seg(rows,side):
    y=[];near=[];reach=[];styles={};obs=0
    for r in rows:
        b=[np.asarray(q['points']) for q in r['boundaries'] if q['side']==side and not q['reprojected']]
        p=max(b,key=lambda p:np.ptp(p[:,0])) if b else None;obs+=p is not None
        y.append(None if p is None else at(p,10.))
        lanes=[q for q in r['lanes'] if q['side']==side and not q['reprojected']]
        pts=[np.asarray(q['points']) for q in lanes]
        near.append(any(q[:,0].min()<8 for q in pts));reach+=[max(q[:,0].max() for q in pts)] if pts else []
        lab=max(lanes,key=lambda q:np.ptp(np.asarray(q['points'])[:,0]))['style'] if lanes else 'none';styles[lab]=styles.get(lab,0)+1
    d=np.array([b-a for a,b in zip(y[:-1],y[1:]) if a is not None and b is not None])
    small=d[np.abs(d)<=.3] if len(d) else d
    return dict(y10_std_cm=round(float(np.std(small))*100,2) if len(small) else None,j5=int(np.sum(np.abs(d)>.05)),obs=round(obs/len(rows),3),
                near8=round(float(np.mean(near)),3),reach=round(float(np.median(reach)),1) if reach else None,styles=styles)
video,out=sys.argv[1],sys.argv[2];res={}
for spec in sys.argv[3:]:
    name,path=spec.split('=');rows=[json.loads(l) for l in gzip.open(path+'/predictions.jsonl.gz','rt')];t0=rows[0]['stamp'];res[name]={}
    for g,a,b in SEGS[video]:
        part=[r for r in rows if a<=r['stamp']-t0<b];key=f'{g} {a}-{b}'
        res[name][key]={s:seg(part,s) for s in ('left','right')}
        L,R=res[name][key]['left'],res[name][key]['right']
        print(f"{name:8s} {key:15s} L std {L['y10_std_cm']} j5 {L['j5']:3d} obs {L['obs']:.2f} near8 {L['near8']:.2f} reach {L['reach']} {L['styles']} | R std {R['y10_std_cm']} j5 {R['j5']:3d} obs {R['obs']:.2f} near8 {R['near8']:.2f} reach {R['reach']} {R['styles']}")
json.dump(res,open(out,'w'),indent=1)
