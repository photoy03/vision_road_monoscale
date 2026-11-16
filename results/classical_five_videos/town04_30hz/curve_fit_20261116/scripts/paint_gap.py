"""Paint-boundary gap: in each frame and side, |y(observed paint lane) - y(boundary)| at x = 6, 8, 10, 15, 20 m,
for every observed (not reprojected) paint lane covering x and the side's longest observed boundary.
Per segment p95 (cm).  For a double line the midpoint of two strands 0.15-0.40 m apart also counts as paint.
usage: paint_gap.py maploop|town05|town04 OUT.json NAME=RUN_DIR ..."""
import sys,gzip,json,numpy as np
SEGS=dict(town04=[('straight',30,35),('curve',15,20),('curve',20,25),('curve',50,55),('curve',55,58),('turn',35,40),('straight',40,45)],
          town05=[('straight',0,9),('curve',15,30),('straight',40,45),('straight',55,60)],
          maploop=[('straight',5,20),('curve',22,35),('straight2',35,42),('island',48,53)])
XS=(6,8,10,15,20)
def at(p,x):return float(np.interp(x,p[:,0],p[:,1])) if p[0,0]<=x<=p[-1,0] else None
def gaps(r,side):
    b=[np.asarray(q['points']) for q in r['boundaries'] if q['side']==side and not q['reprojected']]
    if not b:return {}
    p=max(b,key=lambda p:np.ptp(p[:,0]));out={}
    lanes=[np.asarray(q['points']) for q in r['lanes'] if q['side']==side and not q['reprojected']]
    for x in XS:
        yb=at(p,x)
        if yb is None:continue
        ys=[at(q,x) for q in lanes];ys=[y for y in ys if y is not None]
        # A double line's boundary is the middle of its two strands: pairs 0.15-0.40 m apart add their midpoint.
        ys=ys+[(a+b)/2 for i,a in enumerate(ys) for b in ys[i+1:] if .15<=abs(a-b)<=.40]
        if ys:out[x]=min(abs(y-yb) for y in ys)
    return out
video,outp=sys.argv[1],sys.argv[2];res={}
for spec in sys.argv[3:]:
    name,path=spec.split('=');rows=[json.loads(l) for l in gzip.open(path+'/predictions.jsonl.gz','rt')];t0=rows[0]['stamp'];res[name]={}
    for g,a,b in SEGS[video]:
        part=[r for r in rows if a<=r['stamp']-t0<b];key=f'{g} {a}-{b}';res[name][key]={}
        line=f'{name:8s} {key:14s}'
        for side in ('left','right'):
            v={x:[] for x in XS}
            for r in part:
                for x,d in gaps(r,side).items():v[x].append(d)
            res[name][key][side]={x:(round(float(np.percentile(v[x],95))*100,1) if v[x] else None) for x in XS}
            line+=f' {side[0].upper()} '+'/'.join('-' if res[name][key][side][x] is None else f'{res[name][key][side][x]:.0f}' for x in XS)
        print(line)
json.dump(res,open(outp,'w'),indent=1)
