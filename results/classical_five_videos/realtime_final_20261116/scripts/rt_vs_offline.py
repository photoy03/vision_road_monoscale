"""Real-time boundaries (realtime_eval frames.json) against the offline run of the same code on the same frames:
frames whose published boundary lines are identical, and the largest lateral difference on the common extent.
Per segment: observed-boundary frames and the 10 m frame-to-frame std, both runs.
usage: rt_vs_offline.py VIDEO REALTIME_DIR OFFLINE_RUN_DIR"""
import sys,gzip,json,numpy as np
SEGS=dict(town04=[('junction',0,10),('curve',15,20),('hairpin',20,25),('straight',30,35),('turn',35,40),('curve',50,55)],
          town05=[('straight',0,9),('turn',10,15),('curve',15,30),('turn',35,40),('straight',40,45)],
          maploop=[('straight',5,20),('curve',22,35),('straight2',35,42),('crosswalk',42,48),('island',48,53),('junction2',53,61)])
video,rt,off=sys.argv[1:4]
R=json.load(open(rt+'/frames.json'));O=[json.loads(l) for l in gzip.open(off+'/predictions.jsonl.gz','rt')]
def lines_off(r):return sorted((b['side'],bool(b['reprojected']),[tuple(p) for p in b['points']]) for b in r['boundaries'])
same=0;worst=[];missing=0
for f,o in zip(R,O):
    a=f.get('boundary_lines')
    if a is None:missing+=1;continue
    a=sorted((s,p,[tuple(q) for q in pts]) for s,p,pts in a);b=lines_off(o)
    sa=[(s,p) for s,p,_ in a];sb=[(s,p) for s,p,_ in b]
    if sa==sb and all(np.allclose(np.asarray(x[2]),np.asarray(y[2]),atol=2e-4) for x,y in zip(a,b) if len(x[2])==len(y[2])) and all(len(x[2])==len(y[2]) for x,y in zip(a,b)):same+=1;continue
    d=0.
    for x in a:
        for y in b:
            if x[:2]!=y[:2]:continue
            p,q=np.asarray(x[2]),np.asarray(y[2]);lo,hi=max(p[:,0].min(),q[:,0].min()),min(p[:,0].max(),q[:,0].max())
            if hi>lo:
                s=np.linspace(lo,hi,30);pp=p[np.argsort(p[:,0])];qq=q[np.argsort(q[:,0])]
                d=max(d,float(np.abs(np.interp(s,pp[:,0],pp[:,1])-np.interp(s,qq[:,0],qq[:,1])).max()))
    worst.append((round(f['t'],2),sa,sb,round(d,3)))
print(f'{video}: frames {len(R)}, no real-time boundary message {missing}, identical {same} ({same/len(R):.3f}), differing {len(worst)}')
big=[w for w in worst if w[3]>.05 or w[1]!=w[2]]
print('  differing by >5 cm or in sides:',len(big),big[:8])
def seg_stats(lines_per_frame,ts,a,b):
    out={}
    for side in ('left','right'):
        ys=[];obs=0;n=0
        for lines,t in zip(lines_per_frame,ts):
            if not a<=t<b:continue
            n+=1;c=[np.asarray(p) for s,pr,p in lines if s==side and not pr]
            y=None
            if c:
                obs+=1;p=max(c,key=lambda p:np.ptp(p[:,0]));p=p[np.argsort(p[:,0])]
                if p[0,0]<=10<=p[-1,0]:y=float(np.interp(10,p[:,0],p[:,1]))
            ys.append(y)
        d=np.array([b_-a_ for a_,b_ in zip(ys[:-1],ys[1:]) if a_ is not None and b_ is not None]);d=d[np.abs(d)<=.3] if len(d) else d
        out[side]=(round(obs/max(n,1),3),round(float(np.std(d))*100,2) if len(d) else None)
    return out
ts=[f['t'] for f in R]
rt_lines=[f.get('boundary_lines',[]) for f in R];off_lines=[lines_off(o) for o in O]
for g,a,b in SEGS[video]:
    x=seg_stats(rt_lines,ts,a,b);y=seg_stats(off_lines,ts,a,b)
    print(f'  {g:9s} {a}-{b}s  L obs/std rt {x["left"]} off {y["left"]} | R rt {x["right"]} off {y["right"]}')
