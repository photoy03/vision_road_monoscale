"""Double-line strand metrics per segment from predictions.jsonl.gz.

y6/y10 step: std of frame-to-frame lateral change of the boundary (|d|<=0.3 m), cm.
jumps: frame-to-frame boundary changes > 5 cm at 6 m.
both6/both10: frames where two paint strands 0.15-0.40 m apart cover that distance.
mid3: of those, fraction with the boundary within 3 cm of the strands' midpoint.
"""
import sys,gzip,json,numpy as np
from pathlib import Path
SEG=[('junction',0,5),('straight',5,20),('curve',22,35),('straight2',35,42),('crosswalk',42,48),('island',48,53),('junction2',53,61)]
def at(p,x):
    p=np.asarray(p,float);p=p[np.argsort(p[:,0])]
    return float(np.interp(x,p[:,0],p[:,1])) if p[0,0]<=x<=p[-1,0] else None
def boundary(r,side):
    bs=[x for x in r['boundaries'] if x['side']==side and not x['reprojected']]
    return max(bs,key=lambda x:np.ptp(np.asarray(x['points'])[:,0]),default=None)
def strands(r,side,x):
    v=sorted(y for l in r['lanes'] if l['side']==side and not l['reprojected'] for y in [at(l['points'],x)] if y is not None)
    groups=[]
    for y in v:
        if groups and y-groups[-1][-1]<.08:groups[-1].append(y)
        else:groups.append([y])
    return [float(np.mean(g)) for g in groups]
def pair(s,ref):
    best=None
    for i in range(len(s)):
        for j in range(i+1,len(s)):
            if .15<=s[j]-s[i]<=.40:
                m=(s[i]+s[j])/2
                if best is None or (ref is not None and abs(m-ref)<abs(best-ref)):best=m
    return best
def seg(rows,side):
    out={}
    y={d:[] for d in (6,10)};n={6:[],10:[]};mid={6:[],10:[]}
    for r in rows:
        b=boundary(r,side)
        for d in (6,10):
            v=at(b['points'],d) if b else None;y[d].append(v)
            s=strands(r,side,d);n[d].append(len(s))
            m=pair(s,v)
            if m is not None and v is not None:mid[d].append(abs(v-m))
    for d in (6,10):
        diff=np.array([b-a for a,b in zip(y[d][:-1],y[d][1:]) if a is not None and b is not None])
        out[f'y{d}_step_cm']=round(float(np.std(diff[np.abs(diff)<=.3]))*100,2) if len(diff) else None
        out[f'y{d}_jumps5']=int(np.sum(np.abs(diff)>.05))
        out[f'covered{d}']=round(float(np.mean([v is not None for v in y[d]])),3)
        if side=='left':
            out[f'both{d}']=round(float(np.mean([k>=2 for k in n[d]])),3);out[f'one{d}']=round(float(np.mean([k==1 for k in n[d]])),3)
            out[f'mid3_{d}']=round(float(np.mean(np.array(mid[d])<=.03)),3) if mid[d] else None
            out[f'mid_med_cm_{d}']=round(float(np.median(mid[d]))*100,2) if mid[d] else None
    if side=='left':out['three15']=round(float(np.mean([len(strands(r,side,15))>=3 for r in rows])),3)
    return out
def run(path):
    rows=[json.loads(l) for l in gzip.open(Path(path)/'predictions.jsonl.gz','rt')]
    t0=rows[0]['stamp'];res={}
    for name,a,b in SEG:
        part=[r for r in rows if a<=r['stamp']-t0<b]
        res[name]={s:seg(part,s) for s in ('left','right')}
    return res
if __name__=='__main__':
    res={}
    for spec in sys.argv[1:]:
        name,path=spec.split('=');res[name]=run(path)
        for g,_,_ in SEG:
            L=res[name][g]['left'];R=res[name][g]['right']
            print('%-10s %-9s L y6 %5.2f j6 %3d y10 %5.2f j10 %3d both6 %.2f one6 %.2f both10 %.2f mid3_6 %s mid3_10 %s med6 %s 3@15 %.2f | R y6 %5.2f j6 %3d y10 %5.2f'%(
                name,g,L['y6_step_cm'] or 0,L['y6_jumps5'],L['y10_step_cm'] or 0,L['y10_jumps5'],L['both6'],L['one6'],L['both10'],L['mid3_6'],L['mid3_10'],L['mid_med_cm_6'],L['three15'],R['y6_step_cm'] or 0,R['y6_jumps5'],R['y10_step_cm'] or 0))
    if len(sys.argv)>1:json.dump(res,open(Path(sys.argv[1].split('=')[1])/'strand_metrics.json','w'),indent=1)
