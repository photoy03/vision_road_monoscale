"""Curve (frames 660-1050) and straight (150-600) stability metrics from predictions.jsonl.gz."""
import sys,gzip,json,numpy as np
from pathlib import Path
SEG=dict(straight=(150,600),curve=(660,1050))
def at(p,x):
    p=np.asarray(p,float);p=p[np.argsort(p[:,0])]
    return float(np.interp(x,p[:,0],p[:,1])) if p[0,0]<=x<=p[-1,0] else None
def curvature(p):
    p=np.asarray(p,float)
    if np.ptp(p[:,0])<3:return None
    x=np.arange(p[:,0].min(),p[:,0].max()+1e-6,.5);y=np.interp(x,*p[np.argsort(p[:,0])].T)
    return 2*np.polyfit(x,y,2)[0]
def diffstd(v,lim=None):
    d=np.array([b-a for a,b in zip(v[:-1],v[1:]) if a is not None and b is not None])
    if lim is not None:d=d[np.abs(d)<=lim]
    return float(np.std(d)) if len(d) else None
def metrics(rows,a,b):
    out={}
    for side in ('left','right'):
        k=[];y10=[];y20=[];y6=[];reach=[];span=[];cover=[];pieces=[];paint=[];styles={}
        for r in rows[a:b]:
            bs=[x for x in r['boundaries'] if x['side']==side and not x['reprojected']]
            p=max(bs,key=lambda x:np.ptp(np.asarray(x['points'])[:,0]),default=None)
            if p is None:k.append(None);y10.append(None);y20.append(None);y6.append(None);reach.append(None);span.append(0);cover.append(False)
            else:
                pts=np.asarray(p['points']);k.append(curvature(pts));y10.append(at(pts,10));y20.append(at(pts,20));y6.append(at(pts,6))
                reach.append(pts[:,0].max());span.append(np.ptp(pts[:,0]));cover.append(y10[-1] is not None)
            ls=[l for l in r['lanes'] if l['side']==side and not l['reprojected']]
            pieces.append(len(ls));paint.append(sum(np.ptp(np.asarray(l['points'])[:,0]) for l in ls))
            lab=max((l for l in r['lanes'] if l['side']==side),key=lambda l:np.ptp(np.asarray(l['points'])[:,0]),default=None)
            s='none' if lab is None else lab['style'];styles[s]=styles.get(s,0)+1
        on=[x is not None for x in y10]
        present=[s>0 for s in span]
        out[side]=dict(curv_step_std=round(diffstd(k,.05) or 0,5),y20_step_cm=round((diffstd(y20,.3) or 0)*100,2),y10_step_cm=round((diffstd(y10,.3) or 0)*100,2),
            y6_step_cm=round((diffstd(y6,.3) or 0)*100,2),cover10=round(float(np.mean(cover)),3),toggles=int(sum(1 for x,y in zip(present[:-1],present[1:]) if y and not x)),
            reach_median=round(float(np.median([r for r in reach if r is not None])),1) if any(r is not None for r in reach) else None,
            reach_step_std=round(diffstd(reach) or 0,2),span12=int(sum(s>=12 for s in span)),paint_median=round(float(np.median(paint)),1),
            no_paint=round(float(np.mean([q==0 for q in pieces])),3),multi_piece=round(float(np.mean([q>=2 for q in pieces])),3),styles=styles)
    return out
def run(path):
    rows=[json.loads(l) for l in gzip.open(Path(path)/'predictions.jsonl.gz','rt')]
    return {g:metrics(rows,a,b) for g,(a,b) in SEG.items()}
if __name__=='__main__':
    res={}
    for spec in sys.argv[1:]:
        name,path=spec.split('=');res[name]=run(path)
        for g in SEG:
            for s in ('left','right'):
                m=res[name][g][s];print('%-12s %-8s %-5s curv %.5f y20 %5.2f y10 %5.2f y6 %5.2f cover %.2f tog %2d reach %s±%.1f span12 %3d paint %.1f nopaint %.2f multi %.2f %s'%(name,g,s,m['curv_step_std'],m['y20_step_cm'],m['y10_step_cm'],m['y6_step_cm'],m['cover10'],m['toggles'],m['reach_median'],m['reach_step_std'],m['span12'],m['paint_median'],m['no_paint'],m['multi_piece'],m['styles']))
    json.dump(res,open(Path(sys.argv[0]).with_suffix('.json'),'w'),indent=1)
