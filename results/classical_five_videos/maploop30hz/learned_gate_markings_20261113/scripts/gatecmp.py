"""Compare offline runs by segment group: observation, shake, implausible boundaries, transverse paint, labels."""
import sys,gzip,json,numpy as np
from pathlib import Path
GROUPS=dict(straight=[(5,20),(35,42)],curve=[(22,35)],junction=[(0,5),(42,48),(53,61)])
def at(p,x=10.):
    p=np.asarray(p);p=p[np.argsort(p[:,0])]
    return float(np.interp(x,p[:,0],p[:,1])) if p[0,0]<=x<=p[-1,0] else None
def score(run):
    rows=[json.loads(l) for l in gzip.open(Path(run)/'predictions.jsonl.gz','rt')];t0=rows[0]['stamp'];out={}
    for g,spans in GROUPS.items():
        sel=[r for r in rows if any(a<=r['stamp']-t0<b for a,b in spans)];res={}
        for side in ('left','right'):
            y=[];bad=0;tr=0;frags=0;labels={}
            for r in sel:
                b=[x for x in r['boundaries'] if x['side']==side and not x['reprojected']]
                v=at(b[0]['points']) if b else None;y.append(v)
                if v is not None and not 1.2<=abs(v)<=2.6:bad+=1
                for l in r['lanes']:
                    if l['side']!=side or l['reprojected']:continue
                    p=np.asarray(l['points']);frags+=1
                    if np.ptp(p[:,0])>0 and abs(np.arctan2(np.ptp(p[:,1]),np.ptp(p[:,0])))>np.radians(15) and np.ptp(p[:,1])>.3:tr+=1
                lab=max((l for l in r['lanes'] if l['side']==side),key=lambda l:np.ptp(np.asarray(l['points'])[:,0]),default=None)
                k='none' if lab is None else lab['style'];labels[k]=labels.get(k,0)+1
            d=[b-a for a,b in zip(y[:-1],y[1:]) if a is not None and b is not None]
            small=[x for x in d if abs(x)<=.3]
            res[side]=dict(obs=round(sum(v is not None for v in y)/max(1,len(y)),3),std_cm=round(float(np.std(small))*100,2) if small else None,
                           jumps=len(d)-len(small),implausible=bad,transverse=tr,fragments=frags,labels=labels)
        out[g]=res
    return out
names=sys.argv[2:];base=Path(sys.argv[1])
for g in GROUPS:
    print('==',g)
    for n in names:
        r=score(base/n)[g]
        print('%-14s'%n,' | '.join('%s obs %.3f std %s jump %d bad %d trans %d frag %d %s'%(s[0],v['obs'],v['std_cm'],v['jumps'],v['implausible'],v['transverse'],v['fragments'],{k:c for k,c in v['labels'].items() if k!='none'}) for s,v in r.items()))
