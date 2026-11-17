#!/usr/bin/env python3
"""Plan 26: the C++ result against the Python result (two dump_lines.py outputs), per line kind: own lines
(boundaries, and the paint lanes), outer lines, centrelines.  For each frame and kind, the largest lateral
difference between matching lines (same side/name, the other's y interpolated at this one's x over the common
range); distribution over all frames (median, p95, max), the share of frames over 5 cm, and frames where a
line's side changed or a line appeared/disappeared (or its style changed).
usage: compare_errors.py PYTHON.jsonl.gz NATIVE.jsonl.gz [OUT.json]"""
import sys,json,gzip
import numpy as np
A=[json.loads(l) for l in gzip.open(sys.argv[1],'rt')];B=[json.loads(l) for l in gzip.open(sys.argv[2],'rt')]
KINDS={'own (boundaries)':'bounds','own (paint lanes)':'lanes','outer':'outer','centre':'centres'}
def name(l):return l.get('name') or l.get('side')
def lateral(p,q):
    p=np.asarray(p,float);q=np.asarray(q,float)
    if len(p)<1 or len(q)<2:return 0. if len(p)==len(q) else np.inf
    q=q[np.argsort(q[:,0])];lo,hi=max(p[:,0].min(),q[0,0]),min(p[:,0].max(),q[-1,0])
    sel=(p[:,0]>=lo)&(p[:,0]<=hi)
    if not sel.any():return np.inf
    return float(np.abs(p[sel,1]-np.interp(p[sel,0],q[:,0],q[:,1])).max())
res={};examples={}
for kind,key in KINDS.items():
    diffs=[];changed=[];styles=[]
    for i,(a,b) in enumerate(zip(A,B)):
        la,lb=a.get(key,[]),b.get(key,[])
        if key=='lanes':  # many paint pieces per side: compare the pieces in order when the counts match
            if [name(l) for l in la]!=[name(l) for l in lb]:changed.append(i);continue
            d=max([lateral(x['points'],y['points']) for x,y in zip(la,lb)] or [0.])
        else:
            na=sorted(name(l) for l in la);nb=sorted(name(l) for l in lb)
            if na!=nb:changed.append(i);continue
            da={name(l):l for l in la};db={name(l):l for l in lb}
            d=max([lateral(da[k]['points'],db[k]['points']) for k in da] or [0.])
            if any(da[k].get('style')!=db[k].get('style') for k in da):styles.append(i)
        diffs.append(d)
    v=np.array(diffs);fin=v[np.isfinite(v)]
    res[kind]=dict(frames=len(A),compared=len(v),median_m=float(np.median(fin)) if len(fin) else 0.,p95_m=float(np.percentile(fin,95)) if len(fin) else 0.,
                   max_m=float(fin.max()) if len(fin) else 0.,over_5cm_share=float(np.mean(v>.05)) if len(v) else 0.,no_overlap_frames=int(np.sum(~np.isfinite(v))),
                   side_or_presence_change_frames=len(changed),style_change_frames=len(styles),identical_frames=int(np.sum(v==0)))
    order=np.argsort(-np.where(np.isfinite(v),v,1e9))[:5];examples[kind]=dict(largest=[(int(i),float(v[i])) for i in order if v[i]>0],changed=changed[:10])
out=dict(results=res,examples=examples,identical_whole_frames=sum(a==b for a,b in zip(A,B)))
print(json.dumps(out,indent=1))
if len(sys.argv)>3:json.dump(out,open(sys.argv[3],'w'),indent=1)
