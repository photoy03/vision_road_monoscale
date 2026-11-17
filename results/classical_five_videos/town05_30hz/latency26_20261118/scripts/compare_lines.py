#!/usr/bin/env python3
"""Plan 25: compare two dump_lines.py outputs frame by frame.  Identical frames (every field, every bit), and
for the others the largest position difference of the own lines (paint lanes and boundaries), outer lines and
centrelines (points matched in order; a different point count counts as a structural change), and frames where
a line's side, style or presence changed.  usage: compare_lines.py A.jsonl.gz B.jsonl.gz [OUT.json]"""
import sys,json,gzip
import numpy as np
A=[json.loads(l) for l in gzip.open(sys.argv[1],'rt')];B=[json.loads(l) for l in gzip.open(sys.argv[2],'rt')]
same=0;worst={'lanes':0.,'bounds':0.,'outer':0.,'centres':0.};structural=[];side_changes=[];n=min(len(A),len(B))
def key(l):return (l.get('side') or l.get('name'),l.get('outer'))
for i in range(n):
    a,b=A[i],B[i]
    if a==b:same+=1;continue
    for part in worst:
        la,lb=a.get(part,[]),b.get(part,[])
        if [key(l) for l in la]!=[key(l) for l in lb]:side_changes.append((i,part));continue
        for x,y in zip(la,lb):
            pa,pb=np.asarray(x['points']),np.asarray(y['points'])
            if pa.shape!=pb.shape:structural.append((i,part));continue
            if len(pa):worst[part]=max(worst[part],float(np.abs(pa-pb).max()))
            if x.get('style')!=y.get('style'):side_changes.append((i,part+'.style'))
res=dict(frames=n,identical=same,max_diff_m=worst,structural_frames=len(set(f for f,_ in structural)),side_or_presence_change_frames=len(set(f for f,_ in side_changes)),
         examples=dict(structural=structural[:10],changes=side_changes[:10]),lengths=(len(A),len(B)))
print(json.dumps(res))
if len(sys.argv)>3:json.dump(res,open(sys.argv[3],'w'),indent=1)
