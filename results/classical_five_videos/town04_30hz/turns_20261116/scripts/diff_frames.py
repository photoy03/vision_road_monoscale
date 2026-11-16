"""Frames whose observed boundary differs between two runs: presence, or more than 5 cm somewhere on the
common extent, or its far end moved by more than 1 m.  Grouped into events.
usage: diff_frames.py RUN_A RUN_B"""
import sys,gzip,json,numpy as np
A,B=[[json.loads(l) for l in gzip.open(p+'/predictions.jsonl.gz','rt')] for p in sys.argv[1:3]];t0=A[0]['stamp'];out=[]
for i,(a,b) in enumerate(zip(A,B)):
    for s in ('left','right'):
        pa=[np.asarray(q['points']) for q in a['boundaries'] if q['side']==s and not q['reprojected']]
        pb=[np.asarray(q['points']) for q in b['boundaries'] if q['side']==s and not q['reprojected']]
        if bool(pa)!=bool(pb):out.append((i,s,'lost' if pa else 'gained'));continue
        if not pa:continue
        p=max(pa,key=lambda p:np.ptp(p[:,0]));q=max(pb,key=lambda p:np.ptp(p[:,0]))
        lo,hi=max(p[0,0],q[0,0]),min(p[-1,0],q[-1,0]);d=0.
        if hi>lo:x=np.linspace(lo,hi,30);d=float(np.abs(np.interp(x,p[:,0],p[:,1])-np.interp(x,q[:,0],q[:,1])).max())
        if d>.05 or abs(p[-1,0]-q[-1,0])>1.:out.append((i,s,f'far {p[-1,0]:.1f}->{q[-1,0]:.1f} max {d*100:.0f}cm'))
ev=[]
for f in out:
    if ev and f[0]-ev[-1][-1][0]<=3 and f[1]==ev[-1][-1][1]:ev[-1].append(f)
    else:ev.append([f])
print(f'{len(out)} frame-sides, {len(ev)} events')
for g in ev:print(f'  {A[g[0][0]]["stamp"]-t0:5.2f}-{A[g[-1][0]]["stamp"]-t0:5.2f}s frames {g[0][0]}-{g[-1][0]} {g[0][1]:5s} {g[0][2]}')
