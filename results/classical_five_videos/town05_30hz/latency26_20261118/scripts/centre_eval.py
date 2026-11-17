#!/usr/bin/env python3
"""Plan 20 centreline metrics from dump_lines.py records (runs/NAME_VIDEO.jsonl.gz).
usage: centre_eval.py VIDEO NAME

presence   ego centreline in frames with both own boundaries (should be 100%, not estimated) and in
           frames with one (estimated); per 5 s
width      ego width 3.2-3.8 m share of the both-boundary frames; per 5 s out-of-range frames
wobble     frame to frame, vehicle motion removed (previous line carried into this frame with the
           rear-axle poses), at x = max(5 m, where both lines start): centreline y vs own boundaries' y (cm); also the 0 m
           extrapolation (offset_at_axle_m vs the carried previous line at 0 m, linear past its start),
           heading (mrad, previous heading minus the yaw change) and curvature (1/km)
neighbour  the same for the left/right lanes, and their flags (opposite, crossable, edge_unknown, lane_like)"""
import sys,json,gzip,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
video,name=sys.argv[1],sys.argv[2]
rows=[json.loads(l) for l in gzip.open(HERE.parent/'runs'/f'{name}_{video}.jsonl.gz','rt')]
poses=np.array(json.load(open(f'/media/jaehun/38F7-677B/vision-road-cache/{video}_offline/frames.json'))['poses']);poses[:,3]=np.unwrap(poses[:,3])
t0=rows[0]['stamp']
def pose(t):return np.array([np.interp(t,poses[:,0],poses[:,i]) for i in (1,2,3)])
def relative(a,b):
    c,s=math.cos(a[2]),math.sin(a[2]);wx,wy=b[0]-a[0],b[1]-a[1];return c*wx+s*wy,-s*wx+c*wy,b[2]-a[2]
def carry(p,m):
    dx,dy,dyaw=m;c,s=math.cos(dyaw),math.sin(dyaw);px=p[:,0]-dx;py=p[:,1]-dy
    return np.column_stack([c*px+s*py,-s*px+c*py])
def at(p,x):
    p=np.asarray(p);return float(np.interp(x,p[:,0],p[:,1])) if len(p)>=2 and p[0,0]<=x<=p[-1,0] else None
def stat(v,scale):
    v=np.abs(np.array(v))*scale
    return dict(n=len(v),rms=round(float(np.sqrt(np.mean(v**2))),3) if len(v) else None,p95=round(float(np.percentile(v,95)),3) if len(v) else None,
                max=round(float(v.max()),2) if len(v) else None)
P=[pose(r['stamp']) for r in rows];T=[r['stamp']-t0 for r in rows]
out=dict(video=video,frames=len(rows))
# Presence and width
seg={};both_frames=0;both_ego=0;one_frames=0;one_ego=0;one_est=0;widths=[];bad=[]
for r,t in zip(rows,T):
    sides={b['side'] for b in r['bounds']};ego=next((c for c in r['centres'] if c['name']=='ego'),None);b=int(t//5);s=seg.setdefault(b,[0,0,0,0,0])
    if {'left','right'}<=sides:
        both_frames+=1;s[0]+=1;ok=ego is not None and not ego['estimated'];both_ego+=ok;s[1]+=ok
        if ego is not None:
            widths.append(ego['width_m'])
            if not 3.2<=ego['width_m']<=3.8:bad.append((round(t,2),ego['width_m']));s[4]+=1
    elif sides&{'left','right'}:
        one_frames+=1;s[2]+=1;ok=ego is not None and ego['estimated'];one_est+=ok;one_ego+=ego is not None;s[3]+=ok
w=np.array(widths)
out['presence']=dict(both_frames=both_frames,ego_in_both=round(both_ego/max(1,both_frames),4),one_frames=one_frames,ego_estimated_in_one=round(one_est/max(1,one_frames),4),
                     per5s={f'{k*5}-{k*5+5}':dict(both=v[0],ego=round(v[1]/v[0],3) if v[0] else None,one=v[2],estimated=round(v[3]/v[2],3) if v[2] else None,width_out=v[4]) for k,v in sorted(seg.items())})
out['ego_width']=dict(n=len(w),share_3_2_3_8=round(float(np.mean((w>=3.2)&(w<=3.8))),4),median=round(float(np.median(w)),3),p1=round(float(np.percentile(w,1)),3),p99=round(float(np.percentile(w,99)),3),out=bad)
# Wobble
lanes={}
for i in range(1,len(rows)):
    m=relative(P[i-1],P[i])
    prev={c['name']:c for c in rows[i-1]['centres']};cur={c['name']:c for c in rows[i]['centres']}
    for n,c in cur.items():
        if n not in prev:continue
        q=prev[n];d=lanes.setdefault(n,dict(y5=[],y0=[],head=[],curv=[],y5_est=[]))
        pc=carry(np.asarray(q['points']),m);xs=max(5.,c['points'][0][0],pc[0,0])
        a=at(c['points'],xs);b=at(pc,xs)
        if a is not None and b is not None:(d['y5_est'] if c['estimated'] or q['estimated'] else d['y5']).append(a-b)
        if c['estimated'] or q['estimated']:continue
        # 0 m: this frame's extrapolation against the previous one's, carried (straight line, so carry the point and heading)
        x0,y0=0.,q['offset_at_axle_m'];hx,hy=math.cos(q['heading_rad']),math.sin(q['heading_rad'])
        dx,dy,dyaw=m;cs,sn=math.cos(dyaw),math.sin(dyaw);px,py=x0-dx,y0-dy;X,Y=cs*px+sn*py,-sn*px+cs*py
        H=q['heading_rad']-dyaw;Y0=Y-X*math.tan(H);d['y0'].append(c['offset_at_axle_m']-Y0)
        if abs(c['ref_x_m']-q['ref_x_m'])<.5:d['head'].append(c['heading_rad']-H);d['curv'].append(c['curvature_1pm']-q['curvature_1pm'])
bw={'left':[],'right':[]}
for i in range(1,len(rows)):
    m=relative(P[i-1],P[i]);prev={b['side']:b for b in rows[i-1]['bounds']}
    for b in rows[i]['bounds']:
        q=prev.get(b['side'])
        if q is None or b['rep'] or q['rep']:continue
        pc=carry(np.asarray(q['points']),m);xs=max(5.,b['points'][0][0],pc[0,0])
        a=at(b['points'],xs);c=at(pc,xs)
        if a is not None and c is not None:bw[b['side']].append(a-c)
out['wobble_cm_mrad']={n:dict(y5_cm=stat(d['y5'],100),y5_estimated_cm=stat(d['y5_est'],100),y0_axle_cm=stat(d['y0'],100),heading_mrad=stat(d['head'],1e3),curvature_per_km=stat(d['curv'],1e3)) for n,d in lanes.items()}
out['boundary_wobble_y5_cm']={s:stat(v,100) for s,v in bw.items()}
# Neighbour flags
flags={}
for n in ('left','right'):
    cs=[c for r in rows for c in r['centres'] if c['name']==n]
    if cs:flags[n]=dict(frames=len(cs),width_median=round(float(np.median([c['width_m'] for c in cs])),3),
                         **{k:round(float(np.mean([bool(c.get(k)) for c in cs])),3) for k in ('opposite_direction','crossable','edge_unknown','lane_like','estimated')})
out['neighbours']=flags
out['neighbour_without_ego']=[round(t,2) for r,t in zip(rows,T) if r['centres'] and not any(c['name']=='ego' for c in r['centres'])]
(HERE.parent/'metrics').mkdir(exist_ok=True);(HERE.parent/'metrics'/f'{name}_{video}.json').write_text(json.dumps(out,indent=1))
p=out['presence'];print(f"{video} {name}: ego in both-boundary frames {p['ego_in_both']} ({p['both_frames']}), estimated in one-boundary frames {p['ego_estimated_in_one']} ({p['one_frames']})")
e=out['ego_width'];print(f"  ego width 3.2-3.8 m: {e['share_3_2_3_8']} median {e['median']} p1/p99 {e['p1']}/{e['p99']}, out {len(e['out'])}")
for n,d in out['wobble_cm_mrad'].items():print(f"  {n:5s} y5 {d['y5_cm']} | y0 {d['y0_axle_cm']} | head {d['heading_mrad']} | curv {d['curvature_per_km']} | est y5 {d['y5_estimated_cm']}")
print('  boundaries y5',out['boundary_wobble_y5_cm']);print('  neighbours',flags);print('  neighbour without ego',out['neighbour_without_ego'])
