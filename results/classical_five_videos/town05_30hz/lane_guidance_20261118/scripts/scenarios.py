#!/usr/bin/env python3
"""Fake lateral motion scripts on real Town05 lane centrelines (plan 21).

Town05 has no weaving, no centre-line crossing and no lane change.  A script adds a lateral move
delta(t) (m, + to the left) of the car to a stretch of the recording: every observed centreline moves
by -delta in base_link, the pose moves by +delta along its left normal.  Lanes are then named by
position as the detector would once the car is over a line: the lane the car is in is 'ego', the next
one each side 'left' / 'right', farther ones and ones out of the +-6 m view are dropped.  Each lane keeps
its own boundaries (styles, yellow) and whether it is oncoming; 'crossable' is recomputed for the new
ego.  Every lane also carries 'origin', its name in the recording, to check the target.

scripts (seconds from the stretch start):
  wobble    at 15 s, curve: 0.45 m sine, 4 s period, 12 s: inside the lane (the car's side stays 0.4 m off the lines)
  yellow    at 0 s, straight: over the yellow centre line: +2.2 m in 2 s (car centre 0.44 m past it), held 1 s, back in 2 s
  dashed    at 15 s, curve: onto the white dashed line: -1.15 m in 1.5 s (car side 0.3 m over it), held 1 s, back in 1.5 s
  change    at 15 s, curve: to the right lane: -3.5 m in 4 s (0.9 m/s), held 8 s
  none      the whole recording, no added motion
delta follows a smooth (cosine) ramp between the stated levels."""
import json,gzip,math
from pathlib import Path
import numpy as np
RUN=Path('/home/jaehun/Desktop/vision-road/results/classical_five_videos/town05_30hz/centrelines_20261118/runs/final_town05.jsonl.gz')
# Stretches without junctions (s from the recording start): 0-9 s straight with the oncoming lane on the left,
# 15-30 s a long right-hand curve with a lane on the right (no left lane in view there).


def ramp(t,points):
    """points: [(time, level), ...]; cosine blend between consecutive levels."""
    for (t0,a),(t1,b) in zip(points[:-1],points[1:]):
        if t0<=t<=t1:u=(t-t0)/max(t1-t0,1e-9);return a+(b-a)*(1-math.cos(math.pi*u))/2
    return points[-1][1] if t>points[-1][0] else points[0][1]


SCRIPTS=dict(  # name: (start s, length s, delta(t))
    wobble=(15.,12.,lambda t:.45*math.sin(2*math.pi*(t-1.)/4.) if 1.<=t<=11. else 0.),
    yellow=(.5,8.5,lambda t:ramp(t,[(0,0.),(1.5,0.),(3.5,2.2),(4.5,2.2),(6.5,0.),(8.5,0.)])),
    dashed=(15.,9.,lambda t:ramp(t,[(0,0.),(2.,0.),(3.5,-1.15),(4.5,-1.15),(6.,0.),(9.,0.)])),
    change=(15.,14.,lambda t:ramp(t,[(0,0.),(2.,0.),(6.,-3.5),(14.,-3.5)])),
    none=(0.,1e9,lambda t:0.),
)


def load():
    rows=[json.loads(l) for l in gzip.open(RUN,'rt')];return rows


def shifted_lanes(centres,delta):
    """The frame's lanes after the car moved delta (m, left +), named by position."""
    lanes=[]
    for c in centres:
        p=np.asarray(c['points'],float).copy();p[:,1]-=delta
        lane=dict(c,points=p.tolist(),offset_m=c['offset_m']-delta,offset_at_axle_m=c['offset_at_axle_m']-delta,origin=c['name'])
        lane['origin_opposite']=c['name']=='left' and bool(c.get('opposite_direction'))
        lanes.append(lane)
    # Out of view: centre beyond the +-6 m BEV at the reference point.
    lanes=[l for l in lanes if abs(l['offset_m'])<=6.]
    if not lanes:return []
    lanes.sort(key=lambda l:-l['offset_at_axle_m'])  # left to right
    ego=min(range(len(lanes)),key=lambda i:abs(lanes[i]['offset_at_axle_m']))
    out=[]
    for i,name,index in ((ego,'ego',0),(ego-1,'left',-1),(ego+1,'right',1)):
        if 0<=i<len(lanes):out.append(dict(lanes[i],name=name,index=index))
    e=out[0]
    for l in out[1:]:
        shared=e['left_boundary'] if l['name']=='left' else e['right_boundary']
        l['crossable']=bool(shared is not None and shared['style']=='dashed' and not shared['yellow'])
        l['opposite_direction']=l['origin_opposite']
    e['opposite_direction']=False
    return out


def frames(script,rows=None):
    """[(t, pose (x, y, yaw), lanes, delta)] for a script over the stretch."""
    rows=rows or load();t0=rows[0]['stamp']
    poses=np.array(json.load(open('/media/jaehun/38F7-677B/vision-road-cache/town05_offline/frames.json'))['poses']);poses[:,3]=np.unwrap(poses[:,3])
    start,length,delta=SCRIPTS[script];out=[]
    for r in rows:
        t=r['stamp']-t0-start
        if not 0<=t<=length:continue
        d=delta(t);x,y,yaw=(float(np.interp(r['stamp'],poses[:,0],poses[:,i])) for i in (1,2,3))
        out.append((r['stamp'],(x-d*math.sin(yaw),y+d*math.cos(yaw),yaw),shifted_lanes(r['centres'],d),d,r))
    return out
