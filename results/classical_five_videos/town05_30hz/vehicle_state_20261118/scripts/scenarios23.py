#!/usr/bin/env python3
"""Plan 23 fake-motion scripts: plan 21's scripts (same delta(t), same stretches) with the car also
turning along its path, so a steering angle can explain the motion.

Plan 21 only moved the car sideways (every line by -delta, the pose by +delta along its left normal,
heading unchanged): the car slid without turning, which no steering angle can produce.  Here the car's
heading also turns by psi = atan(d delta/dt / v) (v: the recording's speed there): the pose's yaw gets
+psi, and every observed line is moved by -delta and then turned by -psi about the car (rear axle), as a
camera on a car turned psi would see it.  Lane centrelines are then re-measured (offset, heading and
curvature at 5 m, vision_road.centreline.at_reference) and named by position as in plan 21.  The plain
recording (script 'none') is unchanged."""
import json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path('/home/jaehun/Desktop/vision-road');sys.path.insert(0,str(ROOT/'src/vision_road'))
sys.path.insert(0,str(ROOT/'results/classical_five_videos/town05_30hz/lane_guidance_20261118/scripts'))
import scenarios as S21  # noqa: E402
from vision_road.centreline import at_reference  # noqa: E402
SCRIPTS=S21.SCRIPTS;load=S21.load


def turn(points,delta,psi):
    """Real base_link points into the moved and turned car's base_link."""
    p=np.asarray(points,float);qx=p[:,0];qy=p[:,1]-delta;c,s=math.cos(psi),math.sin(psi)
    out=np.column_stack([c*qx+s*qy,-s*qx+c*qy]);return out[np.argsort(out[:,0],kind='stable')]


def lanes(centres,delta,psi):
    moved=[]
    for c in centres:
        p=turn(c['points'],delta,psi);ref=at_reference(p,5.,15.)
        if ref is None:continue
        moved.append(dict(c,points=p.tolist(),**{k:(round(v,4) if isinstance(v,float) else v) for k,v in ref.items()}))
    return S21.shifted_lanes(moved,0.)


def frames(script,rows=None):
    """[(t, pose (x, y, yaw), lanes, delta, psi, row)]: plan 21's script with the heading turned along the path."""
    rows=rows or load();t0=rows[0]['stamp']
    poses=np.array(json.load(open('/media/jaehun/38F7-677B/vision-road-cache/town05_offline/frames.json'))['poses']);poses[:,3]=np.unwrap(poses[:,3])
    start,length,delta=SCRIPTS[script];out=[];h=1/60.
    speed=np.hypot(np.gradient(poses[:,1],poses[:,0]),np.gradient(poses[:,2],poses[:,0]))
    for r in rows:
        t=r['stamp']-t0-start
        if not 0<=t<=length:continue
        d=delta(t);x,y,yaw=(float(np.interp(r['stamp'],poses[:,0],poses[:,i])) for i in (1,2,3))
        v=max(float(np.interp(r['stamp'],poses[:,0],speed)),1.)
        psi=math.atan2((delta(t+h)-delta(t-h))/(2*h),v) if script!='none' else 0.
        out.append((r['stamp'],(x-d*math.sin(yaw),y+d*math.cos(yaw),yaw+psi),lanes(r['centres'],d,psi) if script!='none' else S21.shifted_lanes(r['centres'],0.),d,psi,r))
    return out
