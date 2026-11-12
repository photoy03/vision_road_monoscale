"""Stop lines, crosswalks and other road markings in the ego lane, from the BEV.

Only "a marking is here" is reported; the content of text or the direction of
an arrow is not read.  Per frame candidates come from the CARLA detector's
gray image and the ego lane boundaries; the learned lane mask, when present,
separates lane lines from other paint.  MarkingTracker confirms candidates
M-of-N in world coordinates so they neither blink nor slide with the vehicle.

Transverse/longitudinal bar separation in the BEV follows the classical
approach surveyed in Bar-Hillel et al., "Recent progress in road and lane
detection: a survey", Machine Vision and Applications 25 (2014).
"""
import math
from dataclasses import dataclass

import cv2
import numpy as np

from .carla import shifted


@dataclass
class MarkingSettings:
    max_forward_m: float = 25.
    default_half_lane_m: float = 1.75
    stop_cover: float = .6          # a stop line covers this fraction of the lane width
    stop_thickness_m: tuple = (.1, 1.)
    grid_bars: int = 3              # this many transverse bars within grid_span_m: a box junction, not a stop line
    grid_span_m: float = 12.
    crosswalk_bar_length_m: float = 1.  # bars cut by the BEV's near edge are short
    crosswalk_bar_width_m: tuple = (.25, .9)  # zebra bars; lane dashes are about 0.15 m wide
    crosswalk_bars: int = 4
    crosswalk_gap_m: float = 1.5    # lateral gap between neighbouring zebra bars
    marking_area_m2: tuple = (1., 12.)  # road studs and manhole covers (up to about 0.9 m2) stay below
    marking_main_piece: float = .8  # an arrow or letter is one solid piece ...
    marking_solidity: float = .4    # ... filling much of its convex hull ...
    marking_gray_std: float = 10.   # ... with even paint inside its edge: arrows 8.4-8.8, repaired patches 10.8-12 (maploop)
    stop_yellow_max: float = .15    # a bar with more yellow is a box junction or lane line
    marking_max_length_m: float = 8.
    marking_max_forward_m: float = 15.  # an arrow's shape is resolved only this close in the BEV
    lane_margin_m: float = .3       # markings start this far inside the boundaries
    contrast_min: float = 18.       # paint is this much brighter than the surrounding mean
    background_m: float = 3.
    stop_contrast_min: float = 15.  # a stop line against 0.45 m before and after it
    confirm_window: int = 5
    confirm_hits: int = 3
    match_m: float = 1.5
    miss_frames: int = 5            # a confirmed marking is kept this many frames without a match


def lane_edges(c,boundaries,x,default):
    edge={}
    for side,sign in (('left',1.),('right',-1.)):
        line=next((np.asarray(b.points) for b in boundaries if b.side==side and not b.reprojected and len(b.points)>=2),None)
        if line is None:edge[side]=np.full(len(x),sign*default)
        else:
            line=line[np.argsort(line[:,0])];edge[side]=np.clip(np.interp(x,line[:,0],line[:,1]),-5.5,5.5)
    return edge


def detect(det,bev,valid,boundaries,learned=None,s=None):
    """Candidates in base_link for one frame.

    stop_line: dict(kind, x, left, right, thickness); crosswalk / marking: dict(kind, x0, x1, y0, y1, area)."""
    s=s or MarkingSettings();c=det.c;mpp=c.mpp
    # Only rows within max_forward_m are examined; all arrays below start at row r0.
    r0=max(0,round((c.forward_max-s.max_forward_m)/mpp));valid=valid[r0:];h,w=valid.shape
    gray=det.gray[r0:];smooth=cv2.blur(gray,(3,3));x=c.forward_max-(r0+np.arange(h))*mpp
    if learned is not None:learned=learned[r0:]
    if getattr(det,'light',None) is not None:light,sat=det.light[r0:],det.saturation[r0:]
    else:_,light,sat=cv2.split(cv2.cvtColor(bev[r0:],cv2.COLOR_BGR2HLS))
    yellow_px=det.yellow[r0:]
    # White paint only: yellow is lane lines and box junction grids here, and a sunny
    # asphalt patch passes the colour test, so paint must also stand out locally.
    k=2*round(s.background_m/mpp/2)+1
    inside_valid=(valid>0).astype(np.float32)
    background=cv2.blur(gray*inside_valid,(k,k))/np.maximum(cv2.blur(inside_valid,(k,k)),1e-3)  # valid pixels only
    bright=gray-background>=s.contrast_min
    white=(sat<120)&(light>c.carla_white_light_min)&~yellow_px&(cv2.erode(valid,np.ones((3,9),np.uint8))>0)
    edge=lane_edges(c,boundaries,x,s.default_half_lane_m)
    lo=np.clip(np.rint((c.lateral_max-edge['left'])/mpp).astype(int),0,w-1)
    hi=np.clip(np.rint((c.lateral_max-edge['right'])/mpp).astype(int),0,w-1)
    cols=np.arange(w)[None,:];margin=round(s.lane_margin_m/mpp)
    inside=(cols>=lo[:,None])&(cols<=hi[:,None]);inner=(cols>=lo[:,None]+margin)&(cols<=hi[:,None]-margin)
    near=x[:,None]<=s.max_forward_m
    out=[]
    # Transverse bars: brighter than 0.45 m before and after, across most of the lane.
    step=max(2,round(.45/mpp))
    bar=(gray-np.maximum(shifted(smooth,-step,0),shifted(smooth,step,0))>=s.stop_contrast_min)&white
    bar=cv2.morphologyEx(bar.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((1,round(.3/mpp)),np.uint8))>0
    width=np.maximum(hi-lo,1);cover=(bar&inside).sum(1)/width
    rows=np.flatnonzero((cover>=s.stop_cover)&(width*mpp>=2.)&(x<=s.max_forward_m))
    groups=np.split(rows,np.flatnonzero(np.diff(rows)>max(2,round(.25/mpp)))+1) if len(rows) else []
    bars=[g for g in groups if len(g) and s.stop_thickness_m[0]<=len(g)*mpp<=s.stop_thickness_m[1]]
    centers=[int(np.median(g)) for g in bars];n=s.grid_bars
    grid=any((centers[j+n-1]-centers[j])*mpp<=s.grid_span_m for j in range(len(centers)-n+1))
    box_rows=getattr(det,'grid_mask',None)
    taken=box_rows[r0:]>0 if box_rows is not None else np.zeros((h,w),bool)  # the detector's box junction rows
    yellow=cv2.dilate(yellow_px.astype(np.uint8),np.ones((3,3),np.uint8))>0
    for g in bars:
        taken[max(0,g[0]-2):g[-1]+3]=True
        if grid:continue  # a box junction's repeated bars are not stop lines
        band=inside[g[0]:g[-1]+1]
        # Antialiased edges of a yellow bar pass the white test; the bar itself does not.
        if np.mean(yellow[g[0]:g[-1]+1][band])>s.stop_yellow_max:continue
        r=int(np.median(g))
        out.append(dict(kind='stop_line',x=float(x[r]),left=float(edge['left'][r]),right=float(edge['right'][r]),thickness=round(len(g)*mpp,2)))
    paint=cv2.morphologyEx((white&bright&near).astype(np.uint8),cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    # Crosswalk: wide longitudinal bars side by side across the road.  The learned
    # mask is not used here: YOLOPv2 paints zebra bars as lane lines.
    bridged=cv2.morphologyEx(paint,cv2.MORPH_CLOSE,np.ones((round(.3/mpp),1),np.uint8))  # cracks across a bar
    count,labels,stats,_=cv2.connectedComponentsWithStats(bridged)
    zebra=[i for i in range(1,count) if stats[i,3]*mpp>=s.crosswalk_bar_length_m and s.crosswalk_bar_width_m[0]<=stats[i,2]*mpp<=s.crosswalk_bar_width_m[1]]
    zebra.sort(key=lambda i:stats[i,0]);groups=[]
    for i in zebra:
        top,bottom=stats[i,1],stats[i,1]+stats[i,3]
        g=groups[-1] if groups else None
        if g and stats[i,0]-g['right']<=s.crosswalk_gap_m/mpp and min(bottom,g['bottom'])-max(top,g['top'])>=.5*stats[i,3]:
            g['bars'].append(i);g['right']=stats[i,0]+stats[i,2];g['top']=min(g['top'],top);g['bottom']=max(g['bottom'],bottom)
        else:groups.append(dict(bars=[i],right=stats[i,0]+stats[i,2],top=top,bottom=bottom,left=stats[i,0]))
    for g in groups:
        if len(g['bars'])<s.crosswalk_bars:continue
        taken[g['top']:g['bottom']+1]=True
        out.append(dict(kind='crosswalk',x0=float(x[min(h-1,g['bottom'])]),x1=float(x[g['top']]),
                        y0=float(c.lateral_max-g['right']*mpp),y1=float(c.lateral_max-g['left']*mpp),bars=len(g['bars'])))
    # Other paint inside the lane that is not a lane line (arrows, text).
    cand=(paint>0)&inner&~taken&(x[:,None]<=s.marking_max_forward_m)
    if learned is not None:cand&=~(cv2.dilate(learned,np.ones((3,2*round(.25/mpp)+1),np.uint8))>0)
    k=round(.6/mpp);count,labels=cv2.connectedComponents(cv2.dilate(cand.astype(np.uint8),np.ones((k,k),np.uint8)))
    for i in range(1,count):
        part=cand&(labels==i);area=float(part.sum())*mpp*mpp
        if area<s.marking_area_m2[0]:continue
        rr,cc=np.nonzero(part);xs=x[rr];ys=c.lateral_max-cc*mpp
        if np.ptp(xs)>s.marking_max_length_m or area>s.marking_area_m2[1]:continue  # kerb, island, light patch
        sub=part[rr.min():rr.max()+1,cc.min():cc.max()+1].astype(np.uint8)
        main=np.bincount(cv2.connectedComponents(sub)[1].ravel())[1:].max()/len(rr)
        hull=cv2.contourArea(cv2.convexHull(np.column_stack([cc,rr]).astype(np.int32)))+1
        core=cv2.erode(part.astype(np.uint8),np.ones((3,3),np.uint8))>0
        if main<s.marking_main_piece or len(rr)/hull<s.marking_solidity or not core.any() or gray[core].std()>s.marking_gray_std:continue
        out.append(dict(kind='marking',x0=float(xs.min()),x1=float(xs.max()),y0=float(ys.min()),y1=float(ys.max()),area=round(area,2)))
    return out


def _world(pose,p):
    c,s=math.cos(pose[2]),math.sin(pose[2]);return np.array([pose[0]+c*p[0]-s*p[1],pose[1]+s*p[0]+c*p[1]])


def _base(pose,q):
    c,s=math.cos(pose[2]),math.sin(pose[2]);d=np.asarray(q)-pose[:2];return np.array([c*d[0]+s*d[1],-s*d[0]+c*d[1]])


class MarkingTracker:
    """M-of-N confirmation of per-frame candidates, matched in world coordinates.

    Without a pose the tracks are cleared: a marking cannot be kept in place."""

    def __init__(self,settings=None):
        self.s=settings or MarkingSettings();self.tracks=[];self.next_id=1

    def reset(self):self.tracks=[]

    @staticmethod
    def corners(m):
        if m['kind']=='stop_line':return np.array([[m['x'],m['left']],[m['x'],m['right']]])
        return np.array([[m['x0'],m['y0']],[m['x0'],m['y1']],[m['x1'],m['y1']],[m['x1'],m['y0']]])

    def update(self,pose,candidates):
        s=self.s
        if pose is None:self.reset();return []
        pose=np.asarray(pose,float)
        world=[(m,np.array([_world(pose,p) for p in self.corners(m)])) for m in candidates]
        used=set()
        for track in self.tracks:
            best=None
            for j,(m,pts) in enumerate(world):
                if j in used or m['kind']!=track['kind']:continue
                d=float(np.linalg.norm(pts.mean(0)-track['points'].mean(0)))
                if d<=s.match_m and (best is None or d<best[0]):best=(d,j)
            hit=best is not None
            track['history']=(track['history']+[hit])[-s.confirm_window:]
            if hit:
                j=best[1];used.add(j);m,pts=world[j]
                track['points']=.6*track['points']+.4*pts if len(pts)==len(track['points']) else pts
                track['misses']=0;track['data']=m
            else:track['misses']+=1
            if sum(track['history'])>=s.confirm_hits:track['confirmed']=True
        self.tracks=[t for t in self.tracks if t['misses']<(s.miss_frames if t['confirmed'] else s.confirm_window)]
        for j,(m,pts) in enumerate(world):
            if j not in used:
                self.tracks.append(dict(id=self.next_id,kind=m['kind'],points=pts,history=[True],misses=0,confirmed=s.confirm_hits<=1,data=m))
                self.next_id+=1
        out=[]
        for t in self.tracks:
            if not t['confirmed']:continue
            base=np.array([_base(pose,q) for q in t['points']])
            if base[:,0].max()<0:continue  # passed under the vehicle
            out.append(dict(id=t['id'],kind=t['kind'],points=base,predicted=t['misses']>0))
        return out
