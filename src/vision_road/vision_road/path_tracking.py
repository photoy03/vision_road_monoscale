"""Bounded continuity bonus and switch hysteresis for observed path candidates."""
from dataclasses import dataclass
import math
import numpy as np


def path_points(path):
    points=np.vstack([node['lane'].points for node in path])
    return points[np.argsort(points[:,0])]


def continuity_error(points,prior):
    """Compare only common observed extent; do not invent paint across a gap."""
    lo=max(points[0,0],prior[0,0],4.);hi=min(points[-1,0],prior[-1,0],22.)
    if hi-lo<.6:return math.inf
    sample=np.linspace(lo,min(hi,lo+10.),12)
    difference=np.abs(np.interp(sample,points[:,0],points[:,1])-np.interp(sample,prior[:,0],prior[:,1]))
    middle=np.partition(difference,(5,6))
    return float((middle[5]+middle[6])*.5)


@dataclass
class Challenge:
    points: np.ndarray
    frame: int
    count: int


class PathTracking:
    def __init__(self,cfg):self.c=cfg;self.reset()
    def reset(self):self.challenges={};self.decisions={}
    def warp(self,matrix):
        if matrix is None:self.challenges.clear();return
        c=self.c
        for challenge in self.challenges.values():
            p=challenge.points
            pixels=np.column_stack([(-p[:,1]-c.lateral_min)/c.mpp,(c.forward_max-p[:,0])/c.mpp])
            pixels=pixels@matrix[:,:2].T+matrix[:,2]
            points=np.column_stack([c.forward_max-pixels[:,1]*c.mpp,-(c.lateral_min+pixels[:,0]*c.mpp)])
            challenge.points=points[np.argsort(points[:,0])]
    def select(self,side,candidates,prior,frame):
        if not candidates:
            self.challenges.pop(side,None);self.decisions[side]=dict(action='no_observation');return None
        if prior is None or not self.c.carla_tracking_enabled:
            self.challenges.pop(side,None);self.decisions[side]=dict(action='untracked');return max(candidates,key=lambda a:a[0])[:2]
        reference,freshness=prior;limit=self.c.carla_tracking_match_m;ranked=[]
        leader=max(candidates,key=lambda a:a[0]);score,path=leader[:2]
        points=leader[2] if len(leader)>2 else path_points(path)
        error=continuity_error(points,reference)
        if error<=limit:
            bonus=self.c.carla_tracking_keep_bonus*freshness*max(0.,1.-error/limit)
            others=[a[0] for a in candidates if a is not leader]
            # Exact upper bound: a lower raw score cannot win even with its
            # maximum possible continuity bonus. Avoid comparing every prefix.
            if not others or score+bonus>max(others)+self.c.carla_tracking_keep_bonus*freshness:
                self.challenges.pop(side,None)
                self.decisions[side]=dict(action='keep',error_m=error,bonus=bonus)
                return score+bonus,path
        for candidate in candidates:
            score,path=candidate[:2]
            points=candidate[2] if len(candidate)>2 else path_points(path)
            error=continuity_error(points,reference)
            matched=error<=limit
            bonus=self.c.carla_tracking_keep_bonus*freshness*max(0.,1.-error/limit) if matched else 0.
            ranked.append((score+bonus,score,path,points,error,matched,bonus))
        best=max(ranked,key=lambda a:a[0]);incumbents=[item for item in ranked if item[5]]
        incumbent=max(incumbents,key=lambda a:a[0]) if incumbents else None
        if best[5]:
            self.challenges.pop(side,None);self.decisions[side]=dict(action='keep',error_m=best[4],bonus=best[6]);return best[0],best[2]
        if incumbent is not None and best[0]<incumbent[0]+self.c.carla_tracking_switch_margin:
            self.challenges.pop(side,None);self.decisions[side]=dict(action='keep_margin',error_m=incumbent[4],bonus=incumbent[6]);return incumbent[0],incumbent[2]
        # A new visible dash may have no common extent with the previous one.
        # Missing overlap is not evidence of a lateral switch.
        if not math.isfinite(best[4]):
            self.challenges.pop(side,None);self.decisions[side]=dict(action='no_common_extent');return best[0],best[2]
        old=self.challenges.get(side)
        same=old is not None and continuity_error(best[3],old.points)<=limit
        count=old.count+(frame!=old.frame) if same and frame-old.frame<=1 else 1
        self.challenges[side]=Challenge(best[3].copy(),frame,count)
        if count>=self.c.carla_tracking_switch_frames:
            self.decisions[side]=dict(action='switch',confirmed_frames=count);return best[0],best[2]
        self.decisions[side]=dict(action='keep_pending' if incumbent is not None else 'pending_without_paint',confirmed_frames=count)
        return (incumbent[0],incumbent[2]) if incumbent is not None else None
