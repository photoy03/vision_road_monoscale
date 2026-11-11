"""Motion-compensated lane history, with short, explicitly labelled gap prediction."""
from dataclasses import dataclass, replace
import math
import cv2
import numpy as np
from .core import Detector, Lane, to_world
from .marking_style import classify_lane
try:
    from . import _native
except ImportError:
    _native=None


def to_base(points, pose):
    x,y,yaw=pose; c,s=math.cos(yaw),math.sin(yaw)
    return (points-[x,y]) @ np.array([[c,-s],[s,c]])


@dataclass
class Track:
    lane: Lane
    last_observed: float
    misses: int = 0
    travel: float = 0.0
    style_candidate: str = 'unknown'
    style_streak: int = 0
    style_time: float = 0.0


class TemporalDetector:
    def __init__(self,cfg):
        if cfg.style_history_frames<1 or cfg.style_history_sec<=0 or cfg.style_hold_sec<0:
            raise ValueError('Invalid style history limits')
        self.c=cfg; self.detector=Detector(cfg); self.reset()

    def reset(self):
        self.tracks={}; self.last_pose=None; self.last_time=None; self.gray=None; self.valid=None
        self.registration='none'; self.measured_count=0; self.predicted_count=0
        self.paint_history=[];self.style_binary=None
        self.next_fragment_id=0
        self.boundaries={};self.boundary_lanes=[]
        self.detector.previous.clear(); self.detector.hits.clear()
        if hasattr(self.detector,'forget'):self.detector.forget()

    def pixel(self, points):
        c=self.c
        return np.column_stack([(-points[:,1]-c.lateral_min)/c.mpp,(c.forward_max-points[:,0])/c.mpp])

    def metric(self, pixels):
        c=self.c
        return np.column_stack([c.forward_max-pixels[:,1]*c.mpp,-(c.lateral_min+pixels[:,0]*c.mpp)])

    def _registration(self,bev,valid,pose,t):
        self.registration='none'; c=self.c
        if self.last_time is None or not (0<t-self.last_time<=c.temporal_max_age_sec): return None,0.
        if pose is not None and self.last_pose is not None:
            distance=float(np.linalg.norm(np.array(pose[:2])-self.last_pose[:2]))
            turn=math.atan2(math.sin(pose[2]-self.last_pose[2]),math.cos(pose[2]-self.last_pose[2]))
            if distance>c.temporal_max_distance_m or abs(turn)>math.radians(10): return None,0.
            base=np.array([[5.,0.],[10.,0.],[5.,1.]])
            current=to_base(to_world(base,self.last_pose),pose)
            self.registration='odometry'
            return cv2.getAffineTransform(self.pixel(base).astype('float32'),self.pixel(current).astype('float32')),distance
        if not c.temporal_use_flow or self.gray is None: return None,0.
        gray=cv2.cvtColor(bev,cv2.COLOR_BGR2GRAY)
        if gray.shape!=self.gray.shape: return None,0.
        mask=cv2.erode(self.valid,np.ones((9,9),'uint8'))
        # Use a corridor around the vehicle; discard invalid borders and far texture.
        h,w=mask.shape; mask[:h//4]=0; mask[:,:w//8]=0; mask[:,7*w//8:]=0
        source=cv2.goodFeaturesToTrack(self.gray,maxCorners=250,qualityLevel=.02,minDistance=5,mask=mask)
        if source is None or len(source)<20: return None,0.
        target,status,_=cv2.calcOpticalFlowPyrLK(self.gray,gray,source,None,winSize=(21,21),maxLevel=3)
        if target is None: return None,0.
        backward,backstatus,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,target,None,winSize=(21,21),maxLevel=3)
        if backward is None: return None,0.
        good=(status.ravel()>0)&(backstatus.ravel()>0)&(np.linalg.norm(source-backward,axis=2).ravel()<1.)
        if np.count_nonzero(good)<20: return None,0.
        a=source[good].reshape(-1,2); b=target[good].reshape(-1,2)
        matrix,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=1000)
        if matrix is None or inliers is None or np.mean(inliers)<.6 or np.sum(inliers)<20: return None,0.
        scale=np.linalg.norm(matrix[:,0]); angle=math.atan2(matrix[1,0],matrix[0,0])
        distance=float(np.linalg.norm(matrix[:,2])*c.mpp)
        if not (.98<scale<1.02) or abs(angle)>math.radians(5) or distance>c.temporal_max_distance_m: return None,0.
        # An affine model is only accepted if it explains held-out residuals too.
        residual=np.linalg.norm(a @ matrix[:,:2].T+matrix[:,2]-b,axis=1)
        if np.median(residual)>1.2: return None,0.
        self.registration='optical_flow'
        return matrix,distance

    def _visible(self,points,valid):
        if self.c.carla_native_enabled and _native is not None:
            c=self.c
            return _native.visible(points,valid,c.mpp,c.forward_max,c.lateral_min,c.min_span_m)
        pixels=self.pixel(points); h,w=valid.shape
        inside=(pixels[:,0]>=0)&(pixels[:,0]<w)&(pixels[:,1]>=0)&(pixels[:,1]<h)
        supported=np.zeros(len(points),bool)
        index=np.flatnonzero(inside)
        supported[index]=valid[pixels[index,1].astype(int),pixels[index,0].astype(int)]>0
        if np.all(supported):
            result=points[np.argsort(points[:,0])]
            return result if np.ptp(result[:,0])>=self.c.min_span_m else None
        edges=np.flatnonzero(np.r_[True,np.diff(supported.astype(int))!=0,True])
        runs=[points[a:b] for a,b in zip(edges[:-1],edges[1:]) if supported[a] and b-a>=3]
        if not runs:return None
        result=max(runs,key=lambda a:np.ptp(a[:,0]))
        result=result[np.argsort(result[:,0])]
        return result if np.ptp(result[:,0])>=self.c.min_span_m else None

    def _fragment_keys(self,observed,records):
        if self.c.carla_native_enabled and _native is not None:
            keys=list(records);old=[records[key].lane for key in keys]
            pairs=_native.match_fragments([l.points for l in observed],[l.points for l in old],
                    [l.side for l in observed],[l.side for l in old],min(.22,self.c.temporal_match_m))
            assigned=set()
            for i,j in pairs:observed[i].track_key=keys[j];assigned.add(i)
        else:
            pairs=[]
            for i,lane in enumerate(observed):
                for key,old in records.items():
                    if lane.side!=old.lane.side:continue
                    a,b=lane.points,old.lane.points
                    lo=max(a[0,0],b[0,0]);hi=min(a[-1,0],b[-1,0])
                    if hi-lo<.20:continue
                    f=np.linspace(lo,hi,8)
                    error=float(np.median(np.abs(np.interp(f,a[:,0],a[:,1])-np.interp(f,b[:,0],b[:,1]))))
                    if error<min(.22,self.c.temporal_match_m):pairs.append((error,i,key))
            assigned=set();used=set()
            for _,i,key in sorted(pairs):
                if i in assigned or key in used:continue
                observed[i].track_key=key;assigned.add(i);used.add(key)
        for i,lane in enumerate(observed):
            if i not in assigned:
                self.next_fragment_id+=1;lane.track_key='carla-'+str(self.next_fragment_id)

    def _boundaries(self, valid, matrix, t):
        """Confirm boundary geometry independently of individual dash identities.

        Only current supported chains are displayed. Gaps between paint pieces
        belong to this display fit, never the paint mask or accumulated map.
        """
        previous=self.boundaries; current={}; self.boundary_lanes=[]
        alpha=self.c.carla_spine_ema_alpha
        for candidate in self.detector.boundary_candidates:
            points=self._visible(candidate.points,valid)
            if points is None:continue
            old=previous.get(candidate.side); hits=1
            if old is not None and self.last_time is not None and 0<t-self.last_time<=max(.4,self.c.temporal_max_age_sec):
                prior,old_hits,observed_time=old
                if matrix is not None:
                    pixels=self.pixel(prior) @ matrix[:,:2].T+matrix[:,2]
                    prior=self.metric(pixels);prior=prior[np.argsort(prior[:,0])]
                lo=max(points[0,0],prior[0,0]);hi=min(points[-1,0],prior[-1,0])
                if hi-lo>=.6:
                    sample=np.linspace(lo,min(hi,lo+8.),20)
                    error=np.median(np.abs(np.interp(sample,points[:,0],points[:,1])-np.interp(sample,prior[:,0],prior[:,1])))
                    if error<=(.30 if matrix is not None else .60):
                        hits=old_hits+1
                        if matrix is not None:
                            # Blend at sample positions, not polynomial terms;
                            # a mismatch above resets the memory instead.
                            overlap=(points[:,0]>=lo)&(points[:,0]<=hi)
                            points[overlap,1]=alpha*points[overlap,1]+(1-alpha)*np.interp(points[overlap,0],prior[:,0],prior[:,1])
            current[candidate.side]=(points,hits,float(t))
            if hits>=self.c.carla_chain_confirm_frames:
                self.boundary_lanes.append(replace(candidate,points=points,hits=hits,source_time=float(t)))
        for side,(prior,hits,observed_time) in previous.items():
            if side in current or t-observed_time>min(.15,self.c.temporal_max_age_sec):continue
            if matrix is None:
                # A missed candidate keeps confirmation memory briefly, but
                # cannot produce a display line or paint without registration.
                current[side]=(prior,hits,observed_time)
                continue
            pixels=self.pixel(prior) @ matrix[:,:2].T+matrix[:,2]
            points=self._visible(self.metric(pixels),valid)
            if points is None:continue
            current[side]=(points,hits,observed_time)
            if hits>=self.c.carla_chain_confirm_frames:
                self.boundary_lanes.append(Lane(side,points,.6,hits=hits,reprojected=True,source_time=observed_time))
        self.boundaries=current

    def process(self,bev,valid,t,pose=None,observation_mask=None,paint_mask=None):
        if self.last_time is not None and t<=self.last_time: self.reset()
        c=self.c
        if c.temporal_enabled:
            matrix,travel=self._registration(bev,valid,pose,t)
        else:
            matrix,travel=None,0.;self.registration='disabled'
        priors={}; records={}
        if c.temporal_enabled and matrix is not None:
            for side,track in self.tracks.items():
                pixels=self.pixel(track.lane.points) @ matrix[:,:2].T+matrix[:,2]
                points=self._visible(self.metric(pixels),valid)
                if points is None: continue
                prior=replace(track.lane,points=points,reprojected=False,source_time=track.last_observed)
                priors[side]=prior
                records[side]=Track(prior,track.last_observed,track.misses,track.travel+travel,track.style_candidate,track.style_streak,track.style_time)
        self.detector.previous.clear(); self.detector.hits.clear()
        binary=self.detector.binary(bev,valid) if paint_mask is None else paint_mask
        observed=self.detector.detect(binary if observation_mask is None else observation_mask,priors=priors)
        # Leave a full BEV pixel margin for subpixel projected endpoints.
        final_valid=cv2.erode(valid,np.ones((3,3),np.uint8))
        grid=getattr(self.detector,'grid_mask',None)
        if grid is not None:final_valid[grid>0]=0
        clipped=[]
        for lane in observed:
            points=self._visible(lane.points,final_valid)
            if points is not None:
                lane.points=points;clipped.append(lane)
        observed=clipped
        chain_filter=(getattr(self.detector,'fragment_tracks',False) and c.carla_chain_filter_enabled)
        if chain_filter:self._boundaries(final_valid,matrix,t)
        if getattr(self.detector,'fragment_tracks',False):self._fragment_keys(observed,records)
        style_binary=getattr(self.detector,'style_mask',binary) if paint_mask is None else paint_mask
        style_valid=getattr(self.detector,'style_valid',valid) if paint_mask is None else valid
        if c.style_temporal_enabled:
            history=[]
            if matrix is not None:
                for stamp,old_mask in self.paint_history:
                    if 0<t-stamp<=c.style_history_sec:
                        moved=cv2.warpAffine(old_mask,matrix,(valid.shape[1],valid.shape[0]),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
                        history.append((stamp,moved))
            history.append((float(t),style_binary.copy()))
            self.paint_history=history[-c.style_history_frames:]
            if len(self.paint_history)>=2:
                votes=np.sum([m>0 for _,m in self.paint_history],axis=0)
                style_binary=((votes>=2)&(valid>0)).astype('uint8')*255
        self.style_binary=style_binary
        result=[]; tracks={};style_cache={}
        for lane in observed:
            key=lane.track_key or lane.side
            old=records.get(key)
            matched=False
            if old is not None:
                a=lane.points; b=old.lane.points
                lo=max(a[0,0],b[0,0]); hi=min(a[-1,0],b[-1,0])
                sample=np.linspace(lo,hi,12) if hi>lo else []
                error=np.median(np.abs(np.interp(sample,a[:,0],a[:,1])-np.interp(sample,b[:,0],b[:,1]))) if len(sample) else math.inf
                if error<c.temporal_match_m:
                    matched=True
                    lane.hits=min(65535,old.lane.hits+1)
                    overlap=(a[:,0]>=lo)&(a[:,0]<=hi)
                    weight=.95 if getattr(self.detector,'fragment_tracks',False) else .75
                    lane.points[overlap,1]=weight*a[overlap,1]+(1-weight)*np.interp(a[overlap,0],b[:,0],b[:,1])
            lane.source_time=float(t)
            style_lane=getattr(self.detector,'style_lanes',{}).get(lane.side,lane)
            cache_key=lane.side if getattr(self.detector,'fragment_tracks',False) else key
            if cache_key not in style_cache:style_cache[cache_key]=classify_lane(style_lane,style_binary,style_valid,c)
            evidence=style_cache[cache_key]
            streak=old.style_streak+1 if matched and old.style_candidate==evidence.kind else 1
            if evidence.kind=='unknown': streak=0
            required=c.style_confirm_frames if c.temporal_enabled else 1
            lane.style=evidence.kind if streak>=required else 'unknown'
            lane.style_confidence=evidence.confidence if lane.style!='unknown' else 0.
            style_time=float(t) if lane.style!='unknown' else 0.
            if (lane.style=='unknown' and matched and old.lane.style!='unknown'
                    and 0<t-old.style_time<=c.style_hold_sec):
                lane.style=old.lane.style
                lane.style_confidence=old.lane.style_confidence*math.exp(-(t-self.last_time)/max(c.style_hold_sec,.01))
                style_time=old.style_time
            lane.style_source_time=style_time
            lane.style_span_m=evidence.span_m; lane.paint_fraction=evidence.paint_fraction; lane.dash_count=evidence.dash_count
            points=self._visible(lane.points,final_valid)
            if points is None:continue
            lane.points=points
            tracks[key]=Track(lane,float(t),style_candidate=evidence.kind,style_streak=streak,style_time=style_time); result.append(lane)
        if getattr(self.detector,'fragment_tracks',False):
            # Style evidence is per side here, so a fragment seen for the first
            # time takes the style its side already has (confirmed or held),
            # unless this frame's evidence for the side says otherwise.
            for track in tracks.values():
                lane=track.lane
                if lane.style!='unknown':continue
                peers=[l for l in result if l.side==lane.side and l.style!='unknown' and track.style_candidate in (l.style,'unknown')]
                if peers:
                    peer=max(peers,key=lambda l:l.style_source_time)
                    lane.style=peer.style;lane.style_confidence=peer.style_confidence
                    lane.style_source_time=track.style_time=peer.style_source_time
        for side,old in records.items():
            if side in tracks: continue
            age=t-old.last_observed; misses=old.misses+1
            paint_confirmed=(not getattr(self.detector,'requires_paint_confirmation_for_prediction',False)
                             or old.lane.style!='unknown')
            if (paint_confirmed and old.lane.hits>=getattr(self.detector,'minimum_prediction_hits',2)
                    and old.lane.confidence>=getattr(self.detector,'minimum_prediction_confidence',0.)
                    and misses<=c.temporal_max_frames and age<=c.temporal_max_age_sec
                    and old.travel<=c.temporal_max_distance_m):
                lane=replace(old.lane,confidence=old.lane.confidence*math.exp(-(t-self.last_time)/.20),reprojected=True,source_time=old.last_observed,style_confidence=old.lane.style_confidence*math.exp(-(t-self.last_time)/.20))
                points=self._visible(lane.points,final_valid)
                if points is None:continue
                lane.points=points
                result.append(lane); tracks[side]=Track(lane,old.last_observed,misses,old.travel,old.style_candidate,old.style_streak,old.style_time)
        if chain_filter:
            # Keep tentative observations internally, but publish only paint
            # inside a temporally confirmed boundary corridor.
            confirmed={lane.side:lane for lane in self.boundary_lanes}
            corridors={lane.side:getattr(lane,'corridor_m',c.carla_chain_band_m) for lane in self.detector.boundary_candidates}
            filtered=[]
            for lane in result:
                boundary=confirmed.get(lane.side)
                if boundary is None:continue
                p=lane.points;b=boundary.points
                inside=(p[:,0]>=b[0,0])&(p[:,0]<=b[-1,0])
                if np.count_nonzero(inside)<3:continue
                error=np.median(np.abs(p[inside,1]-np.interp(p[inside,0],b[:,0],b[:,1])))
                band=corridors.get(lane.side,c.carla_chain_band_m)
                if error<=band:
                    filtered.append(lane)
            result=filtered
        self.tracks=tracks; self.last_pose=pose; self.last_time=float(t)
        self.gray=cv2.cvtColor(bev,cv2.COLOR_BGR2GRAY); self.valid=valid.copy()
        self.measured_count=sum(not l.reprojected for l in result); self.predicted_count=sum(l.reprojected for l in result)
        return result,binary
