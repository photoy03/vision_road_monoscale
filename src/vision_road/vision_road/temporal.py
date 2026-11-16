"""Motion-compensated lane history, with short, explicitly labelled gap prediction."""
from dataclasses import dataclass, replace
import math
import cv2
import numpy as np
from numpy.core.multiarray import interp  # np.interp without its Python wrapper (hundreds of calls per frame)
from .core import Detector, Lane, to_world, median
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

    def _accumulate_style(self,binary,valid,lanes,t):
        """Dashed from paint gathered along the road over the last 20 m travelled.

        In a curve one or two dashes are visible at a time, too little for the
        per-frame profile.  Paint and gaps along each observed boundary (3-20 m) are
        kept on a 0.1 m road coordinate (odometer + forward distance).  With at
        least style_accumulate_m of continuous observation and two or more 2-12 m
        gaps between marks, an UNKNOWN side becomes DASHED.  It never says SOLID,
        and a frame without paint on that side changes nothing."""
        c=self.c;cell=.1;s0=self.odometer;height,width=binary.shape;w=max(1,round(.12/c.mpp))
        for side in ('left','right'):
            cells=self.style_cells.setdefault(side,{})
            bound=next((b for b in self.boundary_lanes if b.side==side and not b.reprojected),None)
            if bound is None:continue
            p=bound.points[np.argsort(bound.points[:,0])];x=np.arange(max(c.forward_min,p[0,0]),min(20.,p[-1,0]),cell)
            if len(x)<2:continue
            y=interp(x,p[:,0],p[:,1]);cols=np.rint((-y-c.lateral_min)/c.mpp).astype(int);rows=np.rint((c.forward_max-x)/c.mpp).astype(int)
            inside=(rows>=0)&(rows<height)&(cols>=w)&(cols<width-w)
            rr=np.clip(rows,0,height-1);cc=np.clip(cols[:,None]+np.arange(-w,w+1),0,width-1)
            seen=inside&np.all(valid[rr[:,None],cc]>0,axis=1);paint=inside&np.any(binary[rr[:,None],cc]>0,axis=1)
            if not paint.any():continue
            for k,pnt in zip(np.rint((s0+x[seen])/cell).astype(int).tolist(),paint[seen].tolist()):cells[k]=cells.get(k,False) or pnt
            first=int((s0-20.)/cell)
            for k in [k for k in cells if k<first]:del cells[k]
            keys=sorted(cells);runs=[[keys[0],keys[0]]] if keys else []
            for k in keys[1:]:
                if k-runs[-1][1]<=3:runs[-1][1]=k
                else:runs.append([k,k])
            a,b=max(runs,key=lambda r:r[1]-r[0])
            if (b-a)*cell<c.style_accumulate_m:continue
            signal=np.array([cells.get(k,False) for k in range(a,b+1)]);edges=np.flatnonzero(np.diff(np.r_[0,signal.astype(int),0]))
            marks=[(lo,hi) for lo,hi in zip(edges[::2],edges[1::2]) if (hi-lo)*cell>=c.style_min_dash_m]
            gaps=sum(1 for (_,h0),(l1,_) in zip(marks[:-1],marks[1:]) if 2.<=(l1-h0)*cell<=12.)
            if gaps<2:continue
            for lane in lanes:
                if lane.side==side and lane.style=='unknown' and not lane.reprojected:
                    lane.style='dashed';lane.style_confidence=.7;lane.style_source_time=float(t)

    def reset(self):
        self.odometer=0.;self.odom_pose=None;self.style_cells={}
        self.tracks={}; self.last_pose=None; self.last_time=None; self.gray=None; self.valid=None
        self.registration='none'; self.measured_count=0; self.predicted_count=0
        self.paint_history=[];self.style_binary=None
        self.next_fragment_id=0
        self.boundaries={};self.boundary_lanes=[];self.confirmed_tracking={}
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
        if median(residual)>1.2: return None,0.
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
                    error=float(median(np.abs(interp(f,a[:,0],a[:,1])-interp(f,b[:,0],b[:,1]))))
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
                    error=median(np.abs(interp(sample,points[:,0],points[:,1])-interp(sample,prior[:,0],prior[:,1])))
                    if error<=(.30 if matrix is not None else .60):
                        hits=old_hits+1
                        if matrix is not None:
                            # Blend at sample positions, not polynomial terms;
                            # a mismatch above resets the memory instead.
                            overlap=(points[:,0]>=lo)&(points[:,0]<=hi)
                            points[overlap,1]=alpha*points[overlap,1]+(1-alpha)*interp(points[overlap,0],prior[:,0],prior[:,1])
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
        if hasattr(self.detector,'path_tracking'):
            moved={};context={}
            if c.carla_tracking_enabled and matrix is not None:
                for side,(points,observed) in self.confirmed_tracking.items():
                    age=t-observed
                    if not 0<=age<=c.carla_tracking_memory_sec:continue
                    pixels=self.pixel(points)@matrix[:,:2].T+matrix[:,2]
                    visible=self._visible(self.metric(pixels),valid)
                    if visible is None:continue
                    moved[side]=(visible,observed);context[side]=(visible,max(0.,1.-age/c.carla_tracking_memory_sec))
            self.confirmed_tracking=moved
            self.detector.tracking_priors=context
            self.detector.path_tracking.warp(matrix if c.carla_tracking_enabled else None)
        self.detector.previous.clear(); self.detector.hits.clear()
        # Distance travelled, the road coordinate of the accumulated style evidence.
        if pose is not None and self.odom_pose is not None and t>self.odom_pose[0]:
            self.odometer+=math.hypot(pose[0]-self.odom_pose[1],pose[1]-self.odom_pose[2])
        self.odom_pose=(t,)+tuple(pose) if pose is not None else None
        self.detector.pose=(self.odometer,)+tuple(pose) if pose is not None else None  # odometer, x, y, yaw
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
        if chain_filter:
            self._boundaries(final_valid,matrix,t)
            if c.carla_tracking_enabled:
                for lane in self.boundary_lanes:
                    if not lane.reprojected:self.confirmed_tracking[lane.side]=(lane.points.copy(),float(t))
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
                votes=np.zeros(valid.shape,np.uint8)
                for _,m in self.paint_history:votes=cv2.add(votes,(m>0).view(np.uint8))
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
                error=median(np.abs(interp(sample,a[:,0],a[:,1])-interp(sample,b[:,0],b[:,1]))) if len(sample) else math.inf
                if error<c.temporal_match_m:
                    matched=True
                    lane.hits=min(65535,old.lane.hits+1)
                    overlap=(a[:,0]>=lo)&(a[:,0]<=hi)
                    weight=.95 if getattr(self.detector,'fragment_tracks',False) else .75
                    lane.points[overlap,1]=weight*a[overlap,1]+(1-weight)*interp(a[overlap,0],b[:,0],b[:,1])
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
        if c.style_accumulate_m>0 and pose is not None:self._accumulate_style(style_binary,style_valid,result,t)
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
                error=median(np.abs(p[inside,1]-interp(p[inside,0],b[:,0],b[:,1])))
                band=corridors.get(lane.side,c.carla_chain_band_m)
                if error<=band:
                    filtered.append(lane)
            result=filtered
        self.tracks=tracks; self.last_pose=pose; self.last_time=float(t)
        self.gray=cv2.cvtColor(bev,cv2.COLOR_BGR2GRAY); self.valid=valid.copy()
        self.measured_count=sum(not l.reprojected for l in result); self.predicted_count=sum(l.reprojected for l in result)
        return result,binary
