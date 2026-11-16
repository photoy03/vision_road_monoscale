"""CARLA paint detection: local paired borders, fragment graph, grid rejection.

No fitted classifier, model weights or simulator map labels are used.
"""
from dataclasses import replace
import math
import statistics
import cv2
import numpy as np
from .core import Detector, Lane
from .path_tracking import PathTracking,continuity_error,path_points
try:
    from . import _native
except ImportError:
    _native=None


def centre_lines(mask,max_width):
    """Per row, the middle 2 pixels of every horizontal run no wider than max_width pixels."""
    h,w=mask.shape;left=np.zeros((h,w),np.int32);right=np.zeros((h,w),np.int32);m=mask.astype(np.int32)
    for j in range(w):left[:,j]=(left[:,j-1]+1)*m[:,j] if j else m[:,j]
    for j in range(w-1,-1,-1):right[:,j]=(right[:,j+1]+1)*m[:,j] if j<w-1 else m[:,j]
    width=left+right-1
    return mask&(np.abs(left-right)<=1)&(width<=max_width)


def shifted(image, offset, axis=1):
    """Replicated boundaries, unlike wrapping a row to the opposite image edge."""
    indices=np.clip(np.arange(image.shape[axis])+offset,0,image.shape[axis]-1)
    return np.take(image,indices,axis=axis)


def grouped_median(points):
    """Exact row medians without one Python/NumPy reduction per BEV row."""
    forward,inverse,counts=np.unique(points[:,0],return_inverse=True,return_counts=True)
    order=np.lexsort((points[:,1],inverse));values=points[order,1]
    starts=np.r_[0,np.cumsum(counts[:-1])]
    center=(values[starts+(counts-1)//2]+values[starts+counts//2])*.5
    return forward,center


class CarlaDetector(Detector):
    fragment_tracks=True
    requires_paint_confirmation_for_prediction=False
    minimum_prediction_hits=3
    minimum_prediction_confidence=.55

    def __init__(self,cfg):
        if cfg.carla_stage not in (1,2,3) or cfg.carla_ridge_min<=0 or cfg.carla_relative_min<=0:
            raise ValueError('Invalid CARLA feature parameters')
        super().__init__(replace(cfg,lane_width_min=-12.,lane_width_max=12.))
        if cfg.carla_chain_confirm_frames<1 or cfg.carla_chain_band_m<=0 or not 0<=cfg.carla_link_curvature_max<=.2 or not 0<cfg.carla_spine_ema_alpha<=1:
            raise ValueError('Invalid CARLA chain filter limits')
        if cfg.carla_learned_gate not in ('off','remove','penalize','replace') or min(cfg.carla_learned_gate_m,cfg.carla_learned_gate_along_m,cfg.carla_learned_gate_far_m)<0 or not 0<=cfg.carla_learned_penalty<=1:
            raise ValueError('Invalid learned gate settings')
        if cfg.carla_tracking_match_m<=0 or cfg.carla_tracking_memory_sec<=0 or cfg.carla_tracking_switch_frames<1 or min(cfg.carla_tracking_keep_bonus,cfg.carla_tracking_switch_margin)<0:
            raise ValueError('Invalid path tracking settings')
        self.path_tracking=PathTracking(cfg);self.tracking_priors={}
        self.diagnostics={}
        self.boundary_candidates=[]
        self.lane_width=3.3  # m; running estimate from frames with both boundaries
        self.last_spines={}  # previous frame boundaries, uncompensated: about 0.3 m of travel at 30 Hz
        self.strand_gap={};self.frame_index=0  # side -> (double line spacing m, frame), kept 30 frames
        self.strand_memory={}  # side -> [(odometer, world centre points)] of seen pairs
        self.shared={}  # short side -> (long side, paint start, paint end) when share_shape gave it the long side's shape
        self.pose=None  # (odometer m, x, y, yaw) of this frame, set by TemporalDetector when odometry is available
        self.learned=None  # BEV learned lane mask for the current frame, set by the caller (optional)
        self.learned_outside=None;self.replaced_paint=None

    def binary(self,bev,valid):
        c=self.c
        gray=cv2.cvtColor(bev,cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray=cv2.GaussianBlur(gray,(3,3),.55);self.gray=gray
        h,l,s=cv2.split(cv2.cvtColor(bev,cv2.COLOR_BGR2HLS));self.light=l;self.saturation=s  # reused by road_markings
        b,g,r=cv2.split(bev.astype(np.float32))
        white=(s<120)&(l>c.carla_white_light_min)
        self.yellow=(h>=16)&(h<=42)&(s>20)&((g-b)>18)&((g-b)>.09*np.maximum(g,1))&((g-b)>1.25*(r-g))
        if c.carla_yellow_relative_min>0:
            # Faint yellow (a pale map, or washed out by light): blue below red and green, measured against
            # the road around it, so a uniformly warm or bluish road is not yellow.
            yellowness=(r+g)/2-b;size=2*round(c.carla_yellow_relative_window_m/c.mpp/2)+1
            road=cv2.medianBlur(np.clip(yellowness+128,0,255).astype(np.uint8),size).astype(np.float32)-128
            self.yellow|=(yellowness>=c.carla_yellow_relative_min)&(yellowness-road>=c.carla_yellow_relative_min)
        color=white|self.yellow
        response=np.zeros_like(gray)
        # Average both flanks at multiple paint widths. Preserve weak ridges;
        # subsequent geometry uses this same response rather than a second
        # incompatible absolute-white threshold.
        smooth=cv2.blur(gray,(3,3))
        for width in (.10,.16,.24,.34):
            radius=max(2,round((width/2+.07)/c.mpp))
            left=shifted(smooth,-radius);right=shifted(smooth,radius)
            ridge=gray-np.maximum(left,right)
            background=np.maximum(np.maximum(left,right),25.)
            eligible=(ridge>=c.carla_ridge_min)&(ridge/background>=c.carla_relative_min)
            response=np.maximum(response,np.where(eligible,ridge,0))
        safe=cv2.erode(valid,np.ones((3,9),np.uint8))>0;self.valid_safe=safe.astype(np.uint8)
        raw=((response>0)&color&safe).astype(np.uint8)*255
        self.response=response;self.raw_mask=raw.copy()
        self.yellow_mask=((self.yellow)&(raw>0)).astype(np.uint8)*255
        self.grid_mask=np.zeros_like(raw)
        if c.carla_stage>=2 and c.carla_grid_enabled:
            # A box junction has repeated long TRANSVERSE painted bars. A
            # crosswalk's parallel longitudinal rectangles are not this test.
            step=max(2,round(.25/c.mpp))
            horizontal=(gray-np.maximum(shifted(smooth,-step,0),shifted(smooth,step,0))>=4)&color&safe
            long=cv2.morphologyEx(horizontal.astype(np.uint8),cv2.MORPH_OPEN,np.ones((1,max(3,round(1.4/c.mpp))),np.uint8))
            rows=np.flatnonzero(np.sum(long,axis=1)*c.mpp>=1.4)
            groups=np.split(rows,np.flatnonzero(np.diff(rows)>max(2,round(.25/c.mpp)))+1) if len(rows) else []
            centers=[int(np.median(a)) for a in groups if len(a)]
            if len(centers)>=3:
                # Suppress only runs of repeated bars within a local 12 m band.
                for j in range(len(centers)-2):
                    a,b=centers[j],centers[j+2]
                    if 1.5<=(b-a)*c.mpp<=12.:
                        self.grid_mask[max(0,a-round(.5/c.mpp)):min(len(raw),b+round(.5/c.mpp)+1)]=255
            raw[self.grid_mask>0]=0
            # Geometry is filtered once in paint_fragments, after the box mask.
        removed=self.learned_gate(raw)
        self.style_mask=raw.copy();self.style_valid=valid.copy()
        self.valid=valid.copy();self.mask=raw.copy()
        self.diagnostics=dict(raw_pixels=int(np.count_nonzero(self.raw_mask)),retained_pixels=int(np.count_nonzero(raw)),grid_pixels=int(np.count_nonzero(self.grid_mask)),candidates=[])
        if removed is not None:self.diagnostics['learned_outside_pixels']=removed
        return raw

    def learned_gate(self,raw):
        """Paint away from the learned lane mask (crosswalk bars, arrows, text, patches) is
        removed or marked for a score penalty, within carla_learned_gate_far_m only."""
        c=self.c;self.learned_outside=None;self.replaced_paint=None
        if c.carla_learned_gate=='off' or self.learned is None or self.learned.shape!=raw.shape:return None
        if c.carla_learned_gate=='replace':return self.learned_replace(raw)
        kernel=np.ones((2*round(c.carla_learned_gate_along_m/c.mpp)+1,2*round(c.carla_learned_gate_m/c.mpp)+1),np.uint8)
        allow=cv2.dilate(self.learned,kernel)>0
        allow[:max(0,round((c.forward_max-c.carla_learned_gate_far_m)/c.mpp))]=True
        outside=(raw>0)&~allow
        if c.carla_learned_gate=='remove':raw[outside]=0;self.yellow_mask[outside]=0
        else:self.learned_outside=outside
        return int(np.count_nonzero(outside))

    def learned_replace(self,raw):
        """Within carla_learned_gate_far_m the learned mask's centre lines replace the paint.
        Yellow and the solid/dashed evidence still come from the measured paint."""
        c=self.c;far=max(0,round((c.forward_max-c.carla_learned_gate_far_m)/c.mpp))
        paint=raw.copy();lines=centre_lines(self.learned>0,round(.9/c.mpp))
        lines&=self.valid_safe>0
        raw[far:]=np.where(lines[far:],255,0).astype(np.uint8)
        raw[self.grid_mask>0]=0
        near_yellow=cv2.dilate(self.yellow_mask,np.ones((3,2*round(.3/c.mpp)+1),np.uint8))>0
        self.yellow_mask=((raw>0)&near_yellow).astype(np.uint8)*255
        self.replaced_paint=paint
        return int(np.count_nonzero(lines[far:]))

    def fragment_graph(self,mask):
        """Local stripe nodes with width/direction-consistent inter-tile links.

        This is a small deterministic fragment graph, not the published graph
        extractor's full implementation. Dash gaps are never rasterized.
        """
        c=self.c;tile=max(8,round(4./c.mpp));nodes=[]
        for start in range(0,len(mask),tile):
            part=mask[start:start+tile]
            count,labels,stats,_=cv2.connectedComponentsWithStats(part)
            for label in range(1,count):
                ys,xs=np.nonzero(labels==label)
                if len(xs)<4 or np.ptp(ys)*c.mpp<.25:continue
                points=np.column_stack([xs,ys+start]).astype(float)
                values,vectors=np.linalg.eigh(np.cov(points.T));axis=vectors[:,-1]
                if abs(axis[1])<.45 or math.sqrt(max(values[0],0))*c.mpp>.16:continue
                fit=np.polyfit(points[:,1],points[:,0],1)
                nodes.append(dict(x=xs,y=ys+start,fit=fit,lo=float(points[:,1].min()),hi=float(points[:,1].max()),linked=False))
        for j,a in enumerate(nodes):
            for b in nodes[j+1:]:
                upper,lower=(a,b) if a['lo']<b['lo'] else (b,a)
                gap=(lower['lo']-upper['hi'])*c.mpp
                if not (-.2<=gap<=10.):continue
                angle=abs(math.atan(a['fit'][0])-math.atan(b['fit'][0]))
                middle=(upper['hi']+lower['lo'])/2
                lateral=abs((a['fit'][0]-b['fit'][0])*middle+a['fit'][1]-b['fit'][1])*c.mpp
                if angle<math.radians(25) and lateral<.30+.08*max(gap,0):
                    a['linked']=b['linked']=True
        result=np.zeros_like(mask)
        for a in nodes:
            if a['linked'] or (a['hi']-a['lo'])*c.mpp>=.60:
                result[a['y'],a['x']]=255
        return result

    def detect(self,mask,priors=None):
        # No old curve is inserted in the mask. It only helps select among
        # candidates supported by current paint.
        if self.c.carla_stage>=2:return self.paint_fragments(mask)
        c=self.c; original=mask; candidates=[];yellow_band=np.zeros_like(mask)
        for color, color_mask in [('yellow',cv2.bitwise_and(mask,self.yellow_mask)),('white',cv2.bitwise_and(mask,cv2.bitwise_not(self.yellow_mask)))]:
            if color=='white':color_mask=cv2.bitwise_and(color_mask,cv2.bitwise_not(yellow_band))
            self.mask=color_mask
            self.previous.clear(); self.hits.clear()
            for lane in super().detect(color_mask,priors,self.candidate_score):
                if color=='yellow':
                    pixels=np.column_stack([(-lane.points[:,1]-c.lateral_min)/c.mpp,(c.forward_max-lane.points[:,0])/c.mpp])
                    cv2.polylines(yellow_band,[np.rint(pixels).astype(np.int32)],False,255,max(3,round(.45/c.mpp)))
                    if lane.side=='right':continue
                candidates.append(lane)
        self.mask=original
        chosen={}
        for lane in candidates:
            if lane.points[0,0]>12.:continue
            anchor=float(np.interp(np.clip(8.,lane.points[0,0],lane.points[-1,0]),lane.points[:,0],lane.points[:,1]))
            side='left' if anchor>0 else 'right'
            if not .65<=abs(anchor)<=4.8:continue
            if lane.side!=side:continue
            span=float(np.ptp(lane.points[:,0]))
            quality=.45*lane.confidence+.45*min(span/15.,1.)-.08*abs(anchor)
            if side not in chosen or quality>chosen[side][0]:chosen[side]=(quality,lane)
        return [chosen[s][1] for s in ['left','right'] if s in chosen]

    def edge_like(self,y,x):
        """A curb or island border is a brightness step, not a ridge on asphalt.

        Painted lines have similar road on both flanks; a curb lip has island
        or sidewalk on one flank and is barely brighter than that flank. Only
        white candidates are tested: yellow paint often runs beside a curb.
        """
        gray=self.gray;offset=max(2,round(.35/self.c.mpp))
        keep=(x-offset-1>=0)&(x+offset+1<gray.shape[1])
        count=int(np.count_nonzero(keep))
        if count<3:return False
        y=y[keep];x=x[keep]
        left=float(gray[y,x-offset].sum()+gray[y,x-offset-1].sum())/(2*count)
        right=float(gray[y,x+offset].sum()+gray[y,x+offset+1].sum())/(2*count)
        center=float(gray[y,x].sum())/count;asymmetry=abs(left-right)
        return asymmetry>=20. and center-max(left,right)<asymmetry/1.5

    def merge_strands(self,nodes,sample,smooth,yellow,side=None):
        """Observed strands for the paint topics, a marking center for the boundary.

        Paint keeps each strand where it was measured, smoothed only along that
        strand, so row-median jitter goes but a double line stays two lines.
        The returned boundary offset puts the spine on the marking center: the
        middle of a double line, also when only one strand is seen this frame.
        Dash gaps stay open in both.
        """
        nodes=sorted(nodes,key=lambda a:a['lane'].points[0,0])
        centers=np.array([np.mean(a['lane'].points[:,0]) for a in nodes])
        offsets=np.array([float(np.mean(a['lane'].points[:,1]-np.interp(a['lane'].points[:,0],sample,smooth))) for a in nodes])
        target=offsets.copy();double=False;strand=np.zeros(len(nodes),bool)
        voted=self.pair_vote(nodes,sample,smooth,side) if yellow and self.c.carla_strand_pair else None
        if voted is not None:
            target,strand=voted;double=True
        elif yellow and len(nodes)>=2:
            order=np.sort(offsets);gaps=np.diff(order);k=int(np.argmax(gaps))
            split=(order[k]+order[k+1])/2;outer=offsets>=split
            extent=lambda flag:sum(np.ptp(a['lane'].points[:,0]) for a,f in zip(nodes,flag) if f)
            if .10<=gaps[k] and order[-1]-order[0]<=.55 and min(extent(outer),extent(~outer))>=2.:
                mid=(np.median(offsets[outer])+np.median(offsets[~outer]))/2
                target[outer]+=mid-np.median(offsets[outer]);target[~outer]+=mid-np.median(offsets[~outer]);double=True;strand=outer
                gap=float(np.median(offsets[outer])-np.median(offsets[~outer]));old=self.strand_gap.get(side)
                if old is not None and self.frame_index-old[1]<=30:gap=.7*old[0]+.3*gap
                self.strand_gap[side]=(gap,self.frame_index)
        remembered=self.strand_gap.get(side);previous=self.last_spines.get(side)
        # The pair vote replaces this previous-output rule: a wrong previous centre would carry on.
        if yellow and not double and not self.c.carla_strand_pair and remembered is not None and self.frame_index-remembered[1]<=30 and previous is not None:
            # One strand of a recent double line: move it half the remembered
            # spacing toward the previous center, on the side it was seen.
            half=remembered[0]/2
            for i,a in enumerate(nodes):
                pts=a['lane'].points
                if not previous[0][0]<=centers[i]<=previous[0][-1]:continue
                delta=float(np.mean(pts[:,1]-np.interp(pts[:,0],previous[0],previous[1])))
                if half/2<=abs(delta)<=1.5*half:target[i]-=math.copysign(half,delta);double=True
        # Neighbour medians within 3 m; n is a few dozen, so plain Python
        # beats a masked n x n NumPy median here.
        positions=centers.tolist();flags=strand.tolist();targets=target.tolist();measured=offsets.tolist()
        near=[[j for j,g in enumerate(positions) if abs(g-f)<=3.] for f in positions]
        center=[statistics.median(targets[j] for j in row) for row in near]
        observed=[statistics.median(measured[j] for j in row if flags[j]==flags[i]) for i,row in enumerate(near)]
        merged=[]
        for flag in (False,True):
            groups=[]
            for a,value,s in zip(nodes,observed,strand):
                if s!=flag:continue
                lo,hi=a['lane'].points[0,0],a['lane'].points[-1,0]
                if groups and lo<=groups[-1]['hi']+.05:
                    g=groups[-1];g['hi']=max(g['hi'],hi);g['items'].append((a,value))
                else:groups.append(dict(lo=lo,hi=hi,items=[(a,value)]))
            for g in groups:
                items=g['items']
                if len(items)==1:
                    # One fragment already has one point per BEV row.
                    points=items[0][0]['lane'].points;forward=points[:,0];measured=points[:,1];offset=items[0][1]
                else:
                    forward,measured=grouped_median(np.vstack([a['lane'].points for a,_ in items]))
                    mids=np.array([np.mean(a['lane'].points[:,0]) for a,_ in items]);values=np.array([v for _,v in items]);order=np.argsort(mids)
                    offset=np.interp(forward,mids[order],values[order])
                # Each strand stays within half a BEV pixel of its measured paint.
                lateral=np.clip(np.interp(forward,sample,smooth)+offset,measured-.45*self.c.mpp,measured+.45*self.c.mpp)
                lane=replace(items[0][0]['lane'],points=np.column_stack([forward,lateral]),
                             confidence=max(a['lane'].confidence for a,_ in items))
                pixels=(np.concatenate([a['pixels'][0] for a,_ in items]),np.concatenate([a['pixels'][1] for a,_ in items]))
                merged.append(dict(lane=lane,pixels=pixels))
        merged.sort(key=lambda a:a['lane'].points[0,0])
        if voted is not None:
            # Section by section: the 3 m neighbour medians, interpolated along the spine.
            order=np.argsort(centers)
            return np.interp(sample,centers[order],np.asarray(center)[order]),merged
        return float(np.median(center)),merged

    def pair_vote(self,nodes,sample,smooth,side):
        """Centre offsets of a double line seen as one pair of rails a fixed gap apart.

        Each fragment is the outer strand (centre half a gap inward), the inner
        strand (half a gap outward) or, when about two strands wide, both strands
        (centre on it).  Fragments with a partner a gap away are labelled by that
        pair; a weighted line through their centres then labels the single strands,
        so far pairs decide which strand the near paint is.  Returns per-fragment
        centre offsets from the spine and the outer-strand flags, or None when no
        fragment is labelled (the caller falls back to its previous-frame rule)."""
        c=self.c;mpp=c.mpp
        remembered=self.strand_gap.get(side);gap=remembered[0] if remembered is not None else c.carla_strand_gap_m;half=gap/2
        residual=[a['lane'].points[:,1]-np.interp(a['lane'].points[:,0],sample,smooth) for a in nodes]
        forward=[a['lane'].points[:,0] for a in nodes]
        label=[None]*len(nodes);votes=[[] for _ in nodes];gaps=[];lines=[];overlap=0.
        for i,a in enumerate(nodes):
            pts=a['lane'].points;run=float(pts[-1,0]-pts[0,0])
            slope=float(pts[-1,1]-pts[0,1])/run if run>0 else 0.;lines.append((slope,float(pts[0,1])-slope*float(pts[0,0])))
            y,x=a['pixels']
            # Paint pixels per BEV row, across the strand: one strand ~0.13 m, two touching ~0.4 m.
            if len(y) and len(y)/len(forward[i])*mpp/math.sqrt(1+slope*slope)>=half+.13:label[i]=0.
        for i in range(len(nodes)):
            for j in range(i+1,len(nodes)):
                if label[i]==0. or label[j]==0.:continue
                lo=max(forward[i][0],forward[j][0]);hi=min(forward[i][-1],forward[j][-1])
                if hi-lo<1.:continue
                middle=(lo+hi)/2
                if abs(abs((lines[i][0]-lines[j][0])*middle+lines[i][1]-lines[j][1])-gap)>.15:continue
                common=(forward[i]>=lo)&(forward[i]<=hi)
                d=float(np.median(nodes[i]['lane'].points[common,1]-np.interp(forward[i][common],forward[j],nodes[j]['lane'].points[:,1])))
                if abs(abs(d)-gap)>.08:continue
                votes[i].append(1. if d>0 else -1.);votes[j].append(-1. if d>0 else 1.);gaps.append(abs(d));overlap+=hi-lo
        # One short side by side pair or one wide blob is no double line (a curb, a chevron
        # edge): 2 m of pairs and two-strand-wide paint at least.
        if overlap+sum(float(forward[i][-1]-forward[i][0]) for i,l in enumerate(label) if l==0.)<2.:
            votes=[[] for _ in nodes];gaps=[];label=[None]*len(nodes)
        for i,v in enumerate(votes):
            if v and label[i] is None and abs(sum(v))==len(v):label[i]=v[0]
        if gaps:
            measured=float(np.median(gaps))
            self.strand_gap[side]=(measured if remembered is None else .9*remembered[0]+.1*measured,self.frame_index)
        memory=c.carla_strand_memory_m>0
        fixed=self.recall_pairs(nodes,label,votes,half,side) if memory else set()
        if memory:self.store_pairs(nodes,label,votes,half,side)
        if all(l is None for l in label):return None
        for _ in range(3):
            known=[i for i,l in enumerate(label) if l is not None]
            xs=np.concatenate([forward[i] for i in known]);cs=np.concatenate([residual[i]-label[i]*half for i in known])
            degree=1 if np.ptp(xs)>=4. else 0
            weights=np.ones(len(xs))
            for _ in range(3):
                fit=np.polyfit(xs,cs,degree,w=weights)
                weights=np.minimum(1.,.05/np.maximum(np.abs(cs-np.polyval(fit,xs)),.001))
            changed=False
            for i in range(len(nodes)):
                if label[i] is not None and (votes[i] or label[i]==0. or i in fixed):continue
                model=np.polyval(fit,forward[i])
                best=min((-1.,1.),key=lambda s:float(np.mean(np.abs(residual[i]-s*half-model))))
                if label[i]!=best:label[i]=best;changed=True
            if not changed:break
        target=np.array([float(np.mean(residual[i]))-label[i]*half for i in range(len(nodes))])
        self.diagnostics['strand_labels_'+side]=label
        return target,np.array([l==1. for l in label])

    def recall_pairs(self,nodes,label,votes,half,side):
        """Label single strands by where a pair's centre was seen on the road.

        Paint 6 m ahead was 10-15 m ahead a second ago, where both strands are
        far more often visible.  Centres of fragments labelled by a partner are
        kept in odometry coordinates for carla_strand_memory_m of travel; a single
        strand lying half a gap from a remembered centre takes that side.  Only
        observations are stored, never the output.  No odometry: nothing is used.
        Returns the fragments it labelled."""
        pose=self.pose;fixed=set()
        if pose is None:self.strand_memory.pop(side,None);return fixed
        odometer,px,py,yaw=pose;co,si=math.cos(yaw),math.sin(yaw)
        memory=[m for m in self.strand_memory.get(side,[]) if odometer-m[0]<=self.c.carla_strand_memory_m]
        if memory:
            world=np.vstack([m[1] for m in memory]);dx=world[:,0]-px;dy=world[:,1]-py
            ahead=dx*co+dy*si;lateral=-dx*si+dy*co
            # Mean remembered centre per 0.2 m of distance ahead.
            cell=np.floor(ahead/.2).astype(int);keep=(cell>=0)&(cell<round(self.c.forward_max/.2)+1)
            count=np.bincount(cell[keep],minlength=round(self.c.forward_max/.2)+1);total=np.bincount(cell[keep],lateral[keep],minlength=len(count))
            for i,a in enumerate(nodes):
                if votes[i] or label[i]==0.:continue
                pts=a['lane'].points;cells=np.clip(np.floor(pts[:,0]/.2).astype(int),0,len(count)-1);seen=count[cells]>0
                if seen.sum()<3:continue
                # Remembered centre minus paint; the outer strand lies half a gap outside the centre.
                d=float(np.median(total[cells[seen]]/count[cells[seen]]-pts[seen,1]))
                if abs(abs(d)-half)<=half*.6:label[i]=-1. if d>0 else 1.;fixed.add(i)
        return fixed

    def store_pairs(self,nodes,label,votes,half,side):
        """Keep this frame's pair centres (partner or two-strand-wide fragments) in odometry coordinates."""
        pose=self.pose
        if pose is None:return
        odometer,px,py,yaw=pose;co,si=math.cos(yaw),math.sin(yaw)
        memory=[m for m in self.strand_memory.get(side,[]) if odometer-m[0]<=self.c.carla_strand_memory_m]
        seen=[a['lane'].points[::3]-[0.,label[i]*half] for i,a in enumerate(nodes) if label[i] is not None and (votes[i] or label[i]==0.)]
        if seen:
            pts=np.vstack(seen)
            memory.append((odometer,np.column_stack([px+pts[:,0]*co-pts[:,1]*si,py+pts[:,0]*si+pts[:,1]*co])))
        self.strand_memory[side]=memory

    def spine(self,group,forward=None,center=None):
        """Robust quadratic (or, for a long dense chain that needs it, cubic) boundary of one selected chain, with its band."""
        c=self.c
        if forward is None:forward,center=grouped_median(np.vstack([a['lane'].points for a in group]))
        # Fit only the selected chain; nearby branches cannot steer
        # the boundary or create paint inside real dash gaps.
        sample=np.linspace(forward[0],forward[-1],max(3,math.ceil((forward[-1]-forward[0])/.20)+1))
        # A robust quadratic within the selected connected chain avoids
        # local polynomial oscillations across the actual dash gaps.
        origin=float(np.mean(forward));x=forward-origin
        near=np.ones(len(x)) if c.carla_spine_near_m<=0 else np.minimum(1.,(c.carla_spine_near_m/np.maximum(forward,1.))**2)
        def robust(degree,weights,ridge=0.):
            # Least squares in x/10 m (rows scaled by the weight, as np.polyfit's w); ridge pulls the cubic term only.
            t=x/10.;A=np.vander(t,degree+1)
            for _ in range(5):
                w=weights*near
                lhs=A*w[:,None];rhs=center*w
                if ridge>0:lhs=np.vstack([lhs,np.r_[math.sqrt(ridge),np.zeros(degree)]]);rhs=np.r_[rhs,0.]
                coef=np.linalg.lstsq(lhs,rhs,rcond=None)[0]
                residual=np.abs(center-A@coef)
                weights=np.minimum(1.,.20/np.maximum(residual,.001))
            return coef,residual,weights
        coef,residual,weights=robust(2,np.ones(len(x)))
        span=float(forward[-1]-forward[0])
        yellow_path=sum(a['yellow'] for a in group)>len(group)/2
        # A double line (a pair of strands measured on this side within 30 frames) keeps its quadratic:
        # the cubic followed the strand scatter and doubled the near jumps (maploop curve).
        paired=self.strand_gap.get(group[0]['lane'].side) if yellow_path else None
        double=c.carla_spine_cubic_double is False and paired is not None and self.frame_index-paired[1]<=30
        if c.carla_spine_cubic_m>0 and not double and span>=c.carla_spine_cubic_span_m and len(forward)*c.mpp>=c.carla_spine_cubic_cover*span:
            # Straight then curving (a curve entry or exit): a quadratic bends the near straight part
            # and flattens the far curve.  Dense paint only; dash gaps keep the quadratic.  The cubic's
            # pull to 0 eases off smoothly as the quadratic misses more, so a gentle curve whose miss
            # hovers near the threshold does not flip between the two from frame to frame.
            # The miss is measured on 1 m means of the signed residual: a double line's row medians
            # alternate between its strands, which is scatter, not a straight-then-curving shape.
            signed=center-np.vander(x/10.,3)@coef;inlier=weights>=.5;bins=np.floor(forward[inlier]).astype(int)
            count=np.bincount(bins-bins.min());total=np.bincount(bins-bins.min(),weights=signed[inlier])
            miss=float(np.percentile(np.abs(total[count>=3]/count[count>=3]),90)) if (count>=3).any() else 0.
            coef,residual,weights=robust(3,weights,c.carla_spine_cubic_ridge*(c.carla_spine_cubic_m/max(miss,1e-3))**4)
            if hasattr(self,'diagnostics'):self.diagnostics['cubic_'+str(group[0]['lane'].side)]=round(float(coef[0]),4)
        fit=coef/np.power(10.,np.arange(len(coef)-1,-1,-1))  # back to polyval in metres
        band=c.carla_chain_band_m
        if c.carla_path_filter_enabled and yellow_path:band=max(band,.45)
        return sample,np.polyval(fit,sample-origin),band,yellow_path,group

    def parallel_support(self,nodes,spines):
        """Find a missing or short boundary from the other one shifted by a lane.

        Only fragments of real paint inside the shifted band are collected;
        nothing is drawn where no paint was measured.
        """
        span=lambda value:value[0][-1]-value[0][0] if value is not None else 0.
        left,right=spines.get('left'),spines.get('right')
        if left is not None and right is not None:
            lo=max(5.,left[0][0],right[0][0]);hi=min(10.,left[0][-1],right[0][-1])
            if hi-lo>=2.:
                f=np.linspace(lo,hi,11)
                width=float(np.mean(np.interp(f,left[0],left[1])-np.interp(f,right[0],right[1])))
                if 2.6<=width<=4.2:self.lane_width=.8*self.lane_width+.2*width
        for side,other in (('right','left'),('left','right')):
            reference=spines.get(other)
            if reference is None or span(reference)<8. or span(spines.get(side))>=6.:continue
            sample,smooth=reference[0],reference[1]
            # Offset along the normal of the reference so curves stay parallel.
            slope=np.gradient(smooth,sample);norm=np.sqrt(1+slope**2)
            sign=-1. if side=='right' else 1.
            forward=sample-sign*self.lane_width*slope/norm;lateral=smooth+sign*self.lane_width/norm
            order=np.argsort(forward);forward=forward[order];lateral=lateral[order]
            used={id(a) for a in reference[4]}
            band=[]
            for a in nodes:
                pts=a['lane'].points
                if id(a) in used or pts[0,0]<forward[0]-.15 or pts[-1,0]>forward[-1]+.15:continue
                if float(np.median(np.abs(pts[:,1]-np.interp(pts[:,0],forward,lateral))))<=.35:band.append(a)
            if not band:continue
            extent=lambda flag:sum(float(np.ptp(a['lane'].points[:,0])) for a in band if a['yellow']==flag)
            color=extent(True)>extent(False);band=[a for a in band if a['yellow']==color]
            current=spines.get(side)
            paint=sum(float(np.ptp(a['lane'].points[:,0])) for a in band)
            if paint<1.2 or (current is not None and paint<=sum(float(np.ptp(a['lane'].points[:,0])) for a in current[4])):continue
            for a in band:a['lane'].side=side
            replacement=self.spine(band)
            prior=self.tracking_priors.get(side)
            if self.c.carla_tracking_enabled and prior is not None:
                candidate=np.column_stack([replacement[0],replacement[1]])
                reference=prior[0]
                if self.path_tracking.decisions.get(side,{}).get('action')=='switch' and current is not None:
                    reference=np.column_stack([current[0],current[1]])
                error=continuity_error(candidate,reference)
                if math.isfinite(error) and error>self.c.carla_tracking_match_m:continue
            spines[side]=replacement
            self.diagnostics['parallel_'+side]=round(paint,2)

    def share_shape(self,spines):
        """A short boundary (one or two dashes in a curve) takes the long one's shape.

        The long boundary is offset along its normal by the median distance of the
        short side's own paint, which keeps the curves parallel (R±W, not a shift
        of the same quadratic).  Only the lateral position comes from the short side."""
        c=self.c
        if 'left' not in spines or 'right' not in spines:return
        span=lambda v:float(v[0][-1]-v[0][0])
        paint=lambda v:sum(float(np.ptp(a['lane'].points[:,0])) for a in v[4])
        for short,long in (('right','left'),('left','right')):
            S,L=spines[short],spines[long]
            if span(L)<c.carla_shared_long_m or paint(S)>=c.carla_shared_short_m:continue
            sample,smooth=L[0],L[1];slope=np.gradient(smooth,sample);norm=np.sqrt(1+slope**2)
            pts=np.vstack([a['lane'].points for a in S[4]])
            # Signed normal distance of the short paint from the long curve.
            nx,ny=-slope/norm,1./norm
            dense=np.column_stack([sample,smooth])
            j=np.abs(pts[:,0][:,None]-sample[None,:]).argmin(1)
            d=float(np.median((pts[:,0]-dense[j,0])*nx[j]+(pts[:,1]-dense[j,1])*ny[j]))
            forward=sample+d*nx;lateral=smooth+d*ny
            ext=c.carla_shared_extend_m
            keep=(forward>=pts[:,0].min()-ext)&(forward<=pts[:,0].max()+ext)
            if keep.sum()<3:continue
            order=np.argsort(forward[keep])
            spines[short]=(forward[keep][order],lateral[keep][order],S[2],S[3],S[4])
            self.diagnostics['shared_'+short]=round(d,2)
            self.shared[short]=(long,float(pts[:,0].min()),float(pts[:,0].max()))

    def prior_penalty(self,side,path,points=None):
        """Rank down a jump outward from the last boundary or off the lane width."""
        if points is None:points=path_points(path)
        if not points[0,0]-2.<=8.<=points[-1,0]+4.:return 0.
        lateral=float(np.interp(8.,points[:,0],points[:,1]));penalty=0.
        def at8(value):
            if value is None or not value[0][0]<=8.<=value[0][-1]:return None
            return float(np.interp(8.,value[0],value[1]))
        tracked=self.tracking_priors.get(side) if self.c.carla_tracking_enabled else None
        previous=at8((tracked[0][:,0],tracked[0][:,1])) if tracked is not None else at8(self.last_spines.get(side))
        jump=0. if previous is None else abs(lateral-previous) if tracked is not None else (lateral-previous if side=='left' else previous-lateral)
        if previous is not None and jump>.5:penalty+=2.
        opposite_side='right' if side=='left' else 'left'
        opposite_track=self.tracking_priors.get(opposite_side) if self.c.carla_tracking_enabled else None
        opposite=at8((opposite_track[0][:,0],opposite_track[0][:,1])) if opposite_track is not None else at8(self.last_spines.get(opposite_side))
        if opposite is not None and abs(abs(lateral-opposite)-self.lane_width)>.6:penalty+=1.5
        return penalty

    def width_error(self,side,points):
        """How far a candidate's distance to the other side's tracked boundary is from the lane width,
        over its first 4 m that both cover (signed: a candidate across the other side is far off)."""
        other=self.tracking_priors.get('right' if side=='left' else 'left') if self.c.carla_tracking_enabled else None
        if other is None:return None
        ref=other[0];lo=max(points[0,0],ref[0,0]);hi=min(points[-1,0],ref[-1,0],lo+4.)
        if hi<lo:return None
        x=np.linspace(lo,hi,5);own=np.interp(x,points[:,0],points[:,1]);them=np.interp(x,ref[:,0],ref[:,1])
        width=float(np.median(them-own if side=='right' else own-them))
        return abs(width-self.lane_width)

    def forget(self):
        """Drop frame-to-frame memory after a time jump or reset."""
        self.last_spines={};self.lane_width=3.3;self.strand_gap={};self.strand_memory={}
        self.tracking_priors={};self.path_tracking.reset()

    def coherent_paths(self,nodes):
        """DAG maximum-support paths; an attached branch cannot pull the fit.

        A paint fragment is a node. Edges cross real dash gaps, never rasterize
        them, and must agree in color, tangent and lateral prediction.
        """
        c=self.c;chosen={};scored={}
        candidates=sorted(nodes,key=lambda a:a['lane'].points[0,0]);n=len(candidates)
        if not n:return chosen
        first=np.array([a['lane'].points[0,0] for a in candidates]);last=np.array([a['lane'].points[-1,0] for a in candidates])
        length=np.array([float(np.ptp(a['lane'].points[:,0])) for a in candidates])
        if self.learned_outside is not None:
            # Paint the learned mask does not call a lane counts for less in the chain score.
            outside=np.array([float(np.mean(self.learned_outside[a['pixels']])) for a in candidates])
            length=length*(1-c.carla_learned_penalty*outside)
        slope=np.array([a['fit'][0] for a in candidates]);intercept=np.array([a['fit'][1] for a in candidates])
        tangent=np.arctan(slope);lo=np.array([a['lo'] for a in candidates]);hi=np.array([a['hi'] for a in candidates])
        yellow=np.array([a['yellow'] for a in candidates]);anchor=np.abs([a['anchor'] for a in candidates])
        # Links k -> j are side independent: same color, tangent and lateral
        # prediction across a real gap of at most 8 m. Computed once.
        gap=first[None,:]-last[:,None];positive=np.maximum(gap,0.)
        middle=(lo[:,None]+hi[None,:])/2;angle=np.abs(tangent[:,None]-tangent[None,:])
        error=np.abs(slope[:,None]*middle+intercept[:,None]-(slope[None,:]*middle+intercept[None,:]))*c.mpp
        # A straight-line prediction misses a curve by about k*g^2/2 and turns
        # by k*g across a gap g; allow up to the configured curvature.
        bend=c.carla_link_curvature_max
        ok=np.triu(np.ones((n,n),bool),1)&(gap>=-.15)&(gap<=8.)&(yellow[:,None]==yellow[None,:])&(angle<=math.radians(22)+bend*positive)&(error<=.20+.025*positive+.5*bend*positive**2)
        weight=length[None,:]+.4-3.*error-1.5*angle-.06*positive
        links=[[] for _ in range(n)]
        for k,j in zip(*np.nonzero(ok)):links[j].append((int(k),float(weight[k,j])))
        base=length+.4+3.*np.maximum(0.,15.-first)
        # The root fixes the side. A far fragment of a curved line may
        # extrapolate to the other side, so off-corridor ones join by chain
        # only. Beyond 15 m a straight extrapolation is trusted only for a
        # root nearly parallel to the heading.
        rootable=(anchor>=.65)&(anchor<=4.2)&((first<=15.)|(np.abs(slope)<=.08))
        def best_path(side,excluded):
            scores=[-math.inf]*n;parent=[-1]*n;origin=list(range(n));paint=list(length)
            for j in range(n):
                if j in excluded:continue
                best=float(base[j]) if rootable[j] and candidates[j]['lane'].side==side else -math.inf;choice=-1
                for k,w in links[j]:
                    if scores[k]==-math.inf:continue
                    value=scores[k]+w
                    if value>best:best=value;choice=k
                scores[j]=best
                if choice>=0:parent[j]=choice;origin[j]=origin[choice];paint[j]=paint[choice]+length[j]
            # A lone short fragment keeps only part of its near-start bonus,
            # so it cannot outrank a long supported chain just behind it.
            adjusted=np.array(scores)-3.*np.maximum(0.,15.-first[origin])*(1-np.minimum(1.,np.array(paint)/3.))
            valid=[]
            for end in np.argsort(-adjusted,kind='stable'):
                if adjusted[end]==-math.inf or len(valid)==(8 if side in self.tracking_priors and c.carla_tracking_enabled else 5):break
                path=[];k=int(end)
                while k>=0:path.append(k);k=parent[k]
                path=[candidates[k] for k in reversed(path)]
                span=path[-1]['lane'].points[-1,0]-path[0]['lane'].points[0,0]
                if (len(path)>=2 and span>=2.5) or (len(path)==1 and span>=1.2 and path[0]['lane'].points[0,0]<=14.):
                    points=path_points(path)
                    if c.carla_width_reject_m>0:
                        off=self.width_error(side,points)
                        if off is not None and off>c.carla_width_reject_m:
                            self.diagnostics['width_rejected_'+side]=round(off,2);continue
                    valid.append((float(adjusted[end])-self.prior_penalty(side,path,points),path,points))
            # Penalties only reorder supported candidates; none is removed.
            return self.path_tracking.select(side,valid,self.tracking_priors.get(side),self.frame_index)
        for side in ('left','right'):
            result=best_path(side,set())
            if result is not None:scored[side]=result
        if len(scored)==2:
            winner,second=sorted(scored,key=lambda s:-scored[s][0])
            used={id(a) for a in scored[winner][1]}
            if used & {id(a) for a in scored[second][1]}:
                result=best_path(second,{j for j,a in enumerate(candidates) if id(a) in used})
                if result is None:del scored[second]
                else:scored[second]=result
        for side,(_,path) in scored.items():
            for a in path:a['lane'].side=side
            chosen[side]=path
        return chosen

    def paint_fragments(self,mask):
        """Publish observed paint fragments rather than a whole-road fitted arc.

        CARLA's boxes and dash gaps must not be filled with invented markings.
        Local row medians retain curved paint without extrapolation.
        """
        c=self.c;tile=max(8,round(2./c.mpp));lanes=[];nodes=[];selected=np.zeros_like(mask);self.frame_index+=1
        self.boundary_candidates=[]
        if c.carla_native_enabled and _native is not None:
            for points,m,b,lo,hi,width,yellow,y,x in _native.extract_fragments(mask,self.yellow,c.mpp,c.forward_max,c.lateral_min):
                anchor=c.lateral_min+(m*((c.forward_max-3.)/c.mpp)+b)*c.mpp
                if not yellow and self.edge_like(y,x):continue
                lane=Lane('left' if anchor<0 else 'right',points,min(.95,max(.55,.9-width*2.)))
                nodes.append(dict(lane=lane,fit=(m,b),lo=lo,hi=hi,yellow=yellow,pixels=(y,x),parent=len(nodes),anchor=anchor))
        else:
            for start in range(0,len(mask),tile):
                part=mask[start:start+tile]
                count,labels,stats,_=cv2.connectedComponentsWithStats(part)
                for label in range(1,count):
                    if stats[label,cv2.CC_STAT_AREA]<6 or (stats[label,cv2.CC_STAT_HEIGHT]-1)*c.mpp<.65:continue
                    y,x=np.nonzero(labels==label);y=y+start
                    if len(x)<6 or np.ptp(y)*c.mpp<.65:continue
                    # Closed-form 2x2 covariance/PCA and OLS, avoiding a generic
                    # eigensolver and least-squares solve for every tiny fragment.
                    xf=x.astype(float);yf=y.astype(float);mx=float(xf.mean());my=float(yf.mean())
                    dx=xf-mx;dy=yf-my;den=len(x)-1
                    vx=float(dx@dx)/den;vy=float(dy@dy)/den;cov=float(dx@dy)/den
                    disc=math.hypot(vx-vy,2*cov)
                    axis_y2=.5*(1-(vx-vy)/disc) if disc else 1.
                    normal_width=math.sqrt(max((vx+vy-disc)*.5,0.))*c.mpp
                    if axis_y2<.35**2 or normal_width>.10 or vy<=0:continue
                    slope=cov/vy;line=(slope,mx-slope*my)
                    anchor=c.lateral_min+(line[0]*((c.forward_max-3.)/c.mpp)+line[1])*c.mpp
                    if c.forward_max-y.max()*c.mpp<=8. and not .65<=abs(anchor)<=4.2:continue
                    rows=np.unique(y);cols=np.bincount(y,weights=x)[rows]/np.bincount(y)[rows]
                    residual=np.sort(np.abs(cols-(line[0]*rows+line[1])))
                    position=.9*(len(residual)-1);lower=int(position);fraction=position-lower
                    q=residual[lower] if lower==len(residual)-1 else residual[lower]+fraction*(residual[lower+1]-residual[lower])
                    if q*c.mpp>.15:continue
                    points=np.column_stack([c.forward_max-rows*c.mpp,-(c.lateral_min+cols*c.mpp)])[::-1]
                    confidence=float(np.clip(.9-normal_width*2.,.55,.95))
                    yellow=float(np.mean(self.yellow[y,x]))>.12
                    if not yellow and self.edge_like(y,x):continue
                    lane=Lane('left' if anchor<0 else 'right',points,confidence)
                    nodes.append(dict(lane=lane,fit=line,lo=float(y.min()),hi=float(y.max()),yellow=yellow,pixels=(y,x),parent=len(nodes),anchor=anchor))
        if c.carla_path_filter_enabled and c.carla_chain_filter_enabled:
            groups=self.coherent_paths(nodes)
        else:
            def parent(i):
                while nodes[i]['parent']!=i:
                    nodes[i]['parent']=nodes[nodes[i]['parent']]['parent'];i=nodes[i]['parent']
                return i
            for j,a in enumerate(nodes):
                for k in range(j+1,len(nodes)):
                    b=nodes[k]
                    if a['lane'].side!=b['lane'].side:continue
                    upper,lower=(a,b) if a['lo']<b['lo'] else (b,a)
                    gap=(lower['lo']-upper['hi'])*c.mpp
                    if not -.15<=gap<=10.:continue
                    angle=abs(math.atan(a['fit'][0])-math.atan(b['fit'][0]))
                    middle=(upper['hi']+lower['lo'])/2
                    error=abs((a['fit'][0]-b['fit'][0])*middle+a['fit'][1]-b['fit'][1])*c.mpp
                    if angle<math.radians(28) and error<.22+.025*max(gap,0):nodes[parent(k)]['parent']=parent(j)
            groups={}
            for j,node in enumerate(nodes):groups.setdefault(parent(j),[]).append(node)
        ranked={}
        rejected_groups=0
        accepted_groups=[]
        for group in groups.values():
            span=(max(a['hi'] for a in group)-min(a['lo'] for a in group))*c.mpp
            paint=sum((a['hi']-a['lo'])*c.mpp for a in group)
            path_filter=c.carla_path_filter_enabled and c.carla_chain_filter_enabled
            if span<(1.2 if path_filter else 6.) or paint<(1.2 if path_filter else 2.) or len(group)<(1 if path_filter else 2):continue
            if not c.carla_chain_filter_enabled:
                accepted_groups.append(group);continue
            points=np.vstack([a['lane'].points for a in group])
            forward,center=grouped_median(points)
            unique_paint=len(forward)*c.mpp
            anchor=float(np.median([c.lateral_min+np.polyval(a['fit'],(c.forward_max-3.)/c.mpp)*c.mpp for a in group]))
            yellow=float(np.mean([a['yellow'] for a in group]))
            side=group[0]['lane'].side
            score=.35*min(span,25.)+.60*min(unique_paint,15.)-1.2*abs(anchor)
            score+=yellow if side=='left' else .5*(1-yellow)
            score+=1.5*max(0.,12.-forward[0])
            if side not in ranked or score>ranked[side][0]:
                ranked[side]=(score,group,forward,center)
        if c.carla_chain_filter_enabled:
            rejected_groups=max(0,len(groups)-len(ranked))
            spines={}
            for side,(_,group,forward,center) in ranked.items():
                spines[side]=self.spine(group,forward,center)
            if c.carla_path_filter_enabled:self.parallel_support(nodes,spines)
            self.shared={}
            if c.carla_shared_shape:self.share_shape(spines)
            # Each fragment supports at most one boundary, whatever side its
            # own straight extrapolation suggested.
            support={side:[] for side in spines}
            members={id(a):side for side,value in spines.items() for a in value[4]}
            for a in nodes:
                pts=a['lane'].points;best=None;middle=len(pts)//2
                for side,(sample,smooth,band,yellow_path,group) in spines.items():
                    if id(a) in members and members[id(a)]!=side:continue
                    if c.carla_path_filter_enabled:
                        if a['yellow']!=yellow_path or pts[0,0]<sample[0]-.15 or pts[-1,0]>sample[-1]+.15:continue
                    elif id(a) not in members:continue
                    # Cheap rejection before the robust median test.
                    if abs(pts[middle,1]-np.interp(pts[middle,0],sample,smooth))>band+.30:continue
                    error=float(np.median(np.abs(pts[:,1]-np.interp(pts[:,0],sample,smooth))))
                    if error<=band and (best is None or error<best[0]):best=(error,side)
                if best is not None:support[best[1]].append(a)
            remembered={};offsets={}
            for side,(sample,smooth,band,yellow_path,group) in spines.items():
                retained=support[side]
                if not retained:continue
                for a in retained:a['lane'].side=side
                offsets[side]=self.merge_strands(retained,sample,smooth,yellow_path,side)
            for side,(offset,merged) in offsets.items():
                sample,smooth,band=spines[side][:3]
                if side in self.shared and self.shared[side][0] in offsets:
                    # A shared shape follows the long side's strand correction along the
                    # road; its own paint already set its position, so only the change counts.
                    long,lo,hi=self.shared[side];source=spines[long][0]
                    shift=np.interp(sample,source,np.broadcast_to(offsets[long][0],source.shape))
                    own=(sample>=lo)&(sample<=hi)
                    offset=offset+shift-(float(np.median(shift[own])) if own.any() else 0.)
                spine=Lane(side,np.column_stack([sample,smooth+offset]),.8)
                spine.corridor_m=band;remembered[side]=(sample,smooth+offset)
                accepted_groups.append(merged);self.boundary_candidates.append(spine)
            self.last_spines=remembered
        for group in accepted_groups:
            lanes.extend(a['lane'] for a in group)
            for a in group:selected[a['pixels']]=255
        self.diagnostics['rejected_branch_groups']=rejected_groups
        self.diagnostics['selected_chains']=len(accepted_groups)
        self.diagnostics['tracking']=dict(self.path_tracking.decisions)
        mask[:]=selected;self.style_mask=selected.copy()
        if self.replaced_paint is not None:
            # Dashes are measured on the real paint along the chosen learned lines.
            near=cv2.dilate(selected,np.ones((1,2*round(.25/c.mpp)+1),np.uint8))>0
            self.style_mask=np.where(near&(self.replaced_paint>0),255,0).astype(np.uint8)
        self.diagnostics['retained_pixels']=int(np.count_nonzero(selected))
        self.diagnostics['fragments']=len(lanes)
        self.style_lanes={}
        for side in ['left','right']:
            parts=[l.points for l in lanes if l.side==side]
            if not parts:continue
            points=np.vstack(parts);forward,center=grouped_median(points)
            self.style_lanes[side]=Lane(side,np.column_stack([forward,center]),.7)
        return lanes

    def _fit(self,xx,yy,idx):
        """Region-stratified quadratic RANSAC before least-squares refinement."""
        c=self.c
        if len(idx)<c.min_pixels:return None
        y=yy[idx].astype(float);x=xx[idx].astype(float)
        if np.ptp(y)*c.mpp<c.min_span_m:return None
        cut=np.quantile(y,[1/3,2/3]);regions=[np.flatnonzero(y<=cut[0]),np.flatnonzero((y>cut[0])&(y<=cut[1])),np.flatnonzero(y>cut[1])]
        if any(not len(a) for a in regions):return None
        rng=np.random.default_rng(37)
        samples=np.column_stack([rng.choice(a,48) for a in regions])
        origin=float(np.mean(y));z=(y-origin)*c.mpp
        a=np.stack([z[samples]**2,z[samples],np.ones_like(z[samples])],axis=2)
        safe=np.abs(np.linalg.det(a))>1e-7
        if not np.any(safe):return None
        coeff=np.linalg.solve(a[safe],x[samples][safe])
        predicted=coeff[:,0,None]*z[None,:]**2+coeff[:,1,None]*z[None,:]+coeff[:,2,None]
        inliers=np.abs(predicted-x[None,:])*c.mpp<.12
        counts=np.sum(inliers,axis=1)
        best=int(np.argmax(counts));keep=inliers[best]
        if counts[best]<c.min_pixels:return None
        fitted=super()._fit(xx,yy,idx[keep])
        if fitted is None:return None
        curve,rows,confidence=fitted
        y=np.arange(max(0,int(rows.min())),min(len(self.mask),int(rows.max())+1),2)
        x=np.rint(np.polyval(curve,y)).astype(int)
        delta=np.arange(-2,3);cols=np.clip(x[:,None]+delta,0,self.mask.shape[1]-1)
        support=np.any(self.mask[y[:,None],cols]>0,axis=1)
        edges=np.flatnonzero(np.r_[True,np.diff(support.astype(int))!=0,True])
        for a,b in zip(edges[:-1],edges[1:]):
            if support[a] and (b-a)*2*c.mpp<.40:support[a:b]=False
        if np.count_nonzero(support)*2*c.mpp<1.2:return None
        good=y[support]
        if np.ptp(good)*c.mpp<c.min_span_m:return None
        rows=np.arange(good.min(),good.max()+.1,max(1,c.segment_m/c.mpp))
        return curve,rows,confidence

    def candidate_score(self,side,coeff,rows,confidence):
        c=self.c;y=np.arange(max(0,int(rows.min())),min(len(self.mask),int(rows.max())+1),2)
        x=np.rint(np.polyval(coeff,y)).astype(int)
        offsets=np.arange(-max(1,round(.10/c.mpp)),max(1,round(.10/c.mpp))+1)
        cols=np.clip(x[:,None]+offsets,0,self.mask.shape[1]-1)
        paint=np.any(self.mask[y[:,None],cols]>0,axis=1)
        edges=np.flatnonzero(np.r_[True,np.diff(paint.astype(int))!=0,True])
        for a,b in zip(edges[:-1],edges[1:]):
            if paint[a] and (b-a)*2*c.mpp<.40:paint[a:b]=False
        support=float(np.mean(paint)) if len(y) else 0.
        anchor=np.clip((c.forward_max-8.)/c.mpp,rows.min(),rows.max())
        lateral=abs(c.lateral_min+np.polyval(coeff,anchor)*c.mpp)
        span=float(np.ptp(rows)*c.mpp)
        reason='accepted'
        if support<.10:reason='insufficient_current_paint'
        elif span<c.min_span_m:reason='short_span'
        elif c.forward_max-rows.max()*c.mpp>12.:reason='no_near_anchor'
        elif not .65<=lateral<=4.8:reason='outside_lane_corridor'
        self.diagnostics['candidates'].append(dict(side=side,support=support,span=span,lateral=lateral,reason=reason))
        if reason!='accepted':return -np.inf
        return .45*confidence+.15*support+.45*min(span/15.,1.)-.08*lateral
