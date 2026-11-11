"""CARLA paint detection: local paired borders, fragment graph, grid rejection.

No fitted classifier, model weights or simulator map labels are used.
"""
from dataclasses import replace
import math
import cv2
import numpy as np
from .core import Detector, Lane


def shifted(image, offset, axis=1):
    """Replicated boundaries, unlike wrapping a row to the opposite image edge."""
    indices=np.clip(np.arange(image.shape[axis])+offset,0,image.shape[axis]-1)
    return np.take(image,indices,axis=axis)


class CarlaDetector(Detector):
    fragment_tracks=True
    requires_paint_confirmation_for_prediction=False
    minimum_prediction_hits=3
    minimum_prediction_confidence=.55

    def __init__(self,cfg):
        if cfg.carla_stage not in (1,2,3) or cfg.carla_ridge_min<=0 or cfg.carla_relative_min<=0:
            raise ValueError('Invalid CARLA feature parameters')
        super().__init__(replace(cfg,lane_width_min=-12.,lane_width_max=12.))
        if cfg.carla_chain_confirm_frames<1 or cfg.carla_chain_band_m<=0:
            raise ValueError('Invalid CARLA chain filter limits')
        self.diagnostics={}
        self.boundary_candidates=[]

    def binary(self,bev,valid):
        c=self.c
        gray=cv2.cvtColor(bev,cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray=cv2.GaussianBlur(gray,(3,3),.55)
        h,l,s=cv2.split(cv2.cvtColor(bev,cv2.COLOR_BGR2HLS))
        b,g,r=cv2.split(bev.astype(np.float32))
        white=(s<120)&(l>c.carla_white_light_min)
        self.yellow=(h>=16)&(h<=42)&(s>20)&((g-b)>18)&((g-b)>.09*np.maximum(g,1))&((g-b)>1.25*(r-g))
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
        safe=cv2.erode(valid,np.ones((3,9),np.uint8))>0
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
        self.style_mask=raw.copy();self.style_valid=valid.copy()
        self.valid=valid.copy();self.mask=raw.copy()
        self.diagnostics=dict(raw_pixels=int(np.count_nonzero(self.raw_mask)),retained_pixels=int(np.count_nonzero(raw)),grid_pixels=int(np.count_nonzero(self.grid_mask)),candidates=[])
        return raw

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

    def paint_fragments(self,mask):
        """Publish observed paint fragments rather than a whole-road fitted arc.

        CARLA's boxes and dash gaps must not be filled with invented markings.
        Local row medians retain curved paint without extrapolation.
        """
        c=self.c;tile=max(8,round(2./c.mpp));lanes=[];nodes=[];selected=np.zeros_like(mask)
        self.boundary_candidates=[]
        for start in range(0,len(mask),tile):
            part=mask[start:start+tile]
            count,labels,stats,_=cv2.connectedComponentsWithStats(part)
            for label in range(1,count):
                y,x=np.nonzero(labels==label);y=y+start
                if len(x)<6 or np.ptp(y)*c.mpp<.65:continue
                xy=np.column_stack([x,y]).astype(float)
                values,vectors=np.linalg.eigh(np.cov(xy.T));axis=vectors[:,-1]
                normal_width=math.sqrt(max(values[0],0))*c.mpp
                if abs(axis[1])<.35 or normal_width>.10:continue
                line=np.polyfit(y,x,1);anchor=c.lateral_min+np.polyval(line,(c.forward_max-3.)/c.mpp)*c.mpp
                if not .65<=abs(anchor)<=4.2:continue
                rows=np.unique(y);cols=np.bincount(y,weights=x)[rows]/np.bincount(y)[rows]
                # Reject erratic paint-like texture; do not smooth it into a lane.
                if np.quantile(np.abs(cols-np.polyval(line,rows)),.9)*c.mpp>.15:continue
                points=np.column_stack([c.forward_max-rows*c.mpp,-(c.lateral_min+cols*c.mpp)])[::-1]
                confidence=float(np.clip(.9-normal_width*2.,.55,.95))
                lane=Lane('left' if anchor<0 else 'right',points,confidence)
                nodes.append(dict(lane=lane,fit=line,lo=float(y.min()),hi=float(y.max()),yellow=float(np.mean(self.yellow[y,x]))>.12,pixels=(y,x),parent=len(nodes)))
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
            if span<6. or paint<2. or len(group)<2:continue
            if not c.carla_chain_filter_enabled:
                accepted_groups.append(group);continue
            points=np.vstack([a['lane'].points for a in group])
            forward=np.unique(points[:,0])
            center=np.array([np.median(points[points[:,0]==f,1]) for f in forward])
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
            for side,(_,group,forward,center) in ranked.items():
                # Smooth only a locally supported spine; no global polynomial
                # is allowed to drag a turn across the opposite boundary.
                sample=np.arange(forward[0],forward[-1]+.001,.20)
                # A robust quadratic within the selected connected chain avoids
                # local polynomial oscillations across the actual dash gaps.
                origin=float(np.mean(forward));x=forward-origin
                weights=np.ones(len(x))
                for _ in range(5):
                    fit=np.polyfit(x,center,2,w=weights)
                    residual=np.abs(center-np.polyval(fit,x))
                    weights=np.minimum(1.,.20/np.maximum(residual,.001))
                smooth=np.polyval(fit,sample-origin)
                spine=Lane(side,np.column_stack([sample,smooth]),.8)
                retained=[]
                for a in group:
                    pts=a['lane'].points
                    error=np.median(np.abs(pts[:,1]-np.interp(pts[:,0],sample,smooth)))
                    if error<=c.carla_chain_band_m:retained.append(a)
                if retained:
                    accepted_groups.append(retained);self.boundary_candidates.append(spine)
        for group in accepted_groups:
            lanes.extend(a['lane'] for a in group)
            for a in group:selected[a['pixels']]=255
        self.diagnostics['rejected_branch_groups']=rejected_groups
        self.diagnostics['selected_chains']=len(accepted_groups)
        mask[:]=selected;self.style_mask=selected.copy()
        self.diagnostics['retained_pixels']=int(np.count_nonzero(selected))
        self.diagnostics['fragments']=len(lanes)
        self.style_lanes={}
        for side in ['left','right']:
            parts=[l.points for l in lanes if l.side==side]
            if not parts:continue
            points=np.vstack(parts);forward=np.unique(points[:,0]);center=np.array([np.median(points[points[:,0]==f,1]) for f in forward])
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
