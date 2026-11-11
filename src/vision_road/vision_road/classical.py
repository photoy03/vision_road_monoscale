"""Unlearned road-paint stripe extraction. No neural models or fitted classifiers."""
from dataclasses import replace
import cv2
import numpy as np
from .core import Detector

def filter_lane_candidates(mask, mpp, max_width_m=.45):
    """Reject broad/transverse paint without rejecting long or curved lanes.

    Horizontal runs measure width locally in metric BEV; total component area
    is deliberately not a rejection criterion. Never join separate dash runs.
    """
    if mpp <= 0 or max_width_m <= 0:
        raise ValueError('Candidate mask dimensions must be positive')
    result = mask.copy()
    max_width = max(2, int(np.ceil(max_width_m / mpp)))
    # Cut only the offending rows, preserving a stripe attached to a broad patch.
    for row in result:
        edges = np.flatnonzero(np.diff(np.r_[False, row > 0, False]))
        for start, end in edges.reshape(-1, 2):
            if end-start > max_width:
                row[start:end] = 0
    # Remove isolated specks. Examine transverse components in short tiles so
    # the bounding box of a long curve does not look like a broad object.
    tile = max(1, round(4. / mpp))
    for start in range(0, len(result), tile):
        part = result[start:start+tile]
        count, labels, stats, _ = cv2.connectedComponentsWithStats(part)
        for label in range(1, count):
            _, _, width, height, area = stats[label]
            speck = area <= 2 and height*mpp < .15
            transverse = width*mpp >= .65 and height*mpp < .30
            if speck or transverse:
                part[labels == label] = 0
    return result

class StripeDetector(Detector):
    def __init__(self,cfg):
        super().__init__(replace(cfg,lane_width_min=0.,lane_width_max=12.))

    def binary(self,bev,valid):
        c=self.c
        hls=cv2.cvtColor(bev,cv2.COLOR_BGR2HLS);h,light,sat=cv2.split(hls)
        gray=cv2.cvtColor(bev,cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray=cv2.GaussianBlur(gray,(3,3),.6)
        dx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)/8.
        white=sat<140;yellow=(h>=12)&(h<=40)&(sat>=80)
        response=np.zeros_like(gray);eligible=np.zeros_like(valid,dtype=bool)
        # Paired opposite edges across plausible physical paint widths.
        for width_m in [.08,.12,.16,.20,.24]:
            radius=max(1,round(width_m/(2*c.mpp)))
            left=np.roll(gray,radius+1,axis=1);right=np.roll(gray,-radius-1,axis=1)
            background=np.maximum(left,right)
            contrast=(gray-background)/np.maximum(background,20.)
            up=np.roll(dx,radius,axis=1);down=-np.roll(dx,-radius,axis=1)
            edge=np.minimum(up,down);balance=np.minimum(up,down)/np.maximum(np.maximum(up,down),1.)
            strength=np.minimum(gray-left,gray-right)
            good=(edge>=3.)&(balance>.35)&(strength>=6.)&(contrast>=.12)&(white|yellow)
            response=np.maximum(response,np.where(good,contrast,0))
            eligible|=good
        safe=cv2.erode(valid,np.ones((3,13),np.uint8))>0
        self.response=response
        mask=(eligible&safe).astype(np.uint8)*255
        mask=cv2.bitwise_and(mask,self.road_surface(bev,valid))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,1),np.uint8))
        result=np.zeros_like(mask)
        # Reject transverse stop stripes and short gravel highlights locally;
        # long curved lines are evaluated in short metric tiles.
        tile=max(12,round(5./c.mpp))
        for start in range(0,mask.shape[0],tile):
            part=mask[start:start+tile]
            n,labels,stats,_=cv2.connectedComponentsWithStats(part)
            for label in range(1,n):
                ys,xs=np.nonzero(labels==label)
                if len(xs)<5 or np.ptp(ys)*c.mpp<.30:continue
                points=np.column_stack([xs,ys]).astype(float)
                cov=np.cov(points.T);values,vectors=np.linalg.eigh(cov)
                if abs(vectors[1,-1])<.65 or np.sqrt(max(values[0],0))*c.mpp>.20:continue
                result[start:start+len(part)][labels==label]=255
        return result

    def road_surface(self,bev,valid):
        """Unlearned edge/texture barriers, flooded from near vehicle road seeds."""
        c=self.c;gray=cv2.cvtColor(bev,cv2.COLOR_BGR2GRAY)
        smooth=cv2.GaussianBlur(gray,(3,3),.6).astype(np.float32)
        gx=cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3)/8.
        gy=cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3)/8.
        mean=cv2.blur(smooth,(5,5));std=np.sqrt(np.maximum(cv2.blur(smooth*smooth,(5,5))-mean*mean,0))
        hsv=cv2.cvtColor(bev,cv2.COLOR_BGR2HSV)
        blocked=(valid==0)|(hsv[:,:,1]>100)|(np.hypot(gx,gy)>12.)|(std>25.)
        flood=np.zeros((gray.shape[0]+2,gray.shape[1]+2),np.uint8);flood[1:-1,1:-1]=blocked.astype(np.uint8)
        source=gray.copy();center=round(-c.lateral_min/c.mpp)
        for forward in np.arange(4.,min(25.,c.forward_max),2.):
            row=round((c.forward_max-forward)/c.mpp)
            for offset in [-.4,0.,.4]:
                col=center+round(offset/c.mpp)
                if 0<=row<gray.shape[0] and 0<=col<gray.shape[1] and flood[row+1,col+1]==0:
                    cv2.floodFill(source,flood,(col,row),0,loDiff=9,upDiff=9,flags=4|cv2.FLOODFILL_MASK_ONLY|(2<<8))
        road=(flood[1:-1,1:-1]==2).astype(np.uint8)*255
        road=cv2.dilate(road,np.ones((3,2*max(1,round(.18/c.mpp))+1),np.uint8))
        return cv2.bitwise_and(road,valid)

class PaintEvidenceDetector(Detector):
    """Photometric paint evidence and independent lane selection, without learning."""
    requires_paint_confirmation_for_prediction=True
    def __init__(self,cfg):
        super().__init__(replace(cfg,lane_width_min=0.,lane_width_max=12.,ridge_width_m=.35,ridge_contrast_min=10.,local_contrast_min=.20))
        self.style_detector=Detector(cfg)
        self.features=None

    def binary(self,bev,valid):
        mask=super().binary(bev,valid)
        gray=cv2.cvtColor(bev,cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.gray=gray;self.saturation=cv2.cvtColor(bev,cv2.COLOR_BGR2HLS)[:,:,2]
        self.style_mask=self.style_detector.binary(bev,valid)
        background=cv2.medianBlur(gray.astype(np.uint8),max(3,round(.65/self.c.mpp))|1)
        self.style_valid=cv2.bitwise_and(valid,(background<235).astype(np.uint8)*255)
        step=max(2,round(.35/self.c.mpp))
        left=cv2.blur(np.roll(gray,step,axis=1),(3,3));right=cv2.blur(np.roll(gray,-step,axis=1),(3,3))
        self.contrast=(gray-np.maximum(left,right))/np.maximum(np.maximum(left,right),20.)
        b,g,r=cv2.split(bev.astype(np.float32))
        self.yellow=(r>80)&(g>80)&(r>=.8*g)&(r<=1.35*g)&(b<.75*g)&(self.contrast>.45)
        step2=max(2,round(.20/self.c.mpp))
        ridge=np.minimum(gray-np.roll(gray,step2,axis=1),gray-np.roll(gray,-step2,axis=1))
        top=cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,np.ones((1,max(5,round(.4/self.c.mpp))|1),np.uint8))
        self.weak=((ridge>=2.)&(top>=2.)&(gray>=225)&(self.saturation<100)&(valid>0)).astype(np.uint8)*255
        self.features=[]
        self.candidate_mask_before = mask.copy()
        if self.c.candidate_mask_enabled:
            mask = filter_lane_candidates(mask, self.c.mpp, self.c.candidate_max_width_m)
        self.candidate_mask_after = mask.copy()
        return mask

    def paint_score(self,side,coeff,rows,confidence):
        c=self.c
        lo=max(0,round(float(rows.min())));hi=min(self.gray.shape[0]-1,round(float(rows.max())))
        y=np.arange(lo,hi+1,2);x=np.rint(np.polyval(coeff,y)).astype(int)
        radius=max(1,round(.12/c.mpp));delta=np.arange(-radius,radius+1)
        xx=np.clip(x[:,None]+delta,0,self.gray.shape[1]-1);yy=y[:,None]
        g=self.gray[yy,xx];ratio=self.contrast[yy,xx];sat=self.saturation[yy,xx]
        evidence=((g>225)&(ratio>.12)&(sat<110))|((g>60)&(ratio>.65)&(sat<130))|self.yellow[yy,xx]
        signal=np.any(evidence,axis=1)
        # Require sustained painted runs, not an isolated white gravel fleck.
        edges=np.flatnonzero(np.r_[True,np.diff(signal.astype(int))!=0,True])
        for a,b in zip(edges[:-1],edges[1:]):
            if signal[a] and (b-a)*2*c.mpp<.5:signal[a:b]=False
        fraction=float(np.mean(signal)) if len(signal) else 0.
        lateral=abs(c.lateral_min+np.polyval(coeff,(c.forward_max-10.)/c.mpp)*c.mpp)
        span=float(np.ptp(rows)*c.mpp)
        self.features.append(dict(side=side,fraction=fraction,confidence=confidence,lateral=lateral,span=span))
        if lateral>self.boundaries.get(side,np.inf)+.25:return -np.inf
        if fraction<.08:
            prior=self.current_priors.get(side)
            if prior is None or prior.hits<3 or prior.style=='unknown':return -np.inf
            forward=c.forward_max-y*c.mpp
            lateral_curve=-(c.lateral_min+np.polyval(coeff,y)*c.mpp)
            inside=(forward>=prior.points[0,0])&(forward<=prior.points[-1,0])
            if np.count_nonzero(inside)<10:return -np.inf
            error=float(np.median(np.abs(lateral_curve[inside]-np.interp(forward[inside],prior.points[:,0],prior.points[:,1]))))
            clipped=float(np.mean(np.max(g,axis=1)>245))
            if error>.25 or clipped<.4:return -np.inf
            return .40-error
        prior_distance=abs(c.lateral_min+np.polyval(coeff,(c.forward_max-10.)/c.mpp)*c.mpp)
        return confidence+.20*fraction-.12*prior_distance

    def detect(self,mask,priors=None):
        self.current_priors=priors or {}
        c=self.c
        for prior in self.current_priors.values():
            if prior.hits<3 or prior.style=='unknown':continue
            pixels=np.column_stack([(-prior.points[:,1]-c.lateral_min)/c.mpp,(c.forward_max-prior.points[:,0])/c.mpp]).astype(np.int32)
            band=np.zeros_like(mask);cv2.polylines(band,[pixels],False,255,max(1,round(.45/c.mpp)))
            # Only current weak paired-ridge pixels enter this mask. Historical
            # geometry constrains the search but is never painted into evidence.
            mask|=cv2.bitwise_and(self.weak,band)
        if c.candidate_mask_enabled:
            mask[:] = filter_lane_candidates(mask, c.mpp, c.candidate_max_width_m)
        self.candidate_mask_after = mask.copy()
        self.boundaries={}
        previous=self.previous.copy();hits=self.hits.copy()
        self.boundary_pass=True
        super().detect(mask,priors,self.paint_score)
        for f in self.features:
            if f['fraction']<.08 and f['confidence']>=.48 and f['span']>=4. and f['lateral']>1.5:
                side=f['side'];self.boundaries[side]=min(self.boundaries.get(side,np.inf),f['lateral'])
        self.previous=previous;self.hits=hits
        self.boundary_pass=False
        return super().detect(mask,priors,self.paint_score)

    def _fit(self,xx,yy,idx):
        base=super()._fit(xx,yy,idx)
        if base is None or getattr(self,'boundary_pass',True):return base
        coeff,rows,confidence=base;c=self.c
        y=np.arange(max(0,round(rows.min())),min(self.gray.shape[0]-1,round(rows.max()))+1,2)
        x=np.polyval(coeff,y);delta=np.arange(-max(2,round(.30/c.mpp)),max(2,round(.30/c.mpp))+1)
        cols=np.clip(np.rint(x).astype(int)[:,None]+delta,0,self.gray.shape[1]-1)
        g=self.gray[y[:,None],cols];r=self.contrast[y[:,None],cols];s=self.saturation[y[:,None],cols]
        good=((g>225)&(r>.10)&(s<110))|((g>60)&(r>.65)&(s<130))|self.yellow[y[:,None],cols]
        strength=np.where(good,np.minimum(np.maximum(r,0),3.)*np.exp(-.5*(delta[None,:]*c.mpp/.25)**2),0.)
        best=np.argmax(strength,axis=1);weight=strength[np.arange(len(y)),best];keep=weight>0
        if np.count_nonzero(keep)<20 or np.ptp(y[keep])*c.mpp<3.:return base
        target=cols[np.arange(len(y)),best].astype(float)
        for _ in range(3):
            fitted=np.polyfit(y[keep],target[keep],2,w=np.sqrt(weight[keep]))
            keep=(weight>0)&(np.abs(target-np.polyval(fitted,y))*c.mpp<.12)
            if np.count_nonzero(keep)<20:return base
        if np.max(np.abs(np.polyval(fitted,rows)-np.polyval(coeff,rows)))*c.mpp>.35:return base
        # A bounded in-frame curve extension spans the end of a dash gap. It
        # depends on current paint, never on drawing an old curve into the mask.
        end=min(self.gray.shape[0]-1,rows.max()+3./c.mpp)
        new_rows=np.arange(rows.min(),end+.1,max(1,c.segment_m/c.mpp))
        return fitted,new_rows,confidence
