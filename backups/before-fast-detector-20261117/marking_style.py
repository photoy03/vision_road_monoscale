"""Classify paint continuity on the measured binary image, never the rendered curve."""
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class StyleEvidence:
    kind: str = 'unknown'
    confidence: float = 0.0
    span_m: float = 0.0
    paint_fraction: float = 0.0
    dash_count: int = 0


def runs(values):
    edges=np.flatnonzero(np.r_[True,np.diff(values.astype(int))!=0,True])
    return [(int(a),int(b),bool(values[a])) for a,b in zip(edges[:-1],edges[1:])]


def classify_profile(paint, visible, spacing=.10, min_span=8., min_dash=.60, min_gap=.70):
    paint=np.asarray(paint,dtype=bool); visible=np.asarray(visible,dtype=bool)
    if len(paint)!=len(visible) or not len(paint): return StyleEvidence()
    # Missing camera support is not a dash gap. Use only a contiguous visible interval.
    intervals=[(a,b) for a,b,value in runs(visible) if value]
    if not intervals: return StyleEvidence()
    a,b=max(intervals,key=lambda pair:pair[1]-pair[0]); signal=paint[a:b].copy()
    span=len(signal)*spacing
    evidence=StyleEvidence(span_m=span,paint_fraction=float(np.mean(signal)))
    if span<min_span: return evidence
    # Suppress sub-mark texture and short cracks; never bridge a true dash gap.
    for lo,hi,value in runs(signal):
        if not value and lo>0 and hi<len(signal) and (hi-lo)*spacing<=.30:
            signal[lo:hi]=True
    for lo,hi,value in runs(signal):
        if value and (hi-lo)*spacing<.30:
            signal[lo:hi]=False
    blocks=runs(signal); fraction=float(np.mean(signal)); evidence.paint_fraction=fraction
    gaps=[(hi-lo)*spacing for lo,hi,value in blocks if not value and lo>0 and hi<len(signal)]
    # Dense paint with small isolated wear/shadow interruptions remains solid.
    # A long occlusion still cannot establish a marking type.
    if (fraction>=.85 and max(gaps or [0.])<min_gap) or (fraction>=.90 and max(gaps or [0.])<2.0):
        evidence.kind='solid'; evidence.confidence=float(min(.99,.55+.45*fraction))
        return evidence
    # Three paint runs and two internal gaps: one shadow/occlusion cannot imply dashed.
    marks=[(lo,hi) for lo,hi,value in blocks if value and (hi-lo)*spacing>=min_dash]
    valid_pairs=[]
    for left,right in zip(marks[:-1],marks[1:]):
        gap=(right[0]-left[1])*spacing
        if min_gap<=gap<=10.:
            valid_pairs.append((left,right,gap))
    if len(marks)<3 or len(valid_pairs)<2 or not (.15<=fraction<=.80): return evidence
    centers=np.array([(lo+hi)*spacing/2 for lo,hi in marks])
    periods=np.diff(centers); lengths=np.array([(hi-lo)*spacing for lo,hi in marks])
    # End runs can be truncated by the view, so evaluate regularity using centers
    # and the untruncated interior paint runs.
    regularity=float(np.std(periods)/max(np.mean(periods),1e-6))
    interior=lengths[1:-1]
    if regularity>.35 or np.median(lengths)>8. or (len(interior)>1 and np.std(interior)/max(np.mean(interior),1e-6)>.6):
        return evidence
    evidence.kind='dashed'; evidence.dash_count=len(marks)
    evidence.confidence=float(np.clip(.65+.20*min(len(valid_pairs)/3,1.)+.15*(1-regularity/.35),0.,.99))
    return evidence


def classify_lane(lane,binary,valid,cfg):
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(lane.points,axis=0),axis=1))]
    spacing=.10; sample=np.arange(0.,arc[-1],spacing)
    if not len(sample): return StyleEvidence()
    forward=np.interp(sample,arc,lane.points[:,0]);left=np.interp(sample,arc,lane.points[:,1])
    cols=np.rint((-left-cfg.lateral_min)/cfg.mpp).astype(int)
    rows=np.rint((cfg.forward_max-forward)/cfg.mpp).astype(int)
    width=max(1,round(.12/cfg.mpp));height,image_width=binary.shape
    inside=(rows>=0)&(rows<height)&(cols>=width)&(cols<image_width-width)
    rr=np.clip(rows,0,height-1)
    cc=np.clip(cols[:,None]+np.arange(-width,width+1),0,image_width-1)
    visible=inside & np.all(valid[rr[:,None],cc]>0,axis=1)
    paint=inside & np.any(binary[rr[:,None],cc]>0,axis=1)
    return classify_profile(paint,visible,spacing,cfg.style_min_span_m,cfg.style_min_dash_m,cfg.style_min_gap_m)


def draw_style_label(image,lane,pixels):
    if not len(pixels): return
    text=('L' if lane.side=='left' else 'R')+':'+lane.style.upper()
    if lane.reprojected: text+=' [PRED]'
    elif lane.style!='unknown' and lane.style_source_time+1e-6<lane.source_time: text+=' [HIST]'
    anchor=pixels[len(pixels)//3]
    if not np.isfinite(anchor).all(): return
    size,_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,.55,2)
    x=int(anchor[0])-(size[0]+12 if lane.side=='left' else -12)
    y=int(anchor[1])-10
    x=max(2,min(image.shape[1]-size[0]-4,x));y=max(size[1]+4,min(image.shape[0]-4,y))
    cv2.rectangle(image,(x-2,y-size[1]-3),(x+size[0]+2,y+3),(0,0,0),-1)
    cv2.putText(image,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255) if lane.reprojected else (255,255,255),2)
