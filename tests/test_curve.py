"""Curve stabilisation: shared boundary shape and dashed judged from paint accumulated along the road."""
from dataclasses import replace
import numpy as np
from vision_road.core import Settings
from vision_road.carla import CarlaDetector
from vision_road.temporal import TemporalDetector

C=Settings(forward_min=3,forward_max=35,lateral_min=-6,lateral_max=6,mpp=.05,min_span_m=.4)
WHITE=(220,220,220);YELLOW=(40,200,220);ROAD=(100,100,100)
K=1/60.  # left curve, radius 60 m


def canvas(cfg=C):
    h=round((cfg.forward_max-cfg.forward_min)/cfg.mpp);w=round((cfg.lateral_max-cfg.lateral_min)/cfg.mpp)
    return np.full((h,w,3),ROAD,np.uint8)


def arc(im,offset,start,end,color,width=.14):
    """Paint along y = offset + K/2 x^2 for x in [start, end] (left positive)."""
    for x in np.arange(start,end,C.mpp/2):
        y=offset+K/2*x*x;r=round((C.forward_max-x)/C.mpp)
        c0=round((C.lateral_max-y-width/2)/C.mpp);c1=round((C.lateral_max-y+width/2)/C.mpp)
        if 0<=r<im.shape[0]:im[r,max(0,c0):c1+1]=color


def at(points,x):return float(np.interp(x,points[:,0],points[:,1]))


def boundaries(im,cfg):
    d=CarlaDetector(cfg);valid=np.full(im.shape[:2],255,np.uint8);d.detect(d.binary(im,valid))
    return {b.side:b for b in d.boundary_candidates}


def test_short_side_takes_the_long_side_shape():
    im=canvas();arc(im,1.75,4,30,YELLOW);arc(im,-1.75,14,17,WHITE)  # long left line, one right dash
    shared=boundaries(im,replace(C,carla_shared_shape=True,carla_shared_extend_m=6.))
    right=shared['right'].points
    # Parallel to the left curve: at the dash it sits on the paint, and it reaches 6 m past it
    # without the straight-ish extrapolation a 3 m dash alone gives.
    assert abs(at(right,15.5)-(-1.75+K/2*15.5**2))<.06
    assert right[0,0]<=8.1 and right[-1,0]>=22.9
    truth=-1.75+K/2*22**2;normal_shift=3.5*(1/np.sqrt(1+(K*22)**2)-1)  # parallel, not the same quadratic shifted
    assert abs(at(right,22.)-truth)<.15+abs(normal_shift)
    plain=boundaries(im,C)['right'].points
    assert plain[0,0]>=13.9 and plain[-1,0]<=17.1  # without sharing, only the dash's own span


def test_no_sharing_when_both_sides_are_long():
    im=canvas();arc(im,1.75,4,30,YELLOW);arc(im,-1.75,4,30,WHITE)
    a=boundaries(im,replace(C,carla_shared_shape=True))['right'].points;b=boundaries(im,C)['right'].points
    assert np.allclose(a,b)


def run_dashes(cfg,dashes,frames=40,step=.5):
    C=cfg
    """Vehicle drives straight at `step` m per frame past right dashes fixed on the road (world x
    ranges), beside a solid left line.  The right boundary covers the gaps only through the
    shared shape; accumulation is judged along the boundary as drawn."""
    t=TemporalDetector(cfg);t.detector=CarlaDetector(cfg);valid=None;styles=[]
    for i in range(frames):
        s=i*step;im=canvas(cfg);im[:,round((C.lateral_max-1.82)/C.mpp):round((C.lateral_max-1.68)/C.mpp)]=YELLOW
        for a,b in dashes:
            lo,hi=a-s,b-s
            if hi>C.forward_min and lo<C.forward_max:
                r0=round((C.forward_max-min(hi,C.forward_max))/C.mpp);r1=round((C.forward_max-max(lo,C.forward_min))/C.mpp)
                im[r0:r1,round((C.lateral_max+1.68)/C.mpp):round((C.lateral_max+1.82)/C.mpp)]=WHITE
        valid=np.full(im.shape[:2],255,np.uint8)
        lanes,_=t.process(im,valid,i/30,(s,0.,0.))
        styles.append({l.style for l in lanes if l.side=='right' and not l.reprojected})
    return styles


def test_far_apart_dashes_become_dashed_only_with_accumulation():
    # 3 m dashes every 12 m and a 20 m view, as in the curve: one or two dashes in view,
    # rarely the three runs a single frame needs.
    dashes=[(x,x+3.) for x in np.arange(10.,120.,12.)]
    shared=replace(C,forward_max=20.,carla_shared_shape=True,carla_shared_extend_m=6.)
    plain=run_dashes(shared,dashes);acc=run_dashes(replace(shared,style_accumulate_m=12.),dashes)
    count=lambda styles:sum('dashed' in s for s in styles)
    assert count(acc)>=count(plain)+10 and not any('solid' in s for s in acc)


def test_accumulation_never_calls_a_solid_line_dashed():
    styles=run_dashes(replace(C,forward_max=20.,carla_shared_shape=True,style_accumulate_m=12.),[(0.,200.)])
    assert not any('dashed' in s for s in styles)
