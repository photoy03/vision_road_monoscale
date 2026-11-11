"""Curb rejection, parallel support, double-line merge and boundary EMA on synthetic BEVs."""
from dataclasses import replace
import numpy as np
from vision_road.core import Settings
from vision_road.carla import CarlaDetector
from vision_road.temporal import TemporalDetector

C=Settings(forward_min=3,forward_max=35,lateral_min=-6,lateral_max=6,mpp=.05,min_span_m=.4)
WHITE=(220,220,220);YELLOW=(40,200,220);ROAD=(100,100,100)


def canvas():
    h=round((C.forward_max-C.forward_min)/C.mpp);w=round((C.lateral_max-C.lateral_min)/C.mpp)
    return np.full((h,w,3),ROAD,np.uint8)


def paint(im,lateral,start,end,color,width=.12):
    """Base-frame line: lateral is left-positive metres, start/end forward metres."""
    left=round((-lateral-width/2-C.lateral_min)/C.mpp);right=round((-lateral+width/2-C.lateral_min)/C.mpp)
    top=round((C.forward_max-end)/C.mpp);bottom=round((C.forward_max-start)/C.mpp)
    im[top:bottom,left:right+1]=color


def detect(im,detector=None):
    d=detector or CarlaDetector(C);valid=np.full(im.shape[:2],255,np.uint8)
    return d,d.detect(d.binary(im,valid))


def lateral(lanes,side):
    points=np.vstack([l.points for l in lanes if l.side==side])
    return float(np.median(points[:,1]))


def test_edge_test_keeps_paint_rejects_curb_and_spares_yellow():
    im=canvas();paint(im,-1.75,4,30,WHITE)
    # Island on the left: bright surface with a slightly brighter curb lip.
    column=round((-2.9-C.lateral_min)/C.mpp);im[:,:column]=(190,190,190);paint(im,2.9,4,30,(200,200,200),.10)
    d,lanes=detect(im)
    assert any(l.side=='right' for l in lanes)
    assert not any(l.side=='left' and abs(np.median(l.points[:,1])-2.9)<.2 for l in lanes)
    # The same step beside yellow paint is not tested: yellow is never a curb.
    paint(im,2.6,4,30,YELLOW)
    d,lanes=detect(im)
    assert any(l.side=='left' and abs(np.median(l.points[:,1])-2.6)<.1 for l in lanes)


def test_parallel_support_only_collects_measured_paint_in_the_band():
    base=canvas();paint(base,1.75,4,30,YELLOW)
    d,_=detect(base.copy())
    assert {b.side for b in d.boundary_candidates}=={'left'}
    # A lone far dash cannot root its own chain, but lies one lane width away.
    im=base.copy();paint(im,1.75-3.3,16,18,WHITE)
    d,lanes=detect(im)
    assert 'right' in {b.side for b in d.boundary_candidates}
    assert all(l.points[0,0]>=15.9 for l in lanes if l.side=='right')
    # The same dash outside the shifted band is not taken.
    im=base.copy();paint(im,1.75-3.3-.9,16,18,WHITE)
    d,_=detect(im)
    assert 'right' not in {b.side for b in d.boundary_candidates}


def boundary(detector,side,forward=10.):
    spine=[b for b in detector.boundary_candidates if b.side==side][0]
    return float(np.interp(forward,spine.points[:,0],spine.points[:,1]))


def strands(lanes,side):
    return sorted({round(float(np.median(l.points[:,1])),2) for l in lanes if l.side==side})


def test_double_yellow_paint_keeps_both_strands_and_boundary_takes_the_center():
    double=canvas();paint(double,1.6,4,30,YELLOW);paint(double,1.9,4,30,YELLOW)
    d,lanes=detect(double)
    # Paint topics carry what was measured: two strands, nothing in between.
    found=strands(lanes,'left')
    assert any(abs(y-1.6)<.05 for y in found) and any(abs(y-1.9)<.05 for y in found)
    assert not any(abs(y-1.75)<.08 for y in found)
    assert abs(boundary(d,'left')-1.75)<.05
    # One strand seen: paint stays on it, the boundary keeps the remembered center.
    single=canvas();paint(single,1.6,4,30,YELLOW)
    _,lanes=detect(single.copy(),d)
    assert all(abs(y-1.6)<.05 for y in strands(lanes,'left'))
    assert abs(boundary(d,'left')-1.75)<.05
    # The spacing memory expires after 30 frames without a double line.
    for _ in range(31):detect(single.copy(),d)
    assert abs(boundary(d,'left')-1.6)<.05


def test_boundary_ema_resets_on_a_jump_or_a_capture_gap():
    c=replace(C,carla_chain_confirm_frames=3)
    def run(frames):
        t=TemporalDetector(c);t.detector=CarlaDetector(c);valid=None;out=[]
        for stamp,y in frames:
            im=canvas();paint(im,y,4,30,YELLOW)
            valid=np.full(im.shape[:2],255,np.uint8)
            t.process(im,valid,stamp,(0.,0.,0.))
            out.append([float(np.interp(10.,b.points[:,0],b.points[:,1])) for b in t.boundary_lanes if b.side=='left'])
        return out
    # A 1 m jump is a different line: no blend toward the old one, and it is
    # shown again only after its own confirmation frames.
    out=run([(1.+.05*i,1.75) for i in range(4)]+[(1.2+.05*i,2.75) for i in range(4)])
    assert out[3] and abs(out[3][0]-1.75)<.05
    assert not out[4] and not out[5]
    assert out[6] and abs(out[6][0]-2.75)<.05
    # A gap above 0.4 s restarts confirmation even at the same place.
    out=run([(1.+.05*i,1.75) for i in range(4)]+[(1.7,1.75)])
    assert not out[4]
