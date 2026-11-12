"""Learned mask gate, stop lines, crosswalks, markings and their confirmation on synthetic BEVs."""
from dataclasses import replace
import numpy as np
from vision_road.core import Settings, Lane
from vision_road.carla import CarlaDetector
from vision_road.road_markings import detect, MarkingTracker

C=Settings(forward_min=3,forward_max=35,lateral_min=-6,lateral_max=6,mpp=.05,min_span_m=.4)
WHITE=(220,220,220);YELLOW=(40,200,220);ROAD=(100,100,100)


def canvas():
    h=round((C.forward_max-C.forward_min)/C.mpp);w=round((C.lateral_max-C.lateral_min)/C.mpp)
    return np.full((h,w,3),ROAD,np.uint8)


def box(im,y0,y1,x0,x1,color=WHITE):
    """Fill lateral y0..y1 (left positive) by forward x0..x1, metres."""
    c0=round((C.lateral_max-y1)/C.mpp);c1=round((C.lateral_max-y0)/C.mpp)
    r0=round((C.forward_max-x1)/C.mpp);r1=round((C.forward_max-x0)/C.mpp)
    im[r0:r1,c0:c1]=color


def lanes(im):
    box(im,1.69,1.81,4,30,YELLOW);box(im,-1.81,-1.69,4,30)
    return [Lane('left',np.array([[4.,1.75],[30.,1.75]]),.9),Lane('right',np.array([[4.,-1.75],[30.,-1.75]]),.9)]


def run(im,settings=C,learned=None):
    d=CarlaDetector(settings);d.learned=learned
    valid=np.full(im.shape[:2],255,np.uint8);raw=d.binary(im,valid)
    return d,raw,valid


def column(y):return round((C.lateral_max-y)/C.mpp)


def test_learned_gate_removes_near_paint_off_the_mask_only():
    im=canvas();box(im,-1.81,-1.69,4,30);box(im,.44,.56,4,30)  # a lane line and a stray stripe
    learned=np.zeros(im.shape[:2],np.uint8);learned[:,column(-1.6):column(-1.9)]=255
    gate=replace(C,carla_learned_gate='remove',carla_learned_gate_far_m=20.)
    _,raw,_=run(im)
    assert raw[:,column(.5)].any()
    _,raw,_=run(im,gate,learned)
    stray=raw[:,column(.5)]>0;rows=np.flatnonzero(stray);x=C.forward_max-rows*C.mpp
    assert raw[:,column(-1.75)].any() and len(rows) and x.min()>20.-C.mpp  # kept beyond 20 m only
    # Without a mask (no GPU, no weights) nothing changes.
    _,plain,_=run(im)
    _,raw,_=run(im,gate,None)
    assert np.array_equal(raw,plain)


def test_penalize_keeps_pixels_and_marks_them():
    im=canvas();box(im,.44,.56,4,30)
    learned=np.zeros(im.shape[:2],np.uint8)
    d,raw,_=run(im,replace(C,carla_learned_gate='penalize'),learned)
    assert raw[:,column(.5)].any() and d.learned_outside is not None and d.learned_outside[:,column(.5)].any()


def test_stop_line_across_the_lane():
    im=canvas();bounds=lanes(im);box(im,-1.6,1.6,12,12.4)
    d,_,valid=run(im)
    stops=[m for m in detect(d,im,valid,bounds) if m['kind']=='stop_line']
    assert len(stops)==1 and abs(stops[0]['x']-12.2)<.2 and stops[0]['left']>1.5 and stops[0]['right']<-1.5


def test_repeated_bars_are_a_grid_not_stop_lines():
    im=canvas();bounds=lanes(im)
    for x in (8,11,14,17):box(im,-1.6,1.6,x,x+.3)
    d,_,valid=run(im)
    assert not [m for m in detect(d,im,valid,bounds) if m['kind']=='stop_line']


def test_yellow_bar_is_not_a_stop_line():
    im=canvas();bounds=lanes(im);box(im,-1.6,1.6,12,12.3,YELLOW)
    d,_,valid=run(im)
    assert not [m for m in detect(d,im,valid,bounds) if m['kind']=='stop_line']


def test_zebra_bars_are_one_crosswalk():
    im=canvas();bounds=lanes(im)
    for y in np.arange(-3.,3.,1.):box(im,y,y+.5,10,13)
    d,_,valid=run(im)
    walks=[m for m in detect(d,im,valid,bounds) if m['kind']=='crosswalk']
    # The bars touching the lane lines merge with them; the other four make the crosswalk.
    assert len(walks)==1 and walks[0]['bars']>=4 and 9.8<walks[0]['x0']<10.3 and 12.7<walks[0]['x1']<13.2
    # Lane dashes side by side are too narrow to be zebra bars.
    im=canvas();bounds=lanes(im)
    for y in (-.5,.5,1.):box(im,y,y+.12,10,13)
    d,_,valid=run(im)
    assert not [m for m in detect(d,im,valid,bounds) if m['kind']=='crosswalk']


def test_solid_symbol_in_the_lane_is_a_marking_and_a_small_plate_is_not():
    im=canvas();bounds=lanes(im);box(im,-.4,.4,8,10)
    d,_,valid=run(im)
    marks=[m for m in detect(d,im,valid,bounds) if m['kind']=='marking']
    assert len(marks)==1 and marks[0]['area']>1.
    im=canvas();bounds=lanes(im);box(im,-.3,.3,8,8.8)  # about 0.5 m2, a manhole cover's size
    d,_,valid=run(im)
    assert not [m for m in detect(d,im,valid,bounds) if m['kind']=='marking']


def stop(x):return dict(kind='stop_line',x=x,left=1.7,right=-1.7,thickness=.3)


def test_tracker_confirms_three_of_five_and_holds_in_the_world():
    t=MarkingTracker()
    # The vehicle drives forward 0.5 m a frame; the stop line stays at world x = 20.
    out=[t.update((.5*k,0.,0.),[stop(20-.5*k)] if k in (0,2,3) else []) for k in range(4)]
    assert not out[0] and not out[1] and not out[2] and out[3]
    assert abs(out[3][0]['points'][0,0]-(20-1.5))<.05 and not out[3][0]['predicted']
    # Missing up to four more frames keeps it; then it is gone.
    held=[t.update((.5*k,0.,0.),[]) for k in range(4,10)]
    assert all(h and h[0]['predicted'] for h in held[:4]) and not held[-1]


def test_single_frame_candidate_never_shows_and_no_pose_clears():
    t=MarkingTracker()
    assert not t.update((0.,0.,0.),[stop(15.)])
    for k in range(1,6):assert not t.update((.5*k,0.,0.),[])
    for k in range(3):t.update((0.,0.,0.),[stop(15.)])
    assert t.update((0.,0.,0.),[stop(15.)])
    assert not t.update(None,[stop(15.)]) and not t.tracks


def test_missing_weights_leave_the_detector_unchanged():
    from vision_road.rt_worker import LearnedMask
    class Projector:  # never initialised: no model is loaded without weights
        size=(1920,1080)
    off=LearnedMask(Projector(),'')
    assert off.status=='off' and off.next(1.,np.zeros((4,4,3),np.uint8)) is None
    missing=LearnedMask(Projector(),'/nonexistent/yolopv2.pt')
    assert missing.status.startswith('unavailable') and missing.next(1.,np.zeros((4,4,3),np.uint8)) is None


def test_centre_lines_keep_narrow_runs_only():
    from vision_road.carla import centre_lines
    m=np.zeros((4,40),bool);m[:,5:11]=True;m[:,20:40]=True
    c=centre_lines(m,10)
    assert c[:,7:9].all() and c.sum()==8 and not c[:,20:].any()


def test_replace_uses_learned_lines_and_keeps_paint_for_style():
    im=canvas()
    for x in range(4,30,6):box(im,-1.81,-1.69,x,x+3)  # dashed paint
    learned=np.zeros(im.shape[:2],np.uint8);learned[:,column(-1.6):column(-1.9)]=255  # one continuous learned line
    d,raw,_=run(im,replace(C,carla_learned_gate='replace',carla_learned_gate_far_m=20.),learned)
    rows=np.flatnonzero(raw[:,column(-1.75)-3:column(-1.75)+4].any(1));x=C.forward_max-rows*C.mpp
    gap=(x>7.2)&(x<9.8)&(x<20)  # between the first two dashes
    assert gap.any()  # the learned line bridges the dash gap within 20 m
    assert d.replaced_paint is not None and not d.replaced_paint[round((C.forward_max-8.5)/C.mpp)].any()
