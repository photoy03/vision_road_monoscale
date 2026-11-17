"""Plan 19: neighbouring lane outer lines (carla_outer_enabled)."""
from dataclasses import replace
import numpy as np
import pytest
from vision_road.core import Settings
from vision_road.carla import CarlaDetector
from vision_road.temporal import TemporalDetector

MPP=.05


def col(y):return round((6-y)/MPP)


def row(x):return round((35-x)/MPP)


def scene(left_outer=True,right_outer=True,right_own=True,diagonal=False):
    """Own left yellow solid at +1.75 m, own right white dashes at -1.65 m, outer right white solid at
    -5.15 m, outer left white dashes at +5.25 m (3.5 m lanes), 4-30 m ahead."""
    c=Settings(forward_min=3,forward_max=35,lateral_min=-6,lateral_max=6,mpp=MPP,min_span_m=.4)
    im=np.full((640,240,3),100,np.uint8)
    im[row(30):row(4),col(1.75):col(1.75)+3]=(60,200,220)
    if right_own:
        for lo in range(4,30,4):im[row(lo+2):row(lo),col(-1.65):col(-1.65)+3]=220
    if right_outer:im[row(30):row(4),col(-5.15):col(-5.15)+3]=220
    if left_outer:
        for lo in range(4,30,4):im[row(lo+2):row(lo),col(5.25):col(5.25)+3]=220
    if diagonal:
        # A line leaving the own line: 2.6 m out at 5 m, 4.6 m out at 25 m (not parallel).
        for x in np.arange(5.,25.,.05):
            y=-1.65-(2.6+(x-5.)*.1);im[row(x),col(y):col(y)+3]=220
    return c,im


def run(c,im,frames=4):
    d=TemporalDetector(c);d.detector=CarlaDetector(c);valid=np.full(im.shape[:2],255,np.uint8)
    for i in range(frames):lanes,_=d.process(im,valid,1+i*.05,(.0,0.,0.))
    return d,lanes


def test_outer_lines_are_off_by_default():
    assert Settings().carla_outer_enabled is False
    c,im=scene();d,_=run(c,im)
    assert d.detector.outer_candidates==[] and d.outer_lanes==[]


def test_own_lines_do_not_change_when_outer_lines_are_on():
    c,im=scene()
    off,a=run(c,im);on,b=run(replace(c,carla_outer_enabled=True),im)
    assert [(l.side,l.style,l.reprojected) for l in a]==[(l.side,l.style,l.reprojected) for l in b]
    for x,y in zip(a,b):assert np.array_equal(x.points,y.points)
    for x,y in zip(off.boundary_lanes,on.boundary_lanes):
        assert x.side==y.side and np.array_equal(x.points,y.points)
    assert on.outer_lanes


def test_each_side_gets_its_outer_line_a_lane_width_out_with_style_and_oncoming_flag():
    c,im=scene();d,_=run(replace(c,carla_outer_enabled=True),im,6)
    outer={l.side:l for l in d.outer_lanes}
    assert set(outer)=={'left_outer','right_outer'}
    right,left=outer['right_outer'],outer['left_outer']
    assert abs(right.outer['gap_m']-3.5)<.12 and abs(left.outer['gap_m']-3.5)<.12
    for lane,y in ((right,-5.15),(left,5.25)):
        p=lane.points[(lane.points[:,0]>=6)&(lane.points[:,0]<=25)]
        assert len(p) and np.max(np.abs(p[:,1]-y))<.12
    # The own left line is yellow: the lane beyond it is oncoming.
    assert left.outer['opposite'] and not right.outer['opposite']
    assert not left.outer['yellow'] and not right.outer['yellow']
    assert right.style=='solid' and left.style=='dashed'


def test_a_line_that_is_not_parallel_to_the_own_line_is_not_an_outer_line():
    c,im=scene(right_outer=False,diagonal=True);d,_=run(replace(c,carla_outer_enabled=True),im)
    assert 'right_outer' not in {l.side for l in d.outer_lanes}


def test_no_own_line_no_outer_line_on_that_side():
    c,im=scene(right_own=False);d,_=run(replace(c,carla_outer_enabled=True),im)
    assert 'right' not in {l.side for l in d.boundary_lanes}
    assert 'right_outer' not in {l.side for l in d.outer_lanes}


def test_outer_line_is_confirmed_over_frames_and_carried_over_briefly():
    c,im=scene();c=replace(c,carla_outer_enabled=True)
    d=TemporalDetector(c);d.detector=CarlaDetector(c);valid=np.full(im.shape[:2],255,np.uint8)
    seen=[]
    for i in range(3):
        d.process(im,valid,1+i*.05,(.0,0.,0.));seen.append({l.side for l in d.outer_lanes})
    assert 'right_outer' not in seen[0] and 'right_outer' in seen[2]
    _,blank=scene(right_outer=False)  # the right outer paint goes; the own lines stay
    d.process(blank,valid,1.15,(.0,0.,0.))
    right=[l for l in d.outer_lanes if l.side=='right_outer']
    assert right and right[0].reprojected


def test_native_and_python_fragments_give_the_same_outer_lines():
    from vision_road.carla import _native
    if _native is None:pytest.skip('Optional native acceleration is not built')
    c,im=scene();c=replace(c,carla_outer_enabled=True)
    a,_=run(c,im);b,_=run(replace(c,carla_native_enabled=False),im)
    assert [l.side for l in a.outer_lanes]==[l.side for l in b.outer_lanes]
    for x,y in zip(a.outer_lanes,b.outer_lanes):assert np.allclose(x.points,y.points,atol=1e-9)
