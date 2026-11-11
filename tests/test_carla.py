from dataclasses import replace
import cv2,numpy as np
from vision_road.core import Settings,Lane,LaneMap
from vision_road.carla import CarlaDetector
from vision_road.temporal import TemporalDetector
from test_core import fixture_bev


def test_low_contrast_curved_paint_keeps_real_dash_gaps():
    c,im=fixture_bev(curve=True,dashed=True)
    c=replace(c,min_span_m=.4,carla_stage=3)
    # White 160 on asphalt 125: old >225 test cannot accept this branch.
    im=np.where(im>240,160,im).astype(np.uint8)
    road=np.all(im==45,axis=2);im[road]=125
    d=CarlaDetector(c);valid=np.full(im.shape[:2],255,np.uint8)
    mask=d.binary(im,valid);lanes=d.detect(mask)
    assert {l.side for l in lanes}=={'left','right'}
    for lane in lanes:
        px=np.rint(np.column_stack([(-lane.points[:,1]-c.lateral_min)/c.mpp,(c.forward_max-lane.points[:,0])/c.mpp])).astype(int)
        assert np.all(mask[px[:,1],px[:,0]]>0)
    # No synthetic stripe is painted into the binary through the dash gaps.
    forward=c.forward_max-np.arange(len(mask))*c.mpp
    gap=(np.floor(forward).astype(int)%3==0)&(forward%1>.3)&(forward%1<.7)
    assert not mask[gap].any()


def test_repeated_transverse_bars_are_not_lane_measurements():
    c=Settings(forward_min=3,forward_max=35,lateral_min=-6,lateral_max=6,mpp=.05,min_span_m=.4)
    im=np.full((640,240,3),100,np.uint8)
    for x in [60,100,140,180]:im[:,x:x+3]=220
    for y in [180,240,300,360]:im[y:y+4,40:200]=220
    d=CarlaDetector(c);valid=np.full(im.shape[:2],255,np.uint8);mask=d.binary(im,valid)
    assert d.grid_mask.any()
    assert not np.any(mask[d.grid_mask>0])


def test_no_paint_blank_and_single_isolated_symbol_do_not_create_lane():
    c=Settings(min_span_m=.4);d=CarlaDetector(c)
    im=np.full((363,200,3),125,np.uint8);v=np.full(im.shape[:2],255,np.uint8)
    assert d.detect(d.binary(im,v))==[]
    cv2.rectangle(im,(115,190),(130,215),(230,230,230),2)
    assert d.detect(d.binary(im,v))==[]


def test_geometry_tracks_unknown_style_without_ghost_map_or_expired_prediction():
    c,im=fixture_bev();c=replace(c,min_span_m=.4,style_min_span_m=100,temporal_max_frames=2,temporal_max_age_sec=.15)
    d=TemporalDetector(c);d.detector=CarlaDetector(c);v=np.full(im.shape[:2],255,np.uint8)
    for t in [1.,1.05,1.10,1.15]:lanes,_=d.process(im,v,t,(0.,0.,0.))
    assert lanes and all(l.style=='unknown' for l in lanes)
    assert len({l.track_key for l in lanes})==len(lanes)
    predicted,binary=d.process(np.full_like(im,125),v,1.20,(0.,0.,0.))
    assert predicted and not binary.any() and all(l.reprojected for l in predicted)
    m=LaneMap();m.update(predicted,(0.,0.,0.),1.20);assert not m.cells
    expired,_=d.process(np.full_like(im,125),v,1.40,(0.,0.,0.));assert not expired


def test_visible_clipping_never_joins_across_invalid_hole():
    c=Settings(min_span_m=.4);d=TemporalDetector(c)
    points=np.column_stack([np.arange(1,14,.1),np.ones(130)])
    valid=np.full((363,200),255,np.uint8);valid[130:180]=0
    clipped=d._visible(points,valid)
    assert clipped is not None
    rows=(c.forward_max-clipped[:,0])/c.mpp
    assert np.all(rows<130) or np.all(rows>=180)


def test_chain_confirmation_keeps_dash_gaps_and_resets():
    c,im=fixture_bev(curve=True,dashed=True)
    c=replace(c,min_span_m=.4,carla_chain_confirm_frames=3)
    d=TemporalDetector(c);d.detector=CarlaDetector(c)
    valid=np.full(im.shape[:2],255,np.uint8)
    for i in range(3):
        lanes,mask=d.process(im,valid,1+i*.05,(0.,0.,0.))
        if i<2:assert not lanes and not d.boundary_lanes
    assert {l.side for l in lanes}=={'left','right'}
    assert len(d.boundary_lanes)==2
    assert all(not l.reprojected for l in lanes)
    assert all(np.ptp(l.points[:,0])<3 for l in lanes)
    # A long capture gap cannot preserve old confirmations or predict paint.
    lanes,_=d.process(im,valid,3.,(0.,0.,0.))
    assert not lanes and not d.boundary_lanes
    d.reset();assert not d.boundaries and not d.boundary_lanes


def test_disconnected_short_speckles_do_not_form_boundary():
    c,im=fixture_bev(dashed=True)
    c=replace(c,min_span_m=.4)
    d=CarlaDetector(c);valid=np.full(im.shape[:2],255,np.uint8)
    rng=np.random.default_rng(42)
    for _ in range(150):
        x=int(rng.integers(2,im.shape[1]-3));y=int(rng.integers(2,im.shape[0]-4))
        im[y:y+3,x:x+2]=255
    lanes=d.detect(d.binary(im,valid))
    assert {l.side for l in lanes}=={'left','right'}
    assert len(d.boundary_candidates)==2
    for lane in lanes:
        assert np.median(np.abs(lane.points[:,1]))>1.


def test_near_boundary_wins_over_disconnected_far_parallel_chain():
    c=Settings(forward_min=3,forward_max=35,lateral_min=-6,lateral_max=6,mpp=.05,min_span_m=.4)
    im=np.full((640,240,3),100,np.uint8)
    # Near ego-lane dashes and a disconnected distant competing marking.
    for lo,hi,x in [(4,7,160),(10,13,160),(16,19,160),(23,26,190),(29,33,190)]:
        a=round((35-hi)/.05);b=round((35-lo)/.05)
        im[a:b,x:x+3]=220
    d=CarlaDetector(c);valid=np.full(im.shape[:2],255,np.uint8)
    lanes=d.detect(d.binary(im,valid))
    assert lanes and len(d.boundary_candidates)==1
    assert all(np.median(np.abs(l.points[:,1]))<2.5 for l in lanes)


def test_confirmation_survives_one_missing_candidate_without_ghost_display():
    c=Settings(min_span_m=.4,carla_chain_confirm_frames=3,temporal_max_age_sec=.15)
    d=TemporalDetector(c);d.detector=CarlaDetector(c)
    valid=np.full((363,200),255,np.uint8)
    lane=Lane('left',np.column_stack([np.arange(4,15,.2),np.full(55,1.7)]),.8)
    for t,present in [(1.,True),(1.05,False),(1.10,True),(1.15,True)]:
        d.detector.boundary_candidates=[lane] if present else []
        d._boundaries(valid,None,t)
        if t<1.15:assert not d.boundary_lanes
        d.last_time=t
    assert len(d.boundary_lanes)==1 and d.boundary_lanes[0].hits==3
    d.detector.boundary_candidates=[];d._boundaries(valid,None,1.20)
    assert not d.boundary_lanes


def test_short_real_dash_survives_without_inventing_other_dashes():
    c=Settings(forward_min=3,forward_max=35,lateral_min=-6,lateral_max=6,mpp=.05,min_span_m=.4)
    im=np.full((640,240,3),100,np.uint8)
    im[round((35-10.8)/.05):round((35-9.3)/.05),160:163]=220
    d=TemporalDetector(c);d.detector=CarlaDetector(c);valid=np.full(im.shape[:2],255,np.uint8)
    for i in range(3):lanes,mask=d.process(im,valid,1+i*.05,(0.,0.,0.))
    assert lanes and all(l.side=='right' for l in lanes)
    for lane in lanes:
        assert lane.points[:,0].min()>=9.2 and lane.points[:,0].max()<=10.9
    assert not mask[:450].any() and not mask[530:].any()


def test_colored_boundary_does_not_follow_attached_white_patch():
    c,im=fixture_bev();c=replace(c,min_span_m=.4)
    # Color-consistent paths must not merge a white diagonal branch into yellow.
    h,w=im.shape[:2]
    for row in range(h//3,2*h//3):
        col=int(w/2-1.7/c.mpp+(row-h//3)*.25)
        if 0<=col<w-3:im[row,col:col+3]=240
    d=CarlaDetector(c);v=np.full(im.shape[:2],255,np.uint8);lanes=d.detect(d.binary(im,v))
    left=[l for l in lanes if l.side=='left']
    assert left and all(abs(np.median(l.points[:,1])-1.7)<.35 for l in left)


def test_spine_includes_exact_observation_endpoints():
    c=Settings(forward_min=3,forward_max=35,lateral_min=-6,lateral_max=6,mpp=.05,min_span_m=.4)
    # The 1.55 m span is not divisible by the 0.2 m display sampling step.
    # Excluding the last sample used to drop this entire genuine dash.
    im=np.full((640,240,3),100,np.uint8);im[484:514,160:163]=220
    d=CarlaDetector(c);v=np.full(im.shape[:2],255,np.uint8);lanes=d.detect(d.binary(im,v))
    assert lanes and len(d.boundary_candidates)==1
    assert d.boundary_candidates[0].points[0,0]==min(l.points[0,0] for l in lanes)
    assert d.boundary_candidates[0].points[-1,0]==max(l.points[-1,0] for l in lanes)


def test_native_and_python_preserve_curved_dash_geometry_and_prediction():
    from vision_road.carla import _native
    import pytest
    if _native is None:pytest.skip('Optional native acceleration is not built')
    c,im=fixture_bev(curve=True,dashed=True);c=replace(c,min_span_m=.4,temporal_max_age_sec=.15)
    native=TemporalDetector(c);native.detector=CarlaDetector(c)
    slow=TemporalDetector(replace(c,carla_native_enabled=False));slow.detector=CarlaDetector(slow.c)
    valid=np.full(im.shape[:2],255,np.uint8)
    for i in range(8):
        current=im if i<6 else np.full_like(im,125)
        a,am=native.process(current,valid,1+i*.05,(0.,0.,0.))
        b,bm=slow.process(current,valid,1+i*.05,(0.,0.,0.))
        assert np.array_equal(am,bm) and len(a)==len(b)
        for x,y in zip(a,b):
            assert (x.side,x.style,x.reprojected)==(y.side,y.style,y.reprojected)
            assert np.allclose(x.points,y.points,rtol=0,atol=1e-10)
