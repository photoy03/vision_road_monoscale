import numpy as np
from vision_road.temporal import TemporalDetector
from test_core import fixture_bev


def test_paint_history_is_warped_and_never_replaces_current_binary():
    c,bev=fixture_bev();c.style_temporal_enabled=True
    valid=np.full(bev.shape[:2],255,np.uint8);d=TemporalDetector(c)
    paint=np.zeros_like(valid);paint[100:120,70:74]=255
    empty=np.zeros_like(paint)
    d.process(bev,valid,1.,(0.,0.,0.),paint_mask=paint)
    _,current=d.process(bev,valid,1.1,(.1,0.,0.),paint_mask=empty)
    assert not current.any()  # Topic evidence remains from the current camera.
    assert d.paint_history[0][1].any()
    expected_shift=round(.1/c.mpp)
    rows=np.flatnonzero(np.any(d.paint_history[0][1]>0,axis=1))
    assert rows.min()==100+expected_shift
    # One historical vote cannot manufacture corroborated paint.
    assert not d.style_binary.any()


def test_paint_history_expires_and_resets_without_registration():
    c,bev=fixture_bev();c.style_temporal_enabled=True;c.style_history_sec=.25;c.style_history_frames=2
    valid=np.full(bev.shape[:2],255,np.uint8);d=TemporalDetector(c)
    for t in [1.,1.1,1.2,1.3]:d.process(bev,valid,t,(0.,0.,0.))
    assert len(d.paint_history)==2
    d.process(bev,valid,2.,(0.,0.,0.))
    assert len(d.paint_history)==1
    d.process(bev,valid,.5,(0.,0.,0.))
    assert len(d.paint_history)==1


def test_confirmed_style_hold_expires_on_current_uncertain_observations():
    c,bev=fixture_bev();c.style_hold_sec=.2
    valid=np.full(bev.shape[:2],255,np.uint8);d=TemporalDetector(c)
    mask=d.detector.binary(bev,valid)
    for t in [1.,1.1]:lanes,_=d.process(bev,valid,t,(0.,0.,0.))
    assert lanes and all(l.style=='solid' for l in lanes)
    lanes,_=d.process(bev,valid,1.2,(0.,0.,0.),observation_mask=mask,paint_mask=np.zeros_like(mask))
    assert lanes and all(l.style=='solid' and l.style_source_time==1.1 and not l.reprojected for l in lanes)
    lanes,_=d.process(bev,valid,1.4,(0.,0.,0.),observation_mask=mask,paint_mask=np.zeros_like(mask))
    assert lanes and all(l.style=='unknown' and l.style_source_time==0 for l in lanes)
