import numpy as np
from vision_road.marking_style import classify_profile, classify_lane
from vision_road.core import Settings, Detector
from vision_road.temporal import TemporalDetector
from test_core import fixture_bev


def test_solid_and_regular_dashes():
    solid=classify_profile(np.ones(300,bool),np.ones(300,bool))
    assert solid.kind=='solid' and solid.confidence>.9
    paint=np.tile(np.r_[np.ones(30,bool),np.zeros(30,bool)],5)
    dashed=classify_profile(paint,np.ones(len(paint),bool))
    assert dashed.kind=='dashed' and dashed.dash_count==5 and dashed.confidence>.8


def test_single_occlusion_and_short_view_are_unknown():
    paint=np.ones(300,bool);paint[100:140]=False
    assert classify_profile(paint,np.ones(300,bool)).kind=='unknown'
    assert classify_profile(np.ones(50,bool),np.ones(50,bool)).kind=='unknown'
    assert classify_profile(np.zeros(300,bool),np.ones(300,bool)).kind=='unknown'


def test_invalid_visibility_not_counted_as_dash_gaps():
    visible=np.tile(np.r_[np.ones(30,bool),np.zeros(30,bool)],5)
    assert classify_profile(np.ones(len(visible),bool),visible).kind=='unknown'


def test_irregular_breaks_do_not_establish_dashes():
    paint=np.zeros(500,bool)
    for start,length in [(0,20),(80,20),(130,20),(370,20)]:paint[start:start+length]=True
    assert classify_profile(paint,np.ones(500,bool)).kind=='unknown'


def test_observed_binary_not_rendered_curve_and_prediction_keeps_style_source():
    c,image=fixture_bev(curve=True,dashed=True);valid=np.full(image.shape[:2],255,'uint8')
    d=TemporalDetector(c)
    first,_=d.process(image,valid,1.,(0.,0.,0.))
    assert first and all(l.style=='unknown' for l in first)
    lanes,mask=d.process(image,valid,1.1,(0.,0.,0.))
    assert len(lanes)==2 and all(l.style=='dashed' for l in lanes)
    assert all(l.dash_count>=3 for l in lanes)
    # Fitted points form a continuous curve, but the source mask contains actual gaps.
    assert classify_lane(lanes[0],np.full_like(mask,255),valid,c).kind=='solid'
    carried,_=d.process(np.full_like(image,45),valid,1.2,(.1,0.,0.))
    assert carried and all(l.reprojected and l.style=='dashed' and l.source_time==1.1 for l in carried)
    expired,_=d.process(np.full_like(image,45),valid,1.6,(.1,0.,0.));assert not expired
