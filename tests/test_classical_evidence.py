import numpy as np
from vision_road.classical import PaintEvidenceDetector
from vision_road.temporal import TemporalDetector
from test_core import fixture_bev


def test_candidate_mask_keeps_long_curves_and_dashes_but_cuts_broad_patch():
    from vision_road.classical import filter_lane_candidates
    mask = np.zeros((400, 220), np.uint8)
    for y in range(400):
        x = 20 + round(.0002*(y-200)**2)
        mask[y, x:x+3] = 255
    for y in range(20, 370, 70):
        mask[y:y+35, 100:103] = 255
    expected = mask.copy()
    mask[180:200, 130:200] = 255
    mask[50:54, 130:200] = 255
    mask[300, 180] = 255
    filtered = filter_lane_candidates(mask, .05)
    assert np.array_equal(filtered, expected)


def test_candidate_mask_preserves_lane_above_and_below_attached_patch():
    from vision_road.classical import filter_lane_candidates
    mask = np.zeros((200, 100), np.uint8)
    mask[:, 40:43] = 255
    mask[80:90, 20:70] = 255
    filtered = filter_lane_candidates(mask, .05)
    assert filtered[:80, 40:43].all() and filtered[90:, 40:43].all()
    assert not filtered[80:90].any()


def test_white_and_yellow_paint_under_shadow_without_models():
    c,image=fixture_bev(curve=True)
    image[50:220]=(image[50:220].astype(float)*[.32,.25,.22]).astype(np.uint8)
    valid=np.full(image.shape[:2],255,np.uint8)
    d=PaintEvidenceDetector(c);lanes=d.detect(d.binary(image,valid))
    assert {l.side for l in lanes}=={'left','right'}
    for lane in lanes:
        f=lane.points[:,0];expected=(1.75 if lane.side=='left' else -1.75)-.003*(f-4)**2
        assert np.median(np.abs(lane.points[:,1]-expected))<.1


def test_new_backend_does_not_turn_prior_into_a_current_measurement():
    c,image=fixture_bev();valid=np.full(image.shape[:2],255,np.uint8)
    d=TemporalDetector(c);d.detector=PaintEvidenceDetector(c)
    for t in [1.,1.1,1.2]:lanes,_=d.process(image,valid,t,(0.,0.,0.))
    assert len(lanes)==2
    blank=np.full_like(image,45)
    predicted,binary=d.process(blank,valid,1.3,(.1,0.,0.))
    assert not binary.any()
    assert predicted and all(l.reprojected and l.source_time==1.2 for l in predicted)
    expired,_=d.process(blank,valid,1.7,(.2,0.,0.))
    assert not expired


def test_unclassified_candidate_is_not_carried_as_confirmed_paint():
    c,image=fixture_bev();c.style_min_span_m=100.
    valid=np.full(image.shape[:2],255,np.uint8);d=TemporalDetector(c);d.detector=PaintEvidenceDetector(c)
    for t in [1.,1.1,1.2]:lanes,_=d.process(image,valid,t,(0.,0.,0.))
    assert lanes and all(l.style=='unknown' for l in lanes)
    lanes,_=d.process(np.full_like(image,45),valid,1.3,(.1,0.,0.))
    assert not lanes
