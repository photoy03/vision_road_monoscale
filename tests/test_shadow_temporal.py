import math
import cv2
import numpy as np
from vision_road.core import Settings, Detector, LaneMap, to_world
from vision_road.temporal import TemporalDetector, to_base
from test_core import fixture_bev


def test_shadow_paint_survives_below_global_brightness_threshold():
    c,image=fixture_bev(curve=True)
    shadow=image.copy(); shadow[50:260]=(shadow[50:260].astype(float)*[.32,.25,.22]).astype('uint8')
    valid=np.full(image.shape[:2],255,'uint8')
    c.shadow_enabled=False; old=Detector(c).binary(shadow,valid)
    c.shadow_enabled=True; new=Detector(c).binary(shadow,valid)
    # Evaluate known painted strips, not all texture pixels.
    locations=[]
    for row in range(60,250):
        f=c.forward_max-row*c.mpp
        col=round((-1.75+.003*(f-4)**2-c.lateral_min)/c.mpp)
        locations.append((row,col))
    assert np.mean([old[y,x]>0 for y,x in locations])<.05
    assert np.mean([new[y,x]>0 for y,x in locations])>.90
    lanes=Detector(c).detect(new); assert len(lanes)==2


def test_broad_shadow_edge_does_not_become_paint():
    c=Settings(); image=np.full((363,200,3),120,'uint8'); image[:,:100]=35
    mask=Detector(c).binary(image,np.full(image.shape[:2],255,'uint8'))
    assert not np.any(mask)


def test_odometry_compensation_short_gap_expiry_and_no_map_reinforcement():
    c,image=fixture_bev(); valid=np.full(image.shape[:2],255,'uint8'); temporal=TemporalDetector(c)
    for t in [1.,1.1]: lanes,_=temporal.process(image,valid,t,(0.,0.,0.))
    assert len(lanes)==2 and all(l.hits>=2 for l in lanes)
    original={l.side:l.points.copy() for l in lanes}
    blank=np.full_like(image,45)
    pose=(.4,.1,math.radians(2))
    predicted,_=temporal.process(blank,valid,1.2,pose)
    assert len(predicted)==2 and all(l.reprojected for l in predicted)
    for lane in predicted:
        world=to_world(lane.points,pose)
        # Reprojected points are a cropped subset of the original ground curve.
        for point in world:
            assert np.min(np.linalg.norm(original[lane.side]-point,axis=1))<1e-5
        assert lane.source_time==1.1 and lane.confidence<lanes[0].confidence+0.1
    lane_map=LaneMap(); lane_map.update(predicted,pose,1.2); assert not lane_map.cells
    output,_=temporal.process(blank,valid,1.5,pose); assert not output


def test_unconfirmed_lane_and_pose_jump_cannot_ghost():
    c,image=fixture_bev(); valid=np.full(image.shape[:2],255,'uint8'); d=TemporalDetector(c)
    d.process(image,valid,1.,(0.,0.,0.))
    output,_=d.process(np.full_like(image,45),valid,1.1,(0.,0.,0.)); assert not output
    d.reset()
    d.process(image,valid,2.,(0.,0.,0.));d.process(image,valid,2.1,(0.,0.,0.))
    output,_=d.process(np.full_like(image,45),valid,2.2,(20.,0.,0.)); assert not output


def test_visual_registration_translation_and_rejection():
    c=Settings(); d=TemporalDetector(c); rng=np.random.default_rng(7)
    gray=rng.integers(20,150,(363,200),dtype='uint8')
    gray=cv2.GaussianBlur(gray,(3,3),0)
    image=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR); valid=np.full(gray.shape,255,'uint8')
    d.gray=gray;d.valid=valid;d.last_time=1.
    movement=np.array([[1.,0.,2.],[0.,1.,8.]])
    moved=cv2.warpAffine(image,movement,(200,363))
    matrix,distance=d._registration(moved,valid,None,1.1)
    assert matrix is not None and d.registration=='optical_flow'
    assert np.allclose(matrix,movement,atol=.15)
    blank=np.full_like(image,45)
    matrix,_=d._registration(blank,valid,None,1.1); assert matrix is None


def test_temporal_disabled_never_predicts():
    c,image=fixture_bev();c.temporal_enabled=False
    d=TemporalDetector(c);valid=np.full(image.shape[:2],255,'uint8')
    for t in [1.,1.1]:d.process(image,valid,t,(0.,0.,0.))
    lanes,_=d.process(np.full_like(image,45),valid,1.2,(0.,0.,0.));assert lanes==[]
