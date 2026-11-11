import math
from pathlib import Path
import cv2
import numpy as np
import pytest
from vision_road.core import Settings, Projector, Detector, Timeline, Lane, LaneMap, to_world
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'src/vision_road/config'


def fixture_bev(curve=True, dashed=False):
    c = Settings(); h = round((c.forward_max-c.forward_min)/c.mpp); w = round((c.lateral_max-c.lateral_min)/c.mpp)
    image = np.full((h, w, 3), 45, dtype='uint8')
    for offset, color in [(-1.75, (255, 255, 255)), (1.75, (0, 220, 220))]:
        for v in range(h):
            f = c.forward_max-v*c.mpp
            lateral = offset + (0.003*(f-4)**2 if curve else 0)
            if not dashed or int(f)%3 != 0:
                cv2.circle(image, (round((lateral-c.lateral_min)/c.mpp), v), 1, color, -1)
    return c, image


@pytest.mark.parametrize('curve,dashed', [(False, False), (True, False), (True, True)])
def test_two_lanes(curve, dashed):
    c, image = fixture_bev(curve, dashed); detector = Detector(c)
    mask = detector.binary(image, np.full(image.shape[:2], 255, 'uint8'))
    lanes = detector.detect(mask)
    assert len(lanes) == 2
    for lane, expected in zip(lanes, [1.75, -1.75]):
        f = lane.points[:, 0]
        reference = expected-(0.003*(f-4)**2 if curve else 0)
        assert np.max(np.abs(lane.points[:, 1]-reference)) < 0.07
        assert np.all(np.diff(f) > 0)
    assert len(detector.detect(mask)) == 2
    assert detector.detect(np.zeros_like(mask)) == []  # No stale predictions.


def test_single_lane_no_invented_partner():
    c, image = fixture_bev(); detector = Detector(c)
    image[:, image.shape[1]//2:] = 45
    mask = detector.binary(image, np.full(image.shape[:2], 255, 'uint8'))
    assert [x.side for x in detector.detect(mask)] == ['left']


def test_flat_black_and_gray_no_divide_by_zero():
    c = Settings(); detector = Detector(c)
    for value in [0, 40, 120]:
        img = np.full((363, 200, 3), value, 'uint8')
        assert not np.any(detector.binary(img, np.full(img.shape[:2], 255, 'uint8')))


def test_invalid_pair_does_not_remove_supported_single_lane():
    c, img = fixture_bev(); c.lane_width_max = 3.0
    d = Detector(c)
    lanes=d.detect(d.binary(img,np.full(img.shape[:2],255,'uint8')))
    assert len(lanes)==1
    assert lanes[0].confidence>.5


def test_projection_metric_and_aspect():
    c = Settings(); rig = yaml.safe_load((CONFIG/'rig.yaml').read_text())
    p = Projector(CONFIG/'calibration/front.npz', rig, c)
    p.initialize(1280, 720)
    # Independent projection of base point, checks the left/right and forward convention.
    u, v = 60, 200
    base = np.array([c.forward_max-v*c.mpp, -(c.lateral_min+u*c.mpp), 0.0])
    cam = p.rotation.T @ (base-p.translation)
    k = p.k.copy(); k[:2, :] *= .5
    pix, _ = cv2.fisheye.projectPoints(cam.reshape(1, 1, 3), np.zeros(3), np.zeros(3), k, p.d)
    assert np.allclose([p.mx[v, u], p.my[v, u]], pix.ravel(), atol=1e-4)
    assert np.count_nonzero(p.valid) > 1000
    with pytest.raises(ValueError): p.initialize(640, 480)


def test_timeline_interpolate_wrapped_yaw_no_extrapolation():
    t = Timeline(.25); t.add(1, 0, 0, math.radians(179)); t.add(1.2, 2, 0, math.radians(-179))
    pose = t.at(1.1)
    assert np.allclose(pose[:2], [1, 0])
    assert abs(abs(pose[2])-math.pi) < 1e-6
    assert t.at(.9) is None and t.at(1.3) is None
    t.add(2, 3, 0, 0); assert t.at(1.5) is None
    t.add(.5, 0, 0, 0); assert t.generation == 1 and len(t.samples) == 1
    assert not t.add(.6, float('nan'), 0, 0)


def test_world_transform():
    assert np.allclose(to_world(np.array([[2., 1.]]), (10, 20, math.pi/2)), [[9, 22]])


def test_map_hits_per_frame_expiry_and_limit():
    lane = Lane('left', np.array([[0., 1.], [2., 1.]]), .9)
    m = LaneMap(resolution=.1, radius=10, ttl=2, max_cells=100)
    m.update([lane, lane], (0, 0, 0), 1)
    assert all(v[2] == 1 for v in m.cells.values())
    m.update([lane], (0, 0, 0), 2)
    assert all(v[2] == 2 for v in m.cells.values())
    m.update([], (0, 0, 0), 5); assert not m.cells
    m.max_cells = 3; m.update([lane], (0, 0, 0), 6); assert len(m.cells) == 3
    m.update([], (100, 0, 0), 7); assert not m.cells


def test_fisheye_image_to_observed_metric_curves():
    c = Settings(); rig = yaml.safe_load((CONFIG/'rig.yaml').read_text())
    p = Projector(CONFIG/'calibration/front.npz', rig, c)
    image = np.full((720, 1280, 3), 45, 'uint8')
    k = p.k.copy(); k[:2, :] *= .5
    for offset, color in [(1.75, (255, 255, 255)), (-1.75, (0, 220, 220))]:
        for f in np.arange(.5, 15, .04):
            points = np.array([[x, offset-.003*(x-4)**2+delta, 0] for x, delta in [(f,-.06),(f,.06),(f+.04,.06),(f+.04,-.06)]])
            cam = (points-p.translation) @ p.rotation
            if np.any(cam[:, 2] <= 0): continue
            uv, _ = cv2.fisheye.projectPoints(cam.reshape(-1,1,3), np.zeros(3), np.zeros(3), k, p.d)
            cv2.fillConvexPoly(image, np.round(uv.reshape(-1,2)).astype('int32'), color)
    bev = p.warp(image); d = Detector(c); mask = d.binary(bev, p.valid)
    lanes = d.detect(mask)
    assert len(lanes) == 2
    for lane, offset in zip(lanes, [1.75, -1.75]):
        assert np.max(np.abs(lane.points[:,1]-(offset-.003*(lane.points[:,0]-4)**2))) < .20
    (ROOT/'artifacts').mkdir(exist_ok=True)
    cv2.imwrite(str(ROOT/'artifacts/synthetic_camera.png'), image)
    cv2.imwrite(str(ROOT/'artifacts/synthetic_bev.png'), bev)
