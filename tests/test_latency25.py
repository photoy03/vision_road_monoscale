"""Plan 25: row-cropped images (CameraInfo roi) and the _native fast paths give the outputs of the full image and
of the NumPy code."""
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from vision_road.core import Projector, Settings
from vision_road import carla as carla_module

CONFIG = Path(__file__).resolve().parents[1]/'src/vision_road/config'
native = pytest.importorskip('vision_road._native')


def projector():
    c = Settings(); rig = yaml.safe_load((CONFIG/'rig.yaml').read_text())
    p = Projector(CONFIG/'calibration/front.npz', rig, c); p.initialize(*p.size)
    return p


def test_row_crop_warp_equals_the_full_image():
    p = projector(); lo, hi = p.rows(.9); h = p.source_size[1]
    assert 0 <= lo < hi <= h
    rng = np.random.default_rng(3); frame = rng.integers(0, 256, (h, p.source_size[0], 3), dtype=np.uint8)
    valid = p.valid.copy(); valid[p.my >= h*.9] = 0
    full = cv2.bitwise_and(p.warp(frame), p.warp(frame), mask=valid)
    for top, bottom in ((lo, hi), (max(0, lo-5), min(h, hi+5))):
        crop = p.warp(np.ascontiguousarray(frame[top:bottom]), top)
        assert np.array_equal(cv2.bitwise_and(crop, crop, mask=valid), full)
    short = p.warp(np.ascontiguousarray(frame[lo+1:hi]), lo+1)  # one needed row missing
    assert not np.array_equal(cv2.bitwise_and(short, short, mask=valid), full)
    with pytest.raises(ValueError):
        p.crop_maps(h-10, 20)


def test_native_binary_equals_numpy():
    rng = np.random.default_rng(5)
    for trial in range(4):
        bev = rng.integers(0, 256, (120, 90, 3), dtype=np.uint8)
        bev[::7] = (40, 200, 230)  # yellow-ish rows
        valid = np.full(bev.shape[:2], 255, np.uint8)
        out = []
        for fast in (False, True):
            c = Settings(); c.carla_native_fast = fast; c.carla_yellow_relative_min = 6.; c.carla_parallel_ops = trial % 2 == 1
            d = carla_module.CarlaDetector(c); raw = d.binary(bev, valid)
            out.append((raw, d.yellow.copy(), d.response, d.yellow_mask))
        for a, b in zip(*out):
            assert a.dtype == b.dtype and np.array_equal(a, b)


def test_native_edge_like_equals_numpy():
    rng = np.random.default_rng(7); c = Settings(); c.carla_native_fast = True
    d = carla_module.CarlaDetector(c); gray = rng.normal(100, 40, (200, 120)).astype(np.float32)
    gray[:, 40:60] += 60  # a brightness step
    d.gray = gray; offset = max(2, round(.35/c.mpp)); frags = []
    for n in (2, 5, 9, 40, 129, 300):
        y = rng.integers(0, 200, n).astype(np.int32); x = rng.integers(0, 120, n).astype(np.int32); frags.append((y, x))
    for y0 in range(0, 190, 10):  # fragments along the step, where the test fires
        y = np.arange(y0, y0+10, dtype=np.int32); frags.append((y, np.full(10, 41+y0 % 17, np.int32)))
    got = native.edge_like(gray, frags, offset)
    want = [d.edge_like(y, x) for y, x in frags]
    assert got == want and any(want)


def numpy_match(a, b, limit, weight):
    from vision_road.core import median
    from numpy.core.multiarray import interp
    lo = max(a[0, 0], b[0, 0]); hi = min(a[-1, 0], b[-1, 0])
    if not hi > lo:
        return np.inf
    s = np.linspace(lo, hi, 12)
    error = median(np.abs(interp(s, a[:, 0], a[:, 1])-interp(s, b[:, 0], b[:, 1])))
    if error < limit:
        o = (a[:, 0] >= lo) & (a[:, 0] <= hi)
        a[o, 1] = weight*a[o, 1]+(1-weight)*interp(a[o, 0], b[:, 0], b[:, 1])
    return error


def test_native_track_match_corridor_and_reproject_equal_numpy():
    from vision_road.core import median
    from numpy.core.multiarray import interp
    from vision_road.temporal import TemporalDetector
    rng = np.random.default_rng(11)
    for _ in range(300):
        n, m = rng.integers(2, 40, 2)
        a = np.column_stack([np.sort(rng.uniform(3, 30, n)), rng.normal(1.7, .2, n)])
        b = np.column_stack([np.sort(rng.uniform(3, 30, m)), rng.normal(1.7, .2, m)])
        if rng.random() < .2:
            b[:, 0] = np.round(b[:, 0])  # repeated stations
        a2 = a.copy(); want = numpy_match(a2, b, .35, .95); got = native.track_match(a, b, .35, .95)
        assert got == want and np.array_equal(a, a2)
        inside = (a[:, 0] >= b[0, 0]) & (a[:, 0] <= b[-1, 0])
        want = median(np.abs(a[inside, 1]-interp(a[inside, 0], b[:, 0], b[:, 1]))) if np.count_nonzero(inside) >= 3 else np.inf
        assert native.corridor_error(a, b) == want
    t = TemporalDetector(Settings()); c = t.c
    for _ in range(200):
        pts = np.column_stack([rng.uniform(3, 30, 25), rng.uniform(-6, 6, 25)])
        matrix = np.array([[1., rng.normal(0, .02), rng.normal(0, 3)], [rng.normal(0, .02), 1., rng.normal(5, 3)]])
        want = t.metric(t.pixel(pts) @ matrix[:, :2].T+matrix[:, 2])
        got = native.reproject(pts, matrix, c.mpp, c.forward_max, c.lateral_min)
        # OpenBLAS's x86-64 FMA dot product: bit-equal here; a different BLAS kernel may differ in the last bit.
        assert np.allclose(got, want, rtol=0, atol=1e-12)


def test_image_in_worker_takes_row_cropped_images(tmp_path):
    """The worker places a row-cropped image by the CameraInfo roi and produces lines; without the roi it skips it."""
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from std_msgs.msg import Header
    from builtin_interfaces.msg import Time
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image, CameraInfo
    from visualization_msgs.msg import MarkerArray
    from cv_bridge import CvBridge
    from vision_road.node import RoadNode
    rclpy.init(args=['--ros-args', '-p', 'detector_backend:=carla', '-p', 'isolated_compute:=true', '-p', 'image_in_worker:=true',
                     '-p', 'display_filter_inline:=true', '-p', 'display_filter_in_worker:=true', '-p', 'display_publish_in_worker:=true',
                     '-p', 'output_thread:=true', '-p', 'image_topic:=/crop/cam/image_raw', '-p', 'image_reliability:=reliable',
                     '-p', 'odom_topic:=/crop/odom', '-p', 'carla_chain_confirm_frames:=1'])
    road = RoadNode(); reader = Node('crop_reader'); executor = SingleThreadedExecutor()
    executor.add_node(road); executor.add_node(reader)
    boundaries = [];reader.create_subscription(MarkerArray, '/road/lane_boundary_markers', boundaries.append, 10)
    images = reader.create_publisher(Image, '/crop/cam/image_raw', 10); infos = reader.create_publisher(CameraInfo, '/crop/cam/camera_info', 10)
    odometry = reader.create_publisher(Odometry, '/crop/odom', 10)
    def pump(duration):
        until = time.monotonic()+duration
        while time.monotonic() < until:executor.spin_once(timeout_sec=.01)
    try:
        assert road.camera_info_topic == '/crop/cam/camera_info'
        w, h = road.projector.size; lo, hi = road.projector.rows(road.image_roi_bottom)
        frame = cv2.resize(cv2.imread(str(Path(__file__).resolve().parents[1]/'artifacts/synthetic_camera.png')), (w, h))
        crop = np.ascontiguousarray(frame[lo:hi]); pump(1.)
        def send(k, with_roi):
            t = 40.+k*.1; stamp = Time(sec=int(t), nanosec=round((t-int(t))*1e9))
            msg = Odometry(); msg.header = Header(stamp=stamp, frame_id='map'); msg.child_frame_id = 'base_link'
            msg.pose.pose.orientation.w = 1.; msg.pose.pose.position.x = 2.+k*.5; odometry.publish(msg)
            info = CameraInfo(); info.header = Header(stamp=stamp, frame_id='camera'); info.width, info.height = w, h
            if with_roi:info.roi.y_offset, info.roi.height, info.roi.width = lo, hi-lo, w
            infos.publish(info); pump(.05)
            image = CvBridge().cv2_to_imgmsg(crop, 'bgr8'); image.header = Header(stamp=stamp, frame_id='camera')
            images.publish(image); pump(.4)
        for k in range(3):send(k, False)  # no roi: skipped
        assert not boundaries
        for k in range(3, 10):send(k, True)
        pump(1.)
        assert len(boundaries) >= 5
        assert any(len(m.points) >= 2 for msg in boundaries[-3:] for m in msg.markers)
    finally:
        road.stop_processing()
        executor.shutdown(); road.destroy_node(); reader.destroy_node(); rclpy.shutdown()
