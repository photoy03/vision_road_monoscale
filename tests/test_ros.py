"""Real ROS serialization/subscription smoke test; run after colcon build."""
import time
from pathlib import Path
import cv2
import numpy as np
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from nav_msgs.msg import Odometry, OccupancyGrid
from vision_lane_interfaces.msg import MarkingSegmentArray, LaneStyleArray
from vision_road.node import RoadNode
from visualization_msgs.msg import Marker, MarkerArray


@pytest.mark.parametrize('backend',['baseline','paint_evidence','carla'])
def test_ros_publish_delayed_pose_map_and_clock_reset(backend):
    rclpy.init()
    road = RoadNode(); reader = Node('road_test_reader'); executor = SingleThreadedExecutor()
    if backend=='paint_evidence':
        from vision_road.classical import PaintEvidenceDetector
        road.detector.detector=PaintEvidenceDetector(road.detector.c)
        road.detector.c.style_temporal_enabled=True
        road.detector.c.style_hold_sec=2.
    if backend=='carla':
        from vision_road.carla import CarlaDetector
        road.detector.c.carla_chain_confirm_frames=1
        road.detector.detector=CarlaDetector(road.detector.c)
    executor.add_node(road); executor.add_node(reader)
    local, global_map, grids, styles = [], [], [], []
    paint_markers,boundary_markers=[],[]
    subscriptions = [reader.create_subscription(typ, topic, dest.append, 10) for typ, topic, dest in [
        (MarkingSegmentArray, '/parking/cpp/marking_segments', local),
        (MarkingSegmentArray, '/parking/cpp/global_marking_segments_fused', global_map),
        (MarkerArray,'/road/lane_markers',paint_markers),
        (MarkerArray,'/road/lane_boundary_markers',boundary_markers),
        (OccupancyGrid, '/road/lane_grid', grids), (LaneStyleArray, '/road/lane_styles', styles)]]
    def pump(duration=.4):
        until = time.monotonic()+duration
        while time.monotonic() < until: executor.spin_once(timeout_sec=.02)
    def stamp(t): return Time(sec=int(t), nanosec=round((t-int(t))*1e9))
    def odom(t):
        msg = Odometry(); msg.header = Header(stamp=stamp(t), frame_id='map'); msg.child_frame_id = 'base_link'
        msg.pose.pose.orientation.w = 1.; msg.pose.pose.position.x = 2.
        road.on_odom(msg)
    try:
        pump(1)
        assert '/road/centerline' not in dict(road.get_topic_names_and_types())
        frame = cv2.imread(str(Path(__file__).resolve().parents[1]/'artifacts/synthetic_camera.png'))
        assert frame is not None
        road.process(frame, Header(stamp=stamp(10.), frame_id='camera'))
        pump(.1)
        assert local and local[-1].segments and not local[-1].pose_valid
        assert not global_map and len(road.pending) == 1
        odom(10.); pump(.3)
        assert global_map and global_map[-1].pose_valid and grids
        assert global_map[-1].header.frame_id == 'map'
        for t in [10.1, 10.2]:
            odom(t); road.process(frame, Header(stamp=stamp(t), frame_id='camera')); pump(.15)
        assert any(s.confirmed for s in local[-1].segments)
        assert paint_markers and boundary_markers
        assert paint_markers[-1].markers[0].action==Marker.DELETEALL
        assert all(m.header.frame_id=='base_link' for m in paint_markers[-1].markers)
        assert all(m.type==Marker.LINE_STRIP and m.points and m.lifetime.nanosec==300000000 for m in paint_markers[-1].markers[1:])
        assert len(boundary_markers[-1].markers)<=3
        assert styles and styles[-1].header.stamp==local[-1].header.stamp
        assert {s.side for s in styles[-1].lanes}=={'left','right'}
        assert all(s.source_stamp.sec == 10 for s in global_map[-1].segments)
        empty = np.zeros_like(frame)
        evidence_before=sum(v[2] for v in road.map.cells.values())
        odom(10.3); road.process(empty, Header(stamp=stamp(10.3),frame_id='camera')); pump(.1)
        assert local[-1].segments and all(s.reprojected for s in local[-1].segments)
        assert styles[-1].lanes and all(s.reprojected for s in styles[-1].lanes)
        assert all(abs(s.source_stamp.nanosec-200000000)<2 for s in local[-1].segments)
        assert sum(v[2] for v in road.map.cells.values())==evidence_before
        odom(10.7); road.process(empty,Header(stamp=stamp(10.7),frame_id='camera')); pump(.1)
        assert not local[-1].segments and not styles[-1].lanes
        odom(5.)
        assert not road.map.cells and not road.pending
        road.process(frame, Header(stamp=stamp(5.1), frame_id='camera')); pump(.7)
        assert not road.map.cells and road.dropped > 0
    finally:
        executor.shutdown(); road.destroy_node(); reader.destroy_node(); rclpy.shutdown()


def test_display_simplification_bounds_error_and_preserves_fragment_ends():
    from types import SimpleNamespace
    forward=np.linspace(5.,15.,201)
    points=np.column_stack([forward,1.7+.03*np.sin(forward*2.)])
    simplified=RoadNode.display_points(SimpleNamespace(display_error=.01),points)
    assert len(simplified)<len(points)
    assert np.allclose(simplified[[0,-1]],points[[0,-1]],atol=1e-6)
    start=simplified[:-1];direction=np.diff(simplified,axis=0)
    delta=points[:,None,:]-start[None,:,:]
    fraction=np.clip(np.sum(delta*direction,axis=2)/np.sum(direction*direction,axis=1),0,1)
    distance=np.linalg.norm(delta-fraction[:,:,None]*direction[None,:,:],axis=2)
    assert np.max(np.min(distance,axis=1))<=.010001
