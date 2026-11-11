#!/usr/bin/env python3
import json
import math
import time
import queue
import threading
from pathlib import Path
from collections import deque
from dataclasses import fields, replace
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose2D
from std_msgs.msg import String, Header
from vision_lane_interfaces.msg import MarkingSegment, MarkingSegmentArray, LaneStyle, LaneStyleArray
from ament_index_python.packages import get_package_share_directory
import yaml
from .core import Settings, Projector, Timeline, LaneMap
from .temporal import TemporalDetector
from .classical import PaintEvidenceDetector
from .carla import CarlaDetector
from .marking_style import draw_style_label


def seconds(stamp):
    return stamp.sec + stamp.nanosec*1e-9


class RoadNode(Node):
    def __init__(self):
        super().__init__('vision_road')
        share = get_package_share_directory('vision_road')
        def param(name, default):
            return self.declare_parameter(name, default).value
        cfg = Settings(**{f.name: param(f.name, getattr(Settings(), f.name)) for f in fields(Settings)})
        if not (cfg.mpp > 0 and cfg.segment_m > 0 and cfg.windows > 0 and cfg.min_pixels >= 3
                and cfg.forward_max > cfg.forward_min >= 0 and cfg.lateral_max > 0 > cfg.lateral_min
                and cfg.margin_m > 0 and cfg.fit_error_m > 0):
            raise ValueError('Invalid detector metric settings')
        if (cfg.style_min_span_m<=0 or cfg.style_min_dash_m<=0 or cfg.style_min_gap_m<=0 or cfg.style_confirm_frames<1
                or cfg.style_history_sec<=0 or cfg.style_history_frames<1 or cfg.style_hold_sec<0):
            raise ValueError('Invalid lane style settings')
        if (cfg.temporal_max_frames<0 or cfg.temporal_max_age_sec<=0 or cfg.temporal_max_distance_m<=0
                or cfg.ridge_width_m<=0 or cfg.local_contrast_min<=0 or cfg.temporal_match_m<=0):
            raise ValueError('Invalid temporal/shadow limits')
        rig = yaml.safe_load(open(param('rig_file', share+'/config/rig.yaml')))
        self.projector = Projector(param('calibration', share+'/config/calibration/front.npz'), rig, cfg)
        self.detector = TemporalDetector(cfg)
        backend=param('detector_backend','baseline')
        if backend=='paint_evidence':self.detector.detector=PaintEvidenceDetector(cfg)
        elif backend=='carla':self.detector.detector=CarlaDetector(cfg)
        elif backend!='baseline':raise ValueError('Unknown detector_backend')
        self.image_roi_bottom=param('image_roi_bottom',1.0)
        self.map_enabled=param('map_enabled',True)
        if not 0<self.image_roi_bottom<=1:raise ValueError('Invalid image_roi_bottom')
        threads=param('opencv_threads',2)
        if threads<1:raise ValueError('opencv_threads must be positive')
        cv2.setNumThreads(threads)
        self.pose_lock=threading.RLock();self.state_lock=threading.RLock()
        self.async_processing=param('async_processing',False)
        self.image_jobs=queue.Queue(maxsize=10);self.worker_stop=threading.Event();self.worker=None;self.reset_pending=False
        self.timeline = Timeline(param('odom_max_gap_sec', 0.25))
        self.map = LaneMap(param('map_resolution_m', 0.10), param('map_radius_m', 80.0), param('map_ttl_sec', 120.0), param('map_max_cells', 100000))
        if min(self.map.resolution, self.map.radius, self.map.ttl, self.map.max_cells, self.timeline.max_gap) <= 0:
            raise ValueError('Map and synchronization limits must be positive')
        self.wait_sec = param('pose_wait_sec', 0.5)
        self.max_pending = param('max_pending_frames', 30)
        if self.max_pending < 1 or self.wait_sec <= 0:
            raise ValueError('Pending queue and wait duration must be positive')
        self.performance_path=param('performance_report','')
        self.performance_rows=[]
        self.display_error=param('display_simplify_error_m',0.0)
        if not math.isfinite(self.display_error) or not 0<=self.display_error<=.05:raise ValueError('Invalid display simplification bound')
        self.overlay_max_width=param('overlay_max_width',0)
        if self.overlay_max_width<0:raise ValueError('Invalid overlay width')
        self.debug_hz=param('debug_publish_hz',0.0);self.last_debug=-math.inf
        if self.debug_hz<0:raise ValueError('Invalid debug image rate')
        self.debug = param('debug_images', True)
        self.map_hz = param('map_publish_hz', 2.0)
        self.world_frame = param('world_frame', 'map')
        self.base_frame = param('base_frame', 'base_link')
        self.compute_worker=None
        if param('isolated_compute',False):
            from .rt_worker import ComputeWorker
            self.projector.initialize(*self.projector.size)
            self.compute_worker=ComputeWorker(self.projector,cfg,backend,self.image_roi_bottom,threads)
        self.pending = deque(); self.last_image = None; self.last_commit = -math.inf
        self.last_map_publish = -math.inf; self.dropped = 0; self.frames = 0
        self.bridge = CvBridge()
        self.local_pub = self.create_publisher(MarkingSegmentArray, '/parking/cpp/marking_segments', 10)
        self.paint_marker_pub=self.create_publisher(MarkerArray,'/road/lane_markers',10)
        self.boundary_marker_pub=self.create_publisher(MarkerArray,'/road/lane_boundary_markers',10)
        self.style_pub = self.create_publisher(LaneStyleArray, '/road/lane_styles', 10)
        self.global_pub = self.create_publisher(MarkingSegmentArray, '/parking/cpp/global_marking_segments_fused', 2)
        self.map_pub = self.create_publisher(Image, '/parking/cpp/global_marking_map', 2)
        self.grid_pub = self.create_publisher(OccupancyGrid, '/road/lane_grid', 2)
        self.status_pub = self.create_publisher(String, '/parking/cpp/status', 10)
        self.binary_pub = self.create_publisher(Image, '/parking/cpp/binary', qos_profile_sensor_data)
        self.bev_pub = self.create_publisher(Image, '/parking/cpp/vector_bev', qos_profile_sensor_data)
        self.overlay_pub = self.create_publisher(Image, '/road/overlay', qos_profile_sensor_data)
        image_topic = param('image_topic', '/sensing/camera/front/fisheye/image_raw')
        compressed = param('compressed', False)
        reliability = param('image_reliability', 'best_effort')
        depth = param('image_queue_depth', 2)
        if reliability not in ('reliable', 'best_effort') or depth < 1:
            raise ValueError('Invalid image QoS settings')
        image_qos = QoSProfile(depth=depth, reliability=ReliabilityPolicy.RELIABLE if reliability=='reliable' else ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CompressedImage if compressed else Image, image_topic, self.on_compressed if compressed else self.on_image, image_qos)
        # Best effort accepts both reliable and best-effort odometry publishers.
        self.create_subscription(Odometry, param('odom_topic', '/localization/kinematic_state'), self.on_odom, QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_timer(0.02, self.timer_flush, clock=Clock(clock_type=ClockType.STEADY_TIME))
        if self.async_processing:
            self.worker=threading.Thread(target=self.work_images,name='road-image-worker',daemon=True);self.worker.start()
        self.get_logger().info('Road detector ready: '+backend+'; image processing only, no learned model')

    def reset(self):
        self.pending.clear(); self.map.cells.clear(); self.detector.reset()
        if self.compute_worker is not None:self.compute_worker.reset_pending=True
        self.last_commit = -math.inf; self.last_map_publish = -math.inf;self.last_debug=-math.inf

    def on_odom(self, msg):
        if msg.header.frame_id != self.world_frame or msg.child_frame_id != self.base_frame:
            self.get_logger().warning('Ignoring odometry: expected '+self.world_frame+' -> '+self.base_frame, throttle_duration_sec=5)
            return
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        norm = math.sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w)
        if not math.isfinite(norm) or norm < 1e-6:
            return
        x, y, z, w = q.x/norm, q.y/norm, q.z/norm, q.w/norm
        yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        with self.pose_lock:
            generation = self.timeline.generation
            self.timeline.add(seconds(msg.header.stamp), p.x, p.y, yaw)
            changed=self.timeline.generation != generation
        if self.async_processing:
            if changed:
                self.reset_pending=True
                while True:
                    try:self.image_jobs.get_nowait()
                    except queue.Empty:break
        else:
            if changed:self.reset();self.last_image=None
            self.flush()

    def pose_at(self,t):
        with self.pose_lock:return self.timeline.at(t)

    def timer_flush(self):
        if self.state_lock.acquire(blocking=False):
            try:self.flush()
            finally:self.state_lock.release()

    def enqueue_image(self,kind,msg):
        try:self.image_jobs.put_nowait((kind,msg))
        except queue.Full:
            try:self.image_jobs.get_nowait();self.dropped+=1
            except queue.Empty:pass
            self.image_jobs.put_nowait((kind,msg))

    def work_images(self):
        while not self.worker_stop.is_set() or not self.image_jobs.empty():
            try:kind,msg=self.image_jobs.get(timeout=.05)
            except queue.Empty:continue
            with self.state_lock:
                if self.reset_pending:self.reset();self.last_image=None;self.reset_pending=False
                try:
                    if kind=='raw':self.process_image(msg)
                    else:self.process_compressed(msg)
                except Exception as exc:
                    self.dropped+=1;self.get_logger().error('Image worker: '+str(exc))

    def stop_processing(self):
        self.worker_stop.set()
        if self.worker is not None:self.worker.join(timeout=8)
        if self.compute_worker is not None:self.compute_worker.close();self.compute_worker=None

    def on_compressed(self,msg):
        if self.async_processing:self.enqueue_image('compressed',msg)
        else:self.process_compressed(msg)

    def process_compressed(self, msg):
        frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self.dropped += 1
            return
        self.process(frame, msg.header)

    def on_image(self,msg):
        if self.async_processing:self.enqueue_image('raw',msg)
        else:self.process_image(msg)

    def process_image(self, msg):
        begin=time.perf_counter_ns()
        try:
            self.process(self.bridge.imgmsg_to_cv2(msg, 'bgr8'), msg.header)
        except (ValueError, cv2.error) as exc:
            self.dropped += 1
            self.get_logger().error(str(exc), throttle_duration_sec=5)
        finally:
            if self.performance_path:self.performance_rows.append(dict(stamp=seconds(msg.header.stamp),arrival_ns=begin,finish_ns=time.perf_counter_ns(),phases=getattr(self,'last_phases',{})))

    def display_points(self,points):
        if self.display_error<=0 or len(points)<3:return points
        return cv2.approxPolyDP(points.astype(np.float32),self.display_error,False).reshape(-1,2).astype(float)

    def save_performance(self):
        if self.performance_path:
            path=Path(self.performance_path);path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(dict(frames=len(self.performance_rows),records=self.performance_rows),separators=(',',':')))

    def process(self, frame, header):
        phase_start=time.perf_counter_ns()
        self.last_phases={}
        def phase(name):
            nonlocal phase_start
            now=time.perf_counter_ns()
            if self.performance_path:self.last_phases[name]=(now-phase_start)/1e6
            phase_start=now
        t = seconds(header.stamp)
        if self.last_image is not None and t <= self.last_image:
            self.dropped += 1
            return
        self.last_image = t; self.frames += 1
        pose = self.pose_at(t)
        if self.compute_worker is not None:
            if self.projector.source_size!=(frame.shape[1],frame.shape[0]):self.projector.initialize(frame.shape[1],frame.shape[0])
            phase('projection')
            result=self.compute_worker.compute(frame,t,pose)
            lanes=result['lanes'];binary=result['binary'];bev=result['bev']
            self.detector.boundary_lanes=result['boundaries'];self.detector.measured_count=result['measured'];self.detector.predicted_count=result['predicted']
            self.detector.registration=result['registration'];self.detector.detector.diagnostics=result['diagnostics']
            phase('detector')
        else:
            bev = self.projector.warp(frame)
            valid=self.projector.valid.copy()
            valid[self.projector.my>=frame.shape[0]*self.image_roi_bottom]=0
            bev=cv2.bitwise_and(bev,bev,mask=valid)
            phase('projection')
            lanes, binary = self.detector.process(bev, valid, t, pose)
            phase('detector')
        display_lanes=[replace(lane,points=self.display_points(lane.points)) for lane in lanes]
        self.local_pub.publish(self.array(display_lanes, header.stamp, self.base_frame, pose))
        self.publish_styles(lanes, header.stamp)
        phase('segments_styles')
        boundaries=[replace(lane,points=self.display_points(lane.points)) for lane in self.detector.boundary_lanes]
        self.paint_marker_pub.publish(self.markers(display_lanes,header.stamp,'observed_paint'))
        self.boundary_marker_pub.publish(self.markers(boundaries,header.stamp,'boundary_fit'))
        phase('markers')
        if self.debug and (self.debug_hz==0 or t-self.last_debug>=1/self.debug_hz-1e-8):
            self.last_debug=t
            scale=min(1.,self.overlay_max_width/frame.shape[1]) if self.overlay_max_width else 1.
            overlay=cv2.resize(frame,(round(frame.shape[1]*scale),round(frame.shape[0]*scale))) if scale<1 else frame.copy()
            label_candidates={}
            for lane_index,lane in enumerate(boundaries+display_lanes):
                pixels, visible = self.projector.image_points(lane.points)
                pixels*=scale
                safe=visible & np.isfinite(pixels).all(axis=1) & (np.max(np.abs(pixels),axis=1)<1e6)
                color=(255,220,0) if lane_index<len(boundaries) else ((0,255,255) if lane.reprojected else (0,255,0))
                if np.all(safe):cv2.polylines(overlay,[np.rint(pixels).astype(np.int32)],False,color,3)
                else:
                    edges=np.flatnonzero(np.r_[True,np.diff(safe.astype(int))!=0,True])
                    for lo,hi in zip(edges[:-1],edges[1:]):
                        if safe[lo] and hi-lo>=2:cv2.polylines(overlay,[np.rint(pixels[lo:hi]).astype(np.int32)],False,color,3)
                if lane_index<len(boundaries):continue
                if lane.side not in label_candidates or np.ptp(lane.points[:,0])>np.ptp(label_candidates[lane.side][0].points[:,0]):label_candidates[lane.side]=(lane,pixels)
            for lane,pixels in label_candidates.values():draw_style_label(overlay,lane,pixels)
            out = self.bridge.cv2_to_imgmsg(overlay, 'bgr8'); out.header = header; self.overlay_pub.publish(out)
            phase('overlay')
            self.publish_image(self.binary_pub, binary, header.stamp, 'mono8')
            c = self.detector.c
            for lane in display_lanes:
                pts = np.column_stack([(-lane.points[:, 1]-c.lateral_min)/c.mpp, (c.forward_max-lane.points[:, 0])/c.mpp]).astype('int32')
                cv2.polylines(bev, [pts], False, (0,255,255) if lane.reprojected else (0,255,0), 2)
            self.publish_image(self.bev_pub, bev, header.stamp, 'bgr8')
        phase('bev_debug')
        if not self.map_enabled:
            self.flush();return
        if len(self.pending) >= self.max_pending:
            self.pending.popleft(); self.dropped += 1
        self.pending.append((t, time.monotonic(), header.stamp, lanes))
        self.flush()

    def flush(self):
        while self.pending:
            t, arrival, stamp, lanes = self.pending[0]
            pose = self.pose_at(t)
            if pose is None and time.monotonic()-arrival < self.wait_sec:
                break
            self.pending.popleft()
            if pose is None or t <= self.last_commit:
                self.dropped += 1
                continue
            self.map.update(lanes, pose, t); self.last_commit = t
            if t-self.last_map_publish >= 1/max(0.1, self.map_hz):
                self.publish_map(stamp, pose)
                self.last_map_publish = t
        status = {'frames': self.frames, 'pending': len(self.pending), 'dropped': self.dropped,
                  'map_cells': len(self.map.cells), 'odom_generation': self.timeline.generation,
                  'last_map_capture_sec': self.last_commit if math.isfinite(self.last_commit) else None,
                  'observed_lanes': self.detector.measured_count, 'predicted_lanes': self.detector.predicted_count,
                  'temporal_registration': self.detector.registration,
                  'map_enabled': self.map_enabled,
                  'detector_diagnostics': getattr(self.detector.detector,'diagnostics',{})}
        # Status at one Hz, using steady time so paused bags do not flood logs.
        now = time.monotonic()
        if now-getattr(self, '_last_status', 0) > 1:
            self.status_pub.publish(String(data=json.dumps(status))); self._last_status = now

    def markers(self,lanes,stamp,namespace):
        msg=MarkerArray()
        clear=Marker();clear.header=Header(stamp=stamp,frame_id=self.base_frame);clear.action=Marker.DELETEALL
        msg.markers.append(clear)
        for ident,lane in enumerate(lanes):
            marker=Marker();marker.header=clear.header;marker.ns=namespace+'/'+lane.side;marker.id=ident
            marker.type=Marker.LINE_STRIP;marker.action=Marker.ADD;marker.pose.orientation.w=1.
            marker.scale.x=.05 if namespace=='observed_paint' else .025
            marker.color.a=.9;marker.color.g=1.
            marker.color.b=1. if namespace=='boundary_fit' else 0.
            marker.color.r=1. if lane.reprojected else 0.
            marker.lifetime.sec=0;marker.lifetime.nanosec=300000000
            marker.points=[Point(x=float(p[0]),y=float(p[1]),z=.02) for p in lane.points]
            msg.markers.append(marker)
        return msg

    def segment(self, a, b, stamp, side, confidence, ident, hits=1):
        seg = MarkingSegment()
        seg.track_id = ident; seg.camera_id = 'front'; seg.family = side
        seg.start = Point(x=float(a[0]), y=float(a[1]), z=0.0)
        seg.end = Point(x=float(b[0]), y=float(b[1]), z=0.0)
        seg.angle_rad = math.atan2(b[1]-a[1], b[0]-a[0]); seg.span_m = float(np.linalg.norm(b-a))
        seg.confidence = confidence; seg.hits_total = hits; seg.hits_in_window = hits
        seg.confirmed = hits >= 2; seg.reprojected = False; seg.source_stamp = stamp
        return seg

    def array(self, lanes, stamp, frame, pose=None):
        msg = MarkingSegmentArray(); msg.header = Header(stamp=stamp, frame_id=frame); msg.camera_id = 'front'
        msg.pose_valid = pose is not None
        if pose is not None:
            msg.observer_pose = Pose2D(x=float(pose[0]), y=float(pose[1]), theta=float(pose[2]))
        for lane in lanes:
            for a, b in zip(lane.points[:-1], lane.points[1:]):
                source = stamp
                if lane.source_time > 0:
                    ns=round(lane.source_time*1e9)
                    source=type(stamp)(sec=ns//1000000000,nanosec=ns%1000000000)
                segment=self.segment(a,b,source,lane.side,lane.confidence,len(msg.segments)+1,lane.hits)
                segment.reprojected=lane.reprojected
                msg.segments.append(segment)
        return msg

    def publish_styles(self,lanes,stamp):
        msg=LaneStyleArray();msg.header=Header(stamp=stamp,frame_id=self.base_frame);msg.camera_id='front'
        kinds={'unknown':LaneStyle.UNKNOWN,'solid':LaneStyle.SOLID,'dashed':LaneStyle.DASHED}
        unique={}
        for lane in lanes:
            if lane.side not in unique or lane.style_confidence>unique[lane.side].style_confidence:unique[lane.side]=lane
        for lane in unique.values():
            style=LaneStyle();style.side=lane.side;style.style=kinds[lane.style]
            style.confidence=float(lane.style_confidence);style.observed_span_m=float(lane.style_span_m)
            style.paint_fraction=float(lane.paint_fraction);style.visible_dash_count=int(lane.dash_count)
            style.reprojected=lane.reprojected
            ns=round(lane.source_time*1e9)
            style.source_stamp=type(stamp)(sec=ns//1000000000,nanosec=ns%1000000000)
            ns=round(lane.style_source_time*1e9)
            style.style_source_stamp=type(stamp)(sec=ns//1000000000,nanosec=ns%1000000000)
            msg.lanes.append(style)
        self.style_pub.publish(msg)

    def publish_image(self, pub, image, stamp, encoding):
        msg = self.bridge.cv2_to_imgmsg(image, encoding)
        msg.header = Header(stamp=stamp, frame_id=self.base_frame); pub.publish(msg)

    def publish_map(self, stamp, pose):
        msg = self.array([], stamp, self.world_frame, pose)
        grid = OccupancyGrid(); grid.header = msg.header
        r = self.map.resolution
        size = math.ceil(2*self.map.radius/r)
        origin = np.floor((np.asarray(pose[:2])-self.map.radius)/r)*r
        grid.info.resolution = r; grid.info.width = size; grid.info.height = size
        grid.info.origin.position.x = float(origin[0]); grid.info.origin.position.y = float(origin[1]); grid.info.origin.orientation.w = 1.0
        raster = np.full((size, size), -1, dtype='int8')
        for ident, (pt, source, hits, confidence, direction) in enumerate(self.map.cells.values(), 1):
            source_stamp = type(stamp)(sec=int(source), nanosec=int(round((source-int(source))*1e9)))
            if source_stamp.nanosec >= 1000000000:
                source_stamp.sec += 1; source_stamp.nanosec = 0
            msg.segments.append(self.segment(pt-direction*r/2, pt+direction*r/2, source_stamp, 'road', confidence, ident, hits))
            ix, iy = np.floor((pt-origin)/r).astype(int)
            if 0 <= ix < size and 0 <= iy < size:
                raster[iy, ix] = min(100, 40+hits*15)
        self.global_pub.publish(msg)
        grid.data = raster.ravel().tolist(); self.grid_pub.publish(grid)
        image = np.where(raster[::-1] >= 0, 255, 0).astype('uint8')
        out = self.bridge.cv2_to_imgmsg(image, 'mono8'); out.header = msg.header; self.map_pub.publish(out)


def main():
    rclpy.init()
    node = RoadNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_processing();node.save_performance();node.destroy_node(); rclpy.try_shutdown()
