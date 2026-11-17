#!/usr/bin/env python3
import json
import struct
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
from rclpy.serialization import deserialize_message
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

MARKING_COLORS=dict(stop_line=(0,0,255),crosswalk=(255,0,255),marking=(0,200,255))  # BGR overlay colours


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
        # Several Python threads share this process (image callback, detection hand-off, outputs): a thread
        # waiting for the interpreter lock otherwise waits up to 5 ms, which showed up as latency.
        switch=param('gil_switch_interval_ms',0.0)
        if switch>0:import sys;sys.setswitchinterval(switch/1000.)
        # 1 keeps only the newest image waiting: a slow frame drops the older one instead of delaying all later frames.
        self.image_jobs=queue.Queue(maxsize=param('image_job_depth',10));self.worker_stop=threading.Event();self.worker=None;self.reset_pending=False
        self.timeline = Timeline(param('odom_max_gap_sec', 0.25))
        self.map = LaneMap(param('map_resolution_m', 0.10), param('map_radius_m', 80.0), param('map_ttl_sec', 120.0), param('map_max_cells', 100000))
        if min(self.map.resolution, self.map.radius, self.map.ttl, self.map.max_cells, self.timeline.max_gap) <= 0:
            raise ValueError('Map and synchronization limits must be positive')
        self.wait_sec = param('pose_wait_sec', 0.5)
        self.max_pending = param('max_pending_frames', 30)
        if self.max_pending < 1 or self.wait_sec <= 0:
            raise ValueError('Pending queue and wait duration must be positive')
        self.performance_path=param('performance_report','')
        self.performance_rows=[];self.received={}  # header stamp -> wall time the image callback ran
        self.display_error=param('display_simplify_error_m',0.0)
        if not math.isfinite(self.display_error) or not 0<=self.display_error<=.05:raise ValueError('Invalid display simplification bound')
        self.overlay_max_width=param('overlay_max_width',0)
        if self.overlay_max_width<0:raise ValueError('Invalid overlay width')
        self.debug_hz=param('debug_publish_hz',0.0);self.last_debug=-math.inf
        if self.debug_hz<0:raise ValueError('Invalid debug image rate')
        self.debug = param('debug_images', True)
        self.debug_jobs=queue.Queue(maxsize=1) if param('debug_thread',False) else None
        # output_thread: lane topics, the inline display and debug images are published by their own thread.
        self.output_jobs=queue.Queue(maxsize=2) if param('output_thread',False) else None
        self.output_lock=threading.Lock();self.output_rows={}
        self.map_hz = param('map_publish_hz', 2.0)
        self.world_frame = param('world_frame', 'map')
        self.base_frame = param('base_frame', 'base_link')
        self.compute_worker=None
        # Optional learned lane mask (gates paint when carla_learned_gate is set) and road markings.
        from .rt_worker import Extras,LearnedMask
        extras=dict(markings=param('road_markings',False))
        weights=param('learned_weights','') if backend=='carla' else ''
        if weights:self.projector.initialize(*self.projector.size)
        self.learned=LearnedMask(self.projector,weights,param('learned_every',1))
        self.get_logger().info('Learned lane mask: '+self.learned.status)
        # display_filter_inline: the HUD display filter (lane_display_filter) runs in this node, with the settings
        # of carla.yaml's lane_display_filter section; one ROS hop less.  display_filter_in_worker: in the
        # perception worker process, next to the detector, away from this process's ROS threads.
        self.display_settings=None
        if param('display_filter_inline',False):
            from .display_filter import settings_from
            section=yaml.safe_load(open(param('display_filter_config',get_package_share_directory('vision_road')+'/config/carla.yaml')))
            self.display_settings=settings_from((section.get('lane_display_filter') or {}).get('ros__parameters',{}))
        in_worker=param('display_filter_in_worker',False) and self.display_settings is not None
        # ... and display_publish_in_worker: the worker publishes the display topics itself, as soon as it has them.
        publish_in_worker=in_worker and param('display_publish_in_worker',False)
        # image_in_worker: the perception worker subscribes the camera and odometry itself (see rt_worker.ros_loop).
        isolated=param('isolated_compute',False)
        # learned_in_worker: with learned weights too, the worker takes the camera itself and hands the frames to
        # the model process (started here); without it the learned mask keeps the road_node image path.
        self.learned_in_worker=bool(weights) and self.learned.model is not None and param('learned_in_worker',False)
        self.image_in_worker=isolated and param('image_in_worker',False) and (not weights or self.learned_in_worker)
        self.worker_display=(self.display_settings,self.display_error,publish_in_worker,self.base_frame) if in_worker else None
        if self.image_in_worker:
            self.compute_worker=None;self.extras=None;self.worker_extras=extras  # started with the subscriptions below
        elif isolated:
            from .rt_worker import ComputeWorker
            self.projector.initialize(*self.projector.size)
            self.compute_worker=ComputeWorker(self.projector,cfg,backend,self.image_roi_bottom,threads,extras,
                                              display=self.worker_display)
            self.extras=None
        else:
            self.extras=Extras(**extras) if backend=='carla' and extras['markings'] else None
        self.pending = deque(); self.last_image = None; self.last_commit = -math.inf
        self.last_map_publish = -math.inf; self.dropped = 0; self.frames = 0
        self.bridge = CvBridge()
        self.local_pub = self.create_publisher(MarkingSegmentArray, '/parking/cpp/marking_segments', 10)
        self.paint_marker_pub=self.create_publisher(MarkerArray,'/road/lane_markers',10)
        self.boundary_marker_pub=self.create_publisher(MarkerArray,'/road/lane_boundary_markers',10)
        self.style_pub = self.create_publisher(LaneStyleArray, '/road/lane_styles', 10)
        self.marking_pub=self.create_publisher(MarkerArray,'/road/road_markings',10)
        self.global_pub = self.create_publisher(MarkingSegmentArray, '/parking/cpp/global_marking_segments_fused', 2)
        self.map_pub = self.create_publisher(Image, '/parking/cpp/global_marking_map', 2)
        self.grid_pub = self.create_publisher(OccupancyGrid, '/road/lane_grid', 2)
        self.status_pub = self.create_publisher(String, '/parking/cpp/status', 10)
        self.binary_pub = self.create_publisher(Image, '/parking/cpp/binary', qos_profile_sensor_data)
        self.bev_pub = self.create_publisher(Image, '/parking/cpp/vector_bev', qos_profile_sensor_data)
        self.overlay_pub = self.create_publisher(Image, '/road/overlay', qos_profile_sensor_data)
        # carla_outer_enabled / carla_centre_enabled: the neighbouring lanes' outer lines and the lane centrelines,
        # apart from the two own lines the HUD and display filter use (side_topics.py).  With image_in_worker
        # the perception worker publishes them after its main result; this node keeps the latest for its overlay.
        outer=bool(getattr(self.detector.c,'carla_outer_enabled',False));centre=bool(getattr(self.detector.c,'carla_centre_enabled',False))
        self.side_config=(outer,centre,self.display_error)
        self.side=None;self.latest_side=dict(outer=[],centres=[])
        if (outer or centre) and not self.image_in_worker:
            from .side_topics import SideTopics
            self.side=SideTopics(self,self.base_frame,self.debug_hz if self.debug else 0.,self.display_error,outer,centre)
        self.display_stage=None
        if self.display_settings is not None:
            from .display_filter import DisplayStage
            self.display_stage=DisplayStage(self.display_settings,self.base_frame)  # in the worker: messages only
            if not (publish_in_worker and (self.compute_worker is not None or self.image_in_worker)):
                self.display_pub=self.create_publisher(MarkerArray,'/road/lane_display_markers',10)
                self.display_style_pub=self.create_publisher(LaneStyleArray,'/road/lane_display_styles',10)
            self.get_logger().info('Display filter '+('in the perception worker'+(', published there' if publish_in_worker else '') if in_worker and (self.compute_worker is not None or self.image_in_worker) else 'inline')+
                                   ': '+str(self.display_settings))
        self.odom_topic=param('odom_topic','/localization/kinematic_state')
        image_topic = param('image_topic', '/sensing/camera/front/fisheye/image_raw')
        compressed = param('compressed', False)
        reliability = param('image_reliability', 'best_effort')
        depth = param('image_queue_depth', 2)
        # A row-cropped image (only the rows the BEV reads, e.g. tools/replay_maploop.py --crop-rows): the camera's
        # CameraInfo keeps the full image size and gives the rows in roi (y_offset, height).  Full images need no
        # CameraInfo.  Default: <image topic's namespace>/camera_info; '' turns row crops off.
        self.camera_info_topic=param('camera_info_topic',image_topic.rsplit('/',1)[0]+'/camera_info')
        self.image_roi=None;self.crop_skipped=0
        if reliability not in ('reliable', 'best_effort') or depth < 1:
            raise ValueError('Invalid image QoS settings')
        image_qos = QoSProfile(depth=depth, reliability=ReliabilityPolicy.RELIABLE if reliability=='reliable' else ReliabilityPolicy.BEST_EFFORT)
        # raw_image_cdr: take the serialized image and view its pixel array in place; building a Python
        # Image (6 MB) and converting it costs several milliseconds per frame.
        self.raw_cdr=param('raw_image_cdr',False) and not compressed;self.raw_shape=None
        if self.image_in_worker:
            from .rt_worker import RosWorker
            self.projector.initialize(*self.projector.size)
            self.compute_worker=RosWorker(self.projector,self.detector.c,backend,self.image_roi_bottom,threads,self.worker_extras,self.worker_display,
                dict(image_topic=image_topic,reliability=reliability,odom_topic=self.odom_topic,camera_info_topic=self.camera_info_topic,
                     world_frame=self.world_frame,base_frame=self.base_frame,odom_max_gap_sec=self.timeline.max_gap,
                     debug_hz=self.debug_hz if self.debug else 0.,overlay_max_width=self.overlay_max_width,side=self.side_config,
                     learned=self.learned.model.handle() if self.learned_in_worker else None,learned_every=self.learned.every))
            self.result_worker=threading.Thread(target=self.take_results,name='road-perception-results',daemon=True);self.result_worker.start()
        elif self.raw_cdr:self.create_subscription(Image,image_topic,self.on_image_cdr,image_qos,raw=True)
        else:self.create_subscription(CompressedImage if compressed else Image, image_topic, self.on_compressed if compressed else self.on_image, image_qos)
        if not self.image_in_worker and self.camera_info_topic:
            from sensor_msgs.msg import CameraInfo
            def on_info(msg):
                r=msg.roi;self.image_roi=(int(r.y_offset),int(r.height),int(r.x_offset),int(r.width)) if r.height else None
            self.create_subscription(CameraInfo,self.camera_info_topic,on_info,QoSProfile(depth=5,reliability=ReliabilityPolicy.BEST_EFFORT))
        # Best effort accepts both reliable and best-effort odometry publishers.
        self.create_subscription(Odometry, self.odom_topic, self.on_odom, QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_timer(0.02, self.timer_flush, clock=Clock(clock_type=ClockType.STEADY_TIME))
        if self.async_processing:
            self.worker=threading.Thread(target=self.work_images,name='road-image-worker',daemon=True);self.worker.start()
        if self.debug_jobs is not None:
            self.debug_worker=threading.Thread(target=self.debug_loop,name='road-debug-images',daemon=True);self.debug_worker.start()
        if self.output_jobs is not None:
            self.output_worker=threading.Thread(target=self.output_loop,name='road-lane-outputs',daemon=True);self.output_worker.start()
        self.get_logger().info('Road detector ready: '+backend+'; image processing only, no learned model')

    def reset(self):
        self.pending.clear(); self.map.cells.clear(); self.detector.reset()
        if self.compute_worker is not None:self.compute_worker.reset_pending=True
        if getattr(self,'extras',None) is not None:self.extras.reset()
        if getattr(self,'learned',None) is not None:self.learned.reset()
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
                    elif kind=='frame':self.process_frame(*msg)
                    else:self.process_compressed(msg)
                except Exception as exc:
                    self.dropped+=1;self.get_logger().error('Image worker: '+str(exc))

    def stop_processing(self):
        self.worker_stop.set()
        if self.worker is not None:self.worker.join(timeout=8)
        if self.output_jobs is not None:self.output_worker.join(timeout=4)
        if self.compute_worker is not None:self.compute_worker.close();self.compute_worker=None
        self.learned.close()

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
        if self.performance_path:self.received[(msg.header.stamp.sec,msg.header.stamp.nanosec)]=time.perf_counter_ns()
        if self.async_processing:self.enqueue_image('raw',msg)
        else:self.process_image(msg)

    def on_image_cdr(self,raw):
        received=time.perf_counter_ns()
        if self.raw_shape is None:
            first=deserialize_message(raw,Image)
            if first.encoding not in ('bgr8','rgb8') or first.step!=first.width*3 or first.is_bigendian:
                raise ValueError('raw_image_cdr needs packed bgr8/rgb8 images')
            self.raw_shape=(first.height,first.width,first.encoding,first.header.frame_id)
        h,w,encoding,frame_id=self.raw_shape
        # CDR: 4-byte encapsulation, then the header stamp; the pixel array is the last field.
        sec,nanosec=struct.unpack_from('<iI',raw,4)
        header=Header(frame_id=frame_id);header.stamp.sec=sec;header.stamp.nanosec=nanosec
        if self.performance_path:self.received[(sec,nanosec)]=received
        frame=np.frombuffer(raw,np.uint8,h*w*3,len(raw)-h*w*3).reshape(h,w,3)
        if self.async_processing:self.enqueue_image('frame',(frame,encoding,header))
        else:self.process_frame(frame,encoding,header)

    def process_frame(self,frame,encoding,header):
        begin=time.perf_counter_ns()
        try:self.process(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR) if encoding=='rgb8' else frame,header)
        except (ValueError,cv2.error) as exc:
            self.dropped+=1
            self.get_logger().error(str(exc),throttle_duration_sec=5)
        finally:
            if self.performance_path:self.performance_rows.append(dict(stamp=seconds(header.stamp),receive_ns=self.received.pop((header.stamp.sec,header.stamp.nanosec),None),
                                                                    arrival_ns=begin,finish_ns=time.perf_counter_ns(),phases=getattr(self,'last_phases',{})))

    def process_image(self, msg):
        begin=time.perf_counter_ns()
        try:
            self.process(self.bridge.imgmsg_to_cv2(msg, 'bgr8'), msg.header)
        except (ValueError, cv2.error) as exc:
            self.dropped += 1
            self.get_logger().error(str(exc), throttle_duration_sec=5)
        finally:
            if self.performance_path:self.performance_rows.append(dict(stamp=seconds(msg.header.stamp),receive_ns=self.received.pop((msg.header.stamp.sec,msg.header.stamp.nanosec),None),
                                                                    arrival_ns=begin,finish_ns=time.perf_counter_ns(),phases=getattr(self,'last_phases',{})))

    def display_points(self,points):
        from .display_filter import simplify
        return simplify(points,self.display_error)

    def save_performance(self):
        if self.performance_path:
            path=Path(self.performance_path);path.parent.mkdir(parents=True,exist_ok=True)
            # Images received but never processed (dropped from the job queue) keep only their receive time.
            unprocessed=[dict(stamp=a+b*1e-9,receive_ns=v) for (a,b),v in self.received.items()]
            for row in self.performance_rows:
                ns=round(row['stamp']*1e9);row.update(self.output_rows.get((ns//1000000000,ns%1000000000),{}))
            path.write_text(json.dumps(dict(frames=len(self.performance_rows),records=self.performance_rows,unprocessed=unprocessed),separators=(',',':')))

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
        top=self.crop_top(frame)
        if top is None:
            self.dropped+=1;self.crop_skipped+=1
            self.get_logger().warning(f'Skipping a {frame.shape[1]}x{frame.shape[0]} image: not the full image and no CameraInfo roi placing '
                                      f'the rows the BEV reads inside it ({self.crop_skipped} so far)',throttle_duration_sec=5)
            return
        self.last_image = t; self.frames += 1
        pose = self.pose_at(t)
        if self.compute_worker is not None:
            if top is False and self.projector.source_size!=(frame.shape[1],frame.shape[0]):self.projector.initialize(frame.shape[1],frame.shape[0])
            phase('projection')
            learned=self.learned.next(t,frame if top is False else self.full_frame(frame,top))
            phase('learned_mask')
            result=self.compute_worker.compute(frame,t,pose,learned,top=top or 0)
            lanes=result['lanes'];binary=result['binary'];bev=result['bev'];markings=result.get('markings',[])
            self.detector.boundary_lanes=result['boundaries'];self.detector.measured_count=result['measured'];self.detector.predicted_count=result['predicted']
            self.detector.registration=result['registration'];self.detector.detector.diagnostics=result['diagnostics']
            precomputed={k:result[k] for k in ('display','display_lanes','display_boundaries','display_published','outer','centres') if k in result}
            phase('detector')
            if self.performance_path and 'worker_ms' in result:self.last_phases.update({'worker.'+k:v for k,v in result['worker_ms'].items()})
        else:
            bev = self.projector.warp(frame,top or 0)
            valid=self.projector.valid.copy()
            valid[self.projector.my>=self.projector.source_size[1]*self.image_roi_bottom]=0
            bev=cv2.bitwise_and(bev,bev,mask=valid)
            phase('projection')
            if isinstance(self.detector.detector,CarlaDetector):self.detector.detector.learned=self.learned.next(t,frame if top is False else self.full_frame(frame,top))
            lanes, binary = self.detector.process(bev, valid, t, pose)
            markings=self.extras.after(self.detector,bev,valid,pose) if self.extras is not None else []
            precomputed=dict(outer=list(getattr(self.detector,'outer_lanes',[])),centres=list(getattr(self.detector,'centre_lanes',[])))
            phase('detector')
        if top is not False and self.debug:frame=self.full_frame(frame,top)  # the overlay draws on the full image
        job=(frame,header,t,pose,lanes,self.detector.boundary_lanes,binary,bev,markings,precomputed)
        if self.output_jobs is not None:
            # The next frame's detection overlaps these outputs; frames stay in order and none is skipped.
            self.output_jobs.put(job);phase('handoff')
        else:self.publish_outputs(*job,phase)

    def crop_top(self,frame):
        """False: a full image (or another size, as before); an int: the full-image row of this row-cropped
        frame's first row (CameraInfo roi); None: a smaller image that cannot be placed."""
        h,w=frame.shape[:2];size=self.projector.source_size or self.projector.size
        if (w,h)==tuple(size) or w!=size[0] or h>=size[1]:return False
        roi=self.image_roi
        if self.projector.source_size is None:self.projector.initialize(*self.projector.size)
        if roi is None or roi[1]!=h or roi[2]!=0 or roi[3] not in (0,w):return None
        lo,hi=self.projector.rows(self.image_roi_bottom)
        return roi[0] if roi[0]<=lo and roi[0]+h>=hi else None

    def full_frame(self,frame,top):
        """The full image with only the cropped rows (black elsewhere), for the learned mask model; its mask
        then differs from one made from the full image above and below the rows (not measured)."""
        w,h=self.projector.source_size;full=np.zeros((h,w,3),np.uint8);full[top:top+frame.shape[0]]=frame;return full

    def take_results(self):
        """image_in_worker: each processed frame's result from the perception worker, in order."""
        connection=self.compute_worker.connection
        while not self.worker_stop.is_set():
            try:
                if not connection.poll(.05):continue
                result=connection.recv()
            except (EOFError,OSError):break
            if 'error' in result:self.dropped+=1;self.get_logger().error('Perception worker: '+result['error'],throttle_duration_sec=5);continue
            if 'side' in result:
                # Outer lines and centrelines (already published by the worker), for the overlay; their time.
                self.latest_side=result['side']
                if self.performance_path and self.performance_rows:self.performance_rows[-1]['phases']['worker.extra']=result['extra_ms']
                continue
            sec,nanosec=result['stamp'];header=Header(frame_id=result['frame_id']);header.stamp.sec=sec;header.stamp.nanosec=nanosec
            t=sec+nanosec*1e-9
            with self.state_lock:
                self.last_image=t;self.frames+=1
                self.detector.boundary_lanes=result['boundaries'];self.detector.measured_count=result['measured'];self.detector.predicted_count=result['predicted']
                self.detector.registration=result['registration'];self.detector.detector.diagnostics=result['diagnostics']
                precomputed={k:result[k] for k in ('display','display_lanes','display_boundaries','display_published','debug_frame','debug_scale','outer','centres') if k in result}
                job=(None,header,t,result['pose'],result['lanes'],result['boundaries'],result['binary'],result['bev'],result['markings'],precomputed)
                if self.output_jobs is not None:self.output_jobs.put(job)
                else:self.publish_outputs(*job,lambda name:None)
            if self.performance_path:
                self.performance_rows.append(dict(stamp=t,receive_ns=result['receive_ns'],arrival_ns=result['start_ns'],finish_ns=result['computed_ns'],
                                                  phases={'worker.'+k:v for k,v in result['worker_ms'].items()}))

    def output_loop(self):
        while not self.worker_stop.is_set() or not self.output_jobs.empty():
            try:job=self.output_jobs.get(timeout=.05)
            except queue.Empty:continue
            start=time.perf_counter_ns();phases={};mark=[start]
            def phase(name):
                now=time.perf_counter_ns();phases[name]=(now-mark[0])/1e6;mark[0]=now
            try:
                with self.output_lock:self.publish_outputs(*job,phase)
            except Exception as exc:self.get_logger().error('Lane outputs: '+str(exc),throttle_duration_sec=5)
            if self.performance_path:
                stamp=job[1].stamp;self.output_rows[(stamp.sec,stamp.nanosec)]=dict(output_start_ns=start,output_finish_ns=time.perf_counter_ns(),output_phases=phases)

    def publish_outputs(self,frame,header,t,pose,lanes,boundary_lanes,binary,bev,markings,precomputed,phase):
        if 'display_lanes' in precomputed:display_lanes,boundaries=precomputed['display_lanes'],precomputed['display_boundaries']
        else:
            display_lanes=[replace(lane,points=self.display_points(lane.points)) for lane in lanes]
            boundaries=[replace(lane,points=self.display_points(lane.points)) for lane in boundary_lanes]
        style_msg=self.style_message(lanes,header.stamp)
        # What the HUD draws goes out first: the display line when the filter runs here (fed the same
        # lanes as the markers below carry), then paint, styles and boundaries; segments and markings after.
        if precomputed.get('display_published'):pass  # the worker has published the display topics
        elif 'display' in precomputed:
            shown=self.display_stage.messages(header.stamp,precomputed['display'])
            self.display_pub.publish(shown[0]);self.display_style_pub.publish(shown[1]);phase('display_filter')
        elif self.display_stage is not None:
            self.display_stage.on_styles(style_msg)
            shown=self.display_stage.on_lanes(header.stamp,[(l.side,l.points,l.reprojected) for l in boundaries if len(l.points)>=2],
                                              [(l.side,l.points) for l in display_lanes if not l.reprojected and len(l.points)>=2],self.pose_at)
            if shown is not None:self.display_pub.publish(shown[0]);self.display_style_pub.publish(shown[1])
            phase('display_filter')
        self.paint_marker_pub.publish(self.markers(display_lanes,header.stamp,'observed_paint'));self.style_pub.publish(style_msg)
        self.boundary_marker_pub.publish(self.markers(boundaries,header.stamp,'boundary_fit'))
        # The worker's side outputs (image_in_worker) come after the main result: the overlay draws the latest.
        outer=precomputed.get('outer',self.latest_side['outer']);centres=precomputed.get('centres',self.latest_side['centres'])
        if self.side is not None:self.side.publish(header.stamp,t,outer,centres)
        phase('markers')
        self.local_pub.publish(self.array(display_lanes, header.stamp, self.base_frame, pose))
        self.marking_pub.publish(self.road_marking_markers(markings,header.stamp))
        phase('segments_markings')
        # image_in_worker: the worker sends a frame (at the overlay size) with the frames to draw.
        debug_frame,debug_scale=precomputed.get('debug_frame'),precomputed.get('debug_scale')
        if debug_frame is not None or (frame is not None and self.debug and (self.debug_hz==0 or t-self.last_debug>=1/self.debug_hz-1e-8)):
            self.last_debug=t
            if debug_frame is not None:frame=debug_frame
            if self.debug_jobs is not None:
                # Drawing waits for the next frame's lanes otherwise; a busy drawer skips this one.
                try:self.debug_jobs.put_nowait((frame,header,boundaries,display_lanes,markings,binary,bev,debug_scale,outer,centres))
                except queue.Full:pass
            else:self.draw_debug(frame,header,boundaries,display_lanes,markings,binary,bev,debug_scale,phase,outer,centres)
        phase('bev_debug')
        if not self.map_enabled:
            self.flush();return
        if len(self.pending) >= self.max_pending:
            self.pending.popleft(); self.dropped += 1
        self.pending.append((t, time.monotonic(), header.stamp, lanes))
        self.flush()

    def debug_loop(self):
        while not self.worker_stop.is_set():
            try:job=self.debug_jobs.get(timeout=.1)
            except queue.Empty:continue
            try:self.draw_debug(*job[:8],phase=lambda name:None,outer=job[8],centres=job[9])
            except Exception as exc:self.get_logger().error('Debug images: '+str(exc),throttle_duration_sec=5)

    def draw_debug(self,frame,header,boundaries,display_lanes,markings,binary,bev,given_scale=None,phase=lambda name:None,outer=(),centres=()):
        """given_scale: the frame is already reduced by it (image_in_worker).  outer: neighbouring lane
        outer lines, magenta (carried over: dark magenta), labelled with style, colour and OPP.  centres: lane
        centrelines, orange dashes (estimated from one boundary: thin)."""
        if given_scale is not None:scale=given_scale;overlay=frame
        else:
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
        for lane in outer:
            pixels,visible=self.projector.image_points(lane.points);pixels*=scale
            safe=visible&np.isfinite(pixels).all(axis=1)&(np.max(np.abs(pixels),axis=1)<1e6)
            color=(110,0,110) if lane.reprojected else (255,0,255)
            edges=np.flatnonzero(np.r_[True,np.diff(safe.astype(int))!=0,True])
            for lo,hi in zip(edges[:-1],edges[1:]):
                if safe[lo] and hi-lo>=2:cv2.polylines(overlay,[np.rint(pixels[lo:hi]).astype(np.int32)],False,color,3)
            if safe.any():
                anchor=pixels[safe][len(pixels[safe])//3];info=getattr(lane,'outer',{})
                text=('L+1' if lane.side.startswith('left') else 'R+1')+':'+lane.style.upper()+(' Y' if info.get('yellow') else '')+(' OPP' if info.get('opposite') else '')
                size,_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,.55,2)
                x=int(anchor[0])-(size[0]+12 if lane.side.startswith('left') else -12);y=int(anchor[1])-10
                x=max(2,min(overlay.shape[1]-size[0]-4,x));y=max(size[1]+4,min(overlay.shape[0]-4,y))
                cv2.rectangle(overlay,(x-2,y-size[1]-3),(x+size[0]+2,y+3),(0,0,0),-1)
                cv2.putText(overlay,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,0,255),2)
        for lane in centres:
            p=np.asarray(lane['points'],float)
            if len(p)<2:continue
            x=np.arange(p[0,0],p[-1,0]+1e-9,1.);p=np.column_stack([x,np.interp(x,p[:,0],p[:,1])])  # 1 m dashes
            pixels,visible=self.projector.image_points(p);pixels*=scale
            safe=visible&np.isfinite(pixels).all(axis=1)&(np.max(np.abs(pixels),axis=1)<1e6)
            for k in range(0,len(pixels)-1,2):
                if safe[k] and safe[k+1]:cv2.line(overlay,tuple(np.rint(pixels[k]).astype(int)),tuple(np.rint(pixels[k+1]).astype(int)),(0,140,255),1 if lane['estimated'] else 3)
        for mark in markings:
            outline=mark['points'] if mark['kind']=='stop_line' else np.vstack([mark['points'],mark['points'][:1]])
            dense=np.vstack([np.linspace(a,b,8) for a,b in zip(outline[:-1],outline[1:])])
            pixels,visible=self.projector.image_points(dense);pixels*=scale
            if visible.sum()>=2:
                color=MARKING_COLORS[mark['kind']]
                cv2.polylines(overlay,[np.rint(pixels[visible]).astype(np.int32)],False,color,4 if mark['kind']=='stop_line' else 2)
        out = self.bridge.cv2_to_imgmsg(overlay, 'bgr8'); out.header = header; self.overlay_pub.publish(out)
        phase('overlay')
        self.publish_image(self.binary_pub, binary, header.stamp, 'mono8')
        c = self.detector.c
        for lane in display_lanes:
            pts = np.column_stack([(-lane.points[:, 1]-c.lateral_min)/c.mpp, (c.forward_max-lane.points[:, 0])/c.mpp]).astype('int32')
            cv2.polylines(bev, [pts], False, (0,255,255) if lane.reprojected else (0,255,0), 2)
        self.publish_image(self.bev_pub, bev, header.stamp, 'bgr8')

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

    def road_marking_markers(self,markings,stamp):
        """Stop lines as a 2-point strip, crosswalks and other markings as a closed outline, base_link.
        ns is the kind; color.r=1 marks a confirmed marking not seen this frame."""
        msg=MarkerArray()
        clear=Marker();clear.header=Header(stamp=stamp,frame_id=self.base_frame);clear.action=Marker.DELETEALL
        msg.markers.append(clear)
        for mark in markings:
            marker=Marker();marker.header=clear.header;marker.ns=mark['kind'];marker.id=int(mark['id'])
            marker.type=Marker.LINE_STRIP;marker.action=Marker.ADD;marker.pose.orientation.w=1.
            marker.scale.x=.3 if mark['kind']=='stop_line' else .1
            marker.color.a=.9;marker.color.g=.5;marker.color.b=1.;marker.color.r=1. if mark['predicted'] else 0.
            marker.lifetime.nanosec=300000000
            points=mark['points'] if mark['kind']=='stop_line' else np.vstack([mark['points'],mark['points'][:1]])
            marker.points=[Point(x=float(p[0]),y=float(p[1]),z=.02) for p in points]
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
        self.style_pub.publish(self.style_message(lanes,stamp))

    def style_message(self,lanes,stamp):
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
        return msg

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
