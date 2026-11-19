"""Bounded shared-image worker: perception has its own Python interpreter/GIL."""
import multiprocessing as mp
import time
from multiprocessing import shared_memory
import cv2
import numpy as np
from .temporal import TemporalDetector
from .carla import CarlaDetector,clone
from .classical import PaintEvidenceDetector


class LearnedMask:
    """Optional learned lane mask from its own process, typically one frame behind.

    Without weights, torch or CUDA it stays off and the detector runs unchanged."""

    def __init__(self,projector,weights='',every=1,max_age=.2,client=None):
        # every=2 runs the model on every other frame: the mask is then 1-2 frames old,
        # which gated maploop as well, and the model's CPU share halves.
        # client: a LaneMaskProcess.handle() from road_node (image_in_worker: the worker submits the frames).
        self.model=None;self.status='off';self.max_age=max_age;self.every=max(1,int(every));self.count=0
        if client is not None:
            from .learned_mask import LaneMaskClient
            self.model=LaneMaskClient(client);self.status='on (in the perception worker)'
        elif weights:
            try:
                from .learned_mask import LaneMaskProcess
                self.model=LaneMaskProcess(weights,projector);self.status='on'
            except Exception as exc:self.status='unavailable: '+str(exc)

    def next(self,stamp,frame,top=None):
        """The newest finished mask no older than max_age, then queue this frame (top: a row crop's first row)."""
        if self.model is None:return None
        latest=self.model.latest();self.count+=1
        if self.count%self.every==0:self.model.submit(stamp,frame,top)
        return latest[1] if latest is not None and 0<=stamp-latest[0]<=self.max_age else None

    def reset(self):
        if self.model is not None:self.model.reset()

    def close(self):
        if self.model is not None:self.model.close()


class Extras:
    """Road markings confirmed over frames; uses the learned mask when there is one."""

    def __init__(self,markings=False):
        self.tracker=None
        if markings:
            from . import road_markings
            self.markings=road_markings;self.tracker=road_markings.MarkingTracker()

    def reset(self):
        if self.tracker is not None:self.tracker.reset()

    def after(self,detector,bev,valid,pose):
        if self.tracker is None:return []
        candidates=self.markings.detect(detector.detector,bev,valid,detector.boundary_lanes,getattr(detector.detector,'learned',None))
        return self.tracker.update(pose,candidates)



class Perception:
    """One process's perception state: projector, detector, optional road markings and display filter.
    display: (FilterSettings, display simplification error, publish, frame); with publish the display topics go
    out from this process (it then runs rclpy), before the result travels back to the node."""

    def __init__(self,projector,cfg,backend,roi,threads,extras=None,display=None,ros=False):
        cv2.setNumThreads(threads)
        projector.initialize(*projector.size);self.projector=projector;self.roi=roi;self.backend=backend
        # The valid area above roi * full height (the same every frame), and the source rows the BEV reads:
        # a row-cropped image (image_crop, CameraInfo roi) must hold at least these.
        self._valid_size=None;self.valid_area()
        self.detector=TemporalDetector(cfg)
        if backend=='carla':self.detector.detector=CarlaDetector(cfg)
        elif backend=='paint_evidence':self.detector.detector=PaintEvidenceDetector(cfg)
        self.extra=Extras(**(extras or {})) if backend=='carla' and extras else None
        self.core=None;self.publisher=None;self.node=None
        if ros or (display is not None and display[2]):
            import rclpy
            rclpy.init(args=[]);self.node=rclpy.create_node('vision_road_perception')
        if display is not None:
            from .display_filter import DisplayCore,DisplayStage
            self.core=DisplayCore(display[0]);self.error=display[1]
            if display[2]:
                from builtin_interfaces.msg import Time
                from visualization_msgs.msg import MarkerArray
                from vision_lane_interfaces.msg import LaneStyleArray
                self.publisher=(self.node.create_publisher(MarkerArray,'/road/lane_display_markers',10),
                                self.node.create_publisher(LaneStyleArray,'/road/lane_display_styles',10),DisplayStage(display[0],display[3]),Time)

    def valid_area(self):
        """projector.valid above roi * full height, kept until the projector is set up for another image size."""
        p=self.projector
        if self._valid_size!=p.source_size:
            self.valid_roi=p.valid.copy();self.valid_roi[p.my>=p.source_size[1]*self.roi]=0
            self.rows=p.rows(self.roi);self._valid_size=p.source_size
        return self.valid_roi

    def reset(self):
        self.detector.reset()
        if self.extra is not None:self.extra.reset()

    def covers(self,top,height):
        """Whether rows [top, top+height) of the full image hold every row the BEV reads."""
        return top<=self.rows[0] and top+height>=self.rows[1]

    def __call__(self,frame,stamp,pose,learned=None,side=True,top=0):
        """side=False: leave the outer lines and centrelines to side_outputs() (after the result is sent).
        top: frame is rows [top, top+height) of the full image (a row crop that covers self.rows)."""
        from .display_filter import simplify,lane_styles
        detector=self.detector;begin=time.perf_counter()
        if self.backend=='carla':detector.detector.learned=learned
        bev=self.projector.warp(frame,top)
        valid=self.valid_area().copy()
        bev=cv2.bitwise_and(bev,bev,mask=valid)
        warped=time.perf_counter()
        lanes,binary=detector.process(bev,valid,stamp,pose,defer=True)
        detected=time.perf_counter()
        shown={}
        if self.core is not None:
            # As road_node does with its own messages: simplified lines, the side styles, this frame's pose.
            # clone: dataclasses.replace without its per-call cost (Lane has no __post_init__; same fields).
            paint=[clone(lane,points=simplify(lane.points,self.error)) for lane in lanes]
            bounds=[clone(lane,points=simplify(lane.points,self.error)) for lane in detector.boundary_lanes]
            self.core.styles(stamp,lane_styles(lanes))
            shown=dict(display_lanes=paint,display_boundaries=bounds,display=self.core.step(stamp,
                [(l.side,l.points,l.reprojected) for l in bounds if len(l.points)>=2],pose,
                [(l.side,l.points) for l in paint if not l.reprojected and len(l.points)>=2]))
            if self.publisher is not None:
                markers,styles,stage,Time=self.publisher;ns=round(stamp*1e9)
                out=stage.messages(Time(sec=ns//1000000000,nanosec=ns%1000000000),shown.pop('display'))
                markers.publish(out[0]);styles.publish(out[1]);shown['display_published']=True
        shown_at=time.perf_counter()
        # Stop lines and crosswalks after the display line too (they only read this frame's detector state).
        markings=self.extra.after(detector,bev,valid,pose) if self.extra is not None else []
        marked=time.perf_counter()
        # After the display line: the outputs beside the own lines (outer lines, centrelines), off the display path.
        if side:detector.finish();shown['outer']=detector.outer_lanes;shown['centres']=detector.centre_lanes
        # Time spent here, for the node's performance report (ms): warp, detector, display filter (and the side outputs).
        finished=time.perf_counter()
        worker=dict(warp=(warped-begin)*1e3,detect=(detected-warped)*1e3,display=(shown_at-detected)*1e3)
        if self.extra is not None:worker['markings']=(marked-shown_at)*1e3
        if side:worker['extra']=(finished-marked)*1e3
        return dict(**shown,worker_ms=worker,lanes=lanes,binary=binary,bev=bev,boundaries=detector.boundary_lanes,markings=markings,
                    measured=detector.measured_count,predicted=detector.predicted_count,
                    registration=detector.registration,diagnostics=getattr(detector.detector,'diagnostics',{}))

    def side_outputs(self,stamp,t,topics=None):
        """The last frame's outer lines and centrelines (finish()), published on topics (SideTopics) when given."""
        self.detector.finish();outer=self.detector.outer_lanes;centres=self.detector.centre_lanes
        if topics is not None:topics.publish(stamp,t,outer,centres)
        return dict(outer=outer,centres=centres)

    def close(self):
        if self.node is not None:
            import rclpy
            try:self.node.destroy_node();rclpy.try_shutdown()
            except Exception:pass


def compute_loop(connection,name,projector,cfg,backend,roi,threads,extras=None,display=None):
    """Images come from road_node through shared memory, one at a time."""
    memory=shared_memory.SharedMemory(name=name)
    frame=None;perceive=None
    try:
        perceive=Perception(projector,cfg,backend,roi,threads,extras,display)
        connection.send('ready')
        while True:
            command=connection.recv()
            if command is None:break
            shape,stamp,pose,reset,learned,top=command
            if reset:perceive.reset()
            frame=np.ndarray(shape,np.uint8,buffer=memory.buf)
            connection.send(perceive(frame,stamp,pose,learned,top=top))
    except (EOFError,BrokenPipeError):pass
    except Exception as exc:
        try:connection.send(dict(error=str(exc)))
        except (EOFError,BrokenPipeError):pass
    finally:
        frame=None;memory.close();connection.close()
        if perceive is not None:perceive.close()


def ros_loop(connection,projector,cfg,backend,roi,threads,extras,display,ros):
    """image_in_worker: this process subscribes the camera and the odometry itself and keeps only the newest
    image waiting (depth 1), so an image is neither copied to road_node nor handed over; results (and, every
    1/debug_hz, a frame reduced to the overlay width) go back to road_node for its other topics."""
    import math
    from rclpy.qos import QoSProfile,ReliabilityPolicy
    from sensor_msgs.msg import Image,CameraInfo
    from nav_msgs.msg import Odometry
    from rclpy.serialization import deserialize_message
    import struct
    from .core import Timeline
    perceive=None
    try:
        perceive=Perception(projector,cfg,backend,roi,threads,extras,display,ros=True);node=perceive.node
        outer,centre,error=ros.get('side',(False,False,0.));topics=None
        if outer or centre:
            from .side_topics import SideTopics
            from builtin_interfaces.msg import Time
            topics=SideTopics(node,ros['base_frame'],ros['debug_hz'],error,outer,centre)
        # Plan 33 (lane_display_filter two_lane_hold_s): the neighbouring lane's outer line while the lane guidance changes
        # lanes, on /road/lane_display_outer, after the outer lines (off the display line's path).
        two=None
        if topics is not None and perceive.core is not None and perceive.core.filter.s.two_lane_hold_s>0:
            import json
            from std_msgs.msg import String
            from visualization_msgs.msg import MarkerArray
            from .display_filter import TwoLane,DisplayStage,simplify
            two=TwoLane(perceive.core.filter.s);two_stage=DisplayStage(perceive.core.filter.s,ros['base_frame'])
            two_pub=node.create_publisher(MarkerArray,'/road/lane_display_outer',10)
            def on_guidance(msg):
                try:d=json.loads(msg.data);two.guidance(float(d['stamp']),d.get('state'))
                except (ValueError,KeyError,TypeError):pass
            node.create_subscription(String,ros.get('guidance_topic','/hero/lane_guidance'),on_guidance,10)
        learned=LearnedMask(projector,every=ros.get('learned_every',1),client=ros['learned']) if ros.get('learned') is not None else None
        share=None
        if ros.get('image_shm_name'):
            from .image_share import ImageShareWriter
            share=ImageShareWriter(ros['image_shm_name'],ros.get('image_shm_every',2))
        timeline=Timeline(ros['odom_max_gap_sec']);state=dict(shape=None,last=None,reset=False,last_debug=-math.inf,roi=None,skipped=0)
        full_w,full_h=projector.source_size
        def on_info(msg):
            # A row-cropped image: CameraInfo keeps the full image size, roi says which rows the image holds.
            r=msg.roi;state['roi']=(int(r.y_offset),int(r.height),int(r.x_offset),int(r.width)) if r.height else None
        def crop_top(h,w):
            """The full-image row of the image's first row, or None when it cannot be placed.  An image of
            another width is not a crop (another resolution: the projector sets itself up for it, as before)."""
            if (h,w)==(full_h,full_w) or w!=full_w or h>full_h:return 0
            roi=state['roi']
            if roi is None or roi[1]!=h or w!=full_w or roi[2]!=0 or roi[3] not in (0,full_w) or not perceive.covers(roi[0],h):return None
            return roi[0]
        def on_odom(msg):
            if msg.header.frame_id!=ros['world_frame'] or msg.child_frame_id!=ros['base_frame']:return
            p,q=msg.pose.pose.position,msg.pose.pose.orientation
            norm=math.sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w)
            if not math.isfinite(norm) or norm<1e-6:return
            x,y,z,w=q.x/norm,q.y/norm,q.z/norm,q.w/norm
            generation=timeline.generation
            timeline.add(msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,p.x,p.y,math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))
            if timeline.generation!=generation:state['reset']=True
        def on_image(raw):
            received=time.perf_counter_ns()
            if state['shape'] is None:
                first=deserialize_message(raw,Image)
                if first.encoding not in ('bgr8','rgb8') or first.step!=first.width*3 or first.is_bigendian:
                    raise ValueError('image_in_worker needs packed bgr8/rgb8 images')
                state['shape']=(first.height,first.width,first.encoding,first.header.frame_id)
            h,w,encoding,frame_id=state['shape'];sec,nanosec=struct.unpack_from('<iI',raw,4);t=sec+nanosec*1e-9
            if state['reset']:
                perceive.reset();state['last']=None;state['reset']=False
                if two is not None:two.reset()
                if learned is not None:learned.reset()
            if state['last'] is not None and t<=state['last']:return
            top=crop_top(h,w)
            if top is None:
                state['skipped']+=1
                node.get_logger().warning(f'Skipping a {w}x{h} image: not the full {full_w}x{full_h} image, and no CameraInfo roi '
                    f'placing rows {perceive.rows[0]}-{perceive.rows[1]-1} inside it ({state["skipped"]} so far)',throttle_duration_sec=5)
                return
            state['last']=t;start=time.perf_counter_ns();cropped=w==full_w and h<full_h
            frame=np.frombuffer(raw,np.uint8,h*w*3,len(raw)-h*w*3).reshape(h,w,3)
            if encoding=='rgb8':frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            mask=None
            if learned is not None:
                # As road_node does without image_in_worker: the newest finished mask, then this frame to the model.
                mask=learned.next(t,frame,top if cropped else None);result_learned=time.perf_counter_ns()
                state['learned_frames']=state.get('learned_frames',0)+1;state['learned_used']=state.get('learned_used',0)+(mask is not None)
            pose=timeline.at(t);result=perceive(frame,t,pose,learned=mask,side=topics is None,top=top)
            if learned is not None:result.setdefault('worker_ms',{})['learned']=(result_learned-start)/1e6
            result.update(stamp=(sec,nanosec),frame_id=frame_id,pose=pose,receive_ns=received,start_ns=start)
            if ros['debug_hz']>0 and t-state['last_debug']>=1/ros['debug_hz']-1e-8:
                state['last_debug']=t;scale=min(1.,ros['overlay_max_width']/w) if ros['overlay_max_width'] else 1.
                if cropped:  # a row crop: the overlay draws on the full image, black outside the rows
                    padded=np.zeros((full_h,full_w,3),np.uint8);padded[top:top+h]=frame;frame=padded;h=full_h
                result.update(debug_frame=cv2.resize(frame,(round(w*scale),round(h*scale))) if scale<1 else frame.copy(),debug_scale=scale)
            else:result.update(bev=None,binary=None)  # only the debug images use them
            result['computed_ns']=time.perf_counter_ns();connection.send(result)
            if topics is not None:
                # Outer lines and centrelines after road_node has the main result: its own outputs go first.
                begin=time.perf_counter();side=perceive.side_outputs(Time(sec=sec,nanosec=nanosec),t,topics)
                if two is not None:
                    third=two.step(t,[(l.side,simplify(l.points,error),l.reprojected) for l in side['outer'] if len(l.points)>=2],pose,
                                   perceive.core.filter.output())
                    two_pub.publish(two_stage.messages(Time(sec=sec,nanosec=nanosec),third)[0])
                connection.send(dict(side=side,extra_ms=(time.perf_counter()-begin)*1e3))
            # image_shm_name (plan 30): this image for the sign detector, after the lane outputs (image_share.py).
            if share is not None and not cropped:share.write(memoryview(raw)[len(raw)-h*w*3:],w,h,encoding,sec,nanosec)
        node.create_subscription(Odometry,ros['odom_topic'],on_odom,QoSProfile(depth=200,reliability=ReliabilityPolicy.BEST_EFFORT))
        if ros.get('camera_info_topic'):
            node.create_subscription(CameraInfo,ros['camera_info_topic'],on_info,QoSProfile(depth=5,reliability=ReliabilityPolicy.BEST_EFFORT))
        node.get_logger().info(f'Perception worker: the BEV reads image rows {perceive.rows[0]}-{perceive.rows[1]-1} of {full_w}x{full_h}; '
                               'a row-cropped image (CameraInfo roi) must hold them')
        node.create_subscription(Image,ros['image_topic'],on_image,QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE
                                 if ros['reliability']=='reliable' else ReliabilityPolicy.BEST_EFFORT),raw=True)
        import rclpy
        connection.send('ready')
        while not connection.poll(0):rclpy.spin_once(node,timeout_sec=.05)
    except (EOFError,BrokenPipeError):pass
    except Exception as exc:
        try:connection.send(dict(error=str(exc)))
        except (EOFError,BrokenPipeError):pass
    finally:
        connection.close()
        if perceive is not None:perceive.close()
        try:
            if share is not None:share.close()
        except NameError:pass
        try:
            if learned is not None:
                print(f"Perception worker: learned mask used in {state.get('learned_used',0)} of {state.get('learned_frames',0)} frames",flush=True)
                learned.close()
        except NameError:pass


class RosWorker:
    """The perception worker with its own image and odometry subscriptions (image_in_worker)."""

    def __init__(self,projector,cfg,backend,roi,threads,extras,display,ros):
        context=mp.get_context('spawn');self.connection,child=context.Pipe()
        self.process=context.Process(target=ros_loop,args=(child,projector,cfg,backend,roi,threads,extras,display,ros),daemon=True)
        self.process.start();child.close();self.reset_pending=False
        if not self.connection.poll(60):self.close();raise RuntimeError('Perception worker startup timeout')
        ready=self.connection.recv()
        if ready!='ready':self.close();raise RuntimeError('Perception worker failed: '+str(ready))

    def close(self):
        if hasattr(self,'process') and self.process.is_alive():
            try:self.connection.send(None)
            except (EOFError,BrokenPipeError):pass
            self.process.join(timeout=3)
            if self.process.is_alive():self.process.terminate();self.process.join(timeout=2)
        self.connection.close()


class ComputeWorker:
    def __init__(self,projector,cfg,backend,roi,threads,extras=None,display=None):
        self.capacity=projector.size[0]*projector.size[1]*3
        self.memory=shared_memory.SharedMemory(create=True,size=self.capacity)
        context=mp.get_context('spawn');self.connection,child=context.Pipe()
        self.process=context.Process(target=compute_loop,args=(child,self.memory.name,projector,cfg,backend,roi,threads,extras,display),daemon=True)
        self.process.start();child.close();self.reset_pending=False
        if not self.connection.poll(60):self.close();raise RuntimeError('Perception worker startup timeout')  # model load
        ready=self.connection.recv()
        if ready!='ready':self.close();raise RuntimeError('Perception worker failed: '+str(ready))

    def compute(self,frame,stamp,pose,learned=None,top=0):
        """top: frame is rows [top, top+height) of the full image (a row crop)."""
        if frame.nbytes>self.capacity:raise ValueError('Input exceeds calibrated shared-image capacity')
        target=np.ndarray(frame.shape,np.uint8,buffer=self.memory.buf);np.copyto(target,frame)
        self.connection.send((frame.shape,stamp,pose,self.reset_pending,learned,top));self.reset_pending=False
        if not self.connection.poll(5):raise RuntimeError('Perception worker timeout')
        result=self.connection.recv()
        if 'error' in result:raise RuntimeError(result['error'])
        return result

    def close(self):
        if hasattr(self,'process') and self.process.is_alive():
            try:self.connection.send(None)
            except (EOFError,BrokenPipeError):pass
            self.process.join(timeout=3)
            if self.process.is_alive():self.process.terminate();self.process.join(timeout=2)
        self.connection.close();self.memory.close();self.memory.unlink()
