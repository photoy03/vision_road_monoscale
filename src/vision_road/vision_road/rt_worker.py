"""Bounded shared-image worker: perception has its own Python interpreter/GIL."""
import multiprocessing as mp
import time
from multiprocessing import shared_memory
import cv2
import numpy as np
from .temporal import TemporalDetector
from .carla import CarlaDetector
from .classical import PaintEvidenceDetector


class LearnedMask:
    """Optional learned lane mask from its own process, typically one frame behind.

    Without weights, torch or CUDA it stays off and the detector runs unchanged."""

    def __init__(self,projector,weights='',every=1,max_age=.2):
        # every=2 runs the model on every other frame: the mask is then 1-2 frames old,
        # which gated maploop as well, and the model's CPU share halves.
        self.model=None;self.status='off';self.max_age=max_age;self.every=max(1,int(every));self.count=0
        if weights:
            try:
                from .learned_mask import LaneMaskProcess
                self.model=LaneMaskProcess(weights,projector);self.status='on'
            except Exception as exc:self.status='unavailable: '+str(exc)

    def next(self,stamp,frame):
        """The newest finished mask no older than max_age, then queue this frame."""
        if self.model is None:return None
        latest=self.model.latest();self.count+=1
        if self.count%self.every==0:self.model.submit(stamp,frame)
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

    def reset(self):
        self.detector.reset()
        if self.extra is not None:self.extra.reset()

    def __call__(self,frame,stamp,pose,learned=None):
        from dataclasses import replace
        from .display_filter import simplify,lane_styles
        detector=self.detector;begin=time.perf_counter()
        if self.backend=='carla':detector.detector.learned=learned
        bev=self.projector.warp(frame)
        valid=self.projector.valid.copy();valid[self.projector.my>=frame.shape[0]*self.roi]=0
        bev=cv2.bitwise_and(bev,bev,mask=valid)
        warped=time.perf_counter()
        lanes,binary=detector.process(bev,valid,stamp,pose)
        markings=self.extra.after(detector,bev,valid,pose) if self.extra is not None else []
        detected=time.perf_counter()
        shown={}
        if self.core is not None:
            # As road_node does with its own messages: simplified lines, the side styles, this frame's pose.
            paint=[replace(lane,points=simplify(lane.points,self.error)) for lane in lanes]
            bounds=[replace(lane,points=simplify(lane.points,self.error)) for lane in detector.boundary_lanes]
            self.core.styles(stamp,lane_styles(lanes))
            shown=dict(display_lanes=paint,display_boundaries=bounds,display=self.core.step(stamp,
                [(l.side,l.points,l.reprojected) for l in bounds if len(l.points)>=2],pose,
                [(l.side,l.points) for l in paint if not l.reprojected and len(l.points)>=2]))
            if self.publisher is not None:
                markers,styles,stage,Time=self.publisher;ns=round(stamp*1e9)
                out=stage.messages(Time(sec=ns//1000000000,nanosec=ns%1000000000),shown.pop('display'))
                markers.publish(out[0]);styles.publish(out[1]);shown['display_published']=True
        # Time spent here, for the node's performance report (ms): warp, detector, display filter.
        worker=dict(warp=(warped-begin)*1e3,detect=(detected-warped)*1e3,display=(time.perf_counter()-detected)*1e3)
        return dict(**shown,worker_ms=worker,lanes=lanes,binary=binary,bev=bev,boundaries=detector.boundary_lanes,markings=markings,
                    measured=detector.measured_count,predicted=detector.predicted_count,
                    registration=detector.registration,diagnostics=getattr(detector.detector,'diagnostics',{}))

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
            shape,stamp,pose,reset,learned=command
            if reset:perceive.reset()
            frame=np.ndarray(shape,np.uint8,buffer=memory.buf)
            connection.send(perceive(frame,stamp,pose,learned))
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
    from sensor_msgs.msg import Image
    from nav_msgs.msg import Odometry
    from rclpy.serialization import deserialize_message
    import struct
    from .core import Timeline
    perceive=None
    try:
        perceive=Perception(projector,cfg,backend,roi,threads,extras,display,ros=True);node=perceive.node
        timeline=Timeline(ros['odom_max_gap_sec']);state=dict(shape=None,last=None,reset=False,last_debug=-math.inf)
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
            if state['reset']:perceive.reset();state['last']=None;state['reset']=False
            if state['last'] is not None and t<=state['last']:return
            state['last']=t;start=time.perf_counter_ns()
            frame=np.frombuffer(raw,np.uint8,h*w*3,len(raw)-h*w*3).reshape(h,w,3)
            if encoding=='rgb8':frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            pose=timeline.at(t);result=perceive(frame,t,pose)
            result.update(stamp=(sec,nanosec),frame_id=frame_id,pose=pose,receive_ns=received,start_ns=start)
            if ros['debug_hz']>0 and t-state['last_debug']>=1/ros['debug_hz']-1e-8:
                state['last_debug']=t;scale=min(1.,ros['overlay_max_width']/w) if ros['overlay_max_width'] else 1.
                result.update(debug_frame=cv2.resize(frame,(round(w*scale),round(h*scale))) if scale<1 else frame.copy(),debug_scale=scale)
            else:result.update(bev=None,binary=None)  # only the debug images use them
            result['computed_ns']=time.perf_counter_ns();connection.send(result)
        node.create_subscription(Odometry,ros['odom_topic'],on_odom,QoSProfile(depth=200,reliability=ReliabilityPolicy.BEST_EFFORT))
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

    def compute(self,frame,stamp,pose,learned=None):
        if frame.nbytes>self.capacity:raise ValueError('Input exceeds calibrated shared-image capacity')
        target=np.ndarray(frame.shape,np.uint8,buffer=self.memory.buf);np.copyto(target,frame)
        self.connection.send((frame.shape,stamp,pose,self.reset_pending,learned));self.reset_pending=False
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
