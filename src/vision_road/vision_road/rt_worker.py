"""Bounded shared-image worker: perception has its own Python interpreter/GIL."""
import multiprocessing as mp
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



def compute_loop(connection,name,projector,cfg,backend,roi,threads,extras=None):
    memory=shared_memory.SharedMemory(name=name)
    frame=None
    try:
        cv2.setNumThreads(threads)
        projector.initialize(*projector.size)
        detector=TemporalDetector(cfg)
        if backend=='carla':detector.detector=CarlaDetector(cfg)
        elif backend=='paint_evidence':detector.detector=PaintEvidenceDetector(cfg)
        extra=Extras(**(extras or {})) if backend=='carla' and extras else None
        connection.send('ready')
        while True:
            command=connection.recv()
            if command is None:break
            shape,stamp,pose,reset,learned=command
            if reset:
                detector.reset()
                if extra is not None:extra.reset()
            frame=np.ndarray(shape,np.uint8,buffer=memory.buf)
            if backend=='carla':detector.detector.learned=learned
            bev=projector.warp(frame)
            valid=projector.valid.copy();valid[projector.my>=shape[0]*roi]=0
            bev=cv2.bitwise_and(bev,bev,mask=valid)
            lanes,binary=detector.process(bev,valid,stamp,pose)
            markings=extra.after(detector,bev,valid,pose) if extra is not None else []
            connection.send(dict(lanes=lanes,binary=binary,bev=bev,boundaries=detector.boundary_lanes,markings=markings,
                measured=detector.measured_count,predicted=detector.predicted_count,
                registration=detector.registration,diagnostics=getattr(detector.detector,'diagnostics',{})))
    except (EOFError,BrokenPipeError):pass
    except Exception as exc:
        try:connection.send(dict(error=str(exc)))
        except (EOFError,BrokenPipeError):pass
    finally:
        frame=None;memory.close();connection.close()


class ComputeWorker:
    def __init__(self,projector,cfg,backend,roi,threads,extras=None):
        self.capacity=projector.size[0]*projector.size[1]*3
        self.memory=shared_memory.SharedMemory(create=True,size=self.capacity)
        context=mp.get_context('spawn');self.connection,child=context.Pipe()
        self.process=context.Process(target=compute_loop,args=(child,self.memory.name,projector,cfg,backend,roi,threads,extras),daemon=True)
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
