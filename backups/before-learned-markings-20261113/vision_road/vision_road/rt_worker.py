"""Bounded shared-image worker: perception has its own Python interpreter/GIL."""
import multiprocessing as mp
from multiprocessing import shared_memory
import cv2
import numpy as np
from .temporal import TemporalDetector
from .carla import CarlaDetector
from .classical import PaintEvidenceDetector


def compute_loop(connection,name,projector,cfg,backend,roi,threads):
    memory=shared_memory.SharedMemory(name=name)
    frame=None
    try:
        cv2.setNumThreads(threads)
        projector.initialize(*projector.size)
        detector=TemporalDetector(cfg)
        if backend=='carla':detector.detector=CarlaDetector(cfg)
        elif backend=='paint_evidence':detector.detector=PaintEvidenceDetector(cfg)
        connection.send('ready')
        while True:
            command=connection.recv()
            if command is None:break
            shape,stamp,pose,reset=command
            if reset:detector.reset()
            frame=np.ndarray(shape,np.uint8,buffer=memory.buf)
            bev=projector.warp(frame)
            valid=projector.valid.copy();valid[projector.my>=shape[0]*roi]=0
            bev=cv2.bitwise_and(bev,bev,mask=valid)
            lanes,binary=detector.process(bev,valid,stamp,pose)
            connection.send(dict(lanes=lanes,binary=binary,bev=bev,boundaries=detector.boundary_lanes,
                measured=detector.measured_count,predicted=detector.predicted_count,
                registration=detector.registration,diagnostics=getattr(detector.detector,'diagnostics',{})))
    except (EOFError,BrokenPipeError):pass
    except Exception as exc:
        try:connection.send(dict(error=str(exc)))
        except (EOFError,BrokenPipeError):pass
    finally:
        frame=None;memory.close();connection.close()


class ComputeWorker:
    def __init__(self,projector,cfg,backend,roi,threads):
        self.capacity=projector.size[0]*projector.size[1]*3
        self.memory=shared_memory.SharedMemory(create=True,size=self.capacity)
        context=mp.get_context('spawn');self.connection,child=context.Pipe()
        self.process=context.Process(target=compute_loop,args=(child,self.memory.name,projector,cfg,backend,roi,threads),daemon=True)
        self.process.start();child.close();self.reset_pending=False
        if not self.connection.poll(20):self.close();raise RuntimeError('Perception worker startup timeout')
        ready=self.connection.recv()
        if ready!='ready':self.close();raise RuntimeError('Perception worker failed: '+str(ready))

    def compute(self,frame,stamp,pose):
        if frame.nbytes>self.capacity:raise ValueError('Input exceeds calibrated shared-image capacity')
        target=np.ndarray(frame.shape,np.uint8,buffer=self.memory.buf);np.copyto(target,frame)
        self.connection.send((frame.shape,stamp,pose,self.reset_pending));self.reset_pending=False
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
