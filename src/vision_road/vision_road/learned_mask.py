"""Optional learned lane mask (YOLOPv2) projected to the BEV, used to gate paint.

The classical detector stays the source of geometry, solid/dashed and double
lines; the learned mask only says where a lane line plausibly is.  Everything
here is optional: without torch, CUDA or the weights nothing is loaded and the
detector runs as before.  The model runs in its own process (LaneMaskProcess) so
the detector keeps 30 Hz and gates with the previous frame's mask.

YOLOPv2: Han et al., "YOLOPv2: Better, Faster, Stronger for Panoptic Driving
Perception", arXiv 2208.11434; weights from github.com/CAIC-AD/YOLOPv2 (V0.0.1).
"""
import multiprocessing as mp
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np

INPUT=(640,360)  # model input (w, h) before the 12 px letterbox rows
PAD=12


class LaneModel:
    """Frame (BGR, camera resolution) -> BEV lane mask (uint8 0/255) in the projector's grid."""

    def __init__(self,weights,projector,device='cuda'):
        import torch  # optional dependency, imported only when enabled
        if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is not available')
        path=Path(weights)
        if not path.is_file():raise RuntimeError(f'learned lane weights not found: {path}')
        self.torch=torch;self.device=device;self.half=device=='cuda'
        self.net=torch.jit.load(str(path),map_location=device).eval()
        if self.half:self.net.half()
        self.projector=projector;self.maps=None

    def _maps(self,shape):
        if self.maps is None or self.maps[0]!=shape:
            h,w=shape;p=self.projector
            sx,sy=INPUT[0]/w,INPUT[1]/h
            self.maps=(shape,(p.mx*sx).astype(np.float32),(p.my*sy).astype(np.float32),p.valid>0)
        return self.maps[1:]

    def image_mask(self,frame):
        """frame: camera image, or one already resized to INPUT."""
        torch=self.torch
        img=frame if frame.shape[1::-1]==INPUT else cv2.resize(frame,INPUT,interpolation=cv2.INTER_LINEAR)
        img=cv2.copyMakeBorder(img,PAD,PAD,0,0,cv2.BORDER_CONSTANT,value=(114,114,114))
        t=torch.from_numpy(np.ascontiguousarray(img[:,:,::-1].transpose(2,0,1))).to(self.device)
        t=(t.half() if self.half else t.float()).div(255).unsqueeze(0)
        with torch.no_grad():
            m=(self.net(t)[2][0,0,PAD:PAD+INPUT[1]]>.5).to(torch.uint8).mul(255)
        if self.device=='cuda':
            # Wait without spinning a CPU core for the whole inference.
            done=torch.cuda.Event(blocking=True);done.record();done.synchronize()
        return m.cpu().numpy()

    def bev(self,frame,source_shape=None):
        """source_shape: the camera image (h, w) when frame was already resized to INPUT."""
        mx,my,valid=self._maps(source_shape or frame.shape[:2])
        out=cv2.remap(self.image_mask(frame),mx,my,cv2.INTER_LINEAR)
        return (((out>127)&valid)*255).astype(np.uint8)


def _mask_loop(connection,frame_name,mask_name,weights,projector,device):
    frames=shared_memory.SharedMemory(name=frame_name);masks=shared_memory.SharedMemory(name=mask_name)
    try:
        try:
            # The GPU does the work; default CPU thread pools would compete with the detector.
            import torch
            torch.set_num_threads(1);cv2.setNumThreads(1)
            projector.initialize(*projector.size);model=LaneModel(weights,projector,device)
        except Exception as exc:
            connection.send(('error',str(exc)));return
        connection.send(('ready',None))
        while True:
            job=connection.recv()
            if job is None:break
            stamp,shape,source=job
            mask=model.bev(np.ndarray(shape,np.uint8,buffer=frames.buf),source)
            np.ndarray(mask.shape,np.uint8,buffer=masks.buf)[:]=mask
            connection.send(('mask',stamp))
    except (EOFError,BrokenPipeError):pass
    finally:
        frames.close();masks.close();connection.close()


class _MaskQueue:
    """submit / poll / latest / reset on a model process's pipe and shared memory (LaneMaskProcess or, in
    another process, LaneMaskClient)."""

    def poll(self):
        while self.busy and self.connection.poll():
            kind,stamp=self.connection.recv();self.busy=False
            self.result=(stamp,np.ndarray(self.shape,np.uint8,buffer=self.masks.buf).copy())
        return self.result

    def submit(self,stamp,frame,top=None):
        """top: frame holds only rows [top, top+height) of a full image of full_shape (a row crop); the
        model then sees that image black outside the rows (its mask there is not the full image's)."""
        self.poll()
        if self.busy:return False
        if top is not None:
            if getattr(self,'_full',None) is None or self._full.shape[:2]!=self.full_shape:self._full=np.zeros(self.full_shape+(3,),np.uint8)
            self._full[top:top+frame.shape[0]]=frame;frame=self._full
        # The model input size is resized here: 0.7 MB to copy instead of a 6 MB frame.
        small=np.ndarray((INPUT[1],INPUT[0],3),np.uint8,buffer=self.frames.buf)
        cv2.resize(frame,INPUT,dst=small,interpolation=cv2.INTER_LINEAR)
        self.connection.send((stamp,small.shape,frame.shape[:2]));self.busy=True
        return True

    def latest(self):return self.poll()

    def reset(self):self.poll();self.result=None


class LaneMaskClient(_MaskQueue):
    """The model process of a LaneMaskProcess used from another process (the perception worker with
    image_in_worker): handle() there, LaneMaskClient(handle) here.  The creating process must not use it too."""

    def __init__(self,handle):
        self.connection,frames,masks,self.shape,self.full_shape=handle
        self.frames=shared_memory.SharedMemory(name=frames);self.masks=shared_memory.SharedMemory(name=masks)
        self.busy=False;self.result=None

    def close(self):
        for memory in (self.frames,self.masks):memory.close()


class LaneMaskProcess(_MaskQueue):
    """LaneModel in its own process, so neither the GIL nor CPU work of the model
    slows the detector.  One frame is in flight at a time: submit() is skipped while
    the model is busy and latest() returns the newest finished (stamp, mask)."""

    def __init__(self,weights,projector,device='cuda',startup=60.):
        if not Path(weights).is_file():raise RuntimeError(f'learned lane weights not found: {weights}')
        self.shape=projector.valid.shape;self.full_shape=(int(projector.size[1]),int(projector.size[0]))
        self.frames=shared_memory.SharedMemory(create=True,size=INPUT[0]*INPUT[1]*3)
        self.masks=shared_memory.SharedMemory(create=True,size=int(np.prod(self.shape)))
        context=mp.get_context('spawn');self.connection,child=context.Pipe()
        self.process=context.Process(target=_mask_loop,args=(child,self.frames.name,self.masks.name,str(weights),projector,device),daemon=True)
        self.process.start();child.close();self.busy=False;self.result=None
        if not self.connection.poll(startup):self.close();raise RuntimeError('learned lane model startup timeout')
        kind,detail=self.connection.recv()
        if kind!='ready':self.close();raise RuntimeError(detail)

    def handle(self):
        """For LaneMaskClient in another process (pass it as a Process argument; then only that process uses it)."""
        return (self.connection,self.frames.name,self.masks.name,self.shape,self.full_shape)

    def close(self):
        if getattr(self,'process',None) is not None and self.process.is_alive():
            try:self.connection.send(None)
            except (EOFError,BrokenPipeError,OSError):pass
            self.process.join(timeout=3)
            if self.process.is_alive():self.process.terminate();self.process.join(timeout=2)
        for memory in (getattr(self,'frames',None),getattr(self,'masks',None)):
            if memory is not None:memory.close();memory.unlink()
