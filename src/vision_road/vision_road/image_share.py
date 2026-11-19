"""The newest camera image in shared memory for other processes on this computer (plan 30, D1).

road_node's perception worker (image_in_worker) writes every image_shm_every-th image into one block after it has
published that frame's lane results; the sign detector (src/yolo_signs, image_shm) reads the block instead of
subscribing the camera topic a second time (a second 6 MB subscriber cost the lane detector about 4 ms).

Block layout (little endian): magic b'VRIM', version u32, seq u64 (odd while a frame is being written, even when
done), sec i32, nanosec u32, width u32, height u32, encoding u32 (0 bgr8, 1 rgb8), 28 reserved bytes; then the
pixels (height x width x 3).  A reader copies the pixels and accepts them when seq is even and unchanged across
the copy (a sequence lock: the writer never waits for readers)."""
import struct
import numpy as np
from multiprocessing import shared_memory

HEADER = struct.Struct('<4sIQiIIII28x')  # 64 bytes
MAGIC = b'VRIM'
ENCODINGS = {'bgr8': 0, 'rgb8': 1}


def untracked(shm):
    """Keep Python's resource tracker from unlinking a block this process only uses (Python < 3.13)."""
    try:
        from multiprocessing import resource_tracker
        resource_tracker.unregister(shm._name, 'shared_memory')
    except Exception:
        pass
    return shm


class ImageShareWriter:
    def __init__(self, name, every=2):
        self.name = name;self.every = max(1, int(every));self.count = 0;self.shm = None;self.seq = 0

    def _open(self, size):
        try:
            shm = shared_memory.SharedMemory(self.name, create=True, size=size)
        except FileExistsError:  # left by an earlier run: reuse it when large enough, else make it anew
            shm = shared_memory.SharedMemory(self.name)
            if shm.size < size:
                shm.close();shm.unlink();shm = shared_memory.SharedMemory(self.name, create=True, size=size)
        return shm

    def write(self, pixels, width, height, encoding, sec, nanosec):
        """pixels: the packed height x width x 3 bytes (a bytes-like view); every every-th call is written."""
        self.count += 1
        if (self.count-1) % self.every:return False
        size = HEADER.size+width*height*3
        if self.shm is None or self.shm.size < size:
            if self.shm is not None:self.close()
            self.shm = self._open(size)
        buf = self.shm.buf
        self.seq += 1  # odd: being written
        buf[:HEADER.size] = HEADER.pack(MAGIC, 1, self.seq, sec, nanosec, width, height, ENCODINGS.get(encoding, 0))
        buf[HEADER.size:size] = pixels
        self.seq += 1  # even: done
        buf[:HEADER.size] = HEADER.pack(MAGIC, 1, self.seq, sec, nanosec, width, height, ENCODINGS.get(encoding, 0))
        return True

    def close(self):
        if self.shm is not None:
            try:self.shm.close();self.shm.unlink()
            except Exception:pass
            self.shm = None


class ImageShareReader:
    def __init__(self, name):
        self.name = name;self.shm = None;self.last = 0

    def read(self):
        """(sec, nanosec, encoding name, frame copy) of an image newer than the last one read, or None."""
        if self.shm is None:
            try:self.shm = untracked(shared_memory.SharedMemory(self.name))
            except FileNotFoundError:return None
        buf = self.shm.buf
        magic, _, seq, sec, nanosec, w, h, enc = HEADER.unpack_from(buf, 0)
        if magic != MAGIC or seq % 2 or seq == self.last or not w or not h:return None
        frame = np.frombuffer(buf, np.uint8, w*h*3, HEADER.size).reshape(h, w, 3).copy()
        if HEADER.unpack_from(buf, 0)[2] != seq:return None  # overwritten while copying
        self.last = seq
        return sec, nanosec, ('rgb8' if enc == 1 else 'bgr8'), frame

    def close(self):
        if self.shm is not None:
            try:self.shm.close()
            except Exception:pass
            self.shm = None
