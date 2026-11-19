#!/usr/bin/env python3
"""Run the trained YOLO26m detector on a ROS 2 Image topic.

Designed for the /sensing/camera/dashcam/image_raw topic in town05_30hz.
The node deliberately does not use cv_bridge because the installed ROS 2
cv_bridge was built against NumPy 1.x while the YOLO environment uses NumPy 2.

vision-road copy (plan 27; the original is ~/Desktop/town05_yolo26m_detector, left unchanged). Added, all
behind parameters whose defaults keep the original behaviour:
  - model_path: best.pt next to this file, else the original folder's (best.pt is not in git).
  - quantize: when this ultralytics rejects predict(quantize=...) ("not a valid YOLO argument", 8.4.48 here),
    the node falls back to half=True for quantize 16 (once, with a warning).
  - full_frame (default false = the original upper-centre 640 x 640 tile): the whole image instead.
  - stats_path: per processed frame (image stamp, receive / start / end times, ultralytics speed, detection
    count) written as JSON when the node stops; for measurements.
  - raw_cdr (default false; plan 30): take the images serialized and decode only the frame that is processed (the
    detector runs at about 11 Hz on 30 Hz images: building a 6 MB Python Image for every message costs the lane
    detector CPU time).  Packed bgr8 / rgb8 only (the first message is checked).
  - image_shm (default '' = subscribe image_topic; plan 30): read the newest image from the shared memory block of
    that name, written by road_node's perception worker (image_shm_name; layout in vision_road/image_share.py),
    instead of subscribing the camera a second time.  image_frame_id names the frame in the detections.
Run with the YOLO environment's Python and ROS sourced (run_lane_hud.sh --signs does this).
"""

from __future__ import annotations

import json
import struct
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO


ORIGINAL_MODEL = Path.home() / "Desktop" / "town05_yolo26m_detector" / "best.pt"
DEFAULT_MODEL = str(Path(__file__).resolve().with_name("best.pt")
                    if Path(__file__).resolve().with_name("best.pt").is_file() else ORIGINAL_MODEL)
EXPECTED_CLASSES = {
    0: "traffic_light_green",
    1: "traffic_light_orange",
    2: "traffic_light_red",
    3: "traffic_sign_30",
    4: "traffic_sign_60",
    5: "traffic_sign_90",
}


def image_to_bgr(message: Image) -> np.ndarray:
    """Convert common 8-bit ROS image encodings to a contiguous BGR array."""
    encoding = message.encoding.lower()
    channels_by_encoding = {
        "bgr8": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
        "mono8": 1,
    }
    if encoding not in channels_by_encoding:
        raise ValueError(f"unsupported image encoding: {message.encoding}")

    channels = channels_by_encoding[encoding]
    row_bytes = message.width * channels
    if message.step < row_bytes:
        raise ValueError(
            f"invalid step {message.step} for width={message.width}, encoding={message.encoding}"
        )

    raw = np.frombuffer(message.data, dtype=np.uint8)
    expected = message.height * message.step
    if raw.size < expected:
        raise ValueError(f"short image buffer: got {raw.size} bytes, expected {expected}")

    # Slice each row before reshaping so images with row padding also work.
    pixels = raw[:expected].reshape(message.height, message.step)[:, :row_bytes]
    image = pixels.reshape(message.height, message.width, channels)

    if encoding == "rgb8":
        image = image[:, :, ::-1]
    elif encoding == "bgra8":
        image = image[:, :, :3]
    elif encoding == "rgba8":
        image = image[:, :, [2, 1, 0]]
    elif encoding == "mono8":
        image = np.repeat(image, 3, axis=2)

    return np.ascontiguousarray(image)


class Town05Yolo26mDetector(Node):
    NODE_NAME = "town05_yolo26m_detector"
    DEFAULT_MODEL_PATH = DEFAULT_MODEL
    DEFAULT_DETECTIONS_TOPIC = "/yolo/detections"
    DEFAULT_IMGSZ = 1280
    USE_FULL_FRAME = False

    def __init__(self) -> None:
        super().__init__(self.NODE_NAME)

        self.declare_parameter("model_path", self.DEFAULT_MODEL_PATH)
        self.declare_parameter("image_topic", "/sensing/camera/dashcam/image_raw")
        self.declare_parameter("detections_topic", self.DEFAULT_DETECTIONS_TOPIC)
        self.declare_parameter("confidence", 0.25)
        self.declare_parameter("imgsz", self.DEFAULT_IMGSZ)
        self.declare_parameter("device", "0")
        self.declare_parameter("quantize", 16)
        # Accept an integer too (max_inference_fps:=10 on the command line is an INTEGER parameter).
        from rcl_interfaces.msg import ParameterDescriptor
        self.declare_parameter("max_inference_fps", 0.0, ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter("full_frame", self.USE_FULL_FRAME)
        self.declare_parameter("stats_path", "")
        self.declare_parameter("raw_cdr", False)
        self.declare_parameter("image_shm", "")
        self.declare_parameter("image_frame_id", "dashcam_optical_link")
        self.declare_parameter("cpu_threads", 0)
        self.declare_parameter("cpu_affinity", "")

        model_path = self.get_parameter("model_path").value
        image_topic = self.get_parameter("image_topic").value
        detections_topic = self.get_parameter("detections_topic").value
        self.confidence = float(self.get_parameter("confidence").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.device = str(self.get_parameter("device").value)
        self.quantize = int(self.get_parameter("quantize").value)
        self.max_inference_fps = float(
            self.get_parameter("max_inference_fps").value
        )
        self.full_frame = bool(self.get_parameter("full_frame").value)
        self.stats_path = str(self.get_parameter("stats_path").value)
        self.raw_cdr = bool(self.get_parameter("raw_cdr").value)
        self.raw_shape = None  # (height, width, encoding, frame_id) from the first raw message
        self.image_shm = str(self.get_parameter("image_shm").value)
        self.image_frame_id = str(self.get_parameter("image_frame_id").value)
        self.shm_reader = ShmImageReader(self.image_shm) if self.image_shm else None
        # Plan 30 (D1): fewer CPU threads and given cores for this process, so the lane detector keeps its CPU time.
        # cpu_threads 0 / cpu_affinity '' (defaults): unchanged.  cpu_affinity e.g. "12-15" or "12,13".
        threads = int(self.get_parameter("cpu_threads").value)
        if threads > 0:
            import cv2
            import torch
            torch.set_num_threads(threads)
            cv2.setNumThreads(threads)
        affinity = str(self.get_parameter("cpu_affinity").value).strip()
        if affinity:
            import os
            cores = set()
            for part in affinity.split(","):
                lo, _, hi = part.partition("-")
                cores.update(range(int(lo), int(hi or lo) + 1))
            os.sched_setaffinity(0, cores)
        self.stats: list[dict] = []
        self.use_half = False  # set when predict(quantize=...) is rejected

        self.get_logger().info(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.names = self.model.names
        if self.names != EXPECTED_CLASSES:
            raise RuntimeError(
                f"unexpected model classes: {self.names}; expected: {EXPECTED_CLASSES}"
            )

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.image_subscription = None if self.shm_reader is not None else self.create_subscription(
            Image, image_topic, self.on_image, sensor_qos, raw=self.raw_cdr
        )
        self.detections_publisher = self.create_publisher(String, detections_topic, 10)

        self.frame_lock = threading.Lock()
        self.frame_event = threading.Event()
        self.latest_message: Image | None = None
        self.running = True
        self.frame_count = 0
        self.started_at: float | None = None
        self.last_speed: dict[str, float] = {}
        self.last_inference_started = 0.0
        self.inference_thread = threading.Thread(
            target=self.inference_loop,
            name="yolo_inference",
            daemon=True,
        )
        self.inference_thread.start()
        self.get_logger().info(
            f"Listening on {image_topic}; publishing detections only on {detections_topic}"
        )

    def on_image(self, message: Image) -> None:
        """Keep only the newest frame and return immediately to the ROS executor."""
        with self.frame_lock:
            self.latest_message = message
            self.latest_received = time.perf_counter()
            self.frame_event.set()

    def inference_loop(self) -> None:
        """Process the newest available frame without blocking image reception."""
        while self.running:
            if self.shm_reader is not None:
                self.shm_step()
                continue
            self.frame_event.wait(timeout=0.5)
            if not self.running:
                break

            if self.max_inference_fps > 0.0:
                period = 1.0 / self.max_inference_fps
                remaining = self.last_inference_started + period - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)

            with self.frame_lock:
                message = self.latest_message
                received = getattr(self, "latest_received", 0.0)
                self.latest_message = None
                self.frame_event.clear()

            if message is None:
                continue

            try:
                self.last_inference_started = time.perf_counter()
                if self.raw_cdr:
                    message, frame = self.from_cdr(message)
                    count = self.process_message(message, frame)
                else:
                    count = self.process_message(message)
                if self.stats_path:
                    end = time.perf_counter()
                    self.stats.append(dict(
                        stamp=message.header.stamp.sec + message.header.stamp.nanosec * 1e-9,
                        received=received, start=self.last_inference_started, end=end,
                        speed={k: round(float(v), 3) for k, v in self.last_speed.items()},
                        detections=count))
            except Exception as error:
                self.get_logger().error(
                    f"inference failed: {error}", throttle_duration_sec=5.0
                )

    def shm_step(self) -> None:
        """image_shm: wait for a new image in the block (polling every 3 ms), then detect on it."""
        if self.max_inference_fps > 0.0:
            remaining = self.last_inference_started + 1.0 / self.max_inference_fps - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
        got = self.shm_reader.read()
        if got is None:
            time.sleep(0.003)
            return
        received = time.perf_counter()
        sec, nanosec, encoding, frame = got
        if encoding == "rgb8":
            frame = np.ascontiguousarray(frame[:, :, ::-1])
        header = SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec), frame_id=self.image_frame_id)
        message = SimpleNamespace(header=header, width=frame.shape[1], height=frame.shape[0])
        try:
            self.last_inference_started = time.perf_counter()
            count = self.process_message(message, frame)
            if self.stats_path:
                self.stats.append(dict(stamp=sec + nanosec * 1e-9, received=received, start=self.last_inference_started,
                                       end=time.perf_counter(), speed={k: round(float(v), 3) for k, v in self.last_speed.items()},
                                       detections=count))
        except Exception as error:
            self.get_logger().error(f"inference failed: {error}", throttle_duration_sec=5.0)

    def from_cdr(self, raw: bytes):
        """(header-like message, BGR frame) from a serialized packed bgr8 / rgb8 Image, the pixels viewed in place."""
        if self.raw_shape is None:
            from rclpy.serialization import deserialize_message
            first = deserialize_message(raw, Image)
            if first.encoding not in ("bgr8", "rgb8") or first.step != first.width * 3 or first.is_bigendian:
                raise ValueError("raw_cdr needs packed bgr8 / rgb8 images")
            self.raw_shape = (first.height, first.width, first.encoding, first.header.frame_id)
        h, w, encoding, frame_id = self.raw_shape
        # CDR: 4-byte encapsulation, then the header stamp; the pixel array is the last field.
        sec, nanosec = struct.unpack_from("<iI", raw, 4)
        frame = np.frombuffer(raw, np.uint8, h * w * 3, len(raw) - h * w * 3).reshape(h, w, 3)
        if encoding == "rgb8":
            frame = np.ascontiguousarray(frame[:, :, ::-1])
        header = SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec), frame_id=frame_id)
        return SimpleNamespace(header=header, width=w, height=h), frame

    def process_message(self, message: Image, frame: np.ndarray | None = None) -> None:
        if frame is None:
            try:
                frame = image_to_bgr(message)
            except ValueError as error:
                self.get_logger().error(str(error), throttle_duration_sec=5.0)
                return

        if self.full_frame:
            sources = [frame]
            offsets = [(0, 0)]
        else:
            # Magnify the upper-center driving view so distant traffic lights retain
            # enough pixels for color classification without scanning five tiles.
            tile_size = min(640, frame.shape[0], frame.shape[1])
            last_x = frame.shape[1] - tile_size
            tile_starts = [last_x // 2]
            sources = []
            offsets = []
            for x in tile_starts:
                sources.append(frame[:tile_size, x : x + tile_size])
                offsets.append((x, 0))

        results = [self.predict(sources[0])]
        self.last_speed = results[0].speed

        candidates = []
        for result, (offset_x, offset_y) in zip(results, offsets):
            if result.boxes is None:
                continue
            coordinates = result.boxes.xyxy.detach().cpu().numpy()
            confidences = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            for xyxy, confidence, class_id in zip(coordinates, confidences, classes):
                shifted = xyxy.copy()
                shifted[[0, 2]] += offset_x
                shifted[[1, 3]] += offset_y
                candidates.append((shifted, float(confidence), int(class_id)))

        # Remove duplicate boxes created by overlapping tiles. Suppression is
        # class-agnostic so the strongest color prediction wins for one lamp.
        candidates.sort(key=lambda item: item[1], reverse=True)
        kept = []
        for candidate in candidates:
            box = candidate[0]
            duplicate = False
            for kept_box, _, _ in kept:
                x1 = max(box[0], kept_box[0])
                y1 = max(box[1], kept_box[1])
                x2 = min(box[2], kept_box[2])
                y2 = min(box[3], kept_box[3])
                intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
                kept_area = max(0.0, kept_box[2] - kept_box[0]) * max(
                    0.0, kept_box[3] - kept_box[1]
                )
                union = box_area + kept_area - intersection
                if union > 0.0 and intersection / union >= 0.5:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)

        detections = [
            {
                "class_id": class_id,
                "class_name": str(self.names[class_id]),
                "confidence": round(confidence, 6),
                "xyxy": [round(float(value), 2) for value in xyxy],
            }
            for xyxy, confidence, class_id in kept
        ]

        detection_message = String()
        detection_message.data = json.dumps(
            {
                "stamp": {
                    "sec": message.header.stamp.sec,
                    "nanosec": message.header.stamp.nanosec,
                },
                "frame_id": message.header.frame_id,
                "image_width": message.width,
                "image_height": message.height,
                "detections": detections,
            },
            ensure_ascii=False,
        )
        self.detections_publisher.publish(detection_message)
        count = len(detections)

        now = time.perf_counter()
        self.frame_count += 1
        if self.started_at is None:
            self.started_at = now
        if self.frame_count % 30 == 0:
            elapsed = max(now - self.started_at, 1e-6)
            preprocess_ms = self.last_speed.get("preprocess", 0.0)
            inference_ms = self.last_speed.get("inference", 0.0)
            postprocess_ms = self.last_speed.get("postprocess", 0.0)
            self.get_logger().info(
                f"frames={self.frame_count}, average={self.frame_count / elapsed:.1f} FPS, "
                f"preprocess={preprocess_ms:.1f} ms, inference={inference_ms:.1f} ms, "
                f"postprocess={postprocess_ms:.1f} ms, detections={len(detections)}"
            )
        return count

    def predict(self, source: np.ndarray):
        """model.predict as the original (quantize=...); if this ultralytics rejects quantize, half=True instead."""
        common = dict(source=source, conf=self.confidence, imgsz=self.imgsz, device=self.device, verbose=False)
        if not self.use_half:
            try:
                return self.model.predict(quantize=self.quantize if self.quantize > 0 else None, **common)[0]
            except Exception as error:  # ultralytics raises SyntaxError for an unknown argument
                if "quantize" not in str(error):
                    raise
                self.use_half = True
                self.get_logger().warning(
                    f"predict(quantize=...) not supported here ({str(error).strip().splitlines()[0][:120]}); "
                    f"using half={self.quantize == 16}")
        return self.model.predict(half=self.quantize == 16, **common)[0]

    def stop(self) -> None:
        """Stop and join the inference worker before destroying the node."""
        self.running = False
        self.frame_event.set()
        self.inference_thread.join(timeout=2.0)
        if self.stats_path:
            Path(self.stats_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.stats_path).write_text(json.dumps(dict(
                full_frame=self.full_frame, imgsz=self.imgsz, half=self.use_half,
                max_inference_fps=self.max_inference_fps, frames=self.stats)))


class ShmImageReader:
    """The reader of vision_road/image_share.py's block (the same layout, kept here so this file runs on its own)."""
    HEADER = struct.Struct("<4sIQiIIII28x")

    def __init__(self, name: str) -> None:
        self.name = name
        self.shm = None
        self.last = 0

    def read(self):
        from multiprocessing import shared_memory
        if self.shm is None:
            try:
                self.shm = shared_memory.SharedMemory(self.name)
            except FileNotFoundError:
                return None
            try:  # only a user of the block: Python's resource tracker must not unlink it at exit (Python < 3.13)
                from multiprocessing import resource_tracker
                resource_tracker.unregister(self.shm._name, "shared_memory")
            except Exception:
                pass
        buf = self.shm.buf
        magic, _, seq, sec, nanosec, w, h, enc = self.HEADER.unpack_from(buf, 0)
        if magic != b"VRIM" or seq % 2 or seq == self.last or not w or not h:
            return None
        frame = np.frombuffer(buf, np.uint8, w * h * 3, self.HEADER.size).reshape(h, w, 3).copy()
        if self.HEADER.unpack_from(buf, 0)[2] != seq:
            return None
        self.last = seq
        return sec, nanosec, ("rgb8" if enc == 1 else "bgr8"), frame


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = Town05Yolo26mDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
