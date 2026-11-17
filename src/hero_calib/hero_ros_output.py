"""ROS 2 outputs for prius_wheel_based_driver_eye.py (--ros-output), added beside it.

The driving tool itself is unchanged: its inputs, parameters, calibration, physics, eye camera and
window are the same with or without this.  With --ros-output it also publishes what the lane
detector (vision_road) and the HUD (hero_hud) read, in the Town05 recording's names and conventions:

  /clock                                rosgraph_msgs/Clock            simulation time, every world tick
  /carla/ground_truth/pose              geometry_msgs/PoseWithCovarianceStamped  frame 'map', every tick:
                                        the vehicle actor's location and rotation, CARLA's left-handed axes
                                        turned right-handed (y and yaw negated), as in the recording
  /hero/vehicle_state                   hero_hud/VehicleState          every tick: gear (D/R from the tool's
                                        reverse toggle; P while the tool holds the car fixed), signed speed
                                        (m/s, reverse negative), front road-wheel angle (rad, left +),
                                        yaw rate (rad/s, left +)
  /hero/steering_wheel                  sensor_msgs/JointState         every tick: 'steering_wheel' = the
                                        handwheel angle the driver turned (rad, left +, from the T300RS axis),
                                        'road_wheel' = the front road-wheel angle (rad, left +)
  /sensing/camera/dashcam/image_raw     sensor_msgs/Image              bgr8 1920x1080, 30 Hz sim time
  /sensing/camera/dashcam/camera_info   sensor_msgs/CameraInfo         fx = fy = 960 (90 deg HFOV), plumb_bob

The dashcam is a second RGB camera rigidly attached where the lane detector expects it: 2.2 m ahead of
the rear axle (base_link) and 1.4 m above the ground, looking straight ahead (vision_road
maploop.launch.py static TF base_link -> dashcam_optical_link).  The rear axle is taken from the
vehicle's rear wheel positions, the ground from the bottom of its bounding box.

rclpy and the message packages are imported only by start(): without ROS the tool runs as before and
--ros-output only prints a warning."""
from __future__ import annotations

import math
import threading
from typing import Any, Optional, Sequence, Tuple

DASHCAM_FORWARD_M = 2.2   # ahead of the rear axle (base_link), vision_road maploop.launch.py static TF
DASHCAM_HEIGHT_M = 1.4    # above the ground
DASHCAM_WIDTH = 1920
DASHCAM_HEIGHT = 1080
DASHCAM_HFOV_DEG = 90.0   # fx = fy = 960 at 1920 px, as the Town05 recording's camera_info
DASHCAM_RATE_HZ = 30.0
DASHCAM_FRAME = "dashcam_optical_link"
WORLD_FRAME = "map"


# --------------------------------------------------------------------------- conversions (pure)
def ros_position(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """CARLA (x forward/east, y right/south, z up; left-handed) to ROS right-handed: y negated."""
    return (float(x), -float(y), float(z))


def ros_rpy(roll_deg: float, pitch_deg: float, yaw_deg: float) -> Tuple[float, float, float]:
    """CARLA rotation (degrees, left-handed) to ROS roll, pitch, yaw (radians, right-handed)."""
    return (math.radians(roll_deg), -math.radians(pitch_deg), -math.radians(yaw_deg))


def quaternion(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """(x, y, z, w) of roll-pitch-yaw (ZYX)."""
    cr, sr = math.cos(roll*.5), math.sin(roll*.5)
    cp, sp = math.cos(pitch*.5), math.sin(pitch*.5)
    cy, sy = math.cos(yaw*.5), math.sin(yaw*.5)
    return (sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy)


def road_wheel_angle_rad(left_deg: float, right_deg: float) -> float:
    """CARLA front wheel steer angles (degrees, + = right) to one road-wheel angle (rad, left +)."""
    return -math.radians(0.5*(float(left_deg)+float(right_deg)))


def handwheel_angle_rad(axis: float, range_deg: float) -> float:
    """T300RS steering axis (-1..1, + = right) to the handwheel angle (rad, left +): the axis spans the
    wheel's whole rotation range (range_deg lock to lock)."""
    axis = max(-1., min(1., float(axis)))
    return -math.radians(axis*range_deg*.5)


def signed_speed(velocity: Sequence[float], forward: Sequence[float]) -> float:
    """Longitudinal speed (m/s): the velocity along the vehicle's forward vector (reverse: negative)."""
    return float(sum(v*f for v, f in zip(velocity, forward)))


def yaw_rate_rps(angular_z_deg_s: float) -> float:
    """CARLA angular velocity about z (deg/s, left-handed: + turns right) to ROS (rad/s, + turns left)."""
    return -math.radians(float(angular_z_deg_s))


def rear_axle_local(wheels: Sequence[Tuple[float, float, float, float]]) -> Tuple[float, float]:
    """(x, y) of the rear axle centre in vehicle-local metres from wheels (x, y, z, max_steer_deg),
    their positions in vehicle-local metres: the mean of the non-steering wheels (else the rear half)."""
    rear = [w for w in wheels if abs(w[3]) < 1e-6] or sorted(wheels, key=lambda w: w[0])[:max(1, len(wheels)//2)]
    return (sum(w[0] for w in rear)/len(rear), sum(w[1] for w in rear)/len(rear))


def dashcam_local(rear_x: float, ground_z: float) -> Tuple[float, float, float]:
    """Dashcam location in vehicle-local metres: 2.2 m ahead of the rear axle, on the centre line, 1.4 m
    above the ground."""
    return (rear_x+DASHCAM_FORWARD_M, 0., ground_z+DASHCAM_HEIGHT_M)


def camera_matrix(width: int = DASHCAM_WIDTH, height: int = DASHCAM_HEIGHT, hfov_deg: float = DASHCAM_HFOV_DEG):
    """K (row-major 3x3) of CARLA's pinhole RGB camera: f = (width / 2) / tan(HFOV / 2), square pixels,
    principal point at the image centre."""
    f = (width*.5)/math.tan(math.radians(hfov_deg)*.5)
    return [f, 0., width*.5, 0., f, height*.5, 0., 0., 1.]


def bgra_to_bgr(raw: Any, width: int, height: int) -> bytes:
    """CARLA's BGRA pixels to packed bgr8 (numpy when available)."""
    try:
        import numpy as np
        return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)[:, :, :3].tobytes()
    except ImportError:
        raw = bytes(raw); bgr = bytearray(width*height*3)
        bgr[0::3] = raw[0::4]; bgr[1::3] = raw[1::4]; bgr[2::3] = raw[2::4]
        return bytes(bgr)


def stamp_of(seconds: float) -> Tuple[int, int]:
    ns = int(round(float(seconds)*1e9))
    return ns//1000000000, ns % 1000000000


# --------------------------------------------------------------------------- publisher
class RosOutput:
    """Started by start(); set_drive() from the tool's driving loop; close() before the vehicle goes."""

    def __init__(self, world: Any, vehicle: Any, args: Any, carla_module: Any, rclpy_module: Any, msgs: dict,
                 range_deg: float):
        self.world = world; self.vehicle = vehicle; self.carla = carla_module; self.rclpy = rclpy_module
        self.msgs = msgs; self.range_deg = range_deg
        self.axis: Optional[float] = None; self.reverse = False
        self.lock = threading.Lock(); self.snapshot = None; self.new = threading.Event(); self.stop = threading.Event()
        self.camera = None; self.tick_id = None; self.max_steer_deg = 0.; self.last_wheel = None
        rclpy_module.init(args=None)
        self.node = rclpy_module.create_node("hero_calib_ros_output")
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        # Reliable: both reliable subscribers (vision_road live mode) and best-effort ones accept it.
        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        node = self.node
        self.pub_clock = node.create_publisher(msgs["Clock"], "/clock", 10)
        self.pub_pose = node.create_publisher(msgs["PoseWithCovarianceStamped"], "/carla/ground_truth/pose",
                                              QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE))
        self.pub_wheel = node.create_publisher(msgs["JointState"], "/hero/steering_wheel", reliable)
        self.pub_state = node.create_publisher(msgs["VehicleState"], "/hero/vehicle_state", reliable) if msgs.get("VehicleState") else None
        self.pub_image = node.create_publisher(msgs["Image"], "/sensing/camera/dashcam/image_raw", reliable)
        self.pub_info = node.create_publisher(msgs["CameraInfo"], "/sensing/camera/dashcam/camera_info", reliable)
        self.rear_x, self.ground_z = self._vehicle_geometry()
        print(f"ROS output: rear axle at x={self.rear_x:.4f} m, ground at z={self.ground_z:.4f} m (vehicle-local)")
        print(f"ROS output: vision_road reference_pose actor_to_rear_axle_m should be {-self.rear_x:.4f} "
              "(its default 1.4096 is the Town05 recording's vehicle)")
        if not getattr(args, "ros_no_dashcam", False):self.camera = self._spawn_dashcam()
        self.worker = threading.Thread(target=self._run, name="hero-ros-output", daemon=True); self.worker.start()
        self.tick_id = world.on_tick(self._on_tick)

    # vehicle geometry (rear axle from the wheels, ground from the bounding box)
    def _vehicle_geometry(self) -> Tuple[float, float]:
        transform = self.vehicle.get_transform(); physics = self.vehicle.get_physics_control(); wheels = []
        for wheel in physics.wheels:
            p = wheel.position  # world, centimetres
            local = transform.inverse_transform(self.carla.Location(x=p.x/100., y=p.y/100., z=p.z/100.))
            wheels.append((local.x, local.y, local.z, wheel.max_steer_angle))
            self.max_steer_deg = max(self.max_steer_deg, wheel.max_steer_angle)
        box = self.vehicle.bounding_box
        return rear_axle_local(wheels)[0], box.location.z-box.extent.z

    def _spawn_dashcam(self) -> Any:
        carla = self.carla
        blueprint = self.world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", str(DASHCAM_WIDTH))
        blueprint.set_attribute("image_size_y", str(DASHCAM_HEIGHT))
        blueprint.set_attribute("fov", str(DASHCAM_HFOV_DEG))
        blueprint.set_attribute("sensor_tick", str(1./DASHCAM_RATE_HZ))
        x, y, z = dashcam_local(self.rear_x, self.ground_z)
        camera = self.world.spawn_actor(blueprint, carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=0., yaw=0., roll=0.)),
                                        attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        camera.listen(self._on_image)
        print(f"ROS output: dashcam at vehicle-local ({x:.3f}, {y:.3f}, {z:.3f}) m, {DASHCAM_WIDTH}x{DASHCAM_HEIGHT}, "
              f"HFOV {DASHCAM_HFOV_DEG} deg, {DASHCAM_RATE_HZ} Hz")
        return camera

    # inputs
    def set_drive(self, axis: float, reverse: bool) -> None:
        """From the tool's driving loop: the raw steering axis and its reverse toggle."""
        with self.lock:self.axis = float(axis); self.reverse = bool(reverse)

    def _on_tick(self, snapshot: Any) -> None:
        # CARLA's callback thread: no client calls here, only hand the snapshot on.
        with self.lock:self.snapshot = snapshot
        self.new.set()

    # outputs
    def _header(self, msg: Any, seconds: float, frame: str) -> None:
        msg.header.stamp.sec, msg.header.stamp.nanosec = stamp_of(seconds); msg.header.frame_id = frame

    def _run(self) -> None:
        while not self.stop.is_set():
            if not self.new.wait(.5):continue
            self.new.clear()
            with self.lock:snapshot = self.snapshot; axis = self.axis; reverse = self.reverse
            try:self._publish_state(snapshot, axis, reverse)
            except Exception as exc:  # noqa: BLE001 - keep driving whatever ROS does
                print(f"WARNING: ROS output: {type(exc).__name__}: {exc}")

    def _publish_state(self, snapshot: Any, axis: Optional[float], reverse: bool) -> None:
        msgs = self.msgs; t = snapshot.timestamp.elapsed_seconds
        clock = msgs["Clock"](); clock.clock.sec, clock.clock.nanosec = stamp_of(t); self.pub_clock.publish(clock)
        actor = snapshot.find(self.vehicle.id)
        if actor is None:return
        tr = actor.get_transform(); v = actor.get_velocity(); w = actor.get_angular_velocity()
        pose = msgs["PoseWithCovarianceStamped"](); self._header(pose, t, WORLD_FRAME)
        x, y, z = ros_position(tr.location.x, tr.location.y, tr.location.z)
        qx, qy, qz, qw = quaternion(*ros_rpy(tr.rotation.roll, tr.rotation.pitch, tr.rotation.yaw))
        p = pose.pose.pose; p.position.x, p.position.y, p.position.z = x, y, z
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = qx, qy, qz, qw
        self.pub_pose.publish(pose)
        forward = tr.get_forward_vector()
        speed = signed_speed((v.x, v.y, v.z), (forward.x, forward.y, forward.z))
        # The front wheels' actual angle (CARLA >= 0.9.13); else the applied command times the lock.
        try:
            loc = self.carla.VehicleWheelLocation
            road = road_wheel_angle_rad(self.vehicle.get_wheel_steer_angle(loc.FL_Wheel), self.vehicle.get_wheel_steer_angle(loc.FR_Wheel))
        except (AttributeError, RuntimeError):
            road = -math.radians(self.vehicle.get_control().steer*self.max_steer_deg)
        if axis is None:  # not driving (the tool holds the car): the applied command
            handwheel = -math.radians(self.vehicle.get_control().steer*self.range_deg*.5); gear = "P"
        else:
            handwheel = handwheel_angle_rad(axis, self.range_deg); gear = "R" if reverse else "D"
        joint = msgs["JointState"](); self._header(joint, t, "base_link")
        joint.name = ["steering_wheel", "road_wheel"]; joint.position = [handwheel, road]
        if self.last_wheel is not None and t > self.last_wheel[0]:
            joint.velocity = [(handwheel-self.last_wheel[1])/(t-self.last_wheel[0]), (road-self.last_wheel[2])/(t-self.last_wheel[0])]
        self.last_wheel = (t, handwheel, road); self.pub_wheel.publish(joint)
        if self.pub_state is not None:
            state = msgs["VehicleState"](); self._header(state, t, "base_link")
            state.gear = {"P": state.GEAR_PARK, "R": state.GEAR_REVERSE, "D": state.GEAR_DRIVE}[gear]
            state.speed_mps = speed; state.steering_angle_rad = road; state.yaw_rate_rps = yaw_rate_rps(w.z)
            self.pub_state.publish(state)

    def _on_image(self, image: Any) -> None:
        # CARLA's sensor thread: BGRA -> bgr8 and publish, stamped with the frame's simulation time.
        try:
            msgs = self.msgs; t = image.timestamp; w, h = image.width, image.height
            bgr = bgra_to_bgr(image.raw_data, w, h)
            out = msgs["Image"](); self._header(out, t, DASHCAM_FRAME)
            out.height, out.width, out.encoding, out.is_bigendian, out.step = h, w, "bgr8", 0, w*3; out.data = bgr
            info = msgs["CameraInfo"](); self._header(info, t, DASHCAM_FRAME)
            info.height, info.width, info.distortion_model = h, w, "plumb_bob"
            k = camera_matrix(w, h, float(image.fov)); info.k = k; info.d = [0.]*5
            info.r = [1., 0., 0., 0., 1., 0., 0., 0., 1.]; info.p = [k[0], 0., k[2], 0., 0., k[4], k[5], 0., 0., 0., 1., 0.]
            self.pub_image.publish(out); self.pub_info.publish(info)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: ROS output dashcam: {type(exc).__name__}: {exc}")

    def close(self) -> None:
        self.stop.set(); self.new.set()
        if self.tick_id is not None:
            try:self.world.remove_on_tick(self.tick_id)
            except RuntimeError:pass
        self.worker.join(2.)
        if self.camera is not None:
            try:
                if self.camera.is_listening:self.camera.stop()
                if self.camera.is_alive:self.camera.destroy()
            except RuntimeError as exc:print(f"WARNING: dashcam cleanup failed: {exc}")
        try:self.node.destroy_node(); self.rclpy.try_shutdown()
        except Exception:pass  # noqa: BLE001
        print("ROS output stopped.")


def start(world: Any, vehicle: Any, args: Any, carla_module: Any, range_deg: float) -> Optional[RosOutput]:
    """RosOutput, or None (with a warning) when ROS 2 is not available: the tool then runs as without."""
    try:
        import rclpy
        from rosgraph_msgs.msg import Clock
        from geometry_msgs.msg import PoseWithCovarianceStamped
        from sensor_msgs.msg import Image, CameraInfo, JointState
    except ImportError as exc:
        print(f"WARNING: --ros-output: ROS 2 is not available ({exc}); running without ROS outputs")
        return None
    msgs = dict(Clock=Clock, PoseWithCovarianceStamped=PoseWithCovarianceStamped, Image=Image, CameraInfo=CameraInfo, JointState=JointState)
    try:
        from hero_hud.msg import VehicleState
        msgs["VehicleState"] = VehicleState
    except ImportError:
        print("WARNING: --ros-output: hero_hud messages not found (source the vision-road install); "
              "/hero/vehicle_state is not published")
    return RosOutput(world, vehicle, args, carla_module, rclpy, msgs, range_deg)
