"""hero_ros_output.py: conversions, and the publisher driven by a fake CARLA world (no simulator needed).
Needs ROS 2 (rclpy) and, for /hero/vehicle_state, the vision-road install sourced (hero_hud messages)."""
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hero_ros_output as R  # noqa: E402


def test_carla_axes_turn_right_handed():
    # A car heading 30 deg in CARLA (towards +y = right/south) heads -30 deg in ROS; its motion agrees.
    assert R.ros_position(10., 5., .3) == (10., -5., .3)
    roll, pitch, yaw = R.ros_rpy(0., 2., 30.)
    assert math.isclose(yaw, -math.radians(30.)) and math.isclose(pitch, -math.radians(2.))
    vx, vy, _ = R.ros_position(math.cos(math.radians(30.)), math.sin(math.radians(30.)), 0.)
    assert math.isclose(math.atan2(vy, vx), yaw)
    x, y, z, w = R.quaternion(0., 0., yaw)
    assert math.isclose(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)), yaw)


def test_steering_and_yaw_rate_are_left_positive():
    assert math.isclose(R.road_wheel_angle_rad(10., 12.), -math.radians(11.))  # CARLA + = right
    assert math.isclose(R.handwheel_angle_rad(.5, 1080.), -math.radians(270.))  # axis + = right
    assert math.isclose(R.handwheel_angle_rad(-2., 1080.), math.radians(540.))  # clipped at the stop
    assert math.isclose(R.yaw_rate_rps(10.), -math.radians(10.))  # CARLA yaw grows turning right
    assert R.signed_speed((-3., 0., 0.), (1., 0., 0.)) == -3.  # reversing


def test_rear_axle_dashcam_and_camera_info():
    wheels = [(1.37, -.78, .33, 70.), (1.37, .78, .33, 70.), (-1.33, -.78, .33, 0.), (-1.33, .78, .33, 0.)]
    assert R.rear_axle_local(wheels) == pytest.approx((-1.33, 0.))
    assert R.dashcam_local(-1.33, 0.) == pytest.approx((.87, 0., 1.4))
    k = R.camera_matrix()
    # The Town05 recording's camera_info: K = [960, 0, 960, 0, 960, 540, 0, 0, 1].
    assert k == pytest.approx([960., 0., 960., 0., 960., 540., 0., 0., 1.], abs=1e-9)
    assert R.stamp_of(2.5666667) == (2, 566666700)


def test_bgra_to_bgr():
    raw = bytes([1, 2, 3, 255, 4, 5, 6, 255])
    assert R.bgra_to_bgr(raw, 2, 1) == bytes([1, 2, 3, 4, 5, 6])


# ---------------------------------------------------------------- fake CARLA
class V:
    def __init__(self, x=0., y=0., z=0.):self.x, self.y, self.z = x, y, z


class Rot:
    def __init__(self, pitch=0., yaw=0., roll=0.):self.pitch, self.yaw, self.roll = pitch, yaw, roll


class Tr:
    def __init__(self, location, rotation):self.location, self.rotation = location, rotation

    def get_forward_vector(self):
        c, s = math.cos(math.radians(self.rotation.yaw)), math.sin(math.radians(self.rotation.yaw));return V(c, s, 0.)

    def transform(self, p):
        c, s = math.cos(math.radians(self.rotation.yaw)), math.sin(math.radians(self.rotation.yaw))
        return V(self.location.x+c*p.x-s*p.y, self.location.y+s*p.x+c*p.y, self.location.z+p.z)

    def inverse_transform(self, p):
        c, s = math.cos(math.radians(self.rotation.yaw)), math.sin(math.radians(self.rotation.yaw))
        dx, dy = p.x-self.location.x, p.y-self.location.y;return V(c*dx+s*dy, -s*dx+c*dy, p.z-self.location.z)


FAKE_CARLA = SimpleNamespace(Location=V, Rotation=Rot, Transform=lambda l, r: Tr(l, r),
                             VehicleWheelLocation=SimpleNamespace(FL_Wheel='FL', FR_Wheel='FR'),
                             AttachmentType=SimpleNamespace(Rigid='rigid'))


class Camera:
    def __init__(self, transform):self.transform, self.callback, self.is_listening, self.is_alive = transform, None, False, True
    def listen(self, cb):self.callback, self.is_listening = cb, True
    def stop(self):self.is_listening = False
    def destroy(self):self.is_alive = False


class Blueprint:
    def __init__(self):self.attributes = {}
    def set_attribute(self, k, v):self.attributes[k] = v


class World:
    def __init__(self):self.cb, self.camera, self.bp = None, None, Blueprint()
    def on_tick(self, cb):self.cb = cb;return 7
    def remove_on_tick(self, i):self.cb = None
    def get_blueprint_library(self):return SimpleNamespace(find=lambda name: self.bp)
    def spawn_actor(self, bp, transform, attach_to=None, attachment_type=None):self.camera = Camera(transform);return self.camera


class Vehicle:
    id = 42

    def __init__(self):
        self.tr = Tr(V(100., 50., 0.), Rot(yaw=30.))
        local = [(1.37, -.78, .33, 70.), (1.37, .78, .33, 70.), (-1.33, -.78, .33, 0.), (-1.33, .78, .33, 0.)]
        wheels = []
        for x, y, z, steer in local:
            p = self.tr.transform(V(x, y, z));wheels.append(SimpleNamespace(position=V(p.x*100., p.y*100., p.z*100.), max_steer_angle=steer))
        self.physics = SimpleNamespace(wheels=wheels)
        self.bounding_box = SimpleNamespace(location=V(0., 0., .75), extent=V(2.25, 1.0, .75))
    def get_transform(self):return self.tr
    def get_physics_control(self):return self.physics
    def get_wheel_steer_angle(self, where):return 10. if where == 'FL' else 12.
    def get_control(self):return SimpleNamespace(steer=.2, reverse=False)


def test_publisher_on_a_fake_world():
    rclpy = pytest.importorskip("rclpy")
    from sensor_msgs.msg import Image, CameraInfo, JointState
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from rosgraph_msgs.msg import Clock
    world, vehicle = World(), Vehicle()
    out = R.start(world, vehicle, SimpleNamespace(ros_no_dashcam=False), FAKE_CARLA, 1080.)
    assert out is not None
    try:
        # The dashcam: 2.2 m ahead of the rear axle (local x -1.33), 1.4 m above the ground (box bottom 0).
        loc = world.camera.transform.location
        assert (loc.x, loc.y, loc.z) == pytest.approx((.87, 0., 1.4))
        assert world.bp.attributes == {"image_size_x": "1920", "image_size_y": "1080", "fov": "90.0", "sensor_tick": str(1./30.)}
        listener = rclpy.create_node("test_listener");got = {}
        for topic, kind in (("/clock", Clock), ("/carla/ground_truth/pose", PoseWithCovarianceStamped), ("/hero/steering_wheel", JointState),
                            ("/sensing/camera/dashcam/image_raw", Image), ("/sensing/camera/dashcam/camera_info", CameraInfo)):
            listener.create_subscription(kind, topic, lambda m, t=topic: got.setdefault(t, m), 10)
        try:
            from hero_hud.msg import VehicleState
            listener.create_subscription(VehicleState, "/hero/vehicle_state", lambda m: got.setdefault("/hero/vehicle_state", m), 10)
            want_state = True
        except ImportError:
            want_state = False
        out.set_drive(-.25, False)  # handwheel 135 deg to the left
        actor = SimpleNamespace(get_transform=lambda: vehicle.tr, get_velocity=lambda: V(math.cos(math.radians(30.))*5., math.sin(math.radians(30.))*5., 0.),
                                get_angular_velocity=lambda: V(0., 0., -6.))
        snapshot = SimpleNamespace(timestamp=SimpleNamespace(elapsed_seconds=12.5), find=lambda i: actor if i == 42 else None)
        image = SimpleNamespace(timestamp=12.5333333, width=1920, height=1080, fov=90., raw_data=bytes([10, 20, 30, 255])*(1920*1080))
        end = time.monotonic()+5.
        while time.monotonic() < end and len(got) < (6 if want_state else 5):
            world.cb(snapshot);world.camera.callback(image);rclpy.spin_once(listener, timeout_sec=.05)
        pose = got["/carla/ground_truth/pose"]
        assert pose.header.frame_id == "map" and (pose.header.stamp.sec, pose.header.stamp.nanosec) == (12, 500000000)
        p, q = pose.pose.pose.position, pose.pose.pose.orientation
        assert (p.x, p.y, p.z) == pytest.approx((100., -50., 0.))
        assert math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z)) == pytest.approx(-math.radians(30.))
        assert got["/clock"].clock.sec == 12
        joint = got["/hero/steering_wheel"]
        assert list(joint.name) == ["steering_wheel", "road_wheel"]
        assert list(joint.position) == pytest.approx([math.radians(135.), -math.radians(11.)])
        img, info = got["/sensing/camera/dashcam/image_raw"], got["/sensing/camera/dashcam/camera_info"]
        assert (img.width, img.height, img.encoding, img.step, img.header.frame_id) == (1920, 1080, "bgr8", 5760, "dashcam_optical_link")
        assert bytes(img.data[:6]) == bytes([10, 20, 30, 10, 20, 30])
        assert (img.header.stamp.sec, img.header.stamp.nanosec) == (12, 533333300)
        assert list(info.k) == pytest.approx([960., 0., 960., 0., 960., 540., 0., 0., 1.]) and info.distortion_model == "plumb_bob"
        if want_state:
            st = got["/hero/vehicle_state"]
            assert st.gear == st.GEAR_DRIVE and st.speed_mps == pytest.approx(5., abs=1e-5)
            assert st.steering_angle_rad == pytest.approx(-math.radians(11.), abs=1e-6)
            assert st.yaw_rate_rps == pytest.approx(math.radians(6.), abs=1e-6)
        listener.destroy_node()
    finally:
        out.close()
    assert world.cb is None and not world.camera.is_alive


def test_without_ros_it_only_warns(monkeypatch, capsys):
    import builtins
    real = builtins.__import__

    def blocked(name, *a, **k):
        if name == "rclpy" or name.startswith("rclpy."):raise ImportError("no rclpy here")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", blocked)
    assert R.start(World(), Vehicle(), SimpleNamespace(), FAKE_CARLA, 1080.) is None
    assert "running without ROS outputs" in capsys.readouterr().out
