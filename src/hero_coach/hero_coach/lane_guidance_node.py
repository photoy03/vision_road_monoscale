"""ROS wrapper of lane_guidance.py: lane return / lane change guidance for the HUD's lane-only mode.

In:  /road/lane_centres (std_msgs/String JSON, vision_road), /road/reference_odom (nav_msgs/Odometry:
     the rear-axle pose; any odometry of the car in a world frame will do), and /hero/vehicle_state
     (hero_hud/VehicleState) when there is one: its speed, if it came within vehicle_state_timeout_sec,
     replaces the speed taken from the pose change (merge length only).
Out: /hero/scene_overlay (hero_hud/SceneOverlay): has_ideal + ideal = the path (base_link, from the car,
     car width) while guidance is shown; has_coach with a short text and severity
     (return: INFO, change: CAUTION, forbidden: WARN) so the HUD can tell them apart.
     /hero/lane_guidance (std_msgs/String JSON): stamp, state (none/keep/return/change/forbidden),
     target (the target lane's current name), offset_m (car from the target centre, + left),
     lateral_speed_mps, speed_mps, shown, merge_m, changes (target moves so far), reason,
     compute_ms, light / speed_sign (what the scene shows, with signs.enabled), stop_line_m (nearest stop line
     ahead, m from the rear axle, or null; with stop_line.enabled), path [[x, y], ...].
     With signs.enabled: /yolo/detections (std_msgs/String JSON, the YOLO detector) -> the scene's has_signal /
     signal and has_sign / sign (traffic_signs.py).
A process of its own, apart from the lane detector: it only reads the detector's output topics.
NumPy's BLAS is held to one thread: its pool of spinning threads would take CPU from the detector for
matrices this small."""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):os.environ.setdefault(_name, '1')
import json
import time
import math
from collections import deque
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point32
from .lane_guidance import LaneGuidance, GuidanceSettings, Lane

TEXT = {'return': ('차로 중앙으로', 0), 'change': ('차선 변경', 1), 'forbidden': ('넘으면 안 되는 선', 2)}  # text, CoachingMessage severity


class LaneGuidanceNode(Node):
    def __init__(self):
        super().__init__('lane_guidance')
        p = lambda name, default: self.declare_parameter(name, default).value
        settings = GuidanceSettings()
        for name in settings.__dataclass_fields__:
            setattr(settings, name, type(getattr(settings, name))(p('guidance.'+name, getattr(settings, name))))
        self.core = LaneGuidance(settings); self.width = float(p('vehicle_width_m', 1.82))
        self.poses = deque(maxlen=400)
        from hero_hud.msg import SceneOverlay
        self.SceneOverlay = SceneOverlay
        self.scene_pub = self.create_publisher(SceneOverlay, p('scene_topic', '/hero/scene_overlay'), 10)
        self.info_pub = self.create_publisher(String, p('guidance_topic', '/hero/lane_guidance'), 10)
        self.create_subscription(Odometry, p('odom_topic', '/road/reference_odom'), self.on_odom,
                                 QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_subscription(String, p('centres_topic', '/road/lane_centres'), self.on_centres, 10)
        from hero_hud.msg import VehicleState
        self.vehicle = None;self.vehicle_timeout = float(p('vehicle_state_timeout_sec', .3))  # (speed, receipt time)
        self.create_subscription(VehicleState, p('vehicle_state_topic', '/hero/vehicle_state'), self.on_vehicle, 10)
        # signs.enabled: the traffic light state and speed sign from the YOLO detector (/yolo/detections, std_msgs/String
        # JSON) go into the same scene (one publisher of /hero/scene_overlay).  Off: no subscription, nothing changes.
        self.signs = None
        if p('signs.enabled', False):
            from .traffic_signs import SignTracker, SignSettings
            sign_settings = SignSettings()
            for name in sign_settings.__dataclass_fields__:
                setattr(sign_settings, name, type(getattr(sign_settings, name))(p('signs.'+name, getattr(sign_settings, name))))
            self.signs = SignTracker(sign_settings)
            self.create_subscription(String, p('signs.topic', '/yolo/detections'), self.on_detections, 10)
            self.get_logger().info('Traffic lights and speed signs from '+self.get_parameter('signs.topic').value+': '+str(sign_settings))
        # stop_line.enabled: the nearest stop line ahead (vision_road /road/road_markings, ns 'stop_line', base_link) as
        # stop_line_m in /hero/lane_guidance, for a later red light + stop line rule.  Off: nothing changes.
        self.stop_lines = None
        if p('stop_line.enabled', False):
            from visualization_msgs.msg import MarkerArray
            self.stop_line_max_age = float(p('stop_line.max_age_s', .35))
            self.stop_lines = (None, [])  # (stamp, [nearest x of each stop line])
            self.create_subscription(MarkerArray, p('stop_line.topic', '/road/road_markings'), self.on_markings, 10)
        self.get_logger().info('Lane guidance: '+str(settings))

    def on_markings(self, msg):
        if not msg.markers:return
        h = msg.markers[0].header.stamp;xs = []
        for m in msg.markers:
            if m.action == 0 and m.ns == 'stop_line' and m.points:xs.append(min(float(q.x) for q in m.points))
        self.stop_lines = (h.sec+h.nanosec*1e-9, xs)

    def stop_line_ahead(self, t):
        """Distance (m, from the rear axle along x) to the nearest stop line ahead of the car at t, or None."""
        stamp, xs = self.stop_lines
        if stamp is None or abs(stamp-t) > self.stop_line_max_age:return None
        ahead = [x for x in xs if x > 0.]
        return round(min(ahead), 2) if ahead else None

    def on_detections(self, msg):
        data = json.loads(msg.data);stamp = data.get('stamp', {})
        self.signs.update(float(stamp.get('sec', 0))+float(stamp.get('nanosec', 0))*1e-9, data.get('detections', []))

    def on_vehicle(self, msg):
        self.vehicle = (float(msg.speed_mps), time.monotonic())

    def on_odom(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        t = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        if self.poses and t < self.poses[-1][0]:self.poses.clear()  # replay restarted
        self.poses.append((t, msg.pose.pose.position.x, msg.pose.pose.position.y, yaw))

    def pose_at(self, t):
        """The pose at t, interpolated between the odometry samples around it (None when not covered)."""
        if not self.poses or t < self.poses[0][0]-.05 or t > self.poses[-1][0]+.1:return None
        prev = None
        for sample in self.poses:
            if sample[0] >= t:
                if prev is None:return sample[1:]
                u = (t-prev[0])/max(sample[0]-prev[0], 1e-9)
                yaw = prev[3]+math.atan2(math.sin(sample[3]-prev[3]), math.cos(sample[3]-prev[3]))*u
                return (prev[1]+(sample[1]-prev[1])*u, prev[2]+(sample[2]-prev[2])*u, yaw)
            prev = sample
        return self.poses[-1][1:]

    def on_centres(self, msg):
        begin = time.perf_counter()
        data = json.loads(msg.data); t = float(data['stamp'])
        if self.core.time is not None and t < self.core.time:self.core.reset()
        lanes = [Lane.from_message(item) for item in data['lanes']]
        speed = self.vehicle[0] if self.vehicle is not None and time.monotonic()-self.vehicle[1] <= self.vehicle_timeout else None
        out = self.core.update(t, self.pose_at(t), lanes, path=False, speed=speed)  # the path only while shown
        sec = int(t); nanosec = int(round((t-sec)*1e9))
        if nanosec >= 1000000000:sec += 1;nanosec -= 1000000000
        scene = self.SceneOverlay();scene.header.stamp.sec = sec;scene.header.stamp.nanosec = nanosec
        scene.header.frame_id = data.get('frame_id', 'base_link')
        if out.shown and out.path is not None:
            scene.has_ideal = True;scene.ideal.width_m = self.width;scene.ideal.confidence = 1.
            scene.ideal.points = [Point32(x=float(x), y=float(y), z=0.) for x, y in out.path]
            text, severity = TEXT[out.state]
            scene.has_coach = True;scene.coach.text = text;scene.coach.severity = severity;scene.coach.ttl_sec = .5
        light = sign = None
        if self.signs is not None:
            light, sign = self.signs.state(t)
            if light is not None:
                scene.has_signal = True;scene.signal.state = {'red': scene.signal.STATE_RED, 'yellow': scene.signal.STATE_YELLOW, 'green': scene.signal.STATE_GREEN}[light]
            if sign is not None:
                scene.has_sign = True;scene.sign.kind = scene.sign.KIND_SPEED;scene.sign.value = float(sign);scene.sign.confidence = 1.
        self.scene_pub.publish(scene)
        compute = (time.perf_counter()-begin)*1e3
        extra = dict(light=light, speed_sign=sign) if self.signs is not None else {}  # only with signs.enabled
        if self.stop_lines is not None:extra['stop_line_m'] = self.stop_line_ahead(t)  # only with stop_line.enabled
        self.info_pub.publish(String(data=json.dumps(dict(
            stamp=t, state=out.state, target=out.target, offset_m=round(out.offset_m, 3), lateral_speed_mps=round(out.lateral_speed_mps, 3),
            speed_mps=round(out.speed_mps, 2), speed_source='vehicle_state' if speed is not None else 'pose', shown=out.shown, merge_m=round(out.merge_m, 1), changes=out.changes, reason=out.reason,
            compute_ms=round(compute, 3), **extra, path=[] if out.path is None else [[round(float(x), 2), round(float(y), 3)] for x, y in out.path]),
            separators=(',', ':'))))


def main():
    rclpy.init();node = LaneGuidanceNode()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.destroy_node();rclpy.try_shutdown()
