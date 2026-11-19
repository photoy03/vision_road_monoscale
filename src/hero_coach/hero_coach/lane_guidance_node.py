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
     With signs.enabled: /hero/traffic_light (SignalIndicator) and /hero/sign (SignMarker) -> the scene's has_signal /
     signal and has_sign / sign, as hero_coach's coach_node does (traffic_sign_bridge makes them from YOLO).
     With traffic.enabled: "traffic" in the JSON (traffic_advice.py: advice, distance, deceleration, speed limit),
     and a stop advice ends the ribbon at the stop line.
     /hero/lane_guidance/ribbon (hero_hud/TrajectoryRibbon, base_link): the guidance ribbon every frame (empty when
     none), for coach_node's lane_ribbon_topic; publish_scene false leaves /hero/scene_overlay to coach_node.
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
from rclpy.executors import ExternalShutdownException
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
        # publish_scene: this node sends /hero/scene_overlay itself (the laptop's lane-only HUD).  false: it does not
        # (with hero_coach's coach_node on the car, which takes the ribbon below as its ideal when it has no parking plan).
        self.publish_scene = bool(p('publish_scene', True))
        from hero_hud.msg import TrajectoryRibbon
        self.TrajectoryRibbon = TrajectoryRibbon
        # The lane guidance ribbon on its own, every frame (no points while no guidance is shown).
        self.ribbon_pub = self.create_publisher(TrajectoryRibbon, p('ribbon_topic', '/hero/lane_guidance/ribbon'), 10)
        # signs.enabled: the traffic light and speed sign as coach_node takes them (plan 29): /hero/traffic_light
        # (SignalIndicator) and /hero/sign (SignMarker), from traffic_sign_bridge (YOLO) or carla_traffic; older than
        # signs.timeout_sec (1 s, as coach_node) or STATE_NONE / KIND_UNKNOWN: none.  Off: no subscription.
        self.signs = None
        if p('signs.enabled', False):
            from hero_hud.msg import SignalIndicator, SignMarker
            self.SignalIndicator, self.SignMarker = SignalIndicator, SignMarker
            self.signs = dict(signal=None, signal_at=None, sign=None, sign_at=None, timeout=float(p('signs.timeout_sec', 1.)))
            self.create_subscription(SignalIndicator, p('signs.signal_topic', '/hero/traffic_light'), self.on_signal, 10)
            self.create_subscription(SignMarker, p('signs.sign_topic', '/hero/sign'), self.on_sign, 10)
        # traffic.enabled: light + stop line advice and the speed limit (traffic_advice.py) as "traffic" in the JSON; a stop
        # advice ends the guidance ribbon at the stop line.  Needs signs.enabled (and stop_line.enabled for distances).
        self.traffic = None
        if p('traffic.enabled', False):
            from .traffic_advice import AdviceSettings, SpeedLimit, advise, cut_at, STOP_ADVICE
            advice = AdviceSettings()
            for name in advice.__dataclass_fields__:
                setattr(advice, name, type(getattr(advice, name))(p('traffic.'+name, getattr(advice, name))))
            self.traffic = dict(settings=advice, limit=SpeedLimit(advice), advise=advise, cut_at=cut_at, stop=STOP_ADVICE)
        # stop_line.enabled: the nearest stop line ahead (vision_road /road/road_markings, ns 'stop_line', base_link) as
        # stop_line_m in /hero/lane_guidance, for a later red light + stop line rule.  Off: nothing changes.
        self.stop_lines = None
        if p('stop_line.enabled', False):
            from visualization_msgs.msg import MarkerArray
            self.stop_line_max_age = float(p('stop_line.max_age_s', .35))
            # Plan 30: with no stop line ahead, the near edge of a crosswalk ahead is where to stop (a junction with a
            # crossing and no stop line); traffic.stop_source says which ('line' / 'crosswalk').  Off by default.
            self.crosswalk_fallback = bool(p('stop_line.crosswalk_fallback', False))
            self.stop_lines = (None, [], [])  # (stamp, [nearest x of each stop line], [nearest x of each crosswalk])
            self.create_subscription(MarkerArray, p('stop_line.topic', '/road/road_markings'), self.on_markings, 10)
        self.get_logger().info('Lane guidance: '+str(settings))

    def on_markings(self, msg):
        if not msg.markers:return
        h = msg.markers[0].header.stamp;xs = [];cs = []
        for m in msg.markers:
            if m.action == 0 and m.ns == 'stop_line' and m.points:xs.append(min(float(q.x) for q in m.points))
            if m.action == 0 and m.ns == 'crosswalk' and m.points:cs.append(min(float(q.x) for q in m.points))
        self.stop_lines = (h.sec+h.nanosec*1e-9, xs, cs)

    def stop_line_ahead(self, t):
        """(distance m from the rear axle along x to where to stop at t, or None; 'line' / 'crosswalk' / None)."""
        stamp, xs, cs = self.stop_lines
        if stamp is None or abs(stamp-t) > self.stop_line_max_age:return None, None
        ahead = [x for x in xs if x > 0.]
        if ahead:return round(min(ahead), 2), 'line'
        ahead = [x for x in cs if x > 0.] if self.crosswalk_fallback else []
        return (round(min(ahead), 2), 'crosswalk') if ahead else (None, None)

    def on_signal(self, msg):
        self.signs['signal'] = msg;self.signs['signal_at'] = time.monotonic()

    def on_sign(self, msg):
        self.signs['sign'] = msg;self.signs['sign_at'] = time.monotonic()

    def signs_now(self):
        """(light 'red'/'yellow'/'green' or None, speed sign km/h or None), as coach_node fills its scene."""
        s = self.signs;now = time.monotonic();light = sign = None
        fresh = lambda at: at is not None and now-at < s['timeout']
        if s['signal'] is not None and fresh(s['signal_at']):
            light = {self.SignalIndicator.STATE_RED: 'red', self.SignalIndicator.STATE_YELLOW: 'yellow',
                     self.SignalIndicator.STATE_GREEN: 'green'}.get(s['signal'].state)
        if s['sign'] is not None and fresh(s['sign_at']) and s['sign'].kind == self.SignMarker.KIND_SPEED:sign = float(s['sign'].value)
        return light, sign

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
        light = sign = None
        if self.signs is not None:
            light, sign = self.signs_now()
            if light is not None:
                scene.has_signal = True;scene.signal = self.signs['signal']
            if sign is not None:
                scene.has_sign = True;scene.sign = self.signs['sign']
        stop_line, stop_source = self.stop_line_ahead(t) if self.stop_lines is not None else (None, None)
        traffic = None;path = out.path if out.shown else None
        if self.traffic is not None:
            tr = self.traffic;limit, over = tr['limit'].update(t, sign, out.speed_mps)
            traffic = dict(light=light, stop_line_m=stop_line, **tr['advise'](light, stop_line, out.speed_mps, tr['settings']),
                           speed_limit_kph=limit, over_limit=over)
            if self.crosswalk_fallback:traffic['stop_source'] = stop_source
            # A red or yellow light with a stop advice: the guidance ribbon ends at the stop line (no path past it).
            if path is not None and light in ('red', 'yellow') and traffic['advice'] in tr['stop'] and stop_line is not None:
                path = tr['cut_at'](list(path), stop_line) or None
            traffic['ribbon_cut'] = path is not (out.path if out.shown else None)
        ribbon = self.TrajectoryRibbon();ribbon.width_m = self.width;ribbon.confidence = 1.
        if path is not None:
            ribbon.points = [Point32(x=float(x), y=float(y), z=0.) for x, y in path]
            scene.has_ideal = True;scene.ideal = ribbon
        if out.shown and out.path is not None:  # the guidance text also when the ribbon was cut away entirely
            text, severity = TEXT[out.state]
            scene.has_coach = True;scene.coach.text = text;scene.coach.severity = severity;scene.coach.ttl_sec = .5
        self.ribbon_pub.publish(ribbon)
        if self.publish_scene:self.scene_pub.publish(scene)
        compute = (time.perf_counter()-begin)*1e3
        extra = dict(light=light, speed_sign=sign) if self.signs is not None else {}  # only with signs.enabled
        if self.stop_lines is not None:extra['stop_line_m'] = stop_line  # only with stop_line.enabled
        if self.stop_lines is not None and self.crosswalk_fallback:extra['stop_source'] = stop_source
        if traffic is not None:extra['traffic'] = traffic  # only with traffic.enabled
        self.info_pub.publish(String(data=json.dumps(dict(
            stamp=t, state=out.state, target=out.target, offset_m=round(out.offset_m, 3), lateral_speed_mps=round(out.lateral_speed_mps, 3),
            speed_mps=round(out.speed_mps, 2), speed_source='vehicle_state' if speed is not None else 'pose', shown=out.shown, merge_m=round(out.merge_m, 1), changes=out.changes, reason=out.reason,
            compute_ms=round(compute, 3), **extra, path=[] if out.path is None else [[round(float(x), 2), round(float(y), 3)] for x, y in out.path]),
            separators=(',', ':'))))


def main():
    rclpy.init();node = LaneGuidanceNode()
    try:rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):pass  # Ctrl+C or a launch shutdown
    finally:
        node.destroy_node();rclpy.try_shutdown()
