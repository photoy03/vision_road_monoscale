"""monoscale 격자·오도메트리 + 계획 궤적 → /hero/scene_overlay.

HUD 는 바보다. 시간 동기, 좌표 변환, "지금 어느 구간인가"는 여기서 끝낸다.
발행은 벽시계 10 Hz — HUD 의 장면 타임아웃이 벽시계이기 때문이다.
"""

import time

from autoware_planning_msgs.msg import Trajectory
from geometry_msgs.msg import Point32
from hero_hud.msg import (
    ObstacleMarker, SceneOverlay, SignalIndicator, SignMarker, SourceHealth,
    TrajectoryRibbon, VehicleState,
)
from nav_msgs.msg import OccupancyGrid, Odometry
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node

from hero_coach.core import (
    UNKNOWN, OdometryWatch, SegmentTracker, advised_gear, densify, grid_obstacles,
    ribbon_points, split_segments, threat_of,
)
from hero_coach.lane_ribbon import lane_ideal
from hero_coach.ros_io import (
    LATCHED, LATEST, declare_footprint, ego_from_odometry, grid_array, trajectory_waypoints,
)


class CoachNode(Node):
    def __init__(self):
        super().__init__('coach_node')
        declare = self.declare_parameter
        declare('grid_topic', '/perception/occupancy_grid_map')
        declare('odometry_topic', '/localization/kinematic_state')
        declare('trajectory_topic', '/hero_coach/trajectory')
        declare('output_topic', '/hero/scene_overlay')
        declare('rate_hz', 10.0)
        declare('obstacle_threshold', 65)
        declare('obstacle_bin_m', 0.3)
        declare('window_behind_m', 10.0)
        declare('window_ahead_m', 15.0)
        declare('window_lateral_m', 6.0)
        declare('arrive_m', 0.5)
        declare('ribbon_step_m', 0.25)
        declare('vehicle_topic', '/hero/vehicle_state')
        # 신호등과 표지판. 시뮬에서는 carla_traffic 이, 실차에서는 영상 검출이
        # 채운다 -- 코치는 어느 쪽인지 묻지 않는다.
        declare('signal_topic', '/hero/traffic_light')
        declare('sign_topic', '/hero/sign')
        # 이보다 오래된 것은 없는 셈 친다. 신호가 바뀌었는데 옛 색을 띄우는
        # 것이 아무것도 안 띄우는 것보다 나쁘다.
        declare('signal_timeout_sec', 1.0)
        # 정상 주기의 지식은 상류에 있다(SourceHealth.msg). 격자는 타이머 5 Hz,
        # 오도메트리는 누적 시차마다 — 기준은 core.OdometryWatch 가 달린 거리로 정한다.
        declare('occupancy_timeout_sec', 2.0)
        # vision-road 사본에 더함(계획 29): 차로 안내 띠(lane_guidance 의 /hero/lane_guidance/ribbon). 주차 계획이 없거나
        # 지금 구간이 없을 때, 이보다 새로운 띠를 ideal 로 싣는다. 빈 문자열(기본)이면 원본과 같다.
        declare('lane_ribbon_topic', '')
        declare('lane_ribbon_timeout_sec', 0.5)
        self.p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.footprint = declare_footprint(self)

        self.grid = None
        self.grid_at = None
        self.signal = None
        self.signal_at = None
        self.sign = None
        self.sign_at = None
        self.ego = None
        self.odometry = OdometryWatch()
        self.tracker = None
        self.lane_ribbon = None
        self.lane_ribbon_at = None

        self.pub = self.create_publisher(SceneOverlay, str(self.p('output_topic')), LATEST)
        self.create_subscription(OccupancyGrid, str(self.p('grid_topic')), self.on_grid, LATCHED)
        self.create_subscription(Odometry, str(self.p('odometry_topic')), self.on_odometry, LATEST)
        self.create_subscription(
            VehicleState, str(self.p('vehicle_topic')), self.on_vehicle, LATEST)
        self.create_subscription(
            SignalIndicator, str(self.p('signal_topic')), self.on_signal, LATEST)
        self.create_subscription(
            SignMarker, str(self.p('sign_topic')), self.on_sign, LATEST)
        self.create_subscription(
            Trajectory, str(self.p('trajectory_topic')), self.on_trajectory, LATCHED)
        if str(self.p('lane_ribbon_topic')):
            self.create_subscription(
                TrajectoryRibbon, str(self.p('lane_ribbon_topic')), self.on_lane_ribbon, LATEST)
        self.create_timer(1.0 / float(self.p('rate_hz')), self.tick,
                          clock=Clock(clock_type=ClockType.STEADY_TIME))

    def on_grid(self, msg: OccupancyGrid):
        self.grid = grid_array(msg)
        self.grid_at = time.monotonic()

    def on_odometry(self, msg: Odometry):
        self.ego = ego_from_odometry(msg)
        self.odometry.on_odometry(time.monotonic())

    def on_signal(self, message):
        self.signal = message
        self.signal_at = time.monotonic()

    def on_sign(self, message):
        self.sign = message
        self.sign_at = time.monotonic()

    def on_lane_ribbon(self, message):
        self.lane_ribbon = message
        self.lane_ribbon_at = time.monotonic()

    def on_vehicle(self, msg: VehicleState):
        self.odometry.on_speed(time.monotonic(), msg.speed_mps)

    def on_trajectory(self, msg: Trajectory):
        waypoints = densify(trajectory_waypoints(msg), float(self.p('ribbon_step_m')))
        self.tracker = SegmentTracker(split_segments(waypoints), float(self.p('arrive_m')))

    @staticmethod
    def fresh(at, timeout) -> bool:
        return at is not None and time.monotonic() - at < timeout

    def tick(self):
        out = SceneOverlay()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'base_link'
        out.advised_gear = UNKNOWN

        now = time.monotonic()
        # 신호등과 표지판은 인지·계획과 무관하게 실린다. 격자가 없어도, 목표가
        # 없어도 빨간불은 띄워야 한다 -- 그것이 가장 먼저 안 보이는 것이다.
        timeout = float(self.p('signal_timeout_sec'))
        if (self.signal is not None and self.fresh(self.signal_at, timeout)
                and self.signal.state != SignalIndicator.STATE_NONE):
            out.has_signal = True
            out.signal = self.signal
        if (self.sign is not None and self.fresh(self.sign_at, timeout)
                and self.sign.kind != SignMarker.KIND_UNKNOWN):
            out.has_sign = True
            out.sign = self.sign

        odometry_ok = self.odometry.ok(now)
        occupancy_ok = self.fresh(self.grid_at, float(self.p('occupancy_timeout_sec')))
        grid_age = -1.0 if self.grid_at is None else now - self.grid_at
        # age -1: 한 번도 안 왔다
        for name, age, ok in (('odometry', self.odometry.age(now), odometry_ok),
                              ('occupancy', grid_age, occupancy_ok)):
            health = SourceHealth()
            health.name = name
            health.age_sec = float(age)
            health.ok = ok
            out.health.append(health)

        if odometry_ok and occupancy_ok:
            self.fill_obstacles(out)
        if odometry_ok and self.tracker is not None:
            self.fill_plan(out)
        if str(self.p('lane_ribbon_topic')):
            plan_active = self.tracker is not None and self.tracker.current is not None
            age = None if self.lane_ribbon_at is None else now - self.lane_ribbon_at
            ribbon = lane_ideal(plan_active, self.lane_ribbon, age,
                                float(self.p('lane_ribbon_timeout_sec')))
            if ribbon is not None and not out.has_ideal:
                out.has_ideal = True
                out.ideal = ribbon
        self.pub.publish(out)

    def fill_obstacles(self, out: SceneOverlay):
        data, res, origin = self.grid
        obstacles = grid_obstacles(
            data, res, origin, self.ego, self.footprint,
            threshold=int(self.p('obstacle_threshold')),
            x_range=(-float(self.p('window_behind_m')), float(self.p('window_ahead_m'))),
            half_lateral=float(self.p('window_lateral_m')),
            bin_m=float(self.p('obstacle_bin_m')))
        for o in obstacles:
            marker = ObstacleMarker()
            marker.position = Point32(x=o.x, y=o.y, z=0.0)
            marker.extent_m = o.extent
            marker.threat = threat_of(o.distance)
            marker.source = ObstacleMarker.SOURCE_FUSED
            marker.confidence = 1.0
            out.obstacles.append(marker)

    def fill_plan(self, out: SceneOverlay):
        tracker = self.tracker
        tracker.update(self.ego[0], self.ego[1])
        out.advised_gear = advised_gear(tracker)
        segment = tracker.current
        if segment is None:
            return
        points = ribbon_points(segment, tracker.progress, self.ego)
        if len(points) < 2:
            return
        out.has_ideal = True
        out.ideal = TrajectoryRibbon(
            points=[Point32(x=x, y=y, z=0.0) for x, y in points],
            width_m=2.0 * self.footprint.half_width,
            confidence=1.0)


def main():
    rclpy.init()
    node = CoachNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
