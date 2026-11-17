#!/usr/bin/env python3
"""Fake vehicle state from the car's pose: what the driving node (src/hero_calib, --ros-output) will send.

Until that node runs on the CARLA computer, this stands in for its /hero/vehicle_state and
/hero/steering_wheel (src/hero_calib/OUTPUT_CONTRACT_KO.md), computed from the rear-axle pose the lane
detector already gets (/road/reference_odom: the recording's /carla/ground_truth/pose moved to the rear
axle, or a script's shifted pose):

  speed_mps            the displacement along the heading per second (reverse: negative), low-passed
  yaw_rate_rps         the heading change per second (left +), low-passed
  steering_angle_rad   the front road-wheel angle of a kinematic bicycle on that path:
                       atan(wheelbase * yaw_rate / speed), wheelbase 2.70 m (Prius gen 3, as the HUD's
                       vehicle.wheelbase_m); 0 below 0.5 m/s
  steering_wheel       road-wheel angle times 14.27 (the driving tool's WHEEL_STEERING_RATIO), clipped to
                       +-540 deg (the T300RS stop)
  gear                 D (the recordings drive forward)

Only past poses are used (a live stand-in): the values lag the motion by about the low-pass time.
usage: ros2 run ... / python3 tools/fake_vehicle_state.py [--odom-topic T] [--tau S]"""
import argparse
import math
from dataclasses import dataclass

WHEELBASE_M = 2.70          # Toyota Prius gen 3 (ZVW30); the HUD's vehicle.wheelbase_m
STEERING_RATIO = 14.27      # src/hero_calib WHEEL_STEERING_RATIO (540 deg handwheel = 37.8 deg road wheel)
HANDWHEEL_LIMIT_RAD = math.radians(540.)


@dataclass
class Estimate:
    speed_mps: float
    yaw_rate_rps: float
    curvature: float
    steering_rad: float
    handwheel_rad: float


class VehicleStateEstimator:
    """Pose samples (t, x, y, yaw) in, Estimate out."""

    def __init__(self, tau=.1, wheelbase=WHEELBASE_M, ratio=STEERING_RATIO, min_speed=.5):
        self.tau, self.wheelbase, self.ratio, self.min_speed = tau, wheelbase, ratio, min_speed
        self.reset()

    def reset(self):
        self.last = None; self.speed = 0.; self.yaw_rate = 0.

    def update(self, t, x, y, yaw):
        if self.last is not None:
            t0, x0, y0, yaw0 = self.last; dt = t-t0
            if not 0. < dt <= .5:self.reset()
            else:
                dx, dy = x-x0, y-y0; along = dx*math.cos(yaw0)+dy*math.sin(yaw0)
                dyaw = math.atan2(math.sin(yaw-yaw0), math.cos(yaw-yaw0))
                a = 1.-math.exp(-dt/self.tau)
                self.speed += (along/dt-self.speed)*a; self.yaw_rate += (dyaw/dt-self.yaw_rate)*a
        self.last = (t, x, y, yaw)
        k = self.yaw_rate/self.speed if abs(self.speed) >= self.min_speed else 0.
        steer = math.atan(self.wheelbase*k)
        hand = max(-HANDWHEEL_LIMIT_RAD, min(HANDWHEEL_LIMIT_RAD, steer*self.ratio))
        return Estimate(self.speed, self.yaw_rate, k, steer, hand)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--odom-topic', default='/road/reference_odom')
    ap.add_argument('--tau', type=float, default=.1)
    ap.add_argument('--rate', type=float, default=30., help='publish at most this often (Hz, simulation time); every pose still updates the estimate')
    a, ros_args = ap.parse_known_args()
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import JointState
    from hero_hud.msg import VehicleState
    rclpy.init(args=ros_args); node = rclpy.create_node('fake_vehicle_state'); est = VehicleStateEstimator(a.tau)
    reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
    pub_state = node.create_publisher(VehicleState, '/hero/vehicle_state', reliable)
    pub_wheel = node.create_publisher(JointState, '/hero/steering_wheel', reliable)
    last = {}

    def on_odom(msg):
        q = msg.pose.pose.orientation; yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        t = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        if est.last is not None and t < est.last[0]:est.reset()  # replay restarted
        e = est.update(t, msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)
        if 'sent' in last and 0. <= t-last['sent'] < 1./a.rate-1e-6:return  # the poses come at 60 Hz; 30 Hz is enough
        last['sent'] = t
        state = VehicleState(); state.header.stamp = msg.header.stamp; state.header.frame_id = 'base_link'
        state.gear = state.GEAR_DRIVE; state.speed_mps = e.speed_mps; state.steering_angle_rad = e.steering_rad; state.yaw_rate_rps = e.yaw_rate_rps
        pub_state.publish(state)
        joint = JointState(); joint.header = state.header; joint.name = ['steering_wheel', 'road_wheel']
        joint.position = [e.handwheel_rad, e.steering_rad]
        if 't' in last and t > last['t']:
            joint.velocity = [(e.handwheel_rad-last['h'])/(t-last['t']), (e.steering_rad-last['s'])/(t-last['t'])]
        last.update(t=t, h=e.handwheel_rad, s=e.steering_rad); pub_wheel.publish(joint)
    node.create_subscription(Odometry, a.odom_topic, on_odom, QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT))
    node.get_logger().info(f'Fake vehicle state from {a.odom_topic} (wheelbase {WHEELBASE_M} m, ratio {STEERING_RATIO})')
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.destroy_node();rclpy.try_shutdown()


if __name__ == '__main__':
    main()
