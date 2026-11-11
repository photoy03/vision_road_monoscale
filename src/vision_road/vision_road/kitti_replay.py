"""Deterministic KITTI playback: /clock, image, CameraInfo, IMU and OXTS reference."""
import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, Imu
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from .kitti import KittiSequence, replay_events
from .rig import quaternion_xyzw


def stamp(t):
    ns = round(float(t)*1e9)
    return Time(sec=ns//1000000000, nanosec=ns%1000000000)


class KittiReplay(Node):
    def __init__(self):
        super().__init__('kitti_replay')
        def p(n,d): return self.declare_parameter(n,d).value
        self.data = KittiSequence(p('sequence',''), p('imu_height_m',.93))
        self.speed = p('rate',1.0); self.delay = p('start_delay_sec',2.0)
        self.limit = p('max_frames',0)
        if not math.isfinite(self.speed) or self.speed <= 0 or self.delay < 0 or self.limit < 0:
            raise ValueError('rate must be positive; delay/max_frames must be nonnegative')
        self.pose_topic = p('pose_topic','/localization/kinematic_state')
        self.publish_pose_tf = p('publish_pose_tf',True)
        self.image_pub = self.create_publisher(Image,'/kitti/image_02/image_rect',QoSProfile(depth=30))
        self.info_pub = self.create_publisher(CameraInfo,'/kitti/image_02/camera_info',qos_profile_sensor_data)
        self.imu_pub = self.create_publisher(Imu,'/kitti/imu',100)
        self.odom_pub = self.create_publisher(Odometry,self.pose_topic,200)
        self.clock_pub = self.create_publisher(Clock,'/clock',10)
        self.static = StaticTransformBroadcaster(self); self.dynamic = TransformBroadcaster(self)
        transforms = []
        for child, translation, rotation in [('kitti_camera_rect02',self.data.t_base_cam[:3,3],self.data.t_base_cam[:3,:3]),
                                               ('imu_link',np.array([0,0,self.data.imu_height]),np.eye(3))]:
            msg = TransformStamped(); msg.header = Header(stamp=stamp(1),frame_id='base_link'); msg.child_frame_id = child
            msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = map(float,translation)
            q = quaternion_xyzw(rotation)
            msg.transform.rotation.x,msg.transform.rotation.y,msg.transform.rotation.z,msg.transform.rotation.w = map(float,q)
            transforms.append(msg)
        self.static.sendTransform(transforms)
        count = min(self.limit or len(self.data.images),len(self.data.images))
        self.events = replay_events(self.data.image_times[:count], self.data.oxts_times)
        self.cursor = 0; self.bridge = CvBridge(); self.started = time.monotonic()+self.delay
        self.finished = False
        self.timer = self.create_timer(.005,self.tick)
        self.get_logger().info(f'KITTI ready: {count} frames, rate={self.speed}, pose={self.pose_topic}; start in {self.delay}s')

    def tick(self):
        if time.monotonic() < self.started:
            self.clock_pub.publish(Clock(clock=stamp(1)))
            return
        target = 1+(time.monotonic()-self.started)*self.speed
        # Do not skip source events. Limit each callback to keep services responsive.
        budget = 20
        while self.cursor < len(self.events) and self.events[self.cursor][0] <= target and budget:
            delivery, kind, i, acquired = self.events[self.cursor]
            self.clock_pub.publish(Clock(clock=stamp(delivery)))
            self.publish_oxts(i,acquired) if kind == 0 else self.publish_image(i,acquired)
            self.cursor += 1; budget -= 1
        if self.cursor == len(self.events):
            self.timer.cancel(); self.finished = True
            self.get_logger().info('KITTI playback complete. Last /clock retained; Ctrl+C to exit.')

    def publish_image(self,i,t):
        msg = self.bridge.cv2_to_imgmsg(self.data.image(i),'bgr8')
        msg.header = Header(stamp=stamp(t),frame_id='kitti_camera_rect02')
        info = CameraInfo(); info.header = msg.header; info.width,info.height = self.data.size
        info.distortion_model = 'plumb_bob'; info.d = [0.]*5; info.k = self.data.k.ravel().tolist()
        info.r = np.eye(3).ravel().tolist(); info.p = np.column_stack([self.data.k,np.zeros(3)]).ravel().tolist()
        self.info_pub.publish(info); self.image_pub.publish(msg)

    def publish_oxts(self,i,t):
        row = self.data.oxts[i]; transform = self.data.poses[i]
        msg = Odometry(); msg.header = Header(stamp=stamp(t),frame_id='map'); msg.child_frame_id = 'base_link'
        msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z = map(float,transform[:3,3])
        q = quaternion_xyzw(transform[:3,:3]); orientation = msg.pose.pose.orientation
        orientation.x,orientation.y,orientation.z,orientation.w = map(float,q)
        # Body-frame velocity of IMU shifted to the ground-frame origin (lever arm).
        omega = row[17:20]
        velocity = np.array(row[8:11])+np.cross(omega,[0,0,-self.data.imu_height])
        msg.twist.twist.linear.x,msg.twist.twist.linear.y,msg.twist.twist.linear.z = map(float,velocity)
        msg.twist.twist.angular.x,msg.twist.twist.angular.y,msg.twist.twist.angular.z = map(float,omega)
        self.odom_pub.publish(msg)
        imu = Imu(); imu.header = Header(stamp=stamp(t),frame_id='imu_link')
        qi = quaternion_xyzw(self.data.imu_rotations[i])
        imu.orientation.x,imu.orientation.y,imu.orientation.z,imu.orientation.w = map(float,qi)
        imu.angular_velocity = msg.twist.twist.angular
        imu.linear_acceleration.x,imu.linear_acceleration.y,imu.linear_acceleration.z = map(float,row[11:14])
        self.imu_pub.publish(imu)
        if self.publish_pose_tf:
            tf = TransformStamped(); tf.header = msg.header; tf.child_frame_id = 'base_link'
            tf.transform.translation.x,tf.transform.translation.y,tf.transform.translation.z = map(float,transform[:3,3])
            tf.transform.rotation = orientation; self.dynamic.sendTransform(tf)


def main():
    rclpy.init(); node = KittiReplay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node(); rclpy.try_shutdown()
