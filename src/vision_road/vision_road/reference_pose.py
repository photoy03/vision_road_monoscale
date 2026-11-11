"""Explicit evaluation-only actor truth -> rear-axle odometry adapter."""
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,ReliabilityPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry


def rear_axle(x,y,yaw,offset=1.4096):
    return x-offset*math.cos(yaw),y-offset*math.sin(yaw)


class ReferencePose(Node):
    def __init__(self):
        super().__init__('maploop_reference_pose')
        self.offset=self.declare_parameter('actor_to_rear_axle_m',1.4096).value
        self.pub=self.create_publisher(Odometry,'/road/reference_odom',20)
        self.create_subscription(PoseWithCovarianceStamped,'/carla/ground_truth/pose',self.on_pose,QoSProfile(depth=100,reliability=ReliabilityPolicy.BEST_EFFORT))
        self.get_logger().warning('Evaluation reference from CARLA truth; this is not estimated odometry')
    def on_pose(self,msg):
        p=msg.pose.pose.position;q=msg.pose.pose.orientation
        norm=math.sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w)
        if not math.isfinite(norm) or norm<1e-6:return
        x,y,z,w=q.x/norm,q.y/norm,q.z/norm,q.w/norm
        yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        bx,by=rear_axle(p.x,p.y,yaw,self.offset)
        if not all(math.isfinite(v) for v in [bx,by]):return
        out=Odometry();out.header=msg.header;out.header.frame_id='map';out.child_frame_id='base_link'
        out.pose=msg.pose;out.pose.pose.position.x=bx;out.pose.pose.position.y=by;out.pose.pose.position.z=0.
        self.pub.publish(out)


def main():
    rclpy.init();node=ReferencePose()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.destroy_node();rclpy.try_shutdown()
