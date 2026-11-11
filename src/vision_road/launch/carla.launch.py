"""CARLA-only road paint node; explicit pinhole profile, no MonoScale dependency."""
from pathlib import Path
import math,tempfile
import numpy as np,yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def build(context):
    def arg(name):return LaunchConfiguration(name).perform(context)
    width,height=int(arg('width')),int(arg('height'))
    fov=float(arg('horizontal_fov'));camera_height=float(arg('camera_height'));pitch=float(arg('pitch'))
    if width<2 or height<2 or not 20<fov<160 or not math.isfinite(camera_height) or camera_height<=0 or not math.isfinite(pitch):raise ValueError('Invalid CARLA camera profile')
    out=Path(tempfile.mkdtemp(prefix='vision_road_carla_'));f=width/(2*math.tan(math.radians(fov/2)))
    np.savez(out/'camera.npz',K=[[f,0,width/2],[0,f,height/2],[0,0,1]],D=np.zeros(5),image_size=[width,height])
    rig=dict(distortion_model='plumb_bob',pitch_deg=pitch,roll_deg=0.,yaw_deg=0.,camera_forward_m=float(arg('camera_forward')),camera_left_m=0.,camera_height_m=camera_height)
    (out/'rig.yaml').write_text(yaml.safe_dump(rig))
    share=Path(get_package_share_directory('vision_road'))
    return [Node(package='vision_road',executable='road_node',name='vision_road',output='screen',parameters=[str(share/'config/carla.yaml'),dict(calibration=str(out/'camera.npz'),rig_file=str(out/'rig.yaml'),image_topic=arg('image_topic'),odom_topic=arg('odom_topic'),use_sim_time=arg('use_sim_time').lower()=='true',map_enabled=arg('map_enabled').lower()=='true')])]


def generate_launch_description():
    args=dict(width='1280',height='720',horizontal_fov='90',camera_height='1.5',pitch='0',camera_forward='0',image_topic='/carla/ego_vehicle/rgb_front/image',odom_topic='/carla/ego_vehicle/odometry',use_sim_time='true',map_enabled='false')
    return LaunchDescription([*[DeclareLaunchArgument(k,default_value=v) for k,v in args.items()],OpaqueFunction(function=build)])
