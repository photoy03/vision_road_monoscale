"""Recorded dashcam profile; optional, explicitly named truth reference adapter."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    share=get_package_share_directory('vision_road')
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time',default_value='true'),
        DeclareLaunchArgument('reference_pose',default_value='true'),
        DeclareLaunchArgument('image_reliability',default_value='reliable'),
        DeclareLaunchArgument('performance_report',default_value=''),
        DeclareLaunchArgument('bev_mpp',default_value='0.07'),
        DeclareLaunchArgument('debug_publish_hz',default_value='5.0'),
        DeclareLaunchArgument('display_simplify_error_m',default_value='0.025'),
        Node(package='vision_road',executable='road_node',name='vision_road',output='screen',parameters=[share+'/config/carla.yaml',{
            'use_sim_time':LaunchConfiguration('use_sim_time'),
            'image_topic':'/sensing/camera/dashcam/image_raw','image_reliability':LaunchConfiguration('image_reliability'),'image_queue_depth':10,
            'odom_topic':'/road/reference_odom','base_frame':'base_link','world_frame':'map',
            'calibration':share+'/config/calibration/maploop_dashcam.npz','rig_file':share+'/config/maploop_rig.yaml',
            'image_roi_bottom':.915,'map_enabled':False,'async_processing':True,'isolated_compute':True,'overlay_max_width':960,
            'performance_report':ParameterValue(LaunchConfiguration('performance_report'),value_type=str),
            'debug_publish_hz':ParameterValue(LaunchConfiguration('debug_publish_hz'),value_type=float),
            'mpp':ParameterValue(LaunchConfiguration('bev_mpp'),value_type=float),
            'display_simplify_error_m':ParameterValue(LaunchConfiguration('display_simplify_error_m'),value_type=float)}]),
        Node(package='vision_road',executable='reference_pose',condition=IfCondition(LaunchConfiguration('reference_pose')),
             parameters=[{'use_sim_time':LaunchConfiguration('use_sim_time')}]),
        Node(package='tf2_ros',executable='static_transform_publisher',arguments=[
            '--x','2.2','--y','0','--z','1.4','--qx','-0.5','--qy','0.5','--qz','-0.5','--qw','0.5',
            '--frame-id','base_link','--child-frame-id','dashcam_optical_link'])])
