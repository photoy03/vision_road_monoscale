"""Front-camera road graph. External images/IMU by default; no parking nodes."""
from pathlib import Path
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from vision_road.rig import read_calibration, rotation_base_from_camera


def build(context):
    def arg(n): return LaunchConfiguration(n).perform(context)
    def enabled(n): return arg(n).lower() == 'true'
    share = Path(get_package_share_directory('vision_road'))
    rig_path = Path(arg('rig_file') or share/'config/rig.yaml')
    calibration = Path(arg('calibration') or share/'config/calibration/front.npz')
    rig = yaml.safe_load(rig_path.read_text())
    if rig.get('imu_axes_aligned_with_base') is not True:
        raise ValueError('MonoScale requires IMU axes aligned with base_link')
    k, d, size = read_calibration(calibration)
    camera = {'front.k': k.flatten().tolist(), 'front.d': d.tolist(),
              'front.calibration_width': size[0], 'front.calibration_height': size[1],
              'front.distortion_model': 'equidistant',
              'front.translation_base_from_camera': [float(rig[x]) for x in ('camera_forward_m', 'camera_left_m', 'camera_height_m')],
              'front.rotation_base_from_camera': rotation_base_from_camera(rig['pitch_deg'], rig['roll_deg'], rig['yaw_deg']).flatten().tolist()}
    image = arg('image_topic'); imu = arg('imu_topic'); sim = enabled('use_sim_time')
    nodes = [Node(package='vision_road', executable='road_node', name='vision_road', output='screen', parameters=[
        str(share/'config/road.yaml'), arg('params_file') or {},
        {'calibration': str(calibration), 'rig_file': str(rig_path), 'use_sim_time': sim,
         'image_topic': image, 'odom_topic': arg('odom_topic')}])]
    if enabled('launch_monoscale'):
        mono = Path(get_package_share_directory('monoscale_odometry'))
        base = str(mono/'config/vision_fisheye.param.yaml'); deploy = str(mono/'config/deployment.param.yaml')
        common = {'use_sim_time': sim, 'cameras': ['front'], 'image_topics': [image],
                  'info_topics': ['/sensing/camera/front/fisheye/camera_info'], **camera,
                  'imu_topic': imu, 'motion_reference': 'front', 'use_cuda': False}
        nodes += [Node(package='monoscale_tracker', executable='feature_tracker', name='feature_tracker', output='screen', parameters=[base, deploy, common]),
                  Node(package='monoscale_odometry', executable='monoscale_odometry', name='monoscale_odometry', output='screen', parameters=[base, deploy, common,
                       {'use_camera_info': False, 'extrinsics_from_tf': False, 'publish_tf': False,
                        'publish_map_to_odom': False, 'map_frame': 'map', 'odom_frame': 'odom',
                        'base_frame': 'base_link', 'odometry_topic': '/localization/monoscale_raw',
                        'mapping_min_period_sec': 0.2, 'use_imu_yaw': True,
                        'imu_yaw_from_gyro': False, 'attitude_from_imu': True,
                        'use_inertial_prediction': False, 'inertial_use_acceleration': False,
                        'fusion_model': 'velocity', 'coast_on_reject': False}]),
                  Node(package='monoscale_pose', executable='odom_stabilizer_node.py', name='lane_odom_stabilizer', output='screen', parameters=[
                       str(Path(get_package_share_directory('monoscale_pose'))/'config/odom_stabilizer.yaml'),
                       {'use_sim_time': sim, 'imu_topic': imu, 'output_odom_topic': arg('odom_topic')}])]
    if enabled('launch_camera'):
        nodes += [Node(package='gscam', executable='gscam_node', name='road_camera', output='screen', parameters=[{
            'gscam_config': 'v4l2src device='+arg('device')+' ! image/jpeg,width=2560,height=1440,framerate=30/1 ! jpegdec ! videoconvert ! video/x-raw,format=RGB',
            'image_encoding': 'rgb8', 'frame_id': 'front_fisheye_camera_optical_frame',
            'camera_name': 'front_fisheye', 'use_sensor_data_qos': True, 'sync_sink': False}],
            remappings=[('/camera/image_raw', image), ('/camera/camera_info', '/sensing/camera/front/fisheye/camera_info')])]
    if enabled('launch_imu'):
        nodes.append(Node(package='monoscale_ublox', executable='f9r_imu', name='f9r_imu', output='screen',
            parameters=[{'device': arg('imu_device'), 'frame_id': 'imu_link', 'gyro_yaw_fallback': True,
                         'gyro_bias_calibration_sec': 2.0, 'gyro_bias_stationary_threshold_rad_s': 0.035,
                         'gyro_yaw_rate_tau_sec': 0.12}], remappings=[('/sensing/imu/imu_data', imu)]))
    return nodes


def generate_launch_description():
    args = {'use_sim_time': 'false', 'launch_monoscale': 'true', 'launch_camera': 'false',
            'launch_imu': 'false', 'imu_device': '/dev/ttyACM0', 'device': '/dev/video2', 'image_topic': '/sensing/camera/front/fisheye/image_raw',
            'imu_topic': '/sensing/imu/imu_data', 'odom_topic': '/localization/kinematic_state',
            'rig_file': '', 'calibration': '', 'params_file': ''}
    return LaunchDescription([*[DeclareLaunchArgument(k, default_value=v) for k, v in args.items()], OpaqueFunction(function=build)])
