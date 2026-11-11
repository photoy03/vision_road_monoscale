"""KITTI Raw synced/rectified: calibrated camera, fixed TF and selectable pose source."""
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from vision_road.kitti import KittiSequence


def build(context):
    def arg(n): return LaunchConfiguration(n).perform(context)
    data = KittiSequence(arg('sequence'),float(arg('imu_height_m')))
    mode = arg('odom_source')
    if mode not in ('oxts','monoscale'):
        raise ValueError('odom_source must be oxts or monoscale')
    out = data.export(arg('output_dir') or '/tmp/vision-road-kitti/'+data.path.name)
    share = Path(get_package_share_directory('vision_road'))
    nodes = [Node(package='vision_road',executable='road_node',name='vision_road',output='screen',parameters=[
        str(share/'config/road.yaml'),str(share/'config/kitti.yaml'),arg('params_file') or {},
        {'use_sim_time':True,'rig_file':str(out/'rig.yaml'),'calibration':str(out/'camera.npz'),
         'image_topic':'/kitti/image_02/image_rect','odom_topic':'/localization/kinematic_state'}]),
        Node(package='vision_road',executable='kitti_replay',name='kitti_replay',output='screen',parameters=[{
            'sequence':str(data.path),'imu_height_m':data.imu_height,'rate':float(arg('rate')),
            'start_delay_sec':float(arg('start_delay_sec')),'max_frames':int(arg('max_frames')),
            'publish_pose_tf':mode=='oxts','pose_topic':'/localization/kinematic_state' if mode=='oxts' else '/kitti/reference_odom'}])]
    if mode == 'monoscale':
        mono = Path(get_package_share_directory('monoscale_odometry'))
        base = str(mono/'config/vision_fisheye.param.yaml'); deploy = str(mono/'config/deployment.param.yaml')
        common = {'use_sim_time':True,'cameras':['front'],'image_topics':['/kitti/image_02/image_rect'],
                  'info_topics':['/kitti/image_02/camera_info'],'imu_topic':'/kitti/imu','motion_reference':'front',
                  'front.k':data.k.ravel().tolist(),'front.d':[0.]*5,'front.distortion_model':'plumb_bob',
                  'front.calibration_width':data.size[0],'front.calibration_height':data.size[1],
                  'front.translation_base_from_camera':data.rig['translation_base_from_camera'],
                  'front.rotation_base_from_camera':data.rig['rotation_base_from_camera'],
                  'use_cuda':False,'executor_threads':1,'input_reliability':'reliable'}
        nodes += [Node(package='monoscale_tracker',executable='feature_tracker',name='feature_tracker',output='screen',parameters=[base,deploy,common]),
                  Node(package='monoscale_odometry',executable='monoscale_odometry',name='monoscale_odometry',output='screen',parameters=[base,deploy,common,{
                      'use_camera_info':False,'extrinsics_from_tf':False,'publish_tf':False,'publish_map_to_odom':False,
                      'map_frame':'map','odom_frame':'odom','base_frame':'base_link','odometry_topic':'/localization/monoscale_raw',
                      'use_imu_yaw':True,'imu_yaw_from_gyro':False,'attitude_from_imu':True,'fusion_model':'velocity',
                      'use_inertial_prediction':False,'inertial_use_acceleration':False,'coast_on_reject':False,
                      'mapping_min_period_sec':0.2}]),
                  Node(package='monoscale_pose',executable='odom_stabilizer_node.py',name='lane_odom_stabilizer',output='screen',parameters=[
                      str(Path(get_package_share_directory('monoscale_pose'))/'config/odom_stabilizer.yaml'),
                      {'use_sim_time':True,'imu_topic':'/kitti/imu','imu_frame_id':'imu_link',
                       'max_linear_speed_mps':40.0,'max_translation_step_m':5.0}])]
    print('KITTI calibration and fixed TF exported to '+str(out))
    print('Pose source: '+mode+' (OXTS is a reference trajectory; MonoScale is an estimate)')
    return nodes


def generate_launch_description():
    defaults = {'imu_height_m':'0.93','odom_source':'oxts','rate':'1.0','start_delay_sec':'3.0',
                'max_frames':'0','output_dir':'','params_file':''}
    return LaunchDescription([DeclareLaunchArgument('sequence',description='Absolute KITTI Raw *_sync directory'),
        *[DeclareLaunchArgument(n,default_value=v) for n,v in defaults.items()],OpaqueFunction(function=build)])
