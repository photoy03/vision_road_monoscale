"""End-to-end replay through ROS input/output topics using a synthetic KITTI rig."""
import os
import signal
import subprocess
import time
import numpy as np
import pytest
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
from rclpy.parameter import parameter_value_to_python
from rcl_interfaces.srv import GetParameters
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from vision_lane_interfaces.msg import MarkingSegmentArray


@pytest.mark.parametrize('mode',['oxts','monoscale'])
def test_kitti_launch_topics_tf_and_model(kitti_sample,tmp_path,mode):
    sequence,k,cam_base=kitti_sample
    log=tmp_path/(mode+'.log')
    env=os.environ.copy(); env.setdefault('ROS_LOG_DIR',str(tmp_path/'roslogs'))
    with log.open('w') as stream:
        process=subprocess.Popen(['ros2','launch','vision_road','kitti.launch.py',f'sequence:={sequence}',
                                  f'odom_source:={mode}',f'output_dir:={tmp_path/mode}',
                                  'start_delay_sec:=2.0','max_frames:=12','rate:=0.5'],
                                 stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        rclpy.init(); reader=Node('kitti_test_reader')
        local=[]; maps=[]; overlays=[]; poses=[]; fixed=[]
        reference='/localization/kinematic_state' if mode=='oxts' else '/kitti/reference_odom'
        subscriptions=[reader.create_subscription(MarkingSegmentArray,'/parking/cpp/marking_segments',local.append,10),
                       reader.create_subscription(MarkingSegmentArray,'/parking/cpp/global_marking_segments_fused',maps.append,10),
                       reader.create_subscription(Image,'/road/overlay',overlays.append,qos_profile_sensor_data),
                       reader.create_subscription(Odometry,reference,poses.append,100),
                       reader.create_subscription(TFMessage,'/tf_static',fixed.append,QoSProfile(depth=10,durability=DurabilityPolicy.TRANSIENT_LOCAL))]
        try:
            deadline=time.monotonic()+18
            while time.monotonic()<deadline and len(local)<12:
                rclpy.spin_once(reader,timeout_sec=.05)
                if process.poll() is not None: break
            assert len(local)>=10,log.read_text()
            assert overlays and poses and any(m.segments for m in local),log.read_text()
            assert '/road/centerline' not in dict(reader.get_topic_names_and_types())
            camera=[t for msg in fixed for t in msg.transforms if t.child_frame_id=='kitti_camera_rect02']
            assert camera
            t=camera[-1].transform.translation
            assert np.allclose([t.x,t.y,t.z],np.linalg.inv(cam_base)[:3,3])
            if mode=='oxts':
                assert maps and maps[-1].segments and maps[-1].header.frame_id=='map'
            else:
                client=reader.create_client(GetParameters,'/feature_tracker/get_parameters')
                assert client.wait_for_service(timeout_sec=5)
                names=['front.distortion_model','front.k','front.translation_base_from_camera','road_from_step','road_step_esm']
                future=client.call_async(GetParameters.Request(names=names))
                rclpy.spin_until_future_complete(reader,future,timeout_sec=5)
                assert future.done()
                values=[parameter_value_to_python(v) for v in future.result().values]
                assert values[0]=='plumb_bob' and np.allclose(values[1],k.ravel())
                assert np.allclose(values[2],np.linalg.inv(cam_base)[:3,3]) and values[3:]==[True,True]
            assert (tmp_path/mode/'dataset.json').is_file()
        finally:
            reader.destroy_node(); rclpy.shutdown()
            process.send_signal(signal.SIGINT)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM); process.wait(timeout=5)
