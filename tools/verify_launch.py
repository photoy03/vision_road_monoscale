#!/usr/bin/env python3
"""Launch graph and verify live MonoScale policy, then stop our own children."""
import os
from pathlib import Path
import signal
import subprocess
import time
import rclpy
from rcl_interfaces.srv import GetParameters
from rclpy.parameter import parameter_value_to_python


def main():
    log_path = Path(os.environ.get('ROAD_VERIFY_LOG', '/tmp/vision-road-live-verify.log'))
    with log_path.open('w') as log:
        launch = subprocess.Popen(['ros2', 'launch', 'vision_road', 'road.launch.py'], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        rclpy.init(); client = rclpy.create_node('road_launch_verifier')
        try:
            expected = {'vision_road', 'feature_tracker', 'monoscale_odometry', 'lane_odom_stabilizer'}
            until = time.monotonic()+20
            while time.monotonic() < until:
                rclpy.spin_once(client, timeout_sec=.1)
                if expected <= set(client.get_node_names()): break
            assert expected <= set(client.get_node_names()), client.get_node_names()
            for remote, checks in {
                '/feature_tracker': {'road_from_step': True, 'road_step_esm': True, 'cameras': ['front'], 'motion_reference': 'front'},
                '/monoscale_odometry': {'cameras': ['front'], 'imu_yaw_from_gyro': False, 'fusion_model': 'velocity', 'publish_tf': False,
                                         'odometry_topic': '/localization/monoscale_raw'},
                '/vision_road': {'odom_topic': '/localization/kinematic_state'}
            }.items():
                params = client.create_client(GetParameters, remote+'/get_parameters')
                assert params.wait_for_service(timeout_sec=5), remote
                future = params.call_async(GetParameters.Request(names=list(checks)))
                rclpy.spin_until_future_complete(client, future, timeout_sec=5)
                assert future.done(), remote
                values = [parameter_value_to_python(x) for x in future.result().values]
                assert values == list(checks.values()), (remote, values, checks)
            print('PASS: four nodes discovered across processes; photometric ESM, front-only calibration path and odometry policy verified.')
        finally:
            client.destroy_node(); rclpy.shutdown()
            launch.send_signal(signal.SIGINT)
            try: launch.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(launch.pid, signal.SIGTERM); launch.wait(timeout=5)
    print('Launch log:', log_path)


if __name__ == '__main__': main()
