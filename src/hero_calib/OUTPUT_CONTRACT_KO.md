# HERO_Calib ROS 2 출력 약속 (`--ros-output`)

운전 노드(이 폴더의 `prius_wheel_based_driver_eye.py --wheel-drive --ros-output`)가 CARLA 컴퓨터에서 낼 토픽이다. 이 컴퓨터에서는 돌릴 수 없으므로, 우리 쪽(차선 인식, 안내, HUD)을 시험할 때는 이 약속대로 가짜로 낸다. 구현은 `hero_ros_output.py` 에 있다.

공통:
- 시각: 모든 머리말 시각은 CARLA 시뮬레이터 시각이다. `/clock` 도 같은 시각이다.
- 좌표: ROS 오른손 좌표(x 앞, y 왼쪽, z 위). 각도는 라디안이다. **왼쪽(반시계)이 양수**다.
- QoS: 모두 reliable 이다.

| 토픽 | 형식 | 주기 | 내용, 단위, 부호 | 기준점 / frame |
|---|---|---|---|---|
| `/clock` | `rosgraph_msgs/Clock` | 서버 틱마다 (CARLA 비동기 모드, 서버 fps) | 시뮬레이터 시각 | – |
| `/carla/ground_truth/pose` | `geometry_msgs/PoseWithCovarianceStamped` | 서버 틱마다 | 위치(m), 자세(쿼터니언). CARLA 좌표에서 y 와 yaw 부호를 바꿈. 공분산 0 | frame `map`. 기준점은 차 액터 원점이다. **미확정**: Prius 원점에서 뒤축까지 거리(차선 인식은 1.4096 m 로 가정, 실제 값은 도구가 시작할 때 출력), 원점이 땅 높이인지 |
| `/hero/vehicle_state` | `hero_hud/VehicleState` | 서버 틱마다 | 아래 표 참고 | frame `base_link` (뒤축) |
| `/hero/steering_wheel` | `sensor_msgs/JointState` | 서버 틱마다 | 아래 표 참고 | frame `base_link` |
| `/sensing/camera/dashcam/image_raw` | `sensor_msgs/Image` | 30 Hz (시뮬레이터 시각, `sensor_tick` 1/30) | bgr8, 1920×1080, step 5760 | frame `dashcam_optical_link`: 뒤축에서 앞 2.2 m, 땅에서 1.4 m, 정면. **미확정**: 실제 Prius 에서의 자리(뒷바퀴 위치와 경계 상자로 계산), 노출 등 카메라 속성 |
| `/sensing/camera/dashcam/camera_info` | `sensor_msgs/CameraInfo` | 영상마다 | K = [960, 0, 960; 0, 960, 540; 0, 0, 1] (수평 90°), plumb_bob, 왜곡 0 | frame `dashcam_optical_link` |

대시캠 행 자르기 (`--ros-crop-rows LO:HI`, 기본은 자르지 않음):
- 차선 인식은 영상의 일부 행만 읽는다. 지금 BEV(앞 3–35 m, 옆 ±6 m, 화면 아래 0.915 까지)는 행 580–981 만 쓴다. 그 행만 보내면 보내고 받는 양이 37% 로 준다.
- 자르면 `image_raw` 는 행 LO 부터 HI−1 까지만 담는다(`height` = HI−LO, `width` 1920, `step` 5760 그대로).
- `camera_info` 는 전체 크기(`height` 1080, `width` 1920)와 같은 K 를 그대로 두고, `roi` 에 어느 행인지 적는다: `roi.y_offset` = LO, `roi.height` = HI−LO, `roi.x_offset` = 0, `roi.width` = 1920. 자르지 않으면 `roi` 는 모두 0 이다(ROS 의 뜻: 전체 영상).
- 받는 쪽(차선 인식 `road_node`, 계산 프로세스)은 `<영상 토픽의 이름공간>/camera_info` 의 `roi` 로 행을 맞춘다. 읽는 행을 모두 담지 않은 영상, 또는 `camera_info` 를 아직 받지 못한 잘린 영상은 버리고 경고를 낸다. 받는 쪽은 시작할 때 읽는 행을 로그에 적는다("the BEV reads image rows 580-981 of 1920x1080").
- 권장 값: `--ros-crop-rows 578:984` (읽는 행 위아래로 2행 여유). 카메라 자리, 보정, BEV 범위를 바꾸면 행도 바뀌므로 받는 쪽 로그를 보고 다시 정한다.

`/hero/vehicle_state`:

| 필드 | 단위 | 부호 / 값 | 출처 |
|---|---|---|---|
| gear | – | D(4) / R(2): 도구의 후진 버튼 전환. `--wheel-drive` 가 아니면 P(1) | 도구 상태 |
| speed_mps | m/s | 앞 +, 후진 − | 차 속도의 진행 방향 성분 |
| steering_angle_rad | rad | 앞바퀴 조향각, 왼쪽 + | CARLA 실제 바퀴 각도(FL, FR 평균). 없으면 명령 × 최대각(37.8°) |
| yaw_rate_rps | rad/s | 왼쪽 회전 + | CARLA 각속도 z 의 부호를 바꿈 |

`/hero/steering_wheel` (JointState):

| name | position | velocity |
|---|---|---|
| `steering_wheel` | 운전자가 돌린 핸들 각도(rad, 왼쪽 +). T300RS 축(−1..1) × 540°, 끝은 ±540° | 프레임 간 변화(rad/s) |
| `road_wheel` | 앞바퀴 조향각(rad, 왼쪽 +). vehicle_state 와 같은 값 | 프레임 간 변화(rad/s) |

참고 관계: 핸들 각도 ≈ 바퀴 각도 × 14.27(도구의 `WHEEL_STEERING_RATIO`, 핸들 540° 에서 바퀴 37.8°). 데드존(축 0.01, 약 5°) 안에서는 바퀴가 0 이다.

미확정 항목 정리:
- 위치의 기준점: 액터 원점에서 뒤축까지 거리. Prius 값이 1.4096 m 와 같은지 모른다.
- 액터 원점의 높이.
- 대시캠의 실제 자리와 카메라 속성.
- 비동기 모드에서 실제 주기와, 영상과 위치의 시각 차이.
- carla 와 rclpy 가 같은 파이썬에서 함께 도는지.
