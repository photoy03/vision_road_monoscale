# 계획 22: 운전 도구 HERO_Calib 노드화 (출력만 더함)

날짜: 2026-11-18. 사본: `src/hero_calib/`. 원본: `/home/jaehun/Desktop/HERO_Calib-main`(읽기만 함). 복사 직후 커밋: `74a0732`.

## 한 일

1. 원본을 `__pycache__` 를 빼고 그대로 복사했다. 원본과 같음을 diff 로 확인했고, 복사 직후 상태를 lane-commit 이 따로 커밋했다(18개).
2. 사본에 출력만 더했다.

| 파일 | 변경 | 지운 줄·바꾼 줄 |
|---|---|---|
| `prius_wheel_based_driver_eye.py` | 더한 줄 27(빈 줄 2 포함) | 0 |
| `README.md` | 끝에 "ROS 2 outputs" 절 101줄 | 0 |
| `hero_ros_output.py` | 새 파일 (출력 전부) | – |
| `tests/test_ros_output.py` | 새 파일 (시험 6개) | – |

`prius_wheel_based_driver_eye.py` 에 더한 것 (`driver_eye.diff`):

- `hero_ros_output` 를 가져오는 줄. 없으면 None 이 된다.
- 전역 `ROS_OUTPUT = None`.
- 새 선택 사항 둘, 기본 꺼짐.
  - `--ros-output`: ROS 2 토픽을 낸다.
  - `--ros-no-dashcam`: 대시캠 없이 낸다.
- `run()` 에서 운전자 눈 카메라를 단 뒤, `--ros-output` 일 때만 `hero_ros_output.start(...)` 를 부른다.
- 운전 고리에서 `vehicle.apply_control(...)` 바로 뒤에 `ROS_OUTPUT.set_drive(핸들 축, 후진)` 를 부른다. 출력이 켜졌을 때만이다.
- `finally` 에서 차를 지우기 전에 `ROS_OUTPUT.close()` 를 부른다. 대시캠이 차에 붙어 있어서 먼저 지워야 한다.

기존 명령줄 인자와 기본값, 키보드·핸들 입력, 보정 JSON, 물리 값, 운전자 눈 카메라(위치·시야각·화면)는 하나도 바꾸지 않았다. 확인한 것:

- `--help` 를 원본과 비교했다(`help.diff`). 새 옵션 두 줄과 그 설명만 더 있다.
- ROS 도 carla 도 없는 파이썬에서 두 파일이 모두 import 된다(`import_without_ros.txt`). rclpy 는 불러오지도 않는다.
- carla 가 없을 때 원본과 사본이 같은 오류와 같은 종료 코드 2 로 끝난다(`run_without_carla.txt`).
- `--ros-output` 에 ROS 2 가 없으면 경고 한 줄만 내고 출력 없이 돈다(시험).

## 출력 (`hero_ros_output.py`)

| 토픽 | 형식 | 내용 |
|---|---|---|
| `/clock` | `rosgraph_msgs/Clock` | 시뮬레이터 시각, 월드 틱마다 |
| `/carla/ground_truth/pose` | `geometry_msgs/PoseWithCovarianceStamped` | frame `map`, 차 액터의 위치와 자세 |
| `/hero/vehicle_state` | `hero_hud/VehicleState` | 기어, 부호 있는 속도, 앞바퀴 조향각(왼쪽 +), 요레이트(왼쪽 +) |
| `/hero/steering_wheel` | `sensor_msgs/JointState` | `steering_wheel` = 운전자가 돌린 핸들 각도(rad, 왼쪽 +), `road_wheel` = 앞바퀴 조향각, velocity = 프레임 간 변화 |
| `/sensing/camera/dashcam/image_raw` | `sensor_msgs/Image` | bgr8 1920×1080, 30 Hz(시뮬레이터 시각) |
| `/sensing/camera/dashcam/camera_info` | `sensor_msgs/CameraInfo` | fx = fy = 960, (960, 540), plumb_bob 0, frame `dashcam_optical_link` |

- 핸들 각도 토픽은 내가 정했다: `/hero/steering_wheel` (`sensor_msgs/JointState`).
  - 머리말 시각이 있어 다른 토픽과 맞출 수 있다. 표준 형식이라 rqt_plot 과 녹화 분석에 바로 쓴다.
  - 값은 T300RS 축(−1..1) × 540°(도구의 `WHEEL_RANGE_DEG` 1080 의 절반)이고, 부호를 바꿔 왼쪽이 + 다.
  - 축 값에서 바로 구하므로, 차에 들어가는 명령의 데드존(0.01 축) 안의 핸들 움직임도 나온다.
- 바퀴 조향각: 명령값이 아니라 **실제 바퀴 각도**다.
  - `Vehicle.get_wheel_steer_angle(FL, FR)` 의 평균이다(CARLA 0.9.13 이상, 도 단위, 오른쪽 +). 부호를 바꿔 rad 로 낸다.
  - 이 함수가 없으면 적용된 명령 × 최대 조향각으로 대신한다.
- 기어: 도구의 후진 버튼 전환을 따른다(D/R).
  - `--wheel-drive` 가 아닐 때는 도구가 차를 붙잡아 두므로 P 다.
- 속도: 차 진행 방향 성분이라 후진은 음수다.
- 요레이트: CARLA 의 z 각속도(도/s, 오른쪽 회전이 +)의 부호를 바꿔 rad/s 로 낸다.
- QoS: 모두 reliable 로 낸다. 녹화는 영상·위치를 best effort 로 냈다. 차선 인식의 live 모드는 영상을 reliable 로 받으므로, reliable 로 내야 둘 다 받을 수 있다.

### 위치 약속 (Town05 녹화와 `reference_pose.py` 에서 알아낸 것)

녹화(`/media/jaehun/38F7-677B/rosbag/town05_30hz`)의 첫 메시지들:

- `/carla/ground_truth/pose`: frame `map`, 시뮬레이터 시각(2.55 s 부터, 1/60 s 간격).
  - 위치 (28.14, −18.66, 0.036): 높이가 땅 높이이므로 액터 원점이다.
  - 방향 yaw −89.4° 이고, 다음 프레임에 y 가 0.09 m 줄어든다. 즉 진행 방향과 yaw 가 오른손 좌표에서 맞는다.
  - 그래서 CARLA 좌표(왼손)를 y 와 yaw 부호를 바꿔 오른손으로 옮긴 것으로 본다(carla-ros-bridge 와 같은 변환).
- `reference_pose.py` 는 이 위치에서 진행 방향으로 1.4096 m 뒤를 뒤축(base_link)으로 쓴다.
- 확신할 수 없는 것:
  - 녹화한 차가 무엇이었는지(1.4096 m 가 Prius 에도 맞는지)는 녹화에 없다.
  - 그래서 도구가 시작할 때 실제 Prius 의 값(뒷바퀴 위치에서 계산)을 출력하게 했다. 다르면 `reference_pose` 의 `actor_to_rear_axle_m` 을 그 값으로 띄워야 한다(README 점검표 4).
  - 액터 원점이 Prius 에서도 땅 높이인지는 CARLA 에서 확인해야 한다.

### 대시캠 자리

- 녹화의 `camera_info`: 1920×1080, K = [960, 0, 960; 0, 960, 540; 0, 0, 1], 왜곡 0, frame `dashcam_optical_link`. bgr8, step 5760.
  - 같은 값을 계산으로 낸다: CARLA 핀홀 카메라에서 f = 960 / tan(45°). 녹화의 960.0000000000001 과 같다.
- 자리: maploop.launch.py 의 static TF(base_link → dashcam_optical_link = 앞 2.2 m, 위 1.4 m, 기울기 없음)와 같게 했다.
  - 뒤축은 물리 설정의 뒷바퀴(조향하지 않는 바퀴) 위치 평균이다(세계 좌표 cm → 차 좌표 m).
  - 땅은 경계 상자 바닥(location.z − extent.z)이다.
  - Prius 의 뒷바퀴가 원점에서 −1.33 m 쯤이면 카메라는 차 좌표 (0.87, 0, 1.4) m 다(시험의 가짜 값 기준).
- `sensor_tick` = 1/30 s. 다른 카메라 속성(노출, 모션 블러 등)은 CARLA 기본값이다. 녹화를 만든 도구가 이 컴퓨터에 없어서 그쪽 설정은 확인하지 못했다.

## 시험 (`tests.log`, 6개 통과)

- 좌표 변환: CARLA yaw 30° → ROS −30°. 그때 움직이는 방향과 yaw 가 맞고, 쿼터니언을 되돌리면 같은 yaw 다.
- 부호: 바퀴(오른쪽 + → 왼쪽 +), 핸들 축 0.5 → −270°, 끝을 넘으면 ±540° 에서 멈춤, 요레이트, 후진 속도 음수.
- 뒤축, 대시캠 자리, camera_info 가 녹화 값과 같은지. 시각 변환.
- BGRA → bgr8.
- **가짜 CARLA 로 발행 경로 전체**:
  - 가짜 월드, 차(바퀴 위치, 경계 상자, 바퀴 각도), 카메라, 스냅샷을 쓴다. 실제 rclpy 로 여섯 토픽을 받아 값을 확인했다.
  - 위치 (100, −50, 0), yaw −30°, 시각 12.5 s.
  - 핸들 135°(왼쪽), 바퀴 −11°.
  - 영상 1920×1080 bgr8, 시각 12.5333 s.
  - camera_info K.
  - vehicle_state: D, 5 m/s, −11°, 요레이트 +6°/s.
  - 닫으면 틱 콜백이 풀리고 대시캠이 지워진다.
- ROS 가 없을 때 경고만 내고 None.

## 확인하지 못한 것

이 컴퓨터에는 CARLA 서버, carla 파이썬 패키지, 핸들이 없다. 설치하지 않았다.

- 실제 CARLA 에서 도는지, 운전 화면이 대시캠 때문에 느려지는지.
- `get_physics_control().wheels[].position` 이 도구가 차를 붙잡아 둔 상태(물리 꺼짐)에서도 맞는 값인지(뒤축 계산에 씀). 시작할 때 출력해 확인하게 했다.
- Prius 의 액터 원점에서 뒤축까지의 거리, 원점이 땅 높이인지.
- 비동기 모드에서 영상과 위치의 시각 맞춤. 둘 다 시뮬레이터 시각으로 찍지만 서버 fps 에 따라 간격이 달라진다.
- carla 패키지와 rclpy 가 같은 파이썬(3.10)에서 함께 import 되는지.
- 대시캠의 다른 속성이 Town05 녹화를 만든 카메라와 같은지.

사용자가 CARLA 컴퓨터에서 할 일은 사본 README 의 "CARLA 컴퓨터에서 확인할 순서 (점검표)" 7단계다.

## 제안 (이번 범위 밖, 하지 않음)

- `reference_pose` 의 뒤축 거리를 launch 인자로 받게 하기. 지금은 노드 파라미터이고 기본값이 1.4096 이다.
- 안내 노드가 `/hero/vehicle_state` 가 있을 때 속도와 요레이트를 그것에서 쓰게 하기. 지금은 위치 변화에서 구한다.
- `run_lane_hud.sh` 가 녹화 이름과 캐시를 인자로 받게 하기. 지금은 세 이름만 알고, 새 녹화는 live 모드와 재생 도구를 따로 띄워야 한다.
- HUD 가 차선 전용 모드에서도 `/hero/vehicle_state` 로 차의 현재 진행 호를 그리게 하기.

## 파일

| 파일 | 내용 |
|---|---|
| `changes.diff` | 복사 커밋 이후 사본의 변경 전부 |
| `driver_eye.diff` | 원본 대비 `prius_wheel_based_driver_eye.py` (더한 줄만) |
| `help_original.txt`, `help_copy.txt`, `help.diff` | `--help` 비교 |
| `import_without_ros.txt`, `run_without_carla.txt` | ROS·carla 없을 때 |
| `tests.log` | 시험 결과 |
