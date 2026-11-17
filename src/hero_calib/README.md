# CARLA Prius Driver-Eye Calibration

CARLA 0.9.16의 `vehicle.toyota.prius`에서 steering-wheel rotation centre를
기준으로 운전자 eye point와 virtual monitor geometry를 계산하고 확인하는
Python 도구입니다.

## Calibrated geometry

CARLA vehicle-local axes are `+X forward`, `+Y right`, `+Z up`.

| Point / parameter | Value |
|---|---:|
| Steering-wheel centre | `(0.815, -0.415, 0.920) m` |
| Wheel → eye offset | `(-0.500, 0.000, +0.210) m` |
| Driver eye | `(0.315, -0.415, 1.130) m` |
| Eye → monitor distance | `0.710 m` |
| Virtual monitor centre | `(1.025, -0.415, 1.130) m` |
| Display width / height | `0.6984 / 0.39285 m` |
| Calculated horizontal FOV | `52.378778°` |

Eye → monitor comes from the seated rig, measured at 1.055 m height:
seatback → monitor 0.910 m minus seatback → eye 0.200 m. The monitor stays
0.210 m ahead of the wheel centre, so eye → wheel is 0.500 m.

The FOV is calculated at runtime:

```text
HFOV = 2 atan((screen_width / 2) / viewing_distance)
```

The calibration belongs to the measured physical rig. It is not a universal
Toyota Prius or human anthropometry value.

## Requirements

- Ubuntu 22.04 or a compatible Linux environment
- CARLA 0.9.16 server running on `127.0.0.1:2000`
- CARLA 0.9.16 Python API available to the selected Python interpreter
- Python packages in `requirements.txt`

Install the ordinary Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Install the CARLA Python API using the package distributed with the same CARLA
0.9.16 installation. Do not install a mismatched CARLA client version.

## Run

Clean FHD eye-camera preview:

```bash
python3 prius_wheel_based_driver_eye.py \
  --preview-eye-camera \
  --clean-preview \
  --lateral-offset 0.0
```

Fullscreen preview on SDL display 0:

```bash
./run_fullscreen.sh
```

For another monitor index:

```bash
DISPLAY_INDEX=1 ./run_fullscreen.sh
```

Drive the Prius with a Thrustmaster T300RS wheel (physics is re-enabled; button
4 toggles reverse, button 5 is the handbrake):

```bash
./run_fullscreen.sh --wheel-drive
```

Wheel drive also swaps CARLA's stock Prius physics for 3rd-generation (ZVW30)
values: 37.8 deg lock reached at the wheel's stop (14.27:1), 1380 kg, Cd 0.25, 0-100 km/h in
10.4 s, hybrid creep. Sources and measured response: `prius_gen3_physics.md`.

Geometry validation view:

```bash
python3 prius_wheel_based_driver_eye.py \
  --preview-eye-camera \
  --debug-geometry
```

The camera starts at the virtual-monitor centre. Press `M` to compare the
monitor-centre and driver-eye camera positions while retaining the same camera
rotation and FOV. Press `Esc` to exit.

## Publish this prepared repository privately

On the original Ubuntu desktop, run:

```bash
./publish_private_github.sh
```

The helper installs Ubuntu's `gh` package when necessary, opens GitHub's web
authentication flow, uses or creates `UnKaiFoo/HERO_Calib` as a private
repository, pushes `main`, and opens the resulting repository page.

## Controls

| Input | Action |
|---|---|
| `M` | Toggle driver eye / monitor centre camera position |
| `PageUp`, `PageDown` | Adjust camera pitch |
| `W/S`, `A/D`, `R/F` | Adjust local wheel cursor X/Y/Z |
| `1`, `2`, `3` | Select 10/5/1 mm calibration step |
| `Enter` or `P` | Confirm and save calibration |
| `Esc` | Exit |

## Files

- `prius_wheel_based_driver_eye.py`: main application
- `prius_steering_wheel_center.json`: confirmed rig calibration
- `prius_steering_wheel_center_method.md`: method and API findings
- `prius_steering_wheel_api_audit.json`: recorded actor API audit
- `prius_gen3_physics.md`: 3rd-generation Prius physics for `--wheel-drive`, sources and measurements
- `run_fullscreen.sh`: FHD fullscreen preview helper
- `publish_private_github.sh`: private GitHub install/auth/create/push helper
- `docs/`: seated-pose research and walker bone mapping
- `legacy/`: earlier mannequin-based tools and measurements retained for traceability

The main application resolves its JSON files relative to the script directory,
so the repository can be cloned under any user account. Some files in
`legacy/` preserve historical absolute output paths and may need explicit CLI
path arguments on another computer.

## CARLA API limitation

Standard CARLA does not document a direct cockpit steering-wheel-centre
property. The tool checks for semantic component, bone, and socket inspection
methods when available, then falls back to the confirmed manual calibration.
Debug primitives cannot guarantee a non-emissive material, so clean preview is
the default and draws no debug geometry.

## ROS 2 outputs (added in vision-road, `--ros-output`)

This copy lives in `vision-road/src/hero_calib`. The original is `~/Desktop/HERO_Calib-main`, which was not changed.
`prius_wheel_based_driver_eye.py` only gained added lines:
- one optional import,
- two new options,
- three calls into `hero_ros_output.py`.

**이 사본에서 `publish_private_github.sh` 를 실행하지 말 것.** 이 사본은 vision-road 저장소 안에
있어서, 실행하면 그 저장소의 원격(origin)을 바꾸고 push 한다. 원본 폴더 전용이다.

Every existing option, default, keyboard and wheel input, calibration, physics value and eye-camera
setting is the same. Without `--ros-output` nothing new runs. With it but without ROS 2, the tool prints
one warning and runs as before.

### 출력

한 장으로 정리한 출력 약속(토픽, 형식, 단위, 부호, 기준점, 주기, 미확정 항목): `OUTPUT_CONTRACT_KO.md`.

| 토픽 | 형식 | 내용 |
|---|---|---|
| `/clock` | `rosgraph_msgs/Clock` | 시뮬레이터 시각, 월드 틱마다 |
| `/carla/ground_truth/pose` | `geometry_msgs/PoseWithCovarianceStamped` | `map`, 차 액터 위치·자세. CARLA 왼손 좌표를 오른손으로(y, yaw 부호 바꿈). Town05 녹화와 같은 약속 |
| `/hero/vehicle_state` | `hero_hud/VehicleState` | 기어(D/R: 도구의 후진 전환, 차를 붙잡아 둔 동안 P), 속도(m/s, 후진 음수), 앞바퀴 조향각(rad, 왼쪽 +, CARLA 의 실제 바퀴 각도), 요레이트(rad/s, 왼쪽 +) |
| `/hero/steering_wheel` | `sensor_msgs/JointState` | `steering_wheel`: 운전자가 돌린 핸들 각도(rad, 왼쪽 +, T300RS 축 × 540°), `road_wheel`: 앞바퀴 조향각(rad, 왼쪽 +). velocity 는 프레임 간 변화 |
| `/sensing/camera/dashcam/image_raw` | `sensor_msgs/Image` | bgr8 1920×1080, 30 Hz(시뮬레이터 시각) |
| `/sensing/camera/dashcam/camera_info` | `sensor_msgs/CameraInfo` | fx = fy = 960(수평 90°), 중심 (960, 540), 왜곡 없음, frame `dashcam_optical_link` |

- 대시캠은 차선 인식이 가정하는 자리에 단다: 뒤축(base_link)에서 앞으로 2.2 m, 땅에서 1.4 m, 정면.
  - vision_road `maploop.launch.py` 의 static TF 와 같다.
  - 뒤축은 뒷바퀴 위치의 가운데, 땅은 차 경계 상자의 바닥으로 잡는다. 시작할 때 둘을 출력한다.
- 차선 인식의 `reference_pose` 는 액터 위치에서 1.4096 m 뒤를 뒤축으로 본다(Town05 녹화의 차).
  - 이 도구는 시작할 때 "actor_to_rear_axle_m should be ..." 를 출력한다. 값이 다르면 그 값으로 바꿔 띄워야 한다(아래 점검표 4).

### 띄우기

```bash
# CARLA 0.9.16 서버가 떠 있고, Town05 가 열려 있어야 한다.
source /opt/ros/humble/setup.bash
source /home/jaehun/Desktop/vision-road/install/setup.bash   # hero_hud 메시지(/hero/vehicle_state)
cd /home/jaehun/Desktop/vision-road/src/hero_calib
./run_fullscreen.sh --wheel-drive --ros-output
# 화면이 느리면 대시캠을 빼고 위치·차 상태만: --ros-output --ros-no-dashcam
```

### 녹화

```bash
source /opt/ros/humble/setup.bash
ros2 bag record -s sqlite3 -o town05_drive_$(date +%Y%m%d_%H%M) \
  /clock /carla/ground_truth/pose /hero/vehicle_state /hero/steering_wheel \
  /sensing/camera/dashcam/image_raw /sensing/camera/dashcam/camera_info
```

녹화할 장면(Town05, 같은 방향 차로가 둘 이상인 구간에서):

| 장면 | 내용 | 길이 |
|---|---|---|
| 차로 안 비틀거림 | 선을 넘지 않고 좌우로 0.3–0.6 m | 20초 이상 |
| 노란 선 넘었다 돌아오기 | 중앙선을 차 반쯤 넘었다가 돌아옴. 두세 번 | 20초 이상 |
| 점선에 걸쳤다 돌아오기 | 오른쪽 점선을 밟았다가 원래 차로로 | 20초 이상 |
| 오른쪽 차로로 옮기기 | 확실하게 옮김. 다시 왼쪽으로 돌아오기도 | 각 두세 번 |
| 그냥 따라가기 | 비교용 | 20초 |

### 녹화를 차선 인식과 HUD 로 돌리기 (vision-road 컴퓨터)

1. 녹화 폴더를 옮긴다(예: `/media/jaehun/38F7-677B/rosbag/town05_drive1/`).
2. 영상 캐시를 만든다(무손실, 녹화 원본은 읽기만 한다):
   `python3 tools/cache_maploop_cdr.py --bag <녹화>/<이름>_0.db3 --output /media/jaehun/38F7-677B/vision-road-cache/town05_drive1_native.cdrz`
3. 터미널 하나에서 `./run_lane_hud.sh live` 를 띄운다(검출기, 표시 필터, 안내, HUD).
4. 다른 터미널에서 재생한다:
   `python3 tools/replay_maploop.py --bag <녹화> --cache <위 캐시> --rate 1`
   - `run_lane_hud.sh` 는 지금 maploop, town04, town05 이름만 알고 있다. 이름으로 부르려면 거기에 녹화 줄을 하나 더해야 한다.
   - 재생 도구는 영상, camera_info, 위치, /clock 만 다시 낸다. `/hero/vehicle_state` 와 `/hero/steering_wheel` 은 분석용으로 녹화에 남는다.

### 실제 실행에서 생길 수 있는 문제

- 화면이 느려질 수 있다. 대시캠(1920×1080, 30 Hz)을 하나 더 그리고 6 MB 영상을 매 프레임 ROS 로 보낸다. 운전 화면이 느리면 `--ros-no-dashcam` 으로 비교해 본다.
- carla 파이썬 패키지와 rclpy(ROS 2 Humble, Python 3.10)가 같은 파이썬에서 import 되어야 한다. carla 패키지가 다른 파이썬용이면 `--ros-output` 은 rclpy 를 찾지 못했다는 경고만 내고 출력 없이 돈다.
- 이 도구는 월드 설정을 바꾸지 않으므로 보통 비동기 모드로 돈다.
  - 위치와 /clock 은 월드 틱마다, 영상은 그 틱의 시각으로 찍힌다.
  - 서버가 30 fps 보다 느리면 영상도 그만큼 줄어든다.
  - 영상과 위치의 시각 차이는 아래 점검표 5 로 확인한다.
- 앞바퀴의 실제 각도는 `Vehicle.get_wheel_steer_angle` (CARLA 0.9.13 이상)로 읽는다. 없으면 명령값 × 최대 조향각으로 대신한다.
- 핸들 각도는 축 값에서 계산하므로 0.01 축(약 5°)의 데드존 안도 그대로 나온다. 차에 들어가는 명령은 원래 도구처럼 데드존이 적용된다.

### CARLA 컴퓨터에서 확인할 순서 (점검표)

1. `python3 prius_wheel_based_driver_eye.py --help` 에 새 옵션 둘(`--ros-output`, `--ros-no-dashcam`)만 더 있는지 본다.
2. `--ros-output` 없이 `./run_fullscreen.sh --wheel-drive` 가 전과 같이 도는지 본다(화면, 핸들, 후진 버튼, 손 브레이크).
3. `--ros-output` 을 붙여 띄우고 `ros2 topic hz` 로 확인한다. 대시캠은 30 Hz 쯤, 위치는 서버 fps 쯤이어야 한다. 다음 여섯 토픽이 나와야 한다:
   `/clock`, `/carla/ground_truth/pose`, `/hero/vehicle_state`, `/hero/steering_wheel`, `/sensing/camera/dashcam/image_raw`, `/sensing/camera/dashcam/camera_info`
4. 시작할 때 출력되는 값을 확인한다:
   - "rear axle at x=..." 는 Prius 라면 약 −1.3 m.
   - "dashcam at vehicle-local (...)" 는 앞으로 약 0.9 m, 높이 약 1.4 m.
   - "actor_to_rear_axle_m should be ..." 가 1.4096 과 많이 다르면, vision_road 의 `reference_pose` 를 그 값으로 띄워야 위치가 맞는다.
5. 직진하면서 `ros2 topic echo /hero/vehicle_state` 를 본다:
   - 앞으로 갈 때 speed_mps 가 +, 후진 버튼 뒤에는 − 이다.
   - 핸들을 왼쪽으로 돌리면 steering_angle_rad 와 `/hero/steering_wheel` 의 steering_wheel 이 + 이다.
   - 왼쪽으로 돌 때 yaw_rate_rps 가 + 이다.
6. 짧게 녹화해서 vision-road 에서 위 "녹화를 돌리기"로 돌린다:
   - HUD 에 차선이 나오는지 본다.
   - 영상의 차선과 위치가 맞는지 본다(`./view_lane_coordinates.sh` 등).
   - 영상 시각과 위치 시각의 차이가 1/60초 안인지 본다.
7. 그다음 위 표의 장면들을 녹화한다.
