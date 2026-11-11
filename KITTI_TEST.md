# KITTI Raw로 차선 검출 시험하기

실행 환경은 바탕화면 `~/Desktop/vision-road`입니다. 실제 KITTI `2011_09_26_drive_0001_sync` 데이터는 현재 `data/kitti`에 압축 해제되어 있습니다. 데이터는 소스 저장소에 포함하지 않습니다. 기존 어안 카메라용 `road.launch.py`는 그대로 사용할 수 있고, KITTI는 새 `kitti.launch.py`로 실행합니다. 중심 경로는 발행하지 않습니다.

## 1. 데이터 준비

공식 페이지 https://www.cvlibs.net/datasets/kitti/raw_data.php 에 로그인하고 **Raw → Road**, **synced+rectified** 시퀀스 하나와 **해당 촬영 날짜의 calibration**을 내려받습니다. 도로 차선이 잘 보이는 시퀀스로 먼저 확인합니다. 아래 명령은 `2011_09_26_drive_0027`이라는 유효한 시퀀스 이름을 예로 든 것이며 다른 시퀀스를 받았다면 날짜·번호를 바꾸면 됩니다. 이 시퀀스가 최적이라는 실영상 비교는 아직 하지 않았습니다.

필수 파일은 왼쪽 컬러 영상 `image_02`, 영상 timestamps, `oxts` 데이터/timestamps와 보정 파일 3개입니다. 라이다·오른쪽 영상은 이 시험에서 사용하지 않습니다. KITTI odometry benchmark의 `sequences/00` 형식이나 PNG를 MP4로 바꾼 파일은 이 입력 모드에 맞지 않습니다.

```bash
cd ~/Desktop/vision-road
mkdir -p data/kitti
unzip ~/Downloads/2011_09_26_calib.zip -d data/kitti
unzip ~/Downloads/2011_09_26_drive_0027_sync.zip -d data/kitti
```

압축 해제 후 구조:

```text
data/kitti/2011_09_26/
├── calib_cam_to_cam.txt
├── calib_imu_to_velo.txt
├── calib_velo_to_cam.txt
└── 2011_09_26_drive_0027_sync/
    ├── image_02/
    │   ├── timestamps.txt
    │   └── data/0000000000.png ...
    └── oxts/
        ├── timestamps.txt
        └── data/0000000000.txt ...
```

## 2. ROS 없이 먼저 확인

```bash
cd ~/Desktop/vision-road
/usr/bin/python3 tools/kitti_offline.py \
  --sequence data/kitti/2011_09_26/2011_09_26_drive_0027_sync \
  --output results/kitti_0027 --check-only
```

필수 파일·크기·타임스탬프·고정 변환을 확인하고, 실제 보정값에서 `camera.npz`, `rig.yaml`, `dataset.json`을 생성합니다. 누락된 파일이나 잘못된 해상도는 오류로 알립니다. 데이터는 읽기만 합니다.

첫 300프레임 검출 영상을 저장하려면:

```bash
/usr/bin/python3 tools/kitti_offline.py \
  --sequence data/kitti/2011_09_26/2011_09_26_drive_0027_sync \
  --output results/kitti_0027 --max-frames 300
```

`results/kitti_0027/overlay.mp4`: 원본에 검출 차선을 녹색으로 그린 영상.
`overlay_*.png`, `bev_*.png`, `binary_*.png`: 처음·중간·마지막 프레임 검사 자료.
`frames.csv`, `summary.json`: 프레임별 검출 수와 pose 정합 수. **검출 프레임 비율은 정확도 수치가 아닙니다.** 차선 정답 라벨을 읽어 평가하는 기능은 없습니다. 오프라인 누적 상태는 실행 중 계산하며 누적 지도 파일의 저장 기능은 현재 제공하지 않습니다.

## 3. ROS 토픽으로 시험 — 첫 단계 권장

```bash
cd ~/Desktop/vision-road
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch vision_road kitti.launch.py \
  sequence:=$PWD/data/kitti/2011_09_26/2011_09_26_drive_0027_sync \
  odom_source:=oxts rate:=0.5 max_frames:=300
```

`rate:=0.5`는 절반 속도 재생이고, 데이터의 원래 시간 간격은 유지합니다. `max_frames:=0`이면 전체 시퀀스를 재생합니다. 시작 후 3초 대기해 구독 연결 시간을 확보합니다. 종료 후 마지막 화면/지도 상태를 유지하고 Ctrl+C로 종료합니다. 다시 시험할 때에는 전체 launch를 종료하고 새로 실행합니다.

기본 모드는 **OXTS GPS/IMU 기준 pose**를 이용합니다. 차선 검출기와 보정값이 적절한지 먼저 보기 위해 MonoScale 오차를 섞지 않는 구성입니다. 이것은 MonoScale 오도메트리의 정확도를 검증한 결과가 아닙니다.

별도 터미널에서도 위 두 setup 파일을 불러온 뒤:

```bash
ros2 run rqt_image_view rqt_image_view
```

화면에서 `/road/overlay`를 선택하면 원본 위 차선이 보이고, `/parking/cpp/vector_bev`를 선택하면 BEV 위 차선이 보입니다. 이 영상 토픽은 best-effort QoS이므로 뷰어 QoS를 best effort로 맞춥니다. 뷰어가 없다면 ROS Humble의 `rqt_image_view` 패키지를 설치합니다.

```bash
ros2 topic echo /parking/cpp/marking_segments --once
ros2 topic echo /parking/cpp/status
ros2 run tf2_ros tf2_echo base_link kitti_camera_rect02
```

기존 이름 그대로 로컬 차선 `/parking/cpp/marking_segments`, 누적 차선 `/parking/cpp/global_marking_segments_fused`, 누적 격자 `/road/lane_grid`가 나옵니다. RViz의 fixed frame은 `map`, 차선 격자는 Map 디스플레이에서 `/road/lane_grid`를 선택합니다. 격자의 양수 값은 차선 관측 증거이며 장애물을 뜻하지 않습니다.

## 4. MonoScale 오도메트리로 교체 — 두 번째 단계

첫 launch를 Ctrl+C로 종료한 뒤:

```bash
ros2 launch vision_road kitti.launch.py \
  sequence:=$PWD/data/kitti/2011_09_26/2011_09_26_drive_0027_sync \
  odom_source:=monoscale rate:=0.5 max_frames:=300
```

동일한 KITTI pinhole K와 고정 장착 변환을 tracker/odometry에 전달합니다. `road_from_step`과 `road_step_esm`은 true이고, KITTI IMU를 입력합니다. MonoScale raw → stabilizer → `/localization/kinematic_state`가 누적 pose가 됩니다. 비교용 OXTS는 `/kitti/reference_odom`에 따로 발행하고 지도에 직접 넣지 않습니다. TF의 map→base_link 발행 권한도 stabilizer만 갖습니다.

KITTI는 10Hz이고 기존 실제 카메라는 30Hz입니다. 기존 알고리즘의 주행 속도별 튜닝과 pose 오차는 검증 전입니다. 이 테스트 모드는 stabilizer의 속도/프레임 이동 제한만 40m/s와 5m로 완화했습니다. 프레임 간 이동이 크면 광합 정합이 실패할 수 있으므로 OXTS 모드부터 확인합니다. 현재 도구는 두 궤적의 수치 오차를 자동 계산하지 않습니다.

## 좌표·보정 처리

- 선택 카메라는 `image_02`의 **rectified pinhole**입니다. `P_rect_02`의 왼쪽 3×3을 K로 쓰고 D=0으로 설정합니다. 기존 어안 투영을 적용하지 않습니다.
- `T_cameraRect02_imu = T_cameraRect02_cameraRect00 × R_rect_00 × T_camera00_velo × T_velo_imu`로 구성합니다. 첫 항은 `P_rect_02`에서 계산해 컬러 카메라의 기준점 이동을 반영합니다.
- `base_link`는 IMU 바로 아래 지면의 점이고 축은 IMU와 같은 x 전방, y 좌측, z 위쪽입니다. IMU 높이는 공식 KITTI 센서 도면의 **0.93m**가 기본이며 `imu_height_m:=...`로 바꿀 수 있습니다. 센서 간 변환과 이 높이를 조합해 카메라 장착을 구성하므로 카메라 높이를 임의로 1.65m로 덮어쓰지 않습니다.
- `/tf_static`에 `base_link → kitti_camera_rect02`, `base_link → imu_link`를 발행합니다. TF와 검출기 투영은 같은 행렬을 사용합니다.
- OXTS 위경도를 로컬 Mercator 좌표로 변환하고 IMU→지면 기준점의 lever arm을 반영합니다. 시작 위치·yaw를 원점으로 삼되 지구 수직 방향은 유지합니다. 누적기는 x/y/yaw를 쓰므로 3D 경사로에서 정밀한 지면 추적을 수행하지 않습니다.
- 영상/OXTS 각각의 원본 timestamp를 읽어 정수 ns로 보존합니다. ROS에는 공통 시작점에서 1초부터의 상대 시각과 `/clock`을 발행합니다. 데이터 자체의 경과 시간은 재생 속도와 무관합니다.
- 보정 파일은 날짜별이고, 다른 날짜의 파일을 섞으면 안 됩니다. 생성된 값은 기본 `/tmp/vision-road-kitti/<sequence>/dataset.json`에서 확인합니다. `output_dir:=/absolute/path`로 바꿀 수 있습니다.

## 검증 범위

자동 검증은 **KITTI 형식으로 만든 합성 데이터**로 다음을 확인합니다: rectification과 baseline을 포함한 고정 TF, 일반 카메라 투영, 소수 ns 시각, 실제 ROS 재생→차선 토픽/overlay→OXTS 누적, MonoScale 카메라 파라미터 연결. 실제 KITTI 0001 영상 시험은 진행했고, 오검출과 누락 및 MonoScale pose 미발행을 확인했습니다. 자세한 결과는 `results/KITTI_0001_REPORT_KO.md`에 기록했습니다.

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ROS_DOMAIN_ID=174 ROS_LOCALHOST_ONLY=1 \
  /usr/bin/python3 -m pytest -q tests/test_core.py tests/test_kitti.py tests/test_ros.py tests/test_kitti_ros.py
```

출처:
- 데이터/다운로드: https://www.cvlibs.net/datasets/kitti/raw_data.php
- 센서 설정: https://www.cvlibs.net/datasets/kitti/setup.php
- 센서 높이·좌표·형식 원문 논문 Fig.3: https://www.cs.utoronto.ca/~urtasun/publications/geiger_et_al_ijrr13.pdf
- rectified camera 변환 참고 구현: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/raw.py

KITTI 재생은 reliable 영상 전달을 사용합니다. 원래 획득 시각은 유지하고, IMU가 영상 시각 뒤 샘플까지 전달된 다음 해당 영상을 전달합니다. `/clock`은 전달 순서대로 진행합니다. 실시간 카메라 입력은 기본 best-effort 설정을 유지합니다.


## 그림자 개선 버전 확인

`bash replay_kitti_0001.sh`는 그림자 대응과 최근 프레임 연결을 기본으로 켭니다. `/road/overlay`에서 녹색은 현재 영상 관측으로 지지되는 곡선, 노란색은 최근 관측에서 현재 좌표로 옮긴 예측입니다. 예측은 최대 3프레임/0.35초/3m 이내이며 정합 실패 시 사라집니다. 예측 선분은 `reprojected=true`로 발행하고 지도에는 추가하지 않습니다.

```bash
# 현재 기본 결과 저장
/usr/bin/python3 tools/kitti_offline.py \
  --sequence data/kitti/2011_09_26/2011_09_26_drive_0001_sync \
  --output results/my_shadow_test --max-frames 108
# 프레임 연결을 끄고 같은 전처리와 비교
/usr/bin/python3 tools/kitti_offline.py \
  --sequence data/kitti/2011_09_26/2011_09_26_drive_0001_sync \
  --output results/my_shadow_only --max-frames 108 --no-temporal
```

기존 `results/kitti_0001_baseline` 영상은 개선 전 상태로 보존했습니다. 전후 비교 영상은 `results/kitti_0001_shadow_comparison/comparison.mp4`입니다. 정확도 평가용 정답 라벨은 사용하지 않았으므로 관측 프레임 수와 인식 정확도를 구분해야 합니다.
