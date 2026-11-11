## CARLA 전용 실행 — 2026-11-11

현재 주행 카메라는 CARLA 전용 `carla` 백엔드와 `carla.launch.py`를 사용합니다. 학습 없이 도색 조각을 검출하며, 중심 경로는 만들지 않습니다.

```bash
cd /home/jaehun/Desktop/vision-road
./run_carla.sh
# 다른 터미널에서 첨부 영상 재생
./replay_carla.sh --rate 0.5
```

카메라 값은 실제 CARLA 설정으로 맞춰야 합니다. 첨부 영상 시험의 FOV90°/높이1.5m/기울기0°는 임시 가정입니다. [CARLA 실행 안내](CARLA_README_KO.md)를 확인하세요. 아래 과거 기록은 KITTI/실물 카메라 개발 이력입니다.

---

# vision-road

2026-11-11 업데이트: **학습 모델 없이 OpenCV/기하·시간 정합만 사용하는 `paint_evidence` 검출기**가 기본입니다. 실제 주행 5개 영상의 전체 처리와 실패 사례는 [5개 영상 검증 보고서](results/CLASSICAL_FIVE_VIDEO_REPORT_KO.md)를 확인하세요. 실행 성공과 인식 정확도 통과를 구분합니다. 모델 가중치·PyTorch·학습 출력 캐시는 실행에 필요하지 않습니다.

기존 vision-lane-monoscale의 카메라와 MonoScale을 사용하면서 **도로의 좌·우 곡선 차선**을 새 알고리즘으로 검출하는 독립 ROS 2 Humble 워크스페이스입니다. 기존 주차 검출기·주차 보조·직선 병합 지도는 실행하지 않습니다. 차량 조향 명령은 발행하지 않습니다.

## 실행

바탕화면에 빌드된 상태로 설치되어 있습니다. 새 터미널에서:

```bash
cd ~/Desktop/vision-road
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch vision_road road.launch.py
```

기본은 외부에서 전방 카메라와 IMU 토픽을 받아, MonoScale tracker → 광합 정합 odometry → pose stabilizer → 차선 누적을 실행합니다. 기존 주차 노드는 함께 실행하지 마세요. 같은 이름의 출력 토픽이 겹칩니다.

카메라와 F9R 드라이버도 이 프로젝트에서 실행하려면:

```bash
ros2 launch vision_road road.launch.py launch_camera:=true device:=/dev/video2 \
  launch_imu:=true imu_device:=/dev/ttyACM0
```

실제 장치 경로에 맞게 변경합니다. 카메라는 기존 MJPG 2560×1440, 30 fps 설정입니다. 외부 드라이버를 이미 실행했다면 해당 launch 옵션은 false로 둡니다. F9R 초기 바이어스 보정 동안 약 2초 정지합니다. IMU 입력은 base_link 축(x 전방, y 좌측, z 위쪽) 기준이며 F9R에서 보정한 orientation을 사용합니다.

이미 만들어지는 오도메트리를 사용할 때:

```bash
ros2 launch vision_road road.launch.py launch_monoscale:=false \
  odom_topic:=/localization/kinematic_state
```

오도메트리는 `nav_msgs/Odometry`, `map → base_link`여야 합니다. 영상과 오도메트리 stamp는 같은 시계를 사용해야 합니다. 센서 raw 영상과 IMU만 담긴 bag은 아래처럼 실행합니다. bag에 기존 차선/오도메트리가 포함되어 있으면 필요한 입력 토픽만 재생하거나 `launch_monoscale:=false`를 선택해 중복 발행을 피합니다.

```bash
ros2 launch vision_road road.launch.py use_sim_time:=true
# 별도 터미널: 자신의 bag 경로 사용
ros2 bag play /path/to/bag --clock --topics \
  /sensing/camera/front/fisheye/image_raw /sensing/imu/imu_data
```

압축 영상만 받을 때에는 검출 노드를 직접 실행할 수 있습니다. 이 모드에서는 MonoScale을 별도로 실행하고 `/localization/kinematic_state`를 제공해야 합니다.

```bash
ros2 run vision_road road_node --ros-args \
  --params-file src/vision_road/config/road.yaml \
  -p compressed:=true -p image_topic:=/measurement/front/image_raw/compressed
```

## 입출력

| 토픽 | 타입 | 의미 |
|---|---|---|
| `/sensing/camera/front/fisheye/image_raw` | sensor_msgs/Image | 전방 raw 영상 입력 |
| `/sensing/imu/imu_data` | sensor_msgs/Imu | 기존 MonoScale IMU 입력 |
| `/localization/monoscale_raw` | nav_msgs/Odometry | MonoScale 원시 추정 |
| `/localization/kinematic_state` | nav_msgs/Odometry | 누적에 사용하는 보정 pose |
| `/parking/cpp/marking_segments` | vision_lane_interfaces/MarkingSegmentArray | 해당 영상에서 관측한 곡선의 짧은 선분, base_link |
| `/parking/cpp/global_marking_segments_fused` | 동일 | map 좌표계 누적 차선 |
| `/parking/cpp/global_marking_map` | sensor_msgs/Image | 누적 차선 래스터 보기 |
| `/parking/cpp/binary` | sensor_msgs/Image | 검출 이진 영상 |
| `/parking/cpp/vector_bev` | sensor_msgs/Image | 차선을 그린 BEV |
| `/parking/cpp/status` | std_msgs/String | 처리 수, pose 대기/드롭, 지도 셀 수 JSON |
| `/road/lane_grid` | nav_msgs/OccupancyGrid | 원점·해상도 포함 누적 차선 격자 |

현재 관측이 없을 때에는 신뢰할 수 있는 움직임 추정과 최근 확정 차선이 있는 경우에 한해 최대 3프레임/0.35초/3m 이내에서 예측을 발행합니다. 조건을 만족하지 않거나 만료되면 빈 배열을 발행합니다. 한쪽만 검출하면 그 차선만 발행합니다. 중심 경로는 생성하거나 발행하지 않습니다. 과거 지도는 TTL까지 남으므로 지도와 현재 검출은 구분해 사용합니다. `lane_grid`는 **차선 표시 지도**이며 장애물/주행 가능 영역 지도가 아닙니다. -1은 미관측, 양수는 관측 증거입니다.

기존 주 토픽 이름과 메시지 타입을 유지했습니다. 새 `family`는 로컬 `left`/`right`, 전역 `road`입니다. `confirmed`는 로컬 정합 관측 2회, 전역 셀 관측 2회부터 true입니다. 과거에서 보완한 선분은 `reprojected=true`이며 `source_stamp`가 마지막 실제 관측 시각입니다. 이 선분은 누적 지도의 새 증거로 사용하지 않습니다. 로컬 `pose_valid=false`여도 영상의 실제 차선은 발행되며 지도는 pose가 정합될 때까지 기다립니다. 로컬 출력은 그 시점에 한 번만 발행됩니다. `track_id`는 해당 메시지 안의 선분 식별자이며 영구 차선 ID가 아닙니다. 폭 추정은 구현하지 않아 `mean_width_m`은 0(미추정)입니다.

이 프로젝트는 누적 지도를 자체 발행합니다. 기존 parking용 lane_map에 다시 넣어 직선 병합하지 않습니다. 기존 `/parking/cpp/lane_observations`, `LaneMapSnapshot`, occupancy와의 `FusedMap` 동기화 계약은 구현하지 않았습니다. 그 소비 노드와 연결하려면 별도 어댑터가 필요합니다.

## 검출과 누적 방식

1. 기존 `front.npz`의 K/D, `rig.yaml`의 장착값을 두 경로가 공유합니다. 어안 지면 투영으로 미터 단위 BEV를 만들기 때문에 예제의 3.7m 차선폭/720px 환산을 거리 스케일로 사용하지 않습니다.
2. CLAHE 국소 대비 보정, 주변 밝기 대비, 좁고 밝은 띠 응답과 HLS 색상 지지로 그림자 속 차선 후보를 찾습니다. BEV 하단 히스토그램 → 슬라이딩 윈도우 → 이상치 제거 → 2차 다항식으로 곡선을 찾습니다. 이전 다항식 주변 탐색에 실패하면 전체 창 탐색으로 복귀합니다.
3. 새 `paint_evidence` 방식은 색/명암의 도색 근거로 후보를 평가하고 각 차선을 독립적으로 선택합니다. 긴 비도색 경계 바깥의 후보를 제한하고, 현재 도색으로 곡선을 보정합니다. 충분한 현재 도색이 있는 곡선은 점선 끝 공백을 최대 3m까지 제한적으로 연장할 수 있습니다. 이는 현재 관측에 대한 곡선 적합이며 거리 전 구간에서 도색이 직접 보였다는 뜻은 아닙니다. 과거 차선만으로 보완하는 경우는 별도로 `reprojected`와 원래 관측 시각을 유지합니다.
4. MonoScale의 `road_from_step=true`, `road_step_esm=true`를 유지하고 전방 단일 카메라로 실행합니다. 기존 bounded 운용의 IMU/velocity 정책을 적용합니다. 이 모듈은 포함된 기존 구현을 재사용합니다.
5. 영상 시각과 정확히 일치하거나 최대 0.25초 간격의 두 pose 사이에서 보간한 오도메트리로만 지도 좌표로 변환합니다. 외삽하지 않습니다. 최대 0.5초·30프레임 대기 후 pose 미확보 프레임은 지도에서 제외합니다. 전역 지도는 0.1m 셀, 반경 40m, 마지막 관측 후 120초, 최대 100000셀로 제한합니다. 시간 역행 시 누적 상태를 초기화합니다.

설정은 `src/vision_road/config/road.yaml`입니다. 다른 파일을 적용할 때:

```bash
ros2 launch vision_road road.launch.py params_file:=/absolute/path/road.yaml
```

카메라 변경은 `calibration:=/absolute/path/front.npz`, 장착 변경은 `rig_file:=/absolute/path/rig.yaml`로 두 모듈에 함께 적용합니다. 기존 장착값은 높이 0.80m, 전방 0.08m, pitch 30°입니다. 원본에도 임시 지지대 측정값이라는 주석이 있으므로 현재 실제 장착과의 일치는 확인이 필요합니다. 촬영 종횡비가 보정값과 다르면 중단합니다.

## 빌드·검증

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --parallel-workers 2 \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
source install/setup.bash
/usr/bin/python3 -m pytest -q tests/test_core.py tests/test_ros.py
```

기본 의존성은 ROS 2 Humble, colcon, cv_bridge, OpenCV, Eigen, numpy, PyYAML, pytest, gscam입니다. 다른 PC에서는 `rosdep install --from-paths src --ignore-src -r -y`로 패키지 의존성을 설치합니다.

합성 직선/곡선/점선, 한쪽 소실, 빈 영상, 폭 이상치, 어안 투영, yaw 보간, 누적/만료/셀 제한과 실제 ROS 메시지 송수신을 검증했습니다. `artifacts/synthetic_camera.png`, `synthetic_bev.png`는 합성 시험 자료입니다. 실제 장착 카메라의 도로 녹화·센서는 아직 제공되지 않았습니다. KITTI 한 시퀀스 시험과 별개로 실차 정확도와 실시간 처리율, MonoScale 주행 오차는 검증하지 않았습니다. 우선 저속 도로 영상으로 색상·시야·동기화를 조정해야 합니다. 기존 pose stabilizer의 기본 최고 속도 제한은 5m/s이며 고속용 보정값은 별도 실측이 필요합니다. 평면 지면 가정이므로 경사 변화·요철·카메라 흔들림에서 거리 오차가 발생할 수 있습니다.

## 출처

- 기존 코드: https://github.com/photoy03/vision-lane-monoscale , 로컬 clean checkout `c88e7962d67579fbe5de6b53ae1bef6920955a60`의 MonoScale/core/tracker/pose/interfaces/F9R 및 카메라 보정 파일 재사용.
- 검출 방식 참고: https://velog.io/@choonsik_mom/Lane-Detection%EC%B0%A8%EC%84%A0-%EC%9D%B8%EC%8B%9D . 글의 처리 흐름을 참고해 새 구현을 작성했고, 예제 코드 전체를 복제하지 않았습니다.
- 새 도로 코드와 ROS 모듈은 Apache-2.0, 포함된 upstream MonoScale의 별도 고지는 `third_party/MONOSCALE_LICENSE`, `MONOSCALE_README.md`에 보존했습니다.

## KITTI 시험 모드

일반 카메라 투영·KITTI Raw 재생·고정 TF·OXTS/MonoScale pose 선택 모드를 추가했습니다. 데이터 준비와 실행은 [KITTI_TEST.md](KITTI_TEST.md)를 따라 진행합니다. 원본 영상에 차선을 그린 `/road/overlay`도 발행합니다. KITTI 0001 실영상의 첫 검증 결과는 `results/KITTI_0001_REPORT_KO.md`에 기록했습니다. 최초 버전에서 연석 오검출과 누락, MonoScale pose 미발행을 확인했습니다. 그림자 개선 후 결과는 `results/SHADOW_TEMPORAL_REPORT_KO.md`를 참조하세요.


## 그림자·여러 프레임 연결

최근 관측을 오도메트리로 현재 좌표에 옮겨 탐색과 곡선 평활화에 사용합니다. pose가 없을 때는 BEV 특징점의 양방향 광류 및 RANSAC 정합이 충분한 경우에만 영상 움직임으로 대신합니다. 입력 간격이 크거나 정합이 불량하면 과거 결과를 버립니다. 만료는 시간·프레임·이동거리 중 먼저 도달하는 제한을 적용합니다.

- `/road/overlay`: 녹색 = 현재 관측 지지 곡선, 노란색 = 과거 관측에서 보완한 예측.
- `/parking/cpp/marking_segments`: 예측은 `reprojected=true`, `source_stamp`는 마지막 실제 관측 시각. 현재 시각은 배열의 header.stamp입니다.
- 누적 지도: 현재 관측만 새 증거로 반영하며 예측을 반복 누적하지 않습니다.
- 설정: `shadow_enabled`, `temporal_enabled`, `temporal_max_frames`(3), `temporal_max_age_sec`(0.35), `temporal_max_distance_m`(3.0), `temporal_use_flow`.
- 비교 실행: `tools/kitti_offline.py`의 `--no-temporal`, `--no-shadow`, `--no-odom`. `--no-shadow`는 새 후보 선택기에 기존 밝기 전처리를 적용하는 비교이며, 최초 버전 전체를 복원하는 옵션은 아닙니다.

실제 KITTI 0001에서 최초 버전 대비 관측 후보 출력 프레임이 59/108에서 108/108로 늘었습니다. 이는 정확도 100%를 뜻하지 않으며, 연석 오검출이 남아 있습니다. 자세한 결과는 `results/SHADOW_TEMPORAL_REPORT_KO.md`에 기록했습니다. MonoScale의 유효 pose 미발행 문제는 이번 변경으로 해결한 항목이 아닙니다.


## 점선·실선 표시

`/road/overlay`에 `L:DASHED`(왼쪽 점선), `R:SOLID`(오른쪽 실선), `UNKNOWN`(미확정)을 표시합니다. 곡선을 연속으로 그렸다는 사실로 실선이라고 판단하지 않고, 원래 이진 영상에서 해당 곡선을 따라 페인트/공백이 반복되는지를 검사합니다. 실제로 보이는 8m 이상 구간에서 세 개 이상 페인트 구간과 두 개 이상 공백의 반복·규칙성이 확인되면 점선 후보입니다. 한 번의 가림이나 짧은 관측만 있으면 미확정으로 둡니다. 기본적으로 움직임 정합된 두 프레임의 판단이 일치해야 분류를 확정합니다.

기존 `/parking/cpp/marking_segments` 메시지와 `family=left/right`는 유지합니다. 분류는 **`/road/lane_styles` (`vision_lane_interfaces/msg/LaneStyleArray`)**에 별도로 발행합니다. 같은 header.stamp/camera_id와 side로 차선 선분 배열에 연결합니다.

- `style`: `UNKNOWN=0`, `SOLID=1`, `DASHED=2`.
- `confidence`: 패턴 분류 confidence이며 차선 검출 confidence와 다릅니다. 미확정이면 0입니다.
- `observed_span_m`, `paint_fraction`, `visible_dash_count`: 현재 카메라에서 보이는 구간의 패턴 근거.
- `reprojected`, `source_stamp`: 과거 차선을 보완한 경우의 구분과 마지막 실제 관측 시각. 새 분류 관측으로 간주하지 않습니다.

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 topic echo /road/lane_styles
```

현재 분류는 **검출된 후보의 패턴**을 분류합니다. 연석을 차선으로 잘못 검출하면 그 경계가 SOLID로 나올 수도 있으며, 이 분류가 실제 차선 페인트임을 보증하지 않습니다. 그림자·차량 가림으로 패턴이 불분명하면 UNKNOWN이 나옵니다. 상세 시험은 `results/LANE_STYLE_REPORT_KO.md` 참조.

## 다른 주행 구간 시험

새로 내려받은 KITTI 0015(297프레임)와 0028(430프레임)을 검출기/설정 변경 없이 전체 처리했습니다. 도심 노면 이음매 오검출과 숲길 분류 불안정을 확인했습니다. 후보가 매 프레임 나온 것을 정확도 100%로 해석하면 안 됩니다. 결과 영상·한계는 `results/NEW_ROUTES_REPORT_KO.md`에 정리했습니다. `bash replay_new_kitti.sh 0015` 또는 `0028`로 다시 재생할 수 있습니다.

## 순수 처리의 시간 누적과 분류 시각

`style_temporal_enabled=true`이면 움직임으로 정합된 최대 8프레임/0.8초의 실제 도색 이진 관측 중 2회 이상 지지를 패턴 판단에 사용합니다. 과거 곡선을 이진 관측으로 그려 넣지 않습니다. 과노출로 주변 노면까지 포화한 곳은 점선의 공백 근거로 쓰지 않습니다. 같은 차선이 현재에도 검출되지만 패턴이 불명확하면 확정된 종류를 최대 2초 유지하며 confidence를 낮춥니다. 이 경우 화면은 `[HIST]`, `style_source_stamp`는 마지막 종류 근거 시각입니다. `source_stamp`는 차선 곡선의 마지막 실제 관측 시각이며 둘은 다를 수 있습니다. 지도에는 과거 예측을 새 관측으로 반영하지 않습니다.

다섯 구간 재생: `bash replay_new_kitti.sh 0056 oxts 0.5` (0001/0015/0028/0018/0056 선택). 오프라인 전체 비교는 `tools/five_video_classical.py`, 평가와 라벨 한계는 `results/classical_five_videos/`에 보관합니다. KITTI 평가의 pose는 OXTS이며 MonoScale 유효 odometry 미발행 문제를 해결했다는 뜻이 아닙니다.

새 기본 검출기는 도색 패턴이 아직 UNKNOWN인 후보를 과거 예측으로 연장하지 않습니다. 현재 영상에서 실제 후보가 검출되면 종류가 UNKNOWN이어도 발행합니다. 이 제한은 불확실한 연석 후보의 이력 연장을 줄이기 위한 것입니다.
