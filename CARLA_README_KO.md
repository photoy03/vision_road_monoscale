# CARLA 전용 차선 도색 검출

학습/가중치 없이 RGB 영상의 국소 대비·도색 폭·방향·조각 연결을 사용한다. 출력은 실제 관측한 도색 조각과 명시된 짧은 과거 예측이며, 중심 경로/조향은 만들지 않는다. 점선의 빈 구간을 연속 실선으로 만들어 발행하지 않는다.

## 첨부 영상으로 ROS 실행

터미널1:
```bash
cd /home/jaehun/Desktop/vision-road
./run_carla.sh
```
터미널2:
```bash
cd /home/jaehun/Desktop/vision-road
./replay_carla.sh --rate 0.5
```

원본 1,207프레임의 불규칙한 캡처 시각을 보존하고 `/clock`과 RGB를 발행한다. 오도메트리는 생성하지 않는다. 기본0.5배속은 이번 노트북의 처리 여유를 위한 값이며, 원본 영상이 느려졌다는 뜻은 아니다. 빠른 입력에서는 최신 영상 위주 처리가 필요하고 입력 전체 처리와 실시간 처리는 별도로 평가해야 한다.

## 실제 CARLA ROS bridge 입력

예:
```bash
./run_carla.sh width:=1280 height:=720 horizontal_fov:=90 camera_height:=1.5 pitch:=0 camera_forward:=0 image_topic:=/carla/ego_vehicle/rgb_front/image odom_topic:=/carla/ego_vehicle/odometry
```

**위 수치는 첨부 영상 시험의 임시 가정이다.** 실제 시뮬레이터의 카메라 해상도/FOV·설치 높이·기울기·앞쪽 위치로 바꿔야 한다. `pitch`는 이 패키지의 아래쪽 양수 각도이며 CARLA Transform의 pitch 부호를 확인해서 변환해야 한다. 카메라높이는 평면 도로 기준이다. `camera_forward`는 차량 기준 앞쪽 위치다. 현재 launch의 roll/yaw/left offset은0이므로 다른 설치라면 생성된 rig 또는 launch를 확장해야 한다.

입력은 distortion 없는 pinhole 가정이다. CARLA에서 lens distortion/postprocess를 켜 두었다면 그 설정도 일치시켜야 한다. `camera_info`의 K와 실제 Transform/TF가 올바른 기준이며, 현재 MP4에서 이 값들을 역으로 정확하게 보정한 것은 아니다.

- [CARLA 공식 ROS sensor 토픽](https://carla.readthedocs.io/projects/ros-bridge/en/latest/ros_sensors/)
- [공식 bridge 카메라 구현](https://github.com/carla-simulator/ros-bridge/blob/master/carla_ros_bridge/src/carla_ros_bridge/camera.py)

이미지 구독은 reliable/depth10이며 publisher가best_effort라면 `carla.yaml`의 `image_reliability`를 맞춘다. role/sensor 이름이 다르면 topic 인자를 바꾼다. 출력 차량 좌표계는 CARLA 프로필에서 `ego_vehicle`, 기존 프로필에서는 `base_link`다. odom의header frame은map, child frame은출력 차량 좌표계와 같아야 한다. 다른 이름은 `world_frame`/`base_frame` 설정으로 맞춘다.

## 토픽

- `/parking/cpp/marking_segments`: 현재 도색 조각·짧은 예측의 선분. 같은left/right에 여러 조각이 있을 수 있다. `reprojected`, `source_stamp`, `confirmed`로 이력을 구분한다.
- `/road/lane_styles`: side별UNKNOWN/SOLID/DASHED. 여러 조각의 기하 추적과 종류 확인을 분리하며, 색·이중선 전용 종류는 아직 없다.
- `/parking/cpp/binary`, `/parking/cpp/vector_bev`, `/road/overlay`: 현재 후보/시각화.
- `/parking/cpp/status`: 적용 백엔드의 raw/retained/grid 픽셀 수와 후보·조각 수, 정합·누적 상태.
- 기존 전역 누적 토픽은 유지되지만 CARLA 프로필의 `map_enabled`는 기본false다. 임시 카메라로 거리 지도를 검증했다고 오인하지 않기 위한 설정이다. 실제 카메라와odom/차량frame을 맞춘 후에 `map_enabled:=true`로 누적을 켤 수 있다. 과거 예측은 새 지도 관측으로 쓰지 않는다.

## 구성과 한계

`src/vision_road/config/carla.yaml`에 CARLA 백엔드를 별도로 둔다. 과거 KITTI/실물 카메라 설정은 이 프로필에 섞지 않는다. 최종 검출기에는 학습 모델·외부 가중치가 없다.

절대밝기225 재검사를 제거하고, 여러 폭에서 주변 노면보다 밝은 응답을 이용한다. 연결 조각은 폭·방향·길이·가까운 지점의 위치·연결된 전체 구간으로 검사한다. 교차로 격자처럼 반복되는 가로 도색은 별도 영역에서 제외한다. 최종 선분은 보닛/유효영역으로 자르고, 확정되지 않은 종류라도 기하가 여러 프레임 일치하면 짧은 예측을 허용한다.

실제 도색이 없는 교차로에서 차선을 억지로 생성하지 않는다. 먼 거리의 작은 문양·마모 패치, 격자 영역 경계, 급한 회전에서 누락/오검출이 남을 수 있다. 첨부 영상 한 개로 개발·평가했으므로 새로운CARLA맵·날씨·카메라에서 성능을 보장하지 않는다. 독립 정답 데이터가 없어 전체 정확도를 측정한 것은 아니다.

현재 순수Python/OpenCV 측정에서는 검출만약58ms중앙값 수준이며, ROS 변환·시각화는 추가된다. 20~24Hz 입력의 실시간 완전 처리를 통과했다고 주장하지 않는다. 시험 수치와 원본을 함께 결과 보고서에서 확인한다.


## 2026-11-11 잡점·점선 개선

학습 없이 밝기 대비, 폭·방향·연결성으로 페인트 조각을 추출하고 좌우 각각 연결 후보를 선택한다. 각 연결 안에서도 선택한 연결 안에서 강건하게 맞춘 2차 경계 곡선에서 0.25 m 이상 벗어난 조각을 제거한다. 0.15초 이내의 짧은 후보 누락은 확인 이력만 유지하며, 일치하는 새 관측 3회가 쌓여야 외부로 발행하며 정합 성공 시 이전 경계를 25% 반영한다. 정합 실패 시 과거 좌표를 섞지 않고 현재 관측만 확인한다. 이 경우 차량 이동 때문에 생길 수 있는 위치 차이는 최대 0.60 m까지 확인에 허용하고, 정합 성공 시에는 0.30 m 기준을 쓴다. 예측은 정합 성공 및 마지막 관측 후 최대 0.15초로 제한하고 누적 지도에 넣지 않는다.

| 토픽 | 메시지 | 의미 |
|---|---|---|
| `/parking/cpp/marking_segments` | 기존 `vision_lane_interfaces/MarkingSegmentArray` | 실제 페인트 조각과 명시적 짧은 예측, 기존 필드 유지 |
| `/road/lane_styles` | 기존 `LaneStyleArray` | 좌우 실선·점선·미확정 |
| `/road/lane_markers` | `visualization_msgs/MarkerArray` | 실제 페인트 조각의 LINE_STRIP, 초록/예측 노랑 |
| `/road/lane_boundary_markers` | `visualization_msgs/MarkerArray` | 좌우 최대 두 경계 곡선, 청록색. 점선 간격을 가로지르는 표시용 추정이며 페인트 검출/중심 경로가 아님 |
| `/road/overlay` | `sensor_msgs/Image`, bgr8 | 원본 카메라 위 표시 |

MarkerArray는 매 프레임 DELETEALL 후 현재 선을 보내며 0.3초 수명이다. Fixed Frame을 `ego_vehicle`(기본 CARLA 설정)로 두고 RViz에 두 MarkerArray 또는 Image를 추가하면 된다. 다른 차량 프레임은 `base_frame`과 일치시킨다. capture timestamp와 차량 로컬 미터 좌표를 사용한다. 세부 파라미터: `carla_chain_filter_enabled`, `carla_chain_confirm_frames`(3), `carla_chain_band_m`(0.25). 필터를 끄면 이전의 미확정 조각 발행으로 돌아간다.

첨부 hero-coach는 SceneOverlay에 장애물과 계획 경로를 쓰며 차선 구독/렌더링은 포함하지 않는다. hero_hud 메시지 정의와 렌더러가 없어 해당 HUD에 직접 표시되는 것은 아직 검증하지 못했다. 위 표준 토픽은 준비되어 있으나 HUD에 구독 어댑터가 필요하다. coach의 `/hero/scene_overlay` 또는 ideal 계획 경로를 차선 데이터로 덮어쓰지 않는다.

또한 첨부 coach 실행 구성의 fisheye 합성 영상과 이번 CARLA MP4용 pinhole 투영은 다르다. 영상 입력을 바꾸면 해당 카메라의 K/D 및 고정 TF를 맞춰야 한다. 현재 MP4의 FOV90도/높이1.5m는 가정이며 실제 TF 검증을 대체하지 않는다.

참고한 원리는 [Graph Model-Based Lane-Marking Feature Extraction (2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8271487/)의 연결된 선 특징 선택과 [Spatial-Temporal Knowledge Filtering (2016)](https://www.mdpi.com/1424-8220/16/8/1276)의 시공간 제약이다. 논문의 전체 알고리즘을 재현한 것은 아니며, 현재 OpenCV 구현에 맞춘 결정론적 필터다.


RViz2가 설치된 환경에서는 다음 설정으로 표시할 수 있다(실제 RViz GUI 실행 자체는 이번 검증에 포함하지 않음).

```bash
source /opt/ros/humble/setup.bash
source /home/jaehun/Desktop/vision-road/install/setup.bash
rviz2 -d /home/jaehun/Desktop/vision-road/src/vision_road/config/carla_lanes.rviz
```

경계 확인은 새 관측 간격 최대 0.5초까지 허용하지만, 좌표 정합이 없으면 과거 선을 섞거나 예측하지 않는다. 예측 수명 제한 0.15초는 별도로 유지한다.
