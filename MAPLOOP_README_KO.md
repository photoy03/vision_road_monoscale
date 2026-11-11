> **2026-11-12 최신 버전:** 추적 유지 가점이 추가됐습니다. 전체 출력 1,807개를 약 30Hz로 수신했지만 전달 지연 p95=150.1ms로 기존 100ms 목표는 미달입니다. 아래의 이전 통과 수치는 당시 버전 기록입니다. [현재 검증 보고서](docs/tracking_bonus_20261112.md)를 우선 확인하세요.

# maploop_30hz 로스백 검증 및 재생

30Hz 처리량 검증을 완료한 기본 설정입니다. 자세한 실측과 한계는 [30Hz 안내](REALTIME_README_KO.md)를 확인하세요. 차선·종류·두 MarkerArray는30Hz, 진단 영상은최대5Hz입니다. 입력은원본1920×1080, 진단 오버레이는960×540입니다.

외장 SSD 원본: `/media/jaehun/38F7-677B/rosbag/maploop_30hz/`.

카메라 `/sensing/camera/dashcam/image_raw`의 **원본 1920×1080 bgr8**를 사용한다. 내부 파라미터는 녹화된 CameraInfo의 fx=fy=960, cx=960, cy=540, 왜곡 0이다. 함께 제공된 센서킷 기준 후륜축에서 전방 2.20 m, 노면에서 높이 1.40 m, 피치 0°이며 fisheye와 구분한다. `maploop_rig.yaml` 및 `maploop_dashcam.npz`에 기록했다. 보닛을 피하도록 영상 높이의 91.5% 아래를 제외한다.

참 위치는 액터 원점이므로 후륜축까지 1.4096 m를 뺀다. `reference_pose`는 이 변환만 하는 **평가용 참 위치 어댑터**이며 MonoScale 오도메트리가 아니다. 평면 차선 좌표의 base_link는 후륜축 아래 노면 z=0으로 둔다. 미리 기록된 원본 참 위치를 재생하고 영상 시각에서 보간한다. 이번 로스백에는 추정 오돔 토픽이 없다.

```bash
# 터미널 1: 차선 노드 + 평가용 참 위치 어댑터 + 고정 카메라 TF
/home/jaehun/Desktop/vision-road/run_maploop.sh

# 터미널 2: SSD 원본을 영상 시뮬레이션 시각으로 재생
/home/jaehun/Desktop/vision-road/replay_maploop.sh --rate 1

# 짧은 구간만 확인할 경우
/home/jaehun/Desktop/vision-road/replay_maploop.sh --start-frame 300 --max-frames 120 --rate 1

# 터미널 3: RViz2가 설치되어 있을 경우
source /opt/ros/humble/setup.bash
source /home/jaehun/Desktop/vision-road/install/setup.bash
rviz2 -d /home/jaehun/Desktop/vision-road/src/vision_road/config/maploop_lanes.rviz
```

실제 RViz GUI 자체는 이번 검사 대상이 아니다. MarkerArray 메시지와 영상 직렬화·구독은 실제 ROS 노드에서 확인한다. 그래픽 화면 없이도 결과 비교 영상을 볼 수 있다.

기록 벽시계 길이는 약 18분이지만 카메라 헤더의 시뮬레이션 길이는 60.2초다. 제공 재생기는 `/clock`을 카메라 헤더에 맞춰 발행한다. 다른 bag 재생기와 동시에 실행하지 않는다. 노드는 기본적으로 reliable 영상을 구독하고 제공 재생기도 reliable로 발행한다. 원본 best-effort 퍼블리셔에 직접 연결해야 하면 `run_maploop.sh image_reliability:=best_effort`로 맞춘다. 기본1배속이 실제30Hz 재생이다. 모든 프레임의33.3ms 이내 완료를 보장하는 하드 실시간과는 구분한다. 원본 메시지의 손실 없는 캐시는 추가SSD에 준비되어 있으며 자동 사용한다. 캐시가 없으면 SSD 원본을 직접 읽지만 저장장치 지연에 따라 입력률이 떨어질 수 있다.

참 위치를 쓰지 않는 영상만의 테스트는 `run_maploop.sh reference_pose:=false`로 실행한다. 이때 움직임 정합은 광류를 사용한다. 참 위치 사용 상태와 성능을 섞어서 비교하지 않는다.

## 화면과 토픽

- `/parking/cpp/marking_segments`: 기존 `vision_lane_interfaces/MarkingSegmentArray` 유지. 실제 페인트 조각, 예측 여부·관측 시각 포함.
- `/road/lane_styles`: 기존 좌우 실선·점선·미확정 메시지 유지.
- `/road/lane_markers`: 실제 조각의 `visualization_msgs/MarkerArray`.
- `/road/lane_boundary_markers`: 좌우 최대 두 개의 경계 추정 곡선. 점선의 빈 구간을 가로지를 수 있는 **표시용 경계**이며 실제 도색 검출이나 중심 경로가 아니다.
- `/road/overlay`: 원본 크기 bgr8 영상. 카메라 위 실제 조각과 경계를 표시.
- 프레임은 `base_link`, 길이는 m, x 전방/y 좌측이다. Marker는 매 프레임 DELETEALL로 교체하고 0.3초 후 만료한다. 예측 페인트는 누적 지도를 갱신하지 않는다. maploop 프로필의 누적 지도는 기본으로 꺼져 있다.

`/home/jaehun/Desktop/hero-coach-main/`에는 실제 HUD 렌더러가 없다. `launch/coach.launch.py`는 외부 `hero_hud/hud_node`를 실행할 뿐이고, coach_node는 격자·계획을 SceneOverlay로 바꾸는 상류 노드다. 현재 coach에는 차선 구독·렌더링이 없으므로 `/hero/scene_overlay`를 임의로 덮어쓰지 않는다. 실제 hero_hud 패키지가 제공되어야 전용 HUD 차선 표시를 직접 연결할 수 있다.

## 수정 원리

이전에는 붙어 있는 조각들을 하나의 큰 집합으로 합쳐, 노면 패치나 원거리 잡점이 경계 곡선을 끌어당겼다. 이제 폭·방향·색·점선 간격이 맞는 조각들 사이에 방향성 그래프를 만들고 가까운 차선에서 이어지는 경로를 선택한다. 이중 황색선은 선택된 경계 주변의 같은 색 평행 조각을 함께 보존한다. 실제 점선은 길이만으로 삭제하지 않고 1.2 m 이상 가까운 단독 조각도 여러 관측에서 확인할 수 있다. 표본의 마지막 위치를 실제 관측 끝에 정확히 맞춰 선 전체가 잘못 잘리는 문제도 수정했다.

`carla_path_filter_enabled`를 false로 두면 이전 집합 방식과 비교할 수 있다. `carla_chain_filter_enabled`까지 false로 두면 경계 확인 없이 원래 페인트 조각을 내보낸다. 학습, 모델 가중치, CARLA 지도·차선 참 라벨은 검출에 사용하지 않는다. CARLA 참 위치는 좌표 정합에만 쓴다.

관련 원리: [그래프 기반 차선 특징 선택, 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8271487/), [공간·시간 제약으로 선분 필터링, 2016](https://www.mdpi.com/1424-8220/16/8/1276). 논문의 전체 재현이 아니라 현재 OpenCV 구현에 맞춘 결정론적 경로 선택이다.

원본 bag을 수정하거나 내장 SSD에 복사하지 않는다. 전체 검증은 원본 해상도에서 만든 손실 없는 BEV 캐시를 같은 노드 내부 검출기에 공급하며, 실제 ROS 통합은 별도로 원본 CDR 영상으로 검사한다. 비교 영상의 배경은 확인용 축소 JPEG지만 검출 입력 BEV는 JPEG에서 만들지 않았다. 같은 영상에서 개발한 결과이므로 독립 평가나 모든 구간 통과를 의미하지 않는다.
