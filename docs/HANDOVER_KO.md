# 인수인계 — vision-road 차선·안내·신호 프로젝트

이 파일만 읽고 바로 이어서 작업할 수 있게 쓴다. **작업 하나가 끝날 때마다 "지금 상태"와 "다음에 할 일"을 고치고, `docs/WORKLOG_KO.md` 에 한 줄을 더한다.**

마지막 갱신: 2026-11-20 02:00 (계획 30 끝, 개선 세션)

---

## 1. 한눈에

- **무엇을 만드나**: CARLA Town05 대시캠 영상에서 차선을 찾고, 차로 중앙선과 복귀·차선 변경 안내 경로를 만들고, 신호등·속도 표지판·정지선·횡단보도와 함께 HUD 화면에 그린다.
- **시연 지도**: Town05. 모든 시험은 Town05 로 한다.
- **작업 폴더**: `/home/jaehun/Desktop/vision-road` (git, 브랜치 `main`). 원격 `origin` 보다 로컬이 앞서 있고 **push 는 하지 않는다**.
- **컴퓨터**: 이 노트북은 Ryzen 7 5800H, 16 GB, RTX 3050 4 GB, Ubuntu 22.04, ROS 2 Humble. CARLA·Autoware·핸들이 없다. 본체 컴퓨터(RTX 5070, 32 GB, CPU 모름)에 CARLA 가 있고 거기서 로스백을 녹화해 외장 SSD 로 가져온다.
- **지금 하는 일**: 계획 30. 새 Town05 로스백 12개와 정답 자료로 전체를 재고, 목표를 채울 때까지 "재기 → 가장 나쁜 것 고치기 → 다시 재기"를 반복한다. → 6절.

## 2. 사용자가 정한 규칙 (반드시 지킬 것)

| 규칙 | 내용 |
|---|---|
| 지도 | Town05 만. **Town04 는 시험·구현에서 건드리지 않는다.** |
| 영상 | 웬만하면 자르지 않는다. 전체 영상 방식이 지연 판정의 기준. `--crop-rows` 는 꺼진 선택 사항. |
| AI 필터 | YOLOPv2 차선 필터(`--learned`)는 Town05 시연에서 끈다. |
| 원본 폴더 | `~/Desktop/hero-hud-main`, `hero-coach-main`, `HERO_Calib-main`, `town05_yolo26m_detector` 는 읽기만. 작업은 `vision-road/src` 안의 사본에서. |
| 금지 스크립트 | `src/hero_calib/publish_private_github.sh` 실행 금지 (저장소 원격을 바꾸고 push 한다). |
| git | 로컬 커밋만. push 금지. |
| 시스템 | 시스템 설정 변경, 패키지 설치 금지 (ultralytics 버전도 그대로). |
| 새 설정 | 코드 기본값은 꺼짐. `yaml` 이나 launch 에서 켠다. 장면마다 다른 값 금지. |
| BEV | 좌우 ±6 m 유지. |
| 선택지 | 설계·설정·순서 같은 선택은 **묻지 말고 판단해서 진행**하고 무엇을 왜 골랐는지 알린다. 위 금지 사항은 예외(물어본다). |
| 지연 목표 | 차선 지연 상위 5% (p95) 25 ms 이하, 30 Hz. |
| 문서 | `docs/lane_improvement_research.md` 는 비전문가용 한국어. 수치·한계·미검증을 그대로 적는다. |
| 개인 정보 | 사용자 이메일 등을 외부 서비스에 보내지 않는다. |
| 이 파일 | 작업 하나 끝날 때마다 이 파일과 `WORKLOG_KO.md` 갱신. |

## 3. 폴더와 파일

```
vision-road/
  run_lane_hud.sh            한 줄 실행 (아래 4절)
  docs/
    HANDOVER_KO.md           이 파일
    WORKLOG_KO.md            작업 기록 (한 일마다 한 줄)
    lane_improvement_research.md   전체 연구 기록과 계획 1–30 (맨 끝 "개선 세션에 보낸 계획" 목록이 색인)
    latency_measure_KO.md    본체 컴퓨터에서 지연 재는 법
  src/
    vision_road/             차선 검출 (파이썬 + C++ `native.cpp`), 설정 `config/carla.yaml`, launch `maploop.launch.py`
      vision_road/carla.py           검출기 (CarlaDetector, TemporalDetector)
      vision_road/display_filter.py  표시선 안정화
      vision_road/rt_worker.py       계산 프로세스
      vision_road/road_markings.py   정지선·횡단보도 (규칙 방식)
      vision_road/node.py            ROS 노드
    vision_lane_interfaces/  차선 메시지
    hero_hud/                HUD 사본 (C++/Qt). 설정 `config/hud.yaml`
    hero_coach/              코치 사본 + 우리가 더한 것:
      lane_guidance.py, lane_guidance_node.py   복귀·차선 변경 안내
      traffic_signs.py, traffic_sign_bridge.py  YOLO 결과 투표 → /hero/traffic_light, /hero/sign
      traffic_advice.py                         신호 + 정지선 + 속도 → 정지 권고
      lane_ribbon.py                            코치 노드가 차로 안내 경로를 싣는 함수
    yolo_signs/              신호등·표지판 YOLO 검출 노드 사본 (가중치 `best.pt` 는 git 에 없음 → `~/Desktop/town05_yolo26m_detector/best.pt`)
    hero_calib/              HERO_Calib 사본 (CARLA 운전 도구에 ROS 출력 추가). 출력 약속 `OUTPUT_CONTRACT_KO.md`
    monoscale_*/             이번 작업과 무관
  tools/                     측정·재생 스크립트 (`realtime_eval.py`, `replay_maploop.py`, `fake_vehicle_state.py`, `measure_latency.sh` …)
  tests/                     vision_road 시험
  results/                   측정 결과 (git 에 없음, 이 노트북에만 있음). 계획마다 `REPORT_KO.md`
  data/, backups/            git 에 없음
```

외장 SSD: `/media/jaehun/38F7-677B`
- `rosbag/town05_30hz` (기준 영상, 1808프레임), `rosbag/town05_signs` (표지판 직진 42초), `rosbag/maploop_30hz` (같은 지도, 보조), `rosbag/town04_30hz` (**쓰지 않음**)
- `rosbag/town05_s*_20261119` 12개 (5절)
- `vision-road-cache/` 재생용 캐시 (만들어도 됨). 로스백 원본은 읽기만.

이어받는 작업은 **이 노트북에서 그대로** 한다(사용자 2026-11-19). 그래서 아래 파일들도 그 자리에 있고 push 는 필요 없다.

git 에 없는 것: `results/`, `data/`, `build/`, `install/`, `best.pt`, `data/models/yolopv2/yolopv2.pt`.

## 4. 빌드, 시험, 실행

```bash
cd ~/Desktop/vision-road
source /opt/ros/humble/setup.bash
colcon build --base-paths src --symlink-install --parallel-workers 2
source install/setup.bash
```

- C++ 빠른 경로(`_native*.so`)는 빌드가 만든다. 없으면 조용히 파이썬으로 돌아 느려진다.
- 시험 (2026-11-20 계획 30 3회차 기준 통과 수): `hero_coach` 102, `vision_road` 157, `hero_hud` 33(계획 29 기준, HUD 는 그 뒤 안 바뀜). 실제로 쓰는 명령(ROS 와 `install/setup.bash` 를 source 한 뒤; source 하지 않으면 hero_coach 시험이 모으는 단계에서 오류가 난다):

```bash
python3 -m pytest -q tests                      # vision_road
python3 -m pytest -q src/hero_coach/test        # hero_coach
colcon test --base-paths src --packages-select hero_hud && colcon test-result --test-result-base build/hero_hud
```

- 실시간 노드(road_node, HUD, lane_guidance 등)가 떠 있으면 ROS 시험이 서로 끼어든다. 그럴 때는 `ROS_DOMAIN_ID=77` 처럼 도메인을 나눠 시험한다.
- 커밋본(스냅숏)으로 시험할 때는 C++ 부분을 다시 빌드한다: `cd src/vision_road && python3 setup.py build_ext --inplace`.
- `colcon build` 는 반드시 `--base-paths src` 를 붙인다(`results/` 아래의 옛 사본과 이름이 겹친다).
- 이 노트북에는 `autoware_planning_msgs` 가 없어 `coach_node`, `parking_planner` 는 뜨지 않는다.

실행:

```bash
./run_lane_hud.sh town05                    # 기준 영상 한 번 재생 (차선 + 안내 + HUD)
./run_lane_hud.sh town05 --signs            # 신호등·표지판 YOLO 도 함께 (GPU, ~/yolo_env 파이썬)
./run_lane_hud.sh live --signs              # 재생 없이, 밖에서 영상·위치 토픽이 들어올 때
```

주요 인자: `--no-hud`, `--loop`, `--frames N`, `--learned`(끔 유지), `--no-guidance`, `--fake-vehicle-state`(조향·속도 토픽이 없는 영상용), `--crop-rows`(쓰지 않음), `--signs`. 환경 변수 `YOLO_PYTHON`, `SIGNS_FULL_FRAME`(기본 true), `SIGNS_FPS`(기본 0.0 = 제한 없음), `SIGNS_IMGSZ`(기본 1920, 계획 30).

YOLO 환경: `/home/jaehun/yolo_env` (ultralytics 8.4.48, torch 2.11, numpy 2.2). 이 버전은 `quantize` 인자를 몰라서 사본이 `half=True` 로 바꿔 돈다. 시스템 python3 로는 모델이 안 뜬다.

지연 재기: `tools/realtime_eval.py` (영상 발행 → `/road/lane_display_markers` 수신, p95). 세 번 재서 본다. **NoMachine 원격 화면이 켜져 있으면 수치가 2배로 나온다.**
- `run_lane_hud.sh` 와 같은 구성으로 재려면 `--launch-arg display_filter_inline:=true lane_guidance:=true fake_vehicle_state:=true` 를 꼭 붙인다(빠뜨리면 화면 필터가 따로 돌아 몇 ms 늘고 안내가 꺼진다).
- 신호·표지판까지: `--signs --signs-arg full_frame:=true imgsz:=1920`. 계획 30 의 새 녹화는 `--video town05_s3a_zigzag_cross` 처럼 부른다.
- 세 영상 × 3 번 한꺼번에: `results/town05_30hz/gt_loop_20261119/scripts/realtime_d.sh 출력폴더 1920 0.0`.

## 5. 자료 흐름과 토픽

```
대시캠 영상 ─→ vision_road (검출 → 표시 필터) ─→ /road/lane_display_markers, /road/lane_styles
                                              ─→ /road/lane_outer(+_markers)      바깥 차선 (좌우 한 줄씩)
                                              ─→ /road/lane_centres(+_markers)    차로 중앙선 (JSON: offset_m, heading_rad, width, opposite_direction, crossable …)
                                              ─→ /road/road_markings              정지선·횡단보도
대시캠 영상 ─→ yolo_signs ─→ /yolo/detections ─→ traffic_sign_bridge ─→ /hero/traffic_light, /hero/sign
/hero/vehicle_state (속도·조향) ─┐
위 토픽들 ──────────────────────┴→ lane_guidance ─→ /hero/scene_overlay (HUD 가 그림)
                                                  ─→ /hero/lane_guidance (JSON: 목표 차로, 금지, stop_line_m, traffic 권고)
                                                  ─→ /hero/lane_guidance/ribbon
```

- 입력: `/sensing/camera/dashcam/image_raw` (1920×1080 bgr8, fx = fy = 960, 30 Hz), `/carla/ground_truth/pose` (map, 차 원점), `/clock`.
- 좌표: 차선 검출은 뒤축 기준. 대시캠은 뒤축 앞 2.2 m, 높이 1.4 m. 코드의 "차 원점 → 뒤축" 가정은 1.4096 m, 실제는 1.3992 m (1 cm 차이, 계획 30 에서 영향 확인 예정).
- 안내 규칙 요약: 차로 중앙에서 0.30 m 넘게 벗어나면 복귀 경로 표시, 0.15 m 안이면 끔. 흰 점선 쪽으로 0.25 m/s 이상 0.6 초 계속 움직이고 선에 0.5 m 안으로 붙으면 옆 차로를 목표로. 노랑·실선·반대 방향 차로는 목표로 삼지 않고 금지 표시.
- 신호 규칙: 최근 3번 중 2번 같은 결과면 확정. 신호등 0.5 초, 표지판 3 초 뒤 꺼짐. 여러 개면 확신도 × 크기 × 가운데 가까움.
- 정지 권고: 필요 감속 2.0 이하 `prepare_stop`, 3.5 이하 `stop`, 초과 `stop_hard`, 넘으면 `over_line`, 정지 중 `hold`, 노랑인데 멈출 수 없으면 `proceed`.
- 본체에서 코치 노드와 함께 쓸 때: `lane_guidance` 의 `publish_scene:=false`, `coach_node` 의 `lane_ribbon_topic:=/hero/lane_guidance/ribbon`. **실행 미검증.**

## 6. 지금 상태

### 끝나서 검증된 것 (Town05 기준, 이 노트북)

- 차선 지연 p95 23.5–23.9 ms (전체 영상, AI 필터 끔; 계획 26). 신호·정지선을 켠 상태로는 아직 안 쟀다.
- 바깥 차선, 차로 중앙선, 표시선 끝 튐 수정 (계획 18–20).
- 차로 안내: 꾸민 장면 4개 통과 (계획 21). 실제 지그재그·차선 변경 영상으로는 아직 안 쟀다.

### 구현은 끝났지만 실제 영상으로 검증 안 된 것

- 신호등·표지판 HUD 결합 (계획 27, 커밋 99aaef9, ddf6414)
- 정지선·횡단보도·초록 신호 HUD 표시, `stop_line_m` (계획 28, f111f20)
- 코치에 맞춘 신호 전달·경로 전달·정지 권고 (계획 29, e64c42e)

### 진행 중: 계획 30 (2026-11-19 시작)

- 전문: `docs/lane_improvement_research.md` 의 "본체에서 온 새 로스백 12개와 정답 기반 반복 계획" 절 (목표표 포함).
- 결과 폴더: `results/town05_30hz/gt_loop_20261119/`
  - `SUMMARY_KO.md`: 회차 비교. `scripts/summary_rounds.py round_00: round_01:_1920 round_02:_1920` 로 만든다.
  - `round_NN/REPORT_KO.md`, `round_NN/table.md`: 회차별 보고서와 목표 대비 표. 표는 `scripts/summary.py round_NN --tag _1920` 으로 만든다.
- 진행 정도 (2026-11-20 01:50):
  - 기준선, 1·2·3회차가 끝났다(커밋 8aa7de4, 985f79f).
  - 마지막 HUD 영상(s3a, s5, s6, s7)도 만들었다: `results/town05_30hz/gt_loop_20261119/final_hud/<장면>/road_and_hud.mp4`(커밋 안 함).
  - 계획 30 은 여기서 끝났다. 남은 미달 항목은 7절에 있다.
  - 3회차:
    - 정지선 정답 정의를 고쳐 C4 를 충족했다.
    - 정지 위치 대체(`stop_line.crosswalk_fallback`)를 넣었다.
    - D1 공유 메모리를 넣었고, `run_lane_hud.sh --signs` 기본으로 켰다.
    - B5·B3·D1 은 닫았다.

#### 도구 (`tools/gt_eval.py`, 사용법은 파일 맨 위)

- `predict`: 차선·정지선·안내를 오프라인으로 돌린다. 검출 설정은 `src/vision_road/config/carla.yaml` 이고 `--set`·`--guide` 로 덮어쓴다.
- `reguide`: 저장된 중앙선으로 안내만 다시 돌린다(장면당 몇 초).
- `yolo`: `~/yolo_env/bin/python` 으로 돌린다. `--full-frame`·`--imgsz`·`--tag`.
- `paint_stops`: semantic 영상에서 칠한 선 정답을 만든다.
- `metrics`: `--tag` 로 쓸 yolo 파일, `--sign` 으로 SignSettings 를 고른다. 출력은 `metrics<tag><name>.json` 이다.
- 스크립트 (`results/town05_30hz/gt_loop_20261119/scripts/`):
  - `eval_round.sh 회차폴더 [--no-yolo] [장면…]`: 기본 설정은 launch 와 같은 값이다. `YOLO_FROM=회차` 로 yolo 결과를 재사용한다.
  - `recompute.sh 회차폴더 태그 [sign 설정…]`: 평가 정의를 바꾼 뒤 지표와 표를 다시 만든다.
  - `reguide_round.sh`, `paint_all.sh 장면…`, `summary.py`, `summary_rounds.py`, `realtime_d.sh 출력폴더 [imgsz] [max_fps]`, `build_caches.sh`, `yolo_tile_all.sh`, `yolo_1920_all.sh`.
- 캐시: SSD `vision-road-cache/town05_<장면>_20261119_native.cdrz` 와 `_offline/` (12 개 모두 있음).
- 칠한 선 정답: `paint_stops/<장면>/paint_stops.jsonl.gz` 를 모든 회차가 같이 쓴다. 들어 있는 것:
  - 칠한 정지선
  - 정답 경계선과 칠한 선의 차이: 거리 5·6·10·15 m 별(`gt_boundary_to_paint_at`)
  - 옆 차로 바깥 선이 BEV 안에 보이는 길이
- `tools/realtime_eval.py --video` 에 새 장면을 넣었다: `town05_s3a_zigzag_cross`, `town05_s7_lights` 등 `town05_<장면>` 이름.

#### 평가 정의 (자세히는 `round_00/REPORT_KO.md` 표)

- 정답 오류: 정답 경계가 칠한 선에서 0.15 m 넘게 어긋난 지점은 A1–A4 에서 뺀다. 모든 장면 출발 구간과 s6 오른쪽이다.
- 선 위: 차 중심이 선에서 0.3 m 안이면 A1–A5 에서 뺀다.
- 반대 방향 차로: 정답이 뒤집어 적은 좌우·방향을 되돌린다(`facing`).
- B5: 칠한 선 기준 오프셋으로 센다.
- B1: 정답 차로 변경과 맞지 않는 목표 이동만 센다.
- C2: 교차로 밖, 내 신호등이 사라진 지 3 초 뒤부터 센다.
- C3: 5 m 앞까지 다가온 표지판만 센다.
- A5: BEV 안에 보이는 바깥 선만 센다. 검출은 lane_like 인 것만 차로로 센다.

#### 켠 것 (코드 기본은 모두 꺼짐 또는 원래 값, launch·yaml 에서 켬)

- 1회차:
  - `carla.yaml`: `style_hold_m: 2.0`
  - `maploop.launch.py` lane_guidance: `guidance.offset_at_m: 1.4096`, `guidance.change_without_neighbour: true`, `guidance.trust_single_line: true`
  - traffic_sign_bridge: `sign.sign_min_height_px: 50`, `sign.light_max_offcentre: 0.3`
  - `run_lane_hud.sh`: `SIGNS_IMGSZ` 기본 1920
- 2회차:
  - `carla.yaml`: `carla_centre_heading_m: 8.0`
  - lane_guidance: `guidance.on_m: 0.25`, `guidance.off_m: 0.20`
  - `run_lane_hud.sh`: YOLO `-p raw_cdr:=true` (검출기 사본의 새 설정. 직렬화된 영상을 받아 처리하는 프레임만 푼다)
- 실시간(2회차):
  - 모두 켰을 때 차선 지연 p95 는 town05 27.7–28.2, s3a 24.8–25.2, s7 26.5–26.8 ms 로 목표 25 ms 에 **미달**이다.
  - 신호·표지판을 끄면 23.1–24.1 ms 다.
  - YOLO 1920 은 11 FPS, 영상 수신 → 검출 p95 113 ms 다.
  - 늘어나는 약 4 ms 는 YOLO 처리 속도·크기와 거의 상관이 없다. 표는 `round_02/REPORT_KO.md` 에 있다.
- 결과 요약:
  - B2 정시: s3a 12 → 19 / 24, s5 0 → 3 / 4, s8c 0 → 2 / 4, s8d 1 → 4 / 4.
  - B5 멀 때: s2 78 → 98 %, s5 32 → 78 %.
  - A3 s7: 49 → 98 %.
  - A4 방향 p95: s1 2.6 → 1.8°.
  - C1 s7: 14 → 80 %.
  - C3 표지판 오류 프레임: 0.

#### 되돌린 시도

- `carla_outer_reach_m` 7.5: 변화 없음(BEV ±6 m).
- 'carried' 목표에서도 차로 변경: s8d 에서 한 번 더 움직임.
- `trust_reprojected_line`: s2 에서 목표가 움직임.
- 내 차로를 뒤축 앞 3 m 기준으로 평가: 나빠짐.
- YOLO 신뢰도 하한 0.4–0.6, 신호 유지 1 초, 투표 1: 확정과 색 오류가 서로 맞바뀜.
- 방향 구간 6·10·12 m: 8 m 보다 나쁨.

#### 3회차에서 켠 것

- `maploop.launch.py`:
  - `image_shm`(기본 '' = 끔)과 `image_shm_every`(2): 차선 작업 프로세스가 결과를 낸 뒤 영상을 공유 메모리에 쓴다(`vision_road/image_share.py`).
  - lane_guidance `stop_line.crosswalk_fallback`: `road_markings` 와 함께 켠다.
- YOLO 사본(`src/yolo_signs`) 파라미터:
  - `image_shm`: 영상 구독 대신 블록을 읽는다.
  - `cpu_threads`, `cpu_affinity`: 시험만 했고 쓰지 않는다.
  - `raw_cdr`.
- `run_lane_hud.sh --signs`: `image_shm:=vision_road_dashcam` 을 양쪽에 준다(`SIGNS_SHM` 으로 바꾸거나 비워 끔).
- 실시간 결과(모두 켬, town05): p95 24.8–25.6 ms(신호 끔 23.1–24.1). D1 은 닫았고 본체에서 다시 잰다.
- 연구 세션 결정으로 더 시도하지 않는 것: C1 신호 확정·s8b 색 오류, C3 미국식 SPEED LIMIT·40 표지판(모델 한계).

#### 지금 돌고 있는 것

- 없음 (2026-11-20 01:50).
- 실행 중인 bash 스크립트는 고치지 않는다. bash 는 읽어 가며 실행하므로 고치면 뒷부분이 깨진다.
- `pgrep -f` 로 기다릴 때 `town05_yolo26m_detector` 같은 이름은 Claude 앱 명령줄에도 있어 기다림이 끝나지 않는다.
- 실시간 측정 중에는 다른 무거운 작업을 돌리지 않는다.

- 새 로스백 12 개: `town05_s1_lanefollow`, `s2_wobble`, `s3a_zigzag_cross`, `s3b_zigzag_inlane`, `s4_straddle`, `s5_lanechange`, `s6_yellow`, `s7_lights`, `s8a_lanefollow_20kmh`, `s8b_lanefollow_50kmh`, `s8c_lanechange_20kmh`, `s8d_lanechange_50kmh` (모두 `_20261119`).
  - 각 폴더에 `README.md`(대본, 토픽, 좌표 약속)와 `sidecar/`가 있다. `sidecar/` 에는 정답 `gt_frames.jsonl`·`.csv`, `lights.csv`, `target.csv`, `ticks.csv`, `static.json`, 지도 `.xodr` 가 들어 있다.
- 정답의 `_vehicle` 점열은 차 원점 기준이다(base_link 는 1.4096 m 뒤). 교차로(`is_junction`)는 평가에서 뺀다.

## 7. 다음에 할 일 (순서대로)

1. (끝) 계획 30 기준선, 1–3회차. 표는 `results/town05_30hz/gt_loop_20261119/SUMMARY_KO.md` 에 있다.
2. (끝) 마지막 HUD 영상 s3a, s5, s6, s7(`final_hud/`). 다시 만들 때 쓰는 명령:
   - `tools/realtime_eval.py --video town05_<장면> --output <폴더> --record --signs --signs-arg full_frame:=true imgsz:=1920 image_shm:=vision_road_dashcam --launch-arg display_filter_inline:=true lane_guidance:=true fake_vehicle_state:=true image_shm:=vision_road_dashcam`
3. 남은 미달(검출기 쪽) → **계획 31 로 진행 중**(2026-11-20 연구 세션이 개선 세션에 전달, 결과 폴더 `results/town05_30hz/gt_loop2_20261120/`, 규칙은 계획 30 과 같고 지연은 신호 끔 24.1 ms·켬 25.6 ms 를 넘기면 되돌림):
   - A5 오른쪽 옆 차로 재현율: s3a 73, s8c 78 %. 바깥 선이 짧게 보이거나 차가 비스듬할 때 놓친다.
   - B2 빠른 차로 변경(s8c 2/4, s3a 19/24): 선을 넘는 동안 검출이 끊긴다.
   - A1 p95 s3a 23 cm, A2 s3a 91 %.
   - A4 방향 p95 커브(3/12 만 1.5° 이하).
   - B5 멀 때 60–98 %(닫음).
4. 모델 한계(재학습은 연구 세션이 사용자에게 알림): 신호등 확정, s8b 빨강 → 초록, 미국식 사각형·40 표지판.
5. 본체 컴퓨터에서 코치 노드와 함께 실행하고, 본체에서 지연을 다시 잰다(D1).

## 8. 알려진 한계와 미해결

- YOLO 모델은 신호등 3색과 속도 표지판 30/60/90 만 안다. 다시 학습하지 않는다. 실제 영상에서 신호등 표시율은 s7 80 %, s1·s8a·s8b 37–49 %, s8b 는 빨간불을 초록으로 약 33 % 읽는다. 미국식 사각형 SPEED LIMIT 표지판과 40 표지판은 못 본다(계획 30 에서 모델 한계로 닫음).
- 신호를 켠 채의 차선 지연 p95 는 이 노트북에서 24.8–25.6 ms(공유 메모리 전달), 끄면 23–24 ms. 본체에서 다시 재야 한다.
- 계획 30 의 평가는 정답 자료의 문제 구간을 뺐다. s4, s6, 그리고 s1·s8a·s8b 의 신호 오표시는 뺀 비율이 30 % 를 넘는다(`gt_loop_20261119/round_03/exclusions.md`).
- 횡단보도는 정답 자료가 없다. 눈으로 본 캡처로만 판단한다.
- HUD 는 안내 경로를 14 m 까지만 그려서, 더 먼 정지선에서 경로를 끊은 것은 화면에 안 보인다.
- 정지 권고는 화면에 표시되지 않고 토픽으로만 나간다.
- 안내는 "지금 벗어난 거리"로만 켜진다. 벗어날 것을 미리 보는 방식은 제안만 했다.
- C++ 이식분과 파이썬의 차이: Town05 0.2 mm 이하, maploop 최대 1.48 cm (원인 미확인).
- 본체 컴퓨터의 CPU 종류를 모른다 (사용자가 나중에 알려 주기로 함).
- 작업 트리에 다른 작업의 미커밋 파일이 있다 (`carla.py`, `path_tracking.py`, `tests/test_path_tracking.py`, `tools/record_maploop_with_hud.py`, `view_maploop_with_hud.sh`). 이번 작업 커밋에 섞지 않는다.

## 9. 일하는 방식 (다른 도구·계정으로 이어받을 때)

지금까지는 Claude Code 세션 셋이 나눠 했다. 한 세션(또는 Codex)이 이어받으면 세 역할을 모두 하면 된다.

| 역할 | 하는 일 |
|---|---|
| 연구·계획 | 문제를 코드에서 특정하고, 목표와 계획을 정하고, `docs/lane_improvement_research.md` 를 쓴다. |
| 구현·측정 | 코드를 고치고, 재고, `results/.../REPORT_KO.md` 를 쓴다. |
| 커밋 | 단계마다 로컬 커밋. 메시지 형식 `[lane] 단계 N: …` (문서), `[lane] 계획 N: …` (코드). 다음 문서 단계 번호는 `WORKLOG_KO.md` 끝을 본다. |

- 한 단계의 순서: 계획 → 구현 → 시험 통과 → 측정 → 보고서 → 연구 문서 갱신 → **이 파일과 WORKLOG 갱신** → 커밋.
- 2분 넘는 명령은 백그라운드로 돌린다.
- 측정하지 못한 것은 "미검증"으로 적는다. 꾸민 입력으로 본 것과 실제 영상으로 본 것을 구분한다.
- Claude 계정의 기억 폴더(`~/.claude/projects/-home-jaehun-Desktop-vision-road/memory/`)에 있는 규칙은 2절에 모두 옮겨 적었다.
