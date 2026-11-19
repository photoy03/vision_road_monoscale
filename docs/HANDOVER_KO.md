# 인수인계 — vision-road 차선·안내·신호 프로젝트

이 파일만 읽고 바로 이어서 작업할 수 있게 쓴다. **작업 하나가 끝날 때마다 "지금 상태"와 "다음에 할 일"을 고치고, `docs/WORKLOG_KO.md` 에 한 줄을 더한다.**

마지막 갱신: 2026-11-19 (계획 30 시작 직후, 기준선 측정 전)

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

git 에 없는 것(다른 컴퓨터로 옮길 때 따로 복사): `results/`, `data/`, `build/`, `install/`, `best.pt`, `data/models/yolopv2/yolopv2.pt`.

## 4. 빌드, 시험, 실행

```bash
cd ~/Desktop/vision-road
source /opt/ros/humble/setup.bash
colcon build --base-paths src --symlink-install --parallel-workers 2
source install/setup.bash
```

- C++ 빠른 경로(`_native*.so`)는 빌드가 만든다. 없으면 조용히 파이썬으로 돌아 느려진다.
- 시험 (2026-11-19 계획 29 기준 통과 수): `hero_coach` 96, `vision_road` 156, `hero_hud` 33. vision_road 는 `/usr/bin/python3 -m pytest -q tests`, hero_coach 는 `/usr/bin/python3 -m pytest -q src/hero_coach/test`. 정확한 명령은 최근 `results/town05_30hz/*/REPORT_KO.md` 와 그 폴더의 `scripts/` 를 본다.
- 이 노트북에는 `autoware_planning_msgs` 가 없어 `coach_node`, `parking_planner` 는 뜨지 않는다.

실행:

```bash
./run_lane_hud.sh town05                    # 기준 영상 한 번 재생 (차선 + 안내 + HUD)
./run_lane_hud.sh town05 --signs            # 신호등·표지판 YOLO 도 함께 (GPU, ~/yolo_env 파이썬)
./run_lane_hud.sh live --signs              # 재생 없이, 밖에서 영상·위치 토픽이 들어올 때
```

주요 인자: `--no-hud`, `--loop`, `--frames N`, `--learned`(끔 유지), `--no-guidance`, `--fake-vehicle-state`(조향·속도 토픽이 없는 영상용), `--crop-rows`(쓰지 않음), `--signs`. 환경 변수 `YOLO_PYTHON`, `SIGNS_FULL_FRAME`(기본 true), `SIGNS_FPS`(기본 0.0 = 제한 없음).

YOLO 환경: `/home/jaehun/yolo_env` (ultralytics 8.4.48, torch 2.11, numpy 2.2). 이 버전은 `quantize` 인자를 몰라서 사본이 `half=True` 로 바꿔 돈다. 시스템 python3 로는 모델이 안 뜬다.

지연 재기: `tools/realtime_eval.py` (영상 발행 → `/road/lane_display_markers` 수신, p95). 세 번 재서 본다. **NoMachine 원격 화면이 켜져 있으면 수치가 2배로 나온다.**

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
- 결과 폴더: `results/town05_30hz/gt_loop_20261119/` (`SUMMARY_KO.md` 누적 표, `round_NN/REPORT_KO.md`).
- 진행 정도: 재생 캐시 만들기 단계 (`scripts/build_caches.sh`, `logs/`). **기준선 측정 결과 아직 없음.**
- 새 로스백 12개: `town05_s1_lanefollow`, `s2_wobble`, `s3a_zigzag_cross`, `s3b_zigzag_inlane`, `s4_straddle`, `s5_lanechange`, `s6_yellow`, `s7_lights`, `s8a_lanefollow_20kmh`, `s8b_lanefollow_50kmh`, `s8c_lanechange_20kmh`, `s8d_lanechange_50kmh` (모두 `_20261119`). 각 폴더에 `README.md`(대본, 토픽, 좌표 약속)와 `sidecar/`(정답 `gt_frames.jsonl`·`.csv`, `lights.csv`, `target.csv`, `ticks.csv`, `static.json`, 지도 `.xodr`).
- 새 로스백에는 실제 `/hero/vehicle_state` 가 있어 `--fake-vehicle-state` 가 필요 없다. 정답의 "_vehicle" 점열 원점이 차 원점인지 base_link 인지는 README·`static.json` 에서 확인해 쓴다. 교차로(`is_junction`)는 평가에서 뺀다.

## 7. 다음에 할 일 (순서대로)

1. 정답 대비 평가 스크립트를 `tools/` 에 만든다 (영상 하나 → metrics JSON).
2. 14개 영상(새 12개 + `town05_30hz` + `town05_signs`) 기준선 측정: AI 필터 끔, 전체 영상, 정지선·횡단보도 켬, `--signs` 켬.
3. 목표표와 비교해 가장 못 미치는 항목 하나를 고치고 14개 전체를 다시 잰다. 다른 곳이 나빠지면 되돌린다. 반복.
4. 끝내는 조건: 모든 목표 충족, 또는 같은 항목에 서로 다른 시도 3번이 실패(원인·수치를 적고 넘어감).
5. 끝나면 장면별 HUD 영상(s3a, s5, s6, s7)을 만든다.
6. 그 뒤 남는 일: 본체 컴퓨터에서 코치 노드와 함께 실행, 본체 지연 측정(`tools/measure_latency.sh`), 정지 권고 음성 안내(제안만 됨).

## 8. 알려진 한계와 미해결

- YOLO 모델은 신호등 3색과 속도 표지판 30/60/90 만 안다. 다시 학습하지 않는다. 실제 영상 정확도는 아직 모른다.
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
