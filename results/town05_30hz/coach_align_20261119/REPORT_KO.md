# 계획 29: 업데이트된 hero_coach 에 맞춰 경로·신호등 로직 다시 맞추기 — 구현 완료 (로스백·coach_node 실행은 미검증)

날짜: 2026-11-19. 설계는 연구 세션이 정했다(사용자 위임).
- 외장 SSD 가 없어 녹화로 하는 확인은 미검증이다.
- 이 노트북에는 `autoware_planning_msgs` 가 없어 `coach_node` 를 띄울 수 없다. 그래서 coach 쪽 추가는 순수 함수의 단위 시험까지만 했다(**실행 미검증**).

## 한 일

### 1. 사본 동기화

- `coach_node.py`, `parking_planner.py` 를 원본(`~/Desktop/hero-coach-main`, 읽기만) 내용으로 갱신했다.
- 그 뒤 사본과 원본의 차이는 우리가 더한 것뿐이다: lane_guidance, traffic_signs, traffic_advice, traffic_sign_bridge, lane_ribbon, 시험, setup.py 의 실행 이름 두 줄, coach_node 의 아래 추가.

### 2. 신호등 로직: 새 노드 `traffic_sign_bridge`

- 입력 `/yolo/detections` 를 `traffic_signs.SignTracker` 의 투표(3 중 2, 신호 0.5 초, 표지판 3 초 유지)에 넣는다.
- 출력은 원본 코치의 약속 그대로다: `/hero/traffic_light` (SignalIndicator), `/hero/sign` (SignMarker: KIND_SPEED, value, SOURCE_FRONT, confidence).
- 확정 상태인 동안 검출 메시지마다 다시 보낸다(코치의 1 초 타임아웃 안). 꺼지면 STATE_NONE / KIND_UNKNOWN 을 한 번 보낸다.
- 여러 신호등이 보일 때:
  - 점수 = 신뢰도 × (상자 넓이/1000 px²)^0.5 × 가로 중앙 가우스(σ = 영상 폭의 0.25)
  - 파라미터 `sign.select_area_power`, `sign.select_centre_sigma`. 둘 다 0 이면 계획 27 의 "신뢰도 최고"와 같다.
  - 시험: 가장자리의 작고 더 확실한 녹색(0.9)보다 가운데의 큰 적색(0.7)을 고른다. 예전 규칙이면 녹색이다.
- `lane_guidance` 는 `/yolo/detections` 직접 구독을 없앴다.
  - 대신 coach_node 와 같은 방식으로 `/hero/traffic_light`, `/hero/sign` 을 구독한다(1 초 타임아웃, NONE/UNKNOWN 무시).
  - 경로는 YOLO → bridge → (coach_node 또는 lane_guidance) 한 줄이다. carla_traffic 과도 바꿔 끼울 수 있다.

### 3. 경로 로직: `/hero/scene_overlay` 발행자를 하나로

- `lane_guidance` 는 차로 안내 띠를 매 프레임 `/hero/lane_guidance/ribbon` (TrajectoryRibbon, base_link)으로 낸다(안내가 없으면 점 0개).
- 새 파라미터 `publish_scene` (기본 true = 지금처럼 SceneOverlay 를 직접 냄). false 면 내지 않는다(본체에서 coach_node 와 같이 돌 때).
- `coach_node` 사본에 최소 추가:
  - 파라미터 `lane_ribbon_topic`, 기본 빈 문자열이면 원본과 같다. 그리고 `lane_ribbon_timeout_sec` 0.5.
  - 주차 계획이 없거나 지금 구간이 없을 때, 0.5 초 안에 온 띠(점 2개 이상)를 `has_ideal/ideal` 로 싣는다.
  - 주차 궤적이 있으면 주차가 이긴다. `advised_gear` 는 건드리지 않는다.
  - 판단은 순수 함수 `lane_ribbon.lane_ideal` 이다. 시험했고, coach_node 실행은 미검증이다.

### 4. 신호 + 정지선 판단: 새 모듈 `traffic_advice.py`

- `lane_guidance` 가 `traffic.enabled` 일 때 JSON 에 `"traffic"` 을 싣는다: light, stop_line_m, advice, distance_m, decel_mps2, speed_limit_kph, over_limit, ribbon_cut.
- 차 앞끝까지 거리 d = stop_line_m − 3.656 m(뒤축→앞범퍼, Prius). 필요 감속도 a = v²/(2·max(d, 0.1)). 속도는 `/hero/vehicle_state`, 없으면 위치 변화.
- 빨강:
  - 정지(|v| < 0.3) → `hold`
  - 정지선 모름 → `stop_unknown_distance` (거리·감속도 null)
  - d ≤ 0 이고 움직임 → `over_line`
  - a ≤ 2.0 → `prepare_stop`, ≤ 3.5 → `stop`, 그 위 → `stop_hard`
- 노랑: a ≤ 3.5 면 빨강과 같고, 넘으면 `proceed`.
- 초록/없음: `none`.
- 제한 속도: 확정된 표지판을 다음 표지판까지 기억한다. 제한 + 5 km/h 를 1 초 넘으면 `over_limit` true.
- 수치는 모두 `traffic.<이름>` 파라미터다. 기능은 코드 기본 꺼짐이고, launch 가 `signs:=true` 일 때 켠다.
- 화면: 빨강·노랑이고 advice 가 정지 계열(hold / prepare_stop / stop / stop_hard)이며 정지선을 알 때, 안내 띠(ideal 과 ribbon 토픽)를 정지선에서 끝낸다.
  - 안내 띠가 없던 상황(차로 중앙 주행)에는 새로 만들지 않는다.
  - 안내 문구(코칭)는 그대로 둔다. HUD 사본은 손대지 않았다.
  - HUD 는 안내 띠를 14 m 까지만 그리므로, 그보다 먼 정지선에서의 자르기는 화면에 드러나지 않는다.

### 5. 실행 연결

- `maploop.launch.py signs:=true` 가 `traffic_sign_bridge` 를 띄우고 lane_guidance 의 `signs.enabled`, `traffic.enabled` 를 켠다.
- `run_lane_hud.sh --signs` 와 `realtime_eval.py --signs` 는 그대로 쓴다(검출 노드 + launch signs:=true).
- 노드가 launch 종료 신호(ExternalShutdownException)로 멈출 때 오류 없이 끝나게 했다(lane_guidance, bridge).

## 확인 (SSD 없이)

- 시험: hero_coach 96(새 `test_traffic_advice.py` 23개: 선택 규칙 2, bridge 메시지, advice 분기 16개와 경계값 2.0·3.5, 숫자, 제한 속도, 띠 자르기, coach 띠 대체), vision_road 156, hero_hud 33 모두 통과.
- **신호 끔일 때 계획 28 직후와 같음:**
  - 같은 가짜 입력(9 초, 270 프레임)을 f111f20 의 lane_guidance 와 지금 것에 넣었다(`runs/identity/`).
  - `/hero/scene_overlay` 와 `/hero/lane_guidance` JSON(compute_ms 제외)이 **270/270 프레임 모두 같다**.
- 빨강 접근 가짜 통합(`scripts/fake_red_approach.py`, `runs/fake_red/`):
  - 차 8 m/s, 차로 중앙에서 0.6 m 벗어나 복귀 안내가 나오는 상황이다. 정지선이 30 m 에서 다가오고 0–6 초 빨강, 그 뒤 초록이다.
  - JSON 의 advice: 정지선 보이기 전 `stop_unknown_distance` → 30 m 에서 `prepare_stop`(1.21 m/s²) → 18 m 에서 `stop`(2.23) → 10 m 에서 `stop_hard`(5.04) → 앞끝이 선을 지난 2 m 에서 `over_line` → 초록 `none`.
  - 장면의 ideal 끝: 정지선을 따라 30 → 27.3 → 22.0 → … → 6.0 m 로 줄고, 초록이 되면 30 m 로 돌아온다.
  - HUD 캡처 `captures/fake_red_approach_sheet.jpg`: 정지선 앞에서 안내 띠와 화살촉이 끝나고, 초록에서는 원래대로다.
- `./run_lane_hud.sh live --signs` 기동(`runs/smoke_live/`):
  - 검출 노드 → bridge → lane_guidance → HUD 가 모두 떴다. `/hero/traffic_light`, `/hero/sign` 은 발행 1·구독 1 이고 `/hero/lane_guidance/ribbon` 은 발행 1 이다.
  - 멈추면 모두 내려갔다. 오류 0.

## 미검증 (나중에)

- SSD 뒤: Town05 두 영상에서 실제 동작. town05_signs 에는 조향·속도 토픽이 없으니 `--fake-vehicle-state` 를 쓴다.
- 본체 컴퓨터에서 `coach_node` + `lane_ribbon_topic` + `publish_scene: false` 실행.

## 후속 제안 (이번 범위 밖)

- 음성(voice_node): `traffic.advice` 가 `stop` / `stop_hard` 로 바뀌는 순간과 `over_limit` 에 짧은 음성 안내를 붙일 수 있다.

## 파일

| 경로 | 내용 |
|---|---|
| `changes.diff` | 변경 전체 |
| `scripts/fake_red_approach.py` | 빨강 접근 가짜 입력 (FULL=1: 동일성 비교용 전체 기록) |
| `runs/fake_red/`, `runs/identity/`, `runs/smoke_live/` | 실행 기록 (frames png 는 커밋하지 않음) |
| `captures/fake_red_approach_sheet.jpg` | HUD 화면 |
| `tests_*.log` | 시험 결과 |

- 백업: `backups/before-coach-align-20261119/`.
