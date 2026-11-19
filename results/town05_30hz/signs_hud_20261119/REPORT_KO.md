# 계획 27: 신호등·속도 표지판(YOLO)을 차선 HUD 에 함께 표시 — 결합 완료 (로스백 검증은 미검증)

날짜: 2026-11-19.
- 범위(사용자 결정 2026-11-19): ultralytics 는 설치·변경하지 않는다(사본이 quantize 대신 half 로 돈다). 이번 목표는 결합 완료까지다.
- 외장 SSD(`/media/jaehun/38F7-677B`)가 빠져 있어 **로스백으로 하는 검증(검출 수, 지연 p95, 타일/전체 비교, 실제 화면 캡처)은 모두 미검증**이다. SSD 가 연결되면 "남은 측정"을 한다.
- 아래는 구현과, 로스백 없이 할 수 있는 확인(단위 시험, 빌드, 가짜 `/yolo/detections` 로 HUD 표시, `run_lane_hud.sh live --signs` 기동)까지다.

## 한 일

### 1. 검출 노드 사본 `src/yolo_signs/town05_yolo26m_detector.py`

- 원본(`~/Desktop/town05_yolo26m_detector/`)은 그대로 두었다. 원본과의 차이는 `changes.diff` 끝에 있다.
- `best.pt` 는 복사하지 않았다. 기본 `model_path` 는 사본 옆의 `best.pt`, 없으면 원본 폴더의 것이다. `.gitignore` 에 `src/yolo_signs/*.pt` 를 넣었다.
- 더한 것(파라미터 기본값은 원본 동작 그대로):
  - `quantize`: 이 컴퓨터의 ultralytics 8.4.48 은 `predict(quantize=16)` 을 "not a valid YOLO argument" 로 거부한다. 그러면 한 번 경고를 내고 `half=True` 로 바꾼다. quantize 를 받는 판에서는 원래 경로 그대로다.
    - 확인: 이 컴퓨터에서 경고가 나온 뒤 검출 메시지가 계속 나왔다.
  - `full_frame` (기본 false = 원본의 위쪽 가운데 640×640 타일): 전체 영상으로 추론.
  - `stats_path`: 처리한 프레임마다 영상 시각, 받은·시작·끝 시각, ultralytics 단계 시간, 검출 수를 노드가 멈출 때 JSON 으로 쓴다(측정용).
- 실행 환경: `~/yolo_env/bin/python` (ultralytics 8.4.48, torch 2.11) 에 ROS 와 이 작업공간을 source 하면 rclpy 와 모델이 함께 뜬다(확인함).

### 2. 안내 노드 사본 (`hero_coach`) — 신호·표지판을 같은 장면에

- `/hero/scene_overlay` 의 발행자는 lane_guidance 하나로 둔다. 신호·표지판도 그 메시지에 싣는다.
- 새 모듈 `hero_coach/traffic_signs.py` (`SignTracker`, ROS 없음). 시각은 검출 메시지의 영상 시각(시뮬레이터 시각)이다.
  - 신호등:
    - 메시지마다 신호등 클래스 중 신뢰도가 가장 높은 것(같으면 큰 상자)이 그 메시지의 상태다. orange 는 YELLOW 다.
    - 최근 3개 메시지 중 2개 이상이 같은 상태면 표시한다.
    - 신호등이 든 마지막 메시지에서 0.5 초가 지나면 끈다.
    - 그 공백 뒤에는 투표를 처음부터 다시 한다(옛 상태가 한 프레임으로 되살아나지 않게).
  - 속도 표지판: 30 / 60 / 90 을 같은 투표로 확정하고, 마지막으로 본 뒤 3 초 유지한다. `KIND_SPEED`, `value` 에 담는다.
  - position: 검출기에 거리가 없어 0 으로 둔다. **HUD 는 position 을 쓰지 않는다.** `ros_bridge.cpp` 가 복사만 하고 그리는 곳(`chrome_item.cpp`)은 고정 자리에 그린다.
- 노드:
  - `signs.enabled` 가 참일 때만 `/yolo/detections` 를 구독한다(코드 기본 꺼짐).
  - 꺼져 있으면 구독이 없다. 장면 메시지와 `/hero/lane_guidance` JSON 이 이전과 같다(JSON 의 `light`, `speed_sign` 은 켰을 때만 더한다).
  - 수치는 모두 `signs.<이름>` 파라미터다(`light_window` 3, `light_votes` 2, `light_hold_s` 0.5, `sign_window` 3, `sign_votes` 2, `sign_hold_s` 3.0, `min_confidence` 0, `topic`).
- 켜는 곳: `maploop.launch.py signs:=true` (`run_lane_hud.sh --signs` 가 넘김).
- 시험: `src/hero_coach/test/test_traffic_signs.py` 5개.

### 3. HUD 사본 — 차선 전용 모드에서 신호와 표지판만

- 새 설정 `road.lane_only_signals` (코드 기본 false, `config/hud.yaml` 에서 true).
  - 켜면 차선 전용 모드에서도 고정 자리 그림(chrome)을 숨기지 않는다. 다만 신호(상단 가운데)와 속도 표지판(우측)만 그린다.
  - 기어와 대기 원은 그리지 않는다.
  - 차량 상태가 없어 HUD 모드가 Normal 이 아니어도, 장면이 신선하면 그린다.
  - 코칭 심각도(차선 안내)로 표지판을 지우지 않는다.
- 신호 모양은 일반 모드와 같다(적: 팔각형, 황: 점멸 삼각형).
  - **녹색은 일반 모드처럼 그리지 않는다.** HUD_전략.md 의 "행동을 요구할 때만 뜬다" 규칙이다.
  - 그리려면 `road.lane_only_green: true` (녹색 원, 코드 기본 false, yaml 도 false). 사용자가 녹색도 보기를 원하면 이 줄 하나만 바꾸면 된다.
- 이 HUD 빌드에는 Qt5Svg 가 없어 표지판 SVG 가 그려지지 않는다(빌드 경고 "sign icons are not drawn").
  - 차선 전용 신호 모드에서는 속도 표지판을 붉은 테두리 원과 숫자로 그린다. 흰 바탕은 가산광에서 빛 덩어리가 되므로 그리지 않는다.
  - 일반 모드의 동작은 바꾸지 않았다.
- 덤으로 고친 것: 고정 자리 그림의 내용 지문에 표지판 값이 빠져 있었다. 그래서 30 → 60 처럼 값만 바뀌면 다시 그리지 않았다. 값을 넣었다(일반 모드에도 해당하는 수정).

### 4. 실행

- `./run_lane_hud.sh town05 --signs`: 차선 + 안내 + HUD 에 검출 노드를 더 띄운다(YOLO 환경 파이썬).
  - 환경변수 `YOLO_PYTHON`(기본 `~/yolo_env/bin/python`), `SIGNS_FULL_FRAME`(기본 true, 아래 비교 뒤 확정), `SIGNS_FPS`(기본 0 = 모든 프레임).
  - 영상 `town05_signs` 도 고를 수 있게 했다(SSD 의 `rosbag/town05_signs`, 캐시가 없으면 녹화 파일을 바로 읽음 — SSD 가 없어 경로는 미확인).
- `tools/realtime_eval.py --signs [--signs-arg full_frame:=true max_inference_fps:=10]`:
  - 검출 노드를 함께 띄우고, 그 발행자가 뜰 때까지 기다린 뒤 재생한다.
  - 프레임별 시간은 `signs_stats.json` 에 쓴다.

## 동작 확인 (녹화 없이, `runs/fake_check/`, `scripts/fake_signs_check.py`)

lane_guidance(`signs.enabled`)와 HUD 사본(차선 전용, 신호 켬)을 띄웠다. 빈 차로 중앙선을 30 Hz 로 보내고, 꾸민 검출을 다음 순서로 보냈다: 적 2 초, 황 2 초, 녹 + 60 표지판 3 초, 아무것도 없음 2 초, 다섯 프레임마다 한 번 적.

장면이 바뀐 시각(대본 시각 기준):

| 시각 | 신호 | 표지판 | 설명 |
|---|---|---|---|
| 0.033 s | 적 | – | 두 번째 프레임에서 확정 |
| 2.033 s | 황 | – | |
| 4.033 s | 녹 | 60 | |
| 7.5 s | – | 60 | 마지막 신호 0.5 초 뒤 꺼짐 |
| 10.0 s | – | – | 마지막 표지판 3 초 뒤 꺼짐 |
| 9–11 s | – | – | 한 프레임짜리 적(5 프레임에 한 번)은 끝까지 표시되지 않음 |

HUD 화면: `captures/fake_check_sheet.jpg`. 적 팔각형, 황 삼각형(점멸), 녹색은 그리지 않음 + 60 표지, 신호가 꺼진 뒤 표지 유지, 그 뒤 아무것도 없음.

시험: hero_hud 33, hero_coach 73(새 5개 포함), vision_road 154 모두 통과.

## 기동 확인 (`run_lane_hud.sh live --signs`, `runs/smoke_live/`)

- 녹화 없이 live 로 띄우고 합성 카메라 영상(1920×1080)을 20장 보냈다. 차선 검출, 표시 필터, lane_guidance(신호 켬), HUD, 검출 노드(YOLO 환경)가 모두 떴다.
- 검출 노드는 영상을 받아 quantize → half 전환 경고 뒤 `/yolo/detections` 를 냈다(합성 영상이라 검출 0개). lane_guidance 가 그 토픽을 구독했다.
- 스크립트를 멈추면 모두 함께 내려갔다.
- 처음 띄울 때 고친 것: `SIGNS_FPS` 기본값 `0` 이 정수 파라미터로 넘어가 검출 노드가 "expecting type DOUBLE" 로 죽었다.
  - 기본값을 `0.0` 으로 바꿨다.
  - 사본이 `max_inference_fps` 를 정수(`:=10`)로 받아도 되게 했다(동적 타입 파라미터).

## 남은 측정 — 미검증 (SSD 가 연결되면)

- 타일 방식과 전체 영상 방식 비교: 원본 해상도 실제 프레임(town05_30hz, town05_signs). 클래스별 검출 프레임 수와 추론 시간을 보고 기본값(`SIGNS_FULL_FRAME`)을 확정한다.
- 깜빡임 횟수, 검출 노드 FPS 와 추론 시간.
- 차선 지연 p95: `--signs` 켬/끔 각 3회(전체 영상). 25 ms 를 넘으면 `max_inference_fps` 10 으로 낮춘 경우도 잰다.
- CPU·GPU 사용률, GPU 메모리.
- HUD 캡처(신호등 보일 때, 표지판 보일 때, 둘 다 없을 때)와 눈으로 본 정확도 몇 장.

## 확인하지 못한 것

- 실제 영상에서의 검출, 지연, 자원 사용(SSD 없음).
- `town05_signs` 녹화의 실제 경로와 캐시 유무.
- 실제 본체 컴퓨터의 ultralytics 에서 `quantize=16` 이 되는지(되면 그대로 쓴다).

## 파일

| 경로 | 내용 |
|---|---|
| `changes.diff` | 변경 전체(검출 노드 사본은 원본과의 diff) |
| `scripts/fake_signs_check.py` | 가짜 입력 동작 확인 |
| `runs/fake_check/` | 그 실행 기록(scenes.json, 로그). frames 는 커밋하지 않음 |
| `captures/fake_check_sheet.jpg` | HUD 화면 |
| `tests_*.log` | 시험 결과 |

- 백업: `backups/before-signs-hud-20261119/`.
