# 계획 28: 규칙 방식 정지선·횡단보도를 켜고 차선 전용 HUD 에 표시 — 결합 완료 (로스백 검증은 미검증)

날짜: 2026-11-19.
- 외장 SSD(`/media/jaehun/38F7-677B`)가 빠져 있다. 로컬 캐시 `data/maploop30hz_native.cdrz` 도 그 SSD 를 가리키는 링크여서, 녹화로 하는 확인(켜진 구간과 맞음/틀림/놓침, 지연 p95, 차선 품질 지표, 실제 HUD 캡처)은 **모두 미검증**이다.
- 아래는 구현과, 녹화 없이 할 수 있는 확인(합성 BEV 단위 시험, 빌드, 가짜 토픽으로 HUD·안내 확인, live 기동)까지다.
- 함께 반영한 결정(연구 세션, 2026-11-19): 차선 전용 HUD 에 **녹색 신호도 표시**한다. 적·황보다 작고 어둡게 그린다.

## 한 일

### 1. 정지선·횡단보도 켜기 (vision_road)

- 지금 구조(계산 프로세스가 영상을 받음, C++ 경로 켬)에서 `road_markings.py` 가 그대로 돈다.
  - `Extras.after` 가 계산 프로세스 안에서 돌고, 결과가 road_node 를 거쳐 `/road/road_markings` 로 나간다.
  - AI 필터(학습 마스크) 없이 돈다.
- 확인:
  - 합성 BEV 의 정지선과 횡단보도 후보가 계획 25·26 의 C++ 설정을 켜도 꺼도 같다(정지선 12.25 m, 횡단보도 1개).
  - 계산 프로세스의 `Extras` 로 세 프레임 뒤 정지선이 확정된다.
  - 새 시험 `tests/test_road_markings_worker.py` 2개.
- 바꾼 것:
  - 정지선·횡단보도 계산을 **표시선 발행 뒤**로 옮겼다(`rt_worker.Perception`). 이전에는 검출과 표시 필터 사이에서 돌아 표시선 지연에 더해졌다.
    - 결과는 같다. 이 프레임의 검출기 상태만 읽는다.
    - 시간은 `worker.markings` 로 따로 잰다.
  - 켜기:
    - `carla.yaml` 에 `road_markings: true` 를 넣었다(코드 기본은 false 그대로).
    - `maploop.launch.py` 의 `road_markings` 인자 기본을 true 로 바꿨다. 이전에는 false 가 yaml 을 덮어써서 yaml 로 켤 수 없었다. `road_markings:=false` 로 끈다.
  - 그래서 `run_lane_hud.sh` 와 `realtime_eval.py` 에서 기본으로 켜진다.
- 노면 표시(화살표·글자) 후보도 `/road/road_markings` 에 그대로 나간다(ns `marking`). 표시하는 쪽(HUD)이 거른다.
- live 기동 확인: `run_lane_hud.sh live --no-hud` 에 합성 영상 40장과 위치를 보냈다. `/road/road_markings` 가 40번 나왔고 오류는 없었다(합성 영상이라 표시는 0개).

### 2. HUD 사본: 차선 전용 모드에서 정지선·횡단보도

- 확인한 사실: 차선 전용 모드는 이미 `drawRoadMarkings` 를 부르고 있었다(`world_item.cpp` 차선 전용 가지). 그래서 토픽이 오면 노면 표시 종류까지 다 그렸다. 지금까지는 토픽이 꺼져 있어 보이지 않았다. `drawRoadPaint` 의 `if (lane_only) return;` 은 원본 칠 조각에 대한 것이다.
- 새 설정 `road.lane_only_markings` (코드 기본 false, `hud.yaml` 에서 true).
  - 끄면 차선 전용 모드는 노면 표시를 하나도 그리지 않는다.
  - 켜면 정지선과 횡단보도만 그린다. 노면 표시 종류(`marking`)는 빼서 화살표·글자는 그리지 않는다.
- 모양:
  - 정지선: 붉은 가로 띠(일반 모드와 같음).
  - 횡단보도: 호박색 테두리. 차선 전용 모드에서는 밝기 0.9 → 0.55 로 어둡게 했다. 안내 경로(호박색 두 줄과 화살촉, 밝기 0.9)와 같은 색이라 안내가 위에 읽히게 하려는 것이다.
  - 그리는 순서는 차선 → 정지선·횡단보도 → 현재 경로 점선 → 안내 경로다. 정지선은 가로, 차선·안내는 세로라 겹쳐도 구별된다(캡처).
- 녹색 신호: `road.lane_only_green: true` (hud.yaml). 녹색 원은 적 팔각형·황 삼각형보다 작게(반지름 0.6 배), 어둡게(밝기 0.55) 그린다.

### 3. 안내 JSON 에 정지선 거리

- `lane_guidance` 사본에 `stop_line.enabled` 를 더했다(코드 기본 false, `maploop.launch.py` 가 `road_markings` 인자와 같이 켬).
- `/road/road_markings` 의 정지선(`stop_line`, 이번 프레임에 안 보였지만 확정된 것 포함) 가운데 차 앞(x > 0)의 가장 가까운 것을 `/hero/lane_guidance` 의 `stop_line_m` 으로 낸다.
  - 단위는 m, 뒤축 기준이다.
  - 없으면 null 이다. 0.35 초보다 오래된 표시는 쓰지 않는다.
  - 꺼져 있으면 JSON 에 이 필드가 없다(이전과 같음).

## 가짜 토픽 확인 (`runs/fake_check/`, `scripts/fake_markings_check.py`)

lane_guidance(신호·정지선 켬)와 HUD 사본(차선 전용)을 띄우고 7초 동안 30 Hz 로 다음을 보냈다:
- ±1.75 m 표시선
- 0–4 초: 20 m 에서 4 m 로 다가오는 정지선
- 4–6 초: 8–12 m 의 횡단보도와 녹색 신호
- 내내: 차 앞 6–7.5 m 의 화살표 노면 표시

- `stop_line_m`: 20.0 → 18.0 → 16.0 … → 6.0(0.5 초마다), 정지선이 사라진 4 초부터 null. 녹색 신호는 4.5–6.0 초에 `light: green`.
- HUD (`captures/fake_markings_sheet.jpg`):
  - 정지선 18 m, 10 m 가 붉은 가로 띠로 보인다.
  - 횡단보도는 어두운 호박색 테두리, 녹색 신호는 작고 어두운 원이다.
  - **화살표 노면 표시는 그리지 않는다.**

시험: hero_hud 33, hero_coach 73, vision_road 156(새 2개) 모두 통과.

## 남은 측정 — 미검증 (SSD 가 연결되면)

- town05_30hz, town05_signs:
  - 정지선·횡단보도가 켜진 구간 목록(시각)
  - 눈으로 본 맞음/틀림/놓침(정답 라벨 없음)
- 차선 지연 p95: road_markings 켬/끔 각 3회. `--signs` 와 함께 켠 것 1조.
  - 계산을 표시선 뒤로 옮겼으므로 표시선 지연은 거의 바뀌지 않을 것으로 본다(재 보지 않은 추정).
- 차선 품질 지표가 켬으로 바뀌지 않는지: 계산이 검출기 상태를 읽기만 하므로 같아야 하지만, 오프라인 비교로 확인해야 한다.
- HUD 캡처: 정지선과 횡단보도가 보일 때.

## 확인하지 못한 것

- Town05 에서 이 규칙 방식의 품질(이전에는 maploop·Town04 몇 장면만 봄).
- 실제 영상에서 횡단보도 테두리와 안내 경로가 겹칠 때 읽히는 정도.

## 파일

| 경로 | 내용 |
|---|---|
| `changes.diff` | 변경 전체 |
| `scripts/fake_markings_check.py`, `runs/fake_check/` | 가짜 토픽 확인(frames 는 커밋하지 않음) |
| `runs/smoke_live/` | live 기동 기록 |
| `captures/fake_markings_sheet.jpg` | HUD 화면 |
| `tests_*.log` | 시험 결과 |

- 백업: `backups/before-road-markings-hud-20261119/`.
