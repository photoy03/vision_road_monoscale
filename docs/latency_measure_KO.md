# 다른 컴퓨터에서 차선 쪽 지연 재기 (RTX 5070 컴퓨터용)

차선 쪽 실시간 지연(영상이 나온 때부터 표시선이 나갈 때까지)을 그 컴퓨터에서 한 줄로 잰다. 기준은 계획 25 와 같다. Town05 녹화를 30 Hz 로 다시 틀고, 시연 설정(바깥 선, 중앙선, 안내, 가짜 차량 상태, 계산 프로세스 안의 표시 필터, HUD 사본)으로 돌린다.

## 필요한 것

| 무엇 | 어디 | 크기 |
|---|---|---|
| 이 작업공간 (빌드된 것) | 아무 곳. `colcon build --symlink-install` (BUILD_AND_TEST.md) | – |
| ROS 2 Humble | `/opt/ros/humble` | – |
| Town05 녹화 | `<디스크>/rosbag/town05_30hz/` | 27 GB |
| Town05 재생 캐시 | `<디스크>/vision-road-cache/town05_30hz_native.cdrz` 와 `.cdrz.json` | 5 GB |
| AI 필터 가중치 (선택) | `data/models/yolopv2/yolopv2.pt` | 150 MB |

- C++ 부분(`src/vision_road/vision_road/native.cpp`)은 `colcon build` 가 그 컴퓨터에서 새로 만든다(만든 파일 `_native*.so` 는 git 에 없다). 만들지 못하면 조용히 파이썬 코드로 돌아 느려지므로, `system.txt` 의 "native fast path" 줄이 `built` 인지 확인한다.
- `<디스크>` 는 기본이 `/media/jaehun/38F7-677B` 다. 지금 쓰는 USB 디스크를 그대로 꽂으면 된다. 다른 곳에 두면 `VISION_ROAD_DISK=<경로>` 를 붙인다.
- 캐시는 녹화 파일의 크기와 수정 시각을 확인한다. 파일을 복사한다면 `rsync -a` 처럼 시각을 지켜 복사한다. 시각이 다르면 재생 도구가 캐시를 쓰지 않고 느린 녹화 파일을 읽는다.
- 패키지를 새로 설치하거나 시스템 설정을 바꾸지 않는다. 다른 무거운 프로그램(원격 화면, 브라우저 영상 등)은 닫고 잰다. 원격 화면 프로그램 하나가 CPU 43% 를 쓸 때 p95 가 47–60 ms 까지 오른 적이 있다.

## 명령

```bash
tools/measure_latency.sh
```

- 기본: 잘라 보내기 3번, 전체 영상 1번, AI 필터 3번(가중치가 있을 때). 한 번에 약 70초, 전체 약 9분이다.
- 짧게 확인만: `RUNS=1 FRAMES=600 tools/measure_latency.sh` (20초짜리 세 번).
- 디스크 경로가 다르면: `VISION_ROAD_DISK=/mnt/data tools/measure_latency.sh`

결과는 `results/latency_<컴퓨터 이름>_<날짜시각>/` 에 쌓인다.

| 파일 | 내용 |
|---|---|
| `system.txt` | CPU 이름·코어 수·최대 클럭, CPU 주파수 정책(governor, energy_performance_preference), 전원 프로필, 전원 연결 여부, GPU 이름·드라이버·전력 한도, 메모리, 시작 전 부하, python·numpy·OpenCV 판. 읽기만 한다 |
| `crop1/` … `ai3/` | 실행마다 `summary.json`(지연 통계), `frames.json`(프레임별 시각), 로그 |
| `SUMMARY.txt` | 아래 표 |

## 결과 읽기 (`SUMMARY.txt`)

```
run                  processed    p95   mean    max gaps |    delivery      warp      detect   display   learned  pass(p95<=25, 100%, no gaps)
crop1                  100.00%  23.40  19.29   34.1    0 |  3.8/  4.2  0.7/ 0.9   11.0/14.2  2.6/ 3.5     -      pass
```

(이 노트북, 계획 25 의 첫 번째 잘라 보내기 실행 값)

- `p95`: 영상 발행 → 표시선 받음의 95 백분위(ms). **기준은 25 ms 이하**다.
- `processed`: 처리한 프레임 비율. 100% 여야 한다.
- `gaps`: 표시선이 100 ms 넘게 끊긴 횟수. 0 이어야 한다.
- 단계(평균/p95, ms):
  - `delivery`: 영상 전달(재생 도구 발행 → 계산 프로세스가 받음). 잘라 보내면 약 3.5 ms, 전체 영상이면 약 6 ms(이 노트북)
  - `warp`: BEV 변환
  - `detect`: 검출
  - `display`: 표시 필터와 발행
  - `learned`: AI 필터가 켜졌을 때 모델에 영상을 넘기는 시간
- 마지막 줄: 종류별로 몇 번 통과했는지(`crop#`: 잘라 보내기, `full#`: 전체 영상, `ai#`: AI 필터).

## 이 노트북의 값과 비교

이 노트북(Ryzen 7 5800H, RTX 3050 Laptop 4 GB, 16 GB)의 값은 `results/classical_five_videos/town05_30hz/latency25_20261118/REPORT_KO.md` 에 있다.

- 차선 검출은 CPU 코어 하나가 시간을 정한다. 다른 컴퓨터의 값은 주로 CPU 의 단일 코어 속도로 바뀐다.
- GPU 가 좋아지면 AI 필터 모델(이 노트북에서 한 번에 약 21 ms, 두 프레임에 한 번)이 빨라진다. 다만 모델은 따로 돌기 때문에 표시선 지연에는 직접 들어가지 않는다.

## 잘라 보내기 행에 대해

- 측정은 재생 도구가 행 578–983 만 보낸다(`--crop-rows 578:984`). 지금 BEV 가 읽는 행이 580–981 이고, 위아래로 2행씩 여유를 둔 것이다. 받는 쪽은 `CameraInfo.roi` 로 행을 맞춘다(약속: `src/hero_calib/OUTPUT_CONTRACT_KO.md`).
- 카메라 보정, 자리, BEV 범위를 바꾸면 읽는 행도 바뀐다. 차선 인식 로그에 "the BEV reads image rows A-B" 로 나오니, 그 값으로 다시 정한다.
- 읽는 행을 다 담지 않은 영상은 버리고 경고를 낸다. 이때 `processed` 가 100% 아래로 떨어진다.
