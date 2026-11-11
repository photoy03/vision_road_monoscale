# 검증 기록

- 날짜: 2026-11-11 (Asia/Seoul)
- 설치 위치: `/home/jaehun/Desktop/vision-road`
- 환경: ROS 2 Humble / Python 3.10 / OpenCV 4.12
- 바탕화면에서 7개 패키지 전체 Release 빌드 성공.
- 핵심 및 실제 ROS 메시지 테스트: **12 passed (3.66s)**.
- tracker / odometry / stabilizer / road 네 프로세스의 ROS 그래프 발견 및 설정 조회 성공.
- 실제 설정 조회: tracker `road_from_step=true`, `road_step_esm=true`, `cameras=['front']`, `motion_reference=front`; odometry `fusion_model=velocity`, `imu_yaw_from_gyro=false`, `publish_tf=false`, raw 출력 토픽 확인.
- 네 프로세스 모두 시험 종료 시 정상 종료 확인.
- 실제 카메라·IMU 입력 및 도로 ground truth 없음: 도로 인식률, 주행 오차, FPS 미측정. 동시 기동·설정 연결 시험은 주행 성능 검증과 다름.
- 기존 monoscale_odometry의 printf 형식 경고와 F9R 코드의 GCC stringop 경고가 남아 있음. 기존 모듈 소스는 변경하지 않음.

제한된 환경의 첫 검사에서는 DDS UDP 접근이 제한되어, 최종 노드 연결 시험은 별도 로컬 ROS 도메인(174)에서 수행했습니다. 실제 사용 시 ROS_DOMAIN_ID를 변경할 필요는 없으며, 카메라와 노드가 같은 도메인을 사용하면 됩니다.

테스트 재현:

```bash
cd ~/Desktop/vision-road
source /opt/ros/humble/setup.bash
source install/setup.bash
/usr/bin/python3 -m pytest -q tests/test_core.py tests/test_ros.py
ROS_DOMAIN_ID=174 ROS_LOCALHOST_ONLY=1 /usr/bin/python3 tools/verify_launch.py
```

`verify_launch.py`는 자신의 노드들을 생성하고 종료하므로 센서/실행 중인 그래프와 분리된 도메인에서 사용합니다.

## KITTI 입력 모드 추가 검증 — 2026-11-11

- 바탕화면 `vision_road` 재빌드 성공.
- 기존 기능 + KITTI 형식 검증 + 두 launch 모드: **17 passed (41.09s)**.
- 합성 KITTI Raw 20프레임 오프라인 시험: 20프레임 모두 좌우 차선 검출, OXTS pose 정합, 677개 누적 셀. 이는 합성 입력 결과이며 실제 KITTI 성능 수치가 아님.
- 실제 프로세스 간 `/kitti/image_02/image_rect` 입력 → 검출 선분, 원본 overlay, 고정 TF 및 OXTS 지도 누적 확인.
- MonoScale 모드에서 KITTI pinhole K/D/장착값, 광합 정합 활성화가 실제 tracker 파라미터에 적용되는지 확인. 실제 주행 오도메트리 정확도는 측정하지 않음.
- 실제 KITTI 원본 데이터는 사용자 PC에 아직 없음. 데이터 다운로드 후 `KITTI_TEST.md` 절차로 실영상 검증 필요.

## KITTI 0001 실제 데이터 — 2026-11-11

전체 108프레임 오프라인 처리 및 ROS 차선 메시지 108개 수신을 확인했습니다. 기본 검출 후보는 59/108프레임이며 연석 오검출과 긴 누락 구간이 있습니다. OXTS 기준 차선 누적은 동작했습니다. MonoScale은 광합 정합 fit 실행 후에도 유효 odometry를 발행하지 못했습니다. 성공한 코드 연결과 실패한 알고리즘 결과를 `results/KITTI_0001_REPORT_KO.md`에 구분해 기록했습니다.

KITTI reliable 영상 입력·원래 timestamp 유지 전달 순서·steady 대기 타이머 수정 후 테스트는 **18 passed (15.51s)**입니다. 실제 영상 인식 정확도나 MonoScale 주행 성능 통과를 의미하지 않습니다.

## 그림자 대응 및 프레임 연결 — 2026-11-11

- 국소 대비·차선 띠 응답·후보 탐색 개선과 움직임 보정 프레임 연결 추가.
- 오도메트리 우선, 신뢰할 수 있는 영상 정합 fallback. 예측은 reprojected/source_stamp로 구분하고 지도 누적에서 제외.
- 최대 3프레임/0.35초/3m 및 pose 점프·정합 실패·시간 역행 제한 적용.
- 테스트: **24 passed (15.92s)**.
- 실제 KITTI ROS 재생: 입력/차선 메시지 각각108개, 현재 관측 포함108개, 과거 보완 포함3개, 최대 관측 나이0.2064초, 전역 예측 선분0개.
- 원본과 전후 비교 영상 및 결과는 `results/SHADOW_TEMPORAL_REPORT_KO.md` 참조. 정답 라벨 없이 관측 후보 출력률을 비교했으며 연석 오검출이 남아 있음.

## 점선·실선 분류 — 2026-11-11

기존 차선 메시지를 유지하고 LaneStyle/LaneStyleArray 두 메시지 및 `/road/lane_styles` 토픽을 추가했습니다. 원래 이진 영상의 페인트·공백 반복과 두 프레임 일치로 분류하며 근거가 부족하면 UNKNOWN을 냅니다. 테스트 **29 passed (16.20s)**. 실제 KITTI 0001의 분류 영상과 상세 한계는 `results/LANE_STYLE_REPORT_KO.md` 참조.

## 추가 실차 구간 검증 — 2026-11-11

KITTI 0015/0028의 왼쪽 컬러 영상·OXTS를 직접 다운로드하고 ZIP 항목 CRC를 검사했습니다. 전체 297+430=727프레임에 대해 OXTS 정합 / 영상 정합만의 두 구성을 실행했습니다. 핵심 검출기·설정 SHA-256이 시험 전후 동일함을 확인했습니다. 추가 튜닝이나 코드 변경을 하지 않아 기존 회귀 테스트를 반복하지 않았습니다. 새 데이터 수신·읽기·전체 프레임 처리 및 영상 출력은 실제 실행으로 확인했습니다. 오검출과 분류 불안정을 포함한 결과는 `results/NEW_ROUTES_REPORT_KO.md` 참조.

## 순수 영상 처리 5개 영상 개발 — 2026-11-11

학습 모델/가중치 없이 동작하는 paint_evidence 백엔드를 기본으로 반영하고 메시지 패키지와 검출 패키지를 재빌드했다(20.4초). 최종 V9 전체 테스트 **35 passed (22.02초)**. 실제 KITTI 5개 구간 1,399프레임을 오프라인/ROS에서 모두 처리했으며 차선·분류 배열은 1,399개씩 수신했다. 과거 예측 선분은 전역 지도에 새 관측으로 들어가지 않았다. 인식 표본 기준은 2/5만 통과했으며 5개 전부 정확하다는 결과가 아니다. 자세한 입력·실패·라벨 한계·재현 자료는 results/CLASSICAL_FIVE_VIDEO_REPORT_KO.md 참조.

최종 V12 보완: 미확정 도색 후보의 이력 연장을 제한했다. 전체 검사 **36 passed (21.96초)**. 실제 5개 ROS 재생에서 입력/차선/분류 각각1,399개를 확인했고 예측 선분은 전역 지도에 새 관측으로 들어가지 않았다. 보조 검토 표본의 위치 조건은3/5, 위치+종류는2/5이며 전체 정확도 5개 통과는 달성하지 못했다. 최종 보고서에 실패와 재현 자료를 기록했다.


## 차선 후보 마스크 — 2026-11-11

차선 후보의 지면 가로 폭과 구간별 연결 형태를 검사하는 필터를 추가했다. 빌드 성공, 전체 검사38개 통과(20.07초). 실제5개 구간1399프레임 재처리, 보조 검토 표본의 TP/FP/FN은 기존과 동일하고 위치+종류 조건은2/5로 유지됐다. 후보 마스크25장 표본에서는4.18% 픽셀이 제거됐다. 결과·전후 이미지: results/classical_five_videos/mask_trial/REPORT_KO.md. 기존5개 동영상은 필터 적용 전 V12다.


## CARLA 전용 도색 조각 검출 — 2026-11-11

최종 전체44개 검사 통과(23.69s), 실제CARLA형ROS 입력18장에 차선/종류/이진영상18개씩 수신. 실행/재생 스크립트6프레임과PTS검증. 첨부1,207프레임 순차 비교, 보조도색지점35곳 중10픽셀이내 기존2→최종35; 독립정답/전체정확도 아님. 보닛밖투영점0, 만료예측0. 검출중앙값58.6ms로원본속도실시간완전처리미검증. 실CARLA서버·TF·거리보정은미검증. 결과: results/classical_five_videos/simulation_20261111/carla_final/REPORT_KO.md
