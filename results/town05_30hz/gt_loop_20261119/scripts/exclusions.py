"""Plan 30: per scene, the frames each evaluation rule leaves out (count and share); over 30 % marked.
usage: exclusions.py ROUND_DIR [--tag _1920]"""
import json, sys
from pathlib import Path
R = Path(sys.argv[1]);tag = sys.argv[sys.argv.index('--tag')+1] if '--tag' in sys.argv else ''
S = ['s1_lanefollow', 's2_wobble', 's3a_zigzag_cross', 's3b_zigzag_inlane', 's4_straddle', 's5_lanechange', 's6_yellow', 's7_lights',
     's8a_lanefollow_20kmh', 's8b_lanefollow_50kmh', 's8c_lanechange_20kmh', 's8d_lanechange_50kmh']
def cell(n, d):
    if not d:return '—'
    v = n/d;return f'{n} ({100*v:.1f} %)'+(' **⚠**' if v > .3 else '')
print('| 장면 | 평가 프레임 (교차로 밖) | 선 위 0.3 m 제외 (A1–A5) | 정답 오류 제외 좌 (A1–A3) | 정답 오류 제외 우 | 정답 오류 제외 (A4) | C2 제외 (교차로·신호 사라진 뒤 3 초) |')
print('|---|---|---|---|---|---|---|')
for s in S:
    f = R/s/f'metrics{tag}.json'
    if not f.exists():f = R/s/'metrics.json'
    if not f.exists():continue
    m = json.load(open(f));used = m['used_frames'];ex = m.get('A1_gt_excluded', {});c2 = m.get('C', {}).get('C2_no_light_shown', {})
    print(f"| {s.split('_')[0]} | {used} | {cell(m.get('A1_on_line_frames', 0), used)} | {cell(ex.get('left', {}).get('frames', 0), used)} | "
          f"{cell(ex.get('right', {}).get('frames', 0), used)} | {cell(m.get('A4_gt_excluded', 0), used)} | "
          f"{cell((c2.get('raw_frames') or 0)-(c2.get('frames') or 0), c2.get('raw_frames')) if c2 else '—'} |")
