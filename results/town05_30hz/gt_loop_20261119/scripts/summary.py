"""Plan 30: one round's table against the targets.  usage: summary.py ROUND_DIR [--tag _tile] > table.md"""
import json, sys
from pathlib import Path
R = Path(sys.argv[1]);tag = sys.argv[sys.argv.index('--tag')+1] if '--tag' in sys.argv else ''
S = ['s1_lanefollow', 's2_wobble', 's3a_zigzag_cross', 's3b_zigzag_inlane', 's4_straddle', 's5_lanechange', 's6_yellow', 's7_lights',
     's8a_lanefollow_20kmh', 's8b_lanefollow_50kmh', 's8c_lanechange_20kmh', 's8d_lanechange_50kmh']
pc = lambda v: '—' if v is None else f'{100*v:.1f}'
cm = lambda v: '—' if v is None else f'{100*v:.1f}'
rows = []
for s in S:
    f = R/s/f'metrics{tag}.json'
    if not f.exists():f = R/s/'metrics.json'
    if not f.exists():continue
    m = json.load(open(f));a5 = m['A5'];b = m['B_lane_changes'];c = m['C']
    st = m['A3_style'];co = m['A3_colour']
    rows.append([s.split('_')[0], cm(m['A1_all']['median']), cm(m['A1_all']['p95']), pc(m['A1_identity']),
                 f"{pc(m['A2_observed']['left'])}/{pc(m['A2_observed']['right'])}",
                 f"{pc(st['left']['accuracy'])}/{pc(st['right']['accuracy'])}", f"{pc(co['left']['accuracy'])}/{pc(co['right']['accuracy'])}",
                 cm(m['A4_offset_m']['median']), cm(m['A4_offset_m']['p95']), '—' if m['A4_heading_deg']['p95'] is None else f"{m['A4_heading_deg']['p95']:.2f}",
                 f"{pc(a5['left']['precision'])}/{pc(a5['right']['precision'])}",
                 f"{pc(a5['left'].get('recall_visible'))}/{pc(a5['right'].get('recall_visible'))}",
                 f"{pc(a5['left']['opposite_accuracy'])}/{pc(a5['right']['opposite_accuracy'])}",
                 f"{pc(a5['left']['crossable_accuracy'])}/{pc(a5['right']['crossable_accuracy'])}",
                 f"{b['ground_truth']}/{b['target_moves']}/{b['on_time']}/{b.get('spurious_moves', '—')}",
                 f"{pc(m['B5_return_shown']['far'])}/{pc(m['B5_return_shown']['centre'])}",
                 f"{m['B3']['over_line_frames']}/{pc(m['B3']['return_shown'])}/{m['B3']['opposite_target_frames']}",
                 f"{pc(c['C1_light']['confirmed'])}/{pc(c['C1_light']['wrong_colour'])}" if c.get('C1_light', {}).get('frames') else '—',
                 pc(c['C2_no_light_shown']['shown']) if 'C2_no_light_shown' in c else '—',
                 f"{c['C3_speed_signs']['signs']}/{pc(c['C3_speed_signs']['confirmed_before'])}/{c['C3_speed_signs']['wrong_frames']}" if 'C3_speed_signs' in c else '—',
                 (lambda q: '—' if not q or not q.get('n') else f"{cm(q['error_m']['median'])}/{cm(q['error_m']['p95'])}/{pc(q['detected'])}/{pc(q['false'])}")(c.get('C4_stop_line')),
                 f"{c['C5_advice']['over_line_frames']}/{pc(c['C5_advice']['hold'])}" if 'C5_advice' in c else '—'])
head = ['장면', 'A1 중앙 cm', 'A1 p95 cm', 'A1 이름 %', 'A2 좌/우 %', 'A3 모양 좌/우 %', 'A3 색 좌/우 %', 'A4 중앙 cm', 'A4 p95 cm', 'A4 방향 p95 °',
        'A5 정밀 좌/우 %', 'A5 재현(보임) 좌/우 %', 'A5 반대 좌/우 %', 'A5 넘음 좌/우 %', 'B 변경 정답/이동/정시/엉뚱', 'B5 멀때/중앙 %', 'B3 넘음/복귀%/반대목표',
        'C1 확정/색오류 %', 'C2 오표시 %', 'C3 표지/미리%/오류프레임', 'C4 중앙/p95 cm/검출/오검출 %', 'C5 선넘음/정지유지 %']
goal = ['목표', '≤5', '≤15', '', '≥95', '≥95', '≥95', '≤5', '≤12', '≤1.5', '≥90', '≥90', '≥95', '≥95', '엉뚱 0 / 정시 ≥90 %', '≥95 / ≤2', '반대목표 0, 복귀 ≥95',
        '≥80 / ≤1', '≤1', '오류 0, 미리 ≥80', '≤50 / ≤100 / ≥80 / ≤1', '0 / 유지']
print('| '+' | '.join(head)+' |');print('|'+'---|'*len(head));print('| '+' | '.join(goal)+' |')
for r in rows:print('| '+' | '.join(r)+' |')
