"""Plan 30: the rounds side by side, per target item: how many scenes meet it and the worst scene.
usage: summary_rounds.py round_00:tag round_01:_1920 ...  (tag: the metrics file suffix, '' for metrics.json)"""
import json, sys
from pathlib import Path
G = Path(__file__).resolve().parents[1]
S = ['s1_lanefollow', 's2_wobble', 's3a_zigzag_cross', 's3b_zigzag_inlane', 's4_straddle', 's5_lanechange', 's6_yellow', 's7_lights',
     's8a_lanefollow_20kmh', 's8b_lanefollow_50kmh', 's8c_lanechange_20kmh', 's8d_lanechange_50kmh']
def load(r, tag):
    out = {}
    for s in S:
        f = G/r/s/f'metrics{tag}.json'
        if f.exists():out[s] = json.load(open(f))
    return out
g = lambda m, *k: (lambda v: v)(m if not k else (g(m.get(k[0]), *k[1:]) if isinstance(m, dict) else None))
B1 = ('s1', 's2', 's3b', 's8a', 's8b');B2 = ('s3a', 's5', 's8c', 's8d')
ITEMS = [  # name, getter(metrics) -> value or None, ok(value), worse = 'max' (bigger is worse) or 'min'
    ('A1 중앙 ≤5 cm', lambda m: g(m, 'A1_all', 'median'), lambda v: v <= .05, 'max', 100),
    ('A1 p95 ≤15 cm', lambda m: g(m, 'A1_all', 'p95'), lambda v: v <= .15, 'max', 100),
    ('A2 관측 ≥95 % (좌우 낮은 쪽)', lambda m: min(m['A2_observed'].values()), lambda v: v >= .95, 'min', 100),
    ('A3 모양 ≥95 % (낮은 쪽)', lambda m: min(v['accuracy'] for v in m['A3_style'].values() if v['accuracy'] is not None), lambda v: v >= .95, 'min', 100),
    ('A3 색 ≥95 %', lambda m: min(v['accuracy'] for v in m['A3_colour'].values() if v['accuracy'] is not None), lambda v: v >= .95, 'min', 100),
    ('A4 오프셋 중앙 ≤5 cm', lambda m: g(m, 'A4_offset_m', 'median'), lambda v: v <= .05, 'max', 100),
    ('A4 오프셋 p95 ≤12 cm', lambda m: g(m, 'A4_offset_m', 'p95'), lambda v: v <= .12, 'max', 100),
    ('A4 방향 p95 ≤1.5°', lambda m: g(m, 'A4_heading_deg', 'p95'), lambda v: v <= 1.5, 'max', 1),
    ('A5 재현(보임) ≥90 %', lambda m: min([v['recall_visible'] for v in m['A5'].values() if v.get('recall_visible') is not None] or [None]) if any(v.get('recall_visible') is not None for v in m['A5'].values()) else None, lambda v: v >= .9, 'min', 100),
    ('A5 넘음 ≥95 %', lambda m: min([v['crossable_accuracy'] for v in m['A5'].values() if v.get('crossable_accuracy') is not None] or [None]) if any(v.get('crossable_accuracy') is not None for v in m['A5'].values()) else None, lambda v: v >= .95, 'min', 100),
    ('B1 엉뚱한 이동 0 (s1·s2·s3b·s8a·s8b)', lambda m: g(m, 'B_lane_changes', 'spurious_moves') if m['scene'].split('_')[0] in B1 else None, lambda v: v == 0, 'max', 1),
    ('B2 정시 ≥90 % (s3a·s5·s8c·s8d)', lambda m: (m['B_lane_changes']['on_time']/m['B_lane_changes']['ground_truth']) if m['scene'].split('_')[0] in B2 and m['B_lane_changes']['ground_truth'] else None, lambda v: v >= .9, 'min', 100),
    ('엉뚱한 이동 (그 밖 장면, 참고)', lambda m: g(m, 'B_lane_changes', 'spurious_moves') if m['scene'].split('_')[0] not in B1 else None, lambda v: v == 0, 'max', 1),
    ('B5 멀 때 ≥95 %', lambda m: g(m, 'B5_return_shown', 'far') if (g(m, 'B5_return_shown', 'far_n') or 0) >= 150 else None, lambda v: v >= .95, 'min', 100),
    ('B5 중앙 ≤2 %', lambda m: g(m, 'B5_return_shown', 'centre'), lambda v: v <= .02, 'max', 100),
    ('B3 넘었을 때 복귀 ≥95 %', lambda m: g(m, 'B3', 'return_shown'), lambda v: v >= .95, 'min', 100),
    ('C1 신호 확정 ≥80 %', lambda m: g(m, 'C', 'C1_light', 'confirmed'), lambda v: v >= .8, 'min', 100),
    ('C1 색 오류 ≤1 %', lambda m: g(m, 'C', 'C1_light', 'wrong_colour'), lambda v: v <= .01, 'max', 100),
    ('C2 오표시 ≤1 %', lambda m: g(m, 'C', 'C2_no_light_shown', 'shown'), lambda v: v <= .01, 'max', 100),
    ('C3 표지 오류 프레임 0', lambda m: g(m, 'C', 'C3_speed_signs', 'wrong_frames'), lambda v: v == 0, 'max', 1),
    ('C3 미리 확정 ≥80 %', lambda m: g(m, 'C', 'C3_speed_signs', 'confirmed_before'), lambda v: v >= .8, 'min', 100),
    ('C4 정지선 검출 ≥80 %', lambda m: g(m, 'C', 'C4_stop_line', 'detected'), lambda v: v >= .8, 'min', 100),
    ('C4 정지선 p95 ≤100 cm', lambda m: g(m, 'C', 'C4_stop_line', 'error_m', 'p95'), lambda v: v <= 1., 'max', 100),
]
rounds = [(a.split(':')[0], a.split(':')[1] if ':' in a else '') for a in sys.argv[1:]]
data = {r: load(r, t) for r, t in rounds}
print('| 항목 | '+' | '.join(f'{r} 충족 / 최악(장면)' for r, _ in rounds)+' |');print('|---|'+'---|'*len(rounds))
for name, get, ok, worse, scale in ITEMS:
    cells = []
    for r, _ in rounds:
        vals = []
        for s, m in data[r].items():
            try:v = get(m)
            except Exception:v = None
            if v is not None:vals.append((v, s))
        if not vals:cells.append('—');continue
        n = sum(ok(v) for v, _ in vals);w = (max if worse == 'max' else min)(vals)
        fmt = (lambda v: f'{v*scale:.1f}') if scale == 100 else (lambda v: f'{v:.2f}' if isinstance(v, float) else str(v))
        cells.append(f'{n}/{len(vals)} · {fmt(w[0])} ({w[1].split("_")[0]})')
    print(f'| {name} | '+' | '.join(cells)+' |')
