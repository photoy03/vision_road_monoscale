"""Plan 31: a try against the base (plan 30 round_03) per scene and item; '!' marks worse by >10 % or newly off target.
usage: cmp.py TRY_DIR [BASE_DIR]"""
import json, sys
from pathlib import Path
T = Path(sys.argv[1]);B = Path(sys.argv[2] if len(sys.argv) > 2 else 'results/town05_30hz/gt_loop_20261119/round_03')
S = ['s1_lanefollow', 's2_wobble', 's3a_zigzag_cross', 's3b_zigzag_inlane', 's4_straddle', 's5_lanechange', 's6_yellow', 's7_lights',
     's8a_lanefollow_20kmh', 's8b_lanefollow_50kmh', 's8c_lanechange_20kmh', 's8d_lanechange_50kmh']
def g(m, *k):
    for x in k:m = m.get(x) if isinstance(m, dict) else None
    return m
ITEMS = [('A1p95', lambda m: g(m, 'A1_all', 'p95'), 'max', .15), ('A1med', lambda m: g(m, 'A1_all', 'median'), 'max', .05),
         ('A2min', lambda m: min(m['A2_observed'].values()), 'min', .95),
         ('A3min', lambda m: min(v['accuracy'] for v in m['A3_style'].values() if v['accuracy'] is not None), 'min', .95),
         ('A4p95', lambda m: g(m, 'A4_offset_m', 'p95'), 'max', .12), ('A4hd', lambda m: g(m, 'A4_heading_deg', 'p95'), 'max', 1.5),
         ('A5prec', lambda m: min(v['precision'] for v in m['A5'].values()), 'min', .9),
         ('A5rec', lambda m: min([v['recall_visible'] for v in m['A5'].values() if v.get('recall_visible') is not None] or [1.]), 'min', .9),
         ('B2', lambda m: (m['B_lane_changes']['on_time']/m['B_lane_changes']['ground_truth']) if m['B_lane_changes']['ground_truth'] else None, 'min', .9),
         ('spur', lambda m: g(m, 'B_lane_changes', 'spurious_moves'), 'max', 0),
         ('B5far', lambda m: g(m, 'B5_return_shown', 'far'), 'min', .95), ('B5cen', lambda m: g(m, 'B5_return_shown', 'centre'), 'max', .02),
         ('B3', lambda m: g(m, 'B3', 'return_shown'), 'min', .95),
         ('C4det', lambda m: g(m, 'C', 'C4_stop_line', 'detected'), 'min', .8), ('C4fal', lambda m: g(m, 'C', 'C4_stop_line', 'false'), 'max', .01)]
bad = 0
for s in S:
    try:t = json.load(open(T/s/'metrics_1920.json'));b = json.load(open(B/s/'metrics_1920.json'))
    except Exception as e:print(s, 'missing', e);continue
    cells = []
    for name, get, worse, goal in ITEMS:
        try:vb, vt = get(b), get(t)
        except Exception:vb = vt = None
        if vb is None or vt is None:continue
        ok = lambda v: v <= goal if worse == 'max' else v >= goal
        flag = ''
        if ok(vb) and not ok(vt):flag = '!'
        elif worse == 'max' and vt > vb*1.1+1e-9 and vt-vb > (1 if name == 'spur' else .002):flag = '!'
        elif worse == 'min' and vt < vb*.9-1e-9:flag = '!'
        if vt != vb or flag:cells.append(f"{name} {vb:.3g}->{vt:.3g}{flag}")
        bad += bool(flag)
    print(s.split('_')[0], ' | '.join(cells))
print('flags', bad)
