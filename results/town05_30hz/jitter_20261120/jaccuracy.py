#!/usr/bin/env python3
"""Plan 32: the HUD display line against the GT own-lane boundary (like A1, but for the 'display' lines), so a smoothing
that lags is seen.  Same exclusions as A1: junctions, the car centre on a line, per distance a GT boundary off the paint
(paint_stops of gt_loop2 round_03).  usage: jaccuracy.py DIR [DIR ...]  ->  per scene p50 / p95 cm at 5, 10, 15 m."""
import sys, gzip, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'tools'))
import gt_eval as G
PS = Path(__file__).resolve().parents[1]/'gt_loop2_20261120'/'round_03'
SCENES = ['s1_lanefollow', 's2_wobble', 's3a_zigzag_cross', 's3b_zigzag_inlane', 's4_straddle', 's5_lanechange', 's6_yellow', 's7_lights',
          's8a_lanefollow_20kmh', 's8b_lanefollow_50kmh', 's8c_lanechange_20kmh', 's8d_lanechange_50kmh']


def errors(d, scene):
    pred = {round(r['stamp'], 4): r for r in (json.loads(l) for l in gzip.open(d/scene/'predictions.jsonl.gz', 'rt'))}
    gtp = {round(r['stamp'], 4): r.get('gt_boundary_to_paint_at') or {} for r in (json.loads(l) for l in gzip.open(PS/scene/'paint_stops.jsonl.gz', 'rt'))}
    e = {x: [] for x in G.XS}
    for r in G.ground_truth(scene):
        st = round(r['stamp'], 4);p = pred.get(st)
        ego = next((l for l in r['lanes'] if l['relation'] == 'ego'), None)
        if p is None or ego is None or r['is_junction'] or ego['is_junction']:continue
        wl, wr = G.y_at(G.to_base(ego['left_boundary_vehicle']), G.BASE_FROM_ORIGIN), G.y_at(G.to_base(ego['right_boundary_vehicle']), G.BASE_FROM_ORIGIN)
        if wl is not None and wr is not None and abs(r['lateral_offset_m']) >= .5*(wl-wr)-G.ON_LINE_M:continue
        disp = {l['side']: l['points'] for l in p.get('display', []) if len(l['points']) >= 2}
        for side in ('left', 'right'):
            if (ego[side+'_marking'] or {}).get('type', 'None') == 'None' or side not in disp:continue
            gl = G.to_base(ego[side+'_boundary_vehicle']);at = gtp.get(st, {}).get(side, {})
            for x in G.XS:
                if at.get(str(int(x))) is not None and abs(at[str(int(x))]) > G.GT_PAINT_M:continue
                yg, yd = G.y_at(gl, x), G.y_at(disp[side], x)
                if yg is not None and yd is not None:e[x].append(abs(yd-yg))
    return e


dirs = [Path(a) for a in sys.argv[1:]]
print('| 영상 | '+' | '.join(f'{d.name} {int(x)} m p50 / p95' for d in dirs for x in G.XS)+' |')
print('|---|'+'---|'*(len(dirs)*len(G.XS)))
for s in SCENES:
    row = []
    for d in dirs:
        e = errors(d, s)
        row += [f'{100*np.median(e[x]):.1f} / {100*np.percentile(e[x], 95):.1f}' if e[x] else '—' for x in G.XS]
    print(f'| {s} | '+' | '.join(row)+' |')
