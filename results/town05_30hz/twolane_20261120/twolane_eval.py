#!/usr/bin/env python3
"""Plan 33: score the third HUD line (predictions 'third' of gt_eval predict with two_lane_hold_s) against the GT.
Per scene: in the GT change windows (+-2 s around each crossing) the share of frames with three lines, and of the third
lines the share within 30 cm of the GT neighbour's outer boundary at 10 m (the lane beyond the own lane's line on that
side, by position); outside the windows the share of frames with a third line (goal <= 1 %), and the same beyond 4 s
of any crossing (a third line there is not part of a change at all); the onset of each change against its crossing.
usage: twolane_eval.py PRED_DIR [PRED_DIR ...]"""
import gzip, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.argv, args = sys.argv[:1], sys.argv[1:]
from twolane_sim import SCENES, expected, yat, G  # noqa: E402


def beyond(g, row, side, x=10.):
    """GT y at x of the next GT lane line beyond the drawn own line of `side` (the third line must be that line: the
    two drawn lanes are then real adjacent lanes, also while the own lines keep their names through a crossing)."""
    own = next((d for d in row['display'] if d['side'] == side), None);yo = None if own is None else yat(own['points'], x)
    if yo is None:return None
    ys = [v for l in g['lanes'] if l.get('lane_type', 'Driving') == 'Driving' for k in ('left_boundary_vehicle', 'right_boundary_vehicle')
          for v in [G.y_at(G.to_base(l[k]), x)] if v is not None]
    cand = [v for v in ys if (v-yo > 1.5 if side == 'left' else yo-v > 1.5)]
    return (min(cand) if side == 'left' else max(cand)) if cand else None


def score(d, scene):
    rows = [json.loads(l) for l in gzip.open(Path(d)/scene/'predictions.jsonl.gz', 'rt')];rows.sort(key=lambda r: r['stamp'])
    gt = {round(r['stamp'], 4): r for r in G.ground_truth(scene)}
    changes = [t for t, _, _ in G.lane_changes(list(gt.values()))] if gt else []
    r = dict(win=0, win3=0, ok=0, ev=0, out=0, out3=0, far=0, far3=0);first = {c: None for c in changes}
    for row in rows:
        t = row['stamp'];third = (row.get('third') or [None])[0];dist = min([abs(t-c) for c in changes], default=1e9)
        if dist <= 2.:
            r['win'] += 1;r['win3'] += third is not None;g = gt.get(round(t, 4))
            if third is not None and g is not None:
                e = beyond(g, row, third['side'].split('_')[0]);y = yat(third['points'], 10.)
                if e is not None and y is not None:r['ev'] += 1;r['ok'] += abs(y-e) <= .3
        else:
            r['out'] += 1;r['out3'] += third is not None
            if dist > 4.:r['far'] += 1;r['far3'] += third is not None
        for c in changes:
            if first[c] is None and third is not None and c-3. <= t <= c+2.:first[c] = round(t-c, 2)
    return r, [first[c] for c in changes]


f = lambda n, m: f'{100*n/m:.1f}' if m else '—'
for d in args:
    print(f'== {d}')
    print('| 영상 | 변경 구간 3선 % | 그중 30 cm 안 % (프레임) | 구간 밖 3선 % | 4 초 밖 3선 % | 켜진 시각(넘는 시각 대비 s) |')
    print('|---|---|---|---|---|---|')
    for s in SCENES+['maploop_30hz']:
        try:r, on = score(d, s)
        except FileNotFoundError:continue
        print(f"| {s} | {f(r['win3'], r['win'])} | {f(r['ok'], r['ev'])} ({r['ev']}) | {f(r['out3'], r['out'])} | {f(r['far3'], r['far'])} | {on} |")
