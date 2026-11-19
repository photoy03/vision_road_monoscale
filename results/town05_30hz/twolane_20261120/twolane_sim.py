#!/usr/bin/env python3
"""Plan 33 (feasibility, before any code change): from saved offline predictions (gt_eval predict: display lines,
outer lines, guidance state per frame) simulate "two lanes while changing lanes" and score it against the GT.

Rule simulated: active while guidance state is 'change' and HOLD s after; the third line is the outer line on the side
of the own display line nearer to the car (before the crossing the line being crossed is ahead on the change side, after
it the lane just left is on the other side).  Optional fallback: the own line moved by the lane width when the outer line
is missing (--shift).  Early trigger candidates: --early (state 'return' with the car moving towards a crossable line is
not available here; the offset and its change are: offset beyond EARLY_M from the target centre and growing).

Scores per scene: in GT change windows (+-2 s around the crossing) the share of frames with three lines and the share of
third lines within 30 cm of the GT neighbour's outer boundary at 10 m; outside the windows the share of frames with a
third line (goal <= 1 %); onset of each change relative to the crossing.
usage: twolane_sim.py PRED_DIR [--hold 1.0] [--shift] [--early M]"""
import argparse, gzip, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'tools'))
import gt_eval as G

SCENES = ['town05_30hz', 's1_lanefollow', 's2_wobble', 's3a_zigzag_cross', 's3b_zigzag_inlane', 's4_straddle', 's5_lanechange', 's6_yellow', 's7_lights',
          's8a_lanefollow_20kmh', 's8b_lanefollow_50kmh', 's8c_lanechange_20kmh', 's8d_lanechange_50kmh']


def yat(p, x):
    p = np.asarray(p, float)
    return None if len(p) < 2 or not (p[0, 0] <= x <= p[-1, 0]) else float(np.interp(x, p[:, 0], p[:, 1]))


def near_y(p):
    p = np.asarray(p, float)
    return float(np.interp(max(p[0, 0], 1.4), p[:, 0], p[:, 1])) if len(p) >= 2 else None


def expected(r, side, x=10.):
    """GT y at x of the outer boundary of the neighbour on `side` of the lane under base_link x = 3 m, or None."""
    own = G.lane_at(r, G.DECIDE_X)
    if own is None:return None
    ys = sorted({round(v, 3) for l in r['lanes'] if l.get('lane_type', 'Driving') == 'Driving' for k in ('left_boundary_vehicle', 'right_boundary_vehicle')
                 for v in [G.y_at(G.to_base(l[k]), x)] if v is not None})
    mine = G.y_at(G.to_base(own[side+'_boundary_vehicle']), x)
    if mine is None:return None
    beyond = [v for v in ys if (v-mine > 1.5 if side == 'left' else mine-v > 1.5)]
    return (min(beyond) if side == 'left' else max(beyond)) if beyond else None


GU = dict(offset_at_m=1.4096, change_without_neighbour=True, trust_single_line=True, on_m=.25, off_m=.20, keep_target_far=True)


def intents(scene, rows, move, toward):
    """Re-run the guidance on the saved centres (as gt_eval reguide) and return per frame whether the car has been moving
    towards a crossable (white dashed) line of the target lane for at least `move` s and is `toward` m or more off the
    target centre towards it (the guidance's own change detector, without its 'close to the line' condition)."""
    sys.path.insert(0, str(G.ROOT/'src/hero_coach'));sys.path.insert(0, str(G.ROOT/'src/vision_road'))
    from hero_coach.lane_guidance import LaneGuidance, Lane, GuidanceSettings, crossable
    from vision_road.core import Timeline
    p = G.paths(scene);meta = json.loads((p['offline']/'frames.json').read_text())
    poses = np.array(meta['poses']);poses[:, 3] = np.unwrap(poses[:, 3]);gs = GuidanceSettings()
    for k, v in GU.items():setattr(gs, k, v)
    guide = LaneGuidance(gs);tl = Timeline(.25);k = 0;gt = {round(r['stamp'], 4): r for r in G.ground_truth(scene)};out = []
    for d in rows:
        t = d['stamp']
        while k < len(poses) and poses[k][0] <= t+.05:tl.add(*poses[k]);k += 1
        pose = tl.at(t);g = gt.get(round(t, 4));speed = g['speed_mps'] if g else None
        gpose = None if pose is None else tuple(float(np.interp(t, poses[:, 0], poses[:, j])) for j in (1, 2, 3))
        o = guide.update(t, gpose, [Lane.from_message(c) for c in d['centres']], path=False, speed=speed)
        hit = False
        if guide.target is not None and o.target not in ('', 'carried') and o.state != 'none':
            left, right = guide.target[3], guide.target[4]
            for side, sign, b in (('left', 1., left), ('right', -1., right)):
                if guide.moving[side] >= move and crossable(b) and sign*o.offset_m >= toward:hit = True
        out.append(hit or o.state == 'change')
    return out


def run(d, scene, hold, shift, early, move=None, toward=0.):
    rows = [json.loads(l) for l in gzip.open(d/scene/'predictions.jsonl.gz', 'rt')];rows.sort(key=lambda r: r['stamp'])
    intent = intents(scene, rows, move, toward) if move is not None else None
    gt = {round(r['stamp'], 4): r for r in G.ground_truth(scene)}
    changes = [t for t, _, _ in G.lane_changes(list(gt.values()))] if gt else []
    last = -1e9;prev_off = None;res = dict(win=0, win3=0, win_ok=0, win_eval=0, out=0, out3=0, onsets=[], shift_used=0)
    first = {c: None for c in changes}
    for i, r in enumerate(rows):
        t = r['stamp'];g = r['guidance'];disp = {x['side']: x['points'] for x in r['display'] if len(x['points']) >= 2}
        on = g['state'] == 'change' if intent is None else intent[i]
        if early is not None and prev_off is not None and g['state'] in ('return', 'change') and abs(g['offset_m']) >= early and abs(g['offset_m']) > abs(prev_off):on = True
        prev_off = g['offset_m']
        if on:last = t
        active = t-last <= hold and len(disp) == 2
        third = None;side = None
        if active:
            side = min(disp, key=lambda s: abs(near_y(disp[s]) or 9.))
            o = next((x for x in r['outer'] if x['side'] == side+'_outer' and len(x['points']) >= 2), None)
            if o is not None:third = o['points']
            elif shift:
                w = (near_y(disp['left']) or 0)-(near_y(disp['right']) or 0)
                if 2.5 <= w <= 4.5:third = [[x, y+(w if side == 'left' else -w)] for x, y in disp[side]];res['shift_used'] += 1
        inwin = any(abs(t-c) <= 2. for c in changes)
        if inwin:
            res['win'] += 1;res['win3'] += third is not None
            gr = gt.get(round(t, 4))
            if third is not None and gr is not None:
                e = expected(gr, side);y = yat(third, 10.)
                if e is not None and y is not None:res['win_eval'] += 1;res['win_ok'] += abs(y-e) <= .3
        else:
            res['out'] += 1;res['out3'] += third is not None
        for c in changes:
            if first[c] is None and third is not None and c-3. <= t <= c+2.:first[c] = round(t-c, 2)
    res['onsets'] = [first[c] for c in changes]
    return res


def main():
    ap = argparse.ArgumentParser();ap.add_argument('dir', type=Path);ap.add_argument('--hold', type=float, default=1.);ap.add_argument('--shift', action='store_true')
    ap.add_argument('--early', type=float);ap.add_argument('--move', type=float);ap.add_argument('--toward', type=float, default=0.);a = ap.parse_args()
    print('| 영상 | 변경 구간 3선 % | 그중 30 cm 안 % | 구간 밖 3선 % | 켜진 시각(넘는 시각 대비 s) |')
    print('|---|---|---|---|---|')
    for s in SCENES:
        r = run(a.dir, s, a.hold, a.shift, a.early, a.move, a.toward)
        f = lambda n, m: f'{100*n/m:.1f}' if m else '—'
        print(f"| {s} | {f(r['win3'], r['win'])} | {f(r['win_ok'], r['win_eval'])} ({r['win_eval']}) | {f(r['out3'], r['out'])} | {r['onsets']} |")


if __name__ == '__main__':
    main()
