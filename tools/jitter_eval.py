#!/usr/bin/env python3
"""Display line jitter (plan 32): how much the HUD's lane lines shake from frame to frame.

Input: an offline predictions.jsonl.gz of tools/gt_eval.py (field 'display') or a realtime frames.json of
tools/realtime_eval.py (field 'display_lines').  For the left and right display line, per frame, y at x = 5, 10, 18 m
(base_link, linear between the line's points, only inside the line); the jitter at a frame is the second difference
|y[t+1] - 2 y[t] + y[t-1]| over three consecutive frames (a smooth drift of the car is near 0).  A second difference
over 1 m is a line changing its name (a lane change), not jitter, and is left out.  End curl: the angle between the
direction of the line's last 5 m and that of the 5 m before it (a far end bending away).

Lane changes (GT, plan 30 gt_eval.lane_changes, +-2 s) are counted apart as well (scene names of the GT recordings).
Goal (plan 32): at 18 m p99 <= 2 cm and at most 5 frames over 5 cm per recording; at 10 m p99 <= 1 cm.

usage: jitter_eval.py FILE [--scene NAME] [--json OUT]
"""
import argparse, gzip, json, math, sys
from pathlib import Path
import numpy as np

XS = (5., 10., 18.)


def frames_of(path):
    """[(stamp, {side: Nx2 points})] from either file kind, in time order."""
    path = Path(path);out = []
    if path.name.endswith('.jsonl.gz'):
        for line in gzip.open(path, 'rt'):
            d = json.loads(line)
            out.append((d['stamp'], {l['side']: np.asarray(l['points'], float) for l in d.get('display', []) if len(l['points']) >= 2}))
    else:
        for d in json.load(open(path)):
            lines = {}
            for side, _, pts in d.get('display_lines') or []:
                if len(pts) >= 2:lines[side] = np.asarray(pts, float)
            out.append((d['t'], lines))
    out.sort(key=lambda f: f[0])
    return out


def y_at(p, x):
    if p is None or not (p[0, 0] <= x <= p[-1, 0]):return None
    return float(np.interp(x, p[:, 0], p[:, 1]))


def curl_deg(p):
    """Angle (deg) between the directions of the last 5 m and the 5 m before, or None (line under 10 m)."""
    if p is None or p[-1, 0]-p[0, 0] < 10.:return None
    x2 = p[-1, 0];x1 = x2-5.;x0 = x2-10.
    y0, y1, y2 = (float(np.interp(x, p[:, 0], p[:, 1])) for x in (x0, x1, x2))
    return abs(math.degrees(math.atan2(y2-y1, 5.)-math.atan2(y1-y0, 5.)))


def changes_of(scene):
    if not scene:return []
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import gt_eval as G
    try:return [t for t, _, _ in G.lane_changes(G.ground_truth(scene))]
    except Exception:return []


def evaluate(path, scene=None):
    frames = frames_of(path);changes = changes_of(scene)
    near_change = lambda t: any(abs(t-c) <= 2. for c in changes)
    res = {};over = {}
    for x in XS:
        d2 = [];d2c = [];times = []
        for side in ('left', 'right'):
            ys = [y_at(f[1].get(side), x) for f in frames]
            for i in range(1, len(frames)-1):
                a, b, c = ys[i-1], ys[i], ys[i+1]
                if a is None or b is None or c is None:continue
                v = abs(c-2*b+a)
                if v > 1.:continue
                (d2c if near_change(frames[i][0]) else d2).append(v)
                if v > .05:times.append(round(frames[i][0], 1))
        allv = np.array(d2+d2c)
        res[f'{int(x)}m'] = dict(n=len(allv), p95=round(float(np.percentile(allv, 95)), 4) if len(allv) else None,
                                 p99=round(float(np.percentile(allv, 99)), 4) if len(allv) else None, over_5cm=int(np.sum(allv > .05)),
                                 p99_outside_changes=round(float(np.percentile(d2, 99)), 4) if d2 else None,
                                 p99_in_changes=round(float(np.percentile(d2c, 99)), 4) if d2c else None, n_in_changes=len(d2c))
        over[f'{int(x)}m'] = sorted(set(times))
    curls = [c for f in frames for p in f[1].values() for c in [curl_deg(p)] if c is not None]
    res['curl_deg'] = dict(n=len(curls), p95=round(float(np.percentile(curls, 95)), 2) if curls else None,
                           p99=round(float(np.percentile(curls, 99)), 2) if curls else None)
    res['over_5cm_times_18m'] = over['18m'];res['frames'] = len(frames)
    res['goal_met'] = bool(res['18m']['p99'] is not None and res['18m']['p99'] <= .02 and res['18m']['over_5cm'] <= 5
                           and res['10m']['p99'] is not None and res['10m']['p99'] <= .01)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('file');ap.add_argument('--scene');ap.add_argument('--json', type=Path)
    a = ap.parse_args();res = evaluate(a.file, a.scene)
    if a.json:a.json.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ('5m', '10m', '18m', 'curl_deg', 'goal_met')}))


if __name__ == '__main__':
    main()
