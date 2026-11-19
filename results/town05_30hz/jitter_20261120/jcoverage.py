#!/usr/bin/env python3
"""Plan 32: how often the HUD display line shows the GT own-lane line, at 15 m: shown % / within 30 cm % / off by more
than 30 cm %, of the GT frames with a painted own-lane line there (A1 exclusions: junctions, GT boundary off the paint).
A smoothing or trimming that starves the line shows as fewer frames within 30 cm.  usage: jcoverage.py DIR [DIR ...]"""
import sys, gzip, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'tools'))
import gt_eval as G
PS = Path(__file__).resolve().parents[1]/'gt_loop2_20261120'/'round_03'
SCENES = ['s1_lanefollow', 's2_wobble', 's3a_zigzag_cross', 's3b_zigzag_inlane', 's4_straddle', 's5_lanechange', 's6_yellow', 's7_lights',
          's8a_lanefollow_20kmh', 's8b_lanefollow_50kmh', 's8c_lanechange_20kmh', 's8d_lanechange_50kmh']


def stats(d, scene, x=15.):
    pred = {round(r['stamp'], 4): r for r in (json.loads(l) for l in gzip.open(d/scene/'predictions.jsonl.gz', 'rt'))}
    gtp = {round(r['stamp'], 4): r.get('gt_boundary_to_paint_at') or {} for r in (json.loads(l) for l in gzip.open(PS/scene/'paint_stops.jsonl.gz', 'rt'))}
    n = shown = good = 0
    for r in G.ground_truth(scene):
        st = round(r['stamp'], 4);p = pred.get(st);ego = next((l for l in r['lanes'] if l['relation'] == 'ego'), None)
        if p is None or ego is None or r['is_junction'] or ego['is_junction']:continue
        disp = {l['side']: l['points'] for l in p.get('display', []) if len(l['points']) >= 2}
        for side in ('left', 'right'):
            if (ego[side+'_marking'] or {}).get('type', 'None') == 'None':continue
            at = gtp.get(st, {}).get(side, {}).get(str(int(x)))
            if at is not None and abs(at) > G.GT_PAINT_M:continue
            yg = G.y_at(G.to_base(ego[side+'_boundary_vehicle']), x)
            if yg is None:continue
            n += 1;yd = G.y_at(disp[side], x) if side in disp else None
            if yd is None:continue
            shown += 1;good += abs(yd-yg) <= .3
    return 100*shown/n, 100*good/n, 100*(shown-good)/n


dirs = [Path(a) for a in sys.argv[1:]]
print('| 영상 | '+' | '.join(f'{d.name} 보임 / 30 cm 안 / 벗어남 %' for d in dirs)+' |')
print('|---|'+'---|'*len(dirs))
for s in SCENES:
    print(f'| {s} | '+' | '.join('%.1f / %.1f / %.1f' % stats(d, s) for d in dirs)+' |')
