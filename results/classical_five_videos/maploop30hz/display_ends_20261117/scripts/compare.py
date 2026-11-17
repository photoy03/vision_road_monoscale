#!/usr/bin/env python3
"""Compact table of ends_eval.py results.  usage: compare.py VIDEO NAME ..."""
import sys,json
from pathlib import Path
M=Path(__file__).resolve().parent.parent/'metrics'
video=sys.argv[1]
KEYS=dict(maploop=['straight 5-20/right','curve 22-35/right','curve 22-35/left','island 48-53/right'],
          town05=['straight 0-9/right','curve 15-30/right','curve 15-30/left','straight2 40-45/right'])
GAP=dict(maploop=['curve 22-35/left','curve 22-35/right'],town05=['curve 15-30/left','curve 15-30/right'])
for name in sys.argv[2:]:
    d=json.load(open(M/f'{name}_{video}.json'))
    for side in ('left','right'):
        e=d['sides'][side];m=d['display_filter_eval'][side]
        print(f"{name:14s} {side[0]} y0>5 {e['y0']['over']:3d} far>20 {e['far']['over']:2d} y10>10 {e['y10']['over']} "
              f"| jit10 {m['jitter_cm']['filtered']['std']} shake {m['shake_cm']['filtered']} tog {m['toggles']['filtered']} shown {m['shown_frames']['filtered']} "
              f"lag {m['curve_lag_m']['mean_abs']} mism {m['observed_mismatch_frames']} from0 {e['from_0m_share']}")
    print(' '*15+'gap '+' '.join(f"{k.split('/')[1][0]}:{d['paint_gap_p95_cm_6_8_10_15_20'][k]}" for k in GAP[video])+
          ' | kink ev/oth '+' '.join(f"{k.split()[0][:5]}{k.split('/')[1][0]} {d['near_head_mrad'][k]['event_rms']}/{d['near_head_mrad'][k]['other_rms']}" for k in KEYS[video])+' | near0-5 '+' '.join(f"{k.split()[0][:5]}{k.split('/')[1][0]} {d['near_gap_p95_cm_0_5m'].get(k)}" for k in KEYS[video]))
