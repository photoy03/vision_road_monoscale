"""Plan 32: jitter table of a try directory (and the guidance/GT metrics are taken with gt_eval separately)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'tools'))
import jitter_eval as J
D = Path(sys.argv[1])
S = ['town05_30hz', 'maploop_30hz', 's1_lanefollow', 's2_wobble', 's3a_zigzag_cross', 's3b_zigzag_inlane', 's4_straddle', 's5_lanechange', 's6_yellow',
     's7_lights', 's8a_lanefollow_20kmh', 's8b_lanefollow_50kmh', 's8c_lanechange_20kmh', 's8d_lanechange_50kmh']
print('| 영상 | 5 m p99 | 10 m p99 | 18 m p95 | 18 m p99 | 18 m >5 cm | 18 m p99 변경 밖 / 안 | 끝 말림 p99 ° | 목표 |')
print('|---|---|---|---|---|---|---|---|---|')
cm = lambda v: '—' if v is None else f'{100*v:.1f}'
for s in S:
    f = D/s/'predictions.jsonl.gz'
    if not f.exists():continue
    r = J.evaluate(f, None if s in ('town05_30hz', 'maploop_30hz') else s);(D/s/'jitter.json').write_text(json.dumps(r, indent=1))
    a = r['18m']
    print(f"| {s.split('_')[0] if s.startswith('s') else s} | {cm(r['5m']['p99'])} | {cm(r['10m']['p99'])} | {cm(a['p95'])} | {cm(a['p99'])} | {a['over_5cm']} | "
          f"{cm(a['p99_outside_changes'])} / {cm(a['p99_in_changes'])} | {r['curl_deg']['p99']} | {'충족' if r['goal_met'] else '미달'} |")
