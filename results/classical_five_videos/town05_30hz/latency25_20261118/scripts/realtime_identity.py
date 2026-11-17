#!/usr/bin/env python3
"""Plan 25: the published lines of two real-time runs, frame by frame (frames.json of tools/realtime_eval.py:
the boundary and display line strips as received, points rounded to 0.1 mm).  Both runs must have processed
every frame.  usage: realtime_identity.py RUN_A RUN_B"""
import sys,json
from pathlib import Path
A,B=(json.load(open(Path(p)/'frames.json')) for p in sys.argv[1:3])
ka={round(r['t'],4):r for r in A};kb={round(r['t'],4):r for r in B};common=sorted(set(ka)&set(kb))
for field in ('boundary_lines','display_lines'):
    same=sum(ka[t].get(field)==kb[t].get(field) for t in common)
    diff=[t for t in common if ka[t].get(field)!=kb[t].get(field)]
    print(f'{field}: identical {same}/{len(common)} frames'+(f' (first differing t={diff[:5]})' if diff else ''))
