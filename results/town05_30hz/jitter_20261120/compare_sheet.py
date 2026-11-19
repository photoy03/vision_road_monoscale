#!/usr/bin/env python3
"""Plan 32: before / after sheet of one interval of a recording.  Top: the display line's y at 18 m (left, right) and its
second difference over the interval, from the realtime frames.json of both recordings.  Below: road_and_hud.mp4 frames
at a few times of the interval, before (upper row) and after (lower row), full frames.
usage: compare_sheet.py BEFORE_DIR AFTER_DIR T0 T1 OUT.png [TITLE]   (times in s from the recording's first frame)"""
import sys, json, subprocess, tempfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'tools'))
import jitter_eval as JE

before, after, t0, t1, out = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), Path(sys.argv[5])
title = sys.argv[6] if len(sys.argv) > 6 else ''
times = np.linspace(t0, t1, 6)


def series(d):
    fr = JE.frames_of(d/'frames.json');s0 = fr[0][0];res = {}
    for side in ('left', 'right'):
        t = np.array([f[0]-s0 for f in fr]);y = np.array([np.nan if (v := JE.y_at(f[1].get(side), 18.)) is None else v for f in fr])
        res[side] = (t, y)
    return res


def frame(d, t, tmp):
    p = Path(tmp)/f'{d.parent.name}_{d.name}_{t:.2f}.png'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{t:.2f}', '-i', str(d/'road_and_hud.mp4'), '-frames:v', '1', str(p)], check=True)
    return plt.imread(p)


fig = plt.figure(figsize=(24, 15));gs = fig.add_gridspec(4, 6, height_ratios=[1.2, .8, 1, 1])
ax = fig.add_subplot(gs[0, :]);ax2 = fig.add_subplot(gs[1, :], sharex=ax)
for d, name, ls in ((before, 'before', '--'), (after, 'after', '-')):
    for side, c in (('left', 'tab:blue'), ('right', 'tab:red')):
        t, y = series(d)[side];m = (t >= t0-.5) & (t <= t1+.5)
        ax.plot(t[m], y[m], ls, color=c, label=f'{name} {side}')
        d2 = np.abs(y[2:]-2*y[1:-1]+y[:-2]);tm = t[1:-1];m2 = (tm >= t0-.5) & (tm <= t1+.5)
        ax2.plot(tm[m2], 100*d2[m2], ls, color=c, label=f'{name} {side}')
ax.set_ylabel('y at 18 m (m)');ax.legend(ncol=4);ax.grid(alpha=.3);ax.set_title(title)
ax2.axhline(5, color='k', lw=.8);ax2.set_ylabel('|2nd difference| at 18 m (cm)');ax2.set_xlabel('time (s)');ax2.grid(alpha=.3)
ax2.set_ylim(0, 30)
for t in times:ax.axvline(t, color='gray', lw=.6)
with tempfile.TemporaryDirectory() as tmp:
    for row, (d, name) in enumerate(((before, 'before'), (after, 'after'))):
        for i, t in enumerate(times):
            a = fig.add_subplot(gs[2+row, i]);a.imshow(frame(d, t, tmp));a.set_axis_off();a.set_title(f'{name} {t:.1f} s', fontsize=10)
fig.tight_layout();fig.savefig(out, dpi=80)
print(out)
