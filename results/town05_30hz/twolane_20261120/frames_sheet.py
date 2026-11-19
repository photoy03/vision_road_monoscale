#!/usr/bin/env python3
"""Plan 33: before / after HUD frames at given times (full frames of road_and_hud.mp4; upper row before, lower after).
usage: frames_sheet.py BEFORE_DIR AFTER_DIR OUT.png TITLE T1 T2 ..."""
import sys, subprocess, tempfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
before, after, out, title = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4];times = [float(t) for t in sys.argv[5:]]
fig, ax = plt.subplots(len(times), 2, figsize=(26, 4.6*len(times)))
with tempfile.TemporaryDirectory() as tmp:
    for i, t in enumerate(times):
        for j, (d, name) in enumerate(((before, 'before'), (after, 'after'))):
            p = Path(tmp)/f'{j}_{i}.png'
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{t:.2f}', '-i', str(d/'road_and_hud.mp4'), '-frames:v', '1', str(p)], check=True)
            ax[i, j].imshow(plt.imread(p));ax[i, j].set_axis_off();ax[i, j].set_title(f'{name} {t:.1f} s', fontsize=13)
fig.suptitle(title, fontsize=16);fig.tight_layout();fig.savefig(out, dpi=55);print(out)
