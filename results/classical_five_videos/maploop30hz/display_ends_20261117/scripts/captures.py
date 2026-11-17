#!/usr/bin/env python3
"""Before/after HUD captures at given replay times: the HUD screen frame (10 Hz, file mtime) and the road overlay
(5 Hz) whose wall time is nearest the display arrival of that replay frame (monotonic -> wall by the current offset).
usage: captures.py BEFORE_RUN AFTER_RUN OUT.jpg T ..."""
import sys,json,time
from pathlib import Path
import numpy as np,cv2
off=time.time_ns()-time.monotonic_ns()
def load(run):
    run=Path(run);fr=json.load(open(run/'frames.json'))
    hud=sorted((run/'hud_frames').glob('frame_*.png'));hs=np.array([p.stat().st_mtime_ns for p in hud])
    ov=sorted((run/'overlay_frames').glob('*.jpg'));os_=np.array([p.stat().st_mtime_ns for p in ov])
    return fr,hud,hs,ov,os_
def pick(data,t):
    fr,hud,hs,ov,os_=data
    f=min(fr,key=lambda f:abs(f['t']-t));w=(f.get('display') or f['publish'])+off
    j=int(np.argmin(np.abs(hs-w)));i=int(np.argmin(np.abs(os_-w)))
    return cv2.imread(str(hud[j])),cv2.imread(str(ov[i])),(hs[j]-w)/1e6
B,A=load(sys.argv[1]),load(sys.argv[2]);rows=[]
for t in map(float,sys.argv[4:]):
    hb,ob,db=pick(B,t);ha,oa,da=pick(A,t)
    cells=[cv2.resize(oa,(560,315)),cv2.resize(hb,(504,315)),cv2.resize(ha,(504,315))]
    row=np.hstack(cells)
    for x,txt in ((8,f't={t:.2f}s camera (after run)'),(568,f'HUD before ({db:+.0f} ms)'),(1072,f'HUD after ({da:+.0f} ms)')):
        cv2.putText(row,txt,(x,28),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),4);cv2.putText(row,txt,(x,28),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
    rows.append(row)
cv2.imwrite(sys.argv[3],np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,88])
