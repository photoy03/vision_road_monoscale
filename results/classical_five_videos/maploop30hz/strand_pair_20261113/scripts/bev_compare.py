"""Curve BEV side by side: boundaries and paint strands of two offline runs over the cached BEVs.

usage: bev_compare.py CACHE OUT.mp4 FIRST LAST NAME=RUN_DIR NAME=RUN_DIR
Paint strands (paint topic) thin yellow, left/right boundary thick (green), 6 m and 10 m ticks.
"""
import sys,gzip,json,subprocess,cv2,numpy as np
from pathlib import Path
MPP=.07;FMAX=35.;LMAX=6.;NEAR=3.;FAR=20.;SCALE=3
cache=Path(sys.argv[1]);out=sys.argv[2];first,last=int(sys.argv[3]),int(sys.argv[4])
runs=[(s.split('=')[0],[json.loads(l) for l in gzip.open(Path(s.split('=')[1])/'predictions.jsonl.gz','rt')]) for s in sys.argv[5:]]
r0=round((FMAX-FAR)/MPP);r1=round((FMAX-NEAR)/MPP)
def px(points):
    p=np.asarray(points,float);p=p[(p[:,0]>=NEAR)&(p[:,0]<=FAR)]
    return np.column_stack([(LMAX-p[:,1])/MPP*SCALE,((FMAX-p[:,0])/MPP-r0)*SCALE]).round().astype(np.int32)
writer=None
for i in range(first,last):
    bev=cv2.imread(str(cache/'bev'/f'{i:05d}.png'))[r0:r1]
    tiles=[]
    for name,rows in runs:
        v=cv2.resize(bev,None,fx=SCALE,fy=SCALE,interpolation=cv2.INTER_NEAREST);v=(v*.7).astype(np.uint8)
        for x in (6.,10.):
            y=round(((FMAX-x)/MPP-r0)*SCALE);cv2.line(v,(0,y),(14,y),(255,255,255),1);cv2.putText(v,f'{x:.0f} m',(16,y+4),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
        r=rows[i]
        for l in r['lanes']:
            if l['reprojected']:continue
            q=px(l['points'])
            if len(q)>1:cv2.polylines(v,[q],False,(0,255,255),1)
        for b in r['boundaries']:
            if b['reprojected']:continue
            q=px(b['points'])
            if len(q)>1:cv2.polylines(v,[q],False,(0,220,0) if b['side']=='left' else (255,160,0),2)
        cv2.putText(v,f'{name}  frame {i}  t {i/30:.1f} s',(8,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        tiles.append(v)
    frame=np.hstack(tiles)
    if frame.shape[0]%2:frame=frame[:-1]
    if frame.shape[1]%2:frame=frame[:,:-1]
    if writer is None:
        writer=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{frame.shape[1]}x{frame.shape[0]}','-r','15','-i','pipe:0',
                                 '-c:v','libx264','-pix_fmt','yuv420p','-crf','20',out],stdin=subprocess.PIPE)
    writer.stdin.write(np.ascontiguousarray(frame).tobytes())
writer.stdin.close();writer.wait()
