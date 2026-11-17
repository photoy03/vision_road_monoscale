#!/usr/bin/env python3
"""Overlay frames of outer_run.py records on the cached camera previews (960x540).
usage: overlay.py VIDEO NAME OUTDIR [seconds ...]   (default every 5 s from 2.5 s)
own boundaries cyan (carried over: grey), outer lines magenta (carried over: dark magenta),
labels: side, style, yellow, gap, OPP (oncoming lane)."""
import sys,json,gzip
from pathlib import Path
import cv2,numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4]
sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import maploop_offline_compare as M  # noqa: E402
video,name,outdir=sys.argv[1],sys.argv[2],Path(sys.argv[3]);outdir.mkdir(parents=True,exist_ok=True)
rows=[json.loads(l) for l in gzip.open(HERE.parent/'runs'/f'{name}_{video}.jsonl.gz','rt')]
cache=Path('/media/jaehun/38F7-677B/vision-road-cache')/f'{video}_offline'
pk=ROOT/'src/vision_road';proj=M.projector(pk,M.settings(pk))
times=[float(a) for a in sys.argv[4:]] or list(np.arange(2.5,rows[-1]['t'],5.))


def draw(img,pts,color,width):
    p,v=proj.image_points(np.asarray(pts,float));p=p*.5
    ok=v&np.isfinite(p).all(1)&(np.abs(p).max(1)<1e5)
    edges=np.flatnonzero(np.r_[True,np.diff(ok.astype(int))!=0,True])
    for a,b in zip(edges[:-1],edges[1:]):
        if ok[a] and b-a>=2:cv2.polylines(img,[np.rint(p[a:b]).astype(np.int32)],False,color,width,cv2.LINE_AA)
    return p[ok]


for ts in times:
    i=min(range(len(rows)),key=lambda j:abs(rows[j]['t']-ts));r=rows[i]
    img=cv2.imread(str(cache/'preview'/f'{r["frame"]:05d}.png'))
    for side,l in r['own'].items():draw(img,l['points'],(160,160,160) if l['reprojected'] else (255,255,0),2)
    y=22
    for side,l in r['outer'].items():
        q=draw(img,l['points'],(110,0,110) if l['reprojected'] else (255,0,255),3)
        text=f"{side}: {l['style']} {'yellow' if l['yellow'] else 'white'} gap {l['gap_m']:.2f} m"+(' OPP' if l['opposite'] else '')+(' [PRED]' if l['reprojected'] else '')
        cv2.putText(img,text,(10,y),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),4);cv2.putText(img,text,(10,y),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,0,255),1);y+=22
    for side in ('left','right'):
        if side+'_outer' not in r['outer']:
            text=f"{side}_outer: none (paint {r['paint'].get(side,0):.1f} m)";cv2.putText(img,text,(10,y),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),4);cv2.putText(img,text,(10,y),cv2.FONT_HERSHEY_SIMPLEX,.55,(200,200,255),1);y+=22
    cv2.putText(img,f"{video} t={r['t']:.2f}s frame {r['frame']}",(10,530),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
    # BEV (x2): raw, and with the lines (1 px) so the paint under them stays visible.
    bev=cv2.imread(str(cache/'bev'/f'{r["frame"]:05d}.png'));S=2;mpp,fmax,lmin=.07,35.,-6.
    raw=cv2.resize(bev,None,fx=S,fy=S,interpolation=cv2.INTER_NEAREST);lined=raw.copy()
    def bevline(pts,color):
        p=np.asarray(pts,float);q=np.column_stack([(-p[:,1]-lmin)/mpp*S,(fmax-p[:,0])/mpp*S])
        cv2.polylines(lined,[np.rint(q).astype(np.int32)],False,color,1,cv2.LINE_AA)
    for side,l in r['own'].items():bevline(l['points'],(160,160,160) if l['reprojected'] else (255,255,0))
    for side,l in r['outer'].items():bevline(l['points'],(110,0,110) if l['reprojected'] else (255,0,255))
    for m in range(5,36,5):
        yy=int((fmax-m)/mpp*S);cv2.putText(lined,f'{m}',(2,yy),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,0),1)
    canvas=np.zeros((max(540,raw.shape[0]),960+2*raw.shape[1]+10,3),np.uint8);canvas[:540,:960]=img
    canvas[:raw.shape[0],965:965+raw.shape[1]]=raw;canvas[:raw.shape[0],970+raw.shape[1]:970+2*raw.shape[1]]=lined
    cv2.imwrite(str(outdir/f't{r["t"]:05.1f}.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,90])
print('wrote',len(times),'frames to',outdir)
