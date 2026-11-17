#!/usr/bin/env python3
"""Camera + BEV overlays of dump_lines.py records with centrelines.  usage: overlay_centres.py VIDEO NAME OUTDIR [seconds ...]
own boundaries cyan, outer lines magenta, centrelines orange dashes (estimated: thin); text: per lane width, offset and
heading at its reference point, curvature, flags.  Writes also OUTDIR/sheet_5s.jpg for 5, 10, ... s."""
import sys,json,gzip
from pathlib import Path
import cv2,numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4]
sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import maploop_offline_compare as M  # noqa: E402
video,name,outdir=sys.argv[1],sys.argv[2],Path(sys.argv[3]);outdir.mkdir(parents=True,exist_ok=True)
rows=[json.loads(l) for l in gzip.open(HERE.parent/'runs'/f'{name}_{video}.jsonl.gz','rt')];t0=rows[0]['stamp']
cache=Path('/media/jaehun/38F7-677B/vision-road-cache')/f'{video}_offline';pk=ROOT/'src/vision_road';proj=M.projector(pk,M.settings(pk))
times=[float(a) for a in sys.argv[4:]] or list(np.arange(2.5,rows[-1]['stamp']-t0,2.5))
S=2;mpp,fmax,lmin=.07,35.,-6.
def cam(img,pts,color,width,dash=False):
    p=np.asarray(pts,float)
    if dash:x=np.arange(p[0,0],p[-1,0]+1e-9,1.);p=np.column_stack([x,np.interp(x,p[:,0],p[:,1])])
    q,v=proj.image_points(p);q=q*.5;ok=v&np.isfinite(q).all(1)&(np.abs(q).max(1)<1e5)
    if dash:
        for k in range(0,len(q)-1,2):
            if ok[k] and ok[k+1]:cv2.line(img,tuple(np.rint(q[k]).astype(int)),tuple(np.rint(q[k+1]).astype(int)),color,width,cv2.LINE_AA)
        return
    e=np.flatnonzero(np.r_[True,np.diff(ok.astype(int))!=0,True])
    for a,b in zip(e[:-1],e[1:]):
        if ok[a] and b-a>=2:cv2.polylines(img,[np.rint(q[a:b]).astype(np.int32)],False,color,width,cv2.LINE_AA)
def bev(img,pts,color,dash=False):
    p=np.asarray(pts,float);q=np.column_stack([(-p[:,1]-lmin)/mpp*S,(fmax-p[:,0])/mpp*S])
    if dash:
        for k in range(0,len(q)-1,6):cv2.polylines(img,[np.rint(q[k:k+4]).astype(np.int32)],False,color,1,cv2.LINE_AA)
    else:cv2.polylines(img,[np.rint(q).astype(np.int32)],False,color,1,cv2.LINE_AA)
out=[]
for ts in times:
    i=min(range(len(rows)),key=lambda j:abs(rows[j]['stamp']-t0-ts));r=rows[i]
    img=cv2.imread(str(cache/'preview'/f'{r["frame"]:05d}.png'));b=cv2.imread(str(cache/'bev'/f'{r["frame"]:05d}.png'))
    raw=cv2.resize(b,None,fx=S,fy=S,interpolation=cv2.INTER_NEAREST);lined=raw.copy()
    for l in r['bounds']:cam(img,l['points'],(160,160,160) if l['rep'] else (255,255,0),2);bev(lined,l['points'],(255,255,0))
    for l in r['outer']:cam(img,l['points'],(110,0,110) if l['rep'] else (255,0,255),2);bev(lined,l['points'],(255,0,255))
    y=20
    for c in sorted(r['centres'],key=lambda c:c['index']):
        cam(img,c['points'],(0,140,255),1 if c['estimated'] else 3,dash=True);bev(lined,c['points'],(0,140,255),dash=True)
        f=[k for k in ('estimated','opposite_direction','crossable','edge_unknown','lane_like') if c.get(k)]
        text=f"{c['name']}: w {c['width_m']:.2f} off {c['offset_m']:+.2f}@{c['ref_x_m']:.1f}m hd {c['heading_rad']*1e3:+.0f}mrad k {c['curvature_1pm']*1e3:+.1f}/km {' '.join(f)}"
        cv2.putText(img,text,(8,y),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),3);cv2.putText(img,text,(8,y),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,170,255),1);y+=18
    cv2.putText(img,f"{video} t={r['stamp']-t0:.2f}s",(8,530),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),2)
    for m in range(5,36,5):cv2.putText(lined,f'{m}',(2,int((fmax-m)/mpp*S)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,0),1)
    canvas=np.zeros((max(540,raw.shape[0]),960+2*raw.shape[1]+10,3),np.uint8);canvas[:540,:960]=img
    canvas[:raw.shape[0],965:965+raw.shape[1]]=raw;canvas[:raw.shape[0],970+raw.shape[1]:970+2*raw.shape[1]]=lined
    f=outdir/f't{r["stamp"]-t0:05.1f}.jpg';cv2.imwrite(str(f),canvas,[cv2.IMWRITE_JPEG_QUALITY,90]);out.append((ts,canvas))
tiles=[]
for ts,c in out:
    if abs(ts/5-round(ts/5))>1e-6:continue
    w=c.shape[1];tile=np.zeros((637,320,3),np.uint8);tile[:180]=cv2.resize(c[:540,:960],(320,180));tile[180:,75:246]=cv2.resize(c[:,w-342:],(171,457))
    cv2.putText(tile,f'{ts:.0f} s',(5,200),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1);tiles.append(tile)
if tiles:
    rows_=[np.hstack(tiles[i:i+6]+[np.zeros_like(tiles[0])]*(6-len(tiles[i:i+6]))) for i in range(0,len(tiles),6)]
    cv2.imwrite(str(outdir/'sheet_5s.jpg'),np.vstack(rows_),[cv2.IMWRITE_JPEG_QUALITY,90])
print('wrote',len(out),'frames to',outdir)
