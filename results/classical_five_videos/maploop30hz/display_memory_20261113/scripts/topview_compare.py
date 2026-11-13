"""Top view of the display line before/after (offline replay of the display filter with odometry).

usage: topview_compare.py PREDICTIONS TRUTH_BASE.json OUT.mp4 FIRST LAST
Left tile: code defaults (from 3 m, no memory).  Right tile: near_m=0, memory_m=10.
Grey: boundary road_node published; yellow: observed paint; thick cyan: display line
(magenta when drawn as predicted); small green dots: remembered paint.  Lines at 3 m (old
start) and 3.6 m (approximate front bumper); the bottom edge is the rear axle (x=0).
"""
import sys,gzip,json,subprocess,cv2,numpy as np
sys.path.insert(0,'/home/jaehun/Desktop/vision-road/src/vision_road')
from vision_road.display_filter import DisplayFilter,FilterSettings,relative
from vision_road.core import Timeline
PX=32.;XMAX=20.;YMAX=4.;W=int(2*YMAX*PX);H=int(XMAX*PX)+40
pred,truth,out,first,last=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),int(sys.argv[5])
rows=[json.loads(l) for l in gzip.open(pred,'rt')];odom=json.load(open(truth))
filters=[('before',DisplayFilter(FilterSettings())),('after',DisplayFilter(FilterSettings(near_m=0.,memory_m=10.)))]
def px(p):
    p=np.asarray(p,float);return np.column_stack([(YMAX-p[:,1])*PX,H-20-p[:,0]*PX]).round().astype(np.int32)
tl=Timeline(.25);k=0;last_pose=None;writer=None;t0=rows[0]['stamp']
for i,r in enumerate(rows[:last]):
    t=r['stamp']
    while k<len(odom) and odom[k][0]<=t+.1:tl.add(*odom[k]);k+=1
    pose=tl.at(t);motion=(0.,0.,0.) if last_pose is None else relative(last_pose,pose);last_pose=pose
    lines=[(b['side'],b['points'],b['reprojected']) for b in r['boundaries']];paint=[(l['side'],l['points']) for l in r['lanes'] if not l['reprojected']]
    tiles=[]
    for name,f in filters:
        for side in ('left','right'):
            st=[l for l in r['lanes'] if l['side']==side and not l['reprojected'] and l['style']!='unknown']
            if st:f.update_style(t,side,max(st,key=lambda l:l['hits'])['style'],1.)
        f.update(t,lines,motion,paint)
        if i<first:continue
        v=np.full((H,W,3),24,np.uint8)
        for x,c in ((3.,(90,90,90)),(3.6,(60,60,160))):
            y=int(H-20-x*PX);cv2.line(v,(0,y),(W,y),c,1)
        cv2.putText(v,'3 m',(2,int(H-20-3*PX)+12),cv2.FONT_HERSHEY_SIMPLEX,.35,(150,150,150),1)
        cv2.putText(v,'bumper',(2,int(H-20-3.6*PX)-3),cv2.FONT_HERSHEY_SIMPLEX,.35,(120,120,220),1)
        cv2.line(v,(0,H-20),(W,H-20),(120,120,120),1)
        for b in r['boundaries']:
            if not b['reprojected']:cv2.polylines(v,[px(b['points'])],False,(150,150,150),1)
        for _,p in paint:cv2.polylines(v,[px(p)],False,(0,220,255),2)
        for tr in f.sides.values():
            for q in px(tr.memory[:,:2]) if len(tr.memory) else []:cv2.circle(v,tuple(int(c) for c in q),1,(0,200,0),-1)
        for side,p,predicted,style in f.output():
            cv2.polylines(v,[px(p)],False,(255,0,255) if predicted else (255,255,0),2)
        cv2.putText(v,f'{name}  t {t-t0:.2f}s  #{i}',(4,14),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
        tiles.append(v)
    if i<first:continue
    frame=np.hstack([tiles[0],np.full((H,4,3),255,np.uint8),tiles[1]])
    if frame.shape[1]%2:frame=frame[:,:-1]
    if frame.shape[0]%2:frame=frame[:-1]
    if writer is None:
        writer=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{frame.shape[1]}x{frame.shape[0]}','-r','15','-i','pipe:0',
                                 '-c:v','libx264','-pix_fmt','yuv420p','-crf','20',out],stdin=subprocess.PIPE)
    writer.stdin.write(np.ascontiguousarray(frame).tobytes())
writer.stdin.close();writer.wait()
