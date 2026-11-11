#!/usr/bin/env python3
"""Offline maploop rerun: identical native-image BEVs and poses for any code version.

  cache: decode the lossless CDR cache once into BEV PNGs, overlay previews, poses
  run:   feed every frame to TemporalDetector(CarlaDetector) from --package-root
         (the node's maploop launch overrides) and draw the node's overlay
"""
import argparse,gzip,json,math,sqlite3,sys
from dataclasses import replace
from pathlib import Path
import cv2,numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1]
LAUNCH=dict(mpp=.07,image_roi_bottom=.915,display_simplify_error_m=.025)
SEGMENTS=[(0,5,'box junction'),(5,20,'straight'),(20,22,'right turn'),(22,35,'curve'),(35,42,'straight'),(42,48,'crosswalk'),(48,53,'island'),(53,61,'box junction')]


def stamp(m):return m.header.stamp.sec+m.header.stamp.nanosec*1e-9


def settings(package):
    from vision_road.core import Settings
    params=yaml.safe_load((package/'config/carla.yaml').read_text())['vision_road']['ros__parameters']
    params.update(mpp=LAUNCH['mpp'])
    if LAUNCH.get('lateral'):params.update(lateral_min=-LAUNCH['lateral'],lateral_max=LAUNCH['lateral'])
    return Settings(**{k:v for k,v in params.items() if k in Settings.__dataclass_fields__})


def projector(package,cfg):
    from vision_road.core import Projector
    rig=yaml.safe_load((package/'config/maploop_rig.yaml').read_text())
    p=Projector(package/'config/calibration/maploop_dashcam.npz',rig,cfg);p.initialize(1920,1080);return p


def cache(a):
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from vision_road.zstd_cdr import cache_records
    from vision_road.reference_pose import rear_axle
    package=ROOT/'src/vision_road';cfg=settings(package);proj=projector(package,cfg)
    (a.cache/'bev').mkdir(parents=True,exist_ok=True);(a.cache/'preview').mkdir(exist_ok=True)
    con=sqlite3.connect('file:'+str(a.bag.resolve())+'?mode=ro&immutable=1',uri=True)
    topics={n:i for i,n in con.execute('select id,name from topics')}
    poses=[]
    for data, in con.execute('select data from messages where topic_id=? order by timestamp',(topics['/carla/ground_truth/pose'],)):
        m=deserialize_message(data,PoseWithCovarianceStamped);p=m.pose.pose.position;q=m.pose.pose.orientation
        yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z));x,y=rear_axle(p.x,p.y,yaw)
        poses.append([stamp(m),x,y,yaw])
    con.close();rows=[]
    for i,data in enumerate(cache_records(a.cache_file,a.bag)):
        image=deserialize_message(data,Image);frame=np.frombuffer(image.data,np.uint8).reshape(image.height,image.width,3)
        if image.encoding=='rgb8':frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        bev=proj.warp(frame);valid=proj.valid.copy();valid[proj.my>=frame.shape[0]*LAUNCH['image_roi_bottom']]=0
        bev=cv2.bitwise_and(bev,bev,mask=valid)
        cv2.imwrite(str(a.cache/'bev'/f'{i:05d}.png'),bev,[cv2.IMWRITE_PNG_COMPRESSION,1])
        cv2.imwrite(str(a.cache/'preview'/f'{i:05d}.png'),cv2.resize(frame,(960,540),interpolation=cv2.INTER_AREA),[cv2.IMWRITE_PNG_COMPRESSION,1])
        rows.append(dict(frame=i,stamp=stamp(image)))
        if i%300==0:print('cache',i,flush=True)
    cv2.imwrite(str(a.cache/'valid.png'),proj.valid)
    (a.cache/'frames.json').write_text(json.dumps(dict(frames=rows,poses=poses,launch=LAUNCH)))


def lane_record(lane):
    return dict(side=lane.side,points=np.round(lane.points,3).tolist(),reprojected=bool(lane.reprojected),style=lane.style,hits=int(lane.hits),
                held=bool(lane.style!='unknown' and lane.style_source_time+1e-6<lane.source_time))


def render(a):
    """Redraw stored frames side by side: one column per run directory."""
    sys.path.insert(0,str(ROOT/'src/vision_road'))
    from vision_road.core import Lane
    from vision_road.marking_style import draw_style_label
    package=ROOT/'src/vision_road';proj=projector(package,settings(package));wanted=set(a.frames);runs=[]
    for run in a.runs:
        frames={}
        for line in gzip.open(run/'predictions.jsonl.gz','rt'):
            r=json.loads(line)
            if r['frame'] in wanted:frames[r['frame']]=r
        runs.append(frames)
    def lanes(items):
        out=[]
        for l in items:
            lane=Lane(l['side'],np.asarray(l['points'],float),.8,reprojected=l['reprojected'],style=l['style'],source_time=1.)
            lane.style_source_time=0. if l.get('held') else 1.;out.append(lane)
        return out
    rows=[]
    for i in a.frames:
        image=cv2.imread(str(a.cache/'preview'/f'{i:05d}.png'));tiles=[]
        for name,frames in zip(a.names,runs):
            r=frames[i];tile=draw(proj,image,lanes(r['boundaries']),lanes(r['lanes']),draw_style_label)
            cv2.rectangle(tile,(0,0),(330,34),(0,0,0),-1)
            cv2.putText(tile,f'{name}  t={r["stamp"]-runs[0][a.frames[0]]["stamp"]+a.frames[0]/30:.1f}s #{i}',(6,24),0,.7,(255,255,255),2)
            tiles.append(tile)
        rows.append(np.hstack(tiles))
    sheet=np.vstack(rows)
    if a.scale!=1:sheet=cv2.resize(sheet,None,fx=a.scale,fy=a.scale,interpolation=cv2.INTER_AREA)
    a.output.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(a.output),sheet,[cv2.IMWRITE_JPEG_QUALITY,90])


def draw(proj,image,boundaries,lanes,draw_style_label):
    overlay=image.copy();scale=.5;labels={}
    for index,lane in enumerate(boundaries+lanes):
        pixels,visible=proj.image_points(lane.points);pixels=pixels*scale
        safe=visible&np.isfinite(pixels).all(axis=1)&(np.max(np.abs(pixels),axis=1)<1e6)
        color=(255,220,0) if index<len(boundaries) else ((0,255,255) if lane.reprojected else (0,255,0))
        edges=np.flatnonzero(np.r_[True,np.diff(safe.astype(int))!=0,True])
        for lo,hi in zip(edges[:-1],edges[1:]):
            if safe[lo] and hi-lo>=2:cv2.polylines(overlay,[np.rint(pixels[lo:hi]).astype(np.int32)],False,color,3)
        if index<len(boundaries):continue
        if lane.side not in labels or np.ptp(lane.points[:,0])>np.ptp(labels[lane.side][0].points[:,0]):labels[lane.side]=(lane,pixels)
    for lane,pixels in labels.values():draw_style_label(overlay,lane,pixels)
    return overlay


def run(a):
    sys.path.insert(0,str(a.package_root))
    from vision_road.core import Timeline
    from vision_road.temporal import TemporalDetector
    from vision_road.carla import CarlaDetector
    from vision_road.marking_style import draw_style_label
    import vision_road
    assert Path(vision_road.__file__).resolve().parent==(a.package_root/'vision_road').resolve(),vision_road.__file__
    cv2.setNumThreads(2)
    cfg=settings(a.package_root);proj=projector(a.package_root,cfg)
    meta=json.loads((a.cache/'frames.json').read_text());timeline=Timeline(.25)
    for t,x,y,yaw in meta['poses']:timeline.add(t,x,y,yaw)
    valid=cv2.imread(str(a.cache/'valid.png'),cv2.IMREAD_GRAYSCALE)
    my=proj.my;valid[my>=1080*LAUNCH['image_roi_bottom']]=0
    det=TemporalDetector(cfg);det.detector=CarlaDetector(cfg)
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'overlay').mkdir(exist_ok=True)
    keys=set(a.key_frames);rows=[];last_debug=-math.inf
    def simplify(points):
        if len(points)<3:return points
        return cv2.approxPolyDP(points.astype(np.float32),LAUNCH['display_simplify_error_m'],False).reshape(-1,2).astype(float)
    video=None
    if a.video:
        import subprocess
        video=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','960x540','-r','30','-i','pipe:0','-c:v','libx264','-pix_fmt','yuv420p','-crf','23','-preset','veryfast',str(a.output/'overlay.mp4')],stdin=subprocess.PIPE)
    held=None
    with gzip.open(a.output/'predictions.jsonl.gz','wt') as out:
        for row in meta['frames']:
            i,t=row['frame'],row['stamp']
            bev=cv2.imread(str(a.cache/'bev'/f'{i:05d}.png'))
            lanes,_=det.process(bev,valid,t,timeline.at(t))
            display=[replace(l,points=simplify(l.points)) for l in lanes]
            boundaries=[replace(l,points=simplify(l.points)) for l in det.boundary_lanes]
            rec=dict(frame=i,stamp=t,lanes=[lane_record(l) for l in display],boundaries=[lane_record(l) for l in boundaries])
            out.write(json.dumps(rec)+'\n');rows.append(rec)
            debug=t-last_debug>=1/5-1e-8
            if debug or i in keys:
                image=cv2.imread(str(a.cache/'preview'/f'{i:05d}.png'));overlay=draw(proj,image,boundaries,display,draw_style_label)
                if i in keys:cv2.imwrite(str(a.output/'overlay'/f'{i:05d}.png'),overlay)
                if debug:last_debug=t;held=overlay
            if video is not None and held is not None:video.stdin.write(held.tobytes())
            if i%300==0:print('run',i,flush=True)
    if video is not None:video.stdin.close();video.wait()
    summary=metrics(rows)
    (a.output/'metrics.json').write_text(json.dumps(summary,indent=1));print(json.dumps(summary['overall'],indent=1))


def jagged(points):
    """Mean absolute second difference of lateral offset, at 0.25 m resampling."""
    if len(points)<2 or np.ptp(points[:,0])<1.:return None
    f=np.arange(points[0,0],points[-1,0],.25)
    if len(f)<4:return None
    y=np.interp(f,points[:,0],points[:,1]);return float(np.mean(np.abs(np.diff(y,2))))


def metrics(rows):
    t0=rows[0]['stamp'];per=[]
    for r in rows:
        rel=r['stamp']-t0;item=dict(t=rel)
        for side in ('left','right'):
            obs=[np.asarray(l['points']) for l in r['lanes'] if l['side']==side and not l['reprojected']]
            pred=[l for l in r['lanes'] if l['side']==side and l['reprojected']]
            bnd=[np.asarray(l['points']) for l in r['boundaries'] if l['side']==side]
            styles=[l['style'] for l in r['lanes'] if l['side']==side]
            item[side]=dict(observed=bool(obs),predicted=bool(pred) and not obs,
                reach=float(max(p[:,0].max() for p in obs)) if obs else 0.,
                boundary_reach=float(max(p[:,0].max() for p in bnd)) if bnd else 0.,
                lateral8=float(np.median([np.interp(8.,p[:,0],p[:,1]) for p in bnd])) if bnd and bnd[0][0,0]<=8.<=bnd[0][-1,0] else None,
                style=max(set(styles),key=styles.count) if styles else 'none',
                jag=[j for j in (jagged(p) for p in obs) if j is not None],
                strands=sum(p[0,0]<=10.<=p[-1,0] for p in obs))
        per.append(item)
    def agg(items):
        out={}
        for side in ('left','right'):
            s=[x[side] for x in items];n=max(1,len(s))
            jag=[j for x in s for j in x['jag']]
            styles={}
            for x in s:styles[x['style']]=styles.get(x['style'],0)+1
            out[side]=dict(observed=round(sum(x['observed'] for x in s)/n,3),predicted_only=round(sum(x['predicted'] for x in s)/n,3),
                median_reach_m=round(float(np.median([x['reach'] for x in s if x['observed']])),2) if any(x['observed'] for x in s) else 0.,
                median_boundary_reach_m=round(float(np.median([x['boundary_reach'] for x in s if x['boundary_reach']])),2) if any(x['boundary_reach'] for x in s) else 0.,
                paint_jag_m=round(float(np.mean(jag)),4) if jag else None,paint_jag_p90_m=round(float(np.percentile(jag,90)),4) if jag else None,
                strands_at_10m=round(float(np.mean([x['strands'] for x in s if x['strands']])),2) if any(x['strands'] for x in s) else 0.,
                styles=styles)
        return out
    return dict(overall=agg(per),segments={f'{a}-{b}s {name}':agg([x for x in per if a<=x['t']<b]) for a,b,name in SEGMENTS},
                per_frame=[dict(t=round(x['t'],3),**{s:dict(o=x[s]['observed'],r=round(x[s]['reach'],2),y8=x[s]['lateral8'],st=x[s]['style']) for s in ('left','right')}) for x in per])


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='cmd',required=True)
    c=sub.add_parser('cache');c.add_argument('--cache',type=Path,required=True)
    c.add_argument('--bag',type=Path,default=Path('/media/jaehun/38F7-677B/rosbag/maploop_30hz/maploop_30hz_0.db3'))
    c.add_argument('--cache-file',type=Path,default=ROOT/'data/maploop30hz_native.cdrz')
    r=sub.add_parser('run');r.add_argument('--cache',type=Path,required=True);r.add_argument('--output',type=Path,required=True)
    r.add_argument('--package-root',type=Path,default=ROOT/'src/vision_road');r.add_argument('--video',action='store_true')
    r.add_argument('--key-frames',type=int,nargs='*',default=[])
    m=sub.add_parser('metrics');m.add_argument('--output',type=Path,required=True)
    v=sub.add_parser('render');v.add_argument('--cache',type=Path,required=True);v.add_argument('--runs',type=Path,nargs='+',required=True)
    v.add_argument('--names',nargs='+',required=True);v.add_argument('--frames',type=int,nargs='+',required=True)
    v.add_argument('--output',type=Path,required=True);v.add_argument('--scale',type=float,default=.5)
    for parser in (c,r,v):parser.add_argument('--lateral',type=float,default=0.,help='BEV half width override in m (default: carla.yaml)')
    a=ap.parse_args()
    if getattr(a,'lateral',0.):LAUNCH['lateral']=a.lateral
    if a.cmd=='render':render(a);return
    if a.cmd=='metrics':
        rows=[json.loads(line) for line in gzip.open(a.output/'predictions.jsonl.gz','rt')]
        (a.output/'metrics.json').write_text(json.dumps(metrics(rows),indent=1));return
    if a.cmd=='cache':
        sys.path.insert(0,str(ROOT/'src/vision_road'));cache(a)
    else:a.package_root=a.package_root.resolve();run(a)
if __name__=='__main__':main()
