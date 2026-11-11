#!/usr/bin/env python3
"""Identical native-image BEVs and reference poses for before/after comparison."""
import argparse,gzip,importlib.util,json,shutil,subprocess,sys,time
from pathlib import Path
from dataclasses import replace,asdict
import cv2,numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src/vision_road'))
from vision_road.core import Settings,Projector
from vision_road.carla import CarlaDetector
from vision_road.temporal import TemporalDetector

def frozen(name,path):
 spec=importlib.util.spec_from_file_location('vision_road.'+name,path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(2)
 rows=json.loads((a.input/'frames.json').read_text());config=json.loads((a.input/'config.json').read_text());params=yaml.safe_load((ROOT/'src/vision_road/config/carla.yaml').read_text())['vision_road']['ros__parameters'];c=Settings(**{k:v for k,v in params.items() if k in Settings.__dataclass_fields__});valid=cv2.imread(str(a.input/'valid.png'),0);projector=Projector(a.input/'dashcam.npz',config['rig'],c);projector.initialize(960,540)
 old=ROOT/'research/2026-11-11/before_maploop';oldc=frozen('maploop_previous_carla',old/'carla.py').CarlaDetector;oldt=frozen('maploop_previous_temporal',old/'temporal.py').TemporalDetector
 detectors={}
 for key,klass,temporal,enabled in [('previous',oldc,oldt,True),('filtered',CarlaDetector,TemporalDetector,True)]:
  cfg=replace(c,carla_chain_filter_enabled=enabled);d=temporal(cfg);d.detector=klass(cfg);detectors[key]=d
 snapshot=a.output/'source_snapshot';snapshot.mkdir(exist_ok=True)
 for name in ['core.py','carla.py','temporal.py','node.py']:shutil.copy2(ROOT/'src/vision_road/vision_road'/name,snapshot/name)
 shutil.copy2(__file__,snapshot/Path(__file__).name);(a.output/'settings.json').write_text(json.dumps(asdict(c),indent=2))
 stats={k:dict(any=0,both=0,empty=0,paint_fragments=0,predicted_fragments=0,invalid_points=0,max_prediction_age=0.,times=[]) for k in detectors};samples=[]
 writer=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1280x720','-r','30','-i','pipe:0','-an','-c:v','libvpx-vp9','-deadline','realtime','-cpu-used','6','-crf','33','-b:v','0','-threads','2',str(a.output/'comparison.webm')],stdin=subprocess.PIPE)
 with gzip.open(a.output/'predictions.jsonl.gz','wt') as result:
  for row in rows:
   i=row['frame'];bev=cv2.imread(str(a.input/'bev'/f'{i:05d}.png'));image=cv2.imread(str(a.input/'preview'/f'{i:05d}.jpg'));canvas=np.zeros((720,1280,3),np.uint8);outputs={}
   canvas[360:720,:640]=cv2.resize(image,(640,360));cv2.putText(canvas,'SOURCE / CALIBRATED DASHCAM',(10,385),0,.6,(255,255,255),1)
   for j,(key,d) in enumerate(detectors.items()):
    before=time.perf_counter();lanes,mask=d.process(bev,valid,row['stamp'],row['pose']);s=stats[key];s['times'].append(time.perf_counter()-before);s['any']+=bool(lanes);s['both']+=len({l.side for l in lanes})==2;s['empty']+=not lanes;s['paint_fragments']+=sum(not l.reprojected for l in lanes);s['predicted_fragments']+=sum(l.reprojected for l in lanes)
    overlay=image.copy();data=[]
    for lane in lanes:
     px=d.pixel(lane.points);inside=(px[:,0]>=0)&(px[:,0]<valid.shape[1])&(px[:,1]>=0)&(px[:,1]<valid.shape[0]);ix=np.flatnonzero(inside);s['invalid_points']+=int(np.count_nonzero(~inside))+int(np.count_nonzero(valid[px[ix,1].astype(int),px[ix,0].astype(int)]==0))
     if lane.reprojected:s['max_prediction_age']=max(s['max_prediction_age'],row['stamp']-lane.source_time)
     points,visible=projector.image_points(lane.points);points=points[visible];cv2.polylines(overlay,[np.rint(points).astype('int32')],False,(0,255,255) if lane.reprojected else (0,255,0),2)
     data.append(dict(side=lane.side,points=points.tolist(),reprojected=lane.reprojected,style=lane.style))
    boundaries=[]
    if key=='filtered':
     for lane in d.boundary_lanes:
      points,_=projector.image_points(lane.points);boundaries.append(dict(side=lane.side,points=points.tolist(),reprojected=lane.reprojected));cv2.polylines(overlay,[np.rint(points).astype('int32')],False,(255,220,0),1)
    outputs[key]=dict(paint=data,boundaries=boundaries)
    y=(j//2)*360;x=(j%2)*640;canvas[y:y+360,x:x+640]=cv2.resize(overlay,(640,360));cv2.rectangle(canvas,(x,y),(x+640,y+30),(20,20,20),-1);cv2.putText(canvas,f'{key.upper()} | {row["seconds"]:.2f}s | {len(lanes)} fragments',(x+8,y+22),0,.57,(255,255,255),1)
    if key=='filtered':
     colored=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR);cv2.putText(canvas,'FILTERED PAINT MASK',(660,390),0,.6,(255,255,255),1);canvas[410:710,880:992]=cv2.resize(colored,(112,300),interpolation=cv2.INTER_NEAREST);cv2.putText(canvas,'Green: paint / cyan: boundary fit',(660,700),0,.5,(200,230,230),1)
   writer.stdin.write(canvas.tobytes());result.write(json.dumps(dict(frame=i,seconds=row['seconds'],outputs=outputs),separators=(',',':'))+'\n')
   if i%150==0 or i==len(rows)-1:
    name=f'review_{i:05d}.jpg';cv2.imwrite(str(a.output/name),canvas);samples.append(dict(frame=i,seconds=row['seconds'],file=name));print(i,round(row['seconds'],2),{k:len(v['paint']) for k,v in outputs.items()},flush=True)
 writer.stdin.close();assert writer.wait()==0
 for s in stats.values():
  times=s.pop('times');s['median_ms']=float(np.median(times)*1000);s['p95_ms']=float(np.quantile(times,.95)*1000)
 (a.output/'summary.json').write_text(json.dumps(dict(frames=len(rows),metrics=stats,reference_pose=True,independent_ground_truth_labels=False,note='Reference pose is not MonoScale odometry. Counts measure output activity, not accuracy.'),indent=2));(a.output/'samples.json').write_text(json.dumps(samples,indent=2));print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
