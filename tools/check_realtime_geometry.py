#!/usr/bin/env python3
import argparse,gzip,json,sys,time
from pathlib import Path
from dataclasses import fields
import cv2,numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src/vision_road'))
from vision_road.core import Settings,Projector
from vision_road.carla import CarlaDetector,_native
from vision_road.temporal import TemporalDetector

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--golden',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(2)
 assert _native is not None,'Native extension missing'
 cfg=yaml.safe_load((ROOT/'src/vision_road/config/carla.yaml').read_text())['vision_road']['ros__parameters'];c=Settings(**{k:v for k,v in cfg.items() if k in Settings.__dataclass_fields__});p=Projector(a.input/'dashcam.npz',yaml.safe_load((a.input/'rig.yaml').read_text()),c);p.initialize(960,540);v=cv2.imread(str(a.input/'valid.png'),0);rows=json.loads((a.input/'frames.json').read_text());d=TemporalDetector(c);d.detector=CarlaDetector(c);times=[];diff=[];max_error=0.;nonempty=0
 with gzip.open(a.golden,'rt') as before,gzip.open(a.output/'predictions.jsonl.gz','wt') as after:
  for row,line in zip(rows,before):
   i=row['frame'];old=json.loads(line);im=cv2.imread(str(a.input/'bev'/f'{i:05d}.png'));t=time.perf_counter();lanes,mask=d.process(im,v,row['stamp'],row['pose']);times.append((time.perf_counter()-t)*1000);nonempty+=bool(lanes);current=[]
   for lane in lanes:
    points,visible=p.image_points(lane.points);current.append(dict(side=lane.side,points=points[visible].tolist(),reprojected=lane.reprojected,style=lane.style))
   expected=old['outputs']['filtered']['paint'];same=len(expected)==len(current)
   if same:
    for x,y in zip(expected,current):
     if x['side']!=y['side'] or x['reprojected']!=y['reprojected'] or x['style']!=y['style'] or np.shape(x['points'])!=np.shape(y['points']):same=False;break
     if len(x['points']):
      error=float(np.max(np.abs(np.array(x['points'])-y['points'])));max_error=max(max_error,error)
      if error>1e-6:same=False;break
   if not same:diff.append(i)
   old['outputs']['filtered']['paint']=current;after.write(json.dumps(old,separators=(',',':'))+'\n')
   if i%300==0:print(i,'different',len(diff),'median',round(float(np.median(times)),2),flush=True)
 result=dict(frames=len(times),native=True,median_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),p99_ms=float(np.quantile(times,.99)),max_ms=max(times),over_33ms=sum(t>1000/30 for t in times),different_frames=diff,max_matching_point_error_pixels=max_error,nonempty=nonempty,note='Engine only, cached native-image BEV; not ROS delivery or source pacing measurement.')
 (a.output/'engine_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
