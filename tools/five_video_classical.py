#!/usr/bin/env python3
"""Run five real routes using only image processing and motion geometry."""
import argparse,hashlib,json,sys,time,shutil
from dataclasses import asdict
from pathlib import Path
import cv2,numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src/vision_road'))
from vision_road.core import Settings,Projector,Timeline,LaneMap
from vision_road.kitti import KittiSequence
from vision_road.temporal import TemporalDetector
from vision_road.classical import StripeDetector,PaintEvidenceDetector
from vision_road.marking_style import draw_style_label

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--data-root',type=Path,default=ROOT/'data/kitti/2011_09_26');ap.add_argument('--sequences',nargs='+',default=['0001','0015','0028','0018','0056']);ap.add_argument('--backend',default='evidence',choices=['baseline','stripes','evidence']);ap.add_argument('--no-video',action='store_true');ap.add_argument('--no-style-history',action='store_true');ap.add_argument('--no-candidate-mask',action='store_true');args=ap.parse_args()
 cv2.setNumThreads(2)
 output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
 cfg=yaml.safe_load((ROOT/'src/vision_road/config/kitti.yaml').read_text())['vision_road']['ros__parameters'];c=Settings(**{k:v for k,v in cfg.items() if k in Settings.__dataclass_fields__})
 c.style_temporal_enabled=not args.no_style_history
 c.candidate_mask_enabled=not args.no_candidate_mask
 c.style_hold_sec=2.0 if args.backend=='evidence' else 0.0
 snapshot=output/'source_snapshot';snapshot.mkdir(exist_ok=True)
 for name in ['core.py','temporal.py','classical.py','marking_style.py']:
  shutil.copy2(ROOT/'src/vision_road/vision_road'/name,snapshot/name)
 shutil.copy2(__file__,snapshot/Path(__file__).name)
 (output/'run_config.json').write_text(json.dumps(dict(learning_used=False,settings=asdict(c),arguments=vars(args),python=sys.version,opencv=cv2.__version__,numpy=np.__version__),indent=2,default=str))
 allpred=[];summaries={}
 for seq in args.sequences:
  d=KittiSequence(args.data_root/f'2011_09_26_drive_{seq}_sync');out=output/seq;d.export(out);p=Projector(out/'camera.npz',d.rig,c)
  det=TemporalDetector(c)
  if args.backend=='stripes':det.detector=StripeDetector(c)
  if args.backend=='evidence':det.detector=PaintEvidenceDetector(c)
  lm=LaneMap(radius=40)
  timeline=Timeline(.25)
  for t,pose in zip(d.oxts_times,d.poses):timeline.add(float(t),*pose[:2,3],np.arctan2(pose[1,0],pose[0,0]))
  fps=1/float(np.median(np.diff(d.image_times)));writer=None;pred=[];timings=[]
  if not args.no_video:
   writer=cv2.VideoWriter(str(out/'overlay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,tuple(n+n%2 for n in d.size))
   if not writer.isOpened():raise RuntimeError('Video writer failed')
  for i in range(len(d.images)):
   t0=time.perf_counter();im=d.image(i);bev=p.warp(im);stamp=float(d.image_times[i]);pose=timeline.at(stamp)
   lanes,binary=det.process(bev,p.valid,stamp,pose);current=[]
   for lane in lanes:
    if not np.isfinite(lane.points).all():raise RuntimeError('Non-finite lane')
    if lane.reprojected and stamp-lane.source_time>c.temporal_max_age_sec+1e-6:raise RuntimeError('Prediction exceeded TTL')
    xy,vis=p.image_points(lane.points);current.append(dict(side=lane.side,points=xy[vis].tolist(),style=lane.style,confidence=lane.confidence,reprojected=lane.reprojected,source_time=lane.source_time,style_source_time=lane.style_source_time))
    cv2.polylines(im,[np.rint(xy[vis]).astype(np.int32)],False,(0,255,255) if lane.reprojected else (0,255,0),2);draw_style_label(im,lane,xy[vis])
   if pose is not None:lm.update(lanes,pose,stamp)
   elapsed=time.perf_counter()-t0;timings.append(elapsed);rec=dict(sequence=seq,frame=i,time=stamp,lanes=current,seconds=elapsed);pred.append(rec);allpred.append(rec)
   if writer is not None:writer.write(cv2.copyMakeBorder(im,0,im.shape[0]%2,0,im.shape[1]%2,cv2.BORDER_CONSTANT))
   if i in {0,len(d.images)//4,len(d.images)//2,3*len(d.images)//4,len(d.images)-1}:cv2.imwrite(str(out/f'frame_{i:06d}.jpg'),im)
   if i%50==0:print(seq,i,'lanes',len(lanes),'seconds',round(elapsed,3),flush=True)
  if writer is not None:writer.release()
  (out/'predictions.json').write_text(json.dumps(pred,indent=2));summaries[seq]=dict(frames=len(d.images),duration_seconds=float(d.image_times[-1]-d.image_times[0]),median_seconds=float(np.median(timings)) if timings else None,map_cells=len(lm.cells),pose_source='KITTI OXTS, not MonoScale',note='Counts/timing do not measure correctness. Pure image processing; no weights, no neural inference or learned evidence cache.')
  print('DONE',seq,json.dumps(summaries[seq]),flush=True)
 (output/'predictions.json').write_text(json.dumps(allpred,indent=2));(output/'summary.json').write_text(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
