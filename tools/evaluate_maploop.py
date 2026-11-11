#!/usr/bin/env python3
"""Sparse source-image paint probes and unpainted ROIs, not full accuracy."""
import argparse,gzip,json
from pathlib import Path
import cv2,numpy as np

def raster(lines):
 mask=np.zeros((540,960),np.uint8)
 for lane in lines:
  points=np.asarray(lane['points'])
  if len(points)>1:cv2.polylines(mask,[np.rint(points).astype('int32')],False,255,3)
 return mask

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--predictions',type=Path,required=True);ap.add_argument('--labels',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();labels=json.loads(a.labels.read_text());wanted={r['frame'] for r in labels['positives']+labels['negatives']};frames={}
 with gzip.open(a.predictions,'rt') as f:
  for line in f:
   r=json.loads(line)
   if r['frame'] in wanted:frames[r['frame']]=r['outputs']
 metrics={};details=[]
 for key in ['previous','filtered']:
  hits=0;count=0;negative=[]
  for item in labels['positives']:
   mask=raster(frames[item['frame']][key]['paint']);distance=cv2.distanceTransform(255-mask,cv2.DIST_L2,5)
   for x,y in item['points']:
    error=float(distance[y,x]) if mask.any() else None;success=error is not None and error<=labels['tolerance_pixels'];hits+=success;count+=1;details.append(dict(variant=key,frame=item['frame'],point=[x,y],error_pixels=error,hit=success))
  for item in labels['negatives']:
   region=np.zeros((540,960),np.uint8);cv2.fillPoly(region,[np.asarray(item['polygon'],'int32')],255);negative.append(dict(frame=item['frame'],pixels=int(np.count_nonzero(region&raster(frames[item['frame']][key]['paint'])))))
  metrics[key]=dict(probes=count,within_8px=hits,negative_region_pixels=negative)
 a.output.write_text(json.dumps(dict(metrics=metrics,details=details,limitations=labels['limitations']),indent=2));print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
