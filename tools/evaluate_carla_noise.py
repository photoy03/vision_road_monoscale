#!/usr/bin/env python3
"""Sparse local paint-point and manually scoped negative-ROI checks, not GT accuracy."""
import argparse,json,gzip
from pathlib import Path
import cv2,numpy as np


def raster(lanes):
    mask=np.zeros((720,1280),np.uint8)
    for lane in lanes:
        p=np.asarray(lane['points'])
        if len(p)>=2:cv2.polylines(mask,[np.rint(p).astype(np.int32)],False,255,3)
    return mask


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--labels',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--predictions',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    labels=json.loads(args.labels.read_text());wanted={r['frame'] for r in labels['positives']+labels['negatives']}
    old=[json.loads(line) for line in gzip.open(args.baseline,'rt')];lookup={r['frame']:{'previous':r['outputs']['stage3']} for r in old if r['frame'] in wanted}
    with gzip.open(args.predictions,'rt') as f:
        for line in f:
            r=json.loads(line)
            if r['frame'] in wanted:lookup[r['frame']].update(r['outputs'])
    summaries={};details=[]
    for key in ['previous','stage3']:
        distances=[];neg=[]
        for r in labels['positives']:
            mask=raster(lookup[r['frame']][key]);distance=cv2.distanceTransform(255-mask,cv2.DIST_L2,5)
            for q in r['points']:
                x,y=q['point'];error=float(distance[y,x]) if mask.any() else None
                distances.append(error);details.append(dict(variant=key,frame=r['frame'],kind=q['kind'],point=q['point'],distance_pixels=error,within_tolerance=error is not None and error<=labels['tolerance_pixels']))
        for r in labels['negatives']:
            region=np.zeros((720,1280),np.uint8);cv2.fillPoly(region,[np.asarray(r['polygon'],np.int32)],255)
            count=int(np.count_nonzero(raster(lookup[r['frame']][key])&region));neg.append(dict(frame=r['frame'],pixels=count))
        hit=sum(d is not None and d<=labels['tolerance_pixels'] for d in distances)
        summaries[key]=dict(paint_probes=len(distances),within_10px=int(hit),negative_region_predicted_pixels=neg)
    result=dict(label_version=labels['version'],summary=summaries,details=details,note=labels['limitations'])
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
