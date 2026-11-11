#!/usr/bin/env python3
"""Summarize candidate continuity and make fixed-time visual review sheets (not accuracy)."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('result_dir');args=p.parse_args()
    root=Path(args.result_dir);summary=json.loads((root/'summary.json').read_text())
    frames=list(csv.DictReader((root/'frames.csv').open()))
    empty=[int(r['observed_lanes'])==0 for r in frames]
    longest=run=0;start=best_start=best_end=0
    for i,missing in enumerate(empty):
        if missing:
            if not run:start=i
            run+=1
            if run>longest:longest=run;best_start=start;best_end=i
        else:run=0
    summary['longest_run_without_current_observation_frames']=longest
    summary['longest_run_without_current_observation_indices']=[best_start,best_end] if longest else []
    summary['candidate_output_fraction']=sum(not x for x in empty)/max(1,len(empty))
    summary['metric_caution']='Continuity/output counts, not correctness. No lane ground-truth annotations.'
    (root/'review_summary.json').write_text(json.dumps(summary,indent=2))
    cap=cv2.VideoCapture(str(root/'overlay.mp4'));count=len(frames)
    indices=sorted(set(round((count-1)*f) for f in [0,.1,.25,.5,.75,.9,1.]))
    if longest:indices+=sorted(set([best_start,(best_start+best_end)//2,best_end])-set(indices))
    tiles=[]
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,image=cap.read()
        if not ok:raise RuntimeError('Cannot read diagnostic frame '+str(i))
        tile=np.zeros((240,720,3),'uint8');tile[24:]=cv2.resize(image,(720,216))
        cv2.putText(tile,'frame '+str(i)+' (green=observed; yellow=predicted)',(8,17),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        tiles.append(tile)
    cap.release();rows=(len(tiles)+1)//2;canvas=np.zeros((rows*240,1440,3),'uint8')
    for i,tile in enumerate(tiles):canvas[(i//2)*240:(i//2+1)*240,(i%2)*720:(i%2+1)*720]=tile
    cv2.imwrite(str(root/'review_contact_sheet.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,92])
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
