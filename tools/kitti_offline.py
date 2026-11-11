#!/usr/bin/env python3
"""Check KITTI folders or render detections without ROS; no data downloads."""
import argparse
import csv
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/vision_road'))
from vision_road.kitti import KittiSequence
from vision_road.core import Settings, Projector, Timeline, LaneMap
from vision_road.temporal import TemporalDetector
from vision_road.classical import PaintEvidenceDetector
from vision_road.marking_style import draw_style_label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence',required=True,help='Extracted KITTI Raw *_sync folder')
    parser.add_argument('--output',required=True,help='Output folder for configuration, preview and counts')
    parser.add_argument('--imu-height-m',type=float,default=.93)
    parser.add_argument('--max-frames',type=int,default=300)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--no-temporal',action='store_true',help='Disable motion-compensated history for comparison')
    parser.add_argument('--no-shadow',action='store_true',help='Use the old brightness threshold, for comparison')
    parser.add_argument('--backend',choices=['baseline','paint_evidence'],help='Default: configured pure image-processing detector')
    parser.add_argument('--no-odom',action='store_true',help='Evaluate visual-motion fallback without OXTS')
    args = parser.parse_args()
    if args.max_frames < 1: parser.error('--max-frames must be positive')
    try:
        data = KittiSequence(args.sequence,args.imu_height_m)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2,'KITTI input error: '+str(exc)+'\nSee KITTI_TEST.md for the required extracted folder layout.\n')
    out = data.export(args.output)
    if args.check_only:
        print((out/'dataset.json').read_text()); return
    defaults = yaml.safe_load((Path(__file__).resolve().parents[1]/'src/vision_road/config/kitti.yaml').read_text())['vision_road']['ros__parameters']
    settings = Settings(**{k:v for k,v in defaults.items() if k in Settings.__dataclass_fields__})
    settings.temporal_enabled=not args.no_temporal; settings.shadow_enabled=not args.no_shadow
    projector = Projector(out/'camera.npz',data.rig,settings); detector = TemporalDetector(settings)
    backend=args.backend or defaults.get('detector_backend','baseline')
    if backend=='paint_evidence':detector.detector=PaintEvidenceDetector(settings)
    cv2.setNumThreads(2)
    timeline = Timeline(.25)
    for t,pose in zip(data.oxts_times,data.poses): timeline.add(float(t),*pose[:2,3],np.arctan2(pose[1,0],pose[0,0]))
    lane_map = LaneMap(radius=40)
    count = min(args.max_frames,len(data.images)); detected=paired=matched=predicted=observed=pred_only=0; registration_counts={};style_counts={}
    fps = 1/float(np.median(np.diff(data.image_times))) if len(data.image_times)>1 else 10.
    writer = cv2.VideoWriter(str(out/'overlay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,tuple(n+n%2 for n in data.size))
    if not writer.isOpened(): raise RuntimeError('Cannot create MP4; install an OpenCV video encoder')
    try:
        with (out/'frames.csv').open('w',newline='') as f:
            csvout = csv.writer(f); csvout.writerow(['frame','time_sec','lanes_detected','segments','pose_valid','observed_lanes','predicted_lanes','registration','lane_styles'])
            for i in range(count):
                image = data.image(i); bev = projector.warp(image); t = float(data.image_times[i])
                pose = None if args.no_odom else timeline.at(t)
                lanes,binary = detector.process(bev,projector.valid,t,pose)
                measured=detector.measured_count; carried=detector.predicted_count
                observed+=measured>0; predicted+=carried>0; pred_only+=(carried>0 and measured==0)
                registration_counts[detector.registration]=registration_counts.get(detector.registration,0)+1
                detected += bool(lanes); paired += len(lanes)==2; matched += pose is not None
                if pose is not None: lane_map.update(lanes,pose,t)
                for lane in lanes:
                    key=lane.side+':'+lane.style
                    style_counts[key]=style_counts.get(key,0)+1
                    pixels, visible = projector.image_points(lane.points)
                    for a,b,ok in zip(pixels[:-1],pixels[1:],visible[:-1]&visible[1:]):
                        if ok and np.isfinite([a,b]).all() and np.max(np.abs([a,b]))<1e6:
                            cv2.line(image,tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)),(0,255,255) if lane.reprojected else (0,255,0),3)
                    draw_style_label(image,lane,pixels)
                    points = np.column_stack([(-lane.points[:,1]-settings.lateral_min)/settings.mpp,(settings.forward_max-lane.points[:,0])/settings.mpp]).astype('int32')
                    cv2.polylines(bev,[points],False,(0,255,255) if lane.reprojected else (0,255,0),2)
                cv2.putText(image,f'frame={i} observed={measured} predicted={carried} {detector.registration}',(15,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
                writer.write(cv2.copyMakeBorder(image,0,image.shape[0]%2,0,image.shape[1]%2,cv2.BORDER_CONSTANT))
                if i in {0,27,count//2,70,90,count-1}:
                    cv2.imwrite(str(out/f'overlay_{i:06d}.png'),image); cv2.imwrite(str(out/f'bev_{i:06d}.png'),bev)
                    cv2.imwrite(str(out/f'binary_{i:06d}.png'),binary)
                csvout.writerow([i,t,len(lanes),sum(len(l.points)-1 for l in lanes),pose is not None,measured,carried,detector.registration,';'.join(l.side+':'+l.style for l in lanes)])
    finally: writer.release()
    result = {'frames_processed':count,'frames_with_any_detection':detected,'frames_with_two_lanes':paired,
              'frames_with_matched_oxts':matched,'accumulated_cells':len(lane_map.cells),
              'frames_with_observed_lanes':observed,'frames_with_predictions':predicted,'prediction_only_frames':pred_only,
              'lane_style_counts':style_counts,'registration_counts':registration_counts,'shadow_enabled':settings.shadow_enabled,'temporal_enabled':settings.temporal_enabled,
              'note':'Detection counts are not accuracy. No lane ground truth is loaded. Pose is KITTI OXTS when enabled, not MonoScale. Green=observed fit; yellow=short history prediction; predictions excluded from mapping.'}
    (out/'summary.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); print('Output: '+str(out))


if __name__ == '__main__': main()
