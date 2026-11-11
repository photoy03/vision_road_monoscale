#!/usr/bin/env python3
"""Diagnostic-only monocular video trial with explicit approximate camera profile.

Does not change production calibration, publish ROS, or claim metric accuracy.
"""
import argparse
from dataclasses import asdict, replace
import hashlib, json, math, subprocess, sys, time
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/vision_road'))
from vision_road.core import Settings, Projector
from vision_road.temporal import TemporalDetector
from vision_road.classical import PaintEvidenceDetector


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--video',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--horizontal-fov',type=float,default=90.)
    ap.add_argument('--camera-height',type=float,default=1.5)
    ap.add_argument('--pitch',type=float,default=0.)
    ap.add_argument('--hood-y',type=float,default=.86)
    args=ap.parse_args()
    if not 20<args.horizontal_fov<160 or args.camera_height<=0 or not 0<args.hood_y<=1:
        raise ValueError('Invalid diagnostic camera parameters')
    cv2.setNumThreads(2)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(args.video));ok,frame=cap.read()
    if not ok:raise ValueError('Video cannot be read')
    h,w=frame.shape[:2]
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time,pkt_duration_time','-of','json',str(args.video)]))['frames']
    pts=np.array([float(f['best_effort_timestamp_time']) for f in probe]);pts-=pts[0]
    if len(pts)<2 or np.any(np.diff(pts)<=0):raise ValueError('Nonmonotonic input PTS')
    end=pts[-1]+float(probe[-1].get('pkt_duration_time',1/24))
    c=Settings(lateral_min=-6,lateral_max=6,forward_min=3.,forward_max=35.,mpp=.05,windows=24,margin_m=.5,min_span_m=2.,style_temporal_enabled=True,style_hold_sec=2.)
    focal=w/(2*math.tan(math.radians(args.horizontal_fov/2)))
    np.savez(out/'APPROXIMATE_camera.npz',K=[[focal,0,w/2],[0,focal,h/2],[0,0,1]],D=np.zeros(5),image_size=[w,h])
    rig=dict(distortion_model='plumb_bob',pitch_deg=args.pitch,roll_deg=0.,yaw_deg=0.,camera_forward_m=0.,camera_left_m=0.,camera_height_m=args.camera_height)
    projector=Projector(out/'APPROXIMATE_camera.npz',rig,c)
    detectors=[]
    for enabled in [False,True]:
        cfg=replace(c,candidate_mask_enabled=enabled)
        d=TemporalDetector(cfg);d.detector=PaintEvidenceDetector(cfg);detectors.append(d)
    run=dict(input=str(args.video),input_sha256=hashlib.sha256(args.video.read_bytes()).hexdigest(),camera_assumptions=vars(args),rig=rig,settings=asdict(c),calibration_verified=False,metric_accuracy_validated=False,odom_available=False,temporal_registration='image optical flow only',timestamps='original decoded frame PTS; output repeats frames onto 24 fps grid',learning_used=False)
    (out/'run_config.json').write_text(json.dumps(run,indent=2,default=str))
    snap=out/'source_snapshot';snap.mkdir(exist_ok=True)
    import shutil
    for name in ['core.py','classical.py','temporal.py','marking_style.py','rig.py']:
        shutil.copy2(ROOT/'src/vision_road/vision_road'/name,snap/name)
    shutil.copy2(__file__,snap/Path(__file__).name)
    command=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1280x720','-r','24','-i','pipe:0','-an','-c:v','libvpx-vp9','-deadline','realtime','-cpu-used','6','-crf','34','-b:v','0','-row-mt','1','-threads','2','-pix_fmt','yuv420p',str(out/'comparison.webm')]
    records=[];samples=[];targets=list(range(0,61,5));ticks=np.rint(np.r_[pts,end]*24).astype(int)
    with (out/'encode.log').open('w') as log:
        writer=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        try:
            i=0
            while ok:
                start=time.perf_counter();bev=projector.warp(frame)
                valid=projector.valid.copy();valid[projector.my>=args.hood_y*h]=0
                bev=cv2.bitwise_and(bev,bev,mask=valid)
                lanes=[];masks=[];overlays=[];runs=[]
                for detector in detectors:
                    found,binary=detector.process(bev,valid,float(pts[i]),None)
                    lanes.append(found);masks.append(binary.copy());im=frame.copy();pred=[]
                    for lane in found:
                        xy,visible=projector.image_points(lane.points);xy=xy[visible]
                        if len(xy)>=2:
                            cv2.polylines(im,[np.rint(xy).astype(np.int32)],False,(0,255,255) if lane.reprojected else (0,255,0),3)
                            label=lane.side+' '+lane.style+(' PRED' if lane.reprojected else '')
                            x,y=np.rint(xy[len(xy)//2]).astype(int);cv2.putText(im,label,(max(0,min(w-210,x)),max(25,min(h-40,y))),0,.6,(0,0,0),4);cv2.putText(im,label,(max(0,min(w-210,x)),max(25,min(h-40,y))),0,.6,(0,255,0),1)
                        pred.append(dict(side=lane.side,style=lane.style,reprojected=lane.reprojected,source_time=lane.source_time,confidence=float(lane.confidence),points=xy.tolist()))
                    runs.append(dict(registration=detector.registration,lanes=pred,candidate_pixels=int(np.count_nonzero(binary))))
                    overlays.append(im)
                canvas=np.full((720,1280,3),20,np.uint8)
                for k,title in [(0,'MASK OFF'),(1,'MASK ON')]:
                    canvas[:360,k*640:(k+1)*640]=cv2.resize(overlays[k],(640,360))
                    cv2.rectangle(canvas,(k*640,0),(k*640+640,38),(15,15,15),-1)
                    cv2.putText(canvas,f'{title} | {pts[i]:.2f}s',(k*640+12,26),0,.7,(255,255,255),1)
                removed=cv2.bitwise_and(masks[0],cv2.bitwise_not(masks[1]));mark=bev.copy();mark[removed>0]=(0,0,255)
                for k,title,image in [(0,'BEV (APPROXIMATE)',bev),(1,'CANDIDATES / OFF',cv2.cvtColor(masks[0],cv2.COLOR_GRAY2BGR)),(2,'CANDIDATES / ON',cv2.cvtColor(masks[1],cv2.COLOR_GRAY2BGR)),(3,'OFF-only pixels / RED',mark)]:
                    iw=round(image.shape[1]*290/image.shape[0]);x=k*320+(320-iw)//2
                    canvas[400:690,x:x+iw]=cv2.resize(image,(iw,290),interpolation=cv2.INTER_NEAREST)
                    cv2.putText(canvas,title,(k*320+8,385),0,.48,(240,240,240),1)
                cv2.putText(canvas,'Estimated camera only | no odometry | green=current fit, yellow=prediction | not calibrated meters',(10,713),0,.48,(170,210,255),1)
                for _ in range(max(1,ticks[i+1]-ticks[i])):writer.stdin.write(canvas.tobytes())
                while targets and (pts[i]>=targets[0] or i==len(pts)-1):
                    target=targets.pop(0);cv2.imwrite(str(out/f'review_{target:02d}s.jpg'),canvas);cv2.imwrite(str(out/f'raw_{target:02d}s.jpg'),frame)
                    samples.append(dict(requested_seconds=target,actual_seconds=float(pts[i]),frame=i,file=f'review_{target:02d}s.jpg'))
                records.append(dict(frame=i,seconds=float(pts[i]),mask_off=runs[0],mask_on=runs[1],processing_seconds=time.perf_counter()-start))
                if i%100==0:print(i,round(pts[i],2),'off/on',len(lanes[0]),len(lanes[1]),flush=True)
                i+=1;ok,frame=cap.read()
            if i!=len(pts):raise ValueError(f'Decoded {i} frames, expected {len(pts)}')
        finally:
            writer.stdin.close();code=writer.wait();cap.release()
        if code:raise RuntimeError('Video encoder failed')
    (out/'predictions.json').write_text(json.dumps(records,indent=2))
    (out/'review_samples.json').write_text(json.dumps(samples,indent=2))
    summary=dict(frames=len(records),duration_seconds=float(end),input_size=[w,h],calibration_verified=False,ground_truth_available=False)
    for variant in ['mask_off','mask_on']:
        a=[r[variant] for r in records]
        summary[variant]=dict(frames_any_lane=sum(bool(r['lanes']) for r in a),frames_two_lanes=sum(len(r['lanes'])==2 for r in a),frames_no_lane=sum(not r['lanes'] for r in a),frames_with_prediction=sum(any(l['reprojected'] for l in r['lanes']) for r in a),frames_flow_registration=sum(r['registration']=='optical_flow' for r in a),unknown_style_lanes=sum(l['style']=='unknown' for r in a for l in r['lanes']))
    summary['note']='Output counts are not correctness or recall. Current-fit curves can include short extrapolation. Actual camera calibration and manual/official GT are unavailable.'
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
