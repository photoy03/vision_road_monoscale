#!/usr/bin/env python3
"""Frozen-baseline vs staged CARLA paint validation, no learned models."""
import argparse,gzip,hashlib,json,subprocess,sys,time,shutil
from pathlib import Path
from dataclasses import asdict,replace
import cv2,numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src/vision_road'))
from vision_road.core import Settings,Projector,LaneMap
from vision_road.classical import PaintEvidenceDetector
from vision_road.carla import CarlaDetector
from vision_road.temporal import TemporalDetector


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--video',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--no-video',action='store_true');args=ap.parse_args()
    cv2.setNumThreads(2);out=args.output;out.mkdir(parents=True,exist_ok=True)
    old=json.loads((args.baseline/'predictions.json').read_text());run=json.loads((args.baseline/'run_config.json').read_text())
    if hashlib.sha256(args.video.read_bytes()).hexdigest()!=run['input_sha256']:raise ValueError('Baseline video mismatch')
    cfg=yaml.safe_load((ROOT/'src/vision_road/config/carla.yaml').read_text())['vision_road']['ros__parameters'];c=Settings(**{k:v for k,v in cfg.items() if k in Settings.__dataclass_fields__})
    projector=Projector(args.baseline/'APPROXIMATE_camera.npz',run['rig'],c)
    detectors={}
    for stage in [1,2,3]:
        settings=replace(c,carla_stage=stage,temporal_enabled=stage==3,style_temporal_enabled=stage==3)
        d=TemporalDetector(settings);d.detector=CarlaDetector(settings);detectors[f'stage{stage}']=d
    snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name in ['core.py','classical.py','temporal.py','carla.py','node.py','marking_style.py']:shutil.copy2(ROOT/'src/vision_road/vision_road'/name,snapshot/name)
    shutil.copy2(ROOT/'src/vision_road/config/carla.yaml',snapshot/'carla.yaml');shutil.copy2(__file__,snapshot/Path(__file__).name)
    (out/'run_config.json').write_text(json.dumps(dict(input=str(args.video),sha256=run['input_sha256'],settings=asdict(c),camera_assumptions=run['camera_assumptions'],calibration_verified=False,learning_used=False,stages='1: local ridge plus robust curves; 2: local paint fragments, graph and grid rejection; 3: stage2 with geometry tracking independent of marking kind'),indent=2,default=str))
    cap=cv2.VideoCapture(str(args.video));writer=None
    if not args.no_video:
        writer=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1280x720','-r','24','-i','pipe:0','-an','-c:v','libvpx-vp9','-deadline','realtime','-cpu-used','6','-crf','33','-b:v','0','-threads','2','-row-mt','1',str(out/'comparison.webm')],stdin=subprocess.PIPE,stderr=(out/'encode.log').open('w'))
    metrics={key:dict(any=0,both_sides=0,none=0,predicted_frames=0,observed_fragments=0,invalid_points=0,max_prediction_age=0.,times=[]) for key in detectors}
    samples=[];targets=list(range(0,61,5));frame_index=0
    with gzip.open(out/'predictions.jsonl.gz','wt') as file:
        while True:
            ok,im=cap.read()
            if not ok:break
            record=old[frame_index];t=record['seconds'];bev=projector.warp(im);valid=projector.valid.copy();valid[projector.my>=.86*im.shape[0]]=0;bev=cv2.bitwise_and(bev,bev,mask=valid)
            output={};final_overlay=None;finalmask=None;stages={}
            for key,d in detectors.items():
                before=time.perf_counter();lanes,binary=d.process(bev,valid,t,None);elapsed=time.perf_counter()-before;m=metrics[key];m['times'].append(elapsed)
                sides={l.side for l in lanes};m['any']+=bool(lanes);m['both_sides']+=len(sides)==2;m['none']+=not lanes;m['predicted_frames']+=any(l.reprojected for l in lanes);m['observed_fragments']+=sum(not l.reprojected for l in lanes)
                current=[]
                for lane in lanes:
                    px=d.pixel(lane.points);inside=(px[:,0]>=0)&(px[:,0]<valid.shape[1])&(px[:,1]>=0)&(px[:,1]<len(valid));ix=np.flatnonzero(inside);bad=int(np.count_nonzero(~inside))+int(np.count_nonzero(valid[px[ix,1].astype(int),px[ix,0].astype(int)]==0));m['invalid_points']+=bad
                    if lane.reprojected:m['max_prediction_age']=max(m['max_prediction_age'],t-lane.source_time)
                    xy,visible=projector.image_points(lane.points);current.append(dict(side=lane.side,points=xy[visible].tolist(),reprojected=lane.reprojected,source_time=lane.source_time,style=lane.style,track_key=lane.track_key))
                output[key]=current;stages[key]=dict(sides=sorted(sides),fragments=len(lanes),diagnostics=d.detector.diagnostics)
                if key=='stage3':
                    final_overlay=im.copy();finalmask=binary.copy()
                    for lane in current:
                        cv2.polylines(final_overlay,[np.rint(lane['points']).astype(np.int32)],False,(0,255,255) if lane['reprojected'] else (0,255,0),2)
            left=im.copy()
            for lane in record['mask_on']['lanes']:cv2.polylines(left,[np.rint(lane['points']).astype(np.int32)],False,(0,255,255) if lane['reprojected'] else (0,255,0),2)
            canvas=np.full((720,1280,3),20,np.uint8)
            for k,(title,image) in enumerate([('PREVIOUS',left),('CARLA / OBSERVED PAINT',final_overlay)]):
                canvas[:360,k*640:(k+1)*640]=cv2.resize(image,(640,360));cv2.rectangle(canvas,(640*k,0),(640*(k+1),35),(15,15,15),-1);cv2.putText(canvas,f'{title} | {t:.2f}s',(640*k+10,25),0,.63,(255,255,255),1)
            grid=detectors['stage3'].detector.grid_mask;marked=bev.copy();marked[grid>0]=(0,0,150)
            for k,(title,image) in enumerate([('BEV / RED: BOX MASK',marked),('CURRENT CANDIDATES',cv2.cvtColor(finalmask,cv2.COLOR_GRAY2BGR))]):
                width=round(image.shape[1]*300/image.shape[0]);x=k*320+(320-width)//2;canvas[400:700,x:x+width]=cv2.resize(image,(width,300),interpolation=cv2.INTER_NEAREST);cv2.putText(canvas,title,(k*320+8,385),0,.48,(230,230,230),1)
            for j,key in enumerate(detectors):
                q=stages[key];cv2.putText(canvas,f"{key}: {','.join(q['sides']) or 'none'} / {q['fragments']} fragments",(660,425+j*44),0,.6,(230,230,230),1)
            cv2.putText(canvas,'Green=current paint | yellow=short prediction',(660,580),0,.6,(130,230,160),1)
            cv2.putText(canvas,'Camera profile is provisional; no metric GT',(660,625),0,.55,(170,210,255),1)
            if writer:
                next_t=old[frame_index+1]['seconds'] if frame_index+1<len(old) else 60.333334
                repeats=round(next_t*24)-round(t*24)
                for _ in range(repeats):writer.stdin.write(canvas.tobytes())
            while targets and (t>=targets[0] or frame_index==len(old)-1):
                target=targets.pop(0);name=f'review_{target:02d}s.jpg';cv2.imwrite(str(out/name),canvas);samples.append(dict(seconds=t,frame=frame_index,file=name,stages=stages))
            file.write(json.dumps(dict(frame=frame_index,seconds=t,outputs=output),separators=(',',':'))+'\n')
            if frame_index%100==0:print(frame_index,round(t,2),{k:v['fragments'] for k,v in stages.items()},flush=True)
            frame_index+=1
    cap.release()
    if writer:writer.stdin.close();exitcode=writer.wait();assert exitcode==0
    assert frame_index==len(old)
    for key,m in metrics.items():
        times=m.pop('times');m['median_ms']=float(np.median(times)*1000);m['p95_ms']=float(np.quantile(times,.95)*1000)
    (out/'summary.json').write_text(json.dumps(dict(frames=frame_index,metrics=metrics,note='Counts are output activity, not correctness. Only the provided CARLA video, no independent ground truth.'),indent=2));(out/'review_samples.json').write_text(json.dumps(samples,indent=2))
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
