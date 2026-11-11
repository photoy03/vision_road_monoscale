#!/usr/bin/env python3
"""Measure real input pacing, delivery and wall latency; never infer Hz from headers."""
import argparse,json,os,signal,subprocess,time,gzip
from pathlib import Path
import numpy as np,rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_lane_interfaces.msg import MarkingSegmentArray,LaneStyleArray
from visualization_msgs.msg import MarkerArray
ROOT=Path(__file__).resolve().parents[1]

def key(stamp):return stamp.sec*1000000000+stamp.nanosec

def stats(values):
    return dict(median=float(np.median(values)),p95=float(np.quantile(values,.95)),p99=float(np.quantile(values,.99)),max=float(max(values))) if values else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--frames',type=int,default=1807);ap.add_argument('--start-frame',type=int,default=0);ap.add_argument('--geometry-input',type=Path);ap.add_argument('--debug-hz',type=float,default=5.);ap.add_argument('--mpp',type=float,default=.07);ap.add_argument('--display-error',type=float,default=.025);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    wanted={};quality={}
    if a.geometry_input:
        from vision_road.core import Settings,Projector
        import yaml
        labels=json.loads((a.geometry_input.parent/'labels/labels.json').read_text());ids={r['frame'] for r in labels['positives']+labels['negatives']}
        frames=json.loads((a.geometry_input/'frames.json').read_text());wanted={round(r['stamp']*1e9):r['frame'] for r in frames if r['frame'] in ids}
        projection=Projector(a.geometry_input/'dashcam.npz',yaml.safe_load((a.geometry_input/'rig.yaml').read_text()),Settings());projection.initialize(960,540)
    rclpy.init();reader=Node('maploop_rate_measurement');observations={k:[] for k in ['local','styles','paint_markers','boundary_markers','overlay','binary','bev']};valid_pose=0;nonempty=0
    def callback(name):
        def receive(msg):
            nonlocal valid_pose,nonempty
            header=msg.markers[0].header if name.endswith('markers') else msg.header
            observations[name].append([key(header.stamp),time.perf_counter_ns()])
            if name=='local':
                valid_pose+=msg.pose_valid;nonempty+=bool(msg.segments)
                if key(header.stamp) in wanted:
                    paint=[]
                    if msg.segments:
                        points=np.array([p for seg in msg.segments for p in [(seg.start.x,seg.start.y),(seg.end.x,seg.end.y)]])
                        pixels,_=projection.image_points(points)
                        paint=[dict(side=seg.family,points=pixels[2*j:2*j+2].tolist(),reprojected=seg.reprojected) for j,seg in enumerate(msg.segments)]
                    quality[wanted[key(header.stamp)]]=paint
        return receive
    subs=[]
    for name,typ,topic,qos in [
        ('local',MarkingSegmentArray,'/parking/cpp/marking_segments',10),('styles',LaneStyleArray,'/road/lane_styles',10),
        ('paint_markers',MarkerArray,'/road/lane_markers',10),('boundary_markers',MarkerArray,'/road/lane_boundary_markers',10),
        ('overlay',Image,'/road/overlay',qos_profile_sensor_data),('binary',Image,'/parking/cpp/binary',qos_profile_sensor_data),('bev',Image,'/parking/cpp/vector_bev',qos_profile_sensor_data)]:
        subs.append(reader.create_subscription(typ,topic,callback(name),qos))
    launch=replay=None;node_timing=a.output/'node_timing.json';source_timing=a.output/'source_timing.json'
    try:
        with (a.output/'launch.log').open('w') as log,(a.output/'replay.log').open('w') as rep:
            launch=subprocess.Popen([str(ROOT/'run_maploop.sh'),'performance_report:='+str(node_timing),'display_simplify_error_m:='+str(a.display_error),'bev_mpp:='+str(a.mpp),'debug_publish_hz:='+str(a.debug_hz)],stdout=log,stderr=log,start_new_session=True)
            deadline=time.perf_counter()+25
            while reader.count_publishers('/parking/cpp/marking_segments')<1 and time.perf_counter()<deadline:
                if launch.poll() is not None:raise RuntimeError('Launch exited; see launch.log')
                rclpy.spin_once(reader,timeout_sec=.05)
            assert reader.count_publishers('/parking/cpp/marking_segments')
            replay=subprocess.Popen([str(ROOT/'replay_maploop.sh'),'--start-frame',str(a.start_frame),'--max-frames',str(a.frames),'--rate','1','--timing-report',str(source_timing)],stdout=rep,stderr=rep)
            deadline=time.perf_counter()+a.frames/30+90
            while replay.poll() is None and time.perf_counter()<deadline:rclpy.spin_once(reader,timeout_sec=.02)
            assert replay.poll()==0,'Replay failed or timed out'
            # Allow the bounded receive queue to drain, and measure its latency.
            end=time.perf_counter()+5
            while time.perf_counter()<end:rclpy.spin_once(reader,timeout_sec=.02)
            launch.send_signal(signal.SIGINT);launch.wait(timeout=12)
        source=json.loads(source_timing.read_text())['records'];processing=json.loads(node_timing.read_text())['records']
        pub={round(row['stamp']*1e9):row['publish_ns'] for row in source};keys=set(pub);delivery={};latency={};missing={}
        for name,records in observations.items():
            current=[(stamp,wall) for stamp,wall in records if stamp in keys];received={stamp for stamp,_ in current};missing[name]=len(keys-received);delivery[name]=len(current)
            latency[name]=stats([(wall-pub[stamp])/1e6 for stamp,wall in current])
        source_hz=(len(source)-1)*1e9/(source[-1]['publish_ns']-source[0]['publish_ns'])
        local=observations['local'];output_hz=(len(local)-1)*1e9/(local[-1][1]-local[0][1]) if len(local)>1 else 0
        service=[(row['finish_ns']-row['arrival_ns'])/1e6 for row in processing]
        throughput_pass=(len(source)==a.frames and 29.7<=source_hz<=30.3 and all(missing[k]==0 for k in ['local','styles','paint_markers','boundary_markers']) and latency['local']['p95']<100 and latency['local']['max']<250)
        result=dict(requested_frames=a.frames,input_frames=len(source),input_hz=source_hz,output_hz=output_hz,received=delivery,missing=missing,pose_valid=valid_pose,nonempty=nonempty,callback_ms=stats(service),callback_over_33ms=sum(v>1000/30 for v in service),latency_ms=latency,source_lateness_ms=stats([(r['publish_ns']-r['scheduled_ns'])/1e6 for r in source]),throughput_pass=throughput_pass,all_callbacks_under_33ms=all(v<=1000/30 for v in service),display_error_m=a.display_error,mpp=a.mpp,debug_hz=a.debug_hz,expected_debug_frames=sum('overlay' in row.get('phases',{}) for row in processing),note='CPU ROS pipeline, native 1920x1080 bag images, debug images enabled and subscribed at the configured diagnostic rate; structured outputs are not throttled. Throughput criterion: source 29.7-30.3Hz, no missing structured lane messages, local p95<100ms/max<250ms. Hard 33.3ms deadline reported separately. CARLA recorded reference poses; not estimated odometry.')
        (a.output/'summary.json').write_text(json.dumps(result,indent=2));(a.output/'arrivals.json').write_text(json.dumps(observations,separators=(',',':')));print(json.dumps(result,indent=2))
        if a.geometry_input:
            golden={}
            with gzip.open(a.geometry_input.parent/'final/predictions.jsonl.gz','rt') as file:
                for line in file:
                    row=json.loads(line)
                    if row['frame'] in wanted.values():golden[row['frame']]=row['outputs']['filtered']['paint']
            with gzip.open(a.output/'quality_predictions.jsonl.gz','wt') as file:
                for frame in sorted(wanted.values()):file.write(json.dumps(dict(frame=frame,outputs=dict(previous=dict(paint=golden[frame]),filtered=dict(paint=quality.get(frame,[])))))+'\n')
    finally:
        if replay is not None and replay.poll() is None:replay.terminate();replay.wait(timeout=5)
        if launch is not None and launch.poll() is None:
            launch.send_signal(signal.SIGINT)
            try:launch.wait(timeout=12)
            except subprocess.TimeoutExpired:os.killpg(launch.pid,signal.SIGTERM);launch.wait(timeout=5)
        reader.destroy_node();rclpy.try_shutdown()
if __name__=='__main__':main()
