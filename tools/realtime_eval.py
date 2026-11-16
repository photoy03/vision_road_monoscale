#!/usr/bin/env python3
"""Real-time check of one recorded video: road_node + lane_display_filter (+ HERO HUD) with the
recording replayed at 30 Hz by its image header stamps (tools/replay_maploop.py).

All wall times are CLOCK_MONOTONIC (time.perf_counter_ns), so per frame (header stamp) the stages line up:
  publish   replay publishes the image                       (replay timing report)
  receive   road_node image callback runs                    (road_node performance report)
  start     processing starts (after the job queue)          (performance report)
  finish    lane topics published                            (performance report; with output_thread the output
            thread's end, and 'computed' is the detection hand-off)
  boundary  /road/lane_boundary_markers arrives here         (this process)
  display   /road/lane_display_markers arrives here          (this process; the HUD subscribes the same topic)

usage: realtime_eval.py --video maploop|town04|town05 --output DIR [--frames N] [--no-hud] [--record]
                        [--launch-arg name:=value ...]
"""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
import numpy as np,rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,ReliabilityPolicy
from visualization_msgs.msg import MarkerArray
ROOT=Path(__file__).resolve().parents[1]
HUD=Path('/home/jaehun/Desktop/hero-hud-main')
DISK=Path('/media/jaehun/38F7-677B')
VIDEOS=dict(maploop=(DISK/'rosbag/maploop_30hz',ROOT/'data/maploop30hz_native.cdrz'),
            town04=(DISK/'rosbag/town04_30hz',DISK/'vision-road-cache/town04_30hz_native.cdrz'),
            town05=(DISK/'rosbag/town05_30hz',DISK/'vision-road-cache/town05_30hz_native.cdrz'))


def key(stamp):return stamp.sec*1000000000+stamp.nanosec


def stats(values):
    if not values:return None
    v=np.asarray(values,float)
    return dict(n=len(v),mean=round(float(v.mean()),2),median=round(float(np.median(v)),2),p95=round(float(np.quantile(v,.95)),2),
                p99=round(float(np.quantile(v,.99)),2),max=round(float(v.max()),2))


def compose(output,overlays):
    """Camera overlay (5 Hz, left) and HUD screen (10 Hz, right) side by side by capture wall time, 10 fps."""
    import cv2
    hud=sorted((output/'hud_frames').glob('frame_*.png'))
    if not overlays or not hud:return
    hs=np.array([p.stat().st_mtime_ns for p in hud]);os_=np.array([t for t,_ in overlays])
    first,last=max(hs[0],os_[0]),min(hs[-1],os_[-1])
    writer=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1600x540','-r','10','-i','pipe:0','-c:v','libx264',
                             '-crf','22','-pix_fmt','yuv420p',str(output/'road_and_hud.mp4')],stdin=subprocess.PIPE)
    cache={}
    for now in range(int(first),int(last),100_000_000):
        i=max(0,int(np.searchsorted(os_,now,'right'))-1);j=max(0,int(np.searchsorted(hs,now,'right'))-1)
        if ('o',i) not in cache:cache.clear();cache[('o',i)]=cv2.resize(cv2.imread(str(overlays[i][1])),(960,540))
        frame=np.zeros((540,1600,3),np.uint8);frame[:,:960]=cache[('o',i)]
        frame[70:470,960:]=cv2.resize(cv2.imread(str(hud[j])),(640,400),interpolation=cv2.INTER_AREA)
        writer.stdin.write(frame.tobytes())
    writer.stdin.close();writer.wait()


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video',choices=sorted(VIDEOS),required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--frames',type=int,default=0,help='0 = whole recording');ap.add_argument('--start-frame',type=int,default=0)
    ap.add_argument('--no-hud',action='store_true');ap.add_argument('--record',action='store_true',help='HUD screen recording (frames to OUTPUT/hud_frames)')
    ap.add_argument('--launch-arg',nargs='*',default=[],help='extra maploop.launch.py arguments, name:=value')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);bag,cache=VIDEOS[a.video]
    rclpy.init();node=Node('road_realtime_eval');arrivals={'boundary':{},'display':{},'paint':{}};content={'boundary':{},'display':{}}
    def receiver(name):
        def receive(msg):
            if not msg.markers:return
            k=key(msg.markers[0].header.stamp);arrivals[name].setdefault(k,time.perf_counter_ns())
            if name in content:  # line strips as published (side, predicted, points)
                content[name][k]=[(m.ns.split('/')[-1],m.color.r>.5,[(round(p.x,4),round(p.y,4)) for p in m.points]) for m in msg.markers if m.action==0 and len(m.points)>=2]
        return receive
    reliable=QoSProfile(depth=10,reliability=ReliabilityPolicy.RELIABLE)
    overlays=[]
    if a.record:
        import cv2
        from cv_bridge import CvBridge
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
        bridge=CvBridge();(a.output/'overlay_frames').mkdir(exist_ok=True)
        def on_overlay(msg):
            path=a.output/'overlay_frames'/f'{len(overlays):05d}.jpg'
            cv2.imwrite(str(path),bridge.imgmsg_to_cv2(msg,'bgr8'),[cv2.IMWRITE_JPEG_QUALITY,85]);overlays.append((time.time_ns(),path))
        node.create_subscription(Image,'/road/overlay',on_overlay,qos_profile_sensor_data)
    subs=[node.create_subscription(MarkerArray,t,receiver(n),reliable) for n,t in
          (('boundary','/road/lane_boundary_markers'),('display','/road/lane_display_markers'),('paint','/road/lane_markers'))]
    procs=[];node_timing=a.output/'node_timing.json';source_timing=a.output/'source_timing.json'
    logs={n:(a.output/f'{n}.log').open('w') for n in ('road','hud','replay')}
    try:
        cmd=['bash','-c','source /opt/ros/humble/setup.bash && source "$0/install/setup.bash" && exec ros2 launch vision_road maploop.launch.py "$@"',
             str(ROOT),'performance_report:='+str(node_timing)]+a.launch_arg
        road=subprocess.Popen(cmd,stdout=logs['road'],stderr=subprocess.STDOUT,start_new_session=True);procs.append(road)
        deadline=time.monotonic()+30
        while node.count_publishers('/road/lane_display_markers')<1 and time.monotonic()<deadline:
            if road.poll() is not None:raise RuntimeError('road launch exited; see road.log')
            rclpy.spin_once(node,timeout_sec=.05)
        if not node.count_publishers('/road/lane_display_markers'):raise RuntimeError('lane_display_filter did not start')
        hud=None
        if not a.no_hud:
            env=os.environ.copy();env.pop('QT_QPA_PLATFORM_PLUGIN_PATH',None);env.pop('QT_QPA_FONTDIR',None)
            if a.record:
                frames=a.frames or 1810
                (a.output/'hud_frames').mkdir(exist_ok=True)
                env.update(HERO_HUD_RECORD=str(a.output/'hud_frames'),HERO_HUD_RECORD_FPS='10',HERO_HUD_RECORD_SECONDS=str(frames//30+8),HERO_HUD_SCREENSHOT_DELAY_MS='2500')
            hud=subprocess.Popen(['bash','-c','source /opt/ros/humble/setup.bash && source "$0/install/setup.bash" && exec "$0/install/hero_hud/lib/hero_hud/hud_node" "$@"',
                                  str(HUD),'--ros-args','--params-file',str(HUD/'config/hud.yaml'),'-p','display.mirror_horizontal:=false','-p','road.lane_only:=true'],
                                 env=env,stdout=logs['hud'],stderr=subprocess.STDOUT,start_new_session=True);procs.append(hud)
            end=time.monotonic()+3
            while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.05)
            if hud.poll() is not None:raise RuntimeError('HUD exited; see hud.log')
        replay_cmd=['bash','-c','source /opt/ros/humble/setup.bash && source "$0/install/setup.bash" && exec /usr/bin/python3 "$0/tools/replay_maploop.py" "$@"',str(ROOT),
                    '--bag',str(bag),'--cache',str(cache),'--start-frame',str(a.start_frame),'--max-frames',str(a.frames),'--rate','1','--timing-report',str(source_timing)]
        replay=subprocess.Popen(replay_cmd,stdout=logs['replay'],stderr=subprocess.STDOUT,start_new_session=True);procs.append(replay)
        deadline=time.monotonic()+(a.frames or 1810)/30+120
        while replay.poll() is None and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.01)
        if replay.poll()!=0:raise RuntimeError('replay failed or timed out; see replay.log')
        end=time.monotonic()+3
        while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.01)
        hud_ok=hud is None or hud.poll() is None
        if hud is not None and a.record:
            try:hud.wait(timeout=30)
            except subprocess.TimeoutExpired:pass
        road.send_signal(signal.SIGINT);road.wait(timeout=20)
    finally:
        for p in reversed(procs):
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGINT)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
        for f in logs.values():f.close()
        node.destroy_node();rclpy.try_shutdown()
    if a.record:compose(a.output,overlays)
    source=json.loads(source_timing.read_text())['records'];perf=json.loads(node_timing.read_text())
    rows={round(r['stamp']*1e9):r for r in perf['records']}
    published={round(r['stamp']*1e9):r['publish_ns'] for r in source};ordered=sorted(published)
    frames=[];t0=ordered[0]
    for k in ordered:
        r=rows.get(k);f=dict(t=round((k-t0)/1e9,3),publish=published[k])
        if r is not None:
            f.update(receive=r.get('receive_ns'),start=r['arrival_ns'],finish=r['finish_ns'],phases=r.get('phases',{}))
            if r.get('output_finish_ns') is not None:  # output_thread: outputs published after the detection hand-off
                f.update(computed=r['finish_ns'],output_start=r['output_start_ns'],finish=r['output_finish_ns'])
                f['phases'].update({'out.'+k:v for k,v in r.get('output_phases',{}).items()})
        for n in ('boundary','display'):
            if k in arrivals[n]:f[n]=arrivals[n][k]
            if k in content[n]:f[n+'_lines']=content[n][k]
        frames.append(f)
    ms=lambda f,a_,b_:(f[b_]-f[a_])/1e6 if f.get(a_) is not None and f.get(b_) is not None else None
    def series(a_,b_):return [v for v in (ms(f,a_,b_) for f in frames) if v is not None]
    total=series('publish','display')
    # Latency drift: mean of the last 10% of frames against the first 10%.
    disp=[(f['t'],ms(f,'publish','display')) for f in frames if f.get('display')]
    tenth=max(1,len(disp)//10)
    drift=round(float(np.mean([v for _,v in disp[-tenth:]])-np.mean([v for _,v in disp[:tenth]])),2) if disp else None
    gaps=np.diff(sorted(arrivals['display'].values()))/1e6 if len(arrivals['display'])>1 else np.array([])
    phases={}
    for f in frames:
        for name,v in f.get('phases',{}).items():phases.setdefault(name,[]).append(v)
    n=len(frames);processed=sum(1 for f in frames if f.get('finish') is not None)
    sides=[l for f in frames for l in f.get('display_lines',[])]
    near=[min(p[0] for p in l[2]) for l in sides]
    summary=dict(video=a.video,frames=n,input_hz=round((n-1)*1e9/(source[-1]['publish_ns']-source[0]['publish_ns']),3) if n>1 else None,
        processed=processed,processed_ratio=round(processed/n,4),
        boundary_received=sum(1 for f in frames if f.get('boundary')),display_received=sum(1 for f in frames if f.get('display')),
        unprocessed_received=len(perf.get('unprocessed',[])),hud=not a.no_hud,hud_recording=a.record,hud_alive_at_end=hud_ok,
        latency_ms=dict(publish_to_display=stats(total),publish_to_receive=stats(series('publish','receive')),receive_to_start=stats(series('receive','start')),
                        start_to_finish=stats(series('start','finish')),start_to_computed=stats(series('start','computed')),
                        computed_to_output_start=stats(series('computed','output_start')),finish_to_boundary=stats(series('finish','boundary')),boundary_to_display=stats(series('boundary','display'))),
        latency_drift_ms_last_vs_first_tenth=drift,display_lines=len(sides),display_lines_from_car=round(float(np.mean(np.array(near)<=.05)),4) if near else None,display_interval_ms=stats(gaps.tolist()),display_gaps_over_100ms=int(np.sum(gaps>100)),
        phases_ms={k:stats(v) for k,v in phases.items()},launch_args=a.launch_arg,
        note='Header-paced replay of the recorded bag (lossless CDR cache); CARLA reference pose as odometry. Display arrival is taken by this '
             'evaluation process, which subscribes /road/lane_display_markers like the HUD does.')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=1))
    with (a.output/'frames.json').open('w') as file:json.dump(frames,file,separators=(',',':'))
    print(json.dumps({k:summary[k] for k in ('video','frames','processed_ratio','display_received','latency_drift_ms_last_vs_first_tenth','display_gaps_over_100ms')}))
    print(json.dumps(summary['latency_ms'],indent=1))


if __name__=='__main__':main()
