#!/usr/bin/env python3
"""Record the actual ROS overlay topic, preserving header time through held frames."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
import cv2,numpy as np,rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from vision_lane_interfaces.msg import MarkingSegmentArray
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    frames=a.output/'frames';frames.mkdir(exist_ok=True)
    rclpy.init();node=Node('lane_overlay_recorder');bridge=CvBridge();captured=[];lanes=[]
    def on_image(msg):
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        image=bridge.imgmsg_to_cv2(msg,'bgr8');name=f'{len(captured):05d}.png'
        assert cv2.imwrite(str(frames/name),image,[cv2.IMWRITE_PNG_COMPRESSION,1])
        captured.append(dict(stamp=stamp,file=name,width=msg.width,height=msg.height))
    def on_lane(msg):lanes.append([msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,time.perf_counter_ns()])
    subscriptions=[node.create_subscription(Image,'/road/overlay',on_image,qos_profile_sensor_data),node.create_subscription(MarkingSegmentArray,'/parking/cpp/marking_segments',on_lane,10)]
    launch=replay=None;timing=a.output/'source_timing.json'
    try:
        with (a.output/'launch.log').open('w') as log,(a.output/'replay.log').open('w') as rep:
            launch=subprocess.Popen([str(ROOT/'run_maploop.sh')],stdout=log,stderr=log,start_new_session=True)
            deadline=time.perf_counter()+25
            while node.count_publishers('/road/overlay')<1 and time.perf_counter()<deadline:
                if launch.poll() is not None:raise RuntimeError('Launch exited')
                rclpy.spin_once(node,timeout_sec=.05)
            assert node.count_publishers('/road/overlay')
            end=time.perf_counter()+1
            while time.perf_counter()<end:rclpy.spin_once(node,timeout_sec=.05)
            replay=subprocess.Popen([str(ROOT/'replay_maploop.sh'),'--rate','1','--timing-report',str(timing)],stdout=rep,stderr=rep)
            deadline=time.perf_counter()+160
            while replay.poll() is None and time.perf_counter()<deadline:rclpy.spin_once(node,timeout_sec=.02)
            assert replay.poll()==0,'Replay failed or exceeded deadline'
            end=time.perf_counter()+3
            while time.perf_counter()<end:rclpy.spin_once(node,timeout_sec=.02)
        assert len(captured)>250 and lanes
        source=json.loads(timing.read_text())['records'];origin=source[0]['stamp'];finish=source[-1]['stamp']+1/30
        # Constant 30 fps container, holding each actual 5 Hz diagnostic frame.
        # No interpolated detections are created to fill dropped diagnostic frames.
        writer=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f"{captured[0]['width']}x{captured[0]['height']}",'-r','30','-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(a.output/'maploop_lane_overlay.mp4')],stdin=subprocess.PIPE)
        position=0;last=None;total=round((finish-origin)*30)
        for i in range(total):
            stamp=origin+i/30
            while position+1<len(captured) and captured[position+1]['stamp']<=stamp+1e-7:position+=1
            if stamp+1e-7<captured[0]['stamp']:
                image=np.zeros((captured[0]['height'],captured[0]['width'],3),np.uint8)
            else:
                if last is None or last[0]!=position:last=(position,cv2.imread(str(frames/captured[position]['file'])))
                image=last[1]
            writer.stdin.write(image.tobytes())
        writer.stdin.close();assert writer.wait()==0
        report=dict(source_frames=len(source),lane_messages=len(lanes),recorded_overlay_images=len(captured),input_hz=(len(source)-1)*1e9/(source[-1]['publish_ns']-source[0]['publish_ns']),duration_seconds=total/30,encoded_frames=total,encoded_fps=30,diagnostic_update_hz=5,resolution=[captured[0]['width'],captured[0]['height']],actual_ros_topic='/road/overlay',held_frames=True,frames=captured,note='Actual live node overlay, not an offline redraw. The diagnostic topic is 5Hz; the MP4 holds observations in a 30fps container. CARLA recorded reference poses, not estimated odometry.')
        (a.output/'recording.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='frames'},indent=2))
        for target in [10,30,50]:
            closest=min(captured,key=lambda row:abs(row['stamp']-origin-target));image=cv2.imread(str(frames/closest['file']));cv2.imwrite(str(a.output/f'preview_{target:02d}s.jpg'),image)
    finally:
        if replay is not None and replay.poll() is None:replay.terminate();replay.wait(timeout=5)
        if launch is not None and launch.poll() is None:
            launch.send_signal(signal.SIGINT)
            try:launch.wait(timeout=12)
            except subprocess.TimeoutExpired:os.killpg(launch.pid,signal.SIGTERM);launch.wait(timeout=5)
        node.destroy_node();rclpy.try_shutdown()
if __name__=='__main__':main()
