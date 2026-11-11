#!/usr/bin/env python3
"""Read-only SSD replay paced by simulation headers, with bounded I/O prefetch."""
import argparse,bisect,json,queue,sqlite3,threading,time
from pathlib import Path
import rclpy
from vision_road.zstd_cdr import cache_records
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image,CameraInfo
from geometry_msgs.msg import PoseWithCovarianceStamped


def stamp(m):return m.header.stamp.sec+m.header.stamp.nanosec*1e-9


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bag',type=Path,default=Path('/media/jaehun/38F7-677B/rosbag/maploop_30hz'))
    ap.add_argument('--cache',type=Path,default=Path(__file__).resolve().parents[1]/'data/maploop30hz_native.cdrz')
    ap.add_argument('--rate',type=float,default=1.)
    ap.add_argument('--start-frame',type=int,default=0)
    ap.add_argument('--max-frames',type=int,default=0)
    ap.add_argument('--prefetch',type=int,default=24)
    ap.add_argument('--timing-report',type=Path)
    a=ap.parse_args()
    if not 0<a.rate<=4 or min(a.start_frame,a.max_frames)<0 or not 1<=a.prefetch<=64:raise ValueError('Invalid replay limits')
    db=a.bag if a.bag.suffix=='.db3' else next(a.bag.glob('*.db3'))
    uri='file:'+str(db.resolve())+'?mode=ro&immutable=1'
    con=sqlite3.connect(uri,uri=True)
    topics={n:i for i,n in con.execute('select id,name from topics')}
    def query(topic):return con.execute('select data from messages where topic_id=? order by timestamp',(topics[topic],))
    poses=[deserialize_message(data,PoseWithCovarianceStamped) for data, in query('/carla/ground_truth/pose')]
    pts=[stamp(m) for m in poses]
    info=deserialize_message(next(query('/sensing/camera/dashcam/camera_info'))[0],CameraInfo)
    con.close()
    buffers=queue.Queue(maxsize=a.prefetch);ready=threading.Event();stop=threading.Event();errors=[]
    def produce():
        connection=sqlite3.connect(uri,uri=True)
        try:
            if a.cache.exists():
                stream=cache_records(a.cache,db,a.start_frame,a.max_frames)
            else:
                cursor=connection.execute('select data from messages where topic_id=? order by timestamp limit ? offset ?',
                    (topics['/sensing/camera/dashcam/image_raw'],a.max_frames or -1,a.start_frame))
                stream=(data for data, in cursor)
            count=0
            for data in stream:
                while not stop.is_set():
                    try:buffers.put(data,timeout=.1);break
                    except queue.Full:continue
                if stop.is_set():return
                count+=1
                if count==min(a.prefetch,a.max_frames or a.prefetch):ready.set()
        except Exception as exc:errors.append(str(exc))
        finally:
            ready.set()
            while not stop.is_set():
                try:buffers.put(None,timeout=.1);break
                except queue.Full:continue
            connection.close()
    reader=threading.Thread(target=produce,daemon=True);reader.start()
    rclpy.init();node=Node('maploop_header_replay')
    images=node.create_publisher(Image,'/sensing/camera/dashcam/image_raw',5)
    infos=node.create_publisher(CameraInfo,'/sensing/camera/dashcam/camera_info',5)
    truth=node.create_publisher(PoseWithCovarianceStamped,'/carla/ground_truth/pose',100)
    clock=node.create_publisher(Clock,'/clock',10)
    def pump(seconds):
        until=time.perf_counter()+seconds
        while rclpy.ok() and time.perf_counter()<until:rclpy.spin_once(node,timeout_sec=min(.01,max(0.,until-time.perf_counter())))
    timing=[]
    try:
        until=time.perf_counter()+20
        while images.get_subscription_count()<1 and time.perf_counter()<until:pump(.1)
        if images.get_subscription_count()<1:raise RuntimeError('Start maploop.launch.py before replay')
        while not ready.wait(.1):pump(.01)
        if errors:raise RuntimeError(errors[0])
        pump(1.)
        origin=None;wall_ns=None;pose_index=0
        while rclpy.ok():
            data=buffers.get()
            if data is None:break
            image=deserialize_message(data,Image);t=stamp(image)
            if origin is None:
                origin=t;pose_index=max(0,bisect.bisect_left(pts,t)-2)
                # Seed the evaluation adapter before starting the 30 Hz clock.
                while pose_index<len(poses) and pts[pose_index]<=t+.05:
                    truth.publish(poses[pose_index]);pose_index+=1
                pump(.05);wall_ns=time.perf_counter_ns()
            target_ns=wall_ns+round((t-origin)/a.rate*1e9)
            while rclpy.ok() and time.perf_counter_ns()<target_ns:pump(min(.01,(target_ns-time.perf_counter_ns())/1e9))
            if not rclpy.ok():break
            tick=Clock();tick.clock=image.header.stamp;clock.publish(tick)
            # Reference poses are available from the recorded bag, independently
            # of vision. A bounded lead keeps them queued before their image.
            while pose_index<len(poses) and pts[pose_index]<=t+.05:
                truth.publish(poses[pose_index]);pose_index+=1
            info.header.stamp=image.header.stamp;infos.publish(info)
            begin=time.perf_counter_ns();images.publish(image);finish=time.perf_counter_ns()
            timing.append(dict(stamp=t,scheduled_ns=target_ns,publish_ns=begin,publish_finish_ns=finish))
            if len(timing)%300==0:node.get_logger().info(f'{len(timing)} frames / {t-origin:.2f} simulation seconds')
        if errors:raise RuntimeError(errors[0])
        pump(.5);node.get_logger().info(f'Done: {len(timing)} recorded images; header-paced, bag unchanged')
    except KeyboardInterrupt:pass
    finally:
        stop.set();reader.join(timeout=2)
        if a.timing_report:
            a.timing_report.parent.mkdir(parents=True,exist_ok=True)
            a.timing_report.write_text(json.dumps(dict(rate=a.rate,prefetch_frames=a.prefetch,lossless_cache=str(a.cache) if a.cache.exists() else None,records=timing),separators=(',',':')))
        node.destroy_node();rclpy.try_shutdown()
if __name__=='__main__':main()
