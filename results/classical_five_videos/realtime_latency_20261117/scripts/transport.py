"""ROS 2 transport time of a bgr8 image of a given size (publisher and subscriber in two processes, reliable,
depth 10, 30 Hz, 300 images): publish -> subscriber callback (raw CDR), CLOCK_MONOTONIC.
usage: transport.py WIDTH HEIGHT"""
import sys,time,json,subprocess,os,numpy as np,rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,ReliabilityPolicy
from sensor_msgs.msg import Image
w,h=int(sys.argv[1]),int(sys.argv[2]);qos=QoSProfile(depth=10,reliability=ReliabilityPolicy.RELIABLE)
if len(sys.argv)>3 and sys.argv[3]=='sub':
    rclpy.init();n=Node('transport_sub');got={}
    def cb(raw):got[int.from_bytes(raw[4:8],'little')*10**9+int.from_bytes(raw[8:12],'little')]=time.perf_counter_ns()
    n.create_subscription(Image,'/transport_test',cb,qos,raw=True);end=time.monotonic()+30
    while time.monotonic()<end and len(got)<300:rclpy.spin_once(n,timeout_sec=.05)
    print(json.dumps(got));sys.exit()
sub=subprocess.Popen([sys.executable,__file__,str(w),str(h),'sub'],stdout=subprocess.PIPE)
rclpy.init();n=Node('transport_pub');pub=n.create_publisher(Image,'/transport_test',qos)
end=time.monotonic()+10
while pub.get_subscription_count()<1 and time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.05)
msg=Image();msg.height=h;msg.width=w;msg.encoding='bgr8';msg.step=w*3;msg.data=np.random.randint(0,255,w*h*3,np.uint8).tobytes()
sent={};t0=time.perf_counter()
for i in range(300):
    while time.perf_counter()<t0+i/30:pass
    msg.header.stamp.sec=i;msg.header.stamp.nanosec=0;sent[i*10**9]=time.perf_counter_ns();pub.publish(msg)
got={int(k):v for k,v in json.loads(sub.communicate()[0]).items()}
d=np.array([(got[k]-v)/1e6 for k,v in sent.items() if k in got])
print(f'{w}x{h} ({w*h*3/1e6:.2f} MB): received {len(d)}/300, mean {d.mean():.2f} p95 {np.percentile(d,95):.2f} max {d.max():.2f} ms')
