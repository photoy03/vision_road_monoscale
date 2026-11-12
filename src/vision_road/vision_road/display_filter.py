"""Display-only smoothing of lane boundaries for the HUD.

road_node publishes boundaries tuned not to lose a lane; a head-up display
needs lines that do not shake or blink.  This node sits between the two: it
confirms a side M-of-N, keeps one quadratic per side, moves it with the vehicle
odometry, smooths it with a 1-euro filter, rejects single-frame jumps, holds a
lost side briefly in the predicted colour and only switches solid/dashed after
the style has been stable.
The detector topics are left untouched.
"""
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

# The quadratic y=a+b·x+c·x² is carried as its lateral value at these forward
# stations: the same three degrees of freedom, but all in metres, so one set of
# 1-euro parameters fits all three.
STATIONS=np.array([5.,10.,15.])
_V=np.vander(STATIONS,3,increasing=True)


@dataclass
class FilterSettings:
    confirm_window:int=8        # M-of-N confirmation: N frames
    confirm_hits:int=5          # ... of which M observed
    start_span_m:float=6.       # an observation must cover this to start a display
    update_span_m:float=3.      # shorter observations are ignored
    max_start_m:float=12.       # so are ones that begin farther out (a 25-35 m piece says little about 3 m)
    min_cutoff_hz:float=.3      # 1-euro: smoothing when the line is still
    beta:float=.5               # 1-euro: speed gain (raise if curves lag); low because motion is compensated
    d_cutoff_hz:float=1.
    gate_m:float=.30            # jump at 10 m treated as an outlier
    gate_frames:int=3           # consecutive consistent outliers accepted as a real change
    near_m:float=3.
    far_m:float=20.
    step_m:float=.5
    extend_m:float=3.           # draw at most this far past the observed end
    hold_s:float=1.             # keep a lost side this long, in the predicted colour
    odom_timeout_s:float=.2     # without vehicle motion for this long ...
    odom_lost_hold_s:float=.2   # ... a lost side is kept only this long (it cannot be moved)
    style_stable_s:float=.5
    restart_s:float=.5          # a stamp this far behind the last one means the replay restarted
    style_memory_s:float=10.    # a side that comes back within this keeps its solid/dashed (gaps of 5-6 s seen)
    dashed_min_confidence:float=.6


def fit(points,ridge=1e-3):
    """Quadratic through a polyline in base_link (x forward, y left), returned as station values.

    The input is usually a simplified polyline with a handful of vertices, so it
    is resampled along x first.  Short spans get a stronger pull toward zero
    curvature so a 3 m piece does not bend the whole display."""
    p=np.asarray(points,float);p=p[np.argsort(p[:,0])]
    x=np.arange(p[0,0],p[-1,0]+1e-6,.5)
    if len(x)<2:x=p[[0,-1],0]
    y=np.interp(x,p[:,0],p[:,1]);span=float(x[-1]-x[0])
    A=np.column_stack([np.ones_like(x),x,x*x])
    lam=len(x)*float(np.mean(x*x))**2*(ridge if span>=8 else 10.)
    coeff=np.linalg.solve(A.T@A+np.diag([0.,0.,lam]),A.T@y)
    return _V@coeff,float(x[0]),float(x[-1])


def evaluate(stations,x):
    coeff=np.linalg.solve(_V,stations)
    return coeff[0]+coeff[1]*x+coeff[2]*x*x


def relative(previous,pose):
    """Motion (dx, dy, dyaw) from one world pose (x, y, yaw) to the next, in the first's frame."""
    c,s=math.cos(previous[2]),math.sin(previous[2]);wx,wy=pose[0]-previous[0],pose[1]-previous[1]
    return (c*wx+s*wy,-s*wx+c*wy,math.atan2(math.sin(pose[2]-previous[2]),math.cos(pose[2]-previous[2])))


def shift(stations,motion):
    """Carry a curve into the next base_link: motion (dx, dy, dyaw) is the vehicle's
    move since the last update, expressed in the previous base_link."""
    dx,dy,dyaw=motion;x=np.arange(0.,25.01,.5)
    c,s=math.cos(dyaw),math.sin(dyaw);px=x-dx;py=evaluate(stations,x)-dy
    nx=c*px+s*py;ny=-s*px+c*py
    return _V@np.polyfit(nx,ny,2)[::-1]


class OneEuro:
    """Casiez et al., CHI 2012: low cutoff when still, rises with the filtered speed."""

    def __init__(self,min_cutoff,beta,d_cutoff):
        self.min_cutoff=min_cutoff;self.beta=beta;self.d_cutoff=d_cutoff;self.reset()

    def reset(self,x=None,t=None):
        self.x=None if x is None else np.array(x,float);self.t=t
        self.dx=None if x is None else np.zeros_like(self.x)

    @staticmethod
    def alpha(cutoff,dt):
        return 1/(1+1/(2*math.pi*cutoff*dt))

    def __call__(self,x,t):
        x=np.asarray(x,float)
        if self.x is None or t<=self.t:
            if self.x is None:self.reset(x,t)
            return self.x
        dt=t-self.t
        self.dx=self.dx+self.alpha(self.d_cutoff,dt)*((x-self.x)/dt-self.dx)
        self.x=self.x+self.alpha(self.min_cutoff+self.beta*np.abs(self.dx),dt)*(x-self.x)
        self.t=t
        return self.x


class SideTrack:
    def __init__(self,s:FilterSettings):
        self.s=s;self.history=deque(maxlen=s.confirm_window)
        self.filter=OneEuro(s.min_cutoff_hz,s.beta,s.d_cutoff_hz)
        self.shown=False;self.predicted=False;self.last_observed=-math.inf;self.far=None;self.pending=[]
        self.style=None;self.candidate=None;self.candidate_since=None

    @property
    def value(self):return self.filter.x

    def move(self,motion):
        if self.value is not None:self.filter.x=shift(self.value,motion)
        self.pending=[shift(p,motion) for p in self.pending]
        if self.far is not None:self.far-=motion[0]

    def drop(self):
        # The style survives a short loss: style evidence is sparse (a dashed line in a
        # curve is often UNKNOWN), and re-learning it showed the line solid on return.
        self.shown=False;self.filter.reset();self.far=None;self.pending=[];self.history.clear()

    def take(self,stations,t):
        """Gate a measurement at 10 m; returns False when it was set aside as an outlier."""
        if self.value is None:
            self.filter.reset(stations,t);return True
        if abs(stations[1]-self.value[1])<=self.s.gate_m:
            self.pending=[];self.filter(stations,t);return True
        self.pending.append(stations)
        recent=[p[1] for p in self.pending[-self.s.gate_frames:]]
        if len(recent)>=self.s.gate_frames and max(recent)-min(recent)<=self.s.gate_m:
            # A sustained change (new line, lane change): jump to it rather than slide.
            self.filter.reset(stations,t);self.pending=[];return True
        return False

    def update(self,t,observed,hold_s):
        s=self.s;obs=None;accepted=False
        if observed is not None:
            stations,near,far=fit(observed)
            if far-near>=s.update_span_m and near<=s.max_start_m:obs=(stations,near,far)
        self.history.append(obs is not None)
        if obs is not None:
            stations,_,far=obs
            accepted=self.take(stations,t)
            if accepted:
                # The drawn end grows quickly and shrinks slowly, so its length does not flicker.
                self.far=far if self.far is None else self.far+(.5 if far>self.far else .1)*(far-self.far)
            self.last_observed=t
        # Only a measurement taken this frame is drawn as observed; a rejected jump shows the kept line.
        self.predicted=not accepted
        if t-self.last_observed>s.hold_s+s.style_memory_s:self.style=self.candidate=None
        if t-self.last_observed>hold_s:
            if self.shown or self.value is not None:self.drop()
            return
        if not self.shown and obs is not None and obs[2]-obs[1]>=s.start_span_m and sum(self.history)>=s.confirm_hits:
            self.shown=True

    def update_style(self,t,style,confidence):
        """style is 'solid', 'dashed' or anything else (ignored as no evidence)."""
        if style=='dashed' and confidence<self.s.dashed_min_confidence:style=None
        if style not in ('solid','dashed'):return
        if style!=self.candidate:self.candidate=style;self.candidate_since=t
        if self.style!=style and t-self.candidate_since>=self.s.style_stable_s:self.style=style

    def points(self):
        if not self.shown or self.value is None:return None
        s=self.s;end=min(s.far_m,max(s.near_m+s.step_m,(self.far or s.far_m)+s.extend_m))
        x=np.arange(s.near_m,end+1e-6,s.step_m)
        return np.column_stack([x,evaluate(self.value,x)])


class DisplayFilter:
    def __init__(self,settings=None):
        self.s=settings or FilterSettings();self.reset()

    def reset(self):
        self.sides={side:SideTrack(self.s) for side in ('left','right')};self.last_t=None;self.last_motion_t=None

    def update(self,t,lines,motion=None):
        """lines: iterable of (side, points Nx2, predicted); the longest observed line per side is used,
        predicted ones are ignored.  motion: vehicle (dx, dy, dyaw) since the previous update in the
        previous base_link, or None when odometry is unavailable."""
        if self.restarted(t):self.reset()
        if self.last_t is not None and t<self.last_t:return  # a late message
        self.last_t=t
        if motion is not None:
            self.last_motion_t=t
            for track in self.sides.values():track.move(motion)
        odom=self.last_motion_t is not None and t-self.last_motion_t<=self.s.odom_timeout_s
        hold=self.s.hold_s if odom else min(self.s.hold_s,self.s.odom_lost_hold_s)
        best={}
        for side,points,predicted in lines:
            p=np.asarray(points,float)
            if side not in self.sides or len(p)<2 or predicted:continue
            span=float(p[:,0].max()-p[:,0].min())
            if side not in best or span>best[side][0]:best[side]=(span,p)
        for side,track in self.sides.items():
            track.update(t,best.get(side,(0,None))[1],hold)

    def restarted(self,t):
        # Styles arrive on their own topic and may trail the markers by a frame or two.
        return self.last_t is not None and t<self.last_t-self.s.restart_s

    def update_style(self,t,side,style,confidence):
        if self.restarted(t):self.reset()
        if side in self.sides:self.sides[side].update_style(t,style,confidence)

    def output(self):
        """[(side, points, predicted, style)] for every displayed side."""
        out=[]
        for side,track in self.sides.items():
            p=track.points()
            if p is not None:out.append((side,p,track.predicted,track.style))
        return out


def main():
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Header
    from geometry_msgs.msg import Point
    from visualization_msgs.msg import Marker, MarkerArray
    from nav_msgs.msg import Odometry
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from vision_lane_interfaces.msg import LaneStyle, LaneStyleArray
    from .core import Timeline

    class LaneDisplayFilter(Node):
        def __init__(self):
            super().__init__('lane_display_filter')
            defaults=FilterSettings()
            values={k:self.declare_parameter(k,v).value for k,v in vars(defaults).items()}
            self.filter=DisplayFilter(FilterSettings(**values))
            self.frame=self.declare_parameter('base_frame','base_link').value
            names={LaneStyle.SOLID:'solid',LaneStyle.DASHED:'dashed'}
            self.kinds={v:k for k,v in names.items()};self.names=names
            self.marker_pub=self.create_publisher(MarkerArray,self.declare_parameter('output_topic','/road/lane_display_markers').value,10)
            self.style_pub=self.create_publisher(LaneStyleArray,self.declare_parameter('style_output_topic','/road/lane_display_styles').value,10)
            self.create_subscription(MarkerArray,self.declare_parameter('boundary_topic','/road/lane_boundary_markers').value,self.on_boundaries,10)
            self.create_subscription(LaneStyleArray,self.declare_parameter('style_topic','/road/lane_styles').value,self.on_styles,10)
            self.timeline=Timeline(.25);self.last_pose=None
            self.create_subscription(Odometry,self.declare_parameter('odom_topic','/road/reference_odom').value,self.on_odom,
                                     QoSProfile(depth=200,reliability=ReliabilityPolicy.BEST_EFFORT))

        def on_odom(self,msg):
            q=msg.pose.pose.orientation
            yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
            self.timeline.add(msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,msg.pose.pose.position.x,msg.pose.pose.position.y,yaw)

        def on_styles(self,msg):
            t=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
            for lane in msg.lanes:
                if not lane.reprojected:self.filter.update_style(t,lane.side,self.names.get(lane.style),lane.confidence)

        def on_boundaries(self,msg):
            lines=[]
            for m in msg.markers:
                if m.action!=Marker.ADD or m.type!=Marker.LINE_STRIP or len(m.points)<2:continue
                side='left' if m.ns.endswith('/left') else 'right' if m.ns.endswith('/right') else None
                if side is None:continue
                lines.append((side,[(p.x,p.y) for p in m.points],m.color.r>.5))
            header=msg.markers[0].header if msg.markers else None
            if header is None:return
            stamp=header.stamp;t=stamp.sec+stamp.nanosec*1e-9
            if self.filter.restarted(t):self.last_pose=None
            pose=self.timeline.at(t);motion=None
            if pose is not None:
                if self.last_pose is not None and t>self.last_pose[0]:motion=relative(self.last_pose[1],pose)
                elif self.last_pose is None:motion=(0.,0.,0.)
                self.last_pose=(t,pose)
            self.filter.update(t,lines,motion)
            out=MarkerArray();head=Header(stamp=stamp,frame_id=self.frame)
            clear=Marker();clear.header=head;clear.action=Marker.DELETEALL;out.markers.append(clear)
            styles=LaneStyleArray();styles.header=head;styles.camera_id='front'
            for ident,(side,points,predicted,style) in enumerate(self.filter.output()):
                m=Marker();m.header=head;m.ns='display/'+side;m.id=ident;m.type=Marker.LINE_STRIP;m.action=Marker.ADD
                m.pose.orientation.w=1.;m.scale.x=.05;m.color.a=.9;m.color.g=1.;m.color.b=1.;m.color.r=1. if predicted else 0.
                m.lifetime.nanosec=300000000
                m.points=[Point(x=float(x),y=float(y),z=.02) for x,y in points];out.markers.append(m)
                if style is not None:
                    lane=LaneStyle();lane.side=side;lane.style=self.kinds[style];lane.confidence=1.
                    lane.reprojected=False;lane.source_stamp=stamp;lane.style_source_stamp=stamp
                    styles.lanes.append(lane)
            self.marker_pub.publish(out);self.style_pub.publish(styles)

    rclpy.init();node=LaneDisplayFilter()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':
    main()
