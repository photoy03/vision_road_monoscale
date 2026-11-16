"""Display-only smoothing of lane boundaries for the HUD.

road_node publishes boundaries tuned not to lose a lane; a head-up display
needs lines that do not shake or blink.  This node sits between the two: it
confirms a side M-of-N, keeps one quadratic per side, moves it with the vehicle
odometry, smooths it with a 1-euro filter, rejects single-frame jumps, holds a
lost side briefly in the predicted colour and only switches solid/dashed after
the style has been stable.  With memory_m set it remembers where it saw paint and
carries it under the vehicle, so the line reaches near_m (down to the rear axle)
from seen paint rather than from the far fragment's direction.
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
# With cubic_m set the curve is a cubic, carried as stations at 5/10/15/20 m (10 m stays index 1).
STATIONS4=np.array([5.,10.,15.,20.])
_V4=np.vander(STATIONS4,4,increasing=True)


def _stations_matrix(n):return _V if n==3 else _V4


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
    near_m:float=3.             # draw from here (base_link x); below the first seen paint it comes from memory or the fit
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
    memory_m:float=0.           # >0: keep seen paint this many metres of travel and fit it below the observation (0 = off)
    memory_store_m:float=4.     # remember the observation's first metres only (the nearest look is the best)
    memory_gate_m:float=.25     # remembered bins this far off the kept line are left out (stray paint)
    fit_near_m:float=0.         # >0: boundary points beyond this count less in the fit, (near/x)^2 (sharp curves)
    cubic_m:float=0.            # >0: the line is a cubic, its third-order term held to 0 unless a quadratic misses by about this (sharp curves)
    cubic_ridge:float=1e-3      # pull of that term at that miss


def fit(points,ridge=1e-3,strength=None,near=0.,cubic=0.,cubic_ridge=1e-3):
    """Quadratic through a polyline in base_link (x forward, y left), returned as station values.

    The input is usually a simplified polyline with a handful of vertices, so it
    is resampled along x first.  Short spans get a stronger pull toward zero
    curvature so a 3 m piece does not bend the whole display.  strength, when given,
    replaces that two-step pull.  near>0: points beyond it count less, (near/x)^2, so the far end
    of a sharp curve does not pull the near line.  cubic>0: a cubic (four stations) whose third-order
    term is held to 0 unless the quadratic misses the line by about cubic (p90): its pull is
    cubic_ridge*(cubic/miss)^4, and as strong as the curvature pull on short spans."""
    p=np.asarray(points,float);p=p[np.argsort(p[:,0])]
    x=np.arange(p[0,0],p[-1,0]+1e-6,.5)
    if len(x)<2:x=p[[0,-1],0]
    y=np.interp(x,p[:,0],p[:,1]);span=float(x[-1]-x[0])
    A=np.column_stack([np.ones_like(x),x,x*x])
    lam=len(x)*float(np.mean(x*x))**2*(strength if strength is not None else ridge if span>=8 else 10.)
    w=np.ones_like(x) if near<=0 else np.minimum(1.,(near/np.maximum(x,1.))**2)
    coeff=np.linalg.solve((A*w[:,None]).T@A+np.diag([0.,0.,lam]),(A*w[:,None]).T@y)
    if cubic<=0:return _V@coeff,float(x[0]),float(x[-1])
    miss=float(np.percentile(np.abs(A@coeff-y),90))
    pull=10. if span<8 and strength is None else cubic_ridge*(cubic/max(miss,1e-3))**4
    A=np.column_stack([A,x**3]);lam3=len(x)*float(np.mean(x*x))**3*pull
    coeff=np.linalg.solve((A*w[:,None]).T@A+np.diag([0.,0.,lam,lam3]),(A*w[:,None]).T@y)
    return _V4@coeff,float(x[0]),float(x[-1])


def evaluate(stations,x):
    coeff=np.linalg.solve(_stations_matrix(len(stations)),stations)
    return np.polyval(coeff[::-1],x)


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
    return _stations_matrix(len(stations))@np.polyfit(nx,ny,len(stations)-1)[::-1]


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
        self.memory=np.empty((0,3))  # x, y in base_link and the travel when seen

    @property
    def value(self):return self.filter.x

    def move(self,motion):
        if self.value is not None:self.filter.x=shift(self.value,motion)
        self.pending=[shift(p,motion) for p in self.pending]
        if self.far is not None:self.far-=motion[0]
        if len(self.memory):
            dx,dy,dyaw=motion;c,s=math.cos(dyaw),math.sin(dyaw);px=self.memory[:,0]-dx;py=self.memory[:,1]-dy
            self.memory=np.column_stack([c*px+s*py,-s*px+c*py,self.memory[:,2]])

    def drop(self):
        # The style survives a short loss: style evidence is sparse (a dashed line in a
        # curve is often UNKNOWN), and re-learning it showed the line solid on return.
        self.shown=False;self.filter.reset();self.far=None;self.pending=[];self.history.clear();self.memory=np.empty((0,3))

    def store(self,observed,paint,travel):
        """Remember the near part of an accepted observation, only where paint was seen this frame.

        observed is the boundary road_node published (a merged line that also spans the gaps
        of a dashed line); paint is that side's observed fragments, or None when the paint
        topic is missing, in which case the boundary is stored as it is."""
        p=observed[np.argsort(observed[:,0])]
        x=np.arange(p[0,0],min(p[-1,0],p[0,0]+self.s.memory_store_m)+1e-6,.25);y=np.interp(x,p[:,0],p[:,1])
        if paint is not None:
            seen=np.zeros(len(x),bool)
            for q in paint:
                q=q[np.argsort(q[:,0])];inside=(x>=q[0,0]-.1)&(x<=q[-1,0]+.1)  # BEV chunks leave 7 cm joins
                if inside.any():seen[inside]|=np.abs(np.interp(x[inside],q[:,0],q[:,1])-y[inside])<=.5
            x,y=x[seen],y[seen]
        if len(x):self.memory=np.vstack([self.memory,np.column_stack([x,y,np.full(len(x),travel)])])

    def recall(self,near,stations):
        """Remembered paint from near_m up to the seen near end, as 0.5 m bin means (mean x and y,
        so a bin in a curve does not snap to its centre).  Bins far off the kept line (or, before
        there is one, the observation's own fit) are left out."""
        lo=self.s.near_m-.25;n_bins=int(math.ceil((near-.25-lo)/.5))
        if n_bins<1 or not len(self.memory):return None
        m=self.memory[(self.memory[:,0]>=lo)&(self.memory[:,0]<near-.25)]
        if not len(m):return None
        i=np.minimum(((m[:,0]-lo)/.5).astype(int),n_bins-1)
        n=np.bincount(i,minlength=n_bins);ok=n>0
        p=np.column_stack([np.bincount(i,weights=m[:,0],minlength=n_bins)[ok],np.bincount(i,weights=m[:,1],minlength=n_bins)[ok]])/n[ok,None]
        p=p[np.abs(p[:,1]-evaluate(stations if self.value is None else self.value,p[:,0]))<=self.s.memory_gate_m]
        return p if len(p) else None

    def take(self,stations,t,alone=None):
        """Gate a measurement at 10 m; returns False when it was set aside as an outlier.
        alone: the observation fitted without remembered paint, used when the line switches."""
        if self.value is None:
            self.filter.reset(stations,t);return True
        if abs(stations[1]-self.value[1])<=self.s.gate_m:
            self.pending=[];self.filter(stations,t);return True
        # Only the latest gate_frames candidates can make a sustained change.
        self.pending=(self.pending+[stations])[-self.s.gate_frames:]
        recent=[p[1] for p in self.pending]
        if len(recent)>=self.s.gate_frames and max(recent)-min(recent)<=self.s.gate_m:
            # A sustained change (new line, lane change): jump to it rather than slide, and forget the old line.
            self.filter.reset(stations if alone is None else alone,t);self.pending=[];self.memory=np.empty((0,3));return True
        return False

    def update(self,t,observed,hold_s,paint=None,travel=None):
        s=self.s;obs=None;accepted=False
        if observed is not None:
            stations,near,far=fit(observed,near=s.fit_near_m,cubic=s.cubic_m,cubic_ridge=s.cubic_ridge)
            if far-near>=s.update_span_m and near<=s.max_start_m:
                # What is seen now wins; remembered paint only fills below the seen near end.
                below=self.recall(near,stations) if travel is not None else None
                alone=stations
                # The curvature pull changes smoothly with the span: spans hover about 8 m, where the
                # two-step pull flipped the curvature from frame to frame in a curve.
                if below is not None:stations=fit(np.vstack([below,observed]),strength=10**np.interp(far-float(below[0,0]),[4.,12.],[1.,-3.]),near=s.fit_near_m,cubic=s.cubic_m,cubic_ridge=s.cubic_ridge)[0]
                obs=(stations,near,far,alone)
        if obs is not None:
            stations,_,far,alone=obs
            accepted=self.take(stations,t,alone)
            if accepted:
                # The drawn end grows quickly and shrinks slowly, so its length does not flicker.
                self.far=far if self.far is None else self.far+(.5 if far>self.far else .1)*(far-self.far)
                # Rejected candidates do not extend the hold: an unstable stream must not keep
                # the last good line on screen past hold_s.
                self.last_observed=t
                if travel is not None:self.store(np.asarray(observed,float),paint,travel)
        # Confirmation counts accepted observations only; counting rejected ones let an unstable
        # stream re-confirm a dropped line within 5 frames, so it blinked about once a second.
        self.history.append(accepted)
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
        self.sides={side:SideTrack(self.s) for side in ('left','right')};self.last_t=None;self.last_motion_t=None;self.travel=0.

    def update(self,t,lines,motion=None,paint=None):
        """lines: iterable of (side, points Nx2, predicted); the longest observed line per side is used,
        predicted ones are ignored.  motion: vehicle (dx, dy, dyaw) since the previous update in the
        previous base_link, or None when odometry is unavailable.  paint: iterable of (side, points)
        observed paint fragments of this frame, used to remember only where paint was seen."""
        if self.restarted(t):self.reset()
        if self.last_t is not None and t<self.last_t:return  # a late message
        self.last_t=t
        if motion is not None:
            self.last_motion_t=t;self.travel+=math.hypot(motion[0],motion[1])
            for track in self.sides.values():track.move(motion)
        memory=self.s.memory_m>0 and motion is not None
        for track in self.sides.values():
            # Memory that cannot be moved with the vehicle, or is older than memory_m of travel, is dropped.
            if not memory:track.memory=np.empty((0,3))
            elif len(track.memory):track.memory=track.memory[self.travel-track.memory[:,2]<=self.s.memory_m]
        fragments={}
        if paint is not None:
            for side,points in paint:fragments.setdefault(side,[]).append(np.asarray(points,float))
        odom=self.last_motion_t is not None and t-self.last_motion_t<=self.s.odom_timeout_s
        hold=self.s.hold_s if odom else min(self.s.hold_s,self.s.odom_lost_hold_s)
        best={}
        for side,points,predicted in lines:
            p=np.asarray(points,float)
            if side not in self.sides or len(p)<2 or predicted:continue
            span=float(p[:,0].max()-p[:,0].min())
            if side not in best or span>best[side][0]:best[side]=(span,p)
        for side,track in self.sides.items():
            track.update(t,best.get(side,(0,None))[1],hold,None if paint is None else fragments.get(side,[]),self.travel if memory else None)

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
            # Observed paint of the same frame (road_node publishes it just before the boundaries).
            self.paint=None
            self.create_subscription(MarkerArray,self.declare_parameter('paint_topic','/road/lane_markers').value,self.on_paint,10)
            self.timeline=Timeline(.25);self.last_pose=None
            self.create_subscription(Odometry,self.declare_parameter('odom_topic','/road/reference_odom').value,self.on_odom,
                                     QoSProfile(depth=200,reliability=ReliabilityPolicy.BEST_EFFORT))

        def on_odom(self,msg):
            q=msg.pose.pose.orientation
            yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
            self.timeline.add(msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,msg.pose.pose.position.x,msg.pose.pose.position.y,yaw)

        def on_paint(self,msg):
            lines=[];stamp=None
            for m in msg.markers:
                if m.action!=Marker.ADD or m.type!=Marker.LINE_STRIP or len(m.points)<2 or m.color.r>.5:continue
                side='left' if m.ns.endswith('/left') else 'right' if m.ns.endswith('/right') else None
                if side is None:continue
                stamp=m.header.stamp;lines.append((side,[(p.x,p.y) for p in m.points]))
            if msg.markers:stamp=msg.markers[0].header.stamp
            if stamp is not None:self.paint=(stamp.sec+stamp.nanosec*1e-9,lines)

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
            paint=self.paint[1] if self.paint is not None and abs(self.paint[0]-t)<1e-6 else None
            self.filter.update(t,lines,motion,paint)
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
