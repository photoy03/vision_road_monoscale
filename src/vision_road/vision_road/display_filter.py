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
from dataclasses import dataclass, fields

import numpy as np
from numpy.core.multiarray import interp as _interp  # np.interp without its Python wrapper (same values)

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
    # Line ends (all off by default).  The "seen span" is where the kept line rests on paint: from the
    # nearest remembered or observed point to the median far end of the last seen_frames accepted observations.
    gate_span:bool=False        # also gate at the near end, middle and far end of the seen span the observation shares
    gate_per_10m:float=0.       # with gate_span: the gate grows by this per 10 m beyond 10 m (far ends move more in curves)
    seen_frames:int=3           # observations the seen far end is taken from (median: one long frame does not count)
    end_from_seen:bool=False    # draw at most extend_m past the seen far end (not past a smoothed longest end)
    start_from_seen:bool=False  # draw at most extend_m below the seen near end (0 m only once paint near the car was seen)
    outside_quadratic:bool=False  # beyond the seen span continue the end's slope and curvature, the curvature fading to 0
    outside_fade_m:float=6.     # over this distance (no third-order bend outside what was seen)
    trim_m:float=0.             # >0: an observation is cut where it leaves the kept line by more than this (plan 32)
    trim_per_10m:float=0.       # ... plus this per 10 m beyond 10 m
    trim_from_m:float=8.        # ... from this x on (nearer, the kept line itself is least sure)
    max_step_m:float=0.         # >0: the kept line moves at most this much per frame at each station (plan 32)
    max_accel_m:float=0.        # >0: the drawn line's per-frame move changes by at most this from frame to frame (plan 32)
    max_accel_leash_m:float=.3  # ... but it stays within this of the kept line at each station
    near_quadratic_m:float=0.   # >0: below this x the line continues its slope and curvature there (the cubic
                                # through the 5-20 m stations swings most at 0 m, where it amplifies their noise)


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
    y=_interp(x,p[:,0],p[:,1]);span=float(x[-1]-x[0])
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


def outside(stations,x,y,span,fade):
    """y with the parts outside span=(near, far) replaced by the line continued from that end: its
    slope, and its curvature fading linearly to 0 over fade metres (then straight on)."""
    c=np.linalg.solve(_stations_matrix(len(stations)),stations)
    d1=np.polynomial.polynomial.polyder(c);d2=np.polynomial.polynomial.polyder(c,2)
    y=y.copy()
    for edge,sign in ((span[1],1.),(span[0],-1.)):
        if math.isinf(edge):continue
        mask=(x-edge)*sign>0
        if not mask.any():continue
        d=np.abs(x[mask]-edge);k=float(np.polynomial.polynomial.polyval(edge,d2))
        if math.isinf(fade):bend=d*d/2  # curvature kept (no fading)
        else:dd=np.minimum(d,fade);bend=dd*dd/2-dd**3/(6*fade)+np.maximum(0.,d-fade)*fade/2
        y[mask]=float(np.polynomial.polynomial.polyval(edge,c))+sign*float(np.polynomial.polynomial.polyval(edge,d1))*d+k*bend
    return y


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
        self.seen=deque(maxlen=s.seen_frames)  # [near, far] in base_link of the last accepted observations
        # max_accel_m: the drawn stations (following the kept line) and their last per-frame move beyond the car's motion
        self.drawn=None;self.velocity=None

    @property
    def value(self):return self.filter.x

    def move(self,motion):
        if self.value is not None:self.filter.x=shift(self.value,motion)
        if self.drawn is not None:self.drawn=shift(self.drawn,motion)
        self.pending=[shift(p,motion) for p in self.pending]
        if self.far is not None:self.far-=motion[0]
        for span in self.seen:span[0]-=motion[0];span[1]-=motion[0]
        if len(self.memory):
            dx,dy,dyaw=motion;c,s=math.cos(dyaw),math.sin(dyaw);px=self.memory[:,0]-dx;py=self.memory[:,1]-dy
            self.memory=np.column_stack([c*px+s*py,-s*px+c*py,self.memory[:,2]])

    def drop(self):
        # The style survives a short loss: style evidence is sparse (a dashed line in a
        # curve is often UNKNOWN), and re-learning it showed the line solid on return.
        self.shown=False;self.filter.reset();self.far=None;self.pending=[];self.history.clear();self.memory=np.empty((0,3))
        self.seen.clear();self.drawn=None

    def span(self):
        """(near, far) of the seen span, or None before an accepted observation: the nearest remembered or
        last observed point, and the median far end of the last seen_frames accepted observations."""
        if not self.seen:return None
        near=self.seen[-1][0]
        if len(self.memory):near=min(near,float(self.memory[:,0].min()))
        # np.median of at most seen_frames values, without its per-call cost: the same value.
        far=sorted(float(f) for _,f in self.seen);k=len(far)//2
        return near,(far[k] if len(far)%2 else (far[k-1]+far[k])/2)

    def span_ok(self,stations,near,far):
        """The observation (stations over near..far) against the kept line at the near end, middle and
        far end of the part both cover (the kept line's part: the seen span and extend_m past it)."""
        kept=self.span()
        if kept is None:return True
        lo,hi=max(near,kept[0]),min(far,kept[1]+self.s.extend_m)  # the drawn line reaches extend_m past the seen end
        if hi<lo:return True
        x=np.array([lo,.5*(lo+hi),hi])
        limit=self.s.gate_m*(1.+self.s.gate_per_10m*np.maximum(0.,x-10.)/10.)
        return bool(np.all(np.abs(evaluate(stations,x)-evaluate(self.value,x))<=limit))

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

    def take(self,stations,t,alone=None,span=None):
        """Gate a measurement at 10 m; returns False when it was set aside as an outlier.
        alone: the observation fitted without remembered paint, used when the line switches.
        span: the observation's (near, far), for gate_span."""
        if self.value is None:
            self.filter.reset(stations,t);return True
        if abs(stations[1]-self.value[1])<=self.s.gate_m and (not self.s.gate_span or span is None or self.span_ok(stations,*span)):
            before=np.array(self.value,float);self.pending=[];self.filter(stations,t)
            if self.s.max_step_m>0:
                # max_step_m (plan 32): the kept line moves at most this much per frame at each station (a line taken
                # again after rejected frames slid instead of jumping); a sustained change below still resets at once.
                step=np.clip(np.asarray(self.filter.x,float)-before,-self.s.max_step_m,self.s.max_step_m);self.filter.x=before+step
            return True
        # Only the latest gate_frames candidates can make a sustained change.
        self.pending=(self.pending+[stations])[-self.s.gate_frames:]
        recent=[p[1] for p in self.pending]
        if len(recent)>=self.s.gate_frames and max(recent)-min(recent)<=self.s.gate_m:
            # A sustained change (new line, lane change): jump to it rather than slide, and forget the old line.
            self.filter.reset(stations if alone is None else alone,t);self.pending=[];self.memory=np.empty((0,3));self.drawn=None;return True
        return False

    def trimmed(self,observed):
        """trim_m (plan 32): the observation up to where it leaves the kept line by more than trim_m (+ trim_per_10m per
        10 m beyond 10 m), from trim_from_m on; a far end that runs onto other paint (a barrier past the road's
        end) is cut off instead of bending the line or being taken whole."""
        if self.value is None or len(observed)<3:return observed
        p=np.asarray(observed,float);kept=evaluate(self.value,p[:,0])
        limit=self.s.trim_m+self.s.trim_per_10m*np.maximum(p[:,0]-10.,0.)/10.
        bad=np.flatnonzero((np.abs(p[:,1]-kept)>limit)&(p[:,0]>=self.s.trim_from_m))
        return p if not len(bad) else p[:bad[0]]

    def update(self,t,observed,hold_s,paint=None,travel=None):
        s=self.s;obs=None;accepted=False
        if observed is not None and s.trim_m>0:
            observed=self.trimmed(observed)
            if len(observed)<2:observed=None
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
            stations,near,far,alone=obs
            accepted=self.take(stations,t,alone,(near,far))
            if accepted:
                # (A cut observation's far end counts as seen too: leaving it out shook the ring-road scenes more,
                # plan 32.)
                self.seen.append([near,far])
                # The drawn end grows quickly and shrinks slowly, so its length does not flicker.
                self.far=far if self.far is None else self.far+(.5 if far>self.far else .1)*(far-self.far)
                # Rejected candidates do not extend the hold: an unstable stream must not keep
                # the last good line on screen past hold_s.
                self.last_observed=t
                if travel is not None:self.store(np.asarray(observed,float),paint,travel)
        # Confirmation counts accepted observations only; counting rejected ones let an unstable
        # stream re-confirm a dropped line within 5 frames, so it blinked about once a second.
        self.history.append(accepted)
        if s.max_accel_m>0:self.draw()
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

    def draw(self):
        """max_accel_m (plan 32): the drawn line follows the kept line with its per-frame move changing by at most
        max_accel_m (the second difference the eye sees as shake), slowing before the kept line so it does not overshoot.
        Only the drawing lags: gating, trimming and memory use the kept line (a gate on the lagging line rejected a
        lane change's observations and held the line away from the paint).  A jump of the kept line (new line, lane
        change) is drawn at once."""
        if self.value is None:self.drawn=None;return
        target=np.asarray(self.value,float)
        if self.drawn is None:self.drawn=target.copy();self.velocity=np.zeros_like(target);return
        a=self.s.max_accel_m;d=target-self.drawn
        lim=.5*(np.sqrt(a*a+8*a*np.abs(d))-a)  # v+(v-a)+... stops within d
        self.velocity=np.clip(self.velocity+np.clip(d-self.velocity,-a,a),-lim,lim);self.drawn=self.drawn+self.velocity
        # A leash: a lag that grows (the far stations in a quick sway; a bent drawn line turned with the car's yaw
        # diverged without it) is dragged along, never more than max_accel_leash_m behind the kept line.
        L=self.s.max_accel_leash_m;self.drawn=target-np.clip(target-self.drawn,-L,L)

    def points(self):
        if not self.shown or self.value is None:return None
        s=self.s;end=min(s.far_m,max(s.near_m+s.step_m,(self.far or s.far_m)+s.extend_m));start=s.near_m
        span=self.span()
        if span is not None:
            if s.end_from_seen:end=min(end,span[1]+s.extend_m)
            if s.start_from_seen:start=max(start,span[0]-s.extend_m)
        if end-start<s.step_m:return None
        x=np.arange(start,end+1e-6,s.step_m);v=self.drawn if s.max_accel_m>0 and self.drawn is not None else self.value
        y=evaluate(v,x)
        if s.outside_quadratic and span is not None and span[1]>span[0]:
            y=outside(v,x,y,span,s.outside_fade_m)
        if s.near_quadratic_m>0:
            y=outside(v,x,y,(s.near_quadratic_m,math.inf),math.inf)
        return np.column_stack([x,y])


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


def simplify(points,error):
    """The display polyline road_node publishes: Douglas-Peucker within error metres (0: unchanged)."""
    import cv2
    if error<=0 or len(points)<3:return points
    return cv2.approxPolyDP(points.astype(np.float32),error,False).reshape(-1,2).astype(float)


def lane_styles(lanes):
    """road_node's /road/lane_styles content as the display filter reads it: per side the lane with the
    highest style confidence (first on ties), observed ones only, confidence as the message's float32."""
    unique={}
    for lane in lanes:
        if lane.side not in unique or lane.style_confidence>unique[lane.side].style_confidence:unique[lane.side]=lane
    return [(lane.side,lane.style if lane.style in ('solid','dashed') else None,float(np.float32(lane.style_confidence)))
            for lane in unique.values() if not lane.reprojected]


class DisplayCore:
    """The display filter's per-frame step without ROS: vehicle motion from consecutive poses and the
    filter update.  road_node's perception worker runs it next to the detector (display_filter_in_worker)."""

    def __init__(self,settings):
        self.filter=DisplayFilter(settings);self.last_pose=None

    def styles(self,t,styles):
        for side,style,confidence in styles:self.filter.update_style(t,side,style,confidence)

    def step(self,t,lines,pose,paint):
        if self.filter.restarted(t):self.last_pose=None
        motion=None
        if pose is not None:
            if self.last_pose is not None and t>self.last_pose[0]:motion=relative(self.last_pose[1],pose)
            elif self.last_pose is None:motion=(0.,0.,0.)
            self.last_pose=(t,pose)
        self.filter.update(t,lines,motion,paint)
        return self.filter.output()


class DisplayStage:
    """The display filter on ROS messages, without a node: road_node can run it inline (display_filter_inline)
    and the lane_display_filter node wraps it.  Inputs are road_node's own topics; on_boundaries returns the
    /road/lane_display_markers and /road/lane_display_styles messages for the boundary message's stamp."""

    def __init__(self,settings,frame='base_link'):
        from vision_lane_interfaces.msg import LaneStyle
        self.core=DisplayCore(settings);self.filter=self.core.filter;self.frame=frame;self.paint=None
        self.names={LaneStyle.SOLID:'solid',LaneStyle.DASHED:'dashed'};self.kinds={v:k for k,v in self.names.items()}

    def on_paint(self,msg):
        from visualization_msgs.msg import Marker
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

    def on_boundaries(self,msg,pose_at):
        """pose_at(t): the vehicle pose (x, y, yaw) at a stamp, or None.  Returns (markers, styles) or None."""
        from visualization_msgs.msg import Marker
        lines=[]
        for m in msg.markers:
            if m.action!=Marker.ADD or m.type!=Marker.LINE_STRIP or len(m.points)<2:continue
            side='left' if m.ns.endswith('/left') else 'right' if m.ns.endswith('/right') else None
            if side is None:continue
            lines.append((side,[(p.x,p.y) for p in m.points],m.color.r>.5))
        header=msg.markers[0].header if msg.markers else None
        if header is None:return None
        return self.update(header.stamp,lines,pose_at)

    def on_lanes(self,stamp,boundaries,paint,pose_at):
        """The same from lanes, without messages (road_node inline): boundaries [(side, points, predicted)],
        paint [(side, points)] observed fragments of this frame."""
        self.paint=(stamp.sec+stamp.nanosec*1e-9,paint)
        return self.update(stamp,boundaries,pose_at)

    def update(self,stamp,lines,pose_at):
        t=stamp.sec+stamp.nanosec*1e-9
        if self.filter.restarted(t):self.core.last_pose=None
        pose=pose_at(t)
        paint=self.paint[1] if self.paint is not None and abs(self.paint[0]-t)<1e-6 else None
        return self.messages(stamp,self.core.step(t,lines,pose,paint))

    def messages(self,stamp,output):
        """/road/lane_display_markers and /road/lane_display_styles for one frame's filter output."""
        if not hasattr(self,'_msgs'):
            from std_msgs.msg import Header
            from geometry_msgs.msg import Point
            from visualization_msgs.msg import Marker, MarkerArray
            from vision_lane_interfaces.msg import LaneStyle, LaneStyleArray
            self._msgs=(Header,Point,Marker,MarkerArray,LaneStyle,LaneStyleArray)
        Header,Point,Marker,MarkerArray,LaneStyle,LaneStyleArray=self._msgs
        def point(x,y,new=Point.__new__):
            # Point(x=, y=, z=) without the per-field checks of its constructor: the same message.
            p=new(Point);p._x=x;p._y=y;p._z=.02;return p
        out=MarkerArray();head=Header(stamp=stamp,frame_id=self.frame)
        clear=Marker();clear.header=head;clear.action=Marker.DELETEALL;out.markers.append(clear)
        styles=LaneStyleArray();styles.header=head;styles.camera_id='front'
        for ident,(side,points,predicted,style) in enumerate(output):
            m=Marker();m.header=head;m.ns='display/'+side;m.id=ident;m.type=Marker.LINE_STRIP;m.action=Marker.ADD
            m.pose.orientation.w=1.;m.scale.x=.05;m.color.a=.9;m.color.g=1.;m.color.b=1.;m.color.r=1. if predicted else 0.
            m.lifetime.nanosec=300000000
            m.points=[point(float(x),float(y)) for x,y in points.tolist()];out.markers.append(m)
            if style is not None:
                lane=LaneStyle();lane.side=side;lane.style=self.kinds[style];lane.confidence=1.
                lane.reprojected=False;lane.source_stamp=stamp;lane.style_source_stamp=stamp
                styles.lanes.append(lane)
        return out,styles


def settings_from(values):
    """FilterSettings from a parameter dict (unknown keys ignored), e.g. carla.yaml's lane_display_filter section."""
    names={f.name for f in fields(FilterSettings)}
    return FilterSettings(**{k:v for k,v in values.items() if k in names})


def main():
    import rclpy
    from rclpy.node import Node
    from visualization_msgs.msg import MarkerArray
    from nav_msgs.msg import Odometry
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from vision_lane_interfaces.msg import LaneStyleArray
    from .core import Timeline

    class LaneDisplayFilter(Node):
        def __init__(self):
            super().__init__('lane_display_filter')
            defaults=FilterSettings()
            values={k:self.declare_parameter(k,v).value for k,v in vars(defaults).items()}
            self.stage=DisplayStage(FilterSettings(**values),self.declare_parameter('base_frame','base_link').value)
            self.filter=self.stage.filter
            self.marker_pub=self.create_publisher(MarkerArray,self.declare_parameter('output_topic','/road/lane_display_markers').value,10)
            self.style_pub=self.create_publisher(LaneStyleArray,self.declare_parameter('style_output_topic','/road/lane_display_styles').value,10)
            self.create_subscription(MarkerArray,self.declare_parameter('boundary_topic','/road/lane_boundary_markers').value,self.on_boundaries,10)
            self.create_subscription(LaneStyleArray,self.declare_parameter('style_topic','/road/lane_styles').value,self.stage.on_styles,10)
            # Observed paint of the same frame (road_node publishes it just before the boundaries).
            self.create_subscription(MarkerArray,self.declare_parameter('paint_topic','/road/lane_markers').value,self.stage.on_paint,10)
            self.timeline=Timeline(.25)
            self.create_subscription(Odometry,self.declare_parameter('odom_topic','/road/reference_odom').value,self.on_odom,
                                     QoSProfile(depth=200,reliability=ReliabilityPolicy.BEST_EFFORT))

        def on_odom(self,msg):
            q=msg.pose.pose.orientation
            yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
            self.timeline.add(msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,msg.pose.pose.position.x,msg.pose.pose.position.y,yaw)

        def on_boundaries(self,msg):
            result=self.stage.on_boundaries(msg,self.timeline.at)
            if result is not None:self.marker_pub.publish(result[0]);self.style_pub.publish(result[1])

    rclpy.init();node=LaneDisplayFilter()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':
    main()
