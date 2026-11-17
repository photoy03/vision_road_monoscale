"""Lane return and lane change guidance from the lane centrelines (vision_road /road/lane_centres).

Pure computation: no ROS, no Autoware; numpy only.  lane_guidance_node.py wraps it.

Target lane.  The guidance keeps one target lane, followed by its position on the road, not by its
name: the detector renames lanes (ego / left / right) the moment the car crosses a line.  Each frame
the target centreline of the last frame is carried into the new base_link with the vehicle's motion and
the current lane whose centre lies within match_m of it (at ref_m ahead) is the target.  Without a
match the carried line stands in for up to carry_sec; after that, or after a sharp turn (a junction:
yaw rate above turn_rate) or an implausible offset, the target is taken afresh: the lane the car is in.

  - The target moves to a neighbouring lane only when all of these hold: the line between them is
    white dashed (crossable), the neighbour is lane-like and not oncoming; the car moves towards
    that line (lateral speed of at least change_speed_mps, gaps under change_gap_sec, for
    change_sec); and the car centre is within change_line_m of the line, or past it.
  - A yellow or solid boundary, or one to an oncoming lane, never moves the target: crossing it keeps
    the target on the home lane ('forbidden' while the car's side is on or past that line).
  - After a move the same must hold towards the other side to move again: on a line the target does
    not flip.

The car's offset from the target centre.  The centreline is seen from about 5 m ahead.  Its points
seen in the last memory_sec (kept in world coordinates, from ref_m to ref_m + 8 m ahead) are, after
the car has moved on, beside and just ahead of the car: a straight line through those within
near_window_m of x = 0 gives the centre's offset and direction at the car.  Until there are enough
(start, a new target), the centreline is continued back from its reference point along its heading
and curvature.

Path.  A quintic in the lateral offset from the target centreline, parameterised by the forward
distance x: from the car (on its position, along its heading, straight) to the centreline at x = L
(offset, slope and curvature 0), then along the centreline for tail_m.  L is the speed times
merge_sec, at least merge_per_m per metre of offset, within merge_min_m..merge_max_m.

States: 'none' (no reliable target: no lane, taken afresh within settle_sec, no measured part of the
target lane within near_m, turning sharply), 'keep' (near the centre: shown from on_m, hidden again
under off_m), 'return' (back to the home lane's centre), 'change' (onto the new lane, until near its
centre), 'forbidden' (the car's side is on or past a line that must not be crossed: back to the home
lane)."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class GuidanceSettings:
    ref_m: float = 5.0  # where lane centres are compared (inside the camera view)
    on_m: float = 0.30  # guidance shown when the car is this far from the target centre
    off_m: float = 0.15  # ... and hidden again when back within this
    change_speed_mps: float = 0.25  # lateral speed towards the line that counts as moving over
    change_sec: float = 0.6  # ... for at least this long
    change_gap_sec: float = 0.15  # a dip below that speed this short does not break it
    change_line_m: float = 0.5  # car centre within this of the line (or past it)
    half_width_m: float = 0.91  # the car's half width (IONIQ 1.82 m): its side is on a line from here
    match_m: float = 0.5  # a carried target centre matches a current lane within this
    carry_sec: float = 1.5  # without a match the carried target stands in this long
    settle_sec: float = 0.5  # after taking a lane afresh no guidance for this long
    near_m: float = 15.0  # no guidance while the target lane has no measured part starting within this
    turn_rate: float = 0.30  # rad/s: turning this sharply (a junction) suspends the guidance
    implausible: float = 1.3  # an offset beyond this many lane widths is a lost lane, not a position
    memory_sec: float = 1.5  # centreline points kept to find the centre beside the car
    near_window_m: float = 3.0
    merge_sec: float = 3.0
    merge_min_m: float = 10.0
    merge_max_m: float = 30.0
    merge_per_m: float = 8.0
    tail_m: float = 6.0
    step_m: float = 0.5
    speed_tau_sec: float = 0.3  # low-pass of the speeds
    offset_tau_sec: float = 0.1  # low-pass of the offset used for the state


def relative(previous, pose):
    """Motion (dx, dy, dyaw) from one world pose (x, y, yaw) to the next, in the first's frame."""
    c, s = math.cos(previous[2]), math.sin(previous[2]); wx, wy = pose[0]-previous[0], pose[1]-previous[1]
    return (c*wx+s*wy, -s*wx+c*wy, math.atan2(math.sin(pose[2]-previous[2]), math.cos(pose[2]-previous[2])))


def carry(points, motion):
    """Points of the previous base_link in the next one."""
    dx, dy, dyaw = motion; c, s = math.cos(dyaw), math.sin(dyaw)
    px = points[:, 0]-dx; py = points[:, 1]-dy
    return np.column_stack([c*px+s*py, -s*px+c*py])


def to_world(points, pose):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.column_stack([pose[0]+c*points[:, 0]-s*points[:, 1], pose[1]+s*points[:, 0]+c*points[:, 1]])


def to_base(points, pose):
    c, s = math.cos(pose[2]), math.sin(pose[2]); dx = points[:, 0]-pose[0]; dy = points[:, 1]-pose[1]
    return np.column_stack([c*dx+s*dy, -s*dx+c*dy])


def centre_at(points, x, ref=None):
    """y of a centreline at forward distances x: the polyline inside it; before its first point the
    line continued from its reference point along its heading and curvature (ref: ref_x_m, offset_m,
    heading_rad, curvature_1pm) or straight; after its last point straight on."""
    p = np.asarray(points, float); x = np.asarray(x, float); y = np.interp(x, p[:, 0], p[:, 1])
    before = x < p[0, 0]
    if before.any():
        if ref is not None:
            x0 = ref['ref_x_m']; y0 = ref['offset_m']; b = math.tan(ref['heading_rad'])
            k = ref['curvature_1pm']*(1+b*b)**1.5; u = x[before]-x0; y[before] = y0+b*u+.5*k*u*u
        elif len(p) >= 2:
            b = (p[1, 1]-p[0, 1])/max(p[1, 0]-p[0, 0], 1e-6); y[before] = p[0, 1]+b*(x[before]-p[0, 0])
    after = x > p[-1, 0]
    if after.any() and len(p) >= 2:
        b = (p[-1, 1]-p[-2, 1])/max(p[-1, 0]-p[-2, 0], 1e-6); y[after] = p[-1, 1]+b*(x[after]-p[-1, 0])
    return y


def quintic(d0, d1, d2, length):
    """Coefficients (lowest power first) of d(x) with d, d', d'' = d0, d1, d2 at 0 and 0, 0, 0 at length."""
    L = length; a0, a1, a2 = d0, d1, d2/2.
    A = np.array([[L**3, L**4, L**5], [3*L**2, 4*L**3, 5*L**4], [6*L, 12*L**2, 20*L**3]])
    b = np.array([-(a0+a1*L+a2*L*L), -(a1+2*a2*L), -2*a2])
    return np.r_[a0, a1, a2, np.linalg.solve(A, b)]


def path_onto(centre, near, length, settings):
    """Path from the car (base_link origin, heading +x, straight) onto a centreline.  centre(x): its y;
    near: (y, slope, curvature) of the centre at x = 0.  Quintic offset over [0, length], then the
    centreline for tail_m.  Returns (n, 2) points."""
    s = settings; y0, slope, curve = near
    coef = quintic(-y0, -slope, -curve, length)
    x = np.arange(0., length+s.tail_m+1e-9, s.step_m)
    d = np.where(x <= length, np.polyval(coef[::-1], np.minimum(x, length)), 0.)
    y = centre(x); y[0] = y0  # the near centre (memory) at 0, the polyline (and its continuation) beyond
    blend = np.clip(x/max(s.near_window_m, 1e-6), 0., 1.)
    y = (1-blend)*(y0+slope*x)+blend*y  # the memory's near line joins the polyline over near_window_m
    return np.column_stack([x, y+d])


@dataclass
class Lane:
    """One centreline of /road/lane_centres, as the guidance reads it."""
    name: str
    points: np.ndarray
    ref: dict
    width: float
    opposite: bool = False
    lane_like: bool = True
    left: dict = None
    right: dict = None
    measured_from: float = math.inf  # where the two boundaries start facing each other (both_m); inf: estimated

    @classmethod
    def from_message(cls, item):
        return cls(item['name'], np.asarray(item['points'], float),
                   dict(ref_x_m=item['ref_x_m'], offset_m=item['offset_m'], heading_rad=item['heading_rad'],
                        curvature_1pm=item['curvature_1pm']),
                   float(item['width_m']), bool(item.get('opposite_direction')), bool(item.get('lane_like', True)),
                   item.get('left_boundary'), item.get('right_boundary'),
                   float(item['both_m'][0]) if item.get('both_m') else math.inf)

    def at(self, x):
        return float(centre_at(self.points, [x], self.ref)[0])


def crossable(boundary):
    """White dashed: one may change lanes over it."""
    return boundary is not None and boundary.get('style') == 'dashed' and not boundary.get('yellow')


def forbidden(boundary):
    """Known yellow or solid: not to be crossed (an unknown style is neither)."""
    return boundary is not None and (bool(boundary.get('yellow')) or boundary.get('style') == 'solid')


@dataclass
class Output:
    state: str  # none / keep / return / change / forbidden
    target: str  # the target lane's current name (ego / left / right), 'carried', or '' when none
    offset_m: float  # the car's lateral offset from the target centre (+: car left of it)
    lateral_speed_mps: float
    speed_mps: float
    shown: bool
    path: np.ndarray = None
    merge_m: float = 0.
    changes: int = 0
    reason: str = ''


class LaneGuidance:
    def __init__(self, settings=None):
        self.s = settings or GuidanceSettings(); self.reset()

    def reset(self):
        self.target = None  # (points, ref, width, left, right) of the target, last base_link
        self.target_time = None; self.pose = None; self.time = None; self.acquired = None
        self.speed = 0.; self.yaw_rate = 0.; self.lateral_speed = 0.; self.offset = None; self.smooth = None
        self.moving = {'left': 0., 'right': 0.}; self.gap = {'left': 0., 'right': 0.}
        self.memory = []; self.shown = False; self.changes = 0; self.state = 'none'

    def _lose(self):
        self.target = None; self.memory = []; self.offset = self.smooth = None

    def _near(self, lane, pose):
        """(y, slope, curvature) of the target centre at x = 0, from the remembered points, else continued."""
        s = self.s
        if pose is not None and self.memory:
            q = to_base(np.vstack([p for _, p in self.memory]), pose)
            q = q[np.abs(q[:, 0]) <= s.near_window_m]
            if len(q) >= 6 and np.ptp(q[:, 0]) >= 1.:
                b, a = np.polyfit(q[:, 0], q[:, 1], 1)
                return float(a), float(b), lane.ref['curvature_1pm'] if lane.ref else 0.
        h = 1e-3; c0, cp, cm = centre_at(lane.points, [0., h, -h], lane.ref)
        return float(c0), float((cp-cm)/(2*h)), float((cp-2*c0+cm)/(h*h))

    def update(self, t, pose, lanes, path=True):
        """t (s), pose (x, y, yaw) in a world frame or None, lanes: Lane list of this frame.  Returns Output.
        path=False: the path only while the guidance is shown."""
        s = self.s
        dt = 0. if self.time is None else t-self.time
        if self.time is not None and not 0. < dt <= 1.:self.reset();dt = 0.
        motion = relative(self.pose, pose) if pose is not None and self.pose is not None and dt > 0 else None
        if motion is not None:
            a = 1-math.exp(-dt/s.speed_tau_sec)
            self.speed += (math.hypot(motion[0], motion[1])/dt-self.speed)*a
            self.yaw_rate += (motion[2]/dt-self.yaw_rate)*a
        self.pose = pose; self.time = t
        named = {l.name: l for l in lanes}
        turning = abs(self.yaw_rate) > s.turn_rate
        if turning:self._lose()
        # The target, by position: last frame's carried into this one, matched among the current lanes.
        target = None
        if self.target is not None and motion is not None:
            points, ref, width, left, right = self.target
            yc = float(centre_at(carry(points, motion), [s.ref_m])[0])
            best = min(lanes, key=lambda l: abs(l.at(s.ref_m)-yc), default=None)
            if best is not None and abs(best.at(s.ref_m)-yc) <= s.match_m:
                target = best; self.target_time = t
            elif t-self.target_time <= s.carry_sec:
                target = Lane('carried', carry(points, motion), None, width, left=left, right=right)
        if target is None:
            target = named.get('ego')
            if target is None:
                self._lose(); self.state = 'none'; self.shown = False
                return Output('none', '', 0., 0., self.speed, False, changes=self.changes, reason='no lane')
            # Taken afresh: the car's lane, settled first.
            self._lose(); self.target_time = t; self.acquired = t
            self.lateral_speed = 0.; self.moving = {'left': 0., 'right': 0.}
        # Unreliable surroundings (a junction: turning, or no measured lane near): the car's own lane,
        # taken afresh every frame, so the target does not stick to a misnamed lane on the way out.
        if target.name != 'carried' and (turning or target.measured_from > s.near_m) and 'ego' in named:
            if target is not named['ego'] or self.acquired is None or t-self.acquired >= s.settle_sec:
                self._lose();target = named['ego']
            self.acquired = t;self.target_time = t;self.lateral_speed = 0.;self.moving = {'left': 0., 'right': 0.}
        near = self._near(target, pose); offset = -near[0]
        limit = (1. if target.name == 'carried' else s.implausible)*target.width
        if abs(offset) > limit:
            self._lose(); self.state = 'none'; self.shown = False
            return Output('none', target.name, offset, self.lateral_speed, self.speed, False, changes=self.changes, reason='implausible offset')
        # Lateral speed from the offset change (the target centre is fixed on the road).
        if self.offset is not None and dt > 0:
            raw = (offset-self.offset)/dt
            if abs(raw) < 5.:self.lateral_speed += (raw-self.lateral_speed)*(1-math.exp(-dt/s.speed_tau_sec))
        self.offset = offset
        self.smooth = offset if self.smooth is None or dt <= 0 else self.smooth+(offset-self.smooth)*(1-math.exp(-dt/s.offset_tau_sec))
        for side, sign in (('left', 1.), ('right', -1.)):
            if sign*self.lateral_speed >= s.change_speed_mps:self.moving[side] += dt;self.gap[side] = 0.
            else:
                self.gap[side] += dt
                if self.gap[side] > s.change_gap_sec:self.moving[side] = 0.
        # Moving over a dashed line into a lane-like, same-direction neighbour (found by position).
        reason = ''; changed = False
        for side, sign in (('left', 1.), ('right', -1.)):
            if self.moving[side] < s.change_sec or target.name == 'carried':continue
            yt = target.at(s.ref_m)
            near_lanes = [l for l in lanes if l is not target and .6*target.width <= sign*(l.at(s.ref_m)-yt) <= 1.5*target.width]
            if not near_lanes:continue
            n = min(near_lanes, key=lambda l: abs(sign*(l.at(s.ref_m)-yt)-target.width))
            shared = target.left if side == 'left' else target.right
            close = sign*offset >= .5*target.width-s.change_line_m
            ok = crossable(shared) and n.lane_like and not n.opposite
            if close and ok:
                target = n;self.memory = [];near = self._near(n, pose);offset = -near[0]
                self.offset = self.smooth = offset;self.changes += 1;changed = True
                self.moving = {'left': 0., 'right': 0.};reason = 'moved over a dashed line to the '+side;break
            if close and not ok:reason = 'the line to the '+side+' may not be crossed'
        self.target = (target.points, target.ref, target.width, target.left, target.right)
        # Remember the measured part of the target centre ahead (world coordinates).
        if pose is not None and target.name != 'carried':
            p = target.points[(target.points[:, 0] >= max(s.ref_m, target.measured_from)) & (target.points[:, 0] <= s.ref_m+8.)]
            if len(p):self.memory.append((t, to_world(p, pose)))
            self.memory = [(u, p) for u, p in self.memory if t-u <= s.memory_sec]
        # State (on the filtered offset).
        e = self.smooth
        unsure = turning or (self.acquired is not None and t-self.acquired < s.settle_sec) or \
            (target.name != 'carried' and target.measured_from > s.near_m) or \
            (target.name == 'carried' and self.state == 'none')  # carried on only what was reliable
        side = 'left' if e > 0 else 'right'
        shared = target.left if side == 'left' else target.right
        on_line = abs(e) >= .5*target.width-s.half_width_m
        if unsure and not changed:state = 'none'
        elif changed or (self.state == 'change' and abs(e) > s.off_m):state = 'change'
        elif on_line and forbidden(shared):state = 'forbidden'
        elif on_line and self.state == 'forbidden':state = 'forbidden'
        else:state = 'return' if abs(e) >= (s.off_m if self.shown else s.on_m) else 'keep'
        self.shown = state not in ('keep', 'none');self.state = state
        merge = float(np.clip(max(self.speed*s.merge_sec, s.merge_per_m*abs(offset)), s.merge_min_m, s.merge_max_m))
        line = path_onto(lambda x: centre_at(target.points, x, target.ref), near, merge, s) if path or self.shown else None
        return Output(state, target.name, offset, self.lateral_speed, self.speed, self.shown, line, merge, self.changes, reason)
