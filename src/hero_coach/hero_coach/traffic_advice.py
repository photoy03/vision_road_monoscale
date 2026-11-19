"""Traffic light + stop line advice, and the speed limit (plan 29).  Pure Python, no ROS.

lane_guidance_node puts the result into /hero/lane_guidance as "traffic" and, for a red or yellow light with a stop
advice, ends the lane guidance ribbon at the stop line.

Distance to the car's front end: d = stop_line_m (rear axle to the line) - front_overhang_m (rear axle to the front
bumper: 3.656 m, Prius gen 3, docs/설계.md).  Deceleration needed: a = v^2 / (2 max(d, 0.1 m)).

  light red:    |v| < stopped_mps                    -> 'hold'
                no stop line known                   -> 'stop_unknown_distance' (distance and deceleration null)
                d <= 0 and moving                    -> 'over_line'
                a <= prepare_decel_mps2 (2.0)        -> 'prepare_stop'
                a <= stop_decel_mps2 (3.5)           -> 'stop'
                else                                 -> 'stop_hard'
  light yellow: no stop line known                   -> 'stop_unknown_distance'
                a <= stop_decel_mps2                 -> as red
                else (cannot stop before the line)   -> 'proceed'
  green / none:                                      -> 'none'

Speed limit: the last confirmed speed sign is kept until the next one (speed_limit_kph); over_limit is true once the
speed has been above the limit + over_limit_kph for over_limit_s.
"""
import math
from dataclasses import dataclass

STOP_ADVICE = ('hold', 'prepare_stop', 'stop', 'stop_hard')  # the ribbon ends at the stop line for these


@dataclass
class AdviceSettings:
    front_overhang_m: float = 3.656   # rear axle to the front bumper (Prius gen 3, docs/설계.md)
    prepare_decel_mps2: float = 2.    # up to this a gentle stop: prepare
    stop_decel_mps2: float = 3.5      # up to this a firm stop; above it hard (red) or go on (yellow)
    stopped_mps: float = .3           # below this the car is standing
    min_distance_m: float = .1        # d in the deceleration is at least this
    over_limit_kph: float = 5.        # over the speed limit by more than this ...
    over_limit_s: float = 1.          # ... for this long: over_limit


def advise(light, stop_line_m, speed_mps, s=None):
    """dict(advice, distance_m (to the car's front, or None), decel_mps2 (or None))."""
    s = s or AdviceSettings()
    v = abs(float(speed_mps or 0.))
    if light not in ('red', 'yellow'):return dict(advice='none', distance_m=None, decel_mps2=None)
    d = None if stop_line_m is None else float(stop_line_m)-s.front_overhang_m
    a = None if d is None else v*v/(2.*max(d, s.min_distance_m))
    out = lambda advice: dict(advice=advice, distance_m=None if d is None else round(d, 2), decel_mps2=None if a is None else round(a, 2))
    if light == 'red' and v < s.stopped_mps:return out('hold')
    if d is None:return dict(advice='stop_unknown_distance', distance_m=None, decel_mps2=None)
    if light == 'yellow' and a > s.stop_decel_mps2:return out('proceed')
    if light == 'yellow' and v < s.stopped_mps:return out('hold')
    if d <= 0. and v >= s.stopped_mps:return out('over_line')
    if a <= s.prepare_decel_mps2:return out('prepare_stop')
    if a <= s.stop_decel_mps2:return out('stop')
    return out('stop_hard')


class SpeedLimit:
    """The last confirmed speed sign until the next one, and whether the car is over it."""

    def __init__(self, settings=None):
        self.s = settings or AdviceSettings();self.reset()

    def reset(self):
        self.limit = None;self.over_since = None;self.last = None

    def update(self, t, sign_kph, speed_mps):
        """t (s), the confirmed sign now (km/h or None) and the speed; returns (limit km/h or None, over_limit)."""
        if self.last is not None and t < self.last-.5:self.reset()  # replay restarted
        self.last = t
        if sign_kph is not None:self.limit = float(sign_kph)
        over = False
        if self.limit is not None and speed_mps is not None and abs(speed_mps)*3.6 > self.limit+self.s.over_limit_kph:
            if self.over_since is None:self.over_since = t
            over = t-self.over_since >= self.s.over_limit_s
        else:self.over_since = None
        return self.limit, over


def cut_at(points, x_end):
    """A ribbon (base_link [(x, y), ...], from the car) ending at x = x_end: points past it removed, the last one
    interpolated onto it.  Fewer than 2 points left: []."""
    out = []
    for i, (x, y) in enumerate(points):
        if x <= x_end:out.append((x, y));continue
        if out:
            x0, y0 = out[-1];u = (x_end-x0)/(x-x0) if x != x0 else 0.;out.append((x_end, y0+u*(y-y0)))
        break
    return out if len(out) >= 2 else []
