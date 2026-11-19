"""Traffic light state and speed limit from the YOLO detector's /yolo/detections (plans 27, 29).

Pure Python, no ROS: traffic_sign_bridge feeds it the detection messages and asks it what is confirmed.  Times
are the detection messages' stamps (the camera image's simulation time).

Traffic light: in each detection message the light class (green / orange / red) with the highest score is that
message's state; a message without a light says "none".  Score (plan 29): confidence x (box area / 1000 px^2)
^ select_area_power x a Gaussian of the box centre's distance from the image's horizontal centre (sigma =
select_centre_sigma x image width): with several lights in view, the large one ahead, the own lane's light more
likely than a side road's, wins over a slightly more confident small one at the edge.  select_area_power 0 and
select_centre_sigma 0 give the plain highest confidence (on a tie the larger box) of plan 27.  A state is shown
once it is the state of at least `light_votes` of the last `light_window` messages, and only while the last
message with a light is at most `light_hold_s` old (no light that long: off).
Speed sign: the same vote on the speed value (30 / 60 / 90) of the most confident sign class per message; a
confirmed value stays shown `sign_hold_s` after the last message that saw it.
The detector's position in the image is not used (it gives no distance); the HUD draws both in fixed places.
"""
import math
from collections import deque
from dataclasses import dataclass

LIGHTS = {'traffic_light_green': 'green', 'traffic_light_orange': 'yellow', 'traffic_light_red': 'red'}
SIGNS = {'traffic_sign_30': 30., 'traffic_sign_60': 60., 'traffic_sign_90': 90.}


@dataclass
class SignSettings:
    light_window: int = 3       # the last N detection messages ...
    light_votes: int = 2        # ... at least M of them with the same light state
    light_hold_s: float = .5    # no light in the messages this long: the light is off
    sign_window: int = 3
    sign_votes: int = 2
    sign_hold_s: float = 3.     # a confirmed speed sign stays this long after it was last seen
    min_confidence: float = 0.  # detections below this are ignored (the detector has its own threshold)
    select_area_power: float = .5    # score: confidence x (area/1000 px^2)^this ...
    select_centre_sigma: float = .25  # ... x exp(-((cx - W/2) / (this x W))^2 / 2); 0: no centre weighting


def area(box):
    return max(0., box[2]-box[0])*max(0., box[3]-box[1])


class SignTracker:
    def __init__(self, settings=None):
        self.s = settings or SignSettings();self.reset()

    def reset(self):
        self.lights = deque(maxlen=self.s.light_window);self.signs = deque(maxlen=self.s.sign_window)
        self.light_seen = None;self.light = None      # last stamp with a light; the confirmed state
        self.sign = None;self.sign_seen = None        # the confirmed value and when it was last seen
        self.sign_confidence = 1.;self.light_confidence = 1.
        self.last = None

    def score(self, conf, box, width):
        """The selection score of one detection (see the module text)."""
        value = conf*((area(box)/1000.)**self.s.select_area_power if self.s.select_area_power else 1.)
        if self.s.select_centre_sigma > 0 and width:
            off = ((box[0]+box[2])/2.-width/2.)/(self.s.select_centre_sigma*width);value *= math.exp(-.5*off*off)
        return value

    def update(self, stamp, detections, image_width=None):
        """One detection message: stamp (s), [{class_name, confidence, xyxy}] and the image width (px; None: no
        centre weighting)."""
        if self.last is not None and stamp < self.last-.5:self.reset()  # replay restarted
        self.last = stamp
        # After a gap longer than the hold, start over: an old confirmed state must not come back on one frame.
        if self.light_seen is not None and stamp-self.light_seen > self.s.light_hold_s:self.light = None;self.lights.clear();self.light_seen = None
        if self.sign_seen is not None and stamp-self.sign_seen > self.s.sign_hold_s:self.sign = None;self.signs.clear();self.sign_seen = None
        best = {}
        for d in detections:
            name = d.get('class_name');conf = float(d.get('confidence', 0.))
            if conf < self.s.min_confidence:continue
            kind = 'light' if name in LIGHTS else 'sign' if name in SIGNS else None
            if kind is None:continue
            box = d.get('xyxy', [0, 0, 0, 0]);key = (self.score(conf, box, image_width), area(box))
            if kind not in best or key > best[kind][0]:best[kind] = (key, name, conf)
        light = LIGHTS[best['light'][1]] if 'light' in best else None
        if light is not None and light == self.light:self.light_confidence = best['light'][2]
        self.lights.append(light)
        if light is not None:self.light_seen = stamp
        votes = [v for v in self.lights if v is not None]
        top = max(set(votes), key=votes.count) if votes else None
        if top is not None and votes.count(top) >= self.s.light_votes:self.light = top
        sign = SIGNS[best['sign'][1]] if 'sign' in best else None
        self.signs.append(sign)
        values = [v for v in self.signs if v is not None]
        top = max(set(values), key=values.count) if values else None
        if top is not None and values.count(top) >= self.s.sign_votes:self.sign = top
        if sign is not None and sign == self.sign:self.sign_seen = stamp;self.sign_confidence = best['sign'][2]

    def state(self, t):
        """(light 'red'/'yellow'/'green' or None, speed value or None) to show at time t."""
        light = self.light if self.light is not None and self.light_seen is not None and -.5 <= t-self.light_seen <= self.s.light_hold_s else None
        sign = self.sign if self.sign is not None and self.sign_seen is not None and -.5 <= t-self.sign_seen <= self.s.sign_hold_s else None
        return light, sign
