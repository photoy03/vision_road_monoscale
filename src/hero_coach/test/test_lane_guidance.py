"""lane_guidance.py: pure functions and the target rules, on synthetic straight lanes (no ROS)."""
import math

import numpy as np

from hero_coach.lane_guidance import (GuidanceSettings, Lane, LaneGuidance, centre_at, path_onto, quintic)

WIDTH = 3.5


def lane(name, centre_y, left_style='dashed', left_yellow=False, right_style='dashed', right_yellow=False,
         opposite=False):
    """A straight lane whose centre is centre_y (base_link) from 5.5 to 30 m ahead."""
    x = np.arange(5.5, 30., .5)
    return Lane(name, np.column_stack([x, np.full_like(x, centre_y)]),
                dict(ref_x_m=5.5, offset_m=centre_y, heading_rad=0., curvature_1pm=0.), WIDTH, opposite, True,
                dict(line='l', style=left_style, yellow=left_yellow), dict(line='r', style=right_style, yellow=right_yellow), 5.5)


def road(delta):
    """Home lane (yellow centre line on its left, white dashed on its right), oncoming lane on the left,
    a same-direction lane on the right; the car delta (m, + left) from the home centre.  Named by position."""
    centres = [(WIDTH, 'oncoming'), (0., 'home'), (-WIDTH, 'right lane')]
    lanes = []
    for y, kind in centres:
        c = y-delta
        if kind == 'oncoming':l = lane('', c, 'dashed', False, 'solid', True, opposite=True)
        elif kind == 'home':l = lane('', c, 'solid', True, 'dashed', False)
        else:l = lane('', c, 'dashed', False, 'solid', False)
        l.kind = kind;lanes.append(l)
    ego = min(range(3), key=lambda i: abs(lanes[i].at(0.)))
    for i, l in enumerate(lanes):l.name = {ego: 'ego', ego-1: 'left', ego+1: 'right'}.get(i, 'far')
    return [l for l in lanes if l.name != 'far']


def drive(profile, seconds, rate=30., speed=10.):
    """Run the guidance along a straight road at speed with lateral position profile(t). Returns outputs and lanes."""
    g = LaneGuidance(); outs = []
    for i in range(int(seconds*rate)):
        t = i/rate; d = profile(t); lanes = road(d)
        o = g.update(t, (speed*t, d, 0.), lanes)
        kind = next((l.kind for l in lanes if l.name == o.target), o.target)
        outs.append((t, d, o, kind))
    return outs


def ramp(t, t0, t1, a, b):
    if t <= t0:return a
    if t >= t1:return b
    u = (t-t0)/(t1-t0);return a+(b-a)*(1-math.cos(math.pi*u))/2


def test_quintic_meets_both_ends():
    c = quintic(.8, -.05, .01, 20.)
    d = np.polynomial.polynomial.Polynomial(c)
    assert abs(d(0.)-.8) < 1e-12 and abs(d.deriv()(0.)+.05) < 1e-12 and abs(d.deriv(2)(0.)-.01) < 1e-12
    for k in range(3):assert abs(d.deriv(k)(20.)) < 1e-9


def test_path_leaves_the_car_along_its_heading_and_ends_on_the_centreline():
    s = GuidanceSettings()
    x = np.arange(5., 40., .5); pts = np.column_stack([x, -1.2+.02*x+.001*x*x])
    ref = dict(ref_x_m=5., offset_m=float(pts[0, 1]), heading_rad=math.atan(.02+.01), curvature_1pm=.002)
    centre = lambda q: centre_at(pts, q, ref)
    near = (float(centre([0.])[0]), .02, .002)
    path = path_onto(centre, near, 20., s)
    assert np.allclose(path[0], [0., 0.], atol=1e-9)
    assert abs((path[1, 1]-path[0, 1])/(path[1, 0]-path[0, 0])) < .01  # along the heading (+x)
    end = path[path[:, 0] >= 20.]
    assert np.max(np.abs(end[:, 1]-centre(end[:, 0]))) < 1e-6  # on the centreline from the merge on
    slope = np.gradient(path[:, 1], path[:, 0]);i = int(np.searchsorted(path[:, 0], 20.))
    assert abs(slope[i]-(.02+.002*20.)) < .01  # and along it (centre slope .02 + .002 x)


def test_near_the_centre_nothing_is_shown_and_on_off_differ():
    outs = drive(lambda t: .25 if t > 1 else 0., 3.)
    assert not any(o.shown for _, _, o, _ in outs)
    outs = drive(lambda t: ramp(t, 1., 1.5, 0., .4) if t < 3. else ramp(t, 3., 3.5, .4, .2), 5.)
    late = [o for t, _, o, _ in outs if t > 4.]
    assert late and all(o.shown and o.state == 'return' for o in late)  # 0.2 m: still shown (off below 0.15 m)


def test_crossing_the_yellow_line_never_targets_the_oncoming_lane():
    outs = drive(lambda t: ramp(t, 1., 3., 0., 2.6) if t < 5. else ramp(t, 5., 7., 2.6, 0.), 9.)
    assert all(kind == 'home' for t, _, o, kind in outs if o.state != 'none')
    assert any(o.state == 'forbidden' for _, _, o, _ in outs)
    assert outs[-1][2].changes == 0


def test_straddling_the_dashed_line_and_back_keeps_the_target():
    outs = drive(lambda t: ramp(t, 1., 2.5, 0., -1.15) if t < 3.5 else ramp(t, 3.5, 5., -1.15, 0.), 7.)
    assert outs[-1][2].changes == 0
    assert all(kind == 'home' for _, _, o, kind in outs if o.state != 'none')


def test_moving_over_the_dashed_line_moves_the_target_once_by_position():
    outs = drive(lambda t: ramp(t, 1., 5., 0., -WIDTH), 12.)
    assert outs[-1][2].changes == 1
    switch = next(t for (t, _, a, _), (_, _, b, _) in zip(outs, outs[1:]) if b.changes != a.changes)
    crossed = next(t for t, d, _, _ in outs if d <= -WIDTH/2)
    assert switch <= crossed+.2
    assert outs[-1][3] == 'right lane' and outs[-1][2].state == 'keep'
    # Renamed by the detector once over the line, the target stays the right lane by position.
    names = {o.target for t, _, o, kind in outs if t > 6.}
    assert names == {'ego'} and {kind for t, _, _, kind in outs if t > 6.} == {'right lane'}


def test_wobble_inside_the_lane_returns_without_moving_the_target():
    outs = drive(lambda t: .45*math.sin(2*math.pi*t/4.), 12.)
    assert outs[-1][2].changes == 0
    assert any(o.state == 'return' for _, _, o, _ in outs) and not any(o.state in ('forbidden', 'change') for _, _, o, _ in outs)


# ---- plan 30 settings (all off by default) ----

def drive_with(settings, lanes_of, profile, seconds, rate=30., speed=10.):
    """drive() with GuidanceSettings and a lanes_of(delta) road."""
    g = LaneGuidance(settings); outs = []
    for i in range(int(seconds*rate)):
        t = i/rate; d = profile(t); lanes = lanes_of(d)
        outs.append((t, d, g.update(t, (speed*t, d, 0.), lanes)))
    return outs


def test_offset_at_m_takes_the_offset_further_along_the_car():
    # The centre runs at 0.1 rad to the car: 1.4 m further forward it is 0.14 m further left.
    x = np.arange(5.5, 30., .5);pts = np.column_stack([x, -.5+.1*x])
    l = Lane('ego', pts, dict(ref_x_m=5.5, offset_m=float(pts[0, 1]), heading_rad=math.atan(.1), curvature_1pm=0.), WIDTH,
             left=dict(style='dashed'), right=dict(style='dashed'), measured_from=5.5)
    for at, want in ((0., .5), (1.4, .5-.14)):
        o = LaneGuidance(GuidanceSettings(offset_at_m=at)).update(0., (0., 0., 0.), [l])
        assert abs(o.offset_m-want) < 1e-6


def only_home(delta, left_style='solid', left_yellow=True):
    """Only the home lane detected (no neighbour beyond either line), the car delta (m, + left) from its centre."""
    l = lane('ego', -delta, left_style, left_yellow, 'dashed', False);l.kind = 'home';return [l]


def test_change_without_neighbour_moves_over_a_white_dashed_line_only():
    move = lambda t: ramp(t, 1., 5., 0., -WIDTH)
    # The detector renames the lanes as it crosses: past the line the car is in "ego" again, centred on the new lane.
    def renamed(d):
        return only_home(d if d > -WIDTH/2 else d+WIDTH)
    off = drive_with(GuidanceSettings(), renamed, move, 10.)
    on = drive_with(GuidanceSettings(change_without_neighbour=True), renamed, move, 10.)
    assert off[-1][2].changes == 0
    assert on[-1][2].changes == 1 and on[-1][2].state == 'keep'
    # Over the yellow line (left) it never moves.
    left = drive_with(GuidanceSettings(change_without_neighbour=True), lambda d: only_home(d if d < WIDTH/2 else d-WIDTH),
                      lambda t: ramp(t, 1., 5., 0., WIDTH), 10.)
    assert left[-1][2].changes == 0


def test_trust_single_line_keeps_guidance_beside_one_measured_line():
    # An estimated own lane (no both_m) beside one measured white line on its right: the car 0.6 m right of the centre.
    def one_line(d):
        x = np.arange(6., 30., .5);c = -d
        return [Lane('ego', np.column_stack([x, np.full_like(x, c)]), dict(ref_x_m=6., offset_m=c, heading_rad=0., curvature_1pm=0.),
                     WIDTH, left=None, right=dict(style='solid', reprojected=False), measured_from=math.inf, single_from=6.)]
    plain = drive_with(GuidanceSettings(), one_line, lambda t: -.6, 3.)
    trusted = drive_with(GuidanceSettings(trust_single_line=True), one_line, lambda t: -.6, 3.)
    assert not any(o.shown for _, _, o in plain)
    assert trusted[-1][2].shown and trusted[-1][2].state == 'return'


def test_on_off_settings_move_the_hysteresis():
    s = GuidanceSettings(on_m=.25, off_m=.20)
    outs = drive_with(s, road, lambda t: .27 if t > 1 else 0., 3.)
    assert outs[-1][2].shown  # 0.27 m: shown from 0.25 m
    outs = drive_with(s, road, lambda t: ramp(t, 1., 1.5, 0., .4) if t < 3. else ramp(t, 3., 3.5, .4, .18), 5.)
    assert not outs[-1][2].shown  # 0.18 m: hidden below 0.20 m
