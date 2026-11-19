"""Plan 29: light selection (centre and size), the bridge's messages, traffic advice, speed limit, ribbon cut and
the coach's lane ribbon fallback."""
from types import SimpleNamespace

import pytest

from hero_coach.traffic_signs import SignTracker, SignSettings
from hero_coach.traffic_advice import advise, AdviceSettings, SpeedLimit, cut_at, STOP_ADVICE
from hero_coach.lane_ribbon import lane_ideal
from hero_coach import traffic_sign_bridge as bridge


def det(name, conf, box):
    return dict(class_name=name, confidence=conf, xyxy=list(box))


def test_central_large_light_beats_a_more_confident_small_edge_light():
    small_edge = det('traffic_light_green', .9, (1850, 100, 1860, 120))   # x 1855 of 1920, 10x20 px
    big_centre = det('traffic_light_red', .7, (940, 80, 980, 170))        # x 960, 40x90 px
    new = SignTracker()
    for k in range(2):new.update(k*.033, [small_edge, big_centre], 1920)
    assert new.state(.04)[0] == 'red'
    old = SignTracker(SignSettings(select_area_power=0., select_centre_sigma=0.))  # plan 27: highest confidence
    for k in range(2):old.update(k*.033, [small_edge, big_centre], 1920)
    assert old.state(.04)[0] == 'green'


def test_without_image_width_no_centre_weighting():
    t = SignTracker(SignSettings(select_area_power=0.))
    a = det('traffic_light_red', .8, (0, 0, 10, 10));b = det('traffic_light_green', .7, (950, 0, 960, 10))
    for k in range(2):t.update(k*.033, [a, b])
    assert t.state(.04)[0] == 'red'


class FakeSignal:
    STATE_NONE, STATE_RED, STATE_YELLOW, STATE_GREEN = 0, 1, 2, 3
    def __init__(self):self.state = 0


class FakeSign:
    KIND_UNKNOWN, KIND_SPEED, SOURCE_FRONT = 0, 2, 1
    def __init__(self):self.kind = 0;self.value = 0.;self.confidence = 0.;self.source = 0


def test_bridge_sends_while_confirmed_and_none_once_when_off():
    tr = SimpleNamespace(sign_confidence=.8)
    m = bridge.messages(FakeSignal, FakeSign, 'red', 60., (None, None), tr)
    assert [type(x).__name__ for x in m] == ['FakeSignal', 'FakeSign'] and m[0].state == 1 and (m[1].kind, m[1].value) == (2, 60.)
    assert m[1].source == 1 and m[1].confidence == .8
    m = bridge.messages(FakeSignal, FakeSign, 'red', 60., ('red', 60.), tr)  # again while confirmed
    assert len(m) == 2
    m = bridge.messages(FakeSignal, FakeSign, None, None, ('red', 60.), tr)  # off: NONE / UNKNOWN once
    assert m[0].state == 0 and m[1].kind == 0
    assert bridge.messages(FakeSignal, FakeSign, None, None, (None, None), tr) == []


S = AdviceSettings()
OH = S.front_overhang_m


@pytest.mark.parametrize('light,stop,v,want', [
    ('green', 10., 10., 'none'), (None, 10., 10., 'none'),
    ('red', 10., .2, 'hold'), ('red', None, .2, 'hold'),
    ('red', None, 10., 'stop_unknown_distance'), ('yellow', None, 10., 'stop_unknown_distance'),
    ('red', OH-.5, 5., 'over_line'),
    ('red', OH+25., 10., 'prepare_stop'),          # a = 100/50 = 2.0: the boundary is prepare
    ('red', OH+25.-1e-6, 10., 'stop'),
    ('red', OH+100/7., 10., 'stop'),               # a = 3.5: the boundary is stop
    ('red', OH+100/7.-1e-6, 10., 'stop_hard'),
    ('yellow', OH+25., 10., 'prepare_stop'), ('yellow', OH+100/7., 10., 'stop'),
    ('yellow', OH+100/7.-1e-6, 10., 'proceed'), ('yellow', OH-1., 10., 'proceed'), ('yellow', OH+5., .1, 'hold'),
])
def test_advice_branches(light, stop, v, want):
    assert advise(light, stop, v, S)['advice'] == want


def test_advice_numbers():
    out = advise('red', OH+20., 10., S)
    assert out == dict(advice='stop', distance_m=20., decel_mps2=2.5)
    assert advise('red', None, 10., S) == dict(advice='stop_unknown_distance', distance_m=None, decel_mps2=None)
    assert set(STOP_ADVICE) == {'hold', 'prepare_stop', 'stop', 'stop_hard'}


def test_speed_limit_kept_and_over_after_one_second():
    s = SpeedLimit(S)
    assert s.update(0., None, 20.) == (None, False)
    assert s.update(1., 60., 60/3.6) == (60., False)
    assert s.update(2., None, 66/3.6) == (60., False)   # over by 6 km/h: starts the second
    assert s.update(2.9, None, 66/3.6) == (60., False)
    assert s.update(3.0, None, 66/3.6) == (60., True)
    assert s.update(3.1, None, 64/3.6) == (60., False)  # within +5
    assert s.update(4., 30., 20/3.6) == (30., False)    # the next sign replaces it


def test_cut_at_the_stop_line():
    pts = [(0., 0.), (5., .5), (10., 1.), (15., 1.)]
    assert cut_at(pts, 7.5) == [(0., 0.), (5., .5), (7.5, .75)]
    assert cut_at(pts, 20.) == pts
    assert cut_at(pts, -1.) == []


def test_lane_ribbon_only_without_a_parking_segment():
    r = SimpleNamespace(points=[1, 2, 3])
    assert lane_ideal(False, r, .1) is r
    assert lane_ideal(True, r, .1) is None          # the parking plan wins
    assert lane_ideal(False, r, .6) is None         # stale
    assert lane_ideal(False, r, None) is None
    assert lane_ideal(False, SimpleNamespace(points=[1]), .1) is None
    assert lane_ideal(False, None, .1) is None
