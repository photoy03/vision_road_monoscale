"""traffic_signs.SignTracker: light state and speed sign from YOLO detection messages (plan 27)."""
from hero_coach.traffic_signs import SignTracker, SignSettings


def det(name, conf=.9, box=(0, 0, 10, 20)):
    return dict(class_name=name, confidence=conf, xyxy=list(box))


def test_nothing_without_detections():
    t = SignTracker()
    assert t.state(1.) == (None, None)
    t.update(1., [])
    assert t.state(1.) == (None, None)


def test_light_needs_two_of_three_and_goes_off_after_half_a_second():
    t = SignTracker()
    t.update(0., [det('traffic_light_red')])
    assert t.state(0.) == (None, None)           # one vote
    t.update(.033, [det('traffic_light_red')])
    assert t.state(.033)[0] == 'red'
    t.update(.066, [det('traffic_light_green')])  # one other frame: still red
    assert t.state(.066)[0] == 'red'
    t.update(.1, [])                               # no light: still within 0.5 s
    assert t.state(.4)[0] == 'red'
    assert t.state(.6)[0] is None                  # 0.5 s after the last light
    # after the gap an old state does not come back on one frame
    t.update(1.5, [det('traffic_light_green')])
    assert t.state(1.5)[0] is None
    t.update(1.533, [det('traffic_light_green')])
    assert t.state(1.533)[0] == 'green'


def test_orange_is_yellow_and_the_most_confident_or_larger_light_wins():
    t = SignTracker()
    for k in range(2):t.update(k*.033, [det('traffic_light_orange', .8), det('traffic_light_red', .5)])
    assert t.state(.04)[0] == 'yellow'
    t = SignTracker()
    for k in range(2):t.update(k*.033, [det('traffic_light_red', .7, (0, 0, 5, 5)), det('traffic_light_green', .7, (0, 0, 20, 40))])
    assert t.state(.04)[0] == 'green'


def test_speed_sign_held_three_seconds():
    t = SignTracker()
    t.update(0., [det('traffic_sign_60')]);t.update(.033, [det('traffic_sign_60'), det('traffic_light_red')])
    assert t.state(.04) == (None, 60.)
    for k in range(2, 30):t.update(k*.033, [])
    assert t.state(.033+2.9)[1] == 60. and t.state(.033+3.1)[1] is None
    t.update(5., [det('traffic_sign_30')])
    assert t.state(5.)[1] is None                  # one vote after the hold ran out
    t.update(5.033, [det('traffic_sign_30')])
    assert t.state(5.04)[1] == 30.


def test_settings_and_replay_restart():
    t = SignTracker(SignSettings(light_votes=1, min_confidence=.5))
    t.update(10., [det('traffic_light_red', .4)])
    assert t.state(10.)[0] is None                 # below min_confidence
    t.update(10.033, [det('traffic_light_red', .6)])
    assert t.state(10.04)[0] == 'red'
    t.update(1., [])                               # stamps went back: a replay restarted
    assert t.state(1.) == (None, None)
