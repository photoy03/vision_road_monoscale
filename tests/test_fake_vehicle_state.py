"""tools/fake_vehicle_state.py: speed, yaw rate and steering of a kinematic bicycle from poses."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from fake_vehicle_state import VehicleStateEstimator, WHEELBASE_M, STEERING_RATIO  # noqa: E402


def drive(radius, speed, seconds=3., rate=60., reverse=False):
    """Rear axle on a circle (radius > 0: turning left; None: straight) at speed; returns the last estimate."""
    est = VehicleStateEstimator(); e = None
    for i in range(int(seconds*rate)):
        t = i/rate; s = speed*t
        if radius is None:x, y, yaw = s, 0., 0.
        else:
            a = s/radius;x, y, yaw = radius*math.sin(a), radius*(1-math.cos(a)), a
        if reverse:x, y = -x, -y  # backing along the same heading
        e = est.update(t, x, y, yaw)
    return e


def test_straight():
    e = drive(None, 10.)
    assert e.speed_mps == pytest.approx(10., rel=1e-3) and abs(e.steering_rad) < 1e-6


def test_left_and_right_circles_give_the_bicycle_angle():
    e = drive(40., 8.)
    assert e.yaw_rate_rps == pytest.approx(8./40., rel=1e-2)
    assert e.steering_rad == pytest.approx(math.atan(WHEELBASE_M/40.), rel=1e-2)
    assert e.handwheel_rad == pytest.approx(e.steering_rad*STEERING_RATIO)
    r = drive(-40., 8.)
    assert r.steering_rad == pytest.approx(-e.steering_rad, rel=1e-6)


def test_reverse_speed_is_negative_and_stopped_is_straight():
    assert drive(None, 3., reverse=True).speed_mps == pytest.approx(-3., rel=1e-3)
    assert drive(10., .1).steering_rad == 0.  # below 0.5 m/s no curvature is claimed
