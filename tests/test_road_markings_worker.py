"""Plan 28: stop lines and crosswalks through the perception worker's Extras, with the plan 25/26 switches on
(as carla.yaml); the same candidates as with them off; confirmed after 3 of 5 frames."""
from dataclasses import replace
from types import SimpleNamespace

import test_road_markings as T
from vision_road.rt_worker import Extras
from vision_road.road_markings import detect

ON = replace(T.C, carla_native_fast=True, carla_native_chain=True, carla_native_fit=True, carla_parallel_binary=True,
             carla_parallel_ops=True, carla_yellow_relative_min=6.)
OFF = replace(T.C, carla_parallel_ops=True, carla_yellow_relative_min=6.)


def scene(kind):
    im = T.canvas(); lanes = T.lanes(im)
    if kind == 'stop':T.box(im, -1.5, 1.5, 12, 12.4)
    else:
        for y in (-1.2, -.4, .4, 1.2):T.box(im, y-.2, y+.2, 10, 13)
    return im, lanes


def test_native_switches_give_the_same_candidates():
    for kind in ('stop', 'zebra'):
        out = []
        for cfg in (OFF, ON):
            im, lanes = scene(kind);d, _, valid = T.run(im, cfg);out.append(detect(d, im, valid, lanes))
        assert out[0] == out[1] and out[1]


def test_worker_extras_confirm_a_stop_line():
    im, lanes = scene('stop');d, _, valid = T.run(im, ON)
    extra = Extras(markings=True);temporal = SimpleNamespace(detector=d, boundary_lanes=lanes)
    got = [extra.after(temporal, im, valid, (0., 0., 0.)) for _ in range(3)]
    assert not got[0] and not got[1]
    assert [m['kind'] for m in got[2]] == ['stop_line']
    assert abs(min(p[0] for p in got[2][0]['points'])-12.) < .5
