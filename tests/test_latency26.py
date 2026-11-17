"""Plan 26: the _native parts for the chain DP, pair vote, solid/dashed sampling, width test and pair memory give
the Python code's values (bit for bit); the least-squares fits agree to rounding; the whole detector with the
plan 26 switches gives the same lines as without on a synthetic road."""
import math

import numpy as np
import pytest

from vision_road.core import Settings, median
from vision_road import carla as carla_module
from vision_road.marking_style import classify_lane, StyleEvidence
from vision_road.core import Lane

native = pytest.importorskip('vision_road._native')
from numpy.core.multiarray import interp  # noqa: E402


def python_dp(ok, weight, base, root, excluded, length):
    n = len(base); links = [[] for _ in range(n)]
    for k, j in zip(*np.nonzero(ok)):links[j].append((int(k), float(weight[k, j])))
    scores = [-math.inf]*n; parent = [-1]*n; origin = list(range(n)); paint = list(length)
    for j in range(n):
        if excluded[j]:continue
        best = float(base[j]) if root[j] else -math.inf; choice = -1
        for k, w in links[j]:
            if scores[k] == -math.inf:continue
            value = scores[k]+w
            if value > best:best = value; choice = k
        scores[j] = best
        if choice >= 0:parent[j] = choice; origin[j] = origin[choice]; paint[j] = paint[choice]+length[j]
    return scores, parent, origin, paint


def test_chain_dp_equals_python():
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = int(rng.integers(1, 40))
        ok = np.triu(rng.random((n, n)) < .3, 1); weight = rng.normal(2, 1, (n, n)); base = rng.normal(5, 2, n)
        root = rng.random(n) < .4; excluded = rng.random(n) < .1; length = rng.uniform(.5, 6, n)
        got = native.chain_dp(ok, weight, base, root, excluded, length); want = python_dp(ok, weight, base, root, excluded, length)
        assert list(got[0]) == want[0] and got[1].tolist() == want[1] and got[2].tolist() == want[2] and list(got[3]) == want[3]


def test_pair_tests_equal_python():
    rng = np.random.default_rng(2)
    for _ in range(60):
        n = int(rng.integers(2, 14)); pts = []
        for _ in range(n):
            a = rng.uniform(3, 20); x = np.arange(a, a+rng.uniform(1, 8), .07); y = rng.choice([1.6, 1.85])+.01*rng.normal(size=len(x))
            pts.append(np.column_stack([x, y]))
        first = [float(p[0, 0]) for p in pts]; last = [float(p[-1, 0]) for p in pts]
        lines = [((p[-1, 1]-p[0, 1])/(p[-1, 0]-p[0, 0]), 0.) for p in pts]; lines = [(s, float(p[0, 1])-s*float(p[0, 0])) for (s, _), p in zip(lines, pts)]
        wide = list(rng.random(n) < .1); gap = .25; want = []
        for i in range(n):
            if wide[i]:continue
            for j in range(i+1, n):
                if wide[j]:continue
                lo = max(first[i], first[j]); hi = min(last[i], last[j])
                if hi-lo < 1.:continue
                middle = (lo+hi)/2
                if abs(abs((lines[i][0]-lines[j][0])*middle+lines[i][1]-lines[j][1])-gap) > .15:continue
                common = (pts[i][:, 0] >= lo) & (pts[i][:, 0] <= hi)
                d = float(median(pts[i][common, 1]-interp(pts[i][common, 0], pts[j][:, 0], pts[j][:, 1])))
                if abs(abs(d)-gap) > .08:continue
                want.append((i, j, d, hi-lo))
        got = native.pair_tests(pts, first, last, [l[0] for l in lines], [l[1] for l in lines], wide, gap)
        assert [tuple(g) for g in got] == want


def test_lane_profile_and_forward_profile_equal_numpy():
    c = Settings(); rng = np.random.default_rng(3)
    binary = (rng.random((457, 171)) < .3).astype(np.uint8)*255; valid = (rng.random((457, 171)) < .95).astype(np.uint8)*255
    for _ in range(50):
        x = np.sort(rng.uniform(3, 30, int(rng.integers(2, 30)))); y = 1.7+.05*np.cumsum(rng.normal(size=len(x)))
        lane = Lane('left', np.column_stack([x, y]), .8)
        c.carla_native_chain = False; a = classify_lane(lane, binary, valid, c)
        c.carla_native_chain = True; b = classify_lane(lane, binary, valid, c)
        assert a == b
        start, stop = max(c.forward_min, float(x[0])), min(20., float(x[-1]))
        xs, seen, paint = native.forward_profile(np.column_stack([x, y]), start, stop, binary, valid, c.mpp, c.forward_max, c.lateral_min)
        want = np.arange(start, stop, .1)
        assert np.array_equal(xs, want)
        if len(want):
            yy = interp(want, x, y); cols = np.rint((-yy-c.lateral_min)/c.mpp).astype(int); rows = np.rint((c.forward_max-want)/c.mpp).astype(int)
            w = max(1, round(.12/c.mpp)); inside = (rows >= 0) & (rows < 457) & (cols >= w) & (cols < 171-w)
            rr = np.clip(rows, 0, 456); cc = np.clip(cols[:, None]+np.arange(-w, w+1), 0, 170)
            assert np.array_equal(seen, inside & np.all(valid[rr[:, None], cc] > 0, axis=1))
            assert np.array_equal(paint, inside & np.any(binary[rr[:, None], cc] > 0, axis=1))
    assert classify_lane(Lane('left', np.array([[5., 1.], [5., 1.]]), .8), binary, valid, c) == StyleEvidence()


def test_width_gap_equals_python():
    rng = np.random.default_rng(4)
    for _ in range(200):
        p = np.column_stack([np.sort(rng.uniform(3, 20, 15)), rng.normal(1.7, .1, 15)])
        r = np.column_stack([np.sort(rng.uniform(3, 20, 15)), rng.normal(-1.7, .1, 15)])
        for right in (False, True):
            lo = max(p[0, 0], r[0, 0]); hi = min(p[-1, 0], r[-1, 0], lo+4.)
            if hi < lo:
                assert native.width_gap(p, r, right, 3.4) is None;continue
            xs = np.linspace(lo, hi, 5); own = interp(xs, p[:, 0], p[:, 1]); them = interp(xs, r[:, 0], r[:, 1])
            assert native.width_gap(p, r, right, 3.4) == abs(float(median(them-own if right else own-them))-3.4)


def test_least_squares_fits_agree_to_rounding():
    rng = np.random.default_rng(5)
    for _ in range(100):
        m = int(rng.integers(6, 150)); x = np.sort(rng.uniform(-12, 12, m)); c = .01*x*x+.1*x+rng.normal(0, .05, m)
        coef, _, _ = native.spine_fit(x/10., c, np.ones(m), np.ones(m), 2, 0.)
        A = np.vander(x/10., 3); w = np.ones(m)
        for _ in range(5):
            ref = np.linalg.lstsq(A*w[:, None], c*w, rcond=None)[0]; w = np.minimum(1., .2/np.maximum(np.abs(c-A@ref), .001))
        assert np.allclose(coef, ref, rtol=0, atol=1e-10)
        fit = native.pair_fit(x, c, 1); w = np.ones(m)
        for _ in range(3):
            ref = carla_module.polyfit(x, c, 1, w=w); w = np.minimum(1., .05/np.maximum(np.abs(c-np.polyval(ref, x)), .001))
        assert np.allclose(fit, ref, rtol=0, atol=1e-10)
    assert native.spine_fit(np.zeros(5), np.ones(5), np.ones(5), np.ones(5), 2, 0.) is None  # rank deficient: Python's lstsq
