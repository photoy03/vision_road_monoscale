#!/usr/bin/env python3
"""Ground-truth evaluation of the lane side on the Town05 scenes recorded with sidecar/gt_frames.jsonl (plan 30).

  predict SCENE OUT   the detector (carla.yaml as road_node runs it: outer lines, centrelines, road markings, the
                      _native paths) over the scene's offline BEV cache, frame by frame, then lane_guidance's
                      LaneGuidance on the centrelines (pose: the cache's rear-axle poses, speed: the sidecar's);
                      writes OUT/predictions.jsonl.gz
  yolo SCENE OUT      (the YOLO environment's Python) the sign detector over every camera frame from the scene's
                      lossless image cache, as src/yolo_signs does it; writes OUT/yolo.jsonl.gz
  metrics SCENE OUT   compares both with the sidecar ground truth; writes OUT/metrics.json (and prints a summary)

Coordinates: the sidecar's *_vehicle points are relative to the car origin (CARLA actor origin; README "좌표 약속");
base_link (the detector's frame, the rear axle as the launch uses it) is 1.4096 m behind it: x_base = x_vehicle +
BASE_FROM_ORIGIN.  Frames in a junction (is_junction) are left out of every lane metric.
Scenes: s1_lanefollow ... s8d_lanechange_50kmh (town05_<name>_20261119 under DISK/rosbag), see SCENES."""
import argparse, gzip, json, math, os, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DISK = Path(os.environ.get('VISION_ROAD_DISK', '/media/jaehun/38F7-677B'))
SCENES = ['s1_lanefollow', 's2_wobble', 's3a_zigzag_cross', 's3b_zigzag_inlane', 's4_straddle', 's5_lanechange', 's6_yellow',
          's7_lights', 's8a_lanefollow_20kmh', 's8b_lanefollow_50kmh', 's8c_lanechange_20kmh', 's8d_lanechange_50kmh']
BASE_FROM_ORIGIN = 1.4096  # base_link is this far behind the car origin (sidecar static.json base_link_x_m)
OUTER_SEEN_M = 2.          # A5 recall_visible: frames whose GT neighbour outer line shows this much paint (x 6-24 m)
PAINT_AT_M = (5., 6., 10., 15.)  # paint_stops: GT boundary to paint also at these base_link distances
PAINT_WINDOW_M = 1.         # paint_stops: paint this far from a GT boundary still counts as that line's (s6: 0.83 m off)
GT_PAINT_M = .15           # A1-A3: a side whose GT boundary is further than this from the paint is a GT error
XS = (5., 10., 15.)        # A1: lateral errors at these base_link distances


def name_of(scene):
    return scene if scene.startswith('town05_') else f'town05_{scene}_20261119'


PLAIN = {'town05_30hz': 'town05', 'maploop_30hz': 'maploop'}  # recordings without GT (plan 32 jitter): offline caches only


def paths(scene):
    if scene in PLAIN:
        k = PLAIN[scene];return dict(bag=DISK/'rosbag'/scene, sidecar=DISK/'rosbag'/scene/'sidecar', offline=DISK/'vision-road-cache'/f'{k}_offline',
                                     cdrz=DISK/'vision-road-cache'/f'{k}_30hz_native.cdrz')
    n = name_of(scene)
    return dict(bag=DISK/'rosbag'/n, sidecar=DISK/'rosbag'/n/'sidecar', offline=DISK/'vision-road-cache'/f'{n}_offline',
                cdrz=DISK/'vision-road-cache'/f'{n}_native.cdrz')


def ground_truth(scene):
    f = paths(scene)['sidecar']/'gt_frames.jsonl'
    if scene in PLAIN and not f.exists():return []
    return [facing(json.loads(l)) for l in open(f)]


def facing(r):
    """A GT frame seen from the car.  When the car origin is in a lane of the other direction (s6 over the yellow line)
    the sidecar describes the lanes in that lane's own direction: heading_error near pi, centrelines running backwards,
    left/right and the offset's sign mirrored, and the lane the car is in called 'same'.  Turn that back: relations,
    boundaries, markings and crossable flags swapped, the offset negated, the heading error wrapped, every lane's
    direction flipped (the car's lane is then 'opposite'); marks the frame flipped=True."""
    if abs(r.get('heading_error_rad', 0.)) <= math.pi/2:return r
    r['flipped'] = True;r['lateral_offset_m'] = -r['lateral_offset_m']
    r['heading_error_rad'] = math.atan2(math.sin(r['heading_error_rad']-math.pi), math.cos(r['heading_error_rad']-math.pi))
    swap = {'left': 'right', 'right': 'left'}
    for l in r['lanes']:
        l['relation'] = swap.get(l['relation'], l['relation']);l['direction'] = {'same': 'opposite', 'opposite': 'same'}.get(l.get('direction'), l.get('direction'))
        for k in ('_boundary_vehicle', '_marking', '_crossable'):
            if 'left'+k in l or 'right'+k in l:l['left'+k], l['right'+k] = l.get('right'+k), l.get('left'+k)
    return r


# ------------------------------------------------------------------ predict
def predict(a):
    sys.path.insert(0, str(ROOT/'src/vision_road'));sys.path.insert(0, str(ROOT/'tools'));sys.path.insert(0, str(ROOT/'src/hero_coach'))
    import cv2
    from dataclasses import replace
    import maploop_offline_compare as M
    from vision_road.core import Timeline
    from vision_road.temporal import TemporalDetector
    from vision_road.carla import CarlaDetector
    from vision_road.rt_worker import Extras
    from vision_road.display_filter import simplify
    from hero_coach.lane_guidance import LaneGuidance, Lane, GuidanceSettings
    cv2.setNumThreads(2)
    p = paths(a.scene);pk = ROOT/'src/vision_road';cfg = M.settings(pk)
    for kv in a.set:
        k, v = kv.split('=', 1);cfg = replace(cfg, **{k: type(getattr(cfg, k))(eval(v))})
    meta = json.loads((p['offline']/'frames.json').read_text())
    valid = cv2.imread(str(p['offline']/'valid.png'), 0);valid[M.projector(pk, cfg).my >= 1080*M.LAUNCH['image_roi_bottom']] = 0
    gt = {round(r['stamp'], 4): r for r in ground_truth(a.scene)}
    det = TemporalDetector(cfg);det.detector = CarlaDetector(cfg);extras = Extras(markings=True)
    # The HUD's display line (road_node display_filter_inline: carla.yaml lane_display_filter, run as rt_worker does).
    from vision_road.display_filter import DisplayCore, settings_from, lane_styles
    import yaml
    section = (yaml.safe_load(open(ROOT/'src/vision_road/config/carla.yaml')).get('lane_display_filter') or {}).get('ros__parameters', {})
    section = dict(section)
    for kv in a.display:
        name, v = kv.split('=', 1);section[name] = eval(v)
    display = DisplayCore(settings_from(section))
    tl = Timeline(.25);k = 0;gs = GuidanceSettings()
    for kv in a.guide:
        name, v = kv.split('=', 1);setattr(gs, name, type(getattr(gs, name))(eval(v)))
    guide = LaneGuidance(gs);err = M.LAUNCH['display_simplify_error_m']
    poses = np.array(meta['poses']);poses[:, 3] = np.unwrap(poses[:, 3])
    a.out.mkdir(parents=True, exist_ok=True);clock = []
    L = lambda l: dict(side=l.side, points=np.round(l.points, 3).tolist(), style=l.style, rep=bool(l.reprojected))
    with gzip.open(a.out/'predictions.jsonl.gz', 'wt') as fo:
        for i, row in enumerate(meta['frames']):
            t = row['stamp']
            while k < len(poses) and poses[k][0] <= t+.05:tl.add(*poses[k]);k += 1
            pose = tl.at(t);bev = cv2.imread(str(p['offline']/'bev'/f'{i:05d}.png'))
            s = time.perf_counter();lanes, _ = det.process(bev, valid, t, pose, defer=True);clock.append((time.perf_counter()-s)*1e3)
            paint = [(l.side, simplify(np.asarray(l.points, float), err)) for l in lanes if not l.reprojected and len(l.points) >= 2]
            shown_bounds = [(l.side, simplify(np.asarray(l.points, float), err), l.reprojected) for l in det.boundary_lanes if len(l.points) >= 2]
            display.styles(t, lane_styles(lanes))
            shown = display.step(t, shown_bounds, pose, paint)
            marks = extras.after(det, bev, valid, pose);det.finish()
            centres = [dict(c, points=np.round(simplify(np.asarray(c['points'], float), err), 3).tolist()) for c in det.centre_lanes]
            g = gt.get(round(t, 4));speed = g['speed_mps'] if g else None
            gpose = None if pose is None else tuple(float(np.interp(t, poses[:, 0], poses[:, j])) for j in (1, 2, 3))
            out = guide.update(t, gpose, [Lane.from_message(c) for c in centres], path=False, speed=speed)
            stops = [min(q[0] for q in m['points']) for m in marks if m['kind'] == 'stop_line']
            fo.write(json.dumps(dict(frame=i, stamp=t, bounds=[L(l) for l in det.boundary_lanes],
                                     display=[dict(side=sd, points=np.round(np.asarray(pt, float), 3).tolist(), predicted=bool(pr), style=st) for sd, pt, pr, st in shown], outer=[L(l) for l in det.outer_lanes], centres=centres,
                                     markings=[dict(kind=m['kind'], points=np.round(np.asarray(m['points']), 3).tolist(), predicted=bool(m['predicted'])) for m in marks],
                                     stop_line_m=min([x for x in stops if x > 0], default=None),
                                     guidance=dict(state=out.state, target=out.target, shown=bool(out.shown), changes=out.changes, offset_m=round(out.offset_m, 3),
                                                   speed_mps=round(out.speed_mps, 3)), process_ms=round(clock[-1], 3)))+'\n')
            if i % 500 == 0:print(a.scene, i, len(meta['frames']), flush=True)
    print(a.scene, 'process mean %.2f p95 %.2f ms' % (np.mean(clock), np.percentile(clock, 95)))


# ------------------------------------------------------------------ guidance again on saved centres
def reguide(a):
    """Re-run only LaneGuidance (with --guide overrides) on the centres of an earlier predictions file (--source DIR):
    the same input as in predict, so a guidance change is measured without the detector.  Writes OUT/predictions.jsonl.gz."""
    sys.path.insert(0, str(ROOT/'src/hero_coach'))
    from hero_coach.lane_guidance import LaneGuidance, Lane, GuidanceSettings
    sys.path.insert(0, str(ROOT/'src/vision_road'));sys.path.insert(0, str(ROOT/'tools'))
    from vision_road.core import Timeline
    p = paths(a.scene);meta = json.loads((p['offline']/'frames.json').read_text())
    poses = np.array(meta['poses']);poses[:, 3] = np.unwrap(poses[:, 3])
    gs = GuidanceSettings()
    for kv in a.guide:
        name, v = kv.split('=', 1);setattr(gs, name, type(getattr(gs, name))(eval(v)))
    guide = LaneGuidance(gs);tl = Timeline(.25);k = 0;gt = {round(r['stamp'], 4): r for r in ground_truth(a.scene)}
    a.out.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.source/'predictions.jsonl.gz', 'rt') as fi, gzip.open(a.out/'predictions.jsonl.gz', 'wt') as fo:
        for line in fi:
            d = json.loads(line);t = d['stamp']
            while k < len(poses) and poses[k][0] <= t+.05:tl.add(*poses[k]);k += 1
            pose = tl.at(t);g = gt.get(round(t, 4));speed = g['speed_mps'] if g else None
            gpose = None if pose is None else tuple(float(np.interp(t, poses[:, 0], poses[:, j])) for j in (1, 2, 3))
            out = guide.update(t, gpose, [Lane.from_message(c) for c in d['centres']], path=False, speed=speed)
            d['guidance'] = dict(state=out.state, target=out.target, shown=bool(out.shown), changes=out.changes, offset_m=round(out.offset_m, 3),
                                 speed_mps=round(out.speed_mps, 3))
            fo.write(json.dumps(d)+'\n')


# ------------------------------------------------------------------ yolo
def yolo(a):
    sys.path.insert(0, str(ROOT/'src/vision_road'))
    from vision_road.zstd_cdr import cache_records
    from ultralytics import YOLO
    p = paths(a.scene);model = YOLO(a.model);db = next(p['bag'].glob('*.db3'));half = None
    a.out.mkdir(parents=True, exist_ok=True);times = []
    with gzip.open(a.out/('yolo%s.jsonl.gz' % a.tag), 'wt') as fo:
        for i, raw in enumerate(cache_records(p['cdrz'], db)):
            # The CDR Image: header stamp at byte 4, the pixels last (1920 x 1080 bgr8, as every recording here).
            sec, nanosec = np.frombuffer(raw, '<i4', 1, 4)[0], np.frombuffer(raw, '<u4', 1, 8)[0];h, w = 1080, 1920
            frame = np.frombuffer(raw, np.uint8, h*w*3, len(raw)-h*w*3).reshape(h, w, 3)
            src, off = (frame, 0) if a.full_frame else (frame[:640, 640:1280], 640)
            s = time.perf_counter()
            if half is None:
                try:r = model.predict(source=src, conf=a.conf, imgsz=a.imgsz, device='0', quantize=16, verbose=False)[0];half = False
                except Exception:half = True
            if half:r = model.predict(source=src, conf=a.conf, imgsz=a.imgsz, device='0', half=True, verbose=False)[0]
            times.append((time.perf_counter()-s)*1e3);dets = []
            if r.boxes is not None:
                for xyxy, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
                    xyxy = xyxy.copy();xyxy[[0, 2]] += off
                    dets.append(dict(class_name=model.names[int(k)], confidence=round(float(c), 4), xyxy=[round(float(v), 1) for v in xyxy]))
            fo.write(json.dumps(dict(stamp=float(sec)+float(nanosec)*1e-9, image_width=w, detections=dets))+'\n')
            if i % 500 == 0:print(a.scene, i, flush=True)
    print(a.scene, 'yolo', 'full' if a.full_frame else 'tile', 'mean %.1f p95 %.1f ms' % (np.mean(times[5:]), np.percentile(times[5:], 95)))


# ------------------------------------------------------------------ painted stop lines from the semantic camera
def paint_stops(a):
    """The painted stop line ahead per frame, from the semantic camera (R = 24 RoadLine) warped into the detector's BEV
    (the same projector, nearest pixel): BEV rows where RoadLine covers at least 80 % of the GT own lane's width (between
    its boundaries, 0.3 m in from each), at least 0.1 m thick; the nearest such bar ahead of base_link.  A zebra crossing's
    bars cover about half a row and stay out.  Writes OUT/paint_stops.jsonl.gz (stamp, stop_m or null)."""
    sys.path.insert(0, str(ROOT/'src/vision_road'));sys.path.insert(0, str(ROOT/'tools'))
    import sqlite3, cv2
    import maploop_offline_compare as M
    pk = ROOT/'src/vision_road';cfg = M.settings(pk);proj = M.projector(pk, cfg);p = paths(a.scene)
    gt = {round(r['stamp'], 4): r for r in ground_truth(a.scene)}
    db = next(p['bag'].glob('*.db3'));con = sqlite3.connect('file:'+str(db)+'?mode=ro&immutable=1', uri=True)
    tid = con.execute('select id from topics where name=?', ('/sensing/camera/dashcam/semantic/image_raw',)).fetchone()[0]
    a.out.mkdir(parents=True, exist_ok=True);h, w = 1080, 1920;mpp = cfg.mpp
    with gzip.open(a.out/'paint_stops.jsonl.gz', 'wt') as fo:
        for i, (raw,) in enumerate(con.execute('select data from messages where topic_id=? order by timestamp', (tid,))):
            sec, nanosec = np.frombuffer(raw, '<i4', 1, 4)[0], np.frombuffer(raw, '<u4', 1, 8)[0];t = float(sec)+float(nanosec)*1e-9
            img = np.frombuffer(raw, np.uint8, h*w*3, len(raw)-h*w*3).reshape(h, w, 3)
            line = ((img[:, :, 2] == 24)*255).astype(np.uint8)
            bev = cv2.remap(line, proj.mx, proj.my, cv2.INTER_NEAREST);bev[proj.valid == 0] = 0;bev[proj.my >= h*M.LAUNCH['image_roi_bottom']] = 0
            r = gt.get(round(t, 4));stop = None;crossing = None
            ego = next((l for l in r['lanes'] if l['relation'] == 'ego'), None) if r else None
            if ego is not None:
                left, right = to_base(ego['left_boundary_vehicle']), to_base(ego['right_boundary_vehicle']);bars = [];stripes = []
                for row in range(bev.shape[0]):
                    x = cfg.forward_max-row*mpp;yl, yr = y_at(left, x), y_at(right, x)
                    if yl is None or yr is None or yl-yr < 2.:bars.append(False);stripes.append(False);continue
                    c0 = int(round((-(yl-.3)-cfg.lateral_min)/mpp));c1 = int(round((-(yr+.3)-cfg.lateral_min)/mpp))
                    seg = bev[row, max(0, c0):min(bev.shape[1], c1)];on = seg > 0
                    # Against the part of the row inside the view (near the car the BEV narrows): at least half the
                    # lane width in view and 80 % of that painted.
                    seen = int(np.count_nonzero(proj.valid[row, max(0, c0):min(bev.shape[1], c1)]))
                    bars.append(len(seg) > 5 and seen >= .5*len(seg) and np.count_nonzero(seg) >= .8*seen)
                    # A zebra crossing's stripes: part of the row painted, in at least 3 pieces.
                    pieces = int(np.count_nonzero(on[1:] & ~on[:-1]))+int(on[0]) if len(on) else 0
                    stripes.append(len(seg) > 5 and .2*len(seg) <= np.count_nonzero(seg) <= .75*len(seg) and pieces >= 3)
                runs = [];start = None
                for row, on in enumerate(bars+[False]):
                    if on and start is None:start = row
                    if not on and start is not None:runs.append((start, row));start = None
                # A full-width bar with zebra stripes within 0.5 m of it is the crossing's border, not a stop line (the
                # last junction of the s1 / s8a / s8b route has a crossing and no stop line): kept apart as crosswalk_m.
                near = max(1, int(round(.5/mpp)));stop_xs = [];cross_xs = []
                for r0, r1 in runs:
                    if (r1-r0)*mpp < .1:continue
                    x = cfg.forward_max-(r0+r1-1)/2*mpp
                    beside = sum(stripes[max(0, r0-near):r0])+sum(stripes[r1:r1+near])
                    (cross_xs if beside*mpp >= .15 else stop_xs).append(x)
                ahead = [x for x in stop_xs if x > 0];stop = round(min(ahead), 3) if ahead else None
                ahead = [x for x in cross_xs if x > 0];crossing = round(min(ahead), 3) if ahead else None
            # Does the GT boundary lie on paint?  At x = 6..24 m: the lateral offset from the GT own-lane boundary to the
            # nearest RoadLine pixel within PAINT_WINDOW_M (rows with paint there only); the median per side, or null (no paint).
            paint = {};paint_at = {}
            if ego is not None:
                for side in ('left', 'right'):
                    line = to_base(ego[side+'_boundary_vehicle']);offs = [];at = []
                    for x in np.arange(2., 24.01, .5):
                        yg = y_at(line, x)
                        if yg is None:continue
                        row = int(round((cfg.forward_max-x)/mpp));c0 = int(round((-(yg+PAINT_WINDOW_M)-cfg.lateral_min)/mpp));c1 = int(round((-(yg-PAINT_WINDOW_M)-cfg.lateral_min)/mpp))
                        if row < 0 or row >= bev.shape[0]:continue
                        cols = np.flatnonzero(bev[row, max(0, c0):min(bev.shape[1], c1+1)]) + max(0, c0)
                        if len(cols):
                            o = float(min((-(cfg.lateral_min+cc*mpp)-yg for cc in cols), key=abs));at.append((x, o))
                            if x >= 6.:offs.append(o)
                    paint[side] = round(float(np.median(offs)), 3) if len(offs) >= 3 else None
                    # The same per distance (A1 points and 6 m for B5): the median within +-3 m of x (a dash gap
                    # in between), null with fewer than 3 painted rows there.
                    for x0 in PAINT_AT_M:
                        near_o = [o for x, o in at if abs(x-x0) <= 3.]
                        paint_at.setdefault(side, {})[str(int(x0))] = round(float(np.median(near_o)), 3) if len(near_o) >= 3 else None
            # The neighbouring lanes' outer lines (A5): metres of x = 6..24 with paint within 0.3 m of that GT line.
            outer = {}
            for side in ('left', 'right'):
                nb = next((l for l in r['lanes'] if l['relation'] == side and l.get('lane_type') == 'Driving'), None) if r else None
                if nb is None:continue
                line = to_base(nb[side+'_boundary_vehicle']);seen = 0
                for x in np.arange(6., 24.01, .5):
                    yg = y_at(line, x)
                    if yg is None or not cfg.lateral_min+.3 <= -yg <= cfg.lateral_max-.3:continue  # inside the BEV only
                    row = int(round((cfg.forward_max-x)/mpp));c0 = int(round((-(yg+.3)-cfg.lateral_min)/mpp));c1 = int(round((-(yg-.3)-cfg.lateral_min)/mpp))
                    if 0 <= row < bev.shape[0] and bev[row, max(0, c0):min(bev.shape[1], c1+1)].any():seen += 1
                outer[side] = seen*.5
            fo.write(json.dumps(dict(stamp=t, stop_m=stop, crosswalk_m=crossing, gt_boundary_to_paint_m=paint, gt_boundary_to_paint_at=paint_at,
                                     outer_paint_m=outer))+'\n')
            if i % 500 == 0:print(a.scene, i, flush=True)
    con.close()


# ------------------------------------------------------------------ metrics
def y_at(points, x):
    """y of a polyline (points sorted by x or not) at x, None outside it."""
    p = np.asarray(points, float)
    if len(p) < 2:return None
    p = p[np.argsort(p[:, 0])]
    return float(np.interp(x, p[:, 0], p[:, 1])) if p[0, 0] <= x <= p[-1, 0] else None


def to_base(points):
    p = np.asarray(points, float)[:, :2].copy();p[:, 0] += BASE_FROM_ORIGIN;return p


def style_of(marking):
    t = (marking or {}).get('type', 'None')
    if t in ('Solid', 'SolidSolid'):return 'solid'
    if t in ('Broken', 'BrokenBroken'):return 'dashed'
    return None  # None (no paint), mixed (SolidBroken / BrokenSolid), others: no single answer


def stats(v):
    v = np.asarray([x for x in v if x is not None], float)
    return dict(n=int(len(v)), median=round(float(np.median(v)), 4) if len(v) else None, p95=round(float(np.percentile(v, 95)), 4) if len(v) else None,
                max=round(float(v.max()), 4) if len(v) else None)


def lane_changes(gt):
    """GT lane changes: consecutive frames outside junctions on the same road with another lane id (the car origin crossed
    the line).  [(stamp, from lane, to lane)]."""
    out = [];prev = None
    for r in gt:
        ego = next((l for l in r['lanes'] if l['relation'] == 'ego'), None)
        if ego is None or r['is_junction'] or ego['is_junction'] or r.get('flipped'):prev = None;continue  # not into oncoming lanes
        key = (ego['road_id'], ego['section_id'], ego['lane_id'])
        if prev is not None and key[0] == prev[0] and key[2] != prev[2]:out.append((r['stamp'], prev[2], key[2]))
        prev = key
    return out


C2_AFTER_S = 3.  # C2: frames this soon after the GT's own light went away are left out (see signs_metrics)
ON_LINE_M = .3  # A1-A4: frames with the car centre this close to a line of its lane are left out (counted)
DECIDE_X = 3.  # the detector picks its own lane at this base_link x (between the lines facing each other there)


def lane_at(r, x):
    """The GT driving lane whose boundaries at base_link x hold y = 0, or None."""
    for l in r['lanes']:
        if l.get('lane_type', 'Driving') != 'Driving':continue
        yl, yr = y_at(to_base(l['left_boundary_vehicle']), x), y_at(to_base(l['right_boundary_vehicle']), x)
        if yl is not None and yr is not None and yr <= 0. <= yl:return l
    return None


def metrics(a):
    gt = ground_truth(a.scene);pred = {round(r['stamp'], 4): r for r in (json.loads(l) for l in gzip.open(a.out/'predictions.jsonl.gz', 'rt'))}
    res = dict(scene=a.scene, frames=len(gt));A1 = {f'{s}_{int(x)}m': [] for s in ('left', 'right') for x in XS};A2 = dict(left=[0, 0], right=[0, 0])
    A3s = dict(left=[0, 0, 0], right=[0, 0, 0]);A3c = dict(left=[0, 0, 0], right=[0, 0, 0]);A4o = [];A4h = [];A4e = [];A4z = [];A4oc = [];A4hc = [];A1p = [];A1m = [];A1i = [0, 0]
    A5 = {s: dict(tp=0, fp=0, fn=0, tn=0, opp=[0, 0], cross=[0, 0], vis=[0, 0]) for s in ('left', 'right')};used = 0;bad_gt = []
    shown_far = [0, 0];shown_centre = [0, 0];guidance_frames = [];shown_far_gt = [0, 0];shown_centre_gt = [0, 0];painted = 0
    # GT errors: frames where the GT own-lane boundary is not on the paint (semantic camera, paint_stops) by more than
    # GT_PAINT_M; that side stays out of A1-A3 and is counted in A1_gt_excluded.
    gtp = {};fp = a.out/'paint_stops.jsonl.gz'
    if fp.exists():
        rows = [json.loads(l) for l in gzip.open(fp, 'rt')]
        gtp = {round(d['stamp'], 4): d.get('gt_boundary_to_paint_m') or {} for d in rows}
        gtp_at = {round(d['stamp'], 4): d['gt_boundary_to_paint_at'] for d in rows if 'gt_boundary_to_paint_at' in d}
        outer_seen = {round(d['stamp'], 4): d['outer_paint_m'] for d in rows if 'outer_paint_m' in d}
    else:outer_seen = {}
    if not fp.exists():gtp_at = {}
    excluded = dict(left=[], right=[]);switched = 0;on_line = 0;a4_gt = 0
    for r in gt:
        p = pred.get(round(r['stamp'], 4))
        if p is None:continue
        ego = next((l for l in r['lanes'] if l['relation'] == 'ego'), None)
        if ego is None or r['is_junction'] or ego['is_junction']:continue
        # The own lane as the detector names it: the lane under base_link x = DECIDE_X (the GT names it at the car
        # origin).  They differ only while the car crosses a line; A1_switched counts those frames.
        mine = lane_at(r, DECIDE_X)
        if mine is not None and mine is not ego:switched += 1  # counted only (naming the own lane there scored worse)
        # The car centre on a line (within ON_LINE_M of it): which lane is the own lane is a matter of centimetres, the GT
        # and the detector can each name either; such frames stay out of A1-A4 (B4 scores straddling) and are counted.
        wl, wr = y_at(to_base(ego['left_boundary_vehicle']), BASE_FROM_ORIGIN), y_at(to_base(ego['right_boundary_vehicle']), BASE_FROM_ORIGIN)
        skip_a = wl is not None and wr is not None and abs(r['lateral_offset_m']) >= .5*(wl-wr)-ON_LINE_M;on_line += skip_a
        used += 1;bounds = {b['side']: b for b in p['bounds']};centres = {c['name']: c for c in p['centres']}
        for side in ('left', 'right'):
            mk = ego[side+'_marking'];paint = (mk or {}).get('type', 'None') != 'None'
            if not paint or skip_a:continue
            st = round(r['stamp'], 4)
            if st in gtp_at:  # per distance: a point whose GT boundary is off the paint is a GT error
                at = gtp_at[st].get(side, {});bad = {x for x in XS if at.get(str(int(x))) is not None and abs(at[str(int(x))]) > GT_PAINT_M}
            else:
                off = gtp.get(st, {}).get(side);bad = set(XS) if off is not None and abs(off) > GT_PAINT_M else set()
            if bad:excluded[side].append(r['stamp'])
            if bad == set(XS):continue
            gtl = to_base(ego[side+'_boundary_vehicle']);b = bounds.get(side)
            A2[side][1] += 1;A2[side][0] += b is not None
            for x in XS:
                if x in bad:continue
                yg = y_at(gtl, x);yd = y_at(b['points'], x) if b else None
                if yg is not None and yd is not None:
                    A1[f'{side}_{int(x)}m'].append(abs(yd-yg))
                    # Reference (plan 31): against the paint itself (the GT boundary moved by its measured offset to the
                    # paint at that distance), where the paint was measured.
                    po = gtp_at.get(st, {}).get(side, {}).get(str(int(x)))
                    if po is not None:A1p.append(abs(yd-(yg+po)))
            # Geometry apart from lane identity: the detected line against the nearest GT lane line (any lane's boundary)
            # at 10 m, and whether that nearest line is the GT own lane's line of the same side.  Near a lane crossing the
            # detector names its own lane at 3 m ahead of the rear axle, the GT at the car origin: they can differ.
            yd = y_at(b['points'], 10.) if b else None
            if yd is not None:
                cand = [(abs(yd-yv), l is ego and k == side+'_boundary_vehicle') for l in r['lanes'] for k in ('left_boundary_vehicle', 'right_boundary_vehicle')
                        for yv in [y_at(to_base(l[k]), 10.)] if yv is not None]
                if cand:
                    e, same = min(cand, key=lambda c: (round(c[0], 3), not c[1]))  # a shared line: the own lane's name wins the tie
                    A1m.append(e);A1i[1] += 1;A1i[0] += same
            if b is not None:
                info = (centres.get('ego') or {}).get(side+'_boundary') or {}
                want = style_of(mk);st = info.get('style')
                if want is not None:A3s[side][2] += 1;A3s[side][1] += st in ('solid', 'dashed');A3s[side][0] += st == want
                A3c[side][2] += 1;A3c[side][1] += 1;A3c[side][0] += bool(info.get('yellow')) == ((mk or {}).get('color') == 'Yellow')
        c = None if skip_a else centres.get('ego')
        # A4 at a GT error (either GT boundary off the paint near the reference point): left out and counted.
        at5 = gtp_at.get(round(r['stamp'], 4), {})
        if c is not None and any(abs(v) > GT_PAINT_M for v in (at5.get(k, {}).get('5') for k in ('left', 'right')) if v is not None):
            c = None;a4_gt += 1
        if c is not None:
            # A4 at the same point: the centre's offset and heading at its reference x (5 m, or its first point) against the
            # GT centreline there (lateral_offset_m is at the car origin, outside the view: comparing there would score
            # the extrapolation).  The car-origin comparison is kept apart (A4_at_origin).
            gc = to_base(ego['centerline_vehicle']);x0 = c['ref_x_m'];yg = y_at(gc, x0)
            if yg is not None:
                d = .5;y1, y2 = y_at(gc, x0-d), y_at(gc, x0+d)
                hg = math.atan2(y2-y1, 2*d) if y1 is not None and y2 is not None else None
                (A4e if c.get('estimated') else A4o).append(abs(c['offset_m']-yg))
                if hg is not None and not c.get('estimated'):A4h.append(abs(math.degrees(c['heading_rad']-hg)))
                # For reference (plan 31): the GT centreline has a point every 2 m, so the chord above is one segment's
                # slope; a quadratic through the GT points within x0 +- 4 m instead (A4_*_fit).  It scored slightly worse
                # (s1 heading p95 1.81 -> 2.01 deg): the tails are real errors at curve starts and ends, not GT steps.
                near = gc[np.abs(gc[:, 0]-x0) <= 4.]
                if len(near) >= 3 and not c.get('estimated'):
                    q = np.polyfit(near[:, 0]-x0, near[:, 1], 2)
                    A4oc.append(abs(c['offset_m']-float(q[2])));A4hc.append(abs(math.degrees(c['heading_rad']-math.atan(float(q[1])))))
            y = c['offset_m']+(BASE_FROM_ORIGIN-c['ref_x_m'])*math.tan(c['heading_rad'])
            A4z.append(abs(-y-r['lateral_offset_m']))
        for side in ('left', 'right'):
            if skip_a:break  # on a line: which lanes are the neighbours depends on the same centimetres
            # A detected neighbour counts as a lane only when the detector calls it lane-like (width like the own lane's;
            # s6's parking strip is sent with lane_like false, and the guidance never moves into such a lane).
            g = next((l for l in r['lanes'] if l['relation'] == side and l.get('lane_type') == 'Driving'), None);d = centres.get(side)
            if d is not None and not d.get('lane_like', True):d = None
            seen = outer_seen.get(round(r['stamp'], 4), {}).get(side)
            if g and seen is not None and seen >= OUTER_SEEN_M:q = A5[side];q['vis'][1] += 1;q['vis'][0] += bool(d)
            q = A5[side];q['tp'] += bool(g and d);q['fp'] += bool(d and not g);q['fn'] += bool(g and not d);q['tn'] += bool(not g and not d)
            if g and d:
                q['opp'][1] += 1;q['opp'][0] += bool(d.get('opposite_direction')) == (g['direction'] == 'opposite')
                q['cross'][1] += 1;q['cross'][0] += bool(d.get('crossable')) == bool(ego[side+'_crossable'])
        # B5 against the painted lane: where the GT boundaries are off the paint (GT error, see GT_PAINT_M) the car's
        # offset from the middle between the painted lines; B5_return_shown_gt keeps the plain GT offset.
        sh = p['guidance']['shown'];st = round(r['stamp'], 4)
        if st in gtp_at:pl, pr = (gtp_at[st].get(k, {}).get('6') for k in ('left', 'right'))
        else:pl, pr = (gtp.get(st, {}).get(k) for k in ('left', 'right'))
        off = r['lateral_offset_m']
        if pl is not None and pr is not None and max(abs(pl), abs(pr)) > GT_PAINT_M:off -= .5*(pl+pr);painted += 1
        off = abs(off);og = abs(r['lateral_offset_m'])
        if og >= .30:shown_far_gt[1] += 1;shown_far_gt[0] += sh
        if og < .15:shown_centre_gt[1] += 1;shown_centre_gt[0] += sh
        if off >= .30:shown_far[1] += 1;shown_far[0] += sh
        if off < .15:shown_centre[1] += 1;shown_centre[0] += sh
        guidance_frames.append((r, p))
    res['used_frames'] = used;res['A1_switched_frames'] = switched;res['A1_on_line_frames'] = on_line;res['A4_gt_excluded'] = a4_gt
    res['A1_gt_excluded'] = {k: dict(frames=len(v), first=round(v[0], 2) if v else None, last=round(v[-1], 2) if v else None) for k, v in excluded.items()}
    res['A1_lateral_error_m'] = {k: stats(v) for k, v in A1.items()}
    res['A1_all'] = stats([x for v in A1.values() for x in v]);res['A1_vs_paint'] = stats(A1p)
    res['A1_nearest_line_10m'] = stats(A1m);res['A1_identity'] = round(A1i[0]/A1i[1], 4) if A1i[1] else None
    res['A2_observed'] = {s: round(v[0]/v[1], 4) if v[1] else None for s, v in A2.items()};res['A2_frames'] = {s: v[1] for s, v in A2.items()}
    res['A3_style'] = {s: dict(accuracy=round(v[0]/v[2], 4) if v[2] else None, known=round(v[1]/v[2], 4) if v[2] else None, n=v[2]) for s, v in A3s.items()}
    res['A3_colour'] = {s: dict(accuracy=round(v[0]/v[2], 4) if v[2] else None, n=v[2]) for s, v in A3c.items()}
    res['A4_offset_m'] = stats(A4o);res['A4_heading_deg'] = stats(A4h);res['A4_coverage'] = round((len(A4o)+len(A4e))/max(used, 1), 4)
    res['A4_offset_fit_m'] = stats(A4oc);res['A4_heading_fit_deg'] = stats(A4hc)
    res['A4_offset_estimated_m'] = stats(A4e);res['A4_at_origin_m'] = stats(A4z)
    res['A5'] = {s: dict(precision=round(q['tp']/max(q['tp']+q['fp'], 1), 4), recall=round(q['tp']/max(q['tp']+q['fn'], 1), 4),
                         opposite_accuracy=round(q['opp'][0]/q['opp'][1], 4) if q['opp'][1] else None,
                         crossable_accuracy=round(q['cross'][0]/q['cross'][1], 4) if q['cross'][1] else None,
                         recall_visible=round(q['vis'][0]/q['vis'][1], 4) if q['vis'][1] else None, visible_n=q['vis'][1], **{k: q[k] for k in ('tp', 'fp', 'fn', 'tn')})
                 for s, q in A5.items()}
    # B: guidance
    changes = lane_changes(gt);pg = [pred[k] for k in sorted(pred)]
    moves = [(b['stamp'], b['guidance']['target']) for a_, b in zip(pg, pg[1:]) if b['guidance']['changes'] != a_['guidance']['changes']]
    on_time = 0;used_moves = set()
    for t, _, _ in changes:
        # The target moved within 3 s before, not after, the crossing; each move answers one crossing only.
        hit = next((i for i, (m, _) in enumerate(moves) if i not in used_moves and t-3. <= m <= t+.05), None)
        if hit is not None:on_time += 1;used_moves.add(hit)
    # B1: a move with no GT lane change from 3 s before it to 1 s after it is a wrong move.
    spurious = sum(1 for m, _ in moves if not any(m-1. <= t <= m+3. for t, _, _ in changes))
    res['B_lane_changes'] = dict(ground_truth=len(changes), target_moves=len(moves), on_time=on_time, spurious_moves=spurious, gt_times=[round(c[0], 2) for c in changes],
                                 move_times=[round(m, 2) for m, _ in moves])
    res['B5_return_shown'] = dict(far=round(shown_far[0]/shown_far[1], 4) if shown_far[1] else None, far_n=shown_far[1],
                                  centre=round(shown_centre[0]/shown_centre[1], 4) if shown_centre[1] else None, centre_n=shown_centre[1])
    res['B5_return_shown']['painted_frames'] = painted
    res['B5_return_shown_gt'] = dict(far=round(shown_far_gt[0]/shown_far_gt[1], 4) if shown_far_gt[1] else None,
                                     centre=round(shown_centre_gt[0]/shown_centre_gt[1], 4) if shown_centre_gt[1] else None)
    # B3: frames over the yellow line (the car origin in an oncoming lane) and targets in oncoming lanes
    over = [0, 0];opposite_target = 0
    for r, p in guidance_frames:
        ego = next(l for l in r['lanes'] if l['relation'] == 'ego')
        if ego['direction'] == 'opposite':over[1] += 1;over[0] += p['guidance']['state'] in ('forbidden', 'return') and p['guidance']['shown']
        tgt = next((c for c in p['centres'] if c['name'] == p['guidance']['target']), None)
        if tgt is not None and p['guidance']['shown']:
            y = tgt['offset_m']+(BASE_FROM_ORIGIN-tgt['ref_x_m'])*math.tan(tgt['heading_rad'])
            best = min(r['lanes'], key=lambda l: abs((y_at(l['centerline_vehicle'], 0.) or 1e9)-y))
            if best['direction'] == 'opposite' and abs((y_at(best['centerline_vehicle'], 0.) or 1e9)-y) < 1.75:opposite_target += 1
    res['B3'] = dict(over_line_frames=over[1], return_shown=round(over[0]/over[1], 4) if over[1] else None, opposite_target_frames=opposite_target)
    # C: lights, signs, stop lines, advice
    res['C'] = signs_metrics(a, gt, pred)
    out = a.out/f'metrics{a.tag}{a.name}.json';out.write_text(json.dumps(res, indent=1));summary(res)


def signs_metrics(a, gt, pred):
    sys.path.insert(0, str(ROOT/'src/hero_coach'))
    from hero_coach.traffic_signs import SignTracker, SignSettings
    from hero_coach.traffic_advice import advise, AdviceSettings
    c = {};f = a.out/('yolo%s.jsonl.gz' % a.tag)
    stop = dict(err=[], seen=[0, 0], false=[0, 0])
    for r in gt:
        p = pred.get(round(r['stamp'], 4))
        if p is None:continue
        # The GT stop line: the stop points (per lane) of a light that affects the car, within 2.5 m to the side (a light
        # can list another road's stop points far off).  NOTE: this is CARLA's stop waypoint, which in Town05 lies about
        # 7 m before the painted line the detector sees (s1 6.0 s, REPORT) -- see C4 in the report.
        tl = [t for t in r['traffic_lights'] if t['affects_ego']]
        want = min((q['vehicle'][0] for t in tl for q in t.get('stop_line', []) if abs(q['vehicle'][1]) <= 2.5 and q['vehicle'][0] > -2.), default=None)
        want = None if want is None else want+BASE_FROM_ORIGIN  # stop line ahead, base_link
        got = p['stop_line_m']
        if want is not None and 0 < want <= 20.:
            stop['seen'][1] += 1;stop['seen'][0] += got is not None
            if got is not None:stop['err'].append(abs(got-want))
        elif want is None or want > 40.:
            stop['false'][1] += 1;stop['false'][0] += got is not None and got <= 20.
    c['C4_stop_waypoint'] = dict(error_m=stats(stop['err']), detected=round(stop['seen'][0]/stop['seen'][1], 4) if stop['seen'][1] else None, n=stop['seen'][1],
                                 false=round(stop['false'][0]/stop['false'][1], 4) if stop['false'][1] else None, false_n=stop['false'][1],
                                 note='against CARLA stop waypoints (stop_line_ahead_m), about 7.6 m before the painted line: not the scored C4')
    f4 = a.out/'paint_stops.jsonl.gz'
    if f4.exists():
        # C4 against the painted line (semantic camera RoadLine, paint_stops): what the detector is meant to find.
        rows4 = [json.loads(l) for l in gzip.open(f4, 'rt')]
        ps = {round(r['stamp'], 4): r['stop_m'] for r in rows4};e = [];seen = [0, 0];false = [0, 0];offset = []
        # C6 (by eye in the plan; counted here for reference): a crossing's border 0-20 m ahead and a crosswalk marking.
        cw = {round(r['stamp'], 4): r.get('crosswalk_m') for r in rows4};cw_seen = [0, 0];cw_err = []
        for r in gt:
            k = round(r['stamp'], 4);p = pred.get(k);want = cw.get(k)
            if p is None or r['is_junction'] or want is None or not 0 < want <= 20.:continue
            got = [min(q[0] for q in m['points']) for m in p['markings'] if m['kind'] == 'crosswalk']
            got = [x for x in got if x > 0];cw_seen[1] += 1;cw_seen[0] += bool(got)
            if got:cw_err.append(abs(min(got, key=lambda x: abs(x-want))-want))
        # C4 with the crosswalk fallback (lane_guidance stop_line.crosswalk_fallback): where to stop is the stop line, else
        # the near border of a crossing; predicted the same way from the markings.
        pe = [];pseen = [0, 0]
        for r in gt:
            k = round(r['stamp'], 4);p = pred.get(k)
            if p is None or r['is_junction'] or k not in ps:continue
            want = ps[k] if ps[k] is not None else cw.get(k)
            if want is None or not 0 < want <= 20.:continue
            got = p['stop_line_m']
            if got is None:
                xs = [min(q[0] for q in m['points']) for m in p['markings'] if m['kind'] == 'crosswalk']
                xs = [x for x in xs if x > 0];got = min(xs) if xs else None
            pseen[1] += 1;pseen[0] += got is not None
            if got is not None:pe.append(abs(got-want))
        c['C4_stop_position'] = dict(detected=round(pseen[0]/pseen[1], 4) if pseen[1] else None, n=pseen[1], error_m=stats(pe),
                                     note='stop line, else the crossing border (truth) / crosswalk marking (detector)')
        c['C6_crosswalk'] = dict(detected=round(cw_seen[0]/cw_seen[1], 4) if cw_seen[1] else None, n=cw_seen[1], near_edge_error_m=stats(cw_err),
                                 note='reference only: the border nearest the car against the crosswalk marking nearest to it')
        for r in gt:
            k = round(r['stamp'], 4);p = pred.get(k)
            if p is None or k not in ps or r['is_junction']:continue
            want, got = ps[k], p['stop_line_m']
            if want is not None and 0 < want <= 20.:
                seen[1] += 1;seen[0] += got is not None
                if got is not None:e.append(abs(got-want))
            elif want is None or want > 40.:false[1] += 1;false[0] += got is not None and got <= 20.
            tl = [t for t in r['traffic_lights'] if t['affects_ego']]
            w = min((q['vehicle'][0] for t in tl for q in t.get('stop_line', []) if abs(q['vehicle'][1]) <= 2.5 and q['vehicle'][0] > -2.), default=None)
            if want is not None and w is not None and 0 < want <= 30.:offset.append(want-(w+BASE_FROM_ORIGIN))
        c['C4_stop_line'] = dict(error_m=stats(e), detected=round(seen[0]/seen[1], 4) if seen[1] else None, n=seen[1],
                                 false=round(false[0]/false[1], 4) if false[1] else None, false_n=false[1],
                                 paint_minus_waypoint_m=stats(offset), truth='semantic camera RoadLine bars across the own lane')
    if not f.exists() or f.stat().st_size == 0:return c  # (an empty file: a stopped run)
    ss = SignSettings()
    for kv in getattr(a, 'sign', None) or []:
        name, v = kv.split('=', 1);setattr(ss, name, type(getattr(ss, name))(eval(v)))
    tracker = SignTracker(ss);yl = {round(r['stamp'], 4): r for r in (json.loads(l) for l in gzip.open(f, 'rt'))}
    light_n = [0, 0, 0];no_light = [0, 0];no_light_raw = [0, 0];last_own = -math.inf;sign_wrong = 0;signs_seen = {};advice = dict(over_line=0, hold=[0, 0], stop_frames=0);s = AdviceSettings()
    state_changes = 0;prev = None;other_signs = set()
    for r in gt:
        y = yl.get(round(r['stamp'], 4))
        if y is not None:tracker.update(y['stamp'], y['detections'], y['image_width'])
        light, sign = tracker.state(r['stamp'])
        if light != prev:state_changes += 1;prev = light
        tl = [t for t in r['traffic_lights'] if t['affects_ego'] and t.get('stop_line_ahead_m') is not None]
        near = [t for t in tl if t['stop_line_ahead_m'] <= 40.]
        if near:
            want = {'Red': 'red', 'Yellow': 'yellow', 'Green': 'green'}.get(near[0]['state'])
            light_n[2] += 1;light_n[0] += light is not None;light_n[1] += light is not None and light != want
        elif not tl:
            no_light_raw[1] += 1;no_light_raw[0] += light is not None
            # The GT drops the own light at its stop waypoint (about 7.6 m before the paint) and lists none inside the
            # junction, where the junction's lights are still in view: C2 counts frames outside junctions at least
            # C2_AFTER_S after the last own light.
            if not r['is_junction'] and r['stamp']-last_own >= C2_AFTER_S:no_light[1] += 1;no_light[0] += light is not None
        if tl:last_own = r['stamp']
        # Speed signs ahead within 10 m to the side (a sign for another road can be listed too); the model knows 30 / 60 / 90
        # only, so a 40 sign is counted apart (a model limit, not scored).
        allsp = [x for x in r['signs'] if 'speed_limit' in x.get('type', '') and abs(x['vehicle'][1]) <= 10.]
        speeds = [x for x in allsp if x['type'].split('.')[-1] in ('30', '60', '90')]
        for x in allsp:
            if x not in speeds and 0 < x['vehicle'][0] <= 40.:other_signs.add((x['id'], x['type']))
        for x in speeds:
            v = float(x['type'].split('.')[-1]);ahead = x['vehicle'][0]
            key = x['id'];rec = signs_seen.setdefault(key, dict(value=v, before=False, seen_close=False, reached=False))
            if 0 < ahead <= 40.:rec['seen_close'] = True
            if ahead <= 5.:rec['reached'] = True  # passed (or nearly): a sign still ahead when the recording ends is not scored
            if sign is not None and ahead > 0:rec['before'] = rec['before'] or sign == v
        if sign is not None and speeds and all(float(x['type'].split('.')[-1]) != sign for x in speeds if 0 < x['vehicle'][0] <= 40.) and any(0 < x['vehicle'][0] <= 40. for x in speeds):sign_wrong += 1
        p = pred.get(round(r['stamp'], 4))
        if p is not None:
            out = advise(light, p['stop_line_m'], r['speed_mps'], s)
            advice['over_line'] += out['advice'] == 'over_line'
            if light == 'red' and abs(r['speed_mps']) < s.stopped_mps:advice['hold'][1] += 1;advice['hold'][0] += out['advice'] == 'hold'
    c['C1_light'] = dict(frames=light_n[2], confirmed=round(light_n[0]/light_n[2], 4) if light_n[2] else None,
                         wrong_colour=round(light_n[1]/light_n[2], 4) if light_n[2] else None, state_changes=state_changes)
    c['C2_no_light_shown'] = dict(frames=no_light[1], shown=round(no_light[0]/no_light[1], 4) if no_light[1] else None,
                                  raw_frames=no_light_raw[1], raw_shown=round(no_light_raw[0]/no_light_raw[1], 4) if no_light_raw[1] else None)
    close = [v for v in signs_seen.values() if v['seen_close'] and v['reached']]
    c['C3_speed_signs'] = dict(signs=len(close), confirmed_before=round(sum(v['before'] for v in close)/len(close), 4) if close else None, wrong_frames=sign_wrong,
                               not_in_model=sorted({t for _, t in other_signs}), not_in_model_n=len(other_signs))
    c['C5_advice'] = dict(over_line_frames=advice['over_line'], hold=round(advice['hold'][0]/advice['hold'][1], 4) if advice['hold'][1] else None, hold_n=advice['hold'][1])
    return c


def summary(res):
    a1 = res['A1_all'];print(f"{res['scene']}: frames {res['used_frames']} | A1 med {a1['median']} p95 {a1['p95']} | A2 {res['A2_observed']} | "
                             f"A3 style {[(k, v['accuracy']) for k, v in res['A3_style'].items()]} colour {[(k, v['accuracy']) for k, v in res['A3_colour'].items()]} | "
                             f"A4 off {res['A4_offset_m']['median']}/{res['A4_offset_m']['p95']} head p95 {res['A4_heading_deg']['p95']} | "
                             f"B changes gt {res['B_lane_changes']['ground_truth']} moves {res['B_lane_changes']['target_moves']} on time {res['B_lane_changes']['on_time']} | "
                             f"B5 {res['B5_return_shown']} | B3 {res['B3']} | C {json.dumps(res['C'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=('predict', 'reguide', 'yolo', 'metrics', 'paint_stops'));ap.add_argument('scene');ap.add_argument('out', type=Path)
    ap.add_argument('--set', nargs='*', default=[], help='predict: Settings overrides name=value')
    ap.add_argument('--source', type=Path, help='reguide: the round directory of the scene whose centres are used')
    ap.add_argument('--name', default='', help='metrics: suffix of the output file (metrics<tag><name>.json)')
    ap.add_argument('--sign', nargs='*', default=[], help='metrics: SignSettings overrides name=value')
    ap.add_argument('--display', nargs='*', default=[], help='predict: lane_display_filter parameter overrides name=value')
    ap.add_argument('--guide', nargs='*', default=[], help='predict: GuidanceSettings overrides name=value')
    ap.add_argument('--model', default=str(Path.home()/'Desktop/town05_yolo26m_detector/best.pt'))
    ap.add_argument('--full-frame', type=int, default=1);ap.add_argument('--conf', type=float, default=.25);ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--tag', default='', help='yolo / metrics: file suffix (yolo<tag>.jsonl.gz), e.g. _tile')
    a = ap.parse_args()
    {'predict': predict, 'reguide': reguide, 'yolo': yolo, 'metrics': metrics, 'paint_stops': paint_stops}[a.cmd](a)


if __name__ == '__main__':
    main()
