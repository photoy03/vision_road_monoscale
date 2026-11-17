"""Metric fisheye BEV, observed-only polynomial lanes and bounded world map."""
from dataclasses import dataclass
from bisect import bisect_left
import math
import cv2
import numpy as np
from .rig import read_calibration, rotation_base_from_camera, validate_mount


@dataclass
class Settings:
    lateral_min: float = -4.0  # BEV right-positive; ROS base y is left-positive
    lateral_max: float = 4.0
    forward_min: float = 0.5
    forward_max: float = 15.0
    mpp: float = 0.04
    windows: int = 20
    margin_m: float = 0.45
    min_pixels: int = 45
    min_span_m: float = 1.0
    fit_error_m: float = 0.15
    white_l_min: int = 170
    white_s_max: int = 100
    yellow_s_min: int = 80
    lane_width_min: float = 2.2
    lane_width_max: float = 5.0
    segment_m: float = 0.4
    shadow_enabled: bool = True
    ridge_width_m: float = 0.50
    ridge_contrast_min: float = 5.0
    local_contrast_min: float = 0.10
    temporal_enabled: bool = True
    temporal_max_frames: int = 3
    temporal_max_age_sec: float = 0.35
    temporal_max_distance_m: float = 3.0
    temporal_match_m: float = 0.45
    temporal_use_flow: bool = True
    style_min_span_m: float = 8.0
    style_min_dash_m: float = 0.60
    style_min_gap_m: float = 0.70
    style_confirm_frames: int = 2
    style_temporal_enabled: bool = False
    style_history_sec: float = 0.8
    style_history_frames: int = 8
    style_hold_sec: float = 0.0
    style_accumulate_m: float = 0.0  # >0: also judge dashed from paint accumulated along the road over the last 20 m travelled
    candidate_mask_enabled: bool = True
    candidate_max_width_m: float = 0.45
    carla_stage: int = 3
    carla_ridge_min: float = 3.0
    carla_relative_min: float = 0.025
    carla_grid_enabled: bool = True
    carla_white_light_min: int = 70
    carla_native_enabled: bool = True
    carla_path_filter_enabled: bool = True
    carla_chain_filter_enabled: bool = True
    carla_chain_confirm_frames: int = 3
    carla_chain_band_m: float = 0.25
    carla_spine_cubic_m: float = 0.0  # >0: a long, densely painted chain gets a cubic (straight-then-curving), held to 0 unless its quadratic misses the paint by about this (p90)
    carla_spine_cubic_span_m: float = 12.0  # ... chain at least this long
    carla_spine_cubic_cover: float = 0.6  # ... with paint along at least this share of it (dashes keep the quadratic)
    carla_spine_cubic_ridge: float = 0.05  # pull of the cubic term towards 0 (x in units of 10 m) at that miss; x(threshold/miss)^4 otherwise
    carla_spine_cubic_double: bool = True  # False: a double line (strand pair measured) keeps the quadratic
    carla_spine_near_m: float = 0.0  # >0: paint beyond this counts less in the fit, (near/x)^2, so far paint cannot bend the near boundary
    carla_tail_gap_m: float = 0.0  # >0: after a paint gap this long that holds a transverse bar (stop line), a chain is cut where it continues off its own near shape
    carla_tail_offset_m: float = 0.5  # ... by more than this (a line ending at a junction picks up the crossing road's edge)
    carla_tail_bar_m: float = 1.0  # transverse paint at least this wide counts as a bar
    carla_native_fit: bool = False  # plan 26: spine and pair-vote least squares in _native (QR; equal up to rounding, not bit for bit)
    carla_parallel_binary: bool = False  # plan 26: the paint mask on its own thread beside the track moves (same results)
    carla_native_chain: bool = False  # plan 26: chain DP, pair vote, style sampling in _native (see REPORT of latency26)
    carla_native_fast: bool = False  # plan 25: the per-pixel binary steps, the edge test and the per-lane loops in _native (same results)
    carla_parallel_ops: bool = False  # True: independent OpenCV steps (median filter, history warps) on two threads; same results
    carla_turn_curvature: float = 0.0  # >0: while the vehicle turns this sharply (1/m, over its last metre), a candidate starting within 15 m is not taken
    carla_turn_angle_deg: float = 25.0  # ... when its first 2 m head against the turn and cross the vehicle's own arc at this angle (a crossing road's line)
    carla_turn_wrong_side_m: float = 0.3  # ... or when it starts this far on the other side of that arc
    carla_tracking_enabled: bool = True
    carla_tracking_keep_bonus: float = 2.0
    carla_tracking_match_m: float = 0.35
    carla_tracking_switch_margin: float = 1.5
    carla_tracking_switch_frames: int = 3
    carla_tracking_memory_sec: float = 0.25
    carla_spine_ema_alpha: float = 0.5  # weight of the current boundary when it matches the previous one
    carla_link_curvature_max: float = 0.0  # 1/m; optional curve allowance for chain links (1/40, 1/100 regressed maploop)
    # Curve stabilisation (each off by default; see docs "커브에서 차선이 흔들리는 이유").
    carla_shared_shape: bool = False  # a short boundary takes the long one's shape, offset along its normal
    carla_shared_long_m: float = 12.0  # the long side spans at least this
    carla_shared_short_m: float = 8.0  # the short side has less paint than this
    carla_shared_extend_m: float = 6.0  # the short side extends past its own paint by up to this along the long side's shape
    carla_strand_pair: bool = False  # a double line is one pair of rails: each fragment votes for the pair's centre
    carla_strand_gap_m: float = .27  # double line spacing before one is measured; then a slow average of measured gaps
    carla_strand_memory_m: float = 0.0  # >0: centres of seen pairs kept in odometry coordinates this far of travel label single strands
    # Optional learned lane mask gate (learned_mask.py); without a mask nothing changes.
    carla_learned_gate: str = 'off'  # off | remove (drop paint outside the mask) | penalize (lower its chain score)
    #                                   | replace (the mask's centre lines are the paint; style still from paint)
    carla_learned_gate_m: float = 0.35  # lateral widening of the learned mask
    carla_learned_gate_along_m: float = 1.0  # forward/backward widening, bridges dash gaps
    carla_learned_gate_far_m: float = 20.0  # gate only this close; the mask thins out and misses far paint
    carla_learned_penalty: float = 0.8  # penalize: fraction of an outside fragment's length removed from its score
    carla_width_reject_m: float = 0.0  # >0: a candidate this far off the lane width from the other side's tracked boundary is not taken
    carla_yellow_relative_min: float = 0.0  # >0: paint whose yellowness (r+g)/2-b is this much above the local road level (and itself) is yellow
    carla_yellow_relative_window_m: float = 1.5  # square window of the local road level (median)
    carla_outer_enabled: bool = False  # also pick each side's neighbouring lane outer line from the paint left over (own lines unchanged)
    carla_outer_reach_m: float = 6.0  # near fragments are kept this far out (lateral at 3 m) for it; the own lines still see only 0.65-4.2 m
    carla_outer_gap_min_m: float = 2.3  # distance from the own line, along its normal
    carla_outer_gap_max_m: float = 4.7
    carla_outer_width_miss_m: float = 0.9  # ... and within this of the own lane width
    carla_outer_parallel_m: float = 0.3  # the distance may vary at most this much along the outer paint (p10-p90)
    carla_outer_cluster_m: float = 0.2  # fragments within this of one distance form one line
    carla_outer_min_paint_m: float = 2.0  # paint along the line, at least
    carla_outer_extend_m: float = 6.0  # past its paint the line follows the own line's shape this far
    carla_outer_keep_m: float = 0.3  # a candidate within this of last frame's distance is preferred
    carla_centre_enabled: bool = False  # lane centrelines: own lane (left-right) and, with carla_outer_enabled, the neighbouring lanes
    carla_centre_ref_m: float = 5.0  # lateral offset, heading and curvature are given at this distance ahead (inside the view)
    carla_centre_curvature_m: float = 15.0  # ... from a quadratic through this much centreline past it
    carla_centre_width_alpha: float = 0.2  # running lane width (for a lane with one boundary): weight of a measured width


class Projector:
    def __init__(self, calibration, rig, cfg):
        self.model = rig.get('distortion_model', 'equidistant')
        self.k, self.d, self.size = read_calibration(calibration, self.model)
        self.cfg = cfg
        if 'rotation_base_from_camera' in rig:
            self.rotation, self.translation = validate_mount(rig)
        else:
            self.rotation = rotation_base_from_camera(rig['pitch_deg'], rig['roll_deg'], rig['yaw_deg'])
            self.translation = np.array([rig['camera_forward_m'], rig['camera_left_m'], rig['camera_height_m']])
        self.source_size = None

    def initialize(self, width, height):
        if abs(width / height - self.size[0] / self.size[1]) > 1e-6:
            raise ValueError('Image aspect ratio differs from calibration')
        c = self.cfg
        self.width = round((c.lateral_max-c.lateral_min)/c.mpp)
        self.height = round((c.forward_max-c.forward_min)/c.mpp)
        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
        base = np.stack([c.forward_max-v*c.mpp, -(c.lateral_min+u*c.mpp), np.zeros_like(u)], axis=-1)
        cam = (base-self.translation) @ self.rotation
        k = self.k.copy(); k[0, :] *= width/self.size[0]; k[1, :] *= height/self.size[1]
        if self.model == 'equidistant':
            pixels, _ = cv2.fisheye.projectPoints(cam.reshape(-1, 1, 3).astype(float), np.zeros(3), np.zeros(3), k, self.d)
        else:
            pixels, _ = cv2.projectPoints(cam.reshape(-1, 1, 3).astype(float), np.zeros(3), np.zeros(3), k, self.d)
        pixels = pixels.reshape(self.height, self.width, 2)
        self.mx, self.my = pixels[:, :, 0].astype('float32'), pixels[:, :, 1].astype('float32')
        self.valid = ((cam[:, :, 2] > 0) & (self.mx >= 0) & (self.mx < width-1) & (self.my >= 0) & (self.my < height-1)).astype('uint8')*255
        self.source_size = (width, height)

    def warp(self, frame, top=0):
        """top: the frame holds only rows [top, top+height) of the full image (a row crop, see rows());
        the BEV is then the same as from the full image wherever rows() covers."""
        if top or (self.source_size is not None and frame.shape[1] == self.source_size[0] and frame.shape[0] < self.source_size[1]):
            mx, my = self.crop_maps(top, frame.shape[0])
            bev = cv2.remap(frame, mx, my, cv2.INTER_LINEAR)
            return cv2.bitwise_and(bev, bev, mask=self.valid)
        size = (frame.shape[1], frame.shape[0])
        if size != self.source_size:
            self.initialize(*size)
        bev = cv2.remap(frame, self.mx, self.my, cv2.INTER_LINEAR)
        return cv2.bitwise_and(bev, bev, mask=self.valid)

    def rows(self, roi_bottom=1.0):
        """Source rows [lo, hi) the BEV reads (bilinear: floor and floor+1) for its valid pixels above
        roi_bottom * height: a row crop covering these gives the same BEV as the full image."""
        used = (self.valid > 0) & (self.my < self.source_size[1]*roi_bottom)
        my = self.my[used]
        return int(np.floor(my.min())), int(np.floor(my.max()))+2

    def crop_maps(self, top, height):
        """remap maps for a frame of rows [top, top+height) (shifted by an integer: the same sampling)."""
        if self.source_size is None:
            raise ValueError('Row crop before the projector knows the full image size')
        if top < 0 or top+height > self.source_size[1]:
            raise ValueError('Row crop outside the full image')
        key = (top, height)
        if getattr(self, '_crop_key', None) != key or getattr(self, '_crop_source', None) != self.source_size:
            self._crop = (self.mx, self.my-np.float32(top))
            self._crop_key, self._crop_source = key, self.source_size
        return self._crop


    def image_points(self, points):
        base = np.column_stack([points, np.zeros(len(points))])
        cam = (base-self.translation) @ self.rotation
        k = self.k.copy(); k[0,:] *= self.source_size[0]/self.size[0]; k[1,:] *= self.source_size[1]/self.size[1]
        if self.model == 'equidistant':
            pixels, _ = cv2.fisheye.projectPoints(cam.reshape(-1,1,3), np.zeros(3), np.zeros(3), k, self.d)
        else:
            pixels, _ = cv2.projectPoints(cam.reshape(-1,1,3), np.zeros(3), np.zeros(3), k, self.d)
        return pixels.reshape(-1,2), cam[:,2] > 1e-6


@dataclass
class Lane:
    side: str
    points: np.ndarray  # forward, left in base_link
    confidence: float
    hits: int = 1
    reprojected: bool = False
    source_time: float = 0.0
    style: str = 'unknown'
    style_confidence: float = 0.0
    style_span_m: float = 0.0
    paint_fraction: float = 0.0
    dash_count: int = 0
    style_source_time: float = 0.0
    track_key: str = ''  # Internal fragment identity; ROS family remains left/right.


class Detector:
    def __init__(self, cfg):
        self.c = cfg
        self.previous = {}
        self.hits = {}

    def binary(self, bev, valid):
        hls = cv2.cvtColor(bev, cv2.COLOR_BGR2HLS)
        h, light, saturation = cv2.split(hls)
        if not self.c.shadow_enabled:
            white = (light >= self.c.white_l_min) & (saturation <= self.c.white_s_max)
            yellow = (h >= 12) & (h <= 40) & (saturation >= self.c.yellow_s_min) & (light >= 60)
            sobel = np.abs(cv2.Sobel(light, cv2.CV_32F, 1, 0, ksize=3))
            scaled = sobel * (255.0/max(float(sobel.max()), 1.0))
            support = cv2.dilate((white | yellow).astype('uint8'), np.ones((3,3),'uint8')) > 0
            return cv2.bitwise_and(((white | yellow | ((scaled>35)&(scaled<220)&support)).astype('uint8')*255), valid)
        # Bright paint is a narrow ridge relative to adjacent asphalt, even in shadow.
        # Unlike an edge detector, requiring darker pavement on both sides suppresses
        # broad illumination boundaries and many curb edges.
        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY).astype('float32')
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(light)
        width = max(5, round(self.c.ridge_width_m/self.c.mpp)) | 1
        response = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, np.ones((1,width),'uint8'))
        background = cv2.GaussianBlur(gray, (0,0), max(2., .40/self.c.mpp))
        ratio = (gray-background)/np.maximum(background, 15.)
        step = max(2, round(.20/self.c.mpp))
        left = np.empty_like(gray); right = np.empty_like(gray)
        left[:,step:] = gray[:,:-step]; left[:,:step] = gray[:,:1]
        right[:,:-step] = gray[:,step:]; right[:,-step:] = gray[:,-1:]
        ridge = np.minimum(gray-left, gray-right)
        white = (saturation <= max(self.c.white_s_max, 150)) & (light >= 20)
        yellow = (h>=12)&(h<=40)&(saturation>=self.c.yellow_s_min)&(light>=25)
        selected = (white | yellow) & (ridge>=self.c.ridge_contrast_min) & (response>=10) & (ratio>=self.c.local_contrast_min)
        safe = cv2.erode(valid, np.ones((3,2*step+1),'uint8'))
        binary = cv2.bitwise_and(selected.astype('uint8')*255, safe)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3,1),'uint8'))

    def detect(self, mask, priors=None, candidate_score=None):
        c = self.c; height, width = mask.shape
        yy, xx = np.nonzero(mask)
        hist = np.count_nonzero(mask[height//2:], axis=0).astype(float)
        mid = int(np.clip(-c.lateral_min/c.mpp, 1, width-1))
        lanes = []
        for side, lo, hi in [('left',0,mid), ('right',mid,width)]:
            reference = self.previous.get(side)
            if priors is not None:
                reference = None
                if side in priors and len(priors[side].points)>=3:
                    pts = priors[side].points
                    reference = np.polyfit((c.forward_max-pts[:,0])/c.mpp, (-pts[:,1]-c.lateral_min)/c.mpp, 2)
            candidates = []
            if reference is not None:
                ids = np.flatnonzero(np.abs(xx-np.polyval(reference,yy)) < c.margin_m/c.mpp)
                fit = self._fit(xx, yy, ids)
                if fit is not None:
                    candidates.append(fit)
            # Several histogram peaks: one bright curb must not suppress a weaker dash.
            scores = hist.copy(); scores[:lo]=0; scores[hi:]=0
            for _ in range(4):
                center = int(np.argmax(scores))
                if scores[center]<3:
                    break
                radius = max(2, round(.50/c.mpp))
                scores[max(0,center-radius):min(width,center+radius+1)] = 0
                ids=[]; cursor=float(center)
                for n in range(c.windows):
                    bottom=height-n*height//c.windows; top=height-(n+1)*height//c.windows
                    ix=np.flatnonzero((yy>=top)&(yy<bottom)&(np.abs(xx-cursor)<c.margin_m/c.mpp))
                    ids.extend(ix)
                    if len(ix)>=6:
                        cursor=float(median(xx[ix]))
                fit=self._fit(xx,yy,np.asarray(ids,dtype=int))
                if fit is not None:
                    candidates.append(fit)
            if not candidates:
                self.previous.pop(side,None); self.hits.pop(side,None)
                continue
            def quality(fit):
                coeff, rows, confidence=fit
                if candidate_score is not None:
                    return candidate_score(side,coeff,rows,confidence)
                # Prefer observed longitudinal support; continuity is only a tie breaker.
                distance = np.mean(np.abs(np.polyval(coeff,rows)-np.polyval(reference,rows)))*c.mpp if reference is not None else 0.
                anchor_row=(c.forward_max-10.)/c.mpp
                lateral=abs(c.lateral_min+np.polyval(coeff,anchor_row)*c.mpp)
                return confidence - .14*lateral - (.20*min(distance,1.) if reference is not None else 0.)
            coeff,rows,confidence=max(candidates,key=quality)
            if not np.isfinite(quality((coeff,rows,confidence))):
                continue
            old=self.previous.get(side)
            matched=old is not None and np.max(np.abs(np.polyval(old,rows)-np.polyval(coeff,rows)))*c.mpp<.35
            self.hits[side]=min(65535,self.hits.get(side,0)+1) if matched else 1
            self.previous[side]=coeff
            cols=np.polyval(coeff,rows)
            points=np.column_stack([c.forward_max-rows*c.mpp,-(c.lateral_min+cols*c.mpp)])
            lanes.append(Lane(side,points[::-1],confidence,self.hits[side]))
        if len(lanes)==2:
            a,b=lanes; low=max(a.points[0,0],b.points[0,0]); high=min(a.points[-1,0],b.points[-1,0])
            if high>low:
                x=np.linspace(low,high,12)
                widths=np.interp(x,a.points[:,0],a.points[:,1])-np.interp(x,b.points[:,0],b.points[:,1])
                if np.any((widths<c.lane_width_min)|(widths>c.lane_width_max)):
                    # A bad partner no longer removes a supported individual lane.
                    keep=max(lanes,key=lambda lane: lane.confidence)
                    for lane in lanes:
                        if lane is not keep:
                            self.previous.pop(lane.side,None); self.hits.pop(lane.side,None)
                    return [keep]
        return lanes

    def _fit(self, xx, yy, idx):
        c = self.c
        if len(idx) < c.min_pixels or np.ptp(yy[idx])*c.mpp < c.min_span_m:
            return None
        x, y = xx[idx].astype(float), yy[idx].astype(float)
        if len(np.unique(y)) < 3:
            return None
        for _ in range(3):
            coeff = np.polyfit(y, x, 2)
            residual = np.abs(x-np.polyval(coeff, y))
            keep = residual*c.mpp <= c.fit_error_m
            if np.count_nonzero(keep) < c.min_pixels:
                return None
            x, y = x[keep], y[keep]
        if np.ptp(y)*c.mpp < c.min_span_m:
            return None
        coeff = np.polyfit(y, x, 2)
        rows = np.arange(y.min(), y.max()+0.1, max(1, c.segment_m/c.mpp))
        coverage = len(np.unique((y*c.mpp/.5).astype(int))) / max(1., np.ptp(y)*c.mpp/.5)
        confidence = float(np.clip((1-np.mean(np.abs(x-np.polyval(coeff,y)))*c.mpp/c.fit_error_m) * (.50+.50*min(coverage,1.)),0,1))
        return coeff, rows, confidence


class Timeline:
    def __init__(self, max_gap=0.25):
        self.samples = []
        self.max_gap = max_gap
        self.generation = 0

    def add(self, t, x, y, yaw):
        if not np.isfinite([t, x, y, yaw]).all():
            return False
        if self.samples and t < self.samples[-1][0]:
            self.samples.clear(); self.generation += 1
        if self.samples and t == self.samples[-1][0]:
            self.samples[-1] = (t, x, y, yaw)
        else:
            self.samples.append((t, x, y, yaw))
        self.samples = self.samples[-2000:]
        return True

    def at(self, t):
        i = bisect_left([a[0] for a in self.samples], t)
        if i < len(self.samples) and abs(self.samples[i][0]-t) < 1e-7:
            return self.samples[i][1:]
        if i == 0 or i == len(self.samples):
            return None
        a, b = self.samples[i-1:i+1]
        if b[0]-a[0] > self.max_gap:
            return None
        f = (t-a[0])/(b[0]-a[0]); dyaw = math.atan2(math.sin(b[3]-a[3]), math.cos(b[3]-a[3]))
        return (a[1]+f*(b[1]-a[1]), a[2]+f*(b[2]-a[2]), a[3]+f*dyaw)


def median(values):
    """np.median of a 1-D array (NaN in, NaN out, empty gives NaN) without its per-call overhead.

    The detector takes about a hundred medians of short arrays per frame; np.partition is
    several times cheaper than np.median there and gives the same value."""
    a=np.asarray(values,dtype=float).ravel();n=a.size
    if n==0:return np.float64(np.nan)
    k=n//2
    if n<=32:
        # A short list sorts faster in Python than np.partition runs; a NaN (or inf-inf) in it goes the general way.
        s=a.tolist()
        if sum(s)==sum(s):
            s.sort();return np.float64(s[k]) if n%2 else np.float64((s[k-1]+s[k])/2)
    if n%2:p=np.partition(a,k);m=p[k]
    else:p=np.partition(a,(k-1,k));m=(p[k-1]+p[k])/2
    return np.float64(np.nan) if np.isnan(p[-1]) else m


def to_world(points, pose):
    x, y, yaw = pose; c, s = math.cos(yaw), math.sin(yaw)
    return points @ np.array([[c, s], [-s, c]]) + [x, y]


class LaneMap:
    def __init__(self, resolution=0.10, radius=80.0, ttl=120.0, max_cells=100000):
        self.resolution, self.radius, self.ttl, self.max_cells = resolution, radius, ttl, max_cells
        self.cells = {}

    def update(self, lanes, pose, stamp):
        touched = set()
        for lane in lanes:
            if lane.reprojected:
                continue  # Predictions never create or reinforce accumulated map evidence.
            world = to_world(lane.points, pose)
            for a, b in zip(world[:-1], world[1:]):
                for pt in np.linspace(a, b, max(2, math.ceil(np.linalg.norm(b-a)/self.resolution)+1)):
                    key = tuple(np.floor(pt/self.resolution).astype(int))
                    if key in touched:
                        continue
                    touched.add(key)
                    old = self.cells.get(key)
                    hits = min(65535, old[2]+1) if old else 1
                    direction = (b-a) / max(np.linalg.norm(b-a), 1e-9)
                    self.cells[key] = (pt, stamp, hits, lane.confidence, direction)
        self.cells = {k: v for k, v in self.cells.items() if stamp-v[1] <= self.ttl and np.linalg.norm(v[0]-pose[:2]) <= self.radius}
        if len(self.cells) > self.max_cells:
            self.cells = dict(sorted(self.cells.items(), key=lambda kv: kv[1][1], reverse=True)[:self.max_cells])
