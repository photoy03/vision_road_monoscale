"""KITTI Raw synced/rectified input. Matrices T_a_b map coordinates b -> a."""
from pathlib import Path
import math
import json
import cv2
import numpy as np
import yaml
from .rig import validate_mount


def read_fields(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        if ':' not in line:
            continue
        key, value = line.split(':', 1)
        try:
            result[key] = np.array([float(x) for x in value.split()])
        except ValueError:
            continue  # calibration timestamp is not a numeric field
    return result


def rigid(path):
    d = read_fields(path); t = np.eye(4)
    t[:3, :3] = d['R'].reshape(3, 3); t[:3, 3] = d['T'].reshape(3)
    if not np.isfinite(t).all() or not np.allclose(t[:3,:3].T @ t[:3,:3], np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(t[:3,:3]), 1, atol=1e-5):
        raise ValueError('Invalid rigid calibration: '+str(path))
    return t


def timestamps(path):
    # Keep integer nanoseconds. No local timezone or float Unix-time conversion.
    lines = Path(path).read_text().splitlines()
    stamps = np.array([np.datetime64(s.strip().replace(' ', 'T'), 'ns').astype(np.int64) for s in lines], dtype=np.int64)
    if not len(stamps) or np.any(np.diff(stamps) <= 0):
        raise ValueError('Timestamps must be nonempty and strictly increasing: '+str(path))
    return stamps


def rpy_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll); cp, sp = math.cos(pitch), math.sin(pitch); cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]]) @ np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]]) @ np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])


class KittiSequence:
    def __init__(self, sequence, imu_height_m=0.93):
        self.path = Path(sequence).expanduser().resolve()
        if not self.path.is_dir() or not self.path.name.endswith('_sync'):
            raise ValueError('Select an extracted KITTI Raw *_sync directory (synced+rectified)')
        if not np.isfinite(imu_height_m) or imu_height_m <= 0:
            raise ValueError('IMU ground height must be positive')
        self.imu_height = float(imu_height_m)
        self.images = sorted((self.path/'image_02/data').glob('*.png'))
        if not self.images:
            raise ValueError('No image_02/data/*.png found: '+str(self.path))
        self.image_ns = timestamps(self.path/'image_02/timestamps.txt')
        self.oxts_ns = timestamps(self.path/'oxts/timestamps.txt')
        if len(self.images) != len(self.image_ns):
            raise ValueError('Image file/timestamp counts differ')
        oxts_files = sorted((self.path/'oxts/data').glob('*.txt'))
        if len(oxts_files) != len(self.oxts_ns):
            raise ValueError('OXTS file/timestamp counts differ')
        if [p.stem for p in self.images] != [f'{i:010d}' for i in range(len(self.images))]:
            raise ValueError('Image frame numbering has holes or does not start at zero')
        if [p.stem for p in oxts_files] != [f'{i:010d}' for i in range(len(oxts_files))]:
            raise ValueError('OXTS frame numbering has holes or does not start at zero')
        self.oxts = np.array([np.loadtxt(p, dtype=float).reshape(-1) for p in oxts_files])
        if self.oxts.ndim != 2 or self.oxts.shape[1] < 30 or not np.isfinite(self.oxts).all():
            raise ValueError('Expected 30 finite KITTI OXTS fields per packet')
        self.epoch_ns = int(min(self.image_ns[0], self.oxts_ns[0]))
        self.image_times = (self.image_ns-self.epoch_ns)*1e-9+1.0
        self.oxts_times = (self.oxts_ns-self.epoch_ns)*1e-9+1.0
        first = cv2.imread(str(self.images[0]))
        if first is None:
            raise ValueError('Cannot decode first KITTI image')
        h, w = first.shape[:2]; self.size = (w, h)
        calib = self.path.parent
        c = read_fields(calib/'calib_cam_to_cam.txt')
        p = c['P_rect_02'].reshape(3, 4); self.k = p[:3,:3].copy()
        if not np.isfinite(p).all() or self.k[0,0] <= 0 or self.k[1,1] <= 0:
            raise ValueError('Invalid P_rect_02')
        if 'S_rect_02' in c and not np.array_equal(np.rint(c['S_rect_02']).astype(int), self.size):
            raise ValueError('Image size differs from S_rect_02; use original rectified files')
        rect = np.eye(4); rect[:3,:3] = c['R_rect_00'].reshape(3,3)
        # P_rect_02 includes the rectified cam0 -> cam2 baseline. Do not use R_02/T_02 a second time.
        cam2_cam0 = np.eye(4); cam2_cam0[:3,3] = np.linalg.solve(self.k, p[:,3])
        self.t_cam_imu = cam2_cam0 @ rect @ rigid(calib/'calib_velo_to_cam.txt') @ rigid(calib/'calib_imu_to_velo.txt')
        self.t_imu_base = np.eye(4); self.t_imu_base[2,3] = -self.imu_height
        self.t_base_cam = np.linalg.inv(self.t_cam_imu @ self.t_imu_base)
        self.rig = {'distortion_model':'plumb_bob', 'imu_axes_aligned_with_base':True,
                    'rotation_base_from_camera':self.t_base_cam[:3,:3].reshape(-1).tolist(),
                    'translation_base_from_camera':self.t_base_cam[:3,3].tolist(),
                    'imu_position_m':[0.0,0.0,self.imu_height],
                    'source':'KITTI Raw calibration; ground origin below IMU',
                    'imu_ground_height_m':self.imu_height}
        validate_mount(self.rig)
        self.poses, self.imu_rotations = self._poses()

    def _poses(self):
        data = self.oxts; earth = 6378137.0
        scale = math.cos(math.radians(data[0,0]))
        positions = np.column_stack([scale*earth*np.radians(data[:,1]), scale*earth*np.log(np.tan(np.pi/4+np.radians(data[:,0])/2)), data[:,2]])
        positions -= positions[0]
        rotations = [rpy_matrix(*row[3:6]) for row in data]
        transforms = []
        for pos, rotation in zip(positions, rotations):
            t = np.eye(4); t[:3,:3] = rotation; t[:3,3] = pos
            transforms.append(t @ self.t_imu_base)
        # Preserve earth vertical; remove only initial yaw and translation.
        origin = np.eye(4); origin[:3,:3] = rpy_matrix(0,0,data[0,5]); origin[:3,3] = transforms[0][:3,3]
        inverse = np.linalg.inv(origin)
        poses = [inverse @ t for t in transforms]
        return poses, [inverse[:3,:3] @ r for r in rotations]

    def image(self, i):
        image = cv2.imread(str(self.images[i]))
        if image is None or (image.shape[1], image.shape[0]) != self.size:
            raise ValueError('Missing/corrupt/changed-size frame: '+str(self.images[i]))
        return image

    def export(self, output):
        out = Path(output).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
        np.savez(out/'camera.npz', K=self.k, D=np.zeros(5), image_size=np.array(self.size))
        (out/'rig.yaml').write_text(yaml.safe_dump(self.rig, sort_keys=False))
        info = {'sequence':str(self.path), 'frames':len(self.images), 'image_size':self.size,
                'imu_height_m':self.imu_height,'camera_height_m':float(self.t_base_cam[2,3]),
                'duration_sec':float(self.image_times[-1]-self.image_times[0]),
                'T_base_camera_rect02':self.t_base_cam.tolist(), 'T_cam02_imu':self.t_cam_imu.tolist(),
                'projection':'P_rect_02; rectified pinhole; no repeated distortion correction',
                'pose_source':'OXTS reference pose (not MonoScale estimate)',
                'original_epoch_ns':self.epoch_ns}
        (out/'dataset.json').write_text(json.dumps(info, indent=2))
        return out


def replay_events(image_times, oxts_times):
    """Delivery order with original acquisition stamps intact.

    Hold each image until the recorded IMU sample after it has been published.
    This supplies the two-sided yaw history required by the photometric tracker.
    A small delivery grace allows the subscriber to ingest the separate IMU topic.
    """
    image_times=np.asarray(image_times); oxts_times=np.asarray(oxts_times)
    events=[]
    for i,t in enumerate(image_times):
        upper=int(np.searchsorted(oxts_times,t,side='right'))
        covered=float(oxts_times[upper]) if upper<len(oxts_times) else float(t)
        events.append((max(float(t),covered)+.02,1,i,float(t)))
    horizon=max(e[0] for e in events)
    events.extend((float(t),0,i,float(t)) for i,t in enumerate(oxts_times) if t<=horizon)
    return sorted(events)
