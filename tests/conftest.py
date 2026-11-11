"""Small, explicitly synthetic KITTI-format rig/sequence for integration tests."""
from pathlib import Path
import cv2
import numpy as np
import pytest


def make_kitti(root,frames=20):
    date = root/'2011_09_26'; seq = date/'2011_09_26_drive_0001_sync'
    (seq/'image_02/data').mkdir(parents=True); (seq/'oxts/data').mkdir(parents=True)
    r = np.array([[0,-1,0],[0,0,-1],[1,0,0]],dtype=float)  # optical <- IMU FLU
    # camera center in IMU = [1.2, .05, .72] -> height 1.65 above ground.
    translation = -r @ [1.2,.05,.72]
    (date/'calib_imu_to_velo.txt').write_text('R: 1 0 0 0 1 0 0 0 1\nT: 0 0 0\n')
    (date/'calib_velo_to_cam.txt').write_text('R: '+' '.join(map(str,r.ravel()))+'\nT: '+' '.join(map(str,translation))+'\n')
    k = np.array([[700.,0,620],[0,700,185],[0,0,1]])
    # Nonzero cam2 baseline: test that P_rect_02 contributes to the fixed TF.
    p = np.column_stack([k,k @ np.array([-.06,.01,0])])
    (date/'calib_cam_to_cam.txt').write_text('R_rect_00: 1 0 0 0 1 0 0 0 1\nP_rect_02: '+' '.join(map(str,p.ravel()))+'\nS_rect_02: 1242 375\n')
    t_cam_imu = np.eye(4); t_cam_imu[:3,:3] = r; t_cam_imu[:3,3] = translation+[-.06,.01,0]
    t_imu_base=np.eye(4); t_imu_base[2,3]=-.93
    cam_base=t_cam_imu @ t_imu_base
    stamps=[]
    for i in range(frames):
        image=np.full((375,1242,3),45,'uint8')
        for side,color in [(1.75,(255,255,255)),(-1.75,(0,230,230))]:
            for f in np.arange(3,35,.15):
                ground=np.array([[x,side-.001*(x+i*.1-4)**2+dy,0] for x,dy in [(f,-.07),(f,.07),(f+.15,.07),(f+.15,-.07)]])
                cam=ground @ cam_base[:3,:3].T+cam_base[:3,3]
                if np.any(cam[:,2]<=0): continue
                uv=(cam @ k.T); uv=uv[:,:2]/uv[:,2:]
                cv2.fillConvexPoly(image,np.rint(uv).astype('int32'),color)
        cv2.imwrite(str(seq/f'image_02/data/{i:010d}.png'),image)
        row=np.zeros(30); row[:6]=[49.,8.+i*.1/(6378137*np.cos(np.radians(49)))*180/np.pi,100.,0,0,0]
        row[8]=1.; row[13]=9.81; row[23:]=[.1,.1,4,10,4,4,4]
        np.savetxt(seq/f'oxts/data/{i:010d}.txt',row.reshape(1,-1))
        stamps.append(str(np.datetime64('2011-09-26T13:00:00.000000000')+np.timedelta64(i*100000000,'ns')).replace('T',' '))
    text='\n'.join(stamps)+'\n'; (seq/'image_02/timestamps.txt').write_text(text); (seq/'oxts/timestamps.txt').write_text(text)
    return seq,k,cam_base


@pytest.fixture
def kitti_sample(tmp_path):
    return make_kitti(tmp_path)
