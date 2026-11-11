import cv2
import numpy as np
import pytest
from vision_road.kitti import KittiSequence, timestamps
from vision_road.core import Projector, Detector, Settings


def test_rectified_transform_baseline_ground_pose(kitti_sample,tmp_path):
    path,k,cam_base=kitti_sample; d=KittiSequence(path)
    assert np.allclose(np.linalg.inv(d.t_base_cam),cam_base)
    assert np.allclose(d.k,k)
    assert np.isclose(d.t_base_cam[2,3],1.66)  # nonzero rectified y translation shifts origin
    assert np.isclose(d.poses[10][0,3],1.,atol=1e-5)
    assert np.isclose(d.poses[0][2,3],0)
    assert np.allclose(d.t_base_cam[:3,:3].T@d.t_base_cam[:3,:3],np.eye(3))
    output=d.export(tmp_path/'generated'); c=Settings(forward_min=3,forward_max=35,lateral_min=-5,lateral_max=5,mpp=.05)
    projector=Projector(output/'camera.npz',d.rig,c); image=d.image(0); bev=projector.warp(image)
    lanes=Detector(c).detect(Detector(c).binary(bev,projector.valid))
    assert len(lanes)==2
    for lane,side in zip(lanes,[1.75,-1.75]):
        assert np.max(np.abs(lane.points[:,1]-(side-.001*(lane.points[:,0]-4)**2)))<.16
    points=np.array([[10.,1.75],[20.,1.5]])
    actual,ok=projector.image_points(points)
    cam=np.column_stack([points,np.zeros(2)])@cam_base[:3,:3].T+cam_base[:3,3]
    expected=cam@k.T; expected=expected[:,:2]/expected[:,2:]
    assert np.all(ok) and np.allclose(actual,expected)


def test_kitti_reject_missing_and_mismatched_inputs(kitti_sample,tmp_path):
    path,_,_=kitti_sample
    with pytest.raises(ValueError,match='positive'): KittiSequence(path,-.1)
    with pytest.raises(ValueError,match='sync'): KittiSequence(tmp_path/'missing')
    (path/'image_02/data/0000000001.png').unlink()
    with pytest.raises(ValueError,match='counts'): KittiSequence(path)


def test_timestamps_preserve_nanoseconds(tmp_path):
    f=tmp_path/'timestamps.txt'; f.write_text('2011-09-26 13:00:00.123456789\n2011-09-26 13:00:00.123456790\n')
    assert np.diff(timestamps(f))[0]==1
    f.write_text('2011-09-26 13:00:00.1\n2011-09-26 13:00:00.1\n')
    with pytest.raises(ValueError,match='increasing'): timestamps(f)


def test_replay_holds_images_until_future_imu_without_retimestamping():
    from vision_road.kitti import replay_events
    events=replay_events([1.,1.1,1.2],[1.003,1.104,1.205])
    assert all(a[0]<=b[0] for a,b in zip(events,events[1:]))
    for delivery,kind,i,acquired in events:
        if kind==1:
            assert acquired==[1.,1.1,1.2][i]
            assert any(e[1]==0 and e[3]>acquired and e[0]<delivery for e in events)
