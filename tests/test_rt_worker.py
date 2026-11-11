"""The isolated path must preserve native camera projection and paint geometry."""
from pathlib import Path
import cv2,numpy as np,yaml
from vision_road.core import Settings,Projector
from vision_road.temporal import TemporalDetector
from vision_road.carla import CarlaDetector
from vision_road.rt_worker import ComputeWorker


def test_shared_worker_matches_inline_native_camera_and_releases_resources():
    root=Path(__file__).resolve().parents[1];config=root/'src/vision_road/config'
    cfg=Settings(min_span_m=.4)
    rig=yaml.safe_load((config/'rig.yaml').read_text())
    calibration=config/'calibration/front.npz'
    projection=Projector(calibration,rig,cfg)
    image=cv2.imread(str(root/'artifacts/synthetic_camera.png'));assert image is not None
    worker=ComputeWorker(Projector(calibration,rig,cfg),cfg,'carla',1.,2)
    detector=TemporalDetector(cfg);detector.detector=CarlaDetector(cfg)
    try:
        for i in range(4):
            stamp=1+i*.05;pose=(0.,0.,0.)
            bev=projection.warp(image);valid=projection.valid.copy()
            expected,mask=detector.process(bev.copy(),valid,stamp,pose)
            result=worker.compute(image,stamp,pose)
            assert np.array_equal(bev,result['bev']) and np.array_equal(mask,result['binary'])
            assert len(expected)==len(result['lanes'])
            for a,b in zip(expected,result['lanes']):
                assert (a.side,a.style,a.reprojected)==(b.side,b.style,b.reprojected)
                assert np.allclose(a.points,b.points,rtol=0,atol=1e-10)
        worker.reset_pending=True
        result=worker.compute(np.zeros_like(image),.5,pose)
        assert not result['lanes']
    finally:worker.close()
    assert not worker.process.is_alive()
