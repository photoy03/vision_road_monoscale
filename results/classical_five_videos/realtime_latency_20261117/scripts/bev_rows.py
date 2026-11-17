"""Which source image rows the BEV warp reads (maploop dashcam calibration, carla.yaml BEV, ROI bottom cut)."""
import sys,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import maploop_offline_compare as M
pk=ROOT/'src/vision_road';cfg=M.settings(pk);p=M.projector(pk,cfg)
my=p.my[p.valid>0];mx=p.mx[p.valid>0];roi=1080*M.LAUNCH['image_roi_bottom']
used=my[my<roi]
lo,hi=int(np.floor(used.min())),int(np.ceil(used.max()))+1
print(f'BEV {p.valid.shape}, source rows read {lo}..{hi} of 1080 ({hi-lo} rows, {100*(hi-lo)/1080:.1f}%), columns {int(mx.min())}..{int(np.ceil(mx.max()))}')
print(f'full frame {1920*1080*3/1e6:.2f} MB, rows only {1920*(hi-lo)*3/1e6:.2f} MB')
