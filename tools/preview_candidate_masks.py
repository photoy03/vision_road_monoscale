#!/usr/bin/env python3
"""Inspect current-frame masks; no temporal predictions are drawn in these panels."""
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/vision_road'))
from vision_road.core import Settings, Projector
from vision_road.classical import PaintEvidenceDetector
from vision_road.kitti import KittiSequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, default=ROOT/'data/kitti/2011_09_26')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT/'src/vision_road/config/kitti.yaml').read_text())['vision_road']['ros__parameters']
    c = Settings(**{k:v for k,v in cfg.items() if k in Settings.__dataclass_fields__})
    stats = []
    best = []
    for seq in ['0001','0015','0028','0018','0056']:
        d = KittiSequence(args.data_root/f'2011_09_26_drive_{seq}_sync')
        out = args.output/seq
        d.export(out)
        p = Projector(out/'camera.npz', d.rig, c)
        det = PaintEvidenceDetector(c)
        panels = []
        for i in sorted({0,len(d.images)//4,len(d.images)//2,3*len(d.images)//4,len(d.images)-1}):
            im = d.image(i)
            bev = p.warp(im)
            after = det.binary(bev, p.valid)
            before = det.candidate_mask_before
            removed = (before > 0) & (after == 0)
            record = dict(sequence=seq, frame=i, before_pixels=int(np.count_nonzero(before)), after_pixels=int(np.count_nonzero(after)), removed_pixels=int(removed.sum()))
            stats.append(record)
            canvas = np.full((710, 1400, 3), 24, np.uint8)
            canvas[70:312, :800] = cv2.resize(im, (800,242))
            cv2.putText(canvas,f'KITTI {seq} / frame {i} - CURRENT FRAME ONLY',(15,32),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
            for text, y in [('White/yellow + local contrast candidates',370),('Reject wide row runs > 0.45 m',410),('Keep long curves and separate dash gaps',450),(f"Pixels: {record['before_pixels']} -> {record['after_pixels']}",510),(f"Removed: {record['removed_pixels']} (not an accuracy score)",550)]:
                cv2.putText(canvas,text,(15,y),cv2.FONT_HERSHEY_SIMPLEX,.65,(230,230,230),1)
            overlay = bev.copy()
            overlay[removed] = (0,0,255)
            for x, title, image in [(800,'BEV / removed red',overlay),(1000,'Before',cv2.cvtColor(before,cv2.COLOR_GRAY2BGR)),(1200,'Filtered',cv2.cvtColor(after,cv2.COLOR_GRAY2BGR))]:
                cv2.putText(canvas,title,(x+3,55),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
                canvas[70:710,x:x+200] = cv2.resize(image,(200,640),interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(out/f'mask_{i:06d}.jpg'),canvas)
            panels.append((record['removed_pixels'],canvas))
        best.append(max(panels,key=lambda item:item[0])[1])
    (args.output/'mask_stats.json').write_text(json.dumps(stats,indent=2))
    cv2.imwrite(str(args.output/'comparison.jpg'),np.vstack([cv2.resize(panel,(1008,511)) for panel in best]))
    print(json.dumps(stats,indent=2))

if __name__=='__main__':
    main()
