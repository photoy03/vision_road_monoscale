"""Every candidate tail cut (gap >= 4 m, near paint >= 6 m) with its gap, the offset of the paint after it
from the near curve, the near curve's degree and whether the near part was dense (solid) - every 3rd frame,
CarlaDetector alone (no temporal).  usage: tail_scan.py CACHE"""
import sys,json,cv2,numpy as np
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import maploop_offline_compare as M
from vision_road.carla import CarlaDetector
cache=Path(sys.argv[1]);cfg=M.settings(ROOT/'src/vision_road');d=CarlaDetector(cfg);valid=cv2.imread(str(cache/'valid.png'),0);rows=[]
orig=d.coherent_paths;meta=json.loads((cache/'frames.json').read_text());frame=[0]
def cp(nodes):
    groups=orig(nodes)
    for side,path in groups.items():
        for k in range(1,len(path)):
            start=path[k]['lane'].points[0,0];end=max(a['lane'].points[-1,0] for a in path[:k]);gap=start-end
            if gap<4:continue
            before=np.vstack([a['lane'].points for a in path[:k]]);span=np.ptp(before[:,0])
            if span<6:continue
            deg=2 if span>=10 else 1;fit=np.polyfit(before[:,0],before[:,1],deg)
            ahead=path[k]['lane'].points;ahead=ahead[ahead[:,0]<=start+2.]
            off=float(np.median(np.abs(ahead[:,1]-np.polyval(fit,ahead[:,0]))))
            dense=sum(np.ptp(a['lane'].points[:,0]) for a in path[:k])/span
            slope_before=np.polyval(np.polyder(fit),end);sa=np.polyfit(path[k]['lane'].points[:,0],path[k]['lane'].points[:,1],1)[0] if np.ptp(path[k]['lane'].points[:,0])>.5 else np.nan
            rows.append((frame[0],side,round(end,1),round(gap,1),round(off,2),deg,round(dense,2),round(float(np.degrees(np.arctan(sa)-np.arctan(slope_before))),0)))
    return groups
d.coherent_paths=cp
for i in range(0,len(meta['frames']),3):
    frame[0]=i;d.detect(d.binary(cv2.imread(str(cache/'bev'/f'{i:05d}.png')),valid))
for r in rows:print(*r)
