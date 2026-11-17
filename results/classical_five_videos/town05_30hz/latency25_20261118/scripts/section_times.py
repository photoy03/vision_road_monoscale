#!/usr/bin/env python3
"""Plan 25 step 2: wall time per detector section (inclusive, perf_counter wrappers; little overhead), on the
cached BEVs, as the worker calls it (process(defer=True), finish()).  Mean and p95 per frame, and the mean on the
heaviest 10% of frames.  usage: section_times.py VIDEO   writes ../metrics/sections_<video>.json"""
import sys,json,time,functools,collections
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import maploop_offline_compare as M
from vision_road.core import Timeline
from vision_road import temporal,carla,marking_style,centreline
from vision_road.temporal import TemporalDetector
from vision_road.carla import CarlaDetector
video=sys.argv[1];threads=int(sys.argv[2]) if len(sys.argv)>2 else 2
cache=Path('/media/jaehun/38F7-677B/vision-road-cache')/f'{video}_offline';pk=ROOT/'src/vision_road';cfg=M.settings(pk)
meta=json.loads((cache/'frames.json').read_text());valid=cv2.imread(str(cache/'valid.png'),0);valid[M.projector(pk,cfg).my>=1080*M.LAUNCH['image_roi_bottom']]=0
cv2.setNumThreads(threads);acc=collections.defaultdict(float);depth=collections.Counter()
def wrap(owner,name,label):
    f=getattr(owner,name)
    @functools.wraps(f)
    def w(*a,**k):
        if depth[label]:return f(*a,**k)  # recursion: count once
        depth[label]+=1;s=time.perf_counter()
        try:return f(*a,**k)
        finally:acc[label]+=time.perf_counter()-s;depth[label]-=1
    setattr(owner,name,w)
for n in ('binary','fragment_graph','detect','merge_strands','pair_vote','spine','parallel_support','coherent_paths','best_path','outer_lines','paint_fragments','share_shape','edge_like','recall_pairs','store_pairs','cut_tail','crossing'):
    if hasattr(CarlaDetector,n):wrap(CarlaDetector,n,'carla.'+n)
for n in ('_registration','_boundaries','_accumulate_style','_visible','_fragment_keys','_outer','_centres','finish'):
    wrap(TemporalDetector,n,'temporal.'+n)
wrap(temporal,'classify_lane','classify_lane') if hasattr(temporal,'classify_lane') else None
det=TemporalDetector(cfg);det.detector=CarlaDetector(cfg);tl=Timeline(.25);k=0;rows=[]
ims=[cv2.imread(str(cache/'bev'/f'{i:05d}.png')) for i in range(len(meta['frames']))]
for i in range(len(ims)):
    t=meta['frames'][i]['stamp']
    while k<len(meta['poses']) and meta['poses'][k][0]<=t+.05:tl.add(*meta['poses'][k]);k+=1
    acc.clear();s=time.perf_counter();det.process(ims[i],valid,t,tl.at(t),defer=True);m=time.perf_counter()
    row={k2:v*1e3 for k2,v in acc.items() if k2!='temporal.finish'};row['process']=(m-s)*1e3
    acc.clear();det.finish();row['finish']=(time.perf_counter()-m)*1e3;row.update({'finish.'+k2:v*1e3 for k2,v in acc.items() if k2!='temporal.finish'});rows.append(row)
keys=sorted({k2 for r in rows for k2 in r});P=np.array([r['process'] for r in rows]);heavy=P>=np.percentile(P,90)
out={}
for k2 in keys:
    v=np.array([r.get(k2,0.) for r in rows]);out[k2]=dict(mean=round(float(v.mean()),3),p95=round(float(np.percentile(v,95)),3),heavy_mean=round(float(v[heavy].mean()),3))
json.dump(out,open(Path(__file__).resolve().parent.parent/'metrics'/f'sections_{video}.json','w'),indent=1)
for k2,v in sorted(out.items(),key=lambda a:-a[1]['heavy_mean']):print(f'{k2:32s} mean {v["mean"]:7.3f}  p95 {v["p95"]:7.3f}  heavy10% {v["heavy_mean"]:7.3f}')
