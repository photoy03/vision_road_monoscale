"""Where the boundary leaves the paint, stage by stage, per side, frames FIRST-LAST (carla.yaml settings).
  center : grouped median of the chosen chain's paint (what spine() fits)
  spine  : spine()'s robust quadratic                      -> fit = |spine - center|
  cand   : the boundary candidate after share_shape / strand offsets (detector output)
  final  : TemporalDetector.boundary_lanes (after temporal smoothing)
  simp   : final after approxPolyDP 0.025 m (as published)
p95 of |stage - center| at x = 6, 8, 10, 15, 20 m (cm).
usage: [SET=name=value,...] stage_gap.py CACHE FIRST-LAST [FIRST-LAST ...]"""
import sys,json,math,cv2,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import maploop_offline_compare as M
from vision_road.core import Timeline
from vision_road.temporal import TemporalDetector
from vision_road.carla import CarlaDetector
import os
from dataclasses import replace
cache=Path(sys.argv[1]);cfg=M.settings(ROOT/'src/vision_road')
for kv in filter(None,os.environ.get('SET','').split(',')):
    k_,v_=kv.split('=');cfg=replace(cfg,**{k_:type(getattr(cfg,k_))(float(v_))})
meta=json.loads((cache/'frames.json').read_text())
valid=cv2.imread(str(cache/'valid.png'),0);valid[M.projector(ROOT/'src/vision_road',cfg).my>=1080*M.LAUNCH['image_roi_bottom']]=0
XS=(6,8,10,15,20)
def at(x_,y_,x):return float(np.interp(x,x_,y_)) if x_[0]<=x<=x_[-1] else None
def simplify(p):return cv2.approxPolyDP(p.astype(np.float32),.025,False).reshape(-1,2).astype(float) if len(p)>=3 else p
for rng in sys.argv[2:]:
    a,b=map(int,rng.split('-'))
    det=TemporalDetector(cfg);det.detector=CarlaDetector(cfg);d=det.detector;tl=Timeline(.25);k=0;spines={}
    orig=d.spine
    def spine(group,forward=None,center=None):
        r=orig(group,forward,center)
        f,cn=(forward,center) if forward is not None else __import__('vision_road.carla',fromlist=['grouped_median']).grouped_median(np.vstack([g['lane'].points for g in group]))
        spines[group[0]['lane'].side]=(np.asarray(f),np.asarray(cn),r[0],r[1]);return r
    d.spine=spine
    stats={s:{st:{x:[] for x in XS} for st in ('fit','cand','final','simp')} for s in ('left','right')}
    for row in meta['frames'][max(0,a-60):b]:
        i,t=row['frame'],row['stamp']
        while k<len(meta['poses']) and meta['poses'][k][0]<=t+.05:tl.add(*meta['poses'][k]);k+=1
        spines.clear();det.process(cv2.imread(str(cache/'bev'/f'{i:05d}.png')),valid,t,tl.at(t))
        if i<a:continue
        cands={l.side:l.points for l in d.boundary_candidates};finals={l.side:l.points for l in det.boundary_lanes if not l.reprojected}
        for side,(f,cn,sx,sy) in spines.items():
            if side not in finals:continue
            for x in XS:
                c0=at(f,cn,x)
                if c0 is None:continue
                for st,(xx,yy) in (('fit',(sx,sy)),('cand',(cands[side][:,0],cands[side][:,1]) if side in cands else (None,None)),('final',(finals[side][:,0],finals[side][:,1])),('simp',tuple(simplify(finals[side]).T))):
                    if xx is None:continue
                    v=at(xx,yy,x)
                    if v is not None:stats[side][st][x].append(abs(v-c0))
    for side in ('left','right'):
        print(f'{rng} {side:5s} '+'  '.join(f"{st} "+'/'.join(f'{np.percentile(v[x],95)*100:.0f}' if v[x] else '-' for x in XS) for st,v in stats[side].items()))
