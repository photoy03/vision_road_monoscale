"""evaluate.py definitions (10 m, probes, jumps, reach) for offline runs of this task."""
import sys,json,cv2,numpy as np
from pathlib import Path
E_DIR=Path('/home/jaehun/Desktop/vision-road/results/classical_five_videos/maploop30hz/learned_compare_20261113')
sys.path.insert(0,str(E_DIR));import evaluate as E
labels=json.loads((E_DIR.parent/'labels/labels.json').read_text())
out={}
for spec in sys.argv[1:]:
    name,path=spec.split('=');rows=E.load(Path(path));res={}
    for side in ('left','right'):
        y,reach=E.side_series(rows,side)
        for a,b,seg in E.SEGMENTS:
            sl=slice(a*30,min(b*30,len(rows)));v=y[sl];d=np.diff(v);d=d[np.isfinite(d)];small=d[np.abs(d)<.3]
            res[f'{seg}/{side}']=dict(observed=round(float(np.isfinite(v).mean()),3),step_std_cm=round(float(small.std()*100),2),jumps_over_30cm=int((np.abs(d)>=.3).sum()),
                                      median_reach_m=round(float(np.nanmedian(reach[sl])),1))
    hits=total=0;neg=[]
    for r in labels['positives']:
        m=E.classical_raster(rows[r['frame']]);dist=cv2.distanceTransform(255-m,cv2.DIST_L2,5) if m.any() else None
        for x,yy in r['points']:total+=1;hits+=int(dist is not None and dist[yy,x]<=labels['tolerance_pixels'])
    for r in labels['negatives']:
        region=np.zeros((540,960),np.uint8);cv2.fillPoly(region,[np.asarray(r['polygon'],np.int32)],255)
        neg.append(int(np.count_nonzero(E.classical_raster(rows[r['frame']])&region)))
    res['probes']=dict(hit=hits,points=total,negative_region_pixels=neg);out[name]=res
    g=lambda k,f:res[k][f]
    print('%-10s probes %d/%d neg %s | curve R obs %.3f L %.3f | reach all L %.1f R %.1f | jumps>30cm all L %d R %d | straight std L %.2f R %.2f'%(name,hits,total,neg,g('curve/right','observed'),g('curve/left','observed'),
        g('all/left','median_reach_m'),g('all/right','median_reach_m'),g('all/left','jumps_over_30cm'),g('all/right','jumps_over_30cm'),g('straight/left','step_std_cm'),g('straight/right','step_std_cm')))
if '--json' in ' '.join(sys.argv):pass
json.dump(out,open('/tmp/claude-1000/-home-jaehun-Desktop-vision-road/29db546f-08e5-4fc6-9d69-1be1602904b8/scratchpad/learned_eval.json','w'),indent=1)
