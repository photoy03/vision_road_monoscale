"""Display filter output (as the HUD gets it) from offline predictions + recorded poses; curve and straight."""
import sys,gzip,json,numpy as np
sys.path.insert(0,'/home/jaehun/Desktop/vision-road/tools');sys.path.insert(0,'/home/jaehun/Desktop/vision-road/src/vision_road')
import display_filter_eval as E
from vision_road.display_filter import FilterSettings
meta=json.load(open(sys.argv[1]));odom=meta['poses']
for spec in sys.argv[2:]:
    name,path=spec.split('=');rows=[json.loads(l) for l in gzip.open(path+'/predictions.jsonl.gz','rt')]
    for r in rows:
        for l in r['lanes']:l.setdefault('hits',1)
    out=E.run(rows,FilterSettings(),odom);line=[name]
    for seg,(a,b) in (('straight',(150,600)),('curve',(660,1050))):
        m=E.metrics(rows[a:b],out[a:b],(rows[660]['stamp'],rows[1049]['stamp']))
        y20={}
        for side in ('left','right'):
            v=[E.at(o[side][0],20.) if side in o else None for o in out[a:b]];d=np.array([q-p for p,q in zip(v[:-1],v[1:]) if p is not None and q is not None]);d=d[np.abs(d)<.3]
            y20[side]=np.std(d)*100 if len(d) else float('nan')
        line.append('%s L shake %.2f y20 %.2f R shake %.2f y20 %.2f shown %d tog %d lag %s'%(seg,m['left']['shake_cm']['filtered'],y20['left'],m['right']['shake_cm']['filtered'],y20['right'],m['right']['shown_frames']['filtered'],m['right']['toggles']['filtered'],m['right']['curve_lag_m']['mean_abs']))
    print(' | '.join(line))
