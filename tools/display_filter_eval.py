#!/usr/bin/env python3
"""Replay recorded boundary predictions through the HUD display filter and compare before/after.

Input is the predictions.jsonl.gz written by maploop_offline_compare.py run: per
30 Hz frame, the boundary polylines road_node publishes (already simplified)
and the paint lanes with their styles.  With --recorded, the input is instead a
JSON list recorded live from /road/lane_boundary_markers and
/road/lane_display_markers, and the node's own output is scored.  Metrics per side, raw vs filtered:
  jitter      std / p90 of the frame-to-frame change of the lateral position at 10 m
              (all frames, and straight frames only); steps above 0.30 m are
              counted as jumps and kept out of the std
  shake       std of the residual from a centred 0.5 s linear trend at 10 m,
              which removes real drift such as a lane change; windows holding
              a step above 0.30 m (line switch) are left out
  toggles     number of times the side appears (off -> on)
  curve lag   mean |filtered - raw observed| at 10 m inside the curve window
"""
import argparse, gzip, json, sys
from dataclasses import replace
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/vision_road'))
from vision_road.display_filter import DisplayFilter, FilterSettings, fit  # noqa: E402


def at(points,x=10.):
    p=np.asarray(points,float);p=p[np.argsort(p[:,0])]
    return float(np.interp(x,p[:,0],p[:,1])) if p[0,0]<=x<=p[-1,0] else None


def run(rows,settings):
    f=DisplayFilter(settings);out=[]
    for r in rows:
        t=r['stamp']
        for side in ('left','right'):
            styled=[l for l in r['lanes'] if l['side']==side and not l['reprojected'] and l['style']!='unknown']
            if styled:
                lane=max(styled,key=lambda l:l['hits']);f.update_style(t,side,lane['style'],1.)
        f.update(t,[(b['side'],b['points'],b['reprojected']) for b in r['boundaries']])
        out.append({side:(p,pred,style) for side,p,pred,style in f.output()})
    return out


def metrics(rows,filtered,curve):
    result={}
    for side in ('left','right'):
        raw=[];obs=[];disp=[];straight=[]
        for r,f in zip(rows,filtered):
            lines=[b for b in r['boundaries'] if b['side']==side]
            best=max(lines,key=lambda b:np.ptp(np.asarray(b['points'])[:,0]),default=None)
            raw.append(None if best is None else at(best['points']))
            observed=[b for b in lines if not b['reprojected']]
            o=max(observed,key=lambda b:np.ptp(np.asarray(b['points'])[:,0]),default=None)
            obs.append(None if o is None else at(o['points']))
            disp.append(None if side not in f else at(f[side][0]))
            if o is not None and np.ptp(np.asarray(o['points'])[:,0])>=8:
                s=fit(o['points'])[0];straight.append(abs(s[0]-2*s[1]+s[2])<.05)
            else:straight.append(None)
        stamps=np.array([r['stamp'] for r in rows])

        def jitter(values,mask=None):
            # Steps above the 0.30 m gate are line switches, not shake; they are counted apart.
            d=np.array([values[i]-values[i-1] for i in range(1,len(values)) if values[i] is not None and values[i-1] is not None
                        and (mask is None or (mask[i] and mask[i-1]))])
            if not len(d):return None
            small=d[np.abs(d)<=.3]
            return dict(std=round(float(np.std(small))*100,2),p90_abs=round(float(np.percentile(np.abs(small),90))*100,2),jumps=int(len(d)-len(small)))

        def shake(values,mask=None,half=7):
            # Residual from a centred 0.5 s linear trend: real drift (curves, lane changes) is removed.
            r=[]
            for i in range(half,len(values)-half):
                w=values[i-half:i+half+1]
                if any(v is None for v in w) or (mask is not None and not all(mask[i-half:i+half+1])):continue
                if np.abs(np.diff(w)).max()>.3:continue  # a line switch inside the window
                k=np.arange(-half,half+1);c=np.polyfit(k,w,1)
                r.append(w[half]-np.polyval(c,0))
            return round(float(np.std(r))*100,2) if r else None

        def toggles(values):
            on=[v is not None for v in values]
            return int(sum(1 for a,b in zip(on[:-1],on[1:]) if b and not a))
        inside=(stamps>=curve[0])&(stamps<=curve[1])
        lag=[d-o for d,o,c in zip(disp,obs,inside) if c and d is not None and o is not None]
        result[side]=dict(
            jitter_cm=dict(raw=jitter(raw),filtered=jitter(disp)),
            jitter_straight_cm=dict(raw=jitter(raw,straight),filtered=jitter(disp,straight)),
            shake_cm=dict(raw=shake(raw),filtered=shake(disp),filtered_where_raw=shake(disp,[v is not None for v in raw])),
            shake_straight_cm=dict(raw=shake(raw,straight),filtered=shake(disp,straight)),
            toggles=dict(raw=toggles(raw),filtered=toggles(disp)),
            shown_frames=dict(raw=sum(v is not None for v in raw),filtered=sum(v is not None for v in disp)),
            curve_lag_m=dict(mean_abs=round(float(np.mean(np.abs(lag))),3) if lag else None,
                             p95_abs=round(float(np.percentile(np.abs(lag),95)),3) if lag else None,frames=len(lag)),
            style_switches=sum(1 for a,b in zip(filtered[:-1],filtered[1:])
                               if side in a and side in b and a[side][2]!=b[side][2]))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('predictions',type=Path)
    ap.add_argument('--recorded',action='store_true',help='score a live topic recording instead of replaying the filter')
    ap.add_argument('--output',type=Path)
    ap.add_argument('--curve',type=float,nargs=2,default=(22.,35.))
    ap.add_argument('--set',nargs='*',default=[],help='FilterSettings overrides, name=value')
    a=ap.parse_args()
    s=FilterSettings()
    for item in a.set:
        k,v=item.split('=');s=replace(s,**{k:type(getattr(s,k))(eval(v))})
    if a.recorded:
        names={1:'solid',2:'dashed'};rows=[];filtered=[]
        for i,r in enumerate(json.loads(a.predictions.read_text())):
            rows.append(dict(frame=i,stamp=r['stamp'],lanes=[],boundaries=[dict(side=b['side'],points=b['points'],reprojected=b['predicted']) for b in r.get('raw',[])]))
            style={l['side']:names.get(l['style']) for l in r.get('display_styles',[])}
            filtered.append({b['side']:(np.asarray(b['points']),b['predicted'],style.get(b['side'])) for b in r.get('display',[])})
        s=None
    else:
        rows=[json.loads(l) for l in gzip.open(a.predictions,'rt')]
        filtered=run(rows,s)
    m=dict(settings=s and vars(s),curve_window_s=list(a.curve),sides=metrics(rows,filtered,a.curve))
    print(json.dumps(m['sides'],indent=1))
    if a.output:
        a.output.mkdir(parents=True,exist_ok=False)
        (a.output/'metrics.json').write_text(json.dumps(m,indent=1))
        if a.recorded:return
        with gzip.open(a.output/'display.jsonl.gz','wt') as out:
            for r,f in zip(rows,filtered):
                out.write(json.dumps(dict(frame=r['frame'],stamp=r['stamp'],sides={k:dict(points=np.round(v[0],3).tolist(),predicted=v[1],style=v[2]) for k,v in f.items()}))+'\n')


if __name__=='__main__':
    main()
