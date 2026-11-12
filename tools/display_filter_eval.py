#!/usr/bin/env python3
"""Replay recorded boundary predictions through the HUD display filter and compare before/after.

Input is the predictions.jsonl.gz written by maploop_offline_compare.py run: per
30 Hz frame, the boundary polylines road_node publishes (already simplified)
and the paint lanes with their styles.  With --recorded, the input is instead a
recording of the live topics (/road/lane_boundary_markers, /road/lane_display_markers,
styles and /road/reference_odom; one or more replay passes) and the node's own
output is scored, or with --refilter the filter is re-run on the recorded input
and odometry.  The predictions file carries no odometry, so there the filter
runs without motion compensation (lost sides held only odom_lost_hold_s).
Metrics per side, raw vs filtered:
  jitter      std / p90 of the frame-to-frame change of the lateral position at 10 m
              (all frames, and straight frames only); steps above 0.30 m are
              counted as jumps and kept out of the std
  shake       std of the residual from a centred 0.5 s linear trend at 10 m,
              which removes real drift such as a lane change; windows holding
              a step above 0.30 m (line switch) are left out
  toggles     number of times the side appears (off -> on)
  curve lag   mean |filtered - raw observed| at 10 m inside the curve window
  mismatch    frames drawn as observed that are > 0.30 m from the raw observation at 10 m
              (at the observation's nearest point when it does not cover 10 m)
  reacquire   when a side held without any observation is observed again,
              |held line (previous frame) - new observation| at 10 m
"""
import argparse, gzip, json, sys
from dataclasses import replace
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/vision_road'))
from vision_road.display_filter import DisplayFilter, FilterSettings, fit, relative  # noqa: E402
from vision_road.core import Timeline  # noqa: E402


def at(points,x=10.):
    p=np.asarray(points,float);p=p[np.argsort(p[:,0])]
    return float(np.interp(x,p[:,0],p[:,1])) if p[0,0]<=x<=p[-1,0] else None


def run(rows,settings,odom=None):
    f=DisplayFilter(settings);out=[];timeline=Timeline(.25);k=0;last=None
    for r in rows:
        t=r['stamp']
        motion=None
        if odom is not None:
            # Same lookup as the node: odometry received up to this frame, interpolated at its stamp.
            while k<len(odom) and odom[k][0]<=t+.1:timeline.add(*odom[k]);k+=1
            pose=timeline.at(t)
            if pose is not None:
                motion=(0.,0.,0.) if last is None or t<=last[0] else relative(last[1],pose);last=(t,pose)
        for side in ('left','right'):
            styled=[l for l in r['lanes'] if l['side']==side and not l['reprojected'] and l['style']!='unknown']
            if styled:
                lane=max(styled,key=lambda l:l['hits']);f.update_style(t,side,lane['style'],1.)
        f.update(t,[(b['side'],b['points'],b['reprojected']) for b in r['boundaries']],motion)
        out.append({side:(p,pred,style) for side,p,pred,style in f.output()})
    return out


def metrics(rows,filtered,curve):
    result={}
    for side in ('left','right'):
        raw=[];obs=[];disp=[];straight=[];near=[]
        for r,f in zip(rows,filtered):
            lines=[b for b in r['boundaries'] if b['side']==side]
            best=max(lines,key=lambda b:np.ptp(np.asarray(b['points'])[:,0]),default=None)
            raw.append(None if best is None else at(best['points']))
            observed=[b for b in lines if not b['reprojected']]
            o=max(observed,key=lambda b:np.ptp(np.asarray(b['points'])[:,0]),default=None)
            obs.append(None if o is None else at(o['points']))
            # Compare at 10 m, or at the nearest point of an observation that does not reach it.
            if o is None or side not in f:near.append(None)
            else:
                ox=np.asarray(o['points'])[:,0];x=float(np.clip(10.,ox.min(),ox.max()))
                a,b=at(f[side][0],x),at(o['points'],x)
                near.append(None if a is None or b is None else abs(a-b))
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
        flags=[f[side][1] if side in f else None for f in filtered]
        mismatch=sum(1 for n,p in zip(near,flags) if p is False and (n is None or n>.3))
        # A hold is a run of shown frames with no observed raw boundary at all (a rejected
        # jump is not a hold: its "new observation" is the jump itself).
        seen=[any(b['side']==side and not b['reprojected'] for b in r['boundaries']) for r in rows]
        reacq=[];run_len=0
        for i in range(len(rows)):
            if i and run_len and seen[i] and disp[i-1] is not None and obs[i] is not None:
                reacq.append((abs(disp[i-1]-obs[i]),run_len))
            run_len=run_len+1 if (not seen[i] and disp[i] is not None) else 0
        long_reacq=[e for e,n in reacq if n>=5];reacq=[e for e,_ in reacq]
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
            observed_mismatch_frames=mismatch,
            reacquire_m=dict(mean=round(float(np.mean(reacq)),3) if reacq else None,
                             max=round(float(np.max(reacq)),3) if reacq else None,events=len(reacq),
                             mean_after_5_frames=round(float(np.mean(long_reacq)),3) if long_reacq else None,events_after_5_frames=len(long_reacq)),
            style_switches=sum(1 for a,b in zip(filtered[:-1],filtered[1:])
                               if side in a and side in b and a[side][2]!=b[side][2]))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('predictions',type=Path)
    ap.add_argument('--recorded',action='store_true',help='score a live topic recording instead of replaying the filter')
    ap.add_argument('--refilter',action='store_true',help='with --recorded: re-run the filter on the recorded input and odometry')
    ap.add_argument('--pass',dest='passno',type=int,default=0,help='with --recorded: which replay pass')
    ap.add_argument('--output',type=Path)
    ap.add_argument('--curve',type=float,nargs=2,default=(22.,35.))
    ap.add_argument('--set',nargs='*',default=[],help='FilterSettings overrides, name=value')
    a=ap.parse_args()
    s=FilterSettings()
    for item in a.set:
        k,v=item.split('=');s=replace(s,**{k:type(getattr(s,k))(eval(v))})
    if a.recorded:
        names={1:'solid',2:'dashed'};rows=[];filtered=[];data=json.loads(a.predictions.read_text())
        recorded=data['rows'] if isinstance(data,dict) else data  # older recordings: one pass, no odometry
        odom=[o[1:] for o in data['odom'] if o[0]==a.passno] if isinstance(data,dict) else None
        for i,r in enumerate(x for x in recorded if x.get('passno',0)==a.passno):
            rows.append(dict(frame=i,stamp=r['stamp'],boundaries=[dict(side=b['side'],points=b['points'],reprojected=b['predicted']) for b in r.get('raw',[])],
                             lanes=[dict(side=l['side'],reprojected=l['reprojected'],hits=l['confidence'],
                                         style={1:'solid',2:'dashed'}.get(l['style'],'unknown')) for l in r.get('raw_styles',[])
                                    if l['style']!=2 or l['confidence']>=s.dashed_min_confidence]))
            style={l['side']:names.get(l['style']) for l in r.get('display_styles',[])}
            filtered.append({b['side']:(np.asarray(b['points']),b['predicted'],style.get(b['side'])) for b in r.get('display',[])})
        if a.refilter:filtered=run(rows,s,odom)
        else:s=None
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
