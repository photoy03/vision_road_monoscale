"""Near-end (3-6 m) direction of the displayed line, frame to frame, vs paint fragments leaving 3 m.

usage: near_end.py PREDICTIONS TRUTH_BASE.json [name=value ...FilterSettings]
Per frame and side, the previous frame's line is carried into the current base_link with the
odometry, then compared:
  head   change of the slope between 3 m and 6 m (mrad)
  y3     change of the lateral value at the line's near end (cm) (display: at near_m)
Done for the boundary road_node publishes (its first 3 m) and for the display filter output.
An event frame is one where the nearest observed paint on that side starts > 1.5 m farther
than in the previous frame (a fragment slid under 3 m).
"""
import sys,gzip,json,math
from pathlib import Path
from dataclasses import replace
import numpy as np
sys.path.insert(0,'/home/jaehun/Desktop/vision-road/src/vision_road')
from vision_road.display_filter import DisplayFilter,FilterSettings,relative
from vision_road.core import Timeline
SEG=[('straight',5,20),('curve',22,35),('straight2',35,42),('junction',0,5),('crosswalk',42,48),('island',48,53),('junction2',53,61)]
def carry(p,m):
    dx,dy,dyaw=m;c,s=math.cos(dyaw),math.sin(dyaw);px=p[:,0]-dx;py=p[:,1]-dy
    return np.column_stack([c*px+s*py,-s*px+c*py])
def at(p,x):
    p=p[np.argsort(p[:,0])]
    return float(np.interp(x,p[:,0],p[:,1])) if p[0,0]-1e-6<=x<=p[-1,0]+1e-6 else None
def near_seg(p):
    """(x0, y at x0, slope over x0..x0+3) of a polyline."""
    p=p[np.argsort(p[:,0])];x0=float(p[0,0]);y0=at(p,x0);y1=at(p,x0+3.)
    return x0,y0,None if y1 is None else (y1-y0)/3.
def main():
    pred,truth=sys.argv[1],sys.argv[2];s=FilterSettings()
    for kv in sys.argv[3:]:
        k,v=kv.split('=');s=replace(s,**{k:type(getattr(s,k))(eval(v))})
    rows=[json.loads(l) for l in gzip.open(pred,'rt')];odom=json.load(open(truth))
    f=DisplayFilter(s);tl=Timeline(.25);k=0;last=None;t0=rows[0]['stamp'];rec=[]
    for r in rows:
        t=r['stamp']
        while k<len(odom) and odom[k][0]<=t+.1:tl.add(*odom[k]);k+=1
        pose=tl.at(t);motion=None
        if pose is not None:
            motion=(0.,0.,0.) if last is None or t<=last[0] else relative(last[1],pose);last=(t,pose)
        for side in ('left','right'):
            st=[l for l in r['lanes'] if l['side']==side and not l['reprojected'] and l['style']!='unknown']
            if st:f.update_style(t,side,max(st,key=lambda l:l['hits'])['style'],1.)
        paint=[(l['side'],l['points']) for l in r['lanes'] if not l['reprojected']]
        f.update(t,[(b['side'],b['points'],b['reprojected']) for b in r['boundaries']],motion,paint)
        out={side:p for side,p,_,_ in f.output()}
        e={'t':t-t0,'motion':motion}
        for side in ('left','right'):
            obs=[np.asarray(b['points'],float) for b in r['boundaries'] if b['side']==side and not b['reprojected']]
            b=max(obs,key=lambda p:np.ptp(p[:,0]),default=None)
            paint=[np.asarray(l['points'],float) for l in r['lanes'] if l['side']==side and not l['reprojected']]
            tr=f.sides[side];m=tr.memory
            e[side+'_mem']=None if out.get(side) is None else ('none' if not len(m) or b is None or not len(m[(m[:,0]<b[:,0].min()-.25)&(m[:,0]>=s.near_m-.25)]) else 'zero' if len(m[np.abs(m[:,0]-s.near_m)<=.25]) else 'part')
            e[side]=dict(bnd=b,disp=out.get(side),paint_near=min((float(p[:,0].min()) for p in paint),default=None),
                         style=next((l['style'] for l in r['lanes'] if l['side']==side and not l['reprojected'] and l['style']!='unknown'),None))
        rec.append(e)
    res={}
    for name,a,bb in SEG:
        for side in ('left','right'):
            d={'bnd_head':[],'bnd_y':[],'disp_head':[],'disp_y3':[],'event':[],'bnd_x0_jump':[],'disp_head03':[],'disp_y0':[]}
            for p,c in zip(rec[:-1],rec[1:]):
                if not(a<=c['t']<bb) or c['motion'] is None:continue
                P,C=p[side],c[side]
                ev=P['paint_near'] is not None and C['paint_near'] is not None and C['paint_near']-P['paint_near']>1.5
                bh=by=dh=dy=None;jump=None
                if P['bnd'] is not None and C['bnd'] is not None:
                    pc=carry(P['bnd'],c['motion']);x0,y0,sl=near_seg(C['bnd']);_,_,psl=near_seg(pc)
                    jump=x0-float(pc[:,0].min())
                    # same stations in both: the current boundary's near 3 m
                    ya,yb=at(pc,x0),at(pc,x0+3.)
                    if sl is not None and ya is not None and yb is not None:bh=(sl-(yb-ya)/3.)*1e3;by=(y0-ya)*100
                if P['disp'] is not None and C['disp'] is not None:
                    pc=carry(P['disp'],c['motion']);cc=C['disp']
                    a3,a6,b3,b6=at(pc,3.),at(pc,6.),at(cc,3.),at(cc,6.)
                    if None not in (a3,a6,b3,b6):dh=((b6-b3)-(a6-a3))/3.*1e3;dy=(b3-a3)*100
                    a0,b0=at(pc,0.),at(cc,0.)
                    if None not in (a0,b0,a3,b3):d['disp_head03'].append(((b3-b0)-(a3-a0))/3.*1e3);d['disp_y0'].append((b0-a0)*100)
                    else:d['disp_head03'].append(None);d['disp_y0'].append(None)
                else:d['disp_head03'].append(None);d['disp_y0'].append(None)
                d['bnd_head'].append(bh);d['bnd_y'].append(by);d['disp_head'].append(dh);d['disp_y3'].append(dy);d['event'].append(ev);d['bnd_x0_jump'].append(jump)
            ev=np.array(d['event'])
            def stat(key,mask):
                v=np.array([x for x,m in zip(d[key],mask) if m and x is not None],float)
                return None if not len(v) else dict(n=len(v),rms=round(float(np.sqrt(np.mean(v*v))),2),p95=round(float(np.percentile(np.abs(v),95)),2),max=round(float(np.abs(v).max()),2))
            near=np.zeros_like(ev)
            for i in np.flatnonzero(ev):near[max(0,i-1):i+2]=True
            mem=[c[side+'_mem'] for c in rec if a<=c['t']<bb and c[side+'_mem'] is not None]
            res[f'{name}/{side}']=dict(events=int(ev.sum()),frames=len(ev),memory={k:round(mem.count(k)/max(1,len(mem)),3) for k in ('zero','part','none')},
                **{f'{k}_{w}':stat(k,m) for k in ('bnd_head','bnd_y','disp_head','disp_y3','disp_head03','disp_y0') for w,m in (('all',np.ones_like(ev)),('event',ev),('other',~near))},
                big_disp_head_on_event=None)
            dh=np.array([np.nan if x is None else abs(x) for x in d['disp_head']])
            big=dh>np.nanpercentile(dh,95) if np.isfinite(dh).any() else np.zeros_like(ev)
            res[f'{name}/{side}']['big_disp_head_on_event']=f'{int((big&near).sum())}/{int(big.sum())} (event±1 covers {near.mean():.2f} of frames)'
            bh=np.array([np.nan if x is None else abs(x) for x in d['bnd_head']])
            bigb=bh>np.nanpercentile(bh,95) if np.isfinite(bh).any() else np.zeros_like(ev)
            res[f'{name}/{side}']['big_bnd_head_on_event']=f'{int((bigb&near).sum())}/{int(bigb.sum())}'
    return res,rec
if __name__=='__main__':
    res,_=main()
    for k,v in res.items():
        print(k,'events',v['events'],'/',v['frames'],'| big disp head on event',v['big_disp_head_on_event'],'| big bnd head on event',v['big_bnd_head_on_event'])
        print('   memory',v['memory'])
        for key in ('disp_head','disp_y3','disp_head03','disp_y0'):
            print('   %-10s all %s\n              event %s\n              other %s'%(key,v[key+'_all'],v[key+'_event'],v[key+'_other']))
    json.dump(res,open(sys.argv[0].replace('.py','')+'_'+'_'.join(a.replace('=','') for a in sys.argv[3:])+'.json','w'),indent=1)
