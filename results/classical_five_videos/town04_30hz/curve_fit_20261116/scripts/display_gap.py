"""HUD line vs paint: |y(display line) - y(observed paint lane)| at x = 6, 8, 10, 15, 20 m, frames whose
display side is drawn as observed (not predicted); p95 per segment, cm.  Display from display_filter_eval --output.
usage: display_gap.py town04|town05|maploop RUN_DIR=DISPLAY_DIR ..."""
import sys,gzip,json,numpy as np
SEGS=dict(town04=[('straight',30,35),('curve',15,20),('curve',20,25),('curve',50,55),('straight',40,45)],
          town05=[('straight',0,9),('curve',15,30)],maploop=[('straight',5,20),('curve',22,35)])
XS=(6,8,10,15,20)
def at(p,x):p=np.asarray(p);return float(np.interp(x,p[:,0],p[:,1])) if p[0,0]<=x<=p[-1,0] else None
for spec in sys.argv[2:]:
    run,disp=spec.split('=');rows=[json.loads(l) for l in gzip.open(run+'/predictions.jsonl.gz','rt')]
    shown={json.loads(l)['frame']:json.loads(l) for l in gzip.open(disp+'/display.jsonl.gz','rt')}
    t0=rows[0]['stamp']
    for g,a,b in SEGS[sys.argv[1]]:
        line=f'{disp.split("/")[-1]:24s} {g} {a}-{b}'
        for side in ('left','right'):
            v={x:[] for x in XS}
            for r in rows:
                if not a<=r['stamp']-t0<b:continue
                d=shown.get(r['frame'],{}).get('sides',{}).get(side)
                if not d or d.get('predicted'):continue
                lanes=[l['points'] for l in r['lanes'] if l['side']==side and not l['reprojected']]
                for x in XS:
                    yd=at(d['points'],x);ys=[y for y in (at(l,x) for l in lanes) if y is not None]
                    ys=ys+[(p+q)/2 for i,p in enumerate(ys) for q in ys[i+1:] if .15<=abs(p-q)<=.40]
                    if yd is not None and ys:v[x].append(min(abs(y-yd) for y in ys))
            line+=f' {side[0].upper()} '+'/'.join(f'{np.percentile(v[x],95)*100:.0f}' if v[x] else '-' for x in XS)
        print(line)
