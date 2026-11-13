import sys
P,T=sys.argv[1],sys.argv[2];extra=sys.argv[3:]
sys.argv=['x']
import near_end as N
def go(args):
    sys.argv=['x',P,T]+args;_,rec=N.main();out={}
    for i,(p,c) in enumerate(zip(rec[:-1],rec[1:])):
        for side in ('left','right'):
            A,B=p[side]['disp'],c[side]['disp']
            if A is None or B is None or c['motion'] is None:continue
            pc=N.carry(A,c['motion']);v=[N.at(pc,x) for x in (0.,3.,6.)]+[N.at(B,x) for x in (0.,3.,6.)]
            out[(i+1,side)]=(c['t'],v,c[side+'_mem'],None if c[side]['bnd'] is None else round(float(c[side]['bnd'][:,0].min()),2))
    return out
a=go(extra);rows=[]
for k,(t,v,m,o) in a.items():
    if None in v:continue
    h03=((v[4]-v[3])-(v[1]-v[0]))/3e-3;h36=((v[5]-v[4])-(v[2]-v[1]))/3e-3
    rows.append((max(abs(h03),abs(h36)),k,round(t,2),'h03',round(h03,1),'h36',round(h36,1),'y0',round((v[3]-v[0])*100,1),m,o))
rows.sort(reverse=True)
for r in rows[:12]:print(r[1:])
