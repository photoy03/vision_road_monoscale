#!/usr/bin/env python3
"""CPU use per process during a real-time run, from /proc (no pidstat here).
usage: cpu_sample.py OUT.json [window_s=50] [gpu]
Waits for the replay tool (replay_maploop.py) to start, waits 4 s, then reads utime+stime of every
process and the whole-system /proc/stat, again after the window, and writes per-process CPU %
(100 % = one hardware thread), grouped by role:
  road_node (main process), road_node.compute (perception worker: detector, display filter),
  road_node.learned (YOLOPv2 mask process, the child holding /dev/nvidia*), hud, replay, eval (realtime_eval.py
  and its subscribers), other (everything else, e.g. the desktop and other sessions).
With 'gpu': nvidia-smi samples every 0.5 s in the window (utilization %, memory MiB) and the per-process GPU memory."""
import os,sys,time,json,subprocess
out=sys.argv[1];window=float(sys.argv[2]) if len(sys.argv)>2 else 50.;gpu='gpu' in sys.argv[3:]
HZ=os.sysconf('SC_CLK_TCK');NCPU=os.cpu_count()
def procs():
    r={}
    for p in os.listdir('/proc'):
        if not p.isdigit():continue
        try:
            st=open(f'/proc/{p}/stat').read();rp=st.rindex(')');f=st[rp+2:].split()
            cmd=open(f'/proc/{p}/cmdline','rb').read().replace(b'\0',b' ').decode(errors='replace').strip()
            r[int(p)]=dict(ppid=int(f[1]),ticks=int(f[11])+int(f[12]),cmd=cmd)
        except (OSError,ValueError):pass
    return r
def total():
    f=open('/proc/stat').readline().split()[1:];v=list(map(int,f));return sum(v),v[3]+v[4]
start=time.time()
while time.time()-start<300:
    if any('replay_maploop' in p['cmd'] for p in procs().values()):break
    time.sleep(.2)
time.sleep(4.)
a=procs();ta=total();t0=time.time();g=[]
if gpu:
    sm=subprocess.Popen(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits','-lms','500'],stdout=subprocess.PIPE,text=True)
time.sleep(window)
b=procs();tb=total();t1=time.time()
apps=''
if gpu:
    sm.terminate();g=[list(map(float,l.split(','))) for l in sm.stdout.read().strip().splitlines() if l.count(',')==2]
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout
dt=t1-t0;rows=[]
for pid,p in b.items():
    if pid not in a:continue
    pct=(p['ticks']-a[pid]['ticks'])/HZ/dt*100
    if pct>0.05:rows.append(dict(pid=pid,ppid=p['ppid'],cpu=round(pct,2),cmd=p['cmd'][:160]))
road=[r['pid'] for r in rows if 'road_node' in r['cmd'] and 'ros2' not in r['cmd'].split()[0:1]]
road_main=min(road) if road else None
nv=set()
for r in rows:
    try:
        if any('nvidia' in os.readlink(f'/proc/{r["pid"]}/fd/{fd}') for fd in os.listdir(f'/proc/{r["pid"]}/fd')):nv.add(r['pid'])
    except OSError:pass
def role(r):
    c=r['cmd']
    if r['pid']==road_main:return 'road_node'
    if r['ppid']==road_main or (road_main and r['ppid'] in [x['pid'] for x in rows if x['ppid']==road_main]):return 'road_node.learned' if r['pid'] in nv else 'road_node.compute'
    if 'hud_node' in c:return 'hud'
    if 'replay_maploop' in c:return 'replay'
    if 'realtime_eval' in c or 'outer_listener' in c:return 'eval'
    if c.startswith(('/usr/bin/python3 /home/jaehun/Desktop/vision-road/install/vision_road/lib/vision_road/road_node','python3 -c from multiprocessing')) or 'lane_display' in c:return 'road_node'
    return 'other'
groups={}
for r in rows:r['role']=role(r);groups[r['role']]=round(groups.get(r['role'],0)+r['cpu'],2)
busy=(tb[0]-ta[0])-(tb[1]-ta[1]);system=round(busy/(tb[0]-ta[0])*NCPU*100,1)
res=dict(window_s=round(dt,1),hardware_threads=NCPU,groups=groups,pipeline_threads=round(sum(v for k,v in groups.items() if k!='other' and k!='eval')/100,2),
         system_busy_pct=system,system_threads_used=round(system/100,2),processes=sorted(rows,key=lambda r:-r['cpu'])[:25])
if gpu and g:
    u=[x[0] for x in g];m=[x[1] for x in g]
    res['gpu']=dict(samples=len(g),util_mean=round(sum(u)/len(u),1),util_max=max(u),mem_used_mib_mean=round(sum(m)/len(m)),mem_used_mib_max=max(m),mem_total_mib=g[0][2],apps=apps.strip())
json.dump(res,open(out,'w'),indent=1);print(json.dumps({k:v for k,v in res.items() if k!='processes'}))
