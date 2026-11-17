"""Sampling line profile of TemporalDetector.process: another thread records the detector thread's current
line (innermost frame in vision_road code) every ~0.3 ms.  usage: line_sampler.py VIDEO FIRST LAST [TOP]"""
import sys,json,time,threading,collections,cv2
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import maploop_offline_compare as M
from vision_road.core import Timeline
from vision_road.temporal import TemporalDetector
from vision_road.carla import CarlaDetector
video,a,b=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]);top=int(sys.argv[4]) if len(sys.argv)>4 else 40
cache=Path('/media/jaehun/38F7-677B/vision-road-cache')/f'{video}_offline';cv2.setNumThreads(2)
pk=ROOT/'src/vision_road';cfg=M.settings(pk);meta=json.loads((cache/'frames.json').read_text())
valid=cv2.imread(str(cache/'valid.png'),0);valid[M.projector(pk,cfg).my>=1080*M.LAUNCH['image_roi_bottom']]=0
det=TemporalDetector(cfg);det.detector=CarlaDetector(cfg);tl=Timeline(.25);k=0
bevs=[cv2.imread(str(cache/'bev'/f'{i:05d}.png')) for i in range(a,b)]
lines=collections.Counter();funcs=collections.Counter();done=threading.Event();target=[None]
def sample():
    while not done.is_set():
        f=sys._current_frames().get(target[0])
        own=None;fn=None
        while f is not None:
            if 'vision_road/vision_road' in f.f_code.co_filename:
                own=(Path(f.f_code.co_filename).name,f.f_lineno,f.f_code.co_name);break
            f=f.f_back
        if own:lines[own]+=1;funcs[own[0]+':'+own[2]]+=1
        time.sleep(.0003)
def run():
    global k
    target[0]=threading.get_ident()
    for j,i in enumerate(range(a,b)):
        t=meta['frames'][i]['stamp']
        while k<len(meta['poses']) and meta['poses'][k][0]<=t+.05:tl.add(*meta['poses'][k]);k+=1
        det.process(bevs[j],valid,t,tl.at(t))
th=threading.Thread(target=sample,daemon=True);w=threading.Thread(target=run)
w.start();th.start();w.join();done.set()
n=sum(lines.values())
print('samples',n);print('-- by function (innermost vision_road frame)')
for (f),c in funcs.most_common(15):print(f'{100*c/n:5.1f}% {f}')
print('-- by line')
src={}
for (fn,ln,name),c in lines.most_common(top):
    if fn not in src:src[fn]=(ROOT/'src/vision_road/vision_road'/fn).read_text().splitlines()
    print(f'{100*c/n:5.1f}% {fn}:{ln} {name}: {src[fn][ln-1].strip()[:110]}')
