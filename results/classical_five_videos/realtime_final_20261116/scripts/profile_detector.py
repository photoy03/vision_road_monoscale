"""cProfile of TemporalDetector.process over cached BEV frames (carla.yaml).  usage: profile_detector.py VIDEO FIRST LAST [N]"""
import sys,json,cProfile,pstats,cv2
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'src/vision_road'));sys.path.insert(0,str(ROOT/'tools'))
import maploop_offline_compare as M
from vision_road.core import Timeline
from vision_road.temporal import TemporalDetector
from vision_road.carla import CarlaDetector
video,a,b=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]);n=int(sys.argv[4]) if len(sys.argv)>4 else 30
cache=Path('/media/jaehun/38F7-677B/vision-road-cache')/f'{video}_offline';cv2.setNumThreads(2)
pk=ROOT/'src/vision_road';cfg=M.settings(pk);meta=json.loads((cache/'frames.json').read_text())
valid=cv2.imread(str(cache/'valid.png'),0);valid[M.projector(pk,cfg).my>=1080*M.LAUNCH['image_roi_bottom']]=0
det=TemporalDetector(cfg);det.detector=CarlaDetector(cfg);tl=Timeline(.25);k=0
bevs=[cv2.imread(str(cache/'bev'/f'{i:05d}.png')) for i in range(a,b)]
def run():
    global k
    for j,i in enumerate(range(a,b)):
        t=meta['frames'][i]['stamp']
        while k<len(meta['poses']) and meta['poses'][k][0]<=t+.05:tl.add(*meta['poses'][k]);k+=1
        det.process(bevs[j],valid,t,tl.at(t))
pr=cProfile.Profile();pr.enable();run();pr.disable()
st=pstats.Stats(pr);st.sort_stats('cumulative').print_stats(n)
st.sort_stats('tottime').print_stats(n)
