#!/usr/bin/env python3
"""Plan 23: plan 21's HUD captures, on scenarios23.py, with the fake vehicle state (tools/fake_vehicle_state.py)
running: the HUD draws the current-path corridor beside the guidance path, the guidance node takes its
speed from /hero/vehicle_state.  hud_scenarios23.py [--no-vehicle-state] [script ...]

HUD captures and short recordings of the fake-motion scripts (plan 21).
For each script: the lane guidance node (hero_coach, this workspace) and the HUD copy (lane-only mode,
recording 10 fps) run while scenario_publisher.py plays the script at 30 Hz.  HUD frames are matched to
script time by their file time, labelled (script, time, lateral move, guidance state and target), and
written as ../hud/<script>.mp4 and capture JPEGs at the script's key moments.
usage: hud_scenarios.py [script ...]   (default: wobble yellow dashed change)"""
import os,sys,json,time,subprocess,signal,shutil,math
from pathlib import Path
import cv2,numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4];OUT=HERE.parent/'hud';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'results/classical_five_videos/town05_30hz/lane_guidance_20261118/scripts'))
import scenarios23 as S  # noqa: E402
NOVS='--no-vehicle-state' in sys.argv
if NOVS:sys.argv.remove('--no-vehicle-state');OUT=HERE.parent/'hud_no_vehicle_state';OUT.mkdir(exist_ok=True)
KEY=dict(wobble=[3.,5.,7.],yellow=[1.,3.,4.,6.],dashed=[1.,3.,4.,7.5],change=[1.,3.,4.,4.5,7.,12.])
SRC='source /opt/ros/humble/setup.bash && source "$0/install/setup.bash" && exec "$@"'


def start(cmd,log,env=None):
    return subprocess.Popen(['bash','-c',SRC,str(ROOT)]+cmd,stdout=open(log,'w'),stderr=subprocess.STDOUT,env=env,start_new_session=True)


def stop(p):
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGINT)
        try:p.wait(8)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)


for name in sys.argv[1:] or ['wobble','yellow','dashed','change']:
    work=OUT/name;shutil.rmtree(work,ignore_errors=True);(work/'frames').mkdir(parents=True)
    length=S.SCRIPTS[name][1]
    guide=start(['ros2','run','hero_coach','lane_guidance'],work/'guidance.log')
    fake=None if NOVS else start(['python3',str(ROOT/'tools/fake_vehicle_state.py')],work/'fake_vehicle_state.log')
    listen=start(['python3',str(ROOT/'results/classical_five_videos/town05_30hz/lane_guidance_20261118/scripts/guidance_listener.py'),str(work/'guidance.json'),str(length+9)],work/'listener.log')
    env=os.environ.copy();env.pop('QT_QPA_PLATFORM_PLUGIN_PATH',None);env.pop('QT_QPA_FONTDIR',None)
    env.update(HERO_HUD_RECORD=str(work/'frames'),HERO_HUD_RECORD_FPS='10',HERO_HUD_RECORD_SECONDS=str(int(length)+5),HERO_HUD_SCREENSHOT_DELAY_MS='2000')
    hud=start([str(ROOT/'install/hero_hud/lib/hero_hud/hud_node'),'--ros-args','--params-file',str(ROOT/'src/hero_hud/config/hud.yaml'),
               '-p','display.mirror_horizontal:=false','-p','road.lane_only:=true'],work/'hud.log',env)
    time.sleep(3.)
    pub=start(['python3',str(HERE/'scenario_publisher23.py'),name,str(work/'publish.json')],work/'publisher.log');pub.wait()
    time.sleep(1.5)
    for p in (hud,guide,fake):
        if p is not None:stop(p)
    listen.wait(20)
    pubs=json.load(open(work/'publish.json'));walls=np.array([p['wall'] for p in pubs])
    guidance=json.load(open(work/'guidance.json'))['rows']
    gstamp=np.array([g['stamp'] for g in guidance]) if guidance else np.zeros(1)
    t0=S.frames(name)[0][0]
    files=sorted((work/'frames').glob('frame_*.png'));video=None;labelled=[]
    for f in files:
        w=f.stat().st_mtime;i=int(np.argmin(np.abs(walls-w)))
        if abs(walls[i]-w)>.2:continue
        p=pubs[i];g=guidance[int(np.argmin(np.abs(gstamp-(t0+p['t']))))] if guidance else {}
        img=cv2.imread(str(f));h=img.shape[0]
        band=np.zeros((70,img.shape[1],3),np.uint8)
        text=f"{name}  t={p['t']:5.2f}s  moved {p['delta']:+.2f} m  turned {math.degrees(p['psi']):+.1f} deg  guidance: {g.get('state','-')}  target {g.get('target','-')}  offset {g.get('offset_m',0):+.2f} m  speed from {g.get('speed_source','-')}"
        cv2.putText(band,text,(12,44),cv2.FONT_HERSHEY_SIMPLEX,.9,(255,255,255),2)
        frame=np.vstack([band,img]);labelled.append((p['t'],frame))
        if video is None:video=cv2.VideoWriter(str(OUT/f'{name}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(frame.shape[1],frame.shape[0]))
        video.write(frame)
    if video is not None:video.release()
    for k in KEY.get(name,[]):
        if labelled:
            t,frame=min(labelled,key=lambda a:abs(a[0]-k));cv2.imwrite(str(OUT/f'{name}_t{k:04.1f}.jpg'),cv2.resize(frame,None,fx=.6,fy=.6),[cv2.IMWRITE_JPEG_QUALITY,88])
    print(name,'HUD frames',len(files),'labelled',len(labelled),'guidance msgs',len(guidance),flush=True)
