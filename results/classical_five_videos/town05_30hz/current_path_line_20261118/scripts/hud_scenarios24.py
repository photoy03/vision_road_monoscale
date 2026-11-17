#!/usr/bin/env python3
"""Plan 24: HUD captures of the current-path drawing styles (lane-only mode, fake vehicle state).
Same run as plan 23 hud_scenarios23.py (script publisher, lane guidance node, fake vehicle state, HUD copy
recording 10 fps), once per variant: road.current_path.style (fill/edges/centre/dots) and length_m.
For each variant and script: the lit share of the HUD frame (pixels with any channel > 16, mean over the
recorded frames), capture JPEGs at the key moments, an mp4 (not committed); the PNG frames are deleted.
usage: hud_scenarios24.py [variant ...]   variants: see VARIANTS.  Writes ../hud/<variant>/ and ../metrics/lit_<variant>.json"""
import os,sys,json,time,subprocess,signal,shutil,math
from pathlib import Path
import cv2,numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4]
P23=ROOT/'results/classical_five_videos/town05_30hz/vehicle_state_20261118/scripts'
sys.path.insert(0,str(P23));import scenarios23 as S  # noqa: E402
VARIANTS={'fill':('fill',0.),'edges':('edges',0.),'centre':('centre',0.),'dots':('dots',0.),
          'edges_6m':('edges',6.),'centre_6m':('centre',6.),'centre_10m':('centre',10.),'none':('none',0.),'dots_8m':('dots',8.),'dots_6m':('dots',6.)}
KEY=dict(wobble=[3.,5.,7.],yellow=[1.,3.,4.,6.],dashed=[1.,3.,4.,7.5],change=[1.,3.,4.,4.5,7.,12.])
SRC='source /opt/ros/humble/setup.bash && source "$0/install/setup.bash" && exec "$@"'
def start(cmd,log,env=None):
    return subprocess.Popen(['bash','-c',SRC,str(ROOT)]+cmd,stdout=open(log,'w'),stderr=subprocess.STDOUT,env=env,start_new_session=True)
def stop(p):
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGINT)
        try:p.wait(8)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
def lit(img):return float(np.mean(img.max(axis=2)>16))
for var in sys.argv[1:] or list(VARIANTS):
    style,length_m=VARIANTS[var];OUT=HERE.parent/'hud'/var;OUT.mkdir(parents=True,exist_ok=True);lits={}
    for name in ['wobble','yellow','dashed','change']:
        work=OUT/name;shutil.rmtree(work,ignore_errors=True);(work/'frames').mkdir(parents=True)
        length=S.SCRIPTS[name][1]
        guide=start(['ros2','run','hero_coach','lane_guidance'],work/'guidance.log')
        fake=start(['python3',str(ROOT/'tools/fake_vehicle_state.py')],work/'fake_vehicle_state.log')
        listen=start(['python3',str(ROOT/'results/classical_five_videos/town05_30hz/lane_guidance_20261118/scripts/guidance_listener.py'),str(work/'guidance.json'),str(length+9)],work/'listener.log')
        env=os.environ.copy();env.pop('QT_QPA_PLATFORM_PLUGIN_PATH',None);env.pop('QT_QPA_FONTDIR',None)
        env.update(HERO_HUD_RECORD=str(work/'frames'),HERO_HUD_RECORD_FPS='10',HERO_HUD_RECORD_SECONDS=str(int(length)+5),HERO_HUD_SCREENSHOT_DELAY_MS='2000')
        hud=start([str(ROOT/'install/hero_hud/lib/hero_hud/hud_node'),'--ros-args','--params-file',str(ROOT/'src/hero_hud/config/hud.yaml'),
                   '-p','display.mirror_horizontal:=false','-p','road.lane_only:=true','-p',f'road.current_path.style:={style}','-p',f'road.current_path.length_m:={length_m}'],work/'hud.log',env)
        time.sleep(3.)
        pub=start(['python3',str(P23/'scenario_publisher23.py'),name,str(work/'publish.json')],work/'publisher.log');pub.wait()
        time.sleep(1.5)
        for p in (hud,guide,fake):stop(p)
        listen.wait(20)
        pubs=json.load(open(work/'publish.json'));walls=np.array([p['wall'] for p in pubs])
        guidance=json.load(open(work/'guidance.json'))['rows']
        gstamp=np.array([g['stamp'] for g in guidance]) if guidance else np.zeros(1)
        t0=S.frames(name)[0][0]
        files=sorted((work/'frames').glob('frame_*.png'));video=None;labelled=[];ls=[]
        for f in files:
            w=f.stat().st_mtime;i=int(np.argmin(np.abs(walls-w)))
            if abs(walls[i]-w)>.2:continue
            p=pubs[i];g=guidance[int(np.argmin(np.abs(gstamp-(t0+p['t']))))] if guidance else {}
            img=cv2.imread(str(f));ls.append(lit(img))
            band=np.zeros((70,img.shape[1],3),np.uint8)
            text=f"{var}  {name}  t={p['t']:5.2f}s  moved {p['delta']:+.2f} m  turned {math.degrees(p['psi']):+.1f} deg  guidance: {g.get('state','-')}"
            cv2.putText(band,text,(12,44),cv2.FONT_HERSHEY_SIMPLEX,.9,(255,255,255),2)
            frame=np.vstack([band,img]);labelled.append((p['t'],frame))
            if video is None:video=cv2.VideoWriter(str(OUT/f'{name}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(frame.shape[1],frame.shape[0]))
            video.write(frame)
        if video is not None:video.release()
        for k in KEY[name]:
            if labelled:
                t,frame=min(labelled,key=lambda a:abs(a[0]-k));cv2.imwrite(str(OUT/f'{name}_t{k:04.1f}.jpg'),cv2.resize(frame,None,fx=.6,fy=.6),[cv2.IMWRITE_JPEG_QUALITY,88])
        lits[name]=dict(frames=len(ls),lit_mean=round(float(np.mean(ls)),4) if ls else None,lit_p95=round(float(np.percentile(ls,95)),4) if ls else None,
                        states={s:sum(1 for g in guidance if g.get('state')==s) for s in ('none','keep','return','change','forbidden')})
        shutil.rmtree(work/'frames')
        print(var,name,lits[name],flush=True)
    json.dump(dict(style=style,length_m=length_m,scripts=lits),open(HERE.parent/'metrics'/f'lit_{var}.json','w'),indent=1)
