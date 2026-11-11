#!/usr/bin/env python3
"""Start a coordinate-only RViz view and one bag replay; close RViz to stop all."""
import argparse,os,signal,subprocess,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]


def stop(process):
    if process is None or process.poll() is not None:return
    process.send_signal(signal.SIGINT)
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=3)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rate',type=float,default=1.);ap.add_argument('--domain',type=int,default=186);a=ap.parse_args()
    if not 0<a.rate<=4 or not 0<=a.domain<=232:raise ValueError('Invalid replay settings')
    env=os.environ.copy();env['ROS_DOMAIN_ID']=str(a.domain);env['ROS_LOCALHOST_ONLY']='1';env.setdefault('QT_QPA_PLATFORM','xcb')
    folder=ROOT/'results'/('coordinate_view_'+datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S'));folder.mkdir(parents=True)
    env['ROS_LOG_DIR']=str(folder/'roslogs')
    road=rviz=replay=None
    try:
        with (folder/'node.log').open('w') as node_log,(folder/'rviz.log').open('w') as rviz_log,(folder/'replay.log').open('w') as replay_log:
            rviz=subprocess.Popen(['rviz2','-d',str(ROOT/'src/vision_road/config/maploop_coordinates.rviz'),'--ros-args','-p','use_sim_time:=true'],env=env,stdout=rviz_log,stderr=rviz_log,start_new_session=True)
            road=subprocess.Popen([str(ROOT/'run_maploop.sh')],env=env,stdout=node_log,stderr=node_log,start_new_session=True)
            replay=subprocess.Popen([str(ROOT/'replay_maploop.sh'),'--rate',str(a.rate)],env=env,stdout=replay_log,stderr=replay_log,start_new_session=True)
            print('차량 기준 좌표계 뷰를 실행했습니다. RViz 창을 닫거나 Ctrl+C를 누르면 함께 종료합니다.',flush=True)
            print('로그: '+str(folder),flush=True)
            finished=False
            while rviz.poll() is None:
                if road.poll() is not None:raise RuntimeError('차선 노드가 종료됐습니다. '+str(folder/'node.log'))
                if replay.poll() is not None and not finished:
                    if replay.returncode!=0:raise RuntimeError('재생에 실패했습니다. '+str(folder/'replay.log'))
                    finished=True;print('한 바퀴 재생을 마쳤습니다. 마지막 좌표 화면에서 확대·이동할 수 있습니다.',flush=True)
                time.sleep(.2)
            if rviz.returncode not in (0,-signal.SIGINT):raise RuntimeError('RViz가 종료됐습니다. '+str(folder/'rviz.log'))
    except KeyboardInterrupt:pass
    finally:
        for process in [replay,road,rviz]:stop(process)
if __name__=='__main__':main()
