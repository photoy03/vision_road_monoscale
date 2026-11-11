#!/usr/bin/env python3
"""Run a real KITTI ROS replay and save received-topic counts, poses and final lane grid."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_lane_interfaces.msg import MarkingSegmentArray, LaneStyleArray
from vision_road.kitti import KittiSequence
from vision_road.core import Timeline


def sec(stamp): return stamp.sec+stamp.nanosec*1e-9


def pose(msg):
    q=msg.pose.pose.orientation; p=msg.pose.pose.position
    return (sec(msg.header.stamp),p.x,p.y,math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z)))


def compare(reference,estimated):
    timeline=Timeline(.25)
    for t,x,y,yaw in reference: timeline.add(t,x,y,yaw)
    matches=[(p,timeline.at(p[0])) for p in estimated]
    matches=[(p,r) for p,r in matches if r is not None]
    if len(matches)<2: return {'matched_samples':len(matches),'note':'Insufficient matched estimated/reference poses'}
    p0,r0=matches[0]; angle=r0[2]-p0[3]
    c,s=math.cos(angle),math.sin(angle); rotation=np.array([[c,-s],[s,c]])
    errors=[]; paired=[]
    for p,r in matches:
        aligned=rotation@(np.array(p[1:3])-p0[1:3])+r0[:2]
        errors.append(float(np.linalg.norm(aligned-r[:2]))); paired.append([p[0],*aligned,*r[:2],errors[-1]])
    return {'matched_samples':len(matches),'position_rmse_m':float(np.sqrt(np.mean(np.square(errors)))),
            'position_final_error_m':errors[-1], 'position_max_error_m':max(errors),
            'alignment':'First matched position/yaw, SE(2), no scale fitting; compare with KITTI OXTS reference, not lane ground truth',
            'pairs':paired}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--mode',choices=['oxts','monoscale'],default='oxts'); parser.add_argument('--rate',type=float,default=.5)
    args=parser.parse_args()
    if args.rate<=0: parser.error('--rate must be positive')
    data=KittiSequence(args.sequence); output=Path(args.output).resolve(); output.mkdir(parents=True,exist_ok=True)
    frames=[]; globals_=[]; inputs=[]; overlays=[]; raw=[]; estimates=[]; refs=[]; status=[]; grids=[]; styles=[]; style_frames=[]
    rclpy.init(); reader=Node('kitti_validation_recorder')
    def segments(msg):
        t=sec(msg.header.stamp)
        frames.append([t,len(msg.segments),msg.pose_valid,sum(s.confirmed for s in msg.segments),
                       sum(not s.reprojected for s in msg.segments),sum(s.reprojected for s in msg.segments),
                       max([t-sec(s.source_stamp) for s in msg.segments if s.reprojected] or [0.])])
    def on_styles(msg):
        style_frames.append(sec(msg.header.stamp))
        for lane in msg.lanes:
            styles.append([sec(msg.header.stamp),lane.side,('unknown','solid','dashed')[lane.style],lane.confidence,lane.reprojected,
                           sec(lane.source_stamp),lane.paint_fraction,lane.visible_dash_count])
    def last_grid(msg): grids[:]=[msg]
    def status_callback(msg):
        try: status.append(json.loads(msg.data))
        except ValueError: pass
    subscriptions=[reader.create_subscription(LaneStyleArray,'/road/lane_styles',on_styles,200),
                   reader.create_subscription(MarkingSegmentArray,'/parking/cpp/marking_segments',segments,200),
                   reader.create_subscription(MarkingSegmentArray,'/parking/cpp/global_marking_segments_fused',lambda m:globals_.append([sec(m.header.stamp),len(m.segments),sum(s.reprojected for s in m.segments)]),20),
                   reader.create_subscription(Image,'/kitti/image_02/image_rect',lambda m:inputs.append(sec(m.header.stamp)),QoSProfile(depth=100)),
                   reader.create_subscription(Image,'/road/overlay',lambda m:overlays.append(sec(m.header.stamp)),qos_profile_sensor_data),
                   reader.create_subscription(Odometry,'/localization/monoscale_raw',lambda m:raw.append(pose(m)),200),
                   reader.create_subscription(Odometry,'/localization/kinematic_state',lambda m:estimates.append(pose(m)),200),
                   reader.create_subscription(Odometry,'/kitti/reference_odom',lambda m:refs.append(pose(m)),200),
                   reader.create_subscription(OccupancyGrid,'/road/lane_grid',last_grid,2),
                   reader.create_subscription(String,'/parking/cpp/status',status_callback,100)]
    started=time.monotonic()
    with (output/'launch.log').open('w') as log:
        process=subprocess.Popen(['ros2','launch','vision_road','kitti.launch.py',f'sequence:={data.path}',f'odom_source:={args.mode}',
                                  f'rate:={args.rate}','start_delay_sec:=4.0',f'output_dir:={output/"calibration"}'],
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            deadline=started+4+float(data.image_times[-1])/args.rate+12
            while time.monotonic()<deadline:
                rclpy.spin_once(reader,timeout_sec=.03)
                if inputs and inputs[-1]>=data.image_times[-1]-1e-5:
                    if time.monotonic()-started>4+float(data.image_times[-1])/args.rate+3: break
                if process.poll() is not None: break
        finally:
            process.send_signal(signal.SIGINT)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM); process.wait(timeout=5)
            reader.destroy_node(); rclpy.shutdown()
    for name,rows,columns in [('local_frames',frames,['stamp_sec','segments','pose_valid_at_local_publish','confirmed_segments','observed_segments','predicted_segments','max_prediction_age_sec']),
                              ('raw_odom',raw,['stamp_sec','x','y','yaw']),('kinematic_state',estimates,['stamp_sec','x','y','yaw']),
                              ('reference_odom',refs,['stamp_sec','x','y','yaw']),('lane_styles',styles,['stamp_sec','side','style','confidence','reprojected','source_stamp_sec','paint_fraction','visible_dash_count']),('global_frames',globals_,['stamp_sec','segments','reprojected_segments'])]:
        with (output/(name+'.csv')).open('w',newline='') as f:
            writer=csv.writer(f); writer.writerow(columns); writer.writerows(rows)
    metrics=compare(refs,estimates) if args.mode=='monoscale' else {}
    pairs=metrics.pop('pairs',[])
    if pairs:
        with (output/'pose_comparison.csv').open('w',newline='') as f:
            writer=csv.writer(f); writer.writerow(['stamp_sec','aligned_x','aligned_y','reference_x','reference_y','position_error_m']); writer.writerows(pairs)
    if grids:
        grid=grids[-1]; raster=np.array(grid.data,dtype='int8').reshape(grid.info.height,grid.info.width)
        cv2.imwrite(str(output/'lane_grid.png'),np.where(raster[::-1]>=0,255,0).astype('uint8'))
        (output/'lane_grid.json').write_text(json.dumps({'frame':grid.header.frame_id,'stamp_sec':sec(grid.header.stamp),'resolution_m':grid.info.resolution,
            'origin_xy':[grid.info.origin.position.x,grid.info.origin.position.y],'width':grid.info.width,'height':grid.info.height,
            'note':'White = lane observation cells; black = unknown, not free space'},indent=2))
    summary={'sequence':str(data.path),'mode':args.mode,'source_frames':len(data.images),'received_input_images':len(set(inputs)),
             'received_local_frames':len(frames),'frames_with_segments':sum(f[1]>0 for f in frames),'frames_with_observed_segments':sum(f[4]>0 for f in frames),
             'frames_with_predicted_segments':sum(f[5]>0 for f in frames),'max_prediction_age_sec':max([f[6] for f in frames] or [0.]),
             'global_predicted_segments':sum(f[2] for f in globals_),'received_overlays':len(overlays),
             'received_global_snapshots':len(globals_),'last_global_segment_count':globals_[-1][1] if globals_ else 0,
             'raw_odom_samples':len(raw),'kinematic_state_samples':len(estimates),'reference_odom_samples':len(refs),
             'last_detector_status':status[-1] if status else None,'pose_comparison':metrics,
             'received_style_frames':len(style_frames),'lane_style_counts':{key:sum(row[1]+':'+row[2]==key for row in styles) for key in sorted({row[1]+':'+row[2] for row in styles})},
             'note':'Topic execution check and OXTS pose comparison, not lane detection accuracy.'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
    if len(frames)<len(data.images)*.9: raise SystemExit('Incomplete ROS replay; inspect launch.log')


if __name__=='__main__': main()
