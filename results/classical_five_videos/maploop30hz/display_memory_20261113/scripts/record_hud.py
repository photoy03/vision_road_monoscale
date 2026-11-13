#!/usr/bin/env python3
"""Record live maploop road overlay and the subscribed HERO HUD in one video."""
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

ROAD = Path('/home/jaehun/Desktop/vision-road')
HUD = Path('/home/jaehun/Desktop/hero-hud-main')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-frame', type=int, default=600)
    parser.add_argument('--frames', type=int, default=360)
    parser.add_argument('--before', action='store_true', help='display filter with code defaults (3 m, no memory), HUD dashes from 3 m')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.output / 'hud_frames'
    raw.mkdir(exist_ok=True)
    bridge = CvBridge()
    overlays = []
    rclpy.init()
    node = Node('road_hud_combined_recorder')

    def on_overlay(msg):
        image = bridge.imgmsg_to_cv2(msg, 'bgr8')
        path = args.output / f'overlay_{len(overlays):04d}.jpg'
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        overlays.append((time.time_ns(), path, msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9))

    sub = node.create_subscription(Image, '/road/overlay', on_overlay, qos_profile_sensor_data)
    processes = []
    try:
        logs = [open(args.output / name, 'w') for name in ('road.log', 'hud.log', 'replay.log')]
        try:
            road = subprocess.Popen([str(ROAD / 'run_maploop.sh')] + (['display_filter:=false'] if args.before else []), stdout=logs[0], stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(road)
            if args.before:
                processes.append(subprocess.Popen(['ros2', 'run', 'vision_road', 'lane_display_filter', '--ros-args', '-p', 'use_sim_time:=true'],
                                                  stdout=logs[0], stderr=subprocess.STDOUT, start_new_session=True))
            deadline = time.monotonic() + 20
            while node.count_publishers('/road/overlay') == 0 and time.monotonic() < deadline:
                if road.poll() is not None: raise RuntimeError('Road node exited')
                rclpy.spin_once(node, timeout_sec=.05)
            if not node.count_publishers('/road/overlay'): raise RuntimeError('No road overlay publisher')

            env = os.environ.copy()
            # Importing cv2 can point Qt at OpenCV's incompatible xcb plugin.
            env.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
            env.pop('QT_QPA_FONTDIR', None)
            env.update(HERO_HUD_RECORD=str(raw), HERO_HUD_RECORD_FPS='10',
                       HERO_HUD_RECORD_SECONDS=str(max(16, args.frames // 30 + 4)),
                       HERO_HUD_SCREENSHOT_DELAY_MS='2500')
            hud = subprocess.Popen([str(HUD / 'install/hero_hud/lib/hero_hud/hud_node'),
                                    '--ros-args', '--params-file', str(HUD / 'config/hud.yaml'),
                                    '-p', 'display.mirror_horizontal:=false',
                                    '-p', 'road.lane_only:=true'] + (['-p', 'road.dash_near_m:=3.0'] if args.before else []),
                                   env=env, stdout=logs[1], stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(hud)
            end = time.monotonic() + 1.8
            while time.monotonic() < end: rclpy.spin_once(node, timeout_sec=.05)
            if hud.poll() is not None: raise RuntimeError('HUD exited before replay')
            hud_subscribers = node.count_subscribers('/road/lane_markers')
            replay = subprocess.Popen([str(ROAD / 'replay_maploop.sh'), '--start-frame', str(args.start_frame),
                                       '--max-frames', str(args.frames), '--rate', '1',
                                       '--timing-report', str(args.output / 'source_timing.json')],
                                      stdout=logs[2], stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(replay)
            deadline = time.monotonic() + args.frames / 30 + 35
            while replay.poll() is None and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=.02)
            if replay.poll() != 0: raise RuntimeError('Replay failed or timed out')
            end = time.monotonic() + 1
            while time.monotonic() < end: rclpy.spin_once(node, timeout_sec=.02)
            hud.wait(timeout=15)
            if hud.returncode != 0: raise RuntimeError('HUD recording failed')
        finally:
            for stream in logs: stream.close()

        source = json.loads((args.output / 'source_timing.json').read_text())['records']
        hud_images = sorted(raw.glob('frame_*.png'))
        if len(overlays) < 20 or len(hud_images) < 80:
            raise RuntimeError(f'Not enough frames: overlay={len(overlays)}, HUD={len(hud_images)}')
        hud_stamps = np.array([p.stat().st_mtime_ns for p in hud_images], np.int64)
        overlay_stamps = np.array([item[0] for item in overlays], np.int64)
        first = max(int(hud_stamps[0]), int(overlay_stamps[0]))
        last = min(int(hud_stamps[-1]), int(overlay_stamps[-1]))
        fps = 10
        total = int((last-first) / 1e9 * fps)
        if total < 80: raise RuntimeError(f'Overlap only {total/fps:.1f}s')
        target = args.output / 'road_and_hud.mp4'
        ffmpeg = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                                   '-s', '1600x540', '-r', str(fps), '-i', 'pipe:0', '-an',
                                   '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                                   '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)],
                                  stdin=subprocess.PIPE)
        previous_overlay = previous_hud = None
        for index in range(total):
            now = first + index * 100_000_000
            camera_index = max(0, int(np.searchsorted(overlay_stamps, now, side='right')-1))
            hud_index = max(0, int(np.searchsorted(hud_stamps, now, side='right')-1))
            if previous_overlay is None or previous_overlay[0] != camera_index:
                previous_overlay = (camera_index, cv2.imread(str(overlays[camera_index][1])))
            if previous_hud is None or previous_hud[0] != hud_index:
                previous_hud = (hud_index, cv2.imread(str(hud_images[hud_index])))
            camera = previous_overlay[1]
            hud_frame = cv2.resize(previous_hud[1], (640, 400), interpolation=cv2.INTER_AREA)
            frame = np.zeros((540, 1600, 3), np.uint8)
            frame[:, :960] = camera
            frame[70:470, 960:] = hud_frame
            cv2.putText(frame, 'CARLA /road/overlay', (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .75, (255,255,255), 2)
            cv2.putText(frame, 'HERO HUD /road/lane_markers + boundaries', (975, 55), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 2)
            ffmpeg.stdin.write(frame.tobytes())
            if index == total//2: cv2.imwrite(str(args.output / 'preview.jpg'), frame)
        ffmpeg.stdin.close()
        if ffmpeg.wait() != 0: raise RuntimeError('Video encoding failed')
        report = dict(source_frames=len(source), overlay_frames=len(overlays), hud_frames=len(hud_images),
                      hud_subscribers_at_start=hud_subscribers, synchronized_seconds=total/fps,
                      video=str(target), note='Two live ROS outputs from the same replay, aligned by capture wall time. HUD is a separate ground-coordinate display; CARLA recorded reference pose is used.')
        (args.output / 'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        shutil.rmtree(raw)
        for image in args.output.glob('overlay_*.jpg'): image.unlink()
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try: process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__': main()
