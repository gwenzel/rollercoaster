import os
import cv2
import argparse
import numpy as np
import pandas as pd
from typing import Tuple

# Simple, modular pipeline: frame extraction -> signals -> path integration -> outputs

def extract_frames(video_path: str, out_dir: str, fps: int = None) -> Tuple[list, list]:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = fps or int(video_fps) or 30
    frame_interval = max(int(video_fps // target_fps), 1) if video_fps > 0 else 1

    frames = []
    times = []
    idx = 0
    t = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frames.append(frame)
            times.append(t)
        idx += 1
        t += 1.0 / (video_fps or target_fps)
    cap.release()
    return frames, times


def estimate_horizon_pitch_roll(frames: list) -> Tuple[np.ndarray, np.ndarray]:
    # Placeholder: robust horizon estimation can use semantic segmentation + RANSAC
    # Here we return zeros (no pitch/roll) as a basic start
    n = len(frames)
    return np.zeros(n, dtype=float), np.zeros(n, dtype=float)


def estimate_yaw_and_speed(frames: list) -> Tuple[np.ndarray, np.ndarray]:
    # Estimate yaw via optical flow bias; speed via expansion magnitude
    prev_gray = None
    yaw = []
    speed = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray
            yaw.append(0.0)
            speed.append(0.0)
            continue
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fx = flow[...,0]
        fy = flow[...,1]
        # central region
        h, w = gray.shape
        cx0, cx1 = int(w*0.3), int(w*0.7)
        cy0, cy1 = int(h*0.3), int(h*0.7)
        fx_c = fx[cy0:cy1, cx0:cx1]
        fy_c = fy[cy0:cy1, cx0:cx1]
        # Yaw proxy: left-right bias in fx
        yaw_val = float(np.clip(np.mean(fx_c), -1.0, 1.0))
        # Speed proxy: magnitude of flow
        mag = np.sqrt(fx_c**2 + fy_c**2)
        speed_val = float(np.clip(np.mean(mag), 0.0, 5.0))
        yaw.append(yaw_val)
        speed.append(speed_val)
        prev_gray = gray
    return np.array(yaw), np.array(speed)


def integrate_path(times: list, yaw: np.ndarray, pitch: np.ndarray, speed: np.ndarray, scale_mps: float = 10.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Convert proxy speed to m/s via a rough scale
    v = speed * scale_mps
    dt = np.diff(times, prepend=times[0])
    n = len(times)
    x = np.zeros(n, dtype=float)
    y = np.zeros(n, dtype=float)
    z = np.zeros(n, dtype=float)
    heading = 0.0
    for i in range(1, n):
        # integrate yaw
        heading += yaw[i] * 0.05  # small gain
        x[i] = x[i-1] + v[i] * dt[i] * np.cos(heading)
        y[i] = y[i-1] + v[i] * dt[i] * np.sin(heading)
        # integrate vertical from pitch
        z[i] = z[i-1] + v[i] * dt[i] * np.sin(pitch[i] * 0.05)
    return x, y, z


def main():
    parser = argparse.ArgumentParser(description='Infer approximate coaster geometry from POV video')
    parser.add_argument('--video', required=True, help='Path to POV video file')
    parser.add_argument('--out', required=True, help='Output folder for results')
    parser.add_argument('--fps', type=int, default=None, help='Target FPS for frame sampling')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print('Extracting frames...')
    frames, times = extract_frames(args.video, os.path.join(args.out, 'frames'), fps=args.fps)
    print(f'Total frames: {len(frames)}')

    print('Estimating visual signals...')
    pitch, roll = estimate_horizon_pitch_roll(frames)
    yaw, speed = estimate_yaw_and_speed(frames)

    print('Integrating path...')
    x, y, z = integrate_path(times, yaw, pitch, speed)

    path_df = pd.DataFrame({'Time': times, 'x': x, 'y': y, 'z': z})
    path_csv = os.path.join(args.out, 'track_path.csv')
    path_df.to_csv(path_csv, index=False)
    print(f'Saved path to {path_csv}')

    print('Done. Consider smoothing and computing accelerations via utils/acceleration.')

if __name__ == '__main__':
    main()
