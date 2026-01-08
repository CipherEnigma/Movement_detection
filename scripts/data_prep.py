"""
Split raw videos into short labeled clips, normalize fps/size, filter low-motion clips,
and write a manifest CSV for training.

Example:
  python data_prep.py --raw_dir raw_videos --out_dir data --clip_len 1.5 --stride 0.75 --fps 10 --min_motion 0.02
"""
import os
import argparse
import csv
import math
import uuid
from tqdm import tqdm
import cv2
import numpy as np

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def frame_diff_motion_ratio(frames, thresh=30):
    # frames: list of BGR images
    if len(frames) < 2:
        return 0.0
    count = 0
    total = 0
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for f in frames[1:]:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        d = cv2.absdiff(prev, g)
        _, b = cv2.threshold(d, thresh, 255, cv2.THRESH_BINARY)
        motion = (b > 0).sum() / float(b.size)
        if motion > 0.001:  # small per-frame threshold
            count += 1
        total += 1
        prev = g
    return (count / total) if total>0 else 0.0

def write_clip(frames, out_path, fps, width=None, height=None):
    if width is None or height is None:
        h,w = frames[0].shape[:2]
    else:
        w,h = width, height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w,h))
    for f in frames:
        if (f.shape[1], f.shape[0]) != (w,h):
            f = cv2.resize(f, (w,h))
        writer.write(f)
    writer.release()

def split_video_to_clips(video_path, clip_len_s, stride_s, target_fps, out_folder, label, min_motion, target_size):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(orig_fps / float(target_fps)))) if target_fps and target_fps>0 else 1
    frames = []
    ret = True
    while ret:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        return []
    # sample to target_fps
    sampled = frames[::step]
    total_frames = len(sampled)
    fw = sampled[0].shape[1]; fh = sampled[0].shape[0]
    target_w, target_h = (target_size if target_size else (fw, fh))
    frames_per_clip = max(2, int(round(clip_len_s * target_fps)))
    stride_frames = max(1, int(round(stride_s * target_fps)))
    saved = []
    for start in range(0, total_frames - frames_per_clip + 1, stride_frames):
        window = sampled[start:start+frames_per_clip]
        motion_ratio = frame_diff_motion_ratio(window)
        if motion_ratio < min_motion:
            continue
        uid = uuid.uuid4().hex[:10]
        out_name = f"{label}_{uid}.mp4"
        out_path = os.path.join(out_folder, out_name)
        write_clip(window, out_path, target_fps, width=target_w, height=target_h)
        duration = len(window) / float(target_fps)
        saved.append({"path": out_path, "label": label, "duration": duration, "frames": len(window), "fps": target_fps, "motion_ratio": motion_ratio})
    return saved

def main():
    parser = argparse.ArgumentParser(description="Prepare training clips from raw videos")
    parser.add_argument('--raw_dir', default='raw_videos', help='raw_videos/<label>/*.mp4')
    parser.add_argument('--out_dir', default='data', help='output data/<label>/ clips and manifest')
    parser.add_argument('--clip_len', type=float, default=1.5)
    parser.add_argument('--stride', type=float, default=0.75)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--min_motion', type=float, default=0.02, help='min fraction of frames with motion in window')
    parser.add_argument('--width', type=int, default=320, help='target width (maintains aspect)')
    parser.add_argument('--height', type=int, default=240, help='target height')
    args = parser.parse_args()

    raw_dir = args.raw_dir
    out_dir = args.out_dir
    ensure_dir(out_dir)
    manifest_path = os.path.join(out_dir, 'manifest.csv')
    rows = []
    labels = sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir,d))])
    for label in labels:
        src_folder = os.path.join(raw_dir, label)
        dst_folder = os.path.join(out_dir, label)
        ensure_dir(dst_folder)
        videos = []
        for ext in ('mp4','avi','mov','mkv'):
            videos += [os.path.join(src_folder, f) for f in os.listdir(src_folder) if f.lower().endswith('.'+ext)]
        for v in tqdm(videos, desc=f"Processing {label}"):
            saved = split_video_to_clips(v, args.clip_len, args.stride, args.fps, dst_folder, label, args.min_motion, (args.width, args.height))
            for s in saved:
                rows.append([s['path'], s['label'], s['duration'], s['frames'], s['fps'], s['motion_ratio']])
    # write manifest
    with open(manifest_path, 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['filepath','label','duration_s','frames','fps','motion_ratio'])
        writer.writerows(rows)
    print(f"Prepared {len(rows)} clips. Manifest: {manifest_path}")

if __name__ == '__main__':
    main()
