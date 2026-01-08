"""
Train classifier for action recognition using bbox-history + optional pose features.

Expected data layout:
  data/
    walking/
      clip1.mp4
    running/
      clip2.mp4
    jumping/
      clip3.mp4

Usage:
  pip install opencv-python mediapipe lightgbm scikit-learn numpy tqdm
  python train_classifier.py --data_dir data --out_model models/classifier.pkl --yolo models/yolov5s.onnx
"""
import os
import glob
import argparse
import pickle
import math
from tqdm import tqdm
from collections import deque, defaultdict

import numpy as np
import cv2

# optional heavy deps
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
except Exception:
    mp_pose = None

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --- helper: YOLO ONNX via OpenCV DNN ---
def load_yolo_onnx(path, input_size=640):
    if not path or not os.path.isfile(path):
        return None
    net = cv2.dnn.readNetFromONNX(path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return {"net": net, "size": input_size}

def yolov5_forward(yolo, frame, conf_thresh=0.3, iou_thresh=0.45):
    """
    Return list of person boxes (x1,y1,x2,y2,conf) in frame coords.
    Assumes model outputs YOLO-like predictions.
    """
    net = yolo["net"]
    input_size = yolo["size"]
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_size, input_size), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    if isinstance(preds, list):
        preds = preds[0]
    if preds.ndim == 3:
        preds = preds[0]
    boxes = []
    scores = []
    class_ids = []
    for row in preds:
        box_conf = float(row[4])
        if box_conf < 0.01:
            continue
        probs = row[5:]
        class_id = int(np.argmax(probs))
        class_score = float(probs[class_id])
        conf = box_conf * class_score
        if conf < conf_thresh:
            continue
        # COCO person class id 0
        if class_id != 0:
            continue
        cx, cy, bw, bh = row[0], row[1], row[2], row[3]
        x1 = int((cx - bw/2) * w / input_size)
        y1 = int((cy - bh/2) * h / input_size)
        x2 = int((cx + bw/2) * w / input_size)
        y2 = int((cy + bh/2) * h / input_size)
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(float(conf))
        class_ids.append(class_id)
    if len(boxes) == 0:
        return []
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
    out = []
    if len(indices) > 0:
        for i in indices.flatten():
            x,y,wd,hd = boxes[i]
            conf = scores[i]
            out.append((max(0,x), max(0,y), x+wd, y+hd, conf))
    return out

# --- simple IoU tracker for short tracklets ---
def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter + 1e-6)

class SimpleTracker:
    def __init__(self, iou_thresh=0.3, max_lost=5):
        self.next_id = 0
        self.tracks = {}  # id -> bbox,last_seen,lost_count
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost

    def update(self, detections):
        """
        detections: list of boxes (x1,y1,x2,y2,conf)
        returns dict id -> box
        """
        assigned = {}
        used_det = set()
        # match existing tracks
        for tid, info in list(self.tracks.items()):
            box_prev = info["box"]
            best_iou = 0
            best_j = -1
            for j, det in enumerate(detections):
                if j in used_det:
                    continue
                i = iou(box_prev, det[:4])
                if i > best_iou:
                    best_iou = i; best_j = j
            if best_iou >= self.iou_thresh and best_j >= 0:
                self.tracks[tid]["box"] = detections[best_j][:4]
                self.tracks[tid]["lost"] = 0
                self.tracks[tid]["last_seen"] = 0
                assigned[tid] = tuple(self.tracks[tid]["box"])
                used_det.add(best_j)
            else:
                self.tracks[tid]["lost"] += 1
            if self.tracks[tid]["lost"] > self.max_lost:
                del self.tracks[tid]
        # create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j in used_det:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {"box": det[:4], "lost": 0, "last_seen": 0}
            assigned[tid] = tuple(self.tracks[tid]["box"])
        # increment last_seen for all
        for tid in self.tracks:
            self.tracks[tid]["last_seen"] += 1
        return assigned

# --- feature computation helpers ---
def bbox_to_center_size(box):
    x1,y1,x2,y2 = box
    w = max(1.0, x2 - x1); h = max(1.0, y2 - y1)
    cx = x1 + w/2.0; cy = y1 + h/2.0
    return cx, cy, w, h

def compute_window_features(bbox_history, frame_diag, pose_history=None):
    """
    bbox_history: list of boxes (x1,y1,x2,y2) oldest->newest (None allowed)
    pose_history: list of pose landmarks (or None) aligned to bbox_history
    Returns feature vector or None if insufficient data.
    Features:
      - center_displacement_norm
      - vertical_range_norm
      - mean_area_ratio (normalized)
      - area_std_ratio
      - aspect_change
      - detect_ratio
      - optional pose wrist amplitudes (rw_amp, lw_amp)
    """
    boxes = [b for b in bbox_history if b is not None]
    if len(boxes) < 2:
        return None
    centers = []; areas = []; ars = []
    for b in boxes:
        cx, cy, w, h = bbox_to_center_size(b)
        centers.append((cx, cy)); areas.append(w*h); ars.append(w/h if h>0 else 1.0)
    x0,y0 = centers[0]; x1,y1 = centers[-1]
    dx = x1 - x0; dy = y1 - y0
    center_disp = math.hypot(dx, dy) / (frame_diag + 1e-6)
    vert_range = (max([c[1] for c in centers]) - min([c[1] for c in centers])) / (frame_diag + 1e-6)
    area_mean = float(np.mean(areas))
    area_std = float(np.std(areas))
    area_norm = area_mean / (frame_diag*frame_diag + 1e-6)
    area_std_norm = area_std / (frame_diag*frame_diag + 1e-6)
    aspect_change = abs(ars[-1] - ars[0])
    detect_ratio = len(bbox_history) and (len([1 for b in bbox_history if b is not None]) / float(len(bbox_history))) or 0.0

    feat = [center_disp, vert_range, area_norm, area_std_norm, aspect_change, detect_ratio]

    # optional pose wrist amplitude features if pose_history provided
    if pose_history:
        rwx = []
        lwx = []
        for p in pose_history:
            if p is None:
                continue
            try:
                rw = p[mp_pose.PoseLandmark.RIGHT_WRIST.value].x
                lw = p[mp_pose.PoseLandmark.LEFT_WRIST.value].x
                rwx.append(rw); lwx.append(lw)
            except Exception:
                continue
        if len(rwx) > 1:
            feat.append(max(rwx)-min(rwx))
        else:
            feat.append(0.0)
        if len(lwx) > 1:
            feat.append(max(lwx)-min(lwx))
        else:
            feat.append(0.0)
    return feat

# --- video processing: extract windows per video with label ---
def extract_windows_from_video(video_path, window_secs, stride_secs, target_fps, yolo=None, use_pose=False):
    """
    Returns list of feature vectors for this video (implicitly labeled by caller)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sample_step = max(1, int(round(orig_fps / float(target_fps)))) if target_fps else 1

    frames = []
    poses = []
    detections_per_frame = []

    # optionally init pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) if (use_pose and mp_pose) else None

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_step != 0:
            idx += 1
            continue
        frames.append(frame.copy())
        # YOLO detections
        if yolo:
            dets = yolov5_forward(yolo, frame)
            detections_per_frame.append(dets)
        else:
            # fallback: treat whole frame as one detection (poor but allowed)
            h,w = frame.shape[:2]
            detections_per_frame.append([(0,0,w,h,1.0)])
        # pose landmarks if requested
        if pose:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            poses.append(res.pose_landmarks.landmark if res and res.pose_landmarks else None)
        else:
            poses.append(None)
        idx += 1
    cap.release()
    if pose:
        pose.close()

    if len(frames) == 0:
        return []

    h0,w0 = frames[0].shape[:2]
    frame_diag = math.hypot(w0, h0)

    # build simple tracklets across frames using IoU tracker
    tracker = SimpleTracker(iou_thresh=0.3, max_lost=3)
    # store per-frame mapping: track_id -> box
    per_frame_tracks = []
    for frame_idx, dets in enumerate(detections_per_frame):
        # dets are (x1,y1,x2,y2,conf); tracker expects same
        assigned = tracker.update(dets)
        per_frame_tracks.append(assigned)

    # aggregate boxes per track id over time
    track_histories = defaultdict(lambda: {"boxes": [], "poses": []})
    num_frames = len(per_frame_tracks)
    for fidx in range(num_frames):
        mapping = per_frame_tracks[fidx]
        # collect all known track ids
        all_ids = set(list(mapping.keys()) + list(track_histories.keys()))
        for tid in all_ids:
            if tid in mapping:
                box = mapping[tid]
                track_histories[tid]["boxes"].append(tuple(box))
                track_histories[tid]["poses"].append(poses[fidx])
            else:
                # pad with None for missing frame
                # but only if track exists and hasn't been deleted; to keep windows aligned, we add None
                if tid in track_histories:
                    track_histories[tid]["boxes"].append(None)
                    track_histories[tid]["poses"].append(None)

    # sliding windows per tracklet
    window_frames = max(2, int(round(window_secs * target_fps)))
    stride_frames = max(1, int(round(stride_secs * target_fps)))

    features = []
    for tid, hist in track_histories.items():
        boxes_seq = hist["boxes"]
        poses_seq = hist["poses"] if use_pose and mp_pose else None
        L = len(boxes_seq)
        if L < window_frames:
            continue
        for start in range(0, L - window_frames + 1, stride_frames):
            window_boxes = boxes_seq[start:start+window_frames]
            window_poses = poses_seq[start:start+window_frames] if poses_seq is not None else None
            feat = compute_window_features(window_boxes, frame_diag, pose_history=window_poses)
            if feat is not None:
                features.append(feat)
    return features

# --- dataset loader / trainer ---
def build_dataset(data_dir, window_secs=1.5, stride_secs=0.75, target_fps=8, yolo_path=None, use_pose=False):
    labels = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,d))])
    if not labels:
        raise RuntimeError("No label subfolders found in data_dir")
    yolo = load_yolo_onnx(yolo_path, input_size=640) if yolo_path else None
    X = []
    Y = []
    for label in labels:
        folder = os.path.join(data_dir, label)
        videos = []
        for ext in ('mp4','avi','mov','mkv'):
            videos += glob.glob(os.path.join(folder, f'*.{ext}'))
        for v in tqdm(videos, desc=f"Processing {label}"):
            feats = extract_windows_from_video(v, window_secs, stride_secs, target_fps, yolo=yolo, use_pose=use_pose)
            for f in feats:
                X.append(f); Y.append(label)
    return np.array(X), np.array(Y)

def train_and_save(X, Y, out_model_path, use_lgb=True, random_state=42):
    if X.shape[0] == 0:
        raise RuntimeError("No training samples")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state, stratify=Y)
    if use_lgb and HAS_LGB:
        train_data = lgb.Dataset(X_train, label=y_train)
        params = {"objective":"multiclass", "num_class":len(np.unique(Y)), "metric":"multi_logloss", "verbosity":-1}
        model = lgb.train(params, train_data, num_boost_round=200)
    else:
        model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=random_state)
        model.fit(X_train, y_train)
    preds = model.predict(X_test) if not (use_lgb and HAS_LGB) else [np.argmax(p) for p in model.predict(X_test)]
    # if LightGBM used, map numeric preds to labels by training with label encoding; for simplicity skip mapping here
    if not (use_lgb and HAS_LGB):
        print("Eval:")
        print(classification_report(y_test, preds))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, preds))
    os.makedirs(os.path.dirname(out_model_path) or ".", exist_ok=True)
    with open(out_model_path, 'wb') as f:
        pickle.dump({"model": model, "use_lgb": use_lgb and HAS_LGB}, f)
    print(f"Saved model to {out_model_path}")

def main():
    parser = argparse.ArgumentParser(description="Train action classifier from labeled videos")
    parser.add_argument('--data_dir', required=True, help='data directory with subfolders per label')
    parser.add_argument('--out_model', default='models/classifier.pkl', help='where to save classifier')
    parser.add_argument('--window_secs', type=float, default=1.5)
    parser.add_argument('--stride_secs', type=float, default=0.75)
    parser.add_argument('--target_fps', type=int, default=8)
    parser.add_argument('--yolo', default=None, help='path to YOLO ONNX (optional)')
    parser.add_argument('--use_pose', action='store_true', help='extract pose features (requires mediapipe)')
    parser.add_argument('--no_lgb', action='store_true', help='disable LightGBM even if installed')
    args = parser.parse_args()

    print("Building dataset...")
    X, Y = build_dataset(args.data_dir, window_secs=args.window_secs, stride_secs=args.stride_secs,
                         target_fps=args.target_fps, yolo_path=args.yolo, use_pose=args.use_pose)
    print(f"Extracted {len(X)} samples across {len(np.unique(Y)) if len(Y)>0 else 0} labels")
    if len(X) == 0:
        print("No samples extracted; check data and parameters.")
        return
    use_lgb = not args.no_lgb and HAS_LGB
    train_and_save(X, Y, args.out_model, use_lgb=use_lgb)

if __name__ == "__main__":
    main()

