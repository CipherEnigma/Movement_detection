import cv2
import numpy as np
import os

def _load_yolo_net(model_path):
    if not model_path or not os.path.isfile(model_path):
        return None
    try:
        net = cv2.dnn.readNetFromONNX(model_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net
    except Exception:
        return None

def _yolo_detect_persons(net, frame, input_size=640, conf_thresh=0.4, iou_thresh=0.45):
    """
    Returns list of person boxes in (x1,y1,x2,y2,conf) in original frame coordinates.
    Assumes COCO class 0 is 'person'.
    """
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_size, input_size), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    # normalize preds shape
    if isinstance(preds, list):
        preds = preds[0]
    if preds.ndim == 3:
        preds = preds[0]
    boxes = []
    confidences = []
    class_ids = []
    for row in preds:
        box_conf = float(row[4])
        if box_conf < 0.01:
            continue
        scores = row[5:]
        class_id = int(np.argmax(scores))
        class_score = float(scores[class_id])
        conf = box_conf * class_score
        if conf < conf_thresh:
            continue
        # only person class (COCO id 0)
        if class_id != 0:
            continue
        cx, cy, bw, bh = row[0], row[1], row[2], row[3]
        # coords are relative to input_size
        x1 = int((cx - bw/2) * w / input_size)
        y1 = int((cy - bh/2) * h / input_size)
        x2 = int((cx + bw/2) * w / input_size)
        y2 = int((cy + bh/2) * h / input_size)
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        confidences.append(float(conf))
        class_ids.append(class_id)
    # apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, iou_thresh)
    persons = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            conf = confidences[i]
            persons.append((max(0,x), max(0,y), min(w, x+bw), min(h, y+bh), conf))
    return persons

def _rects_overlap(r1, r2):
    # r: (x1,y1,x2,y2)
    ax1, ay1, ax2, ay2 = r1
    bx1, by1, bx2, by2 = r2
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    return (inter_x2 > inter_x1) and (inter_y2 > inter_y1)

def detect_motion(video_source=0, output_path=None, max_frames=None, min_area=700,
                  yolo_model_path=None, yolo_interval=5, yolo_input_size=640):
    """
    Headless motion detector with optional YOLO person filtering.

    - video_source: webcam index (int) or file path (str)
    - output_path: if provided, writes annotated mp4
    - yolo_model_path: path to ONNX YOLO model (e.g., models/yolov5s.onnx). If missing, fallback to previous behavior.
    - yolo_interval: run YOLO every N frames (to save CPU)
    - returns: summary dict
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {video_source}")

    # read first two frames
    ret, frame1 = cap.read()
    ret2, frame2 = cap.read()
    if not ret or not ret2:
        cap.release()
        raise IOError("Unable to read initial frames from source")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    height, width = frame1.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # attempt to load YOLO net
    yolo_net = _load_yolo_net(yolo_model_path) if yolo_model_path else None
    using_yolo = yolo_net is not None

    processed = 0
    motion_count = 0
    yolo_boxes = []  # current person boxes (x1,y1,x2,y2,conf)
    frame_idx = 0

    while cap.isOpened():
        # compute frame diff
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        frame_idx += 1
        # run YOLO periodically
        if using_yolo and (frame_idx % max(1, yolo_interval) == 0):
            try:
                yolo_boxes = _yolo_detect_persons(yolo_net, frame1, input_size=yolo_input_size)
            except Exception:
                yolo_boxes = []

        # draw YOLO person boxes
        for (x1,y1,x2,y2,conf) in yolo_boxes:
            cv2.rectangle(frame1, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(frame1, f"person:{conf:.2f}", (x1, max(15,y1+15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        # draw rectangles for contours above area threshold and that overlap a person box (if YOLO used)
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            rect = (x, y, x+w, y+h)
            draw_it = True
            if using_yolo and len(yolo_boxes) > 0:
                # require overlap with at least one person bbox
                overlaps = False
                for b in yolo_boxes:
                    box = (b[0], b[1], b[2], b[3])
                    if _rects_overlap(rect, box):
                        overlaps = True
                        break
                draw_it = overlaps
            if draw_it:
                cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
                motion_count += 1

        # write annotated frame if requested
        if writer:
            writer.write(frame1)

        processed += 1
        # advance frames
        frame1 = frame2
        ret, frame2 = cap.read()
        if not ret:
            break
        if max_frames and processed >= max_frames:
            break

    cap.release()
    if writer:
        writer.release()

    summary = {
        "processed_frames": processed,
        "motion_rects_drawn": motion_count,
        "yolo_used": using_yolo,
        "yolo_model_path": yolo_model_path if using_yolo else None
    }
    return summary
