"""
app.py
Flask backend that runs YOLOv7 object detection (filtered to 'person' and
'car', plus 'motorcycle'/'bus'/'truck' since they're common on roads) on
either a live webcam feed or an uploaded video file, and streams the
annotated frames back as MJPEG so the React frontend can just drop them
into an <img> tag.

It also does simple motion filtering: a background-subtractor is used to
tag each YOLO detection as "moving" or "static" (nice extra talking point
for a project demo / viva -- "how do you know it's actually moving and
not a parked car?").

Run with:  python app.py
Requires:  pip install -r requirements.txt
"""

import os
import time
import threading

import cv2
import torch
import numpy as np
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# COCO class ids we care about: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
TARGET_CLASSES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CONF_THRESHOLD = 0.45

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)
CORS(app)  # allow the React dev server (localhost:3000) to hit this API

# --------------------------------------------------------------------------- #
# Load YOLOv7 (pretrained weights auto-download on first run)
# --------------------------------------------------------------------------- #

print(f"[app.py] Loading YOLOv7 on {DEVICE} ... (first run downloads yolov7.pt)")
model = torch.hub.load(
    "WongKinYiu/yolov7",
    "custom",
    "yolov7.pt",
    trust_repo=True,
)
model.to(DEVICE)
model.conf = CONF_THRESHOLD
print("[app.py] Model loaded.")

# --------------------------------------------------------------------------- #
# Shared state for the webcam stream (so multiple clients can hit /video_feed
# without opening the camera multiple times)
# --------------------------------------------------------------------------- #

class CameraStream:
    def __init__(self, source=0):
        self.source = source
        self.cap = None
        self.lock = threading.Lock()
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=False
        )
        self.running = False

    def start(self):
        with self.lock:
            if not self.running:
                self.cap = cv2.VideoCapture(self.source)
                self.running = True

    def stop(self):
        with self.lock:
            self.running = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    def frames(self):
        self.start()
        while self.running:
            success, frame = self.cap.read()
            if not success:
                break
            annotated = detect_and_annotate(frame, self.bg_subtractor)
            ok, buffer = cv2.imencode(".jpg", annotated)
            if not ok:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
        self.stop()


webcam_stream = CameraStream(source=0)


# --------------------------------------------------------------------------- #
# Core detection + annotation logic (shared by webcam + uploaded video)
# --------------------------------------------------------------------------- #

def detect_and_annotate(frame, bg_subtractor):
    """
    Runs YOLOv7 on a single BGR frame, keeps only person/vehicle classes,
    tags each detection as 'moving' or 'static' using a motion mask, and
    draws labeled boxes. Returns the annotated frame.
    """
    # 1. Motion mask (helps decide moving vs static for each detection)
    fg_mask = bg_subtractor.apply(frame)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # 2. YOLOv7 inference (expects RGB)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb, size=640)
    detections = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, conf, cls

    for x1, y1, x2, y2, conf, cls_id in detections:
        cls_id = int(cls_id)
        if cls_id not in TARGET_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        label_name = TARGET_CLASSES[cls_id]

        # Is this box "moving"? Check overlap with the motion mask.
        box_mask = fg_mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
        moving = box_mask.size > 0 and (np.mean(box_mask) > 15)  # % of white px

        color = (0, 200, 0) if moving else (0, 165, 255)  # green=moving, orange=static
        status = "moving" if moving else "static"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label_name} {conf:.2f} ({status})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, text, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )

    return frame


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.route("/video_feed")
def video_feed():
    """MJPEG stream from the webcam, with live YOLOv7 detection drawn on it."""
    return Response(
        webcam_stream.frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stop_feed", methods=["POST"])
def stop_feed():
    webcam_stream.stop()
    return jsonify({"status": "stopped"})


@app.route("/upload", methods=["POST"])
def upload_video():
    """
    Accepts a video file upload, saves it, and returns a URL the frontend
    can use to stream the annotated version back via /processed_feed.
    """
    if "video" not in request.files:
        return jsonify({"error": "No file part named 'video'"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    return jsonify({
        "message": "Upload successful",
        "stream_url": f"/processed_feed?filename={filename}",
    })


@app.route("/processed_feed")
def processed_feed():
    """Streams annotated frames from a previously uploaded video file."""
    filename = request.args.get("filename")
    if not filename:
        return jsonify({"error": "filename query param required"}), 400

    path = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404

    def generate():
        cap = cv2.VideoCapture(path)
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=False
        )
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        delay = 1.0 / fps

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            annotated = detect_and_annotate(frame, bg_subtractor)
            ok, buffer = cv2.imencode(".jpg", annotated)
            if not ok:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
            time.sleep(delay)  # roughly match original playback speed
        cap.release()

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": DEVICE})


if __name__ == "__main__":
    # threaded=True so /video_feed and /upload can be handled concurrently
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
