"""
app.py
------
Streamlit frontend for the YOLOv8-powered Moving Object Detection system.

YOLOv8 weights are handled automatically by the `ultralytics` package: point
the sidebar "weights" field at any official model name (e.g. "yolov8n.pt",
"yolov8s.pt", "yolov8m.pt") and, if it isn't already cached locally, it will
be downloaded on first use. A path to your own custom-trained .pt file also
works.

Run with:
    streamlit run app.py

Two input modes are supported:
  - Upload Video  : full offline processing with a progress bar + downloadable result
  - Upload Image  : single-frame object detection (motion needs a frame history,
                     so a single image is shown with objects detected but unclassified
                     for motion)
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from ultralytics import YOLO

from detector import MotionEstimator
from tracker import CentroidTracker
from utils import COCO_CLASSES, CLASS_COLORS, draw_detection, draw_trail, draw_hud, FPSMeter


# --------------------------------------------------------------------------
# YOLOv8 detector wrapper (drop-in replacement for the old YOLOv7Detector)
# --------------------------------------------------------------------------
class Detection:
    """A single detection result. Matches the shape process_frame() expects:
    .box (x1, y1, x2, y2 ints), .label (str), .confidence (float)."""

    __slots__ = ("box", "label", "confidence")

    def __init__(self, box, label, confidence):
        self.box = box
        self.label = label
        self.confidence = confidence


class YOLOv8Detector:
    """Thin wrapper around ultralytics.YOLO so the rest of the app doesn't
    need to know it's no longer talking to YOLOv7.

    `weights_path` can be an official model name (e.g. "yolov8n.pt") or a
    path to a custom-trained .pt file. Either way, ultralytics resolves and
    downloads the weights automatically on first use -- no manual download
    step is required.
    """

    def __init__(self, weights_path: str, device: str, conf_thres: float, iou_thres: float, img_size: int):
        self.model = YOLO(weights_path)  # auto-downloads weights if not cached locally
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size

    def set_thresholds(self, conf_thres: float, iou_thres: float):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def detect(self, frame_bgr, classes_filter=None):
        results = self.model.predict(
            source=frame_bgr,
            device=self.device,
            conf=self.conf_thres,
            iou=self.iou_thres,
            imgsz=self.img_size,
            classes=classes_filter,
            verbose=False,
        )

        detections = []
        if not results:
            return detections

        result = results[0]
        names = result.names
        for box in result.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            detections.append(Detection((x1, y1, x2, y2), label, conf))
        return detections

# --------------------------------------------------------------------------
# Page config + styling
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Moving Object Detection | YOLOv8",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}

    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #0f766e 130%);
        color: #f8fafc;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .hero h1 {margin: 0; font-size: 1.65rem; font-weight: 700;}
    .hero p {margin: 0.35rem 0 0 0; color: #cbd5e1; font-size: 0.95rem;}
    .pill {
        display: inline-block; padding: 3px 10px; border-radius: 999px;
        background: rgba(45, 212, 191, 0.15); color: #2dd4bf;
        font-size: 0.72rem; font-weight: 600; margin-right: 6px; letter-spacing: .02em;
    }

    div[data-testid="stMetric"] {
        background: #ffffff0d;
        border: 1px solid rgba(148,163,184,0.25);
        border-radius: 12px;
        padding: 0.6rem 0.8rem 0.3rem 0.8rem;
    }
    div[data-testid="stMetricLabel"] {font-size: 0.78rem; opacity: 0.85;}

    section[data-testid="stSidebar"] > div {padding-top: 1rem;}
    .legend-dot {
        display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px;
    }
    .footer-note {opacity: 0.55; font-size: 0.78rem; margin-top: 2rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <span class="pill">YOLOv8</span><span class="pill">OpenCV</span><span class="pill">Real-time</span>
        <h1>🎯 Moving Object Detection</h1>
        <p>Detects <b>what</b> an object is with a pretrained YOLOv8 model, and <b>whether it's moving</b>
        using background-subtraction + multi-frame centroid tracking. Works on uploaded video or images.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Sidebar -- configuration
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Model")
    weights_path = st.text_input(
        "YOLOv8 weights",
        value="yolov8n.pt",
        help=(
            "An official model name (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt) "
            "or a path to your own custom-trained .pt file. Official weights are downloaded "
            "automatically the first time they're used -- no manual download needed."
        ),
    )
    device = st.selectbox("Device", options=["cpu", "cuda"], index=0, help="Select 'cuda' only if a GPU + CUDA torch build is available")
    img_size = st.select_slider("Inference size", options=[320, 416, 512, 640, 768], value=640)

    st.subheader("Detection thresholds")
    conf_thres = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
    iou_thres = st.slider("NMS IoU threshold", 0.05, 0.95, 0.45, 0.05)

    st.subheader("Motion sensitivity")
    motion_overlap_thres = st.slider(
        "Min. foreground overlap to count as moving", 0.0, 1.0, 0.15, 0.05,
        help="Fraction of a detected box that must overlap the motion mask"
    )
    displacement_thres = st.slider(
        "Min. centroid displacement (px)", 0, 100, 18, 2,
        help="Minimum pixel movement over the last few frames to count as moving"
    )
    motion_logic = st.radio(
        "Combine signals with", ["OR (more sensitive)", "AND (more strict)"], index=0, horizontal=True
    )

    st.subheader("Filters & display")
    class_options = st.multiselect(
        "Only show these classes (empty = all)", options=COCO_CLASSES, default=[]
    )
    show_static = st.checkbox("Show static (non-moving) objects too", value=True)
    show_trails = st.checkbox("Show motion trails", value=True)
    show_hud = st.checkbox("Show stats overlay on frame", value=True)

    st.markdown("---")
    st.caption(
        "Official YOLOv8 weights aren't bundled with this app, but they're fetched "
        "and cached automatically the first time you run detection -- no manual download needed."
    )

classes_filter = None
if class_options:
    classes_filter = [COCO_CLASSES.index(c) for c in class_options]


# --------------------------------------------------------------------------
# Cached resource loaders
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLOv8 model (downloading weights on first use)...")
def load_detector(weights_path: str, device: str, conf_thres: float, iou_thres: float, img_size: int):
    return YOLOv8Detector(
        weights_path=weights_path, device=device,
        conf_thres=conf_thres, iou_thres=iou_thres, img_size=img_size,
    )


# --------------------------------------------------------------------------
# Core per-frame processing (shared by all modes)
# --------------------------------------------------------------------------
def process_frame(frame_bgr, detector, motion_estimator, tracker, frame_idx):
    """Run detection + motion classification + tracking on one frame.
    Returns (annotated_frame, stats_dict, moving_count, static_count)."""
    mask = motion_estimator.apply(frame_bgr)

    detections = detector.detect(frame_bgr, classes_filter=classes_filter)
    det_tuples = [(d.box, d.label, d.confidence) for d in detections]

    tracked = tracker.update(det_tuples) if frame_idx > 0 else tracker.update(det_tuples)

    # Match tracked objects back to their (box, label, confidence) for drawing
    annotated = frame_bgr.copy()
    moving_count, static_count = 0, 0

    for obj_id, obj in tracked.items():
        if obj.frames_missing > 0:
            continue  # not seen this frame

        overlap = MotionEstimator.box_motion_ratio(mask, obj.box)
        disp = obj.displacement(lookback=8)

        moving_by_mask = overlap >= motion_overlap_thres
        moving_by_track = disp >= displacement_thres

        is_moving = (moving_by_mask or moving_by_track) if motion_logic.startswith("OR") else (moving_by_mask and moving_by_track)

        if is_moving:
            moving_count += 1
        else:
            static_count += 1
            if not show_static:
                continue

        draw_detection(annotated, obj.box, obj.label, obj.confidence, is_moving, track_id=obj_id)
        if show_trails and is_moving:
            draw_trail(annotated, obj.centroid_history)

    if show_hud:
        draw_hud(annotated, {
            "Objects": str(moving_count + static_count),
            "Moving": str(moving_count),
            "Static": str(static_count),
        })

    return annotated, moving_count, static_count


# --------------------------------------------------------------------------
# Mode tabs
# --------------------------------------------------------------------------
tab_video, tab_image, tab_about = st.tabs(
    ["📹 Upload Video", "🖼️ Upload Image", "ℹ️ About"]
)

# ---------------------------- VIDEO MODE ----------------------------------
with tab_video:
    left, right = st.columns([2, 1])
    with left:
        video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"], key="video_uploader")
    with right:
        st.write("")
        st.write("")
        run_video = st.button("▶️ Run detection", type="primary", use_container_width=True, disabled=video_file is None)

    if video_file is not None and run_video:
        try:
            detector = load_detector(weights_path, device, conf_thres, iou_thres, img_size)
        except Exception as e:
            detector = None
            st.error(f"Couldn't load YOLOv8 weights `{weights_path}`: {e}")

        if detector is None:
            pass
        else:
            detector.set_thresholds(conf_thres, iou_thres)
            motion_estimator = MotionEstimator()
            tracker = CentroidTracker()

            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            in_path = tfile.name

            cap = cv2.VideoCapture(in_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

            out_path = str(Path(tempfile.gettempdir()) / f"moving_objects_{int(time.time())}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

            preview = st.empty()
            progress = st.progress(0.0, text="Processing video...")
            metric_cols = st.columns(4)
            m_frame = metric_cols[0].empty()
            m_moving = metric_cols[1].empty()
            m_static = metric_cols[2].empty()
            m_fps = metric_cols[3].empty()

            history = []
            fps_meter = FPSMeter()
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated, moving_c, static_c = process_frame(frame, detector, motion_estimator, tracker, frame_idx)
                writer.write(annotated)

                if frame_idx % 2 == 0:  # throttle UI updates for speed
                    preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                cur_fps = fps_meter.tick()
                history.append({"frame": frame_idx, "moving": moving_c, "static": static_c})

                m_frame.metric("Frame", f"{frame_idx + 1}/{total_frames}")
                m_moving.metric("Moving objects", moving_c)
                m_static.metric("Static objects", static_c)
                m_fps.metric("Processing FPS", f"{cur_fps:.1f}")

                frame_idx += 1
                progress.progress(min(frame_idx / total_frames, 1.0), text=f"Processing video... {frame_idx}/{total_frames}")

            cap.release()
            writer.release()
            progress.empty()

            st.success("Done! Preview and download the processed video below.")
            st.video(out_path)
            with open(out_path, "rb") as f:
                st.download_button("⬇️ Download processed video", f, file_name="moving_objects_output.mp4", mime="video/mp4")

            if history:
                df = pd.DataFrame(history)
                st.subheader("📈 Objects detected per frame")
                st.line_chart(df.set_index("frame")[["moving", "static"]])

# ---------------------------- IMAGE MODE -----------------------------------
with tab_image:
    st.caption("A single image has no motion history, so objects are detected and labeled, but every box is neutral (no moving/static verdict).")
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"], key="image_uploader")

    if image_file is not None:
        try:
            detector = load_detector(weights_path, device, conf_thres, iou_thres, img_size)
        except Exception as e:
            detector = None
            st.error(f"Couldn't load YOLOv8 weights `{weights_path}`: {e}")

        if detector is None:
            pass
        else:
            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            detector.set_thresholds(conf_thres, iou_thres)
            detections = detector.detect(frame, classes_filter=classes_filter)

            annotated = frame.copy()
            for d in detections:
                x1, y1, x2, y2 = d.box
                color = CLASS_COLORS.get(d.label, (0, 200, 255))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                text = f"{d.label} {d.confidence * 100:.0f}%"
                cv2.putText(annotated, text, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            col1, col2 = st.columns(2)
            col1.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
            col2.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Detected ({len(detections)} objects)", use_container_width=True)

            if detections:
                st.dataframe(
                    pd.DataFrame([{"label": d.label, "confidence": round(d.confidence, 3)} for d in detections]),
                    use_container_width=True, hide_index=True,
                )

# ---------------------------- ABOUT TAB -------------------------------------
with tab_about:
    st.markdown(
        """
### How it works

1. **Object detection** — every frame is passed through a pretrained **YOLOv8** model
   (80 COCO classes), giving bounding boxes, class labels, and confidence scores.
   Weights are resolved and downloaded automatically by `ultralytics` the first
   time a given model name is used, then cached locally for subsequent runs.
2. **Motion masking** — an MOG2 background subtractor builds a running model of the
   static background and flags pixels that differ from it (the "foreground mask").
3. **Motion classification** — each YOLO box is checked two ways:
   - **Overlap** — what fraction of the box lies inside the foreground mask.
   - **Displacement** — a lightweight centroid tracker follows each object across
     frames and measures how far its center has moved recently.
   An object is labeled **MOVING** (green) if either (or both, depending on your
   sidebar setting) of these signals cross their threshold — otherwise it's **static** (gray).
4. **Tracking** — the centroid tracker also assigns a persistent ID to each object so
   trails and per-object motion history stay stable across frames.

### Why not just use frame differencing alone?
Pure background subtraction can't tell you *what* is moving (a person vs. a car vs. a
shadow) and is easily fooled by camera shake, lighting changes, or waving trees.
Combining it with YOLOv8 gives semantic labels, while the two-signal motion check
keeps false positives low.

### Tech stack
`YOLOv8 (ultralytics)` · `OpenCV` · `Streamlit`
        """
    )

st.markdown('<p class="footer-note">Moving Object Detection · YOLOv8 + OpenCV + Streamlit</p>', unsafe_allow_html=True)
