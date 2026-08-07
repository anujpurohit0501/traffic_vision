"""
app.py
------
Streamlit frontend for the YOLOv8-powered Moving Object Detection system.
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


class Detection:
    __slots__ = ("box", "label", "confidence")

    def __init__(self, box, label, confidence):
        self.box = box
        self.label = label
        self.confidence = confidence


class YOLOv8Detector:
    def __init__(self, weights_path: str, device: str, conf_thres: float, iou_thres: float, img_size: int):
        self.model = YOLO(weights_path)
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


# Page configuration with a professional light/clean theme
st.set_page_config(
    page_title="Moving Object Detection | YOLOv8",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional Professional Light/Off-White UI Styling
CUSTOM_CSS = """
<style>
    /* Global Clean Styling */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
    }
    
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}

    /* Hero Section */
    .hero {
        padding: 1.5rem 1.8rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        color: #0f172a;
        margin-bottom: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .hero h1 {margin: 0; font-size: 1.75rem; font-weight: 700; color: #0f172a;}
    .hero p {margin: 0.4rem 0 0 0; color: #475569; font-size: 0.95rem;}
    
    .pill {
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        background: #e0f2fe; color: #0369a1;
        font-size: 0.75rem; font-weight: 600; margin-right: 6px; letter-spacing: .02em;
    }

    /* Metric Cards Styling */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    div[data-testid="stMetricLabel"] {font-size: 0.8rem; color: #64748b; font-weight: 500;}
    div[data-testid="stMetricValue"] {color: #0f172a; font-weight: 700;}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] > div {padding-top: 1rem;}

    .footer-note {color: #94a3b8; font-size: 0.8rem; text-align: center; margin-top: 3rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <span class="pill">YOLOv8</span><span class="pill">OpenCV</span><span class="pill">Real-time</span>
        <h1>🎯 Smart Moving Object Detection</h1>
        <p>Detects <b>what</b> an object is using a pretrained YOLOv8 model, and <b>whether it's moving</b> 
        via background-subtraction and multi-frame centroid tracking.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Model Settings")
    weights_path = st.text_input(
        "YOLOv8 weights",
        value="yolov8n.pt",
        help="Model name like yolov8n.pt, yolov8s.pt or path to custom weights.",
    )
    device = st.selectbox("Device", options=["cpu", "cuda"], index=0)
    img_size = st.select_slider("Inference size", options=[320, 416, 512, 640, 768], value=640)

    st.subheader("Detection thresholds")
    conf_thres = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
    iou_thres = st.slider("NMS IoU threshold", 0.05, 0.95, 0.45, 0.05)

    st.subheader("Motion sensitivity")
    motion_overlap_thres = st.slider("Min. foreground overlap", 0.0, 1.0, 0.15, 0.05)
    displacement_thres = st.slider("Min. centroid displacement (px)", 0, 100, 18, 2)
    motion_logic = st.radio("Combine signals with", ["OR (sensitive)", "AND (strict)"], index=0, horizontal=True)

    st.subheader("Display Options")
    class_options = st.multiselect("Filter Classes (empty = all)", options=COCO_CLASSES, default=[])
    show_static = st.checkbox("Show static objects", value=True)
    show_trails = st.checkbox("Show motion trails", value=True)
    show_hud = st.checkbox("Show stats overlay", value=True)

classes_filter = None
if class_options:
    classes_filter = [COCO_CLASSES.index(c) for c in class_options]


@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def load_detector(weights_path: str, device: str, conf_thres: float, iou_thres: float, img_size: int):
    return YOLOv8Detector(
        weights_path=weights_path, device=device,
        conf_thres=conf_thres, iou_thres=iou_thres, img_size=img_size,
    )


def process_frame(frame_bgr, detector, motion_estimator, tracker, frame_idx):
    mask = motion_estimator.apply(frame_bgr)
    detections = detector.detect(frame_bgr, classes_filter=classes_filter)
    det_tuples = [(d.box, d.label, d.confidence) for d in detections]

    tracked = tracker.update(det_tuples)

    annotated = frame_bgr.copy()
    moving_count, static_count = 0, 0

    for obj_id, obj in tracked.items():
        if obj.frames_missing > 0:
            continue

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


# App Layout Tabs
tab_video, tab_image, tab_about = st.tabs(["📹 Upload Video", "🖼️ Upload Image", "ℹ️ About"])

with tab_video:
    left, right = st.columns([2, 1])
    with left:
        video_file = st.file_uploader("Upload video file", type=["mp4", "mov", "avi", "mkv"])
    with right:
        st.write("")
        st.write("")
        run_video = st.button("▶️ Run Detection", type="primary", use_container_width=True, disabled=video_file is None)

    if video_file is not None and run_video:
        try:
            detector = load_detector(weights_path, device, conf_thres, iou_thres, img_size)
        except Exception as e:
            detector = None
            st.error(f"Error loading model: {e}")

        if detector is not None:
            detector.set_thresholds(conf_thres, iou_thres)
            motion_estimator = MotionEstimator()
            tracker = CentroidTracker()

            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

            out_path = str(Path(tempfile.gettempdir()) / f"output_{int(time.time())}.mp4")
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

            preview = st.empty()
            progress = st.progress(0.0, text="Processing video...")
            metric_cols = st.columns(4)
            m_frame, m_moving, m_static, m_fps = metric_cols[0].empty(), metric_cols[1].empty(), metric_cols[2].empty(), metric_cols[3].empty()

            history, fps_meter, frame_idx = [], FPSMeter(), 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated, moving_c, static_c = process_frame(frame, detector, motion_estimator, tracker, frame_idx)
                writer.write(annotated)

                if frame_idx % 2 == 0:
                    preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                cur_fps = fps_meter.tick()
                history.append({"frame": frame_idx, "moving": moving_c, "static": static_c})

                m_frame.metric("Frame", f"{frame_idx + 1}/{total_frames}")
                m_moving.metric("Moving", moving_c)
                m_static.metric("Static", static_c)
                m_fps.metric("FPS", f"{cur_fps:.1f}")

                frame_idx += 1
                progress.progress(min(frame_idx / total_frames, 1.0), text=f"Processing... {frame_idx}/{total_frames}")

            cap.release()
            writer.release()
            progress.empty()

            st.success("Processing complete!")
            st.video(out_path)
            with open(out_path, "rb") as f:
                st.download_button("⬇️ Download Video", f, file_name="detected_output.mp4", mime="video/mp4")

            if history:
                df = pd.DataFrame(history)
                st.subheader("📈 Object counts over time")
                st.line_chart(df.set_index("frame")[["moving", "static"]])

with tab_image:
    image_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if image_file is not None:
        detector = load_detector(weights_path, device, conf_thres, iou_thres, img_size)
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        detector.set_thresholds(conf_thres, iou_thres)
        detections = detector.detect(frame, classes_filter=classes_filter)

        annotated = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d.box
            color = CLASS_COLORS.get(d.label, (0, 200, 255))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{d.label} {d.confidence*100:.0f}%", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        col1, col2 = st.columns(2)
        col1.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
        col2.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Detected ({len(detections)})", use_container_width=True)

with tab_about:
    st.markdown("### System Architecture\nPowered by **YOLOv8**, **OpenCV**, and **Streamlit** for real-time tracking.")

st.markdown('<p class="footer-note">Moving Object Detection System · Built with Streamlit</p>', unsafe_allow_html=True)
