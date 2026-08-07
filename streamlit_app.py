"""
streamlit_app.py
Streamlit frontend + backend (single file) for YOLOv7 person/vehicle
detection, with moving/static tagging via background subtraction.

Two input modes:
  1. Upload a video file  -> processed frame-by-frame and shown live.
  2. Webcam                -> only works when running locally
                               (Streamlit Cloud has no camera access).

Run with:  streamlit run streamlit_app.py
Requires:  pip install -r requirements.txt
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, Tuple

import cv2
import numpy as np
import streamlit as st
import torch

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("detector")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Config:
    conf_threshold: float = float(os.environ.get("CONF_THRESHOLD", 0.45))
    inference_size: int = int(os.environ.get("INFERENCE_SIZE", 640))
    motion_history: int = 300
    motion_var_threshold: int = 40
    motion_pixel_ratio: float = 15.0  # % white pixels inside a box to call it "moving"

    # COCO class ids we care about: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
    target_classes: Dict[int, str] = field(default_factory=lambda: {
        0: "person",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    })

    # BGR colors per detection state -- teal for moving, amber for static.
    color_moving: Tuple[int, int, int] = (168, 145, 30)
    color_static: Tuple[int, int, int] = (30, 160, 235)
    label_text_color: Tuple[int, int, int] = (255, 255, 255)


CONFIG = Config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Model loading (cached so it only loads once per session)
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Loading YOLOv7 model...")
def load_model(device: str) -> torch.nn.Module:
    logger.info("Loading YOLOv7 on %s (first run downloads yolov7.pt)...", device)
    try:
        yolo_model = torch.hub.load(
            "WongKinYiu/yolov7",
            "custom",
            "yolov7.pt",
            trust_repo=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load YOLOv7 model.")
        raise RuntimeError("Could not load YOLOv7 model") from exc

    yolo_model.to(device)
    yolo_model.conf = CONFIG.conf_threshold
    logger.info("Model loaded successfully.")
    return yolo_model


# --------------------------------------------------------------------------- #
# Core detection + annotation logic
# --------------------------------------------------------------------------- #

def detect_and_annotate(frame: np.ndarray, model: torch.nn.Module, bg_subtractor) -> np.ndarray:
    """
    Runs YOLOv7 on a single BGR frame, keeps only person/vehicle classes,
    tags each detection as "moving" or "static" using a motion mask, and
    draws labeled boxes. Returns the annotated frame.
    """
    fg_mask = bg_subtractor.apply(frame)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        results = model(rgb, size=CONFIG.inference_size)
    except Exception:  # noqa: BLE001
        logger.exception("Inference failed on a frame; skipping annotation.")
        return frame

    detections = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, conf, cls

    for x1, y1, x2, y2, conf, cls_id in detections:
        cls_id = int(cls_id)
        if cls_id not in CONFIG.target_classes:
            continue

        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        label_name = CONFIG.target_classes[cls_id]

        box_mask = fg_mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
        moving = box_mask.size > 0 and (np.mean(box_mask) > CONFIG.motion_pixel_ratio)

        color = CONFIG.color_moving if moving else CONFIG.color_static
        status = "moving" if moving else "static"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label_name} {conf:.2f} ({status})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, text, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, CONFIG.label_text_color, 1, cv2.LINE_AA,
        )

    return frame


def new_bg_subtractor():
    return cv2.createBackgroundSubtractorMOG2(
        history=CONFIG.motion_history,
        varThreshold=CONFIG.motion_var_threshold,
        detectShadows=False,
    )


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="Person & Vehicle Detector", page_icon="🚦", layout="wide")

st.title("🚦 Person & Vehicle Detection")
st.caption("YOLOv7-based detector with moving vs. static tagging.")

with st.sidebar:
    st.header("Settings")
    source_mode = st.radio("Input source", ["Upload video", "Webcam (local only)"])
    st.markdown(f"**Device:** `{DEVICE}`")
    st.markdown("---")
    st.markdown(
        "Legend:\n"
        "- 🟦 Teal box = moving\n"
        "- 🟧 Amber box = static"
    )

try:
    model = load_model(DEVICE)
except RuntimeError as exc:
    st.error("Failed to load the YOLOv7 model. See details below.")
    st.exception(exc)
    st.stop()
frame_placeholder = st.empty()
status_placeholder = st.empty()

# --------------------------------------------------------------------------- #
# Mode 1: Uploaded video file
# --------------------------------------------------------------------------- #

if source_mode == "Upload video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    run_button = st.button("▶ Start processing", disabled=uploaded_file is None)
    stop_button = st.button("⏹ Stop")

    if "processing" not in st.session_state:
        st.session_state.processing = False

    if run_button and uploaded_file is not None:
        st.session_state.processing = True

    if stop_button:
        st.session_state.processing = False

    if uploaded_file is not None and st.session_state.processing:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Could not open the uploaded video.")
        else:
            bg_subtractor = new_bg_subtractor()
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            delay = 1.0 / fps

            status_placeholder.info("Processing video... click Stop to end early.")
            while cap.isOpened() and st.session_state.processing:
                success, frame = cap.read()
                if not success:
                    break
                annotated = detect_and_annotate(frame, model, bg_subtractor)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)
                time.sleep(delay)

            cap.release()
            os.unlink(video_path)
            st.session_state.processing = False
            status_placeholder.success("Done processing video.")

# --------------------------------------------------------------------------- #
# Mode 2: Webcam (only works when run locally, not on Streamlit Cloud)
# --------------------------------------------------------------------------- #

else:
    st.warning(
        "Webcam mode only works when the app is run **locally** "
        "(`streamlit run streamlit_app.py`). It will not work on Streamlit "
        "Community Cloud since there's no camera access on the server."
    )
    start_cam = st.button("▶ Start webcam")
    stop_cam = st.button("⏹ Stop webcam")

    if "cam_running" not in st.session_state:
        st.session_state.cam_running = False

    if start_cam:
        st.session_state.cam_running = True
    if stop_cam:
        st.session_state.cam_running = False

    if st.session_state.cam_running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access the webcam.")
            st.session_state.cam_running = False
        else:
            bg_subtractor = new_bg_subtractor()
            status_placeholder.info("Webcam live... click Stop webcam to end.")
            while st.session_state.cam_running:
                success, frame = cap.read()
                if not success:
                    st.error("Failed to read from webcam.")
                    break
                annotated = detect_and_annotate(frame, model, bg_subtractor)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)
            cap.release()
