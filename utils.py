"""
utils.py
--------
Shared helper utilities for the Moving Object Detection app:
- COCO class names (used by the pretrained YOLOv7 model)
- Deterministic per-class colors
- Drawing helpers (boxes, labels, trails, HUD overlay)
- A tiny rolling FPS meter

Keeping these separate from detector.py / tracker.py / app.py keeps each
file focused and easy to review.
"""

from __future__ import annotations

import time
import colorsys
from collections import deque
from typing import Deque, Dict, List, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# COCO class names (80 classes) -- the classes YOLOv7's official pretrained
# weights (yolov7.pt / yolov7x.pt / etc.) were trained on.
# ---------------------------------------------------------------------------
COCO_CLASSES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def _generate_palette(n: int) -> List[Tuple[int, int, int]]:
    """Generate `n` visually distinct BGR colors (evenly spaced hues)."""
    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))  # BGR for OpenCV
    return colors


CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    name: color for name, color in zip(COCO_CLASSES, _generate_palette(len(COCO_CLASSES)))
}

# Fixed semantic colors for motion state (overrides class color for the box
# outline so "moving vs static" is instantly readable regardless of class).
COLOR_MOVING = (0, 220, 0)      # green
COLOR_STATIC = (150, 150, 150)  # gray
COLOR_TEXT_BG = (0, 0, 0)


def draw_detection(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    confidence: float,
    is_moving: bool,
    track_id: int | None = None,
    thickness: int = 2,
) -> None:
    """Draw a single detection box + label pill onto `frame` in-place."""
    x1, y1, x2, y2 = box
    color = COLOR_MOVING if is_moving else COLOR_STATIC

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

    tag = "MOVING" if is_moving else "static"
    id_part = f"#{track_id} " if track_id is not None else ""
    text = f"{id_part}{label} {confidence * 100:.0f}% [{tag}]"

    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_y1 = max(0, y1 - th - baseline - 6)
    cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 8, y1), color, -1, lineType=cv2.LINE_AA)
    cv2.putText(
        frame, text, (x1 + 4, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )


def draw_trail(frame: np.ndarray, points: Deque[Tuple[int, int]], color=(0, 220, 0)) -> None:
    """Draw a fading motion trail for a tracked object's centroid history."""
    pts = list(points)
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        thickness = max(1, int(np.sqrt(len(pts) / float(i + 1)) * 2))
        cv2.line(frame, pts[i - 1], pts[i], color, thickness, lineType=cv2.LINE_AA)


def draw_hud(frame: np.ndarray, stats: Dict[str, str]) -> None:
    """Draw a semi-transparent stats HUD in the top-left corner."""
    overlay = frame.copy()
    pad = 10
    line_h = 22
    w = 260
    h = pad * 2 + line_h * len(stats)
    cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 20), -1)
    frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    y = pad + 14
    for key, val in stats.items():
        cv2.putText(
            frame, f"{key}: {val}", (pad, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
        y += line_h


class FPSMeter:
    """Small rolling-average FPS counter."""

    def __init__(self, window: int = 30):
        self._times: Deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        now = time.time()
        self._times.append(now)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._times) - 1) / elapsed
