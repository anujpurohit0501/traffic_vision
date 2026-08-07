from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from utils import COCO_CLASSES

BBox = Tuple[int, int, int, int]


@dataclass
class Detection:
    box: BBox
    label: str
    confidence: float
    class_id: int


class YOLOv7Detector:
    """
    Wrapper around Ultralytics YOLOv8.
    The class name is kept as YOLOv7Detector so app.py
    doesn't require major changes.
    """

    def __init__(
        self,
        weights_path: str = "yolov8n.pt",
        device: str = "cpu",
        conf_thres: float = 0.35,
        iou_thres: float = 0.45,
        img_size: int = 640,
        local_repo_dir=None,
    ):
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size

        # Automatically downloads yolov8n.pt if it isn't already present.
        self.model = YOLO(weights_path)

    def set_thresholds(self, conf_thres: float, iou_thres: float):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def detect(
        self,
        frame_bgr: np.ndarray,
        classes_filter: List[int] | None = None,
    ) -> List[Detection]:

        results = self.model.predict(
            frame_bgr,
            imgsz=self.img_size,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            verbose=False,
        )

        detections = []

        for result in results:

            for box in result.boxes:

                cls_id = int(box.cls.item())

                if classes_filter is not None and cls_id not in classes_filter:
                    continue

                conf = float(box.conf.item())

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                label = (
                    COCO_CLASSES[cls_id]
                    if cls_id < len(COCO_CLASSES)
                    else str(cls_id)
                )

                detections.append(
                    Detection(
                        box=(x1, y1, x2, y2),
                        label=label,
                        confidence=conf,
                        class_id=cls_id,
                    )
                )

        return detections


class MotionEstimator:
    """
    Background subtraction using OpenCV MOG2.
    Used to determine whether detected objects are moving.
    """

    def __init__(
        self,
        history: int = 300,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
    ):

        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )

        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        )

    def apply(self, frame_bgr: np.ndarray):

        mask = self.subtractor.apply(frame_bgr)

        _, mask = cv2.threshold(
            mask,
            200,
            255,
            cv2.THRESH_BINARY,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self.kernel,
            iterations=1,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_DILATE,
            self.kernel,
            iterations=2,
        )

        return mask

    @staticmethod
    def box_motion_ratio(mask: np.ndarray, box: BBox) -> float:

        h, w = mask.shape[:2]

        x1, y1, x2, y2 = box

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        roi = mask[y1:y2, x1:x2]

        if roi.size == 0:
            return 0.0

        return float(np.count_nonzero(roi)) / float(roi.size)
