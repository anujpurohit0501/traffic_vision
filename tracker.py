"""
tracker.py
----------
A lightweight, dependency-free centroid tracker.

YOLOv7 gives us per-frame detections but no notion of "this is the same
object as last frame". To decide whether a detected object is *moving* we
need to follow it across frames, so this module assigns a stable integer ID
to each detection using a greedy nearest-centroid + IoU match, and keeps a
short history of centroids per ID (used both to compute displacement and to
draw a motion trail).

This is intentionally simple (no Kalman filter / Hungarian algorithm) so the
whole project stays easy to read and to explain -- swap in `motpy`,
`deep-sort-realtime`, or `ByteTrack` later if you need more robust tracking.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from typing import Deque, Dict, List, Tuple

import numpy as np

BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2


def _centroid(box: BBox) -> Tuple[int, int]:
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def _iou(box_a: BBox, box_b: BBox) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


class TrackedObject:
    __slots__ = ("id", "box", "label", "confidence", "centroid_history", "frames_missing", "start_centroid")

    def __init__(self, obj_id: int, box: BBox, label: str, confidence: float, history_len: int = 20):
        self.id = obj_id
        self.box = box
        self.label = label
        self.confidence = confidence
        self.centroid_history: Deque[Tuple[int, int]] = deque(maxlen=history_len)
        self.centroid_history.append(_centroid(box))
        self.start_centroid = self.centroid_history[0]
        self.frames_missing = 0

    def update(self, box: BBox, label: str, confidence: float) -> None:
        self.box = box
        self.label = label
        self.confidence = confidence
        self.centroid_history.append(_centroid(box))
        self.frames_missing = 0

    def displacement(self, lookback: int = 8) -> float:
        """Euclidean pixel distance between the oldest and newest centroid
        within the last `lookback` frames -- used as a motion signal."""
        pts = list(self.centroid_history)[-lookback:]
        if len(pts) < 2:
            return 0.0
        (x1, y1), (x2, y2) = pts[0], pts[-1]
        return float(np.hypot(x2 - x1, y2 - y1))


class CentroidTracker:
    """Greedy IoU + centroid-distance based multi-object tracker."""

    def __init__(self, max_missing: int = 15, max_distance: int = 120, history_len: int = 20):
        self.next_id = 0
        self.objects: "OrderedDict[int, TrackedObject]" = OrderedDict()
        self.max_missing = max_missing
        self.max_distance = max_distance
        self.history_len = history_len

    def reset(self) -> None:
        self.next_id = 0
        self.objects.clear()

    def update(self, detections: List[Tuple[BBox, str, float]]) -> Dict[int, TrackedObject]:
        """
        detections: list of (box, label, confidence) for the current frame.
        Returns the current dict of {id: TrackedObject} after matching.
        """
        if not detections:
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id].frames_missing += 1
                if self.objects[obj_id].frames_missing > self.max_missing:
                    del self.objects[obj_id]
            return self.objects

        if not self.objects:
            for box, label, conf in detections:
                self._register(box, label, conf)
            return self.objects

        existing_ids = list(self.objects.keys())
        existing_boxes = [self.objects[i].box for i in existing_ids]
        existing_centroids = [_centroid(b) for b in existing_boxes]

        det_centroids = [_centroid(d[0]) for d in detections]

        # Cost = distance, but only allow a match if class matches and IoU is plausible
        cost = np.zeros((len(existing_ids), len(detections)), dtype=np.float32)
        for i, (ex_c, ex_box) in enumerate(zip(existing_centroids, existing_boxes)):
            for j, (det_box, det_label, _conf) in enumerate(detections):
                same_class = self.objects[existing_ids[i]].label == det_label
                dist = float(np.hypot(ex_c[0] - det_centroids[j][0], ex_c[1] - det_centroids[j][1]))
                iou = _iou(ex_box, det_box)
                if not same_class or (dist > self.max_distance and iou == 0):
                    cost[i, j] = 1e6
                else:
                    # Blend distance and (1 - iou) so overlapping boxes are preferred
                    cost[i, j] = dist * 0.7 + (1 - iou) * 100 * 0.3

        matched_rows, matched_cols = set(), set()
        # Greedy matching, cheapest pairs first
        flat = [(cost[i, j], i, j) for i in range(cost.shape[0]) for j in range(cost.shape[1])]
        flat.sort(key=lambda t: t[0])
        for c, i, j in flat:
            if c >= 1e6:
                break
            if i in matched_rows or j in matched_cols:
                continue
            matched_rows.add(i)
            matched_cols.add(j)
            obj_id = existing_ids[i]
            box, label, conf = detections[j]
            self.objects[obj_id].update(box, label, conf)

        # Unmatched existing objects -> mark missing / evict
        for i, obj_id in enumerate(existing_ids):
            if i not in matched_rows:
                self.objects[obj_id].frames_missing += 1
                if self.objects[obj_id].frames_missing > self.max_missing:
                    del self.objects[obj_id]

        # Unmatched detections -> register as new objects
        for j, (box, label, conf) in enumerate(detections):
            if j not in matched_cols:
                self._register(box, label, conf)

        return self.objects

    def _register(self, box: BBox, label: str, confidence: float) -> None:
        self.objects[self.next_id] = TrackedObject(self.next_id, box, label, confidence, self.history_len)
        self.next_id += 1
