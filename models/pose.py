"""
Pose Landmarker wrapper for LookThePerson.
Detects up to 4 full-body poses with segmentation masks.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from models import ensure_model, model_path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_POSES = 4
MIN_VISIBILITY = 0.35
MASK_ALPHA = 0.55
MASK_THRESHOLD = 0.35

POINT_COLOR = (0, 255, 80)

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


# ---------------------------------------------------------------------------
# PoseModel
# ---------------------------------------------------------------------------

class PoseModel:
    """Manages pose detection, skeleton drawing, and segmentation tinting."""

    def __init__(self):
        self._detector = None

    def start(self):
        ensure_model("pose")
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path("pose")),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=MAX_POSES,
            min_pose_detection_confidence=0.35,
            min_pose_presence_confidence=0.35,
            min_tracking_confidence=0.35,
            output_segmentation_masks=True,
        )
        self._detector = vision.PoseLandmarker.create_from_options(options)

    def stop(self):
        if self._detector:
            self._detector.close()
            self._detector = None

    def detect(self, mp_image, timestamp_ms):
        """Run pose detection. Returns the PoseLandmarkerResult or None."""
        if not self._detector:
            return None
        return self._detector.detect_for_video(mp_image, timestamp_ms)

    # -- Drawing helpers -----------------------------------------------------

    @staticmethod
    def landmark_visible(landmark):
        return getattr(landmark, "visibility", 1.0) >= MIN_VISIBILITY

    @staticmethod
    def draw_skeleton(frame, landmarks, width, height, color):
        points = []
        for lm in landmarks:
            x = int(lm.x * width)
            y = int(lm.y * height)
            vis = PoseModel.landmark_visible(lm)
            inframe = 0 <= x < width and 0 <= y < height
            points.append((x, y, vis and inframe))

        for s, e in POSE_CONNECTIONS:
            x1, y1, v1 = points[s]
            x2, y2, v2 = points[e]
            if v1 and v2:
                cv2.line(frame, (x1, y1), (x2, y2), color, 3)

        for x, y, vis in points:
            if vis:
                cv2.circle(frame, (x, y), 5, POINT_COLOR, -1)
                cv2.circle(frame, (x, y), 7, (20, 20, 20), 1)

    @staticmethod
    def tint_body(frame, segmentation_mask, color):
        mask = segmentation_mask.numpy_view()
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

        body_mask = mask > MASK_THRESHOLD
        color_layer = np.zeros_like(frame)
        color_layer[:] = color
        tinted = cv2.addWeighted(frame, 1.0 - MASK_ALPHA, color_layer, MASK_ALPHA, 0)
        frame[body_mask] = tinted[body_mask]

    @staticmethod
    def body_center(landmarks, width, height):
        ids = (11, 12, 23, 24)
        pts = []
        for i in ids:
            lm = landmarks[i]
            if PoseModel.landmark_visible(lm):
                x, y = int(lm.x * width), int(lm.y * height)
                if 0 <= x < width and 0 <= y < height:
                    pts.append((x, y))
        if not pts:
            return None
        cx = sum(p[0] for p in pts) // len(pts)
        cy = sum(p[1] for p in pts) // len(pts)
        return cx, cy
