"""
Object Detector wrapper for LookThePerson.
EfficientDet-Lite0 — detects 80 COCO classes with colored bounding boxes.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from models import ensure_model, model_path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Color palette for different object categories (BGR)
CATEGORY_COLORS = [
    (255, 80, 80),    # red-ish
    (80, 255, 80),    # green-ish
    (80, 80, 255),    # blue-ish
    (255, 200, 0),    # yellow
    (0, 200, 255),    # cyan
    (255, 0, 200),    # magenta
    (200, 255, 0),    # lime
    (0, 255, 200),    # teal
    (255, 140, 0),    # orange
    (140, 0, 255),    # purple
    (0, 140, 255),    # sky blue
    (255, 0, 140),    # pink
]

MAX_RESULTS = 10
MIN_CONFIDENCE = 0.35


def _color_for_category(category_name):
    """Deterministic color assignment based on category name hash."""
    idx = hash(category_name) % len(CATEGORY_COLORS)
    return CATEGORY_COLORS[idx]


# ---------------------------------------------------------------------------
# ObjectDetectionModel
# ---------------------------------------------------------------------------

class ObjectDetectionModel:
    """Manages object detection with labeled bounding boxes."""

    def __init__(self):
        self._detector = None
        self._confidence = MIN_CONFIDENCE

    @property
    def confidence(self):
        return self._confidence

    @confidence.setter
    def confidence(self, value):
        self._confidence = max(0.1, min(0.95, value))

    def start(self):
        ensure_model("object_detection")
        options = vision.ObjectDetectorOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=model_path("object_detection"),
            ),
            running_mode=vision.RunningMode.VIDEO,
            max_results=MAX_RESULTS,
            score_threshold=self._confidence,
        )
        self._detector = vision.ObjectDetector.create_from_options(options)

    def stop(self):
        if self._detector:
            self._detector.close()
            self._detector = None

    def restart_with_confidence(self, new_confidence):
        """Restart detector with a new confidence threshold."""
        self._confidence = max(0.1, min(0.95, new_confidence))
        self.stop()
        self.start()

    def detect(self, mp_image, timestamp_ms):
        if not self._detector:
            return None
        return self._detector.detect_for_video(mp_image, timestamp_ms)

    @staticmethod
    def draw_detections(frame, result, width, height):
        """Draw bounding boxes with labels and confidence for each object."""
        if not result or not result.detections:
            return

        for detection in result.detections:
            bbox = detection.bounding_box
            x = int(bbox.origin_x)
            y = int(bbox.origin_y)
            w = int(bbox.width)
            h = int(bbox.height)

            category = detection.categories[0] if detection.categories else None
            if not category:
                continue

            name = category.category_name or "unknown"
            score = category.score
            color = _color_for_category(name)

            # Bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Label background + text
            label = f"{name} {score:.0%}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                frame,
                (x, y - label_size[1] - 8),
                (x + label_size[0] + 6, y),
                color,
                -1,
            )
            cv2.putText(
                frame, label, (x + 3, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
            )
