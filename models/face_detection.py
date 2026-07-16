"""
Face Detector wrapper for LookThePerson.
Fast face detection with bounding boxes and keypoints.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from models import ensure_model, model_path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FACE_BOX_COLOR = (0, 255, 180)
FACE_KEYPOINT_COLOR = (255, 100, 255)

KEYPOINT_NAMES = [
    "Right Eye",
    "Left Eye",
    "Nose Tip",
    "Mouth Center",
    "Right Ear Tragion",
    "Left Ear Tragion",
]


# ---------------------------------------------------------------------------
# FaceDetectionModel
# ---------------------------------------------------------------------------

class FaceDetectionModel:
    """Manages fast face detection with bounding boxes and 6 keypoints."""

    def __init__(self):
        self._detector = None

    def start(self):
        ensure_model("face_detection")
        options = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=model_path("face_detection"),
            ),
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=0.4,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def stop(self):
        if self._detector:
            self._detector.close()
            self._detector = None

    def detect(self, mp_image, timestamp_ms):
        if not self._detector:
            return None
        return self._detector.detect_for_video(mp_image, timestamp_ms)

    @staticmethod
    def draw_detections(frame, result, width, height):
        """Draw bounding boxes and keypoints for each detected face."""
        if not result or not result.detections:
            return

        for detection in result.detections:
            bbox = detection.bounding_box
            x = int(bbox.origin_x)
            y = int(bbox.origin_y)
            w = int(bbox.width)
            h = int(bbox.height)

            # Bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), FACE_BOX_COLOR, 2)

            # Confidence label
            score = detection.categories[0].score if detection.categories else 0.0
            label = f"Face {score:.0%}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                frame,
                (x, y - label_size[1] - 6),
                (x + label_size[0] + 4, y),
                FACE_BOX_COLOR,
                -1,
            )
            cv2.putText(
                frame, label, (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
            )

            # Keypoints
            if detection.keypoints:
                for i, kp in enumerate(detection.keypoints):
                    px = int(kp.x * width)
                    py = int(kp.y * height)
                    cv2.circle(frame, (px, py), 3, FACE_KEYPOINT_COLOR, -1)
