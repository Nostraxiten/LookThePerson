"""
Face Landmarker (Face Mesh) wrapper for LookThePerson.
Detects 478 facial landmarks, draws tesselation mesh, detects expressions.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from models import ensure_model, model_path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MESH_COLOR = (180, 255, 180)
MESH_COLOR_DIM = (80, 130, 80)
IRIS_COLOR = (255, 100, 255)
CONTOUR_COLOR = (0, 255, 200)

# Key landmark indices for expression detection
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

LEFT_EYEBROW_TOP = 105
LEFT_EYEBROW_INNER = 107
RIGHT_EYEBROW_TOP = 334
RIGHT_EYEBROW_INNER = 336

LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

# Tesselation connections (subset — full mesh is huge, we use contours)
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
]

LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362]
LIPS_CONTOUR = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61]


# ---------------------------------------------------------------------------
# Expression helpers
# ---------------------------------------------------------------------------

def _dist(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    return (dx * dx + dy * dy) ** 0.5


def detect_expressions(landmarks):
    """
    Detect facial expressions from 478-point face mesh.
    Returns dict of booleans: mouth_open, left_eye_closed, right_eye_closed,
    eyebrows_raised, smile.
    """
    results = {
        "mouth_open": False,
        "left_eye_closed": False,
        "right_eye_closed": False,
        "left_wink": False,
        "right_wink": False,
        "eyebrows_raised": False,
        "smile": False,
    }

    if len(landmarks) < 478:
        return results

    # Eye openness
    left_eye_dist = _dist(landmarks[LEFT_EYE_TOP], landmarks[LEFT_EYE_BOTTOM])
    right_eye_dist = _dist(landmarks[RIGHT_EYE_TOP], landmarks[RIGHT_EYE_BOTTOM])

    face_height = _dist(landmarks[10], landmarks[152])
    if face_height < 0.01:
        return results

    left_eye_ratio = left_eye_dist / face_height
    right_eye_ratio = right_eye_dist / face_height

    results["left_eye_closed"] = left_eye_ratio < 0.018
    results["right_eye_closed"] = right_eye_ratio < 0.018

    # Wink detection: one eye closed, other open
    results["left_wink"] = results["left_eye_closed"] and not results["right_eye_closed"]
    results["right_wink"] = results["right_eye_closed"] and not results["left_eye_closed"]

    # Mouth open
    mouth_dist = _dist(landmarks[MOUTH_TOP], landmarks[MOUTH_BOTTOM])
    mouth_width = _dist(landmarks[MOUTH_LEFT], landmarks[MOUTH_RIGHT])
    mouth_ratio = mouth_dist / max(mouth_width, 0.001)
    results["mouth_open"] = mouth_ratio > 0.35

    # Smile detection: mouth corners higher relative to center
    mouth_center_y = (landmarks[MOUTH_TOP].y + landmarks[MOUTH_BOTTOM].y) / 2
    left_corner_y = landmarks[MOUTH_LEFT].y
    right_corner_y = landmarks[MOUTH_RIGHT].y
    avg_corner_y = (left_corner_y + right_corner_y) / 2
    results["smile"] = avg_corner_y < mouth_center_y and mouth_width / face_height > 0.28

    # Eyebrows raised
    left_brow_dist = _dist(landmarks[LEFT_EYEBROW_TOP], landmarks[LEFT_EYE_TOP])
    right_brow_dist = _dist(landmarks[RIGHT_EYEBROW_TOP], landmarks[RIGHT_EYE_TOP])
    avg_brow_dist = (left_brow_dist + right_brow_dist) / 2
    results["eyebrows_raised"] = avg_brow_dist / face_height > 0.065

    return results


def estimate_gaze_direction(landmarks):
    """
    Estimate gaze direction from iris landmarks.
    Returns (horizontal, vertical) where negative=left/up, positive=right/down.
    Values roughly in range [-1, 1].
    """
    if len(landmarks) < 478:
        return 0.0, 0.0

    # Left eye boundaries
    left_inner = landmarks[133]
    left_outer = landmarks[33]
    left_iris = landmarks[LEFT_IRIS_CENTER]

    # Right eye boundaries
    right_inner = landmarks[362]
    right_outer = landmarks[263]
    right_iris = landmarks[RIGHT_IRIS_CENTER]

    # Horizontal: where is iris between inner and outer corner
    left_range = left_inner.x - left_outer.x
    right_range = right_outer.x - right_inner.x

    if abs(left_range) > 0.001 and abs(right_range) > 0.001:
        left_h = (left_iris.x - left_outer.x) / left_range
        right_h = (right_iris.x - right_inner.x) / right_range
        h = (left_h + right_h) / 2
        h = (h - 0.5) * 2  # normalize to [-1, 1]
    else:
        h = 0.0

    # Vertical: how high is iris relative to eye height
    left_top = landmarks[LEFT_EYE_TOP]
    left_bot = landmarks[LEFT_EYE_BOTTOM]
    right_top = landmarks[RIGHT_EYE_TOP]
    right_bot = landmarks[RIGHT_EYE_BOTTOM]

    left_v_range = left_bot.y - left_top.y
    right_v_range = right_bot.y - right_top.y

    if abs(left_v_range) > 0.001 and abs(right_v_range) > 0.001:
        left_v = (left_iris.y - left_top.y) / left_v_range
        right_v = (right_iris.y - right_top.y) / right_v_range
        v = (left_v + right_v) / 2
        v = (v - 0.5) * 2
    else:
        v = 0.0

    return h, v


# ---------------------------------------------------------------------------
# FaceMeshModel
# ---------------------------------------------------------------------------

class FaceMeshModel:
    """Manages face landmark detection with mesh overlay and expressions."""

    def __init__(self):
        self._detector = None

    def start(self):
        ensure_model("face_mesh")
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path("face_mesh")),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=2,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)

    def stop(self):
        if self._detector:
            self._detector.close()
            self._detector = None

    def detect(self, mp_image, timestamp_ms):
        if not self._detector:
            return None
        return self._detector.detect_for_video(mp_image, timestamp_ms)

    @staticmethod
    def draw_mesh(frame, landmarks, width, height, draw_tesselation=True):
        """Draw face mesh overlay on frame."""
        points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]

        # Draw face oval
        for i in range(len(FACE_OVAL) - 1):
            a, b = FACE_OVAL[i], FACE_OVAL[i + 1]
            if a < len(points) and b < len(points):
                cv2.line(frame, points[a], points[b], CONTOUR_COLOR, 1)

        # Draw eye contours
        for contour in (LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR):
            for i in range(len(contour) - 1):
                a, b = contour[i], contour[i + 1]
                if a < len(points) and b < len(points):
                    cv2.line(frame, points[a], points[b], CONTOUR_COLOR, 1)

        # Draw lip contour
        for i in range(len(LIPS_CONTOUR) - 1):
            a, b = LIPS_CONTOUR[i], LIPS_CONTOUR[i + 1]
            if a < len(points) and b < len(points):
                cv2.line(frame, points[a], points[b], (0, 140, 255), 1)

        # Draw iris points
        if len(landmarks) >= 478:
            for iris_idx in (LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER):
                cv2.circle(frame, points[iris_idx], 3, IRIS_COLOR, -1)

        # Draw sparse tesselation dots
        if draw_tesselation:
            for i in range(0, min(468, len(points)), 3):
                cv2.circle(frame, points[i], 1, MESH_COLOR_DIM, -1)

    @staticmethod
    def draw_gaze_indicator(frame, landmarks, width, height):
        """Draw gaze direction indicator."""
        if len(landmarks) < 478:
            return

        h, v = estimate_gaze_direction(landmarks)

        # Draw gaze indicator in top-right of face
        nose = landmarks[1]
        cx, cy = int(nose.x * width), int(nose.y * height) - 40

        # Background circle
        cv2.circle(frame, (cx, cy), 20, (40, 40, 40), -1)
        cv2.circle(frame, (cx, cy), 20, IRIS_COLOR, 1)

        # Dot showing gaze direction
        gx = int(cx + h * 14)
        gy = int(cy + v * 14)
        cv2.circle(frame, (gx, gy), 4, IRIS_COLOR, -1)
