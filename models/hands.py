"""
Hand Landmarker wrapper for LookThePerson.
Detects up to 2 hands, counts fingers, draws skeleton.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from models import ensure_model, model_path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HAND_COLOR = (255, 180, 0)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


# ---------------------------------------------------------------------------
# Hand helpers
# ---------------------------------------------------------------------------

def _distance_2d(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    return (dx * dx + dy * dy) ** 0.5


def count_extended_fingers(hand_landmarks):
    """Count how many fingers are extended (0-5)."""
    wrist = hand_landmarks[0]
    thumb_tip = hand_landmarks[4]
    thumb_ip = hand_landmarks[3]
    extended = 0

    if _distance_2d(thumb_tip, wrist) > _distance_2d(thumb_ip, wrist) + 0.035:
        extended += 1

    for tip_id, pip_id in ((8, 6), (12, 10), (16, 14), (20, 18)):
        if hand_landmarks[tip_id].y < hand_landmarks[pip_id].y - 0.02:
            extended += 1

    return extended


def detect_hand_gesture(hand_landmarks):
    """
    Detect specific hand gestures based on finger positions.
    Returns a string identifier or None.
    """
    fingers = count_extended_fingers(hand_landmarks)

    # Check thumbs up: only thumb extended, hand roughly upright
    if fingers == 1:
        thumb_tip = hand_landmarks[4]
        index_tip = hand_landmarks[8]
        if thumb_tip.y < index_tip.y:
            return "thumbs_up"

    # Check peace sign: index + middle extended
    if fingers == 2:
        index_tip = hand_landmarks[8]
        middle_tip = hand_landmarks[12]
        ring_tip = hand_landmarks[16]
        if index_tip.y < hand_landmarks[6].y and middle_tip.y < hand_landmarks[10].y:
            if ring_tip.y > hand_landmarks[14].y:
                return "peace"

    # Rock sign: index + pinky extended
    if fingers == 2:
        index_tip = hand_landmarks[8]
        pinky_tip = hand_landmarks[20]
        middle_tip = hand_landmarks[12]
        ring_tip = hand_landmarks[16]
        if (
            index_tip.y < hand_landmarks[6].y
            and pinky_tip.y < hand_landmarks[18].y
            and middle_tip.y > hand_landmarks[10].y
            and ring_tip.y > hand_landmarks[14].y
        ):
            return "rock"

    # Open palm
    if fingers == 5:
        return "open_palm"

    # Fist
    if fingers == 0:
        return "fist"

    return f"{fingers}_fingers"


def hand_calculator_gesture(hand_landmarks):
    """Map finger count to calculator input."""
    count = count_extended_fingers(hand_landmarks)
    if count == 0:
        return "+"
    if 1 <= count <= 4:
        return str(count)
    if count == 5:
        return "+"
    return None


# ---------------------------------------------------------------------------
# HandModel
# ---------------------------------------------------------------------------

class HandModel:
    """Manages hand detection and skeleton rendering."""

    def __init__(self):
        self._detector = None

    def start(self):
        ensure_model("hand")
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path("hand")),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.45,
            min_hand_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)

    def stop(self):
        if self._detector:
            self._detector.close()
            self._detector = None

    def detect(self, mp_image, timestamp_ms):
        if not self._detector:
            return None
        return self._detector.detect_for_video(mp_image, timestamp_ms)

    @staticmethod
    def draw_skeleton(frame, hand_landmarks, width, height, color=None):
        color = color or HAND_COLOR
        points = [(int(p.x * width), int(p.y * height)) for p in hand_landmarks]

        for s, e in HAND_CONNECTIONS:
            cv2.line(frame, points[s], points[e], color, 2)

        for pt in points:
            cv2.circle(frame, pt, 4, color, -1)
