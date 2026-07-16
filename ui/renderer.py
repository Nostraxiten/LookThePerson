"""
Renderer utilities for LookThePerson.
Grid overlay, night mode, head circle, frame fitting.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Frame utilities
# ---------------------------------------------------------------------------

def fit_frame_to_screen(frame, screen_width, screen_height):
    """Scale and center-pad frame to fill the screen."""
    fh, fw = frame.shape[:2]
    scale = min(screen_width / fw, screen_height / fh)
    rw = max(1, int(fw * scale))
    rh = max(1, int(fh * scale))

    resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
    output = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    x = (screen_width - rw) // 2
    y = (screen_height - rh) // 2
    output[y:y + rh, x:x + rw] = resized
    return output


# ---------------------------------------------------------------------------
# Grid overlay
# ---------------------------------------------------------------------------

GRID_COLOR = (60, 60, 60)
GRID_COLOR_CENTER = (100, 100, 100)


def draw_grid(frame, divisions=6):
    """Draw a reference grid over the frame."""
    h, w = frame.shape[:2]
    step_x = w // divisions
    step_y = h // divisions

    for i in range(1, divisions):
        x = i * step_x
        y = i * step_y
        color = GRID_COLOR_CENTER if i == divisions // 2 else GRID_COLOR
        thickness = 2 if i == divisions // 2 else 1
        cv2.line(frame, (x, 0), (x, h), color, thickness)
        cv2.line(frame, (0, y), (w, y), color, thickness)

    # Center crosshair
    cx, cy = w // 2, h // 2
    size = 15
    cv2.line(frame, (cx - size, cy), (cx + size, cy), GRID_COLOR_CENTER, 2)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), GRID_COLOR_CENTER, 2)


# ---------------------------------------------------------------------------
# Night mode
# ---------------------------------------------------------------------------

def apply_night_mode(frame):
    """Invert colors for night mode effect."""
    return cv2.bitwise_not(frame)


# ---------------------------------------------------------------------------
# Head circle
# ---------------------------------------------------------------------------

HEAD_COLOR = (0, 255, 0)
HEAD_TOUCH_COLOR = (0, 180, 255)


def draw_head_circle(frame, landmarks, width, height, touching=False):
    """Draw a circle around the detected head."""
    from gestures.body_gestures import head_circle

    circle = head_circle(landmarks, width, height)
    if circle is None:
        return

    cx, cy, radius, _nx, _ny, _nr = circle
    color = HEAD_TOUCH_COLOR if touching else HEAD_COLOR
    cv2.circle(frame, (cx, cy), radius, color, 2)
    cv2.line(
        frame,
        (cx - radius, int(cy + radius * 0.15)),
        (cx + radius, int(cy + radius * 0.15)),
        color, 1,
    )


# ---------------------------------------------------------------------------
# Bounding box overlay for all detections
# ---------------------------------------------------------------------------

def draw_bounding_boxes(frame, result, width, height, color=(0, 255, 255), label_prefix=""):
    """Generic bounding box drawer for any detection result with bounding_box."""
    if not result or not hasattr(result, "detections") or not result.detections:
        return

    for det in result.detections:
        bbox = det.bounding_box
        x, y = int(bbox.origin_x), int(bbox.origin_y)
        w, h = int(bbox.width), int(bbox.height)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        if det.categories:
            cat = det.categories[0]
            label = f"{label_prefix}{cat.category_name} {cat.score:.0%}"
            cv2.putText(frame, label, (x + 2, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
