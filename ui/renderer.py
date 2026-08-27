"""
Frame rendering utilities for LookThePerson.

Window fitting, reference guides, and the theme-aware drawing of skeletons,
head circles and detection boxes.

The original function signatures (``fit_frame_to_screen``, ``draw_grid``,
``apply_night_mode``, ``draw_head_circle``, ``draw_bounding_boxes``) are
unchanged, with optional arguments added for theming.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.theme import Theme, get_theme

__all__ = [
    "fit_frame_to_screen",
    "letterbox",
    "draw_grid",
    "draw_thirds",
    "draw_safe_area",
    "draw_crosshair",
    "apply_night_mode",
    "draw_head_circle",
    "draw_bounding_boxes",
    "draw_corner_box",
    "draw_watermark",
    "draw_border",
    "resize_keep_aspect",
]

Color = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# Frame fitting
# ---------------------------------------------------------------------------

def resize_keep_aspect(
    frame: np.ndarray,
    max_width: int,
    max_height: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Scale a frame to fit inside a box without distorting it."""
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, new_size, interpolation=interpolation)


def letterbox(
    frame: np.ndarray,
    target_width: int,
    target_height: int,
    background: Color = (0, 0, 0),
) -> np.ndarray:
    """Scale and centre a frame on a fixed-size canvas, preserving aspect."""
    resized = resize_keep_aspect(frame, target_width, target_height)
    rh, rw = resized.shape[:2]

    canvas = np.full((target_height, target_width, 3), background, dtype=np.uint8)
    x = (target_width - rw) // 2
    y = (target_height - rh) // 2
    canvas[y:y + rh, x:x + rw] = resized
    return canvas


def fit_frame_to_screen(
    frame: np.ndarray,
    screen_width: int,
    screen_height: int,
) -> np.ndarray:
    """Scale and centre-pad a frame to fill the screen (letterboxed)."""
    return letterbox(frame, screen_width, screen_height)


# ---------------------------------------------------------------------------
# Guides
# ---------------------------------------------------------------------------

GRID_COLOR = (60, 60, 60)
GRID_COLOR_CENTER = (100, 100, 100)


def draw_grid(
    frame: np.ndarray,
    divisions: int = 6,
    color: Optional[Color] = None,
    center_color: Optional[Color] = None,
) -> None:
    """Reference grid with a highlighted centre line and crosshair."""
    color = color or GRID_COLOR
    center_color = center_color or GRID_COLOR_CENTER

    height, width = frame.shape[:2]
    divisions = max(2, divisions)
    step_x = width // divisions
    step_y = height // divisions

    for index in range(1, divisions):
        is_center = index == divisions // 2
        line_color = center_color if is_center else color
        thickness = 2 if is_center else 1
        cv2.line(frame, (index * step_x, 0), (index * step_x, height), line_color, thickness)
        cv2.line(frame, (0, index * step_y), (width, index * step_y), line_color, thickness)

    draw_crosshair(frame, center_color)


def draw_thirds(frame: np.ndarray, color: Color = (90, 90, 90)) -> None:
    """Rule-of-thirds guides for framing shots."""
    height, width = frame.shape[:2]
    for index in (1, 2):
        x = width * index // 3
        y = height * index // 3
        cv2.line(frame, (x, 0), (x, height), color, 1)
        cv2.line(frame, (0, y), (width, y), color, 1)


def draw_safe_area(
    frame: np.ndarray,
    margin: float = 0.08,
    color: Color = (120, 120, 120),
) -> None:
    """Rectangle marking the area guaranteed to stay visible when cropped."""
    height, width = frame.shape[:2]
    mx = int(width * margin)
    my = int(height * margin)
    cv2.rectangle(frame, (mx, my), (width - mx, height - my), color, 1)


def draw_crosshair(frame: np.ndarray, color: Color = GRID_COLOR_CENTER, size: int = 15) -> None:
    """Small crosshair at the exact centre of the frame."""
    height, width = frame.shape[:2]
    cx, cy = width // 2, height // 2
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 2)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 2)


def draw_border(frame: np.ndarray, color: Color, thickness: int = 4) -> None:
    """Full-frame border, used to signal alert states."""
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), color, thickness)


def draw_watermark(
    frame: np.ndarray,
    text: str = "LookThePerson",
    color: Color = (150, 150, 150),
    scale: float = 0.45,
) -> None:
    """Discreet corner watermark for exported footage."""
    height, width = frame.shape[:2]
    (tw, _th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.putText(
        frame, text, (width - tw - 14, height - 14),
        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Image adjustments
# ---------------------------------------------------------------------------

def apply_night_mode(frame: np.ndarray) -> np.ndarray:
    """Invert the image — the original 'night mode'."""
    return cv2.bitwise_not(frame)


# ---------------------------------------------------------------------------
# Detection drawing
# ---------------------------------------------------------------------------

HEAD_COLOR = (0, 255, 0)
HEAD_TOUCH_COLOR = (0, 180, 255)


def draw_head_circle(
    frame: np.ndarray,
    landmarks: Sequence[Any],
    width: int,
    height: int,
    touching: bool = False,
    color: Optional[Color] = None,
) -> None:
    """Circle around the detected head, highlighted when it is being touched."""
    from gestures.body_gestures import head_circle

    circle = head_circle(landmarks, width, height)
    if circle is None:
        return

    cx, cy, radius, _nx, _ny, _nr = circle
    stroke = color or (HEAD_TOUCH_COLOR if touching else HEAD_COLOR)
    cv2.circle(frame, (cx, cy), radius, stroke, 2, cv2.LINE_AA)
    cv2.line(
        frame,
        (cx - radius, int(cy + radius * 0.15)),
        (cx + radius, int(cy + radius * 0.15)),
        stroke, 1, cv2.LINE_AA,
    )


def draw_corner_box(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    color: Color,
    thickness: int = 2,
    corner_length: int = 18,
) -> None:
    """
    Bounding box drawn as four corner brackets.

    Reads as a targeting reticle and, unlike a full rectangle, does not hide
    the edges of whatever it is framing.
    """
    x, y, w, h = box
    x2, y2 = x + w, y + h
    length = min(corner_length, max(4, w // 3), max(4, h // 3))

    for (px, py), (dx, dy) in (
        ((x, y), (1, 1)), ((x2, y), (-1, 1)),
        ((x, y2), (1, -1)), ((x2, y2), (-1, -1)),
    ):
        cv2.line(frame, (px, py), (px + dx * length, py), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (px, py), (px, py + dy * length), color, thickness, cv2.LINE_AA)


def draw_bounding_boxes(
    frame: np.ndarray,
    result: Any,
    width: int,
    height: int,
    color: Color = (0, 255, 255),
    label_prefix: str = "",
    style: str = "corner",
    theme: Optional[Theme] = None,
) -> None:
    """
    Draw every detection in a MediaPipe result.

    With a *theme*, each category gets its own stable colour; otherwise the
    single *color* is used for all of them.
    """
    if not result or not getattr(result, "detections", None):
        return

    for detection in result.detections:
        box = detection.bounding_box
        x, y = int(box.origin_x), int(box.origin_y)
        w, h = int(box.width), int(box.height)

        label = ""
        stroke = color
        if detection.categories:
            category = detection.categories[0]
            name = category.category_name or "?"
            label = f"{label_prefix}{name} {category.score:.0%}"
            if theme is not None:
                stroke = theme.color_for_name(name)

        if style == "corner":
            draw_corner_box(frame, (x, y, w, h), stroke)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), stroke, 2)

        if not label:
            continue

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        # Flip the label below the box when it would fall off the top edge.
        label_y = y - 6 if y - th - 8 >= 0 else y + h + th + 6
        cv2.rectangle(
            frame, (x, label_y - th - 5), (x + tw + 8, label_y + 3), stroke, -1,
        )
        cv2.putText(
            frame, label, (x + 4, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA,
        )
