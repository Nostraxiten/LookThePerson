"""
Reusable HUD widgets for LookThePerson.

Small, composable drawing primitives — panels, badges, bars, sparklines,
toasts — that the HUD and the modes build their interfaces from. Keeping them
here means every panel in the app shares one look and one set of conventions.

All coordinates are in pixels and all drawing happens in place.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.theme import Theme

__all__ = [
    "FONT",
    "text_size",
    "draw_text",
    "draw_panel",
    "draw_badge",
    "draw_bar",
    "draw_sparkline",
    "draw_key_hint",
    "draw_table",
    "Toast",
    "ToastManager",
    "truncate",
]

FONT = cv2.FONT_HERSHEY_SIMPLEX
Color = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def text_size(text: str, scale: float = 0.5, thickness: int = 1) -> Tuple[int, int]:
    """Pixel ``(width, height)`` of rendered text."""
    (width, height), _baseline = cv2.getTextSize(text, FONT, scale, thickness)
    return width, height


def truncate(text: str, max_chars: int) -> str:
    """Shorten with an ellipsis so long labels cannot break a panel's layout."""
    if max_chars <= 1 or len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def draw_text(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Color,
    scale: float = 0.5,
    thickness: int = 1,
    shadow: bool = True,
) -> Tuple[int, int]:
    """
    Draw text with an optional drop shadow and return its size.

    The shadow is what keeps white HUD text readable over a bright background —
    without it the overlay disappears against a window or a lamp.
    """
    if shadow:
        cv2.putText(frame, text, (position[0] + 1, position[1] + 1),
                    FONT, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, position, FONT, scale, color, thickness, cv2.LINE_AA)
    return text_size(text, scale, thickness)


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------

def draw_panel(
    frame: np.ndarray,
    top_left: Tuple[int, int],
    size: Tuple[int, int],
    theme: Theme,
    alpha: float = 0.72,
    border: bool = True,
    title: str = "",
) -> Tuple[int, int]:
    """
    Draw a translucent panel and return the content origin (inside padding).

    Clipped to the frame, so a panel positioned near an edge degrades
    gracefully instead of raising.
    """
    x, y = top_left
    width, height = size
    frame_h, frame_w = frame.shape[:2]

    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame_w, x + width), min(frame_h, y + height)
    if x1 <= x0 or y1 <= y0:
        return x0 + 12, y0 + 12

    region = frame[y0:y1, x0:x1]
    backdrop = np.full_like(region, theme.background)
    cv2.addWeighted(backdrop, alpha, region, 1.0 - alpha, 0, region)

    if border:
        cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), theme.accent, 1)

    content_y = y0 + 12
    if title:
        draw_text(frame, title, (x0 + 12, y0 + 22), theme.accent, 0.5, 1)
        cv2.line(frame, (x0 + 10, y0 + 30), (x1 - 10, y0 + 30),
                 tuple(int(c * 0.4) for c in theme.accent), 1)
        content_y = y0 + 48

    return x0 + 12, content_y


def draw_badge(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Color,
    text_color: Color = (12, 12, 12),
    scale: float = 0.42,
    padding: int = 6,
) -> int:
    """Filled rounded label. Returns the badge width in pixels."""
    width, height = text_size(text, scale, 1)
    x, y = position
    cv2.rectangle(
        frame, (x, y - height - padding), (x + width + padding * 2, y + padding),
        color, -1,
    )
    cv2.putText(frame, text, (x + padding, y), FONT, scale, text_color, 1, cv2.LINE_AA)
    return width + padding * 2


def draw_key_hint(
    frame: np.ndarray,
    key: str,
    description: str,
    position: Tuple[int, int],
    theme: Theme,
    active: Optional[bool] = None,
) -> None:
    """One row of the help panel: key badge, optional state, description."""
    x, y = position
    badge_width = draw_badge(frame, key.upper(), (x, y), theme.accent, (10, 10, 10), 0.38)

    offset = x + badge_width + 8
    if active is not None:
        state_color = theme.good if active else theme.text_dim
        draw_text(frame, "ON " if active else "OFF", (offset, y), state_color, 0.36, 1)
        offset += 34

    draw_text(frame, truncate(description, 30), (offset, y), theme.text, 0.38, 1)


def draw_table(
    frame: np.ndarray,
    rows: Sequence[Tuple[str, str]],
    origin: Tuple[int, int],
    theme: Theme,
    line_height: int = 18,
    key_width: int = 110,
    scale: float = 0.4,
) -> int:
    """
    Draw label/value rows and return the y coordinate after the last one.
    """
    x, y = origin
    for label, value in rows:
        draw_text(frame, truncate(label, 18), (x, y), theme.text_dim, scale, 1)
        draw_text(frame, str(value), (x + key_width, y), theme.text, scale, 1)
        y += line_height
    return y


# ---------------------------------------------------------------------------
# Data display
# ---------------------------------------------------------------------------

def draw_bar(
    frame: np.ndarray,
    origin: Tuple[int, int],
    size: Tuple[int, int],
    value: float,
    color: Color,
    background: Color = (48, 48, 48),
    border: Optional[Color] = None,
    label: str = "",
    text_color: Optional[Color] = None,
) -> None:
    """Horizontal progress/level bar; *value* is 0..1 and is clamped."""
    x, y = origin
    width, height = size
    value = max(0.0, min(1.0, value))

    cv2.rectangle(frame, (x, y), (x + width, y + height), background, -1)
    if value > 0:
        cv2.rectangle(frame, (x, y), (x + int(width * value), y + height), color, -1)
    if border:
        cv2.rectangle(frame, (x, y), (x + width, y + height), border, 1)
    if label:
        draw_text(frame, label, (x + 4, y + height - 4), text_color or (250, 250, 250), 0.36, 1)


def draw_sparkline(
    frame: np.ndarray,
    values: Sequence[float],
    origin: Tuple[int, int],
    size: Tuple[int, int],
    color: Color,
    background: Optional[Color] = None,
    baseline: Optional[float] = None,
    baseline_color: Optional[Color] = None,
) -> None:
    """
    Compact line chart of a value history.

    The vertical scale spans the data's own min and max, so the shape of the
    variation is always visible regardless of the absolute values. Pass
    *baseline* to draw a reference line (a target FPS, for instance).
    """
    x, y = origin
    width, height = size
    if width <= 2 or height <= 2:
        return

    if background is not None:
        cv2.rectangle(frame, (x, y), (x + width, y + height), background, -1)

    if len(values) < 2:
        return

    low = min(values)
    high = max(values)
    if baseline is not None:
        low = min(low, baseline)
        high = max(high, baseline)
    span = high - low
    if span < 1e-6:
        span = 1.0
        low -= 0.5

    def to_y(value: float) -> int:
        return int(y + height - (value - low) / span * height)

    if baseline is not None and baseline_color is not None:
        by = to_y(baseline)
        cv2.line(frame, (x, by), (x + width, by), baseline_color, 1)

    step = width / (len(values) - 1)
    points = np.array(
        [[int(x + index * step), to_y(value)] for index, value in enumerate(values)],
        dtype=np.int32,
    )
    cv2.polylines(frame, [points], False, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------

@dataclass
class Toast:
    """A transient on-screen message."""

    message: str
    level: str = "info"          # info | warn | danger | good
    created: float = 0.0
    duration: float = 2.6

    def age(self, now: float) -> float:
        return now - self.created

    def alive(self, now: float) -> bool:
        return self.age(now) < self.duration

    def opacity(self, now: float) -> float:
        """Fade in quickly, hold, then fade out over the last third."""
        age = self.age(now)
        if age < 0.15:
            return age / 0.15
        remaining = self.duration - age
        if remaining < 0.5:
            return max(0.0, remaining / 0.5)
        return 1.0


class ToastManager:
    """
    Queue of transient notifications drawn in a corner of the screen.

    Subscribing this to the ``ui.notify`` event is all it takes for any part of
    the app to surface a message without knowing anything about rendering.
    """

    def __init__(self, max_visible: int = 4, duration: float = 2.6):
        self.max_visible = max_visible
        self.duration = duration
        self._toasts: Deque[Toast] = deque(maxlen=12)

    def push(self, message: str, level: str = "info", now: Optional[float] = None) -> Toast:
        """Add a notification."""
        toast = Toast(
            message=message,
            level=level,
            created=now if now is not None else time.monotonic(),
            duration=self.duration,
        )
        self._toasts.append(toast)
        return toast

    def handle_event(self, event) -> None:
        """Event-bus handler for ``ui.notify`` events."""
        self.push(event.get("message", ""), event.get("level", "info"))

    def prune(self, now: float) -> None:
        while self._toasts and not self._toasts[0].alive(now):
            self._toasts.popleft()

    def visible(self, now: float) -> List[Toast]:
        self.prune(now)
        return list(self._toasts)[-self.max_visible:]

    def draw(self, frame: np.ndarray, theme: Theme, now: float) -> None:
        """Render the visible toasts, stacked upward from the bottom-right."""
        toasts = self.visible(now)
        if not toasts:
            return

        height, width = frame.shape[:2]
        y = height - 90

        colors = {
            "info": theme.accent, "good": theme.good,
            "warn": theme.warn, "danger": theme.danger,
        }

        for toast in reversed(toasts):
            opacity = toast.opacity(now)
            if opacity <= 0.01:
                continue

            color = colors.get(toast.level, theme.accent)
            text = truncate(toast.message, 42)
            tw, th = text_size(text, 0.46, 1)
            box_w = tw + 26
            box_h = th + 18
            x = width - box_w - 20

            if x < 0 or y - box_h < 0:
                break

            region = frame[y - box_h:y, x:x + box_w]
            if region.size:
                backdrop = np.full_like(region, theme.background)
                cv2.addWeighted(backdrop, 0.82 * opacity, region, 1.0 - 0.82 * opacity, 0, region)

            # Accent stripe on the left edge carries the severity.
            cv2.rectangle(frame, (x, y - box_h), (x + 4, y), color, -1)
            cv2.rectangle(frame, (x, y - box_h), (x + box_w - 1, y - 1),
                          tuple(int(c * opacity) for c in color), 1)
            draw_text(
                frame, text, (x + 14, y - 10),
                tuple(int(c * opacity + 20 * (1 - opacity)) for c in theme.text), 0.46, 1,
            )
            y -= box_h + 8

    def clear(self) -> None:
        self._toasts.clear()

    def __len__(self) -> int:
        return len(self._toasts)
