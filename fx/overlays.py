"""
Drawing overlays for LookThePerson.

Things drawn *on top* of the frame rather than transformations of it:
skeleton styles, motion trails, particles, heat maps, radar and glow.

Everything draws in place on the supplied BGR frame.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np

__all__ = [
    "draw_trail",
    "draw_glow_line",
    "MotionHeatmap",
    "ParticleSystem",
    "draw_radar",
    "draw_progress_ring",
    "draw_skeleton_styled",
    "draw_motion_blur_ghost",
    "draw_scan_line",
    "draw_landmark_ids",
    "SKELETON_STYLES",
]

Color = Tuple[int, int, int]

SKELETON_STYLES = ("classic", "glow", "dots", "thick", "wire", "neon", "bones")


# ---------------------------------------------------------------------------
# Lines and skeletons
# ---------------------------------------------------------------------------

def draw_glow_line(
    frame: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Color,
    thickness: int = 3,
    glow: int = 3,
) -> None:
    """
    Line with a soft halo.

    Drawn as progressively wider, darker strokes underneath the core line —
    much cheaper than blurring a separate layer every frame.
    """
    for step in range(glow, 0, -1):
        faded = tuple(int(c * (0.20 + 0.1 * (glow - step))) for c in color)
        cv2.line(frame, start, end, faded, thickness + step * 3, cv2.LINE_AA)
    cv2.line(frame, start, end, color, thickness, cv2.LINE_AA)


def draw_skeleton_styled(
    frame: np.ndarray,
    points: Sequence[Tuple[int, int, bool]],
    connections: Sequence[Tuple[int, int]],
    color: Color,
    style: str = "classic",
    thickness: int = 3,
    joint_color: Optional[Color] = None,
    radius: int = 5,
) -> None:
    """
    Draw a skeleton in one of several visual styles.

    Args:
        points: ``(x, y, visible)`` per landmark, already in pixel space.
        connections: index pairs to join.
        style: one of :data:`SKELETON_STYLES`.
    """
    joint_color = joint_color or color

    if style == "dots":
        for x, y, visible in points:
            if visible:
                cv2.circle(frame, (x, y), radius, joint_color, -1, cv2.LINE_AA)
        return

    for start_index, end_index in connections:
        if start_index >= len(points) or end_index >= len(points):
            continue
        x1, y1, v1 = points[start_index]
        x2, y2, v2 = points[end_index]
        if not (v1 and v2):
            continue

        if style == "glow" or style == "neon":
            draw_glow_line(frame, (x1, y1), (x2, y2), color, thickness, glow=3)
        elif style == "thick":
            cv2.line(frame, (x1, y1), (x2, y2), color, thickness + 4, cv2.LINE_AA)
        elif style == "wire":
            cv2.line(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        elif style == "bones":
            _draw_bone(frame, (x1, y1), (x2, y2), color, thickness)
        else:  # classic
            cv2.line(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    if style == "wire":
        return

    for x, y, visible in points:
        if not visible:
            continue
        if style == "neon":
            cv2.circle(frame, (x, y), radius + 3, tuple(int(c * 0.4) for c in joint_color), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), radius, joint_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), radius + 2, (20, 20, 20), 1, cv2.LINE_AA)


def _draw_bone(
    frame: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Color,
    thickness: int,
) -> None:
    """Tapered segment that reads as an anatomical bone."""
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    if length < 2:
        return
    angle = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))
    center = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
    cv2.ellipse(
        frame, center, (int(length / 2), max(2, thickness)),
        angle, 0, 360, color, -1, cv2.LINE_AA,
    )
    cv2.circle(frame, start, thickness + 2, color, -1, cv2.LINE_AA)
    cv2.circle(frame, end, thickness + 2, color, -1, cv2.LINE_AA)


def draw_trail(
    frame: np.ndarray,
    path: Sequence[Tuple[float, float, float]],
    color: Color,
    width: int,
    height: int,
    max_thickness: int = 6,
    fade: bool = True,
) -> None:
    """
    Draw a motion trail from a normalized path.

    Older points are thinner and dimmer, which gives the trail direction
    without needing an arrowhead.
    """
    if len(path) < 2:
        return

    count = len(path)
    for index in range(1, count):
        x1 = int(path[index - 1][0] * width)
        y1 = int(path[index - 1][1] * height)
        x2 = int(path[index][0] * width)
        y2 = int(path[index][1] * height)

        ratio = index / count
        thickness = max(1, int(max_thickness * ratio))
        stroke = tuple(int(c * (0.25 + 0.75 * ratio)) for c in color) if fade else color
        cv2.line(frame, (x1, y1), (x2, y2), stroke, thickness, cv2.LINE_AA)


def draw_motion_blur_ghost(
    frame: np.ndarray,
    previous: Optional[np.ndarray],
    strength: float = 0.55,
) -> np.ndarray:
    """
    Blend the previous frame into this one for a motion-echo effect.

    Returns the blended frame; the caller stores it as the next *previous*.
    """
    if previous is None or previous.shape != frame.shape:
        return frame.copy()
    return cv2.addWeighted(frame, 1.0 - strength, previous, strength, 0)


def draw_scan_line(
    frame: np.ndarray,
    position: float,
    color: Color = (0, 255, 180),
    thickness: int = 2,
    glow: bool = True,
) -> None:
    """
    Horizontal scanner sweep at *position* (0..1 down the frame).
    """
    height, width = frame.shape[:2]
    y = int(max(0.0, min(1.0, position)) * (height - 1))

    if glow:
        overlay = frame.copy()
        span = 26
        cv2.rectangle(overlay, (0, max(0, y - span)), (width, y), tuple(int(c * 0.5) for c in color), -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.line(frame, (0, y), (width, y), color, thickness, cv2.LINE_AA)


def draw_landmark_ids(
    frame: np.ndarray,
    points: Sequence[Tuple[int, int, bool]],
    color: Color = (200, 200, 200),
    step: int = 1,
) -> None:
    """Label each visible landmark with its index — a debugging aid."""
    for index in range(0, len(points), max(1, step)):
        x, y, visible = points[index]
        if visible:
            cv2.putText(
                frame, str(index), (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA,
            )


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

class MotionHeatmap:
    """
    Accumulates where in the frame movement happens.

    The map is kept at reduced resolution — motion heat is inherently blurry,
    so a small buffer is both faster and visually identical once upscaled.
    """

    def __init__(self, width: int = 96, height: int = 54, decay: float = 0.985):
        self.width = max(8, width)
        self.height = max(8, height)
        self.decay = decay
        self._map = np.zeros((self.height, self.width), dtype=np.float32)
        self._samples = 0

    def add_point(self, x: float, y: float, weight: float = 1.0, radius: int = 3) -> None:
        """Add heat at a normalized ``(x, y)`` position."""
        px = int(max(0.0, min(1.0, x)) * (self.width - 1))
        py = int(max(0.0, min(1.0, y)) * (self.height - 1))
        cv2.circle(self._map, (px, py), radius, float(weight), -1)
        self._samples += 1

    def add_landmarks(self, landmarks: Sequence[Any], weight: float = 1.0) -> None:
        """Add heat at every visible landmark."""
        for landmark in landmarks:
            if getattr(landmark, "visibility", 1.0) >= 0.4:
                self.add_point(landmark.x, landmark.y, weight)

    def decay_step(self) -> None:
        """Fade the whole map — call once per frame."""
        self._map *= self.decay

    def render(
        self,
        frame: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Blend the heat map over *frame* and return the result."""
        if self._map.max() <= 1e-6:
            return frame

        normalized = self._map / self._map.max()
        height, width = frame.shape[:2]
        upscaled = cv2.resize(normalized, (width, height), interpolation=cv2.INTER_CUBIC)
        colored = cv2.applyColorMap((upscaled * 255).astype(np.uint8), colormap)

        # Keep cold areas transparent so the video still reads through.
        mask = np.clip(upscaled * 1.8, 0.0, 1.0)[:, :, None] * alpha
        return (frame * (1.0 - mask) + colored * mask).astype(np.uint8)

    @property
    def sample_count(self) -> int:
        return self._samples

    def hotspot(self) -> Optional[Tuple[float, float]]:
        """Normalized position of the hottest cell, or None when empty."""
        if self._map.max() <= 1e-6:
            return None
        index = int(np.argmax(self._map))
        y, x = divmod(index, self.width)
        return x / (self.width - 1), y / (self.height - 1)

    def coverage(self, threshold: float = 0.1) -> float:
        """Fraction of the frame that has seen meaningful motion, 0..1."""
        if self._map.max() <= 1e-6:
            return 0.0
        normalized = self._map / self._map.max()
        return float((normalized > threshold).sum()) / normalized.size

    def clear(self) -> None:
        self._map[:] = 0.0
        self._samples = 0


# ---------------------------------------------------------------------------
# Particles
# ---------------------------------------------------------------------------

@dataclass
class Particle:
    """One particle in the effect system."""

    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    color: Color
    size: int = 3

    @property
    def alive(self) -> bool:
        return self.life > 0.0

    @property
    def fade(self) -> float:
        """Remaining life as 0..1."""
        return max(0.0, self.life / self.max_life) if self.max_life > 0 else 0.0


class ParticleSystem:
    """
    Lightweight particle emitter used for gesture feedback and celebrations.

    Positions are normalized so the system is resolution-independent.
    """

    def __init__(self, max_particles: int = 220, gravity: float = 0.35):
        self.max_particles = max_particles
        self.gravity = gravity
        self._particles: List[Particle] = []
        self._rng = random.Random(0xA11CE)

    def emit(
        self,
        x: float,
        y: float,
        count: int = 20,
        color: Color = (0, 220, 255),
        speed: float = 0.35,
        life: float = 1.0,
        spread: float = math.tau,
    ) -> None:
        """Spawn *count* particles at a normalized position."""
        room = self.max_particles - len(self._particles)
        for _ in range(min(count, max(0, room))):
            angle = self._rng.uniform(0.0, spread)
            velocity = self._rng.uniform(0.25, 1.0) * speed
            jitter = self._rng.uniform(0.75, 1.25)
            self._particles.append(Particle(
                x=x, y=y,
                vx=math.cos(angle) * velocity,
                vy=math.sin(angle) * velocity,
                life=life * jitter,
                max_life=life * jitter,
                color=color,
                size=self._rng.randint(2, 4),
            ))

    def burst(self, x: float, y: float, color: Color = (0, 220, 255)) -> None:
        """A dense celebratory burst — used when a rep or gesture lands."""
        self.emit(x, y, count=45, color=color, speed=0.6, life=1.2)

    def update(self, dt: float) -> None:
        """Advance the simulation by *dt* seconds."""
        for particle in self._particles:
            particle.life -= dt
            particle.vy += self.gravity * dt
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt
        self._particles = [p for p in self._particles if p.alive and -0.2 < p.y < 1.2]

    def draw(self, frame: np.ndarray) -> None:
        """Render every live particle onto the frame."""
        height, width = frame.shape[:2]
        for particle in self._particles:
            px = int(particle.x * width)
            py = int(particle.y * height)
            if not (0 <= px < width and 0 <= py < height):
                continue
            fade = particle.fade
            color = tuple(int(c * fade) for c in particle.color)
            cv2.circle(frame, (px, py), max(1, int(particle.size * fade)), color, -1, cv2.LINE_AA)

    @property
    def count(self) -> int:
        return len(self._particles)

    def clear(self) -> None:
        self._particles.clear()


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------

def draw_radar(
    frame: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    targets: Sequence[Tuple[float, float]],
    sweep_angle: float = 0.0,
    color: Color = (0, 255, 160),
) -> None:
    """
    Radar scope with a rotating sweep and plotted targets.

    *targets* are ``(dx, dy)`` offsets in -1..1 relative to the scope centre.
    """
    cv2.circle(frame, center, radius, color, 1, cv2.LINE_AA)
    cv2.circle(frame, center, radius * 2 // 3, tuple(int(c * 0.6) for c in color), 1, cv2.LINE_AA)
    cv2.circle(frame, center, radius // 3, tuple(int(c * 0.4) for c in color), 1, cv2.LINE_AA)
    cv2.line(frame, (center[0] - radius, center[1]), (center[0] + radius, center[1]),
             tuple(int(c * 0.35) for c in color), 1)
    cv2.line(frame, (center[0], center[1] - radius), (center[0], center[1] + radius),
             tuple(int(c * 0.35) for c in color), 1)

    end = (
        int(center[0] + math.cos(sweep_angle) * radius),
        int(center[1] + math.sin(sweep_angle) * radius),
    )
    cv2.line(frame, center, end, color, 2, cv2.LINE_AA)

    for dx, dy in targets:
        tx = int(center[0] + max(-1.0, min(1.0, dx)) * radius)
        ty = int(center[1] + max(-1.0, min(1.0, dy)) * radius)
        cv2.circle(frame, (tx, ty), 4, color, -1, cv2.LINE_AA)


def draw_progress_ring(
    frame: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    progress: float,
    color: Color = (0, 220, 255),
    background: Color = (60, 60, 60),
    thickness: int = 6,
    label: str = "",
) -> None:
    """
    Circular progress indicator, filling clockwise from the top.

    Used for rep depth, gesture hold timers and countdowns.
    """
    progress = max(0.0, min(1.0, progress))
    cv2.circle(frame, center, radius, background, thickness, cv2.LINE_AA)
    if progress > 0:
        cv2.ellipse(
            frame, center, (radius, radius), -90, 0, int(360 * progress),
            color, thickness, cv2.LINE_AA,
        )
    if label:
        size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(
            frame, label,
            (center[0] - size[0] // 2, center[1] + size[1] // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )
