"""
Central application state for LookThePerson.

Two objects carry everything the app knows:

* :class:`AppState` — lives for the whole session: config, event bus, metrics,
  toggles, counters, the active mode.
* :class:`FrameContext` — rebuilt every frame and handed to each mode: the
  image, the raw detector results and the derived analytics for that instant.

Modes read from these instead of reaching back into the main loop, which is
what keeps them independent of one another.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.config import Config
from core.events import EventBus, Events
from core.geometry import is_finite_point, to_pixels, to_pixels_clamped
from core.metrics import FPSTracker, PerformanceMonitor, StageProfiler
from core.theme import Theme, get_theme

__all__ = ["FrameContext", "AppState"]


# ---------------------------------------------------------------------------
# Per-frame context
# ---------------------------------------------------------------------------

@dataclass
class FrameContext:
    """
    Everything known about a single frame.

    ``frame`` is the working BGR image: modes draw on it in place and may
    replace it wholesale (a filter mode swapping in a processed copy) by
    assigning to the attribute.
    """

    frame: Any                       # BGR numpy array
    width: int
    height: int
    timestamp_ms: int                # monotonic ms since app start, for MediaPipe
    now: float                       # time.monotonic() at frame start
    delta: float = 0.0               # seconds since the previous frame
    frame_index: int = 0

    # Raw detector output (MediaPipe result objects, or None when not run)
    pose_result: Any = None
    hand_result: Any = None
    face_mesh_result: Any = None
    face_detect_result: Any = None
    object_result: Any = None

    # Convenience views derived once per frame
    pose_landmarks: List[Any] = field(default_factory=list)   # list of persons
    hand_landmarks: List[Any] = field(default_factory=list)
    face_landmarks: List[Any] = field(default_factory=list)
    handedness: List[str] = field(default_factory=list)       # "Left" / "Right"

    # Derived analytics, filled by the pipeline before modes run
    body_center: Optional[Tuple[int, int]] = None
    body_gestures: Dict[str, bool] = field(default_factory=dict)
    hand_info: Dict[str, Any] = field(default_factory=dict)
    face_info: Dict[str, Any] = field(default_factory=dict)
    angles: Dict[str, float] = field(default_factory=dict)
    posture: Dict[str, Any] = field(default_factory=dict)
    motion: Dict[str, float] = field(default_factory=dict)

    # Free-form slot for modes to stash their own per-frame data
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_pose(self) -> bool:
        return bool(self.pose_landmarks)

    @property
    def has_hands(self) -> bool:
        return bool(self.hand_landmarks)

    @property
    def has_face(self) -> bool:
        return bool(self.face_landmarks)

    @property
    def person_count(self) -> int:
        return len(self.pose_landmarks)

    @property
    def hand_count(self) -> int:
        return len(self.hand_landmarks)

    @property
    def primary_pose(self) -> Optional[Any]:
        """Landmarks of the first detected person, or None."""
        return self.pose_landmarks[0] if self.pose_landmarks else None

    @property
    def primary_face(self) -> Optional[Any]:
        return self.face_landmarks[0] if self.face_landmarks else None

    def gesture(self, name: str) -> bool:
        """Whether a body gesture is active this frame."""
        return bool(self.body_gestures.get(name))

    def angle(self, name: str, default: float = 0.0) -> float:
        return self.angles.get(name, default)

    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 1.0

    # -- Pixel-space helpers ------------------------------------------------
    #
    # Modes convert landmarks to pixels constantly. Routing that through the
    # context keeps the NaN and range handling in one place instead of every
    # mode re-deriving `int(lm.x * ctx.width)` and inheriting its crashes.

    def px(self, point) -> Tuple[int, int]:
        """Landmark to pixel coordinates, safe for drawing (may be off-frame)."""
        return to_pixels(point, self.width, self.height)

    def px_clamped(self, point) -> Tuple[int, int]:
        """Landmark to pixel coordinates guaranteed to index into the frame."""
        return to_pixels_clamped(point, self.width, self.height)

    def usable(self, point) -> bool:
        """Whether a landmark can be converted to pixels at all."""
        return is_finite_point(point)

    def landmark_points(
        self,
        landmarks: Any,
        visibility: float = 0.0,
    ) -> List[Tuple[int, int, bool]]:
        """
        ``(x, y, visible)`` triples for a landmark list, in pixel space.

        A landmark that is non-finite is reported as not visible rather than
        dropped, so indices still line up with the skeleton connection table.
        """
        points: List[Tuple[int, int, bool]] = []
        for landmark in landmarks or ():
            ok = is_finite_point(landmark)
            x, y = self.px(landmark) if ok else (0, 0)
            seen = ok and getattr(landmark, "visibility", 1.0) >= visibility
            points.append((x, y, bool(seen)))
        return points

    def ensure_drawable(self) -> Any:
        """
        Guarantee ``frame`` is something OpenCV will draw into.

        OpenCV refuses any array that is not C-contiguous ("Layout of the
        output array img is incompatible with cv::Mat") and silently discards
        writes to a read-only one. A mode that crops or transposes the frame
        produces exactly that, so the pipeline normalises once per frame
        instead of every draw call defending itself.
        """
        frame = self.frame
        if frame is None:
            return frame
        flags = getattr(frame, "flags", None)
        if flags is not None and not (flags.c_contiguous and flags.writeable):
            import numpy as np
            self.frame = np.ascontiguousarray(frame)
        return self.frame


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

class AppState:
    """
    Long-lived application state shared by every subsystem.

    Toggles are stored here (rather than only in the key handler) so modes can
    query and set them by name without knowing which key owns them.
    """

    def __init__(self, config: Optional[Config] = None, bus: Optional[EventBus] = None):
        self.config = config or Config()
        self.bus = bus or EventBus()
        self.fps = FPSTracker()
        self.profiler = StageProfiler(enabled=True)
        self.performance = PerformanceMonitor(target_fps=float(self.config.camera.fps) * 0.8)

        self.started_at = time.monotonic()
        self.frame_index = 0
        self.running = True

        self._theme_name = self.config.display.theme
        self._theme = get_theme(self._theme_name)

        self.mode_name: str = self.config.mode
        self.previous_mode: Optional[str] = None

        self._toggles: Dict[str, bool] = {}
        self._counters: Dict[str, int] = {}
        self._notes: Dict[str, Any] = {}

        self.status_text: str = ""
        self.last_gesture: str = ""
        self.last_gesture_time: float = 0.0
        self.body_color: Tuple[int, int, int] = self._theme.skeleton
        self.recording: bool = False
        self.paused: bool = False

        self._init_toggles()

    # -- Toggles ------------------------------------------------------------

    def _init_toggles(self) -> None:
        display = self.config.display
        detection = self.config.detection
        self._toggles.update({
            "segmentation": detection.segmentation,
            "face_mesh": False,
            "face_detect": False,
            "object_detect": False,
            "skeleton": True,
            "bounding_boxes": True,
            "grid": display.show_grid,
            "help": display.show_help,
            "telemetry": display.show_hud,
            "night_mode": False,
            "mirror": self.config.camera.mirror,
            "fps_graph": display.show_fps_graph,
            "landmark_ids": display.show_landmark_ids,
            "trails": False,
            "heatmap": False,
            "debug": False,
            "gestures": self.config.gestures.enabled,
            "analytics": self.config.analytics.enabled,
        })

    def is_active(self, name: str) -> bool:
        return bool(self._toggles.get(name, False))

    def set_toggle(self, name: str, active: bool, announce: bool = True) -> bool:
        """Set a toggle; emits ``toggle.changed`` when the value actually moves."""
        previous = self._toggles.get(name)
        self._toggles[name] = bool(active)
        if announce and previous != bool(active):
            self.bus.emit(Events.TOGGLE_CHANGED, name=name, active=bool(active))
        return bool(active)

    def toggle(self, name: str) -> bool:
        """Flip a toggle and return its new value."""
        return self.set_toggle(name, not self.is_active(name))

    def active_toggles(self) -> Dict[str, bool]:
        return dict(self._toggles)

    def apply_toggles(self, values: Dict[str, bool], announce: bool = False) -> None:
        """Bulk-apply toggle values, as used by mode presets."""
        for name, active in values.items():
            self.set_toggle(name, active, announce=announce)

    # -- Theme --------------------------------------------------------------

    @property
    def theme(self) -> Theme:
        return self._theme

    @property
    def theme_name(self) -> str:
        return self._theme_name

    def set_theme(self, name: str) -> Theme:
        """Switch palette and retint the skeleton to match."""
        self._theme_name = name
        self._theme = get_theme(name)
        self.config.display.theme = name
        self.body_color = self._theme.skeleton
        self.bus.emit(Events.NOTIFY, message=f"Tema: {name}", level="info")
        return self._theme

    # -- Mode ---------------------------------------------------------------

    def set_mode(self, name: str) -> None:
        """Record a mode change; the ModeManager performs the actual switch."""
        if name == self.mode_name:
            return
        self.previous_mode = self.mode_name
        self.mode_name = name
        self.config.mode = name

    # -- Counters and notes -------------------------------------------------

    def increment(self, name: str, amount: int = 1) -> int:
        """Bump a named session counter and return the new total."""
        self._counters[name] = self._counters.get(name, 0) + amount
        return self._counters[name]

    def counter(self, name: str) -> int:
        return self._counters.get(name, 0)

    def counters(self) -> Dict[str, int]:
        return dict(self._counters)

    def note(self, key: str, value: Any) -> None:
        """Store arbitrary cross-subsystem state (last export path, etc.)."""
        self._notes[key] = value

    def get_note(self, key: str, default: Any = None) -> Any:
        return self._notes.get(key, default)

    # -- Status -------------------------------------------------------------

    def set_gesture(self, name: str, now: Optional[float] = None) -> None:
        """Record the most recent gesture for the HUD."""
        self.last_gesture = name
        self.last_gesture_time = now if now is not None else time.monotonic()
        self.increment("gestures_total")

    def gesture_age(self, now: Optional[float] = None) -> float:
        """Seconds since the last gesture, or a large number if none yet."""
        if not self.last_gesture:
            return float("inf")
        now = time.monotonic() if now is None else now
        return now - self.last_gesture_time

    @property
    def uptime(self) -> float:
        """Seconds since the session started."""
        return time.monotonic() - self.started_at

    def uptime_text(self) -> str:
        """Uptime formatted as ``M:SS`` or ``H:MM:SS``."""
        total = int(self.uptime)
        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    def notify(self, message: str, level: str = "info") -> None:
        """Raise a toast notification through the event bus."""
        self.bus.emit(Events.NOTIFY, message=message, level=level)

    def shutdown(self) -> None:
        self.running = False
        self.bus.emit(Events.SHUTDOWN, uptime=self.uptime, frames=self.frame_index)

    def summary(self) -> Dict[str, Any]:
        """Session summary, used by the exit report and session export."""
        return {
            "uptime_seconds": round(self.uptime, 1),
            "frames": self.frame_index,
            "average_fps": round(self.fps.average_fps, 1),
            "mode": self.mode_name,
            "theme": self._theme_name,
            "counters": self.counters(),
            "events": self.bus.counts(),
        }
