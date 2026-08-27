"""
Facial metrics for LookThePerson.

Quantities derived from the 478-point face mesh:

* Eye Aspect Ratio (EAR) and blink detection
* PERCLOS-based drowsiness estimation
* Mouth Aspect Ratio, yawn detection
* Head pose (yaw / pitch / roll) from facial geometry
* Attention and screen-time tracking from gaze

These support the drowsiness monitor, the focus timer and the attention HUD.
All ratios are dimensionless, normalised by face size so they hold at any
distance from the camera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.filters import Cooldown, EdgeDetector, ExponentialFilter, RingBuffer
from core.geometry import clamp, distance, midpoint

__all__ = [
    "FaceIndex",
    "eye_aspect_ratio",
    "mouth_aspect_ratio",
    "head_pose",
    "BlinkDetector",
    "DrowsinessMonitor",
    "AttentionTracker",
    "face_size",
]


class FaceIndex:
    """Landmark indices in the MediaPipe 478-point face mesh."""

    # Eye corners and lids (left/right are the subject's own sides)
    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    LEFT_EYE_TOP_2 = 158
    LEFT_EYE_BOTTOM_2 = 153

    RIGHT_EYE_OUTER = 263
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    RIGHT_EYE_TOP_2 = 385
    RIGHT_EYE_BOTTOM_2 = 380

    # Irises (only present when refine_landmarks / 478 points are available)
    LEFT_IRIS = 468
    RIGHT_IRIS = 473

    # Mouth
    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    UPPER_LIP = 0
    LOWER_LIP = 17

    # Structure
    NOSE_TIP = 1
    CHIN = 152
    FOREHEAD = 10
    LEFT_CHEEK = 234
    RIGHT_CHEEK = 454

    REFINED_COUNT = 478


F = FaceIndex


# ---------------------------------------------------------------------------
# Basic ratios
# ---------------------------------------------------------------------------

def face_size(landmarks: Sequence[Any]) -> float:
    """Forehead-to-chin distance — the normalisation scale for every ratio."""
    if len(landmarks) <= F.CHIN:
        return 0.0
    return distance(landmarks[F.FOREHEAD], landmarks[F.CHIN])


def eye_aspect_ratio(landmarks: Sequence[Any], side: str = "left") -> Optional[float]:
    """
    Eye Aspect Ratio: eye height divided by eye width.

    Roughly 0.3 for an open eye and under 0.15 when closed. Two vertical
    measurements are averaged, which is markedly more stable than one.
    """
    if side == "left":
        outer, inner = F.LEFT_EYE_OUTER, F.LEFT_EYE_INNER
        top1, bottom1 = F.LEFT_EYE_TOP, F.LEFT_EYE_BOTTOM
        top2, bottom2 = F.LEFT_EYE_TOP_2, F.LEFT_EYE_BOTTOM_2
    else:
        outer, inner = F.RIGHT_EYE_OUTER, F.RIGHT_EYE_INNER
        top1, bottom1 = F.RIGHT_EYE_TOP, F.RIGHT_EYE_BOTTOM
        top2, bottom2 = F.RIGHT_EYE_TOP_2, F.RIGHT_EYE_BOTTOM_2

    needed = max(outer, inner, top1, bottom1, top2, bottom2)
    if len(landmarks) <= needed:
        return None

    width = distance(landmarks[outer], landmarks[inner])
    if width < 1e-6:
        return None
    height = (
        distance(landmarks[top1], landmarks[bottom1])
        + distance(landmarks[top2], landmarks[bottom2])
    ) / 2.0
    return height / width


def mouth_aspect_ratio(landmarks: Sequence[Any]) -> Optional[float]:
    """Mouth opening relative to its width — drives yawn detection."""
    if len(landmarks) <= F.MOUTH_RIGHT:
        return None
    width = distance(landmarks[F.MOUTH_LEFT], landmarks[F.MOUTH_RIGHT])
    if width < 1e-6:
        return None
    height = distance(landmarks[F.MOUTH_TOP], landmarks[F.MOUTH_BOTTOM])
    return height / width


def head_pose(landmarks: Sequence[Any]) -> Dict[str, float]:
    """
    Approximate head orientation in degrees.

    Returns ``yaw`` (turn left/right), ``pitch`` (nod up/down) and ``roll``
    (tilt). This is a geometric approximation from facial proportions, not a
    full PnP solve — good enough for attention and gesture work, not for
    metrology.
    """
    result = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    if len(landmarks) <= F.RIGHT_CHEEK:
        return result

    left_cheek = landmarks[F.LEFT_CHEEK]
    right_cheek = landmarks[F.RIGHT_CHEEK]
    nose = landmarks[F.NOSE_TIP]
    chin = landmarks[F.CHIN]
    forehead = landmarks[F.FOREHEAD]

    # Roll: tilt of the cheek line.
    result["roll"] = math.degrees(
        math.atan2(right_cheek.y - left_cheek.y, right_cheek.x - left_cheek.x)
    )

    # Yaw: how far the nose sits from the midpoint of the cheeks.
    cheek_mid = midpoint(left_cheek, right_cheek)
    cheek_width = distance(left_cheek, right_cheek)
    if cheek_width > 1e-6:
        offset = (nose.x - cheek_mid.x) / cheek_width
        result["yaw"] = clamp(offset * 180.0, -90.0, 90.0)

    # Pitch: nose height within the forehead-chin span.
    face_height = distance(forehead, chin)
    if face_height > 1e-6:
        vertical_mid = (forehead.y + chin.y) / 2.0
        offset = (nose.y - vertical_mid) / face_height
        result["pitch"] = clamp(offset * 180.0, -90.0, 90.0)

    return result


# ---------------------------------------------------------------------------
# Blinks
# ---------------------------------------------------------------------------

class BlinkDetector:
    """
    Counts blinks from the eye aspect ratio.

    A blink is a closure that lasts between *min_duration* and *max_duration* —
    shorter is tracking noise, longer is a deliberate eye close (which the
    drowsiness monitor cares about instead).
    """

    def __init__(
        self,
        closed_threshold: float = 0.17,
        open_threshold: float = 0.23,
        min_duration: float = 0.05,
        max_duration: float = 0.5,
    ):
        self.closed_threshold = closed_threshold
        self.open_threshold = open_threshold
        self.min_duration = min_duration
        self.max_duration = max_duration

        self._closed = False
        self._closed_since: Optional[float] = None
        self._count = 0
        self._blink_times: List[float] = []
        self._smoother = ExponentialFilter(alpha=0.5)
        self._last_ear = 0.3

    def update(self, ear: Optional[float], now: float) -> bool:
        """Feed one EAR sample; returns True on the frame a blink completes."""
        if ear is None:
            return False
        smoothed = self._smoother.update(ear)
        self._last_ear = smoothed

        if not self._closed and smoothed < self.closed_threshold:
            self._closed = True
            self._closed_since = now
            return False

        if self._closed and smoothed > self.open_threshold:
            self._closed = False
            duration = now - (self._closed_since or now)
            self._closed_since = None
            if self.min_duration <= duration <= self.max_duration:
                self._count += 1
                self._blink_times.append(now)
                return True
        return False

    @property
    def count(self) -> int:
        return self._count

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def ear(self) -> float:
        """Most recent smoothed EAR."""
        return self._last_ear

    def closed_duration(self, now: float) -> float:
        """How long the eyes have been shut right now."""
        return 0.0 if self._closed_since is None else now - self._closed_since

    def blink_rate(self, now: float, window: float = 60.0) -> float:
        """
        Blinks per minute over the recent window.

        A typical resting rate is 15-20; markedly lower often accompanies
        screen concentration, markedly higher can indicate eye strain.
        """
        recent = [t for t in self._blink_times if now - t <= window]
        elapsed = min(window, now - self._blink_times[0]) if self._blink_times else 0.0
        if elapsed < 1.0:
            return 0.0
        return len(recent) / (elapsed / 60.0)

    def reset(self) -> None:
        self._closed = False
        self._closed_since = None
        self._count = 0
        self._blink_times.clear()
        self._smoother.reset()


# ---------------------------------------------------------------------------
# Drowsiness
# ---------------------------------------------------------------------------

@dataclass
class DrowsinessReport:
    """Drowsiness assessment for one frame."""

    perclos: float = 0.0          # fraction of recent time with eyes closed
    long_closure: float = 0.0     # current continuous closure, seconds
    yawning: bool = False
    yawn_count: int = 0
    level: str = "alerta"         # alerta | cansancio | somnolencia
    alert: bool = False

    @property
    def severity(self) -> float:
        """0..1 combined drowsiness score."""
        return clamp(self.perclos * 2.0 + min(self.long_closure / 2.0, 1.0) * 0.5, 0.0, 1.0)


class DrowsinessMonitor:
    """
    Estimates fatigue from eye closure and yawning.

    PERCLOS — the percentage of time the eyes are closed over a rolling window
    — is the standard measure used in driver-monitoring research, and it is
    what this class tracks, alongside sustained closures and yawn frequency.
    """

    def __init__(
        self,
        window_seconds: float = 60.0,
        perclos_warn: float = 0.15,
        perclos_alert: float = 0.30,
        closure_alert: float = 1.2,
        yawn_threshold: float = 0.6,
    ):
        self.window_seconds = window_seconds
        self.perclos_warn = perclos_warn
        self.perclos_alert = perclos_alert
        self.closure_alert = closure_alert
        self.yawn_threshold = yawn_threshold

        self._samples: List[Tuple[float, bool]] = []   # (timestamp, closed)
        self._yawn_edge = EdgeDetector()
        self._yawn_count = 0
        self._yawn_cooldown = Cooldown(4.0)
        self._alert_cooldown = Cooldown(20.0)

    def update(
        self,
        ear: Optional[float],
        mar: Optional[float],
        now: float,
        closed_threshold: float = 0.17,
    ) -> DrowsinessReport:
        """Feed eye and mouth ratios; returns the current assessment."""
        report = DrowsinessReport()
        if ear is None:
            return report

        closed = ear < closed_threshold
        self._samples.append((now, closed))
        cutoff = now - self.window_seconds
        self._samples = [s for s in self._samples if s[0] >= cutoff]

        if self._samples:
            report.perclos = sum(1 for _t, c in self._samples if c) / len(self._samples)

        # Longest run of consecutive closed samples ending now.
        closure = 0.0
        for timestamp, is_closed in reversed(self._samples):
            if not is_closed:
                break
            closure = now - timestamp
        report.long_closure = closure

        if mar is not None and mar > self.yawn_threshold:
            report.yawning = True
            if self._yawn_edge.rising(True) and self._yawn_cooldown.trigger(now):
                self._yawn_count += 1
        else:
            self._yawn_edge.update(False)
        report.yawn_count = self._yawn_count

        if report.perclos >= self.perclos_alert or closure >= self.closure_alert:
            report.level = "somnolencia"
        elif report.perclos >= self.perclos_warn or self._yawn_count >= 3:
            report.level = "cansancio"
        else:
            report.level = "alerta"

        if report.level == "somnolencia" and self._alert_cooldown.trigger(now):
            report.alert = True
        return report

    def reset(self) -> None:
        self._samples.clear()
        self._yawn_count = 0
        self._yawn_edge = EdgeDetector()


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class AttentionTracker:
    """
    Measures how much of the time someone is looking at the screen.

    Attention is credited when the head is roughly forward-facing and the gaze
    is near-centred. Powers the focus timer and the study/work session report.
    """

    def __init__(
        self,
        yaw_tolerance: float = 25.0,
        pitch_tolerance: float = 22.0,
        gaze_tolerance: float = 0.55,
        history: int = 300,
    ):
        self.yaw_tolerance = yaw_tolerance
        self.pitch_tolerance = pitch_tolerance
        self.gaze_tolerance = gaze_tolerance

        self._attentive_seconds = 0.0
        self._distracted_seconds = 0.0
        self._last_update: Optional[float] = None
        self._attentive = False
        self._history = RingBuffer(history)
        self._distraction_count = 0
        self._edge = EdgeDetector(initial=True)
        self._longest_focus = 0.0
        self._focus_since: Optional[float] = None

    def update(
        self,
        pose: Optional[Dict[str, float]],
        gaze: Optional[Tuple[float, float]],
        now: float,
    ) -> bool:
        """
        Feed head pose and gaze; returns whether the user is attentive now.

        Passing ``None`` for *pose* (no face detected) counts as distracted.
        """
        attentive = False
        if pose is not None:
            facing = (
                abs(pose.get("yaw", 0.0)) <= self.yaw_tolerance
                and abs(pose.get("pitch", 0.0)) <= self.pitch_tolerance
            )
            looking = True
            if gaze is not None:
                looking = (
                    abs(gaze[0]) <= self.gaze_tolerance
                    and abs(gaze[1]) <= self.gaze_tolerance
                )
            attentive = facing and looking

        if self._last_update is not None:
            elapsed = max(0.0, now - self._last_update)
            if attentive:
                self._attentive_seconds += elapsed
            else:
                self._distracted_seconds += elapsed
        self._last_update = now

        transition = self._edge.update(attentive)
        if transition == "falling":
            self._distraction_count += 1
            if self._focus_since is not None:
                self._longest_focus = max(self._longest_focus, now - self._focus_since)
            self._focus_since = None
        elif transition == "rising":
            self._focus_since = now
        elif attentive and self._focus_since is None:
            self._focus_since = now

        self._attentive = attentive
        self._history.append(1.0 if attentive else 0.0)
        return attentive

    @property
    def is_attentive(self) -> bool:
        return self._attentive

    @property
    def attention_ratio(self) -> float:
        """Share of session time spent attentive, 0..1."""
        total = self._attentive_seconds + self._distracted_seconds
        return self._attentive_seconds / total if total > 0 else 0.0

    @property
    def recent_ratio(self) -> float:
        """Attention over the recent history window only."""
        return self._history.mean()

    def current_focus_streak(self, now: float) -> float:
        """Seconds of uninterrupted attention right now."""
        if not self._attentive or self._focus_since is None:
            return 0.0
        return now - self._focus_since

    def stats(self, now: Optional[float] = None) -> Dict[str, Any]:
        longest = self._longest_focus
        if now is not None and self._focus_since is not None:
            longest = max(longest, now - self._focus_since)
        return {
            "attentive_seconds": round(self._attentive_seconds, 1),
            "distracted_seconds": round(self._distracted_seconds, 1),
            "attention_ratio": round(self.attention_ratio, 3),
            "distractions": self._distraction_count,
            "longest_focus": round(longest, 1),
        }

    def reset(self) -> None:
        self._attentive_seconds = 0.0
        self._distracted_seconds = 0.0
        self._last_update = None
        self._distraction_count = 0
        self._history.clear()
        self._longest_focus = 0.0
        self._focus_since = None
