"""
Motion, balance and symmetry analysis for LookThePerson.

Derives movement quantities from the pose stream:

* :class:`MotionAnalyzer` — per-landmark speed, whole-body energy, direction.
* :class:`BalanceAnalyzer` — centre of mass, sway and a stability score.
* :func:`symmetry_report` — left/right comparison for form feedback.
* :class:`TrajectoryRecorder` — landmark trails for the visual overlay.
* :class:`StillnessDetector` — flags when someone stops moving.

Everything is normalized-image-space and numpy-free.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from analytics.angles import PoseLandmark as L
from analytics.angles import landmark_visible
from core.filters import ExponentialFilter, RingBuffer, VelocityTracker
from core.geometry import clamp, distance, midpoint, rolling_direction

__all__ = [
    "MotionAnalyzer",
    "BalanceAnalyzer",
    "TrajectoryRecorder",
    "StillnessDetector",
    "symmetry_report",
    "jump_height_estimate",
    "SEGMENT_MASS_FRACTIONS",
]

# Approximate body segment mass fractions (Winter, Biomechanics), used to
# weight the centre-of-mass estimate. Only landmarks we can see contribute.
SEGMENT_MASS_FRACTIONS: Dict[int, float] = {
    L.NOSE: 0.081,
    L.LEFT_SHOULDER: 0.078, L.RIGHT_SHOULDER: 0.078,
    L.LEFT_ELBOW: 0.016, L.RIGHT_ELBOW: 0.016,
    L.LEFT_WRIST: 0.006, L.RIGHT_WRIST: 0.006,
    L.LEFT_HIP: 0.142, L.RIGHT_HIP: 0.142,
    L.LEFT_KNEE: 0.043, L.RIGHT_KNEE: 0.043,
    L.LEFT_ANKLE: 0.014, L.RIGHT_ANKLE: 0.014,
}


# ---------------------------------------------------------------------------
# Motion
# ---------------------------------------------------------------------------

class MotionAnalyzer:
    """
    Tracks how fast the body and its parts are moving.

    "Energy" is the visibility-weighted mean landmark speed — a single number
    that rises with any movement, used to drive intensity meters, motion
    alarms and stillness detection.
    """

    TRACKED = (
        L.NOSE, L.LEFT_WRIST, L.RIGHT_WRIST, L.LEFT_ELBOW, L.RIGHT_ELBOW,
        L.LEFT_SHOULDER, L.RIGHT_SHOULDER, L.LEFT_HIP, L.RIGHT_HIP,
        L.LEFT_KNEE, L.RIGHT_KNEE, L.LEFT_ANKLE, L.RIGHT_ANKLE,
    )

    def __init__(self, window_seconds: float = 0.25, history: int = 180):
        self._trackers: Dict[int, VelocityTracker] = {
            index: VelocityTracker(window_seconds) for index in self.TRACKED
        }
        self._speeds: Dict[int, float] = {}
        self._energy = ExponentialFilter(alpha=0.25)
        self._energy_history = RingBuffer(history)
        self._peak_energy = 0.0
        self._center_tracker = VelocityTracker(window_seconds)
        self._last_direction = ""

    def update(self, landmarks: Sequence[Any], now: float) -> Dict[str, float]:
        """
        Feed one pose and return the motion metrics for this frame.

        Keys: ``energy``, ``peak_energy``, ``max_speed``, ``center_speed``,
        ``left_hand_speed``, ``right_hand_speed``.
        """
        if not landmarks or len(landmarks) < L.COUNT:
            return self._empty_metrics()

        total_speed = 0.0
        total_weight = 0.0
        max_speed = 0.0

        for index, tracker in self._trackers.items():
            landmark = landmarks[index]
            if not landmark_visible(landmark, 0.3):
                continue
            _vx, _vy, speed = tracker.update(landmark.x, landmark.y, now)
            self._speeds[index] = speed
            weight = getattr(landmark, "visibility", 1.0)
            total_speed += speed * weight
            total_weight += weight
            max_speed = max(max_speed, speed)

        raw_energy = total_speed / total_weight if total_weight > 0 else 0.0
        energy = self._energy.update(raw_energy)
        self._energy_history.append(energy)
        self._peak_energy = max(self._peak_energy, energy)

        center_speed = 0.0
        if landmark_visible(landmarks[L.LEFT_HIP], 0.3) and landmark_visible(landmarks[L.RIGHT_HIP], 0.3):
            hip = midpoint(landmarks[L.LEFT_HIP], landmarks[L.RIGHT_HIP])
            vx, vy, center_speed = self._center_tracker.update(hip.x, hip.y, now)
            if center_speed > 0.05:
                self._last_direction = rolling_direction(math.degrees(math.atan2(-vy, vx)))

        return {
            "energy": energy,
            "peak_energy": self._peak_energy,
            "max_speed": max_speed,
            "center_speed": center_speed,
            "left_hand_speed": self._speeds.get(L.LEFT_WRIST, 0.0),
            "right_hand_speed": self._speeds.get(L.RIGHT_WRIST, 0.0),
        }

    @staticmethod
    def _empty_metrics() -> Dict[str, float]:
        return {
            "energy": 0.0, "peak_energy": 0.0, "max_speed": 0.0,
            "center_speed": 0.0, "left_hand_speed": 0.0, "right_hand_speed": 0.0,
        }

    def speed_of(self, landmark_index: int) -> float:
        """Most recent speed of a single landmark."""
        return self._speeds.get(landmark_index, 0.0)

    @property
    def energy(self) -> float:
        return self._energy.value or 0.0

    @property
    def direction(self) -> str:
        """Compass label for the body's travel direction, e.g. ``"NE"``."""
        return self._last_direction

    def energy_history(self) -> List[float]:
        return self._energy_history.values()

    def average_energy(self) -> float:
        return self._energy_history.mean()

    def reset(self) -> None:
        for tracker in self._trackers.values():
            tracker.reset()
        self._center_tracker.reset()
        self._energy.reset()
        self._energy_history.clear()
        self._speeds.clear()
        self._peak_energy = 0.0


# ---------------------------------------------------------------------------
# Balance
# ---------------------------------------------------------------------------

@dataclass
class BalanceReport:
    """Balance metrics for one frame."""

    center_of_mass: Optional[Tuple[float, float]] = None
    base_center: Optional[float] = None
    offset: float = 0.0          # CoM horizontal offset from the support base
    sway: float = 0.0            # recent CoM movement
    stability: float = 100.0     # 0..100
    valid: bool = False

    @property
    def grade(self) -> str:
        if not self.valid:
            return "-"
        for threshold, label in ((85, "excelente"), (70, "buena"), (50, "regular")):
            if self.stability >= threshold:
                return label
        return "inestable"


class BalanceAnalyzer:
    """
    Estimates the centre of mass and how steady it is.

    Stability penalises both a CoM that sits outside the feet (leaning) and one
    that keeps moving (wobbling), which together capture what "good balance"
    means for standing poses and single-leg work.
    """

    def __init__(self, history: int = 90):
        self._com_history: Deque[Tuple[float, float]] = deque(maxlen=history)
        self._stability = ExponentialFilter(alpha=0.2)
        self._best = 0.0
        self._samples = 0
        self._total = 0.0

    def update(self, landmarks: Sequence[Any]) -> BalanceReport:
        """Compute the balance report for one pose."""
        report = BalanceReport()
        if not landmarks or len(landmarks) < L.COUNT:
            return report

        weighted_x = weighted_y = total_mass = 0.0
        for index, mass in SEGMENT_MASS_FRACTIONS.items():
            landmark = landmarks[index]
            if not landmark_visible(landmark, 0.3):
                continue
            weighted_x += landmark.x * mass
            weighted_y += landmark.y * mass
            total_mass += mass

        if total_mass < 0.3:      # too little of the body visible to judge
            return report

        com = (weighted_x / total_mass, weighted_y / total_mass)
        report.center_of_mass = com
        report.valid = True
        self._com_history.append(com)

        # Support base: whatever feet we can see, else the hips.
        feet = [
            landmarks[i] for i in (L.LEFT_ANKLE, L.RIGHT_ANKLE)
            if landmark_visible(landmarks[i], 0.3)
        ]
        if feet:
            base_center = sum(f.x for f in feet) / len(feet)
            base_width = (
                abs(feet[0].x - feet[1].x) if len(feet) > 1 else 0.12
            )
        else:
            hips = [
                landmarks[i] for i in (L.LEFT_HIP, L.RIGHT_HIP)
                if landmark_visible(landmarks[i], 0.3)
            ]
            if not hips:
                return report
            base_center = sum(h.x for h in hips) / len(hips)
            base_width = 0.15

        base_width = max(base_width, 0.06)
        report.base_center = base_center
        report.offset = abs(com[0] - base_center) / base_width

        # Sway: mean distance between consecutive CoM samples.
        if len(self._com_history) >= 2:
            deltas = [
                math.hypot(b[0] - a[0], b[1] - a[1])
                for a, b in zip(self._com_history, list(self._com_history)[1:])
            ]
            report.sway = sum(deltas) / len(deltas)

        lean_penalty = clamp(report.offset, 0.0, 2.0) * 30.0
        sway_penalty = clamp(report.sway * 1500.0, 0.0, 40.0)
        raw = clamp(100.0 - lean_penalty - sway_penalty, 0.0, 100.0)
        report.stability = self._stability.update(raw)

        self._samples += 1
        self._total += report.stability
        self._best = max(self._best, report.stability)
        return report

    def average_stability(self) -> float:
        return self._total / self._samples if self._samples else 0.0

    def stats(self) -> Dict[str, Any]:
        return {
            "samples": self._samples,
            "average_stability": round(self.average_stability(), 1),
            "best_stability": round(self._best, 1),
        }

    def reset(self) -> None:
        self._com_history.clear()
        self._stability.reset()
        self._samples = 0
        self._total = 0.0
        self._best = 0.0


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------

def symmetry_report(angles: Dict[str, float], tolerance: float = 12.0) -> Dict[str, Any]:
    """
    Compare left and right joint angles.

    Returns the per-joint difference, an overall 0..100 symmetry score and the
    joints that exceed *tolerance* degrees of asymmetry.
    """
    pairs = ("elbow", "shoulder", "hip", "knee", "ankle")
    differences: Dict[str, float] = {}
    asymmetric: List[str] = []

    for base in pairs:
        left = angles.get(f"left_{base}")
        right = angles.get(f"right_{base}")
        if left is None or right is None:
            continue
        delta = abs(left - right)
        differences[base] = delta
        if delta > tolerance:
            asymmetric.append(base)

    if not differences:
        return {"score": 0.0, "differences": {}, "asymmetric": [], "valid": False}

    mean_delta = sum(differences.values()) / len(differences)
    score = clamp(100.0 - mean_delta * 2.0, 0.0, 100.0)
    return {
        "score": score,
        "differences": {k: round(v, 1) for k, v in differences.items()},
        "asymmetric": asymmetric,
        "dominant_side": _dominant_side(angles),
        "valid": True,
    }


def _dominant_side(angles: Dict[str, float]) -> str:
    """Which side is more extended overall — a rough dominance hint."""
    left = [v for k, v in angles.items() if k.startswith("left_")]
    right = [v for k, v in angles.items() if k.startswith("right_")]
    if not left or not right:
        return "unknown"
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    if abs(left_mean - right_mean) < 5.0:
        return "equilibrado"
    return "izquierda" if left_mean > right_mean else "derecha"


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------

class TrajectoryRecorder:
    """
    Keeps a rolling trail of positions for chosen landmarks.

    Used by the trail overlay, the air-drawing mode and gesture path matching.
    """

    def __init__(self, landmark_indices: Sequence[int], max_points: int = 64):
        self.max_points = max(2, max_points)
        self._paths: Dict[int, Deque[Tuple[float, float, float]]] = {
            index: deque(maxlen=self.max_points) for index in landmark_indices
        }

    def update(self, landmarks: Sequence[Any], now: float, min_visibility: float = 0.4) -> None:
        """Append the current position of every tracked landmark."""
        if not landmarks:
            return
        for index, path in self._paths.items():
            if index >= len(landmarks):
                continue
            landmark = landmarks[index]
            if landmark_visible(landmark, min_visibility):
                path.append((landmark.x, landmark.y, now))

    def path(self, landmark_index: int) -> List[Tuple[float, float, float]]:
        """Trail for one landmark as ``(x, y, timestamp)`` tuples, oldest first."""
        return list(self._paths.get(landmark_index, ()))

    def paths(self) -> Dict[int, List[Tuple[float, float, float]]]:
        return {index: list(path) for index, path in self._paths.items()}

    def path_length(self, landmark_index: int) -> float:
        """Total distance travelled along a trail, in normalized units."""
        points = self.path(landmark_index)
        return sum(
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(points, points[1:])
        )

    def prune(self, now: float, max_age: float = 2.0) -> None:
        """Drop trail points older than *max_age* seconds."""
        for path in self._paths.values():
            while path and now - path[0][2] > max_age:
                path.popleft()

    def track(self, landmark_index: int) -> None:
        """Start recording an additional landmark."""
        self._paths.setdefault(landmark_index, deque(maxlen=self.max_points))

    def clear(self) -> None:
        for path in self._paths.values():
            path.clear()


# ---------------------------------------------------------------------------
# Stillness / presence
# ---------------------------------------------------------------------------

class StillnessDetector:
    """
    Detects when a person stops moving.

    Powers the meditation timer, the "are you still there" check and the
    security mode's inverse — motion alarms.
    """

    def __init__(self, threshold: float = 0.012, hold_seconds: float = 1.5):
        self.threshold = threshold
        self.hold_seconds = hold_seconds
        self._still_since: Optional[float] = None
        self._longest = 0.0

    def update(self, energy: float, now: float) -> bool:
        """Feed motion energy; returns True once stillness has been held."""
        if energy <= self.threshold:
            self._still_since = self._still_since or now
            duration = now - self._still_since
            self._longest = max(self._longest, duration)
            return duration >= self.hold_seconds
        self._still_since = None
        return False

    def still_seconds(self, now: float) -> float:
        """How long the person has been still right now."""
        return 0.0 if self._still_since is None else now - self._still_since

    @property
    def longest_stillness(self) -> float:
        return self._longest

    def reset(self) -> None:
        self._still_since = None
        self._longest = 0.0


# ---------------------------------------------------------------------------
# Sport-specific helpers
# ---------------------------------------------------------------------------

def jump_height_estimate(
    baseline_hip_y: float,
    peak_hip_y: float,
    body_height_normalized: float,
    real_height_cm: float = 170.0,
) -> float:
    """
    Rough jump height in centimetres.

    Scales the hip's vertical displacement by the ratio between the person's
    real height and their on-screen height. Accuracy depends on the camera
    being roughly perpendicular to the jump.
    """
    if body_height_normalized <= 1e-6:
        return 0.0
    displacement = max(0.0, baseline_hip_y - peak_hip_y)
    cm_per_unit = real_height_cm / body_height_normalized
    return displacement * cm_per_unit
