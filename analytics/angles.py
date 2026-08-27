"""
Joint angle computation for LookThePerson.

Turns the 33-point MediaPipe pose skeleton into named joint angles in degrees.
Every downstream feature — rep counting, posture scoring, form feedback, yoga
pose matching — is built on top of these numbers.

Convention: angles are the interior angle at the joint, so a fully extended
limb reads ~180 and a fully folded one approaches 0.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.geometry import angle_between, distance, joint_angle, midpoint, vector

__all__ = [
    "PoseLandmark",
    "JOINT_DEFINITIONS",
    "ANGLE_RANGES",
    "landmark_visible",
    "compute_joint_angles",
    "compute_angle",
    "limb_lengths",
    "trunk_inclination",
    "shoulder_tilt",
    "hip_tilt",
    "head_tilt",
    "body_orientation",
    "is_angle_within",
    "describe_angle",
    "flag_implausible",
]


class PoseLandmark:
    """Named indices into the MediaPipe pose landmark list."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

    COUNT = 33


L = PoseLandmark

# name -> (point_a, joint_vertex, point_c)
JOINT_DEFINITIONS: Dict[str, Tuple[int, int, int]] = {
    "left_elbow": (L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST),
    "right_elbow": (L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST),
    "left_shoulder": (L.LEFT_ELBOW, L.LEFT_SHOULDER, L.LEFT_HIP),
    "right_shoulder": (L.RIGHT_ELBOW, L.RIGHT_SHOULDER, L.RIGHT_HIP),
    "left_hip": (L.LEFT_SHOULDER, L.LEFT_HIP, L.LEFT_KNEE),
    "right_hip": (L.RIGHT_SHOULDER, L.RIGHT_HIP, L.RIGHT_KNEE),
    "left_knee": (L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE),
    "right_knee": (L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE),
    "left_ankle": (L.LEFT_KNEE, L.LEFT_ANKLE, L.LEFT_FOOT_INDEX),
    "right_ankle": (L.RIGHT_KNEE, L.RIGHT_ANKLE, L.RIGHT_FOOT_INDEX),
    "left_wrist": (L.LEFT_ELBOW, L.LEFT_WRIST, L.LEFT_INDEX),
    "right_wrist": (L.RIGHT_ELBOW, L.RIGHT_WRIST, L.RIGHT_INDEX),
}

# Comfortable human range per joint, used to flag implausible readings.
ANGLE_RANGES: Dict[str, Tuple[float, float]] = {
    "left_elbow": (25.0, 180.0),
    "right_elbow": (25.0, 180.0),
    "left_shoulder": (0.0, 180.0),
    "right_shoulder": (0.0, 180.0),
    "left_hip": (30.0, 180.0),
    "right_hip": (30.0, 180.0),
    "left_knee": (30.0, 180.0),
    "right_knee": (30.0, 180.0),
    "left_ankle": (60.0, 180.0),
    "right_ankle": (60.0, 180.0),
    "left_wrist": (90.0, 180.0),
    "right_wrist": (90.0, 180.0),
}

MIN_VISIBILITY = 0.35


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def landmark_visible(landmark: Any, threshold: float = MIN_VISIBILITY) -> bool:
    """Whether a landmark's tracking confidence clears *threshold*."""
    return getattr(landmark, "visibility", 1.0) >= threshold


def _all_visible(landmarks: Sequence[Any], indices: Sequence[int], threshold: float) -> bool:
    return all(
        index < len(landmarks) and landmark_visible(landmarks[index], threshold)
        for index in indices
    )


# ---------------------------------------------------------------------------
# Angle computation
# ---------------------------------------------------------------------------

def compute_angle(
    landmarks: Sequence[Any],
    joint_name: str,
    min_visibility: float = MIN_VISIBILITY,
) -> Optional[float]:
    """
    Angle at a single named joint, or ``None`` when its points are not visible.
    """
    definition = JOINT_DEFINITIONS.get(joint_name)
    if definition is None:
        raise KeyError(f"Unknown joint: {joint_name}")
    if not _all_visible(landmarks, definition, min_visibility):
        return None
    a, b, c = definition
    return joint_angle(landmarks[a], landmarks[b], landmarks[c])


def compute_joint_angles(
    landmarks: Sequence[Any],
    min_visibility: float = MIN_VISIBILITY,
) -> Dict[str, float]:
    """
    Every computable joint angle for one person.

    Joints whose landmarks are occluded are omitted rather than reported as
    zero, so callers can distinguish "straight" from "unknown".
    """
    if not landmarks or len(landmarks) < PoseLandmark.COUNT:
        return {}

    angles: Dict[str, float] = {}
    for name in JOINT_DEFINITIONS:
        value = compute_angle(landmarks, name, min_visibility)
        if value is not None:
            angles[name] = value

    trunk = trunk_inclination(landmarks, min_visibility)
    if trunk is not None:
        angles["trunk"] = trunk

    shoulders = shoulder_tilt(landmarks, min_visibility)
    if shoulders is not None:
        angles["shoulder_tilt"] = shoulders

    hips = hip_tilt(landmarks, min_visibility)
    if hips is not None:
        angles["hip_tilt"] = hips

    head = head_tilt(landmarks, min_visibility)
    if head is not None:
        angles["head_tilt"] = head

    # Symmetric pairs get an average, convenient for two-sided exercises.
    for base in ("elbow", "shoulder", "hip", "knee"):
        left, right = angles.get(f"left_{base}"), angles.get(f"right_{base}")
        if left is not None and right is not None:
            angles[f"avg_{base}"] = (left + right) / 2.0

    return angles


# ---------------------------------------------------------------------------
# Whole-body orientation
# ---------------------------------------------------------------------------

def trunk_inclination(
    landmarks: Sequence[Any],
    min_visibility: float = MIN_VISIBILITY,
) -> Optional[float]:
    """
    Lean of the torso away from vertical, in degrees.

    0 means upright; larger values mean the shoulders are further forward or
    back over the hips. Sign is not encoded — use :func:`body_orientation` for
    direction.
    """
    required = (L.LEFT_SHOULDER, L.RIGHT_SHOULDER, L.LEFT_HIP, L.RIGHT_HIP)
    if not _all_visible(landmarks, required, min_visibility):
        return None

    shoulder_mid = midpoint(landmarks[L.LEFT_SHOULDER], landmarks[L.RIGHT_SHOULDER])
    hip_mid = midpoint(landmarks[L.LEFT_HIP], landmarks[L.RIGHT_HIP])
    torso = vector(hip_mid, shoulder_mid)
    # Screen "up" is negative Y in normalized image space.
    return angle_between(torso, (0.0, -1.0))


def shoulder_tilt(
    landmarks: Sequence[Any],
    min_visibility: float = MIN_VISIBILITY,
) -> Optional[float]:
    """
    Roll of the shoulder line off horizontal, in degrees.

    Positive means the left shoulder (viewer's left) sits lower.
    """
    if not _all_visible(landmarks, (L.LEFT_SHOULDER, L.RIGHT_SHOULDER), min_visibility):
        return None
    left, right = landmarks[L.LEFT_SHOULDER], landmarks[L.RIGHT_SHOULDER]
    return math.degrees(math.atan2(left.y - right.y, abs(left.x - right.x) or 1e-6))


def hip_tilt(
    landmarks: Sequence[Any],
    min_visibility: float = MIN_VISIBILITY,
) -> Optional[float]:
    """Roll of the hip line off horizontal, in degrees."""
    if not _all_visible(landmarks, (L.LEFT_HIP, L.RIGHT_HIP), min_visibility):
        return None
    left, right = landmarks[L.LEFT_HIP], landmarks[L.RIGHT_HIP]
    return math.degrees(math.atan2(left.y - right.y, abs(left.x - right.x) or 1e-6))


def head_tilt(
    landmarks: Sequence[Any],
    min_visibility: float = MIN_VISIBILITY,
) -> Optional[float]:
    """Sideways head tilt derived from the ear line, in degrees."""
    if not _all_visible(landmarks, (L.LEFT_EAR, L.RIGHT_EAR), min_visibility):
        return None
    left, right = landmarks[L.LEFT_EAR], landmarks[L.RIGHT_EAR]
    return math.degrees(math.atan2(left.y - right.y, abs(left.x - right.x) or 1e-6))


def body_orientation(
    landmarks: Sequence[Any],
    min_visibility: float = MIN_VISIBILITY,
) -> str:
    """
    Coarse facing direction: ``"front"``, ``"left"``, ``"right"`` or ``"unknown"``.

    Inferred from how compressed the shoulder line is relative to the torso
    height — a person in profile shows a very narrow shoulder span.
    """
    required = (L.LEFT_SHOULDER, L.RIGHT_SHOULDER, L.LEFT_HIP, L.RIGHT_HIP)
    if not _all_visible(landmarks, required, min_visibility):
        return "unknown"

    left_s, right_s = landmarks[L.LEFT_SHOULDER], landmarks[L.RIGHT_SHOULDER]
    shoulder_span = abs(left_s.x - right_s.x)
    shoulder_mid = midpoint(left_s, right_s)
    hip_mid = midpoint(landmarks[L.LEFT_HIP], landmarks[L.RIGHT_HIP])
    torso_height = abs(shoulder_mid.y - hip_mid.y) or 1e-6

    # Anatomically the shoulder span is roughly 0.6-0.9 of torso height when
    # facing the camera, and collapses below ~0.25 in profile.
    if shoulder_span / torso_height > 0.5:
        return "front"
    # In profile the nearer shoulder wins on depth (smaller z is closer).
    left_z = getattr(left_s, "z", 0.0)
    right_z = getattr(right_s, "z", 0.0)
    return "left" if left_z < right_z else "right"


# ---------------------------------------------------------------------------
# Proportions
# ---------------------------------------------------------------------------

def limb_lengths(
    landmarks: Sequence[Any],
    min_visibility: float = MIN_VISIBILITY,
) -> Dict[str, float]:
    """
    Normalized segment lengths, useful for calibration and body measurement.

    Values are in normalized image units, so they only mean something relative
    to each other or after scaling by a known reference.
    """
    segments = {
        "left_upper_arm": (L.LEFT_SHOULDER, L.LEFT_ELBOW),
        "left_forearm": (L.LEFT_ELBOW, L.LEFT_WRIST),
        "right_upper_arm": (L.RIGHT_SHOULDER, L.RIGHT_ELBOW),
        "right_forearm": (L.RIGHT_ELBOW, L.RIGHT_WRIST),
        "left_thigh": (L.LEFT_HIP, L.LEFT_KNEE),
        "left_shin": (L.LEFT_KNEE, L.LEFT_ANKLE),
        "right_thigh": (L.RIGHT_HIP, L.RIGHT_KNEE),
        "right_shin": (L.RIGHT_KNEE, L.RIGHT_ANKLE),
        "shoulder_width": (L.LEFT_SHOULDER, L.RIGHT_SHOULDER),
        "hip_width": (L.LEFT_HIP, L.RIGHT_HIP),
    }

    out: Dict[str, float] = {}
    for name, (a, b) in segments.items():
        if _all_visible(landmarks, (a, b), min_visibility):
            out[name] = distance(landmarks[a], landmarks[b])

    shoulder_ok = _all_visible(landmarks, (L.LEFT_SHOULDER, L.RIGHT_SHOULDER), min_visibility)
    hip_ok = _all_visible(landmarks, (L.LEFT_HIP, L.RIGHT_HIP), min_visibility)
    if shoulder_ok and hip_ok:
        shoulder_mid = midpoint(landmarks[L.LEFT_SHOULDER], landmarks[L.RIGHT_SHOULDER])
        hip_mid = midpoint(landmarks[L.LEFT_HIP], landmarks[L.RIGHT_HIP])
        out["torso"] = distance(shoulder_mid, hip_mid)
    return out


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

def is_angle_within(value: Optional[float], target: float, tolerance: float) -> bool:
    """Whether an angle sits within *tolerance* degrees of *target*."""
    if value is None:
        return False
    return abs(value - target) <= tolerance


def describe_angle(joint_name: str, value: Optional[float]) -> str:
    """Short human-readable description of a joint angle, for the HUD."""
    if value is None:
        return f"{joint_name}: --"

    if "elbow" in joint_name or "knee" in joint_name:
        if value > 160:
            state = "extendido"
        elif value > 110:
            state = "medio"
        elif value > 70:
            state = "flexionado"
        else:
            state = "cerrado"
        return f"{joint_name}: {value:.0f}° ({state})"
    return f"{joint_name}: {value:.0f}°"


def flag_implausible(angles: Dict[str, float]) -> List[str]:
    """
    Names of joints whose angle falls outside the plausible human range.

    A useful tracking-quality signal: several flagged joints at once usually
    means the skeleton has latched onto the wrong thing.
    """
    flagged: List[str] = []
    for name, value in angles.items():
        bounds = ANGLE_RANGES.get(name)
        if bounds and not (bounds[0] <= value <= bounds[1]):
            flagged.append(name)
    return flagged
