"""
Shared test fixtures for LookThePerson.

Provides synthetic skeletons and hands so the analytics and gesture layers can
be tested exactly, with no camera, no model downloads and no randomness.
"""

from __future__ import annotations

import math
import os
import sys
from typing import List

import pytest

# Make the project importable when pytest is run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.angles import PoseLandmark as L  # noqa: E402
from core.geometry import Point  # noqa: E402


def make_pose(knee_angle: float = 180.0, **overrides) -> List[Point]:
    """
    Build a front-facing standing skeleton with realistic proportions.

    *knee_angle* rotates the shin around the knee so a caller can drive a
    squat through its full range; any landmark can be replaced by passing its
    :class:`PoseLandmark` name as a keyword.
    """
    points = [Point(0.5, 0.5) for _ in range(L.COUNT)]

    points[L.NOSE] = Point(0.50, 0.10)
    points[L.LEFT_EYE_INNER] = Point(0.48, 0.09)
    points[L.LEFT_EYE] = Point(0.47, 0.09)
    points[L.LEFT_EYE_OUTER] = Point(0.46, 0.09)
    points[L.RIGHT_EYE_INNER] = Point(0.52, 0.09)
    points[L.RIGHT_EYE] = Point(0.53, 0.09)
    points[L.RIGHT_EYE_OUTER] = Point(0.54, 0.09)
    points[L.LEFT_EAR] = Point(0.45, 0.11)
    points[L.RIGHT_EAR] = Point(0.55, 0.11)
    points[L.MOUTH_LEFT] = Point(0.48, 0.14)
    points[L.MOUTH_RIGHT] = Point(0.52, 0.14)

    points[L.LEFT_SHOULDER] = Point(0.42, 0.26)
    points[L.RIGHT_SHOULDER] = Point(0.58, 0.26)
    points[L.LEFT_ELBOW] = Point(0.40, 0.40)
    points[L.RIGHT_ELBOW] = Point(0.60, 0.40)
    points[L.LEFT_WRIST] = Point(0.39, 0.54)
    points[L.RIGHT_WRIST] = Point(0.61, 0.54)
    points[L.LEFT_INDEX] = Point(0.39, 0.58)
    points[L.RIGHT_INDEX] = Point(0.61, 0.58)

    points[L.LEFT_HIP] = Point(0.45, 0.53)
    points[L.RIGHT_HIP] = Point(0.55, 0.53)
    points[L.LEFT_KNEE] = Point(0.45, 0.74)
    points[L.RIGHT_KNEE] = Point(0.55, 0.74)

    # Place each ankle so that hip-knee-ankle equals the requested angle.
    radians = math.radians(knee_angle)
    for knee, hip, ankle, foot in (
        (L.LEFT_KNEE, L.LEFT_HIP, L.LEFT_ANKLE, L.LEFT_FOOT_INDEX),
        (L.RIGHT_KNEE, L.RIGHT_HIP, L.RIGHT_ANKLE, L.RIGHT_FOOT_INDEX),
    ):
        k = points[knee]
        vx, vy = points[hip].x - k.x, points[hip].y - k.y
        length = math.hypot(vx, vy) or 1e-6
        vx, vy = vx / length, vy / length
        shin = 0.20
        ax = k.x + shin * (vx * math.cos(radians) - vy * math.sin(radians))
        ay = k.y + shin * (vx * math.sin(radians) + vy * math.cos(radians))
        points[ankle] = Point(ax, ay)
        points[foot] = Point(ax, ay + 0.04)

    for name, value in overrides.items():
        points[getattr(L, name)] = value
    return points


def make_hand(x: float = 0.5, y: float = 0.5, fingers: int = 5) -> List[Point]:
    """
    Build a hand skeleton with a chosen number of extended fingers.

    Extended fingertips sit above their PIP joint; folded ones sit below,
    which is exactly what the classifier looks at.
    """
    hand = [Point(x, y) for _ in range(21)]
    hand[0] = Point(x, y)                      # wrist
    hand[9] = Point(x, y - 0.10)               # middle MCP sets the hand scale

    # Thumb: extended reaches away from the wrist, folded stays close.
    if fingers >= 1:
        hand[1] = Point(x - 0.03, y - 0.02)
        hand[2] = Point(x - 0.06, y - 0.04)
        hand[3] = Point(x - 0.08, y - 0.05)
        hand[4] = Point(x - 0.13, y - 0.07)
    else:
        hand[1] = Point(x - 0.02, y - 0.01)
        hand[2] = Point(x - 0.03, y - 0.02)
        hand[3] = Point(x - 0.035, y - 0.02)
        hand[4] = Point(x - 0.03, y - 0.015)

    long_fingers = (
        (5, 6, 7, 8, -0.045),      # index
        (9, 10, 11, 12, -0.015),   # middle
        (13, 14, 15, 16, 0.015),   # ring
        (17, 18, 19, 20, 0.045),   # pinky
    )
    for index, (mcp, pip, dip, tip, offset) in enumerate(long_fingers, start=1):
        hand[mcp] = Point(x + offset, y - 0.09)
        hand[pip] = Point(x + offset, y - 0.13)
        if index < fingers:
            hand[dip] = Point(x + offset, y - 0.17)
            hand[tip] = Point(x + offset, y - 0.21)   # extended: above the PIP
        else:
            hand[dip] = Point(x + offset, y - 0.11)
            hand[tip] = Point(x + offset, y - 0.07)   # folded: below the PIP
    return hand


def make_face(mouth_open: float = 0.02, eye_open: float = 0.03) -> List[Point]:
    """
    Build a 478-point face mesh with controllable eye and mouth openness.

    Only the landmarks the metrics actually read are positioned meaningfully;
    the rest fill the mesh so length checks pass.
    """
    face = [Point(0.5, 0.3) for _ in range(478)]

    face[10] = Point(0.50, 0.14)     # forehead
    face[152] = Point(0.50, 0.46)    # chin
    face[1] = Point(0.50, 0.30)      # nose tip
    face[234] = Point(0.40, 0.30)    # left cheek
    face[454] = Point(0.60, 0.30)    # right cheek

    # Left eye
    face[33] = Point(0.43, 0.25)
    face[133] = Point(0.48, 0.25)
    face[159] = Point(0.455, 0.25 - eye_open)
    face[145] = Point(0.455, 0.25 + eye_open)
    face[158] = Point(0.46, 0.25 - eye_open)
    face[153] = Point(0.46, 0.25 + eye_open)

    # Right eye
    face[263] = Point(0.57, 0.25)
    face[362] = Point(0.52, 0.25)
    face[386] = Point(0.545, 0.25 - eye_open)
    face[374] = Point(0.545, 0.25 + eye_open)
    face[385] = Point(0.54, 0.25 - eye_open)
    face[380] = Point(0.54, 0.25 + eye_open)

    # Mouth
    face[61] = Point(0.455, 0.39)
    face[291] = Point(0.545, 0.39)
    face[13] = Point(0.50, 0.39 - mouth_open)
    face[14] = Point(0.50, 0.39 + mouth_open)
    face[0] = Point(0.50, 0.36)
    face[17] = Point(0.50, 0.42)

    # Irises, centred
    face[468] = Point(0.455, 0.25)
    face[473] = Point(0.545, 0.25)
    return face


@pytest.fixture
def standing_pose() -> List[Point]:
    return make_pose(178.0)


@pytest.fixture
def squat_pose() -> List[Point]:
    return make_pose(80.0)


@pytest.fixture
def open_hand() -> List[Point]:
    return make_hand(fingers=5)


@pytest.fixture
def fist_hand() -> List[Point]:
    return make_hand(fingers=0)


@pytest.fixture
def neutral_face() -> List[Point]:
    return make_face()
