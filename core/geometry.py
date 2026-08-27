"""
Geometry helpers for LookThePerson.

Pure-Python vector math shared by gesture detection, analytics and rendering.
Deliberately free of numpy/cv2 imports so it stays cheap to import and simple
to unit-test.

Landmarks are duck-typed: any object exposing ``.x`` / ``.y`` (and optionally
``.z`` and ``.visibility``) works, which covers MediaPipe landmarks, plain
tuples wrapped by :class:`Point`, and test doubles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

__all__ = [
    "Point",
    "as_point",
    "distance",
    "distance_3d",
    "midpoint",
    "centroid",
    "angle_between",
    "joint_angle",
    "signed_angle",
    "vector",
    "norm",
    "normalize",
    "dot",
    "cross_z",
    "clamp",
    "lerp",
    "inverse_lerp",
    "remap",
    "to_pixels",
    "bounding_box",
    "expand_box",
    "box_area",
    "box_iou",
    "point_in_box",
    "point_in_circle",
    "polygon_area",
    "smooth_step",
    "rolling_direction",
    "is_finite_point",
    "safe_int",
    "to_pixels_clamped",
    "PIXEL_LIMIT",
]

#: Hard bound applied to every pixel coordinate the app produces.
#:
#: OpenCV's drawing primitives take C ``int`` coordinates and raise
#: ``OverflowError`` well before Python's unbounded ints do. A landmark only
#: has to be a few hundred units outside the frame — which happens whenever a
#: limb leaves the shot and the tracker extrapolates — for ``x * width`` to
#: exceed that. Clamping to a generous multiple of any real frame keeps
#: off-screen geometry pointing the right way while staying safely in range.
PIXEL_LIMIT = 1 << 20


# ---------------------------------------------------------------------------
# Basic types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Point:
    """A minimal landmark-compatible 3D point."""

    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0

    def as_tuple(self) -> Tuple[float, float]:
        return self.x, self.y

    def as_tuple_3d(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.z


def as_point(obj) -> Point:
    """Coerce a landmark-like object or ``(x, y[, z])`` tuple into a Point."""
    if isinstance(obj, Point):
        return obj
    if hasattr(obj, "x") and hasattr(obj, "y"):
        visibility = getattr(obj, "visibility", 1.0)
        return Point(
            float(obj.x),
            float(obj.y),
            float(getattr(obj, "z", 0.0) or 0.0),
            float(1.0 if visibility is None else visibility),
        )
    seq = tuple(obj)
    if len(seq) == 2:
        return Point(float(seq[0]), float(seq[1]))
    if len(seq) >= 3:
        return Point(float(seq[0]), float(seq[1]), float(seq[2]))
    raise TypeError(f"Cannot interpret {obj!r} as a point")


# ---------------------------------------------------------------------------
# Distances and vectors
# ---------------------------------------------------------------------------

def distance(a, b) -> float:
    """Euclidean distance in the XY plane."""
    pa, pb = as_point(a), as_point(b)
    return math.hypot(pa.x - pb.x, pa.y - pb.y)


def distance_3d(a, b) -> float:
    """Euclidean distance including the Z axis."""
    pa, pb = as_point(a), as_point(b)
    return math.sqrt((pa.x - pb.x) ** 2 + (pa.y - pb.y) ** 2 + (pa.z - pb.z) ** 2)


def midpoint(a, b) -> Point:
    """Point halfway between *a* and *b*."""
    pa, pb = as_point(a), as_point(b)
    return Point(
        (pa.x + pb.x) / 2.0,
        (pa.y + pb.y) / 2.0,
        (pa.z + pb.z) / 2.0,
        min(pa.visibility, pb.visibility),
    )


def centroid(points: Iterable) -> Optional[Point]:
    """Average of every point given, or ``None`` when the iterable is empty."""
    pts = [as_point(p) for p in points]
    if not pts:
        return None
    n = float(len(pts))
    return Point(
        sum(p.x for p in pts) / n,
        sum(p.y for p in pts) / n,
        sum(p.z for p in pts) / n,
        min(p.visibility for p in pts),
    )


def vector(a, b) -> Tuple[float, float]:
    """2D vector pointing from *a* to *b*."""
    pa, pb = as_point(a), as_point(b)
    return pb.x - pa.x, pb.y - pa.y


def norm(v: Sequence[float]) -> float:
    """Magnitude of a 2D vector."""
    return math.hypot(v[0], v[1])


def normalize(v: Sequence[float]) -> Tuple[float, float]:
    """Unit vector; returns ``(0, 0)`` for a zero-length input."""
    length = norm(v)
    if length < 1e-9:
        return 0.0, 0.0
    return v[0] / length, v[1] / length


def dot(v1: Sequence[float], v2: Sequence[float]) -> float:
    return v1[0] * v2[0] + v1[1] * v2[1]


def cross_z(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Z component of the 3D cross product — the sign gives turn direction."""
    return v1[0] * v2[1] - v1[1] * v2[0]


# ---------------------------------------------------------------------------
# Angles
# ---------------------------------------------------------------------------

def angle_between(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Unsigned angle between two vectors, in degrees (0-180)."""
    n1, n2 = norm(v1), norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cosine = clamp(dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def joint_angle(a, b, c) -> float:
    """
    Interior angle at joint *b* formed by the segments b->a and b->c.

    This is the standard convention for limb angles: ``joint_angle(shoulder,
    elbow, wrist)`` returns 180 for a fully extended arm and approaches 0 as
    the arm folds shut.
    """
    return angle_between(vector(b, a), vector(b, c))


def signed_angle(v1: Sequence[float], v2: Sequence[float]) -> float:
    """
    Angle from *v1* to *v2* in degrees, within (-180, 180].

    Positive means counter-clockwise in standard math orientation. Note that
    image coordinates have Y growing downward, so on screen a positive value
    reads as clockwise.
    """
    return math.degrees(math.atan2(cross_z(v1, v2), dot(v1, v2)))


def rolling_direction(angle_degrees: float) -> str:
    """Map an angle in degrees to one of eight compass-style directions."""
    labels = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    index = int(((angle_degrees % 360) + 22.5) // 45) % 8
    return labels[index]


# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------

def clamp(value: float, low: float, high: float) -> float:
    """Constrain *value* to the inclusive range [low, high]."""
    if low > high:
        low, high = high, low
    return max(low, min(high, value))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between *a* and *b* (t is not clamped)."""
    return a + (b - a) * t


def inverse_lerp(a: float, b: float, value: float) -> float:
    """Where *value* falls between a and b, as 0..1. Returns 0 if a == b."""
    if abs(b - a) < 1e-12:
        return 0.0
    return clamp((value - a) / (b - a), 0.0, 1.0)


def remap(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """Rescale *value* from one range to another, clamped to the output range."""
    return lerp(out_min, out_max, inverse_lerp(in_min, in_max, value))


def smooth_step(edge0: float, edge1: float, value: float) -> float:
    """Hermite smoothing curve — a soft 0..1 ramp between two edges."""
    t = inverse_lerp(edge0, edge1, value)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# Pixel space and boxes
# ---------------------------------------------------------------------------

def is_finite_point(point) -> bool:
    """
    Whether a landmark carries usable coordinates.

    MediaPipe emits NaN for a landmark it cannot solve, and the One-Euro
    smoother propagates that NaN forward for the rest of the session once it
    enters the filter state. Callers use this to skip a landmark rather than
    drawing it at a garbage position.
    """
    try:
        p = as_point(point)
    except (TypeError, ValueError):
        return False
    return math.isfinite(p.x) and math.isfinite(p.y)


def safe_int(value: float, fallback: int = 0) -> int:
    """
    Convert to ``int`` without ever raising.

    ``int(nan)`` raises ``ValueError`` and ``int(inf)`` raises
    ``OverflowError``; both reach the drawing code through landmarks. The
    result is clamped to :data:`PIXEL_LIMIT` so OpenCV never sees a coordinate
    it cannot represent.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(number):
        return fallback
    if number > PIXEL_LIMIT:
        return PIXEL_LIMIT
    if number < -PIXEL_LIMIT:
        return -PIXEL_LIMIT
    return int(number)


def to_pixels(point, width: int, height: int) -> Tuple[int, int]:
    """
    Convert a normalized landmark to integer pixel coordinates.

    Non-finite and wildly out-of-range coordinates are absorbed here rather
    than at each of the ~30 call sites, because this is the single funnel every
    landmark passes through on its way to OpenCV.
    """
    p = as_point(point)
    return safe_int(p.x * width), safe_int(p.y * height)


def to_pixels_clamped(point, width: int, height: int) -> Tuple[int, int]:
    """
    Like :func:`to_pixels` but confined to the frame itself.

    Use this when the coordinate indexes into the image (a crop, a mask, a
    region of interest); use :func:`to_pixels` when it is only being drawn,
    since OpenCV clips strokes for free and clamping would visibly pin an
    off-screen limb to the edge.
    """
    px, py = to_pixels(point, width, height)
    return (
        int(clamp(px, 0, max(0, width - 1))),
        int(clamp(py, 0, max(0, height - 1))),
    )


def bounding_box(points: Iterable) -> Optional[Tuple[float, float, float, float]]:
    """Axis-aligned bounds of the points as ``(x_min, y_min, x_max, y_max)``."""
    pts = [as_point(p) for p in points]
    if not pts:
        return None
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def expand_box(
    box: Tuple[float, float, float, float],
    factor: float = 1.0,
    clamp_to: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[float, float, float, float]:
    """
    Grow (or shrink) a box around its centre by *factor*.

    ``factor=1.2`` yields a box 20% larger in each dimension. When *clamp_to*
    is given the result is trimmed to that outer box.
    """
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half_w = (x1 - x0) / 2.0 * factor
    half_h = (y1 - y0) / 2.0 * factor
    out = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
    if clamp_to:
        cx0, cy0, cx1, cy1 = clamp_to
        out = (
            clamp(out[0], cx0, cx1),
            clamp(out[1], cy0, cy1),
            clamp(out[2], cx0, cx1),
            clamp(out[3], cy0, cy1),
        )
    return out


def box_area(box: Tuple[float, float, float, float]) -> float:
    """Area of a box; zero when the box is degenerate or inverted."""
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def box_iou(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    """Intersection-over-union of two boxes — used for detection tracking."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter = box_area((
        max(ax0, bx0), max(ay0, by0),
        min(ax1, bx1), min(ay1, by1),
    ))
    if inter <= 0.0:
        return 0.0
    union = box_area(box_a) + box_area(box_b) - inter
    return inter / union if union > 0 else 0.0


def point_in_box(point, box: Tuple[float, float, float, float]) -> bool:
    """Whether a point lies inside an axis-aligned box (edges inclusive)."""
    p = as_point(point)
    x0, y0, x1, y1 = box
    return x0 <= p.x <= x1 and y0 <= p.y <= y1


def point_in_circle(point, center, radius: float) -> bool:
    """Whether a point lies within *radius* of *center*."""
    return distance(point, center) <= radius


def polygon_area(points: Sequence) -> float:
    """Absolute area of a simple polygon via the shoelace formula."""
    pts = [as_point(p) for p in points]
    if len(pts) < 3:
        return 0.0
    total = 0.0
    for i in range(len(pts)):
        j = (i + 1) % len(pts)
        total += pts[i].x * pts[j].y - pts[j].x * pts[i].y
    return abs(total) / 2.0
