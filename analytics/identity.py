"""
Person identification and tracking for LookThePerson.

Turns per-frame pose detections into *persistent subjects*: each person keeps
the same identifier while they are in view, and reclaims it when they leave and
come back.

Two signals do the work:

* **Spatial continuity** — box overlap plus centroid distance links a detection
  to the track it most plausibly continues. This is what carries an identity
  across consecutive frames.
* **Body signature** — a vector of limb ratios normalised by torso length.
  Because every entry is a ratio, the signature barely moves as a subject walks
  toward or away from the camera, which is what makes it usable for
  re-identification once the spatial link has been broken.

The signature describes *build*, not identity in any biometric sense. It can
tell two or three people in a room apart; it cannot recognise anyone outside
the set it has seen, it is derived from skeleton proportions rather than
appearance, and nothing here is written to disk or persisted between sessions.

Pure Python — no numpy, OpenCV or MediaPipe — so every part is directly
unit-testable without a camera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from analytics.angles import PoseLandmark as L
from analytics.angles import landmark_visible, limb_lengths
from core.geometry import box_iou, clamp, is_finite_point

__all__ = [
    "SIGNATURE_KEYS",
    "BodySignature",
    "build_signature",
    "pose_box",
    "stature_span",
    "Detection",
    "TrackedPerson",
    "PersonTracker",
    "describe_build",
]

#: Ratios that make up a signature. All are divided by torso length, so the
#: vector is invariant to how far the subject stands from the camera.
SIGNATURE_KEYS: Tuple[str, ...] = (
    "shoulder_width",
    "hip_width",
    "left_upper_arm",
    "right_upper_arm",
    "left_forearm",
    "right_forearm",
    "left_thigh",
    "right_thigh",
    "left_shin",
    "right_shin",
)

#: Below this many shared ratios two signatures cannot be meaningfully
#: compared, so the tracker treats them as "no opinion" rather than guessing.
MIN_SHARED_RATIOS = 4


# ---------------------------------------------------------------------------
# Body signature
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BodySignature:
    """
    A scale-invariant description of someone's build.

    ``ratios`` maps a segment name to its length divided by the subject's torso
    length. Two signatures are compared with :meth:`distance`, which is the
    mean absolute difference over whatever ratios both of them carry.
    """

    ratios: Dict[str, float] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return len(self.ratios) >= MIN_SHARED_RATIOS

    def distance(self, other: "BodySignature") -> Optional[float]:
        """
        Mean absolute ratio difference, or ``None`` when not comparable.

        ``None`` is deliberately distinct from a large distance: it means the
        two signatures have too little in common to judge, which the caller
        must not read as "different person".
        """
        shared = [k for k in self.ratios if k in other.ratios]
        if len(shared) < MIN_SHARED_RATIOS:
            return None
        total = sum(abs(self.ratios[k] - other.ratios[k]) for k in shared)
        return total / len(shared)

    def similarity(self, other: "BodySignature") -> Optional[float]:
        """Distance mapped onto 0..1, where 1 is an exact match."""
        gap = self.distance(other)
        if gap is None:
            return None
        return clamp(1.0 - gap / 0.5, 0.0, 1.0)

    def blend(self, other: "BodySignature", weight: float = 0.2) -> "BodySignature":
        """
        Fold a fresh observation into a running signature.

        Averaging over time is what makes the signature usable: a single frame
        carries whatever noise the tracker had that instant, while the running
        mean settles onto the subject's actual proportions.
        """
        weight = clamp(weight, 0.0, 1.0)
        merged = dict(self.ratios)
        for key, value in other.ratios.items():
            if key in merged:
                merged[key] += (value - merged[key]) * weight
            else:
                merged[key] = value
        return BodySignature(merged)

    def as_code(self) -> str:
        """
        Short printable digest, stable for a given set of ratios.

        Shown in the HUD so an operator can tell two subjects apart at a
        glance. It is a display aid derived from the ratios, not an identifier
        that means anything outside this session.
        """
        if not self.ratios:
            return "----"
        digest = 0
        for key in sorted(self.ratios):
            quantised = int(round(self.ratios[key] * 100))
            digest = (digest * 131 + hash(key) % 997 + quantised) & 0xFFFFFFFF
        alphabet = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
        return "".join(alphabet[(digest >> (i * 5)) % len(alphabet)] for i in range(4))


def build_signature(
    landmarks: Sequence[Any],
    min_visibility: float = 0.4,
) -> Optional[BodySignature]:
    """
    Derive a :class:`BodySignature` from one person's pose landmarks.

    Returns ``None`` when the pose does not carry enough visible geometry —
    which is the normal case for someone half out of frame, and the reason the
    tracker must cope with signature-less detections.
    """
    if landmarks is None or len(landmarks) < L.COUNT:
        return None

    lengths = limb_lengths(landmarks, min_visibility=min_visibility)
    torso = lengths.get("torso")
    if not torso or not math.isfinite(torso) or torso < 1e-4:
        return None

    ratios: Dict[str, float] = {}
    for key in SIGNATURE_KEYS:
        value = lengths.get(key)
        if value is None or not math.isfinite(value):
            continue
        ratio = value / torso
        # A ratio outside this band means the skeleton is broken rather than
        # the person unusual; letting it in would poison the running mean.
        if 0.05 <= ratio <= 4.0:
            ratios[key] = ratio

    signature = BodySignature(ratios)
    return signature if signature.is_valid else None


def describe_build(signature: Optional[BodySignature]) -> str:
    """One-word build description for the HUD, from shoulder-to-hip ratio."""
    if signature is None:
        return "desconocida"
    shoulders = signature.ratios.get("shoulder_width")
    hips = signature.ratios.get("hip_width")
    if not shoulders or not hips or hips < 1e-6:
        return "desconocida"
    spread = shoulders / hips
    if spread >= 1.45:
        return "ancha"
    if spread >= 1.15:
        return "media"
    return "estrecha"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def pose_box(
    landmarks: Sequence[Any],
    min_visibility: float = 0.3,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Normalized ``(x0, y0, x1, y1)`` around the visible landmarks of one person.

    Non-finite landmarks are skipped rather than expanding the box to infinity.
    """
    xs: List[float] = []
    ys: List[float] = []
    for landmark in landmarks or ():
        if not is_finite_point(landmark):
            continue
        if not landmark_visible(landmark, min_visibility):
            continue
        xs.append(float(landmark.x))
        ys.append(float(landmark.y))

    if len(xs) < 2:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def stature_span(landmarks: Sequence[Any], min_visibility: float = 0.4) -> Optional[float]:
    """
    Vertical span from the head to the lowest visible foot, in frame units.

    Returned as a fraction of frame height, so it is only a stature estimate
    once combined with a calibration reference — see
    :meth:`TrackedPerson.estimated_height_cm`.
    """
    if landmarks is None or len(landmarks) < L.COUNT:
        return None

    head = landmarks[L.NOSE]
    if not is_finite_point(head):
        return None

    feet = [
        landmarks[i]
        for i in (L.LEFT_ANKLE, L.RIGHT_ANKLE, L.LEFT_HEEL, L.RIGHT_HEEL,
                  L.LEFT_FOOT_INDEX, L.RIGHT_FOOT_INDEX)
        if is_finite_point(landmarks[i]) and landmark_visible(landmarks[i], min_visibility)
    ]
    if not feet:
        return None

    span = max(float(f.y) for f in feet) - float(head.y)
    return span if span > 0.05 else None


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """One person found in the current frame."""

    box: Tuple[float, float, float, float]
    signature: Optional[BodySignature] = None
    span: Optional[float] = None
    has_face: bool = False

    @property
    def centroid(self) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0


@dataclass
class TrackedPerson:
    """A subject the tracker is following, with its accumulated history."""

    pid: int
    box: Tuple[float, float, float, float]
    first_seen: float
    last_seen: float
    signature: Optional[BodySignature] = None
    span: Optional[float] = None
    frames: int = 1
    reappearances: int = 0
    has_face: bool = False
    #: Seconds the subject has been in view across all of their appearances.
    visible_seconds: float = 0.0

    @property
    def label(self) -> str:
        """Operator-facing subject label, e.g. ``SUJ-03``."""
        return f"SUJ-{self.pid:02d}"

    @property
    def centroid(self) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0

    @property
    def code(self) -> str:
        return self.signature.as_code() if self.signature else "----"

    @property
    def build(self) -> str:
        return describe_build(self.signature)

    def dwell(self, now: float) -> float:
        """Seconds since this subject was first seen."""
        return max(0.0, now - self.first_seen)

    def confidence(self) -> float:
        """
        0..1 confidence that this really is a tracked person.

        Rises with how long the track has survived and whether a usable
        signature was ever extracted — a one-frame blob scores near zero.
        """
        stability = clamp(self.frames / 30.0, 0.0, 1.0)
        described = 0.35 if (self.signature and self.signature.is_valid) else 0.0
        return clamp(0.25 + stability * 0.4 + described, 0.0, 1.0)

    #: Stature estimates outside this band are reported as unknown. A figure
    #: this far off does not mean an unusual person, it means the subject is
    #: crouched, partly out of frame or angled away from the camera — and an
    #: operator display that prints "201 cm" with confidence is worse than one
    #: that admits it cannot tell.
    PLAUSIBLE_HEIGHT_CM = (110.0, 215.0)

    def estimated_height_cm(self, reference_cm: float = 170.0) -> Optional[float]:
        """
        Rough stature estimate in centimetres, or ``None`` when not credible.

        This is a proportional estimate, not a measurement: it assumes the
        subject is standing square to the camera and fully in frame, and scales
        against *reference_cm*. Treat it as a sorting aid — "taller than the
        other subject" — rather than a figure to record.
        """
        if not self.span or self.span <= 0.05:
            return None
        # A full-body subject standing upright fills roughly 0.82 of the frame
        # height at a typical surveillance framing; the ratio to that is what
        # scales the reference.
        estimate = reference_cm * (self.span / 0.82)
        low, high = self.PLAUSIBLE_HEIGHT_CM
        return estimate if low <= estimate <= high else None


class PersonTracker:
    """
    Assigns stable identities to per-frame person detections.

    A detection is matched to an existing track by a weighted blend of box
    overlap, centroid distance and signature difference. A detection that
    matches nothing is compared against a short-lived gallery of departed
    tracks before a new identity is minted, which is what lets a subject walk
    out of frame and return as the same subject rather than as a new one.
    """

    def __init__(
        self,
        forget_seconds: float = 1.2,
        gallery_seconds: float = 90.0,
        match_gate: float = 0.62,
        reid_similarity: float = 0.80,
        max_tracks: int = 12,
    ):
        self.forget_seconds = max(0.0, forget_seconds)
        self.gallery_seconds = max(0.0, gallery_seconds)
        self.match_gate = match_gate
        self.reid_similarity = reid_similarity
        self.max_tracks = max(1, max_tracks)

        self._tracks: Dict[int, TrackedPerson] = {}
        self._gallery: List[TrackedPerson] = []
        self._next_id = 1
        self._total_identified = 0
        self._last_now: Optional[float] = None

    # -- Query --------------------------------------------------------------

    @property
    def tracks(self) -> List[TrackedPerson]:
        """Currently visible subjects, oldest identity first."""
        return sorted(self._tracks.values(), key=lambda t: t.pid)

    @property
    def active_count(self) -> int:
        return len(self._tracks)

    @property
    def total_identified(self) -> int:
        """How many distinct identities have been minted this session."""
        return self._total_identified

    @property
    def gallery_size(self) -> int:
        return len(self._gallery)

    def get(self, pid: int) -> Optional[TrackedPerson]:
        return self._tracks.get(pid)

    # -- Update -------------------------------------------------------------

    def update(self, detections: Sequence[Detection], now: float) -> List[TrackedPerson]:
        """
        Advance the tracker one frame and return the visible subjects.

        Returns the same :class:`TrackedPerson` objects across frames, so a
        caller may hold on to one and watch its history grow.
        """
        elapsed = 0.0 if self._last_now is None else max(0.0, now - self._last_now)
        self._last_now = now

        pairs = self._match(list(detections), now)
        matched_ids = set()

        for pid, detection in pairs:
            track = self._tracks[pid]
            track.box = detection.box
            track.last_seen = now
            track.frames += 1
            track.has_face = detection.has_face
            track.visible_seconds += elapsed
            if detection.span:
                track.span = (
                    detection.span if track.span is None
                    else track.span + (detection.span - track.span) * 0.15
                )
            if detection.signature is not None:
                track.signature = (
                    detection.signature if track.signature is None
                    else track.signature.blend(detection.signature, 0.18)
                )
            matched_ids.add(pid)

        used = {id(d) for _pid, d in pairs}
        for detection in detections:
            if id(detection) in used:
                continue
            self._admit(detection, now)

        self._retire(matched_ids, now)
        self._expire_gallery(now)
        return self.tracks

    def _match(
        self,
        detections: List[Detection],
        now: float,
    ) -> List[Tuple[int, Detection]]:
        """
        Greedy lowest-cost assignment between tracks and detections.

        Greedy rather than optimal (Hungarian) on purpose: with at most a dozen
        subjects the assignments agree, and this stays trivial to reason about
        when a match looks wrong on screen.
        """
        candidates: List[Tuple[float, int, Detection]] = []
        for pid, track in self._tracks.items():
            for detection in detections:
                cost = self._cost(track, detection)
                if cost is not None and cost <= self.match_gate:
                    candidates.append((cost, pid, detection))

        candidates.sort(key=lambda item: item[0])
        pairs: List[Tuple[int, Detection]] = []
        taken_tracks: set = set()
        taken_detections: set = set()
        for _cost, pid, detection in candidates:
            if pid in taken_tracks or id(detection) in taken_detections:
                continue
            taken_tracks.add(pid)
            taken_detections.add(id(detection))
            pairs.append((pid, detection))
        return pairs

    def _cost(self, track: TrackedPerson, detection: Detection) -> Optional[float]:
        """
        Blended match cost in 0..1, or ``None`` when the pair is impossible.

        Overlap dominates because it is the most reliable signal frame to
        frame; the signature only breaks ties, since it is missing whenever the
        subject is partly out of shot.
        """
        overlap = box_iou(track.box, detection.box)
        tx, ty = track.centroid
        dx, dy = detection.centroid
        gap = math.hypot(tx - dx, ty - dy)

        # Two boxes that neither overlap nor sit close together are different
        # people, whatever their builds look like.
        if overlap <= 0.0 and gap > 0.35:
            return None

        cost = 0.55 * (1.0 - overlap) + 0.30 * clamp(gap / 0.35, 0.0, 1.0)

        if track.signature is not None and detection.signature is not None:
            difference = track.signature.distance(detection.signature)
            if difference is not None:
                cost += 0.15 * clamp(difference / 0.5, 0.0, 1.0)
                return cost
        # No comparable signature: charge the average so a pair with evidence
        # is preferred over one without, rather than rewarded for silence.
        return cost + 0.15 * 0.5

    def _admit(self, detection: Detection, now: float) -> None:
        """Re-identify a returning subject, or mint a new identity."""
        if len(self._tracks) >= self.max_tracks:
            return

        revived = self._reidentify(detection)
        if revived is not None:
            self._gallery.remove(revived)
            revived.box = detection.box
            revived.last_seen = now
            revived.frames += 1
            revived.reappearances += 1
            revived.has_face = detection.has_face
            if detection.signature is not None and revived.signature is not None:
                revived.signature = revived.signature.blend(detection.signature, 0.25)
            self._tracks[revived.pid] = revived
            return

        pid = self._next_id
        self._next_id += 1
        self._total_identified += 1
        self._tracks[pid] = TrackedPerson(
            pid=pid,
            box=detection.box,
            first_seen=now,
            last_seen=now,
            signature=detection.signature,
            span=detection.span,
            has_face=detection.has_face,
        )

    def _reidentify(self, detection: Detection) -> Optional[TrackedPerson]:
        """Best gallery match for a detection, if any clears the threshold."""
        if detection.signature is None:
            return None

        best: Optional[TrackedPerson] = None
        best_score = 0.0
        for candidate in self._gallery:
            if candidate.signature is None:
                continue
            score = candidate.signature.similarity(detection.signature)
            if score is None or score < self.reid_similarity:
                continue
            if score > best_score:
                best, best_score = candidate, score
        return best

    def _retire(self, matched_ids: set, now: float) -> None:
        """Move tracks that have not been seen recently into the gallery."""
        for pid in list(self._tracks):
            if pid in matched_ids:
                continue
            track = self._tracks[pid]
            if now - track.last_seen < self.forget_seconds:
                continue
            del self._tracks[pid]
            # Only a track with a usable signature is worth remembering; the
            # rest could never be re-identified anyway.
            if track.signature is not None and track.signature.is_valid:
                self._gallery.append(track)

    def _expire_gallery(self, now: float) -> None:
        self._gallery = [
            track for track in self._gallery
            if now - track.last_seen <= self.gallery_seconds
        ]

    def reset(self) -> None:
        """Forget every subject and start numbering from one again."""
        self._tracks.clear()
        self._gallery.clear()
        self._next_id = 1
        self._total_identified = 0
        self._last_now = None
