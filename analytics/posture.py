"""
Posture analysis for LookThePerson.

Scores how someone is holding themselves and names the specific problems —
slouching, forward head, uneven shoulders, leaning — so a coaching mode can
give actionable feedback rather than a bare number.

Designed for a seated or standing user facing the camera. Readings degrade
gracefully when landmarks are occluded: each check reports ``None`` instead of
guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from analytics.angles import PoseLandmark as L
from analytics.angles import head_tilt, landmark_visible, shoulder_tilt, trunk_inclination
from core.filters import Debouncer, ExponentialFilter
from core.geometry import clamp, distance, midpoint

__all__ = [
    "PostureIssue",
    "PostureReport",
    "analyze_posture",
    "PostureMonitor",
    "SLOUCH_THRESHOLD",
    "FORWARD_HEAD_THRESHOLD",
    "SHOULDER_TILT_THRESHOLD",
    "LEAN_THRESHOLD",
]

# Tuning thresholds — degrees except where noted.
SLOUCH_THRESHOLD = 0.62          # neck length ratio below which we call it slouch
FORWARD_HEAD_THRESHOLD = 0.22    # ear-ahead-of-shoulder offset, torso-relative
SHOULDER_TILT_THRESHOLD = 7.0
LEAN_THRESHOLD = 12.0
HEAD_TILT_THRESHOLD = 10.0


@dataclass(frozen=True)
class PostureIssue:
    """One specific posture problem with a severity in 0..1."""

    code: str
    label: str
    severity: float
    advice: str

    @property
    def level(self) -> str:
        """``"warn"`` for mild problems, ``"danger"`` once severe."""
        return "danger" if self.severity >= 0.6 else "warn"


@dataclass
class PostureReport:
    """The result of one posture evaluation."""

    score: float = 100.0                      # 0..100, higher is better
    issues: List[PostureIssue] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    valid: bool = False

    @property
    def is_good(self) -> bool:
        return self.valid and self.score >= 80.0

    @property
    def grade(self) -> str:
        """Letter grade, for a compact HUD readout."""
        if not self.valid:
            return "-"
        for threshold, grade in ((90, "A"), (80, "B"), (70, "C"), (60, "D")):
            if self.score >= threshold:
                return grade
        return "F"

    @property
    def worst_issue(self) -> Optional[PostureIssue]:
        return max(self.issues, key=lambda i: i.severity, default=None)

    def summary(self) -> str:
        """One-line status suitable for the HUD."""
        if not self.valid:
            return "Postura: sin datos"
        worst = self.worst_issue
        if worst is None:
            return f"Postura: {self.score:.0f}/100 — correcta"
        return f"Postura: {self.score:.0f}/100 — {worst.label}"


# ---------------------------------------------------------------------------
# Single-frame analysis
# ---------------------------------------------------------------------------

def analyze_posture(
    landmarks: Sequence[Any],
    min_visibility: float = 0.4,
) -> PostureReport:
    """
    Evaluate posture from one pose skeleton.

    Returns a report with an overall score, the list of detected issues and
    the raw metrics behind them. ``valid`` is False when the upper body is not
    visible enough to judge.
    """
    report = PostureReport()
    if not landmarks or len(landmarks) < L.COUNT:
        return report

    required = (L.LEFT_SHOULDER, L.RIGHT_SHOULDER, L.LEFT_EAR, L.RIGHT_EAR)
    if not all(landmark_visible(landmarks[i], min_visibility) for i in required):
        return report

    report.valid = True
    penalties = 0.0

    shoulder_mid = midpoint(landmarks[L.LEFT_SHOULDER], landmarks[L.RIGHT_SHOULDER])
    ear_mid = midpoint(landmarks[L.LEFT_EAR], landmarks[L.RIGHT_EAR])
    shoulder_width = distance(landmarks[L.LEFT_SHOULDER], landmarks[L.RIGHT_SHOULDER]) or 1e-6

    # -- Slouch: the head sinks toward the shoulders -----------------------
    neck_length = abs(shoulder_mid.y - ear_mid.y)
    neck_ratio = neck_length / shoulder_width
    report.metrics["neck_ratio"] = neck_ratio

    if neck_ratio < SLOUCH_THRESHOLD:
        severity = clamp((SLOUCH_THRESHOLD - neck_ratio) / SLOUCH_THRESHOLD, 0.0, 1.0)
        penalties += severity * 35.0
        report.issues.append(PostureIssue(
            code="slouch",
            label="Espalda encorvada",
            severity=severity,
            advice="Estira la columna y baja los hombros",
        ))

    # -- Forward head: ears drift ahead of the shoulders --------------------
    forward_offset = abs(ear_mid.x - shoulder_mid.x) / shoulder_width
    report.metrics["forward_head"] = forward_offset
    if forward_offset > FORWARD_HEAD_THRESHOLD:
        severity = clamp(
            (forward_offset - FORWARD_HEAD_THRESHOLD) / (FORWARD_HEAD_THRESHOLD * 2), 0.0, 1.0
        )
        penalties += severity * 25.0
        report.issues.append(PostureIssue(
            code="forward_head",
            label="Cabeza adelantada",
            severity=severity,
            advice="Lleva la barbilla hacia atras, alinea las orejas con los hombros",
        ))

    # -- Uneven shoulders ---------------------------------------------------
    tilt = shoulder_tilt(landmarks, min_visibility)
    if tilt is not None:
        report.metrics["shoulder_tilt"] = tilt
        if abs(tilt) > SHOULDER_TILT_THRESHOLD:
            severity = clamp((abs(tilt) - SHOULDER_TILT_THRESHOLD) / 20.0, 0.0, 1.0)
            penalties += severity * 20.0
            side = "izquierdo" if tilt > 0 else "derecho"
            report.issues.append(PostureIssue(
                code="shoulder_tilt",
                label=f"Hombro {side} caido",
                severity=severity,
                advice="Nivela los hombros, reparte el peso",
            ))

    # -- Trunk lean ---------------------------------------------------------
    lean = trunk_inclination(landmarks, min_visibility)
    if lean is not None:
        report.metrics["trunk_lean"] = lean
        if lean > LEAN_THRESHOLD:
            severity = clamp((lean - LEAN_THRESHOLD) / 30.0, 0.0, 1.0)
            penalties += severity * 20.0
            report.issues.append(PostureIssue(
                code="lean",
                label="Torso inclinado",
                severity=severity,
                advice="Centra el torso sobre las caderas",
            ))

    # -- Head tilt ----------------------------------------------------------
    tilt_head = head_tilt(landmarks, min_visibility)
    if tilt_head is not None:
        report.metrics["head_tilt"] = tilt_head
        if abs(tilt_head) > HEAD_TILT_THRESHOLD:
            severity = clamp((abs(tilt_head) - HEAD_TILT_THRESHOLD) / 25.0, 0.0, 1.0)
            penalties += severity * 10.0
            report.issues.append(PostureIssue(
                code="head_tilt",
                label="Cabeza ladeada",
                severity=severity,
                advice="Endereza la cabeza",
            ))

    report.score = clamp(100.0 - penalties, 0.0, 100.0)
    report.issues.sort(key=lambda i: i.severity, reverse=True)
    return report


# ---------------------------------------------------------------------------
# Temporal monitoring
# ---------------------------------------------------------------------------

class PostureMonitor:
    """
    Tracks posture over time and raises an alert once it stays bad.

    A single slouched frame means nothing — people shift constantly. An alert
    fires only after the score stays below *alert_score* for *alert_seconds*,
    and then at most once per *alert_cooldown*.
    """

    def __init__(
        self,
        alert_score: float = 65.0,
        alert_seconds: float = 5.0,
        alert_cooldown: float = 30.0,
        smoothing: float = 0.15,
    ):
        self.alert_score = alert_score
        self.alert_seconds = alert_seconds
        self.alert_cooldown = alert_cooldown
        self._smoother = ExponentialFilter(alpha=smoothing)
        self._bad = Debouncer(rise_seconds=alert_seconds, fall_seconds=1.5)
        self._last_alert = float("-inf")
        self._last_report = PostureReport()

        # Session accumulators
        self._samples = 0
        self._score_total = 0.0
        self._good_frames = 0
        self._issue_counts: Dict[str, int] = {}
        self._best = 0.0
        self._worst = 100.0

    def update(self, landmarks: Sequence[Any], now: float) -> PostureReport:
        """Evaluate one frame and update the running session statistics."""
        report = analyze_posture(landmarks)
        self._last_report = report
        if not report.valid:
            return report

        smoothed = self._smoother.update(report.score)
        report.metrics["smoothed_score"] = smoothed

        self._samples += 1
        self._score_total += report.score
        self._best = max(self._best, report.score)
        self._worst = min(self._worst, report.score)
        if report.is_good:
            self._good_frames += 1
        for issue in report.issues:
            self._issue_counts[issue.code] = self._issue_counts.get(issue.code, 0) + 1

        self._bad.update(smoothed < self.alert_score, now)
        return report

    def should_alert(self, now: float) -> bool:
        """Whether a posture alert is due; consumes the cooldown when True."""
        if not self._bad.state:
            return False
        if now - self._last_alert < self.alert_cooldown:
            return False
        self._last_alert = now
        return True

    @property
    def last_report(self) -> PostureReport:
        return self._last_report

    @property
    def average_score(self) -> float:
        return self._score_total / self._samples if self._samples else 0.0

    @property
    def good_ratio(self) -> float:
        """Fraction of evaluated frames with good posture, 0..1."""
        return self._good_frames / self._samples if self._samples else 0.0

    def most_common_issue(self) -> Optional[str]:
        if not self._issue_counts:
            return None
        return max(self._issue_counts.items(), key=lambda item: item[1])[0]

    def session_stats(self) -> Dict[str, Any]:
        """Everything worth writing into the session report."""
        return {
            "samples": self._samples,
            "average_score": round(self.average_score, 1),
            "best_score": round(self._best, 1),
            "worst_score": round(self._worst, 1) if self._samples else 0.0,
            "good_ratio": round(self.good_ratio, 3),
            "issues": dict(self._issue_counts),
            "most_common_issue": self.most_common_issue(),
        }

    def reset(self) -> None:
        self._smoother.reset()
        self._bad.reset()
        self._samples = 0
        self._score_total = 0.0
        self._good_frames = 0
        self._issue_counts.clear()
        self._best = 0.0
        self._worst = 100.0
