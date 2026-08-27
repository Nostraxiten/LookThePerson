"""
Exercise repetition counting for LookThePerson.

A repetition is detected as a full traversal of a joint angle between two
thresholds: the joint must reach the "down" position and return to the "up"
position before the counter advances. Hysteresis plus a minimum duration keeps
noise and half-reps from inflating the count.

Adding a new exercise means adding one :class:`ExerciseDefinition` — no changes
to the state machine itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from analytics.angles import compute_joint_angles
from core.geometry import clamp, inverse_lerp

__all__ = [
    "ExerciseDefinition",
    "RepEvent",
    "EXERCISES",
    "RepCounter",
    "MultiExerciseCounter",
    "exercise_names",
    "get_exercise",
]


# ---------------------------------------------------------------------------
# Exercise catalogue
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExerciseDefinition:
    """
    Describes how to count one exercise.

    Attributes:
        key: stable identifier.
        label: display name.
        angle_name: which joint angle drives the state machine.
        down_angle: angle at or below which the rep is "down".
        up_angle: angle at or above which the rep is "up".
        min_rep_seconds: reps faster than this are rejected as noise.
        max_rep_seconds: a rep taking longer resets the machine.
        met: metabolic equivalent, for calorie estimation.
        form_checks: extra ``(angle_name, min, max, message)`` constraints
            evaluated at the bottom of the rep.
        descending: True when the "down" phase is a *larger* angle than "up"
            (used by exercises like the plank hold or leg raise).
    """

    key: str
    label: str
    angle_name: str
    down_angle: float
    up_angle: float
    min_rep_seconds: float = 0.6
    max_rep_seconds: float = 12.0
    met: float = 5.0
    form_checks: Tuple[Tuple[str, float, float, str], ...] = ()
    descending: bool = False
    muscle_group: str = "general"


EXERCISES: Dict[str, ExerciseDefinition] = {
    "squat": ExerciseDefinition(
        key="squat", label="Sentadilla", angle_name="avg_knee",
        down_angle=100.0, up_angle=160.0, met=5.5, muscle_group="piernas",
        form_checks=(("trunk", 0.0, 45.0, "Mantén el torso más erguido"),),
    ),
    "pushup": ExerciseDefinition(
        key="pushup", label="Flexión", angle_name="avg_elbow",
        down_angle=95.0, up_angle=160.0, met=8.0, muscle_group="pecho",
        form_checks=(("avg_hip", 150.0, 190.0, "No hundas la cadera"),),
    ),
    "bicep_curl": ExerciseDefinition(
        key="bicep_curl", label="Curl de bíceps", angle_name="avg_elbow",
        down_angle=55.0, up_angle=155.0, met=3.5, muscle_group="brazos",
    ),
    "left_curl": ExerciseDefinition(
        key="left_curl", label="Curl izquierdo", angle_name="left_elbow",
        down_angle=55.0, up_angle=155.0, met=3.5, muscle_group="brazos",
    ),
    "right_curl": ExerciseDefinition(
        key="right_curl", label="Curl derecho", angle_name="right_elbow",
        down_angle=55.0, up_angle=155.0, met=3.5, muscle_group="brazos",
    ),
    "shoulder_press": ExerciseDefinition(
        key="shoulder_press", label="Press militar", angle_name="avg_elbow",
        down_angle=85.0, up_angle=165.0, met=5.0, muscle_group="hombros",
    ),
    "lunge": ExerciseDefinition(
        key="lunge", label="Zancada", angle_name="left_knee",
        down_angle=105.0, up_angle=165.0, met=5.0, muscle_group="piernas",
    ),
    "jumping_jack": ExerciseDefinition(
        key="jumping_jack", label="Jumping jack", angle_name="avg_shoulder",
        down_angle=35.0, up_angle=140.0, min_rep_seconds=0.35,
        met=8.0, muscle_group="cardio",
    ),
    "sit_up": ExerciseDefinition(
        key="sit_up", label="Abdominal", angle_name="avg_hip",
        down_angle=100.0, up_angle=150.0, met=6.0, muscle_group="core",
    ),
    "lateral_raise": ExerciseDefinition(
        key="lateral_raise", label="Elevación lateral", angle_name="avg_shoulder",
        down_angle=25.0, up_angle=85.0, met=4.0, muscle_group="hombros",
    ),
    "deadlift": ExerciseDefinition(
        key="deadlift", label="Peso muerto", angle_name="avg_hip",
        down_angle=105.0, up_angle=165.0, met=6.0, muscle_group="espalda",
    ),
    "calf_raise": ExerciseDefinition(
        key="calf_raise", label="Elevación de talones", angle_name="avg_ankle",
        down_angle=95.0, up_angle=130.0, min_rep_seconds=0.4,
        met=3.0, muscle_group="piernas",
    ),
}


def exercise_names() -> List[str]:
    """Keys of every known exercise, in catalogue order."""
    return list(EXERCISES.keys())


def get_exercise(key: str) -> Optional[ExerciseDefinition]:
    return EXERCISES.get(key)


# ---------------------------------------------------------------------------
# Rep events
# ---------------------------------------------------------------------------

@dataclass
class RepEvent:
    """A completed repetition."""

    exercise: str
    index: int                       # 1-based rep number in the current set
    duration: float                  # seconds from top to top
    depth: float                     # 0..1, how fully the rep reached the bottom
    form_score: float = 100.0        # 0..100
    warnings: List[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return self.form_score >= 80.0 and not self.warnings

    @property
    def tempo(self) -> str:
        """Qualitative pace label."""
        if self.duration < 1.0:
            return "rapida"
        if self.duration < 2.5:
            return "normal"
        return "lenta"


# ---------------------------------------------------------------------------
# The counter
# ---------------------------------------------------------------------------

class RepCounter:
    """
    Counts repetitions of a single exercise from a stream of joint angles.

    The machine has two states, ``"up"`` and ``"down"``. A rep completes on the
    down -> up transition, provided the elapsed time is plausible.
    """

    def __init__(self, exercise: str = "squat"):
        definition = EXERCISES.get(exercise)
        if definition is None:
            raise KeyError(f"Unknown exercise: {exercise}")
        self.definition = definition

        self._state = "up"
        self._count = 0
        self._set_count = 0
        self._phase_started: Optional[float] = None
        self._rep_started: Optional[float] = None
        self._extreme_angle: Optional[float] = None
        self._history: List[RepEvent] = []
        self._last_angle: Optional[float] = None
        self._warnings: List[str] = []
        self._last_rep_time: Optional[float] = None
        self._rest_since: Optional[float] = None

    # -- Configuration ------------------------------------------------------

    def set_exercise(self, exercise: str) -> None:
        """Switch exercise, banking the current reps as a completed set."""
        definition = EXERCISES.get(exercise)
        if definition is None:
            raise KeyError(f"Unknown exercise: {exercise}")
        if self._count:
            self.complete_set()
        self.definition = definition
        self._reset_machine()

    @property
    def exercise(self) -> str:
        return self.definition.key

    @property
    def label(self) -> str:
        return self.definition.label

    # -- Counting -----------------------------------------------------------

    def update(self, angles: Dict[str, float], now: float) -> Optional[RepEvent]:
        """
        Feed one frame of joint angles.

        Returns a :class:`RepEvent` on the frame a rep completes, else None.
        """
        angle = angles.get(self.definition.angle_name)
        if angle is None:
            return None

        self._last_angle = angle
        down_at, up_at = self.definition.down_angle, self.definition.up_angle
        going_down = angle <= down_at if not self.definition.descending else angle >= down_at
        going_up = angle >= up_at if not self.definition.descending else angle <= up_at

        if self._state == "up" and going_down:
            self._state = "down"
            self._phase_started = now
            self._rep_started = self._rep_started or now
            self._extreme_angle = angle
            self._warnings = list(self._check_form(angles))
            self._rest_since = None
            return None

        if self._state == "down":
            # Track the deepest point reached and keep checking form there.
            if self._extreme_angle is None:
                self._extreme_angle = angle
            elif self.definition.descending:
                self._extreme_angle = max(self._extreme_angle, angle)
            else:
                self._extreme_angle = min(self._extreme_angle, angle)

            for warning in self._check_form(angles):
                if warning not in self._warnings:
                    self._warnings.append(warning)

            # Abandon a rep that stalls at the bottom.
            if self._rep_started and now - self._rep_started > self.definition.max_rep_seconds:
                self._reset_rep(now)
                return None

            if going_up:
                return self._complete_rep(now)

        return None

    def _complete_rep(self, now: float) -> Optional[RepEvent]:
        started = self._rep_started or now
        duration = now - started
        self._state = "up"
        self._rep_started = None

        if duration < self.definition.min_rep_seconds:
            # Too fast to be real — most likely tracking jitter.
            self._reset_rep(now)
            return None

        self._count += 1
        depth = self._depth_ratio()
        form_score = clamp(100.0 - 15.0 * len(self._warnings), 0.0, 100.0)
        form_score *= clamp(0.6 + 0.4 * depth, 0.0, 1.0)

        event = RepEvent(
            exercise=self.definition.key,
            index=self._count,
            duration=duration,
            depth=depth,
            form_score=form_score,
            warnings=list(self._warnings),
        )
        self._history.append(event)
        self._last_rep_time = now
        self._rest_since = now
        self._warnings = []
        self._extreme_angle = None
        return event

    def _depth_ratio(self) -> float:
        """How far into the target range the rep actually went, 0..1."""
        if self._extreme_angle is None:
            return 0.0
        return clamp(
            inverse_lerp(self.definition.up_angle, self.definition.down_angle, self._extreme_angle),
            0.0, 1.0,
        )

    def _check_form(self, angles: Dict[str, float]) -> List[str]:
        """Evaluate the exercise's extra constraints for this frame."""
        warnings: List[str] = []
        for angle_name, low, high, message in self.definition.form_checks:
            value = angles.get(angle_name)
            if value is not None and not (low <= value <= high):
                warnings.append(message)
        return warnings

    def _reset_rep(self, now: float) -> None:
        self._state = "up"
        self._rep_started = None
        self._extreme_angle = None
        self._warnings = []

    def _reset_machine(self) -> None:
        self._state = "up"
        self._count = 0
        self._rep_started = None
        self._extreme_angle = None
        self._warnings = []

    # -- Set management -----------------------------------------------------

    def complete_set(self) -> int:
        """Bank the current reps as a set and start a fresh count."""
        if self._count:
            self._set_count += 1
        self._count = 0
        self._reset_machine()
        return self._set_count

    # -- Introspection ------------------------------------------------------

    @property
    def count(self) -> int:
        """Reps in the current set."""
        return self._count

    @property
    def total_reps(self) -> int:
        """Reps across every set this session."""
        return len(self._history)

    @property
    def sets(self) -> int:
        return self._set_count

    @property
    def state(self) -> str:
        return self._state

    @property
    def history(self) -> List[RepEvent]:
        return list(self._history)

    @property
    def progress(self) -> float:
        """
        How far the current rep has travelled, 0..1.

        Drives the on-screen progress ring during a rep.
        """
        if self._last_angle is None:
            return 0.0
        return clamp(
            inverse_lerp(self.definition.up_angle, self.definition.down_angle, self._last_angle),
            0.0, 1.0,
        )

    def rest_seconds(self, now: float) -> float:
        """Seconds since the last completed rep, 0 while mid-set."""
        if self._rest_since is None:
            return 0.0
        return now - self._rest_since

    def average_form(self) -> float:
        if not self._history:
            return 0.0
        return sum(e.form_score for e in self._history) / len(self._history)

    def average_tempo(self) -> float:
        if not self._history:
            return 0.0
        return sum(e.duration for e in self._history) / len(self._history)

    def clean_reps(self) -> int:
        return sum(1 for e in self._history if e.is_clean)

    def stats(self) -> Dict[str, Any]:
        """Session statistics for this exercise."""
        return {
            "exercise": self.definition.key,
            "label": self.definition.label,
            "current_set": self._count,
            "sets": self._set_count,
            "total_reps": self.total_reps,
            "clean_reps": self.clean_reps(),
            "average_form": round(self.average_form(), 1),
            "average_tempo": round(self.average_tempo(), 2),
            "muscle_group": self.definition.muscle_group,
        }

    def reset(self) -> None:
        self._reset_machine()
        self._set_count = 0
        self._history.clear()
        self._last_rep_time = None
        self._rest_since = None


# ---------------------------------------------------------------------------
# Multi-exercise tracking
# ---------------------------------------------------------------------------

class MultiExerciseCounter:
    """
    Runs several :class:`RepCounter` instances at once.

    Used by the workout mode so a session can mix exercises without the user
    having to declare which one they are doing — every counter watches the same
    angle stream and only the matching one advances.
    """

    def __init__(self, exercises: Optional[Sequence[str]] = None):
        keys = list(exercises) if exercises else ["squat", "pushup", "bicep_curl", "jumping_jack"]
        self._counters: Dict[str, RepCounter] = {k: RepCounter(k) for k in keys}
        self._events: List[RepEvent] = []

    def update(self, angles: Dict[str, float], now: float) -> List[RepEvent]:
        """Advance every counter; returns the reps completed this frame."""
        completed: List[RepEvent] = []
        for counter in self._counters.values():
            event = counter.update(angles, now)
            if event:
                completed.append(event)
                self._events.append(event)
        return completed

    def counter(self, exercise: str) -> Optional[RepCounter]:
        return self._counters.get(exercise)

    def add(self, exercise: str) -> RepCounter:
        """Start tracking an additional exercise."""
        if exercise not in self._counters:
            self._counters[exercise] = RepCounter(exercise)
        return self._counters[exercise]

    def remove(self, exercise: str) -> bool:
        return self._counters.pop(exercise, None) is not None

    @property
    def total_reps(self) -> int:
        return len(self._events)

    def active_exercise(self) -> Optional[str]:
        """Whichever exercise most recently completed a rep."""
        return self._events[-1].exercise if self._events else None

    def leaderboard(self) -> List[Tuple[str, int]]:
        """``(exercise, reps)`` sorted by volume, most first."""
        totals: Dict[str, int] = {}
        for event in self._events:
            totals[event.exercise] = totals.get(event.exercise, 0) + 1
        return sorted(totals.items(), key=lambda item: item[1], reverse=True)

    def stats(self) -> Dict[str, Any]:
        return {
            "total_reps": self.total_reps,
            "by_exercise": {k: c.stats() for k, c in self._counters.items() if c.total_reps},
            "leaderboard": self.leaderboard(),
        }

    def reset(self) -> None:
        for counter in self._counters.values():
            counter.reset()
        self._events.clear()


def count_reps_from_landmarks(
    counter: RepCounter,
    landmarks: Sequence[Any],
    now: float,
) -> Optional[RepEvent]:
    """Convenience wrapper: compute angles from landmarks and update *counter*."""
    return counter.update(compute_joint_angles(landmarks), now)
