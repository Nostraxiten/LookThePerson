"""
Workout session tracking for LookThePerson.

Wraps rep counting with the things that make a workout a workout: sets, rest
timers, calorie estimation, intensity zones and a summary you can export.

Calorie figures use the standard MET formula and are estimates, not medical
measurements.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from analytics.reps import EXERCISES, RepEvent
from core.geometry import clamp

__all__ = [
    "calories_burned",
    "WorkoutSet",
    "WorkoutSession",
    "IntensityTracker",
    "estimate_intensity",
]


# ---------------------------------------------------------------------------
# Energy expenditure
# ---------------------------------------------------------------------------

def calories_burned(met: float, weight_kg: float, seconds: float) -> float:
    """
    Kilocalories burned, from the standard MET equation.

    ``kcal/min = MET * 3.5 * weight_kg / 200``
    """
    if seconds <= 0 or weight_kg <= 0:
        return 0.0
    minutes = seconds / 60.0
    return met * 3.5 * weight_kg / 200.0 * minutes


def estimate_intensity(reps_per_minute: float, met: float) -> str:
    """Classify effort as ``"ligera"``, ``"moderada"`` or ``"intensa"``."""
    score = reps_per_minute * max(met, 1.0)
    if score < 40:
        return "ligera"
    if score < 110:
        return "moderada"
    return "intensa"


# ---------------------------------------------------------------------------
# Sets
# ---------------------------------------------------------------------------

@dataclass
class WorkoutSet:
    """One completed set of a single exercise."""

    exercise: str
    reps: int
    started_at: float
    ended_at: float
    average_form: float = 100.0
    warnings: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.ended_at - self.started_at)

    @property
    def tempo(self) -> float:
        """Average seconds per rep."""
        return self.duration / self.reps if self.reps else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exercise": self.exercise,
            "reps": self.reps,
            "duration": round(self.duration, 1),
            "tempo": round(self.tempo, 2),
            "average_form": round(self.average_form, 1),
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class WorkoutSession:
    """
    Accumulates a whole training session.

    A set closes automatically after *rest_threshold* seconds without a rep,
    which matches how people actually train — they stop, breathe, and start
    the next set without pressing anything.
    """

    def __init__(
        self,
        weight_kg: float = 70.0,
        rest_threshold: float = 12.0,
        started_at: Optional[float] = None,
    ):
        self.weight_kg = max(1.0, weight_kg)
        self.rest_threshold = rest_threshold
        self.started_at = started_at if started_at is not None else time.monotonic()

        self._sets: List[WorkoutSet] = []
        self._current_exercise: Optional[str] = None
        self._current_reps: List[RepEvent] = []
        self._current_started: Optional[float] = None
        self._last_rep_at: Optional[float] = None
        self._calories = 0.0
        self._active_seconds = 0.0
        self._paused = False

    # -- Recording ----------------------------------------------------------

    def record_rep(self, event: RepEvent, now: Optional[float] = None) -> None:
        """Add a completed repetition to the session."""
        now = time.monotonic() if now is None else now

        # Switching exercise closes the previous set.
        if self._current_exercise and event.exercise != self._current_exercise:
            self.close_set(now)

        if self._current_exercise is None:
            self._current_exercise = event.exercise
            self._current_started = now - event.duration

        self._current_reps.append(event)
        self._last_rep_at = now

        definition = EXERCISES.get(event.exercise)
        met = definition.met if definition else 5.0
        self._calories += calories_burned(met, self.weight_kg, event.duration)
        self._active_seconds += event.duration

    def tick(self, now: Optional[float] = None) -> Optional[WorkoutSet]:
        """
        Call once per frame; closes the current set after enough rest.

        Returns the set that was closed, if any.
        """
        now = time.monotonic() if now is None else now
        if (
            self._current_reps
            and self._last_rep_at is not None
            and now - self._last_rep_at >= self.rest_threshold
        ):
            return self.close_set(now)
        return None

    def close_set(self, now: Optional[float] = None) -> Optional[WorkoutSet]:
        """Force the current set closed and bank it."""
        if not self._current_reps or self._current_exercise is None:
            return None
        now = time.monotonic() if now is None else now

        warnings: List[str] = []
        for rep in self._current_reps:
            for warning in rep.warnings:
                if warning not in warnings:
                    warnings.append(warning)

        completed = WorkoutSet(
            exercise=self._current_exercise,
            reps=len(self._current_reps),
            started_at=self._current_started or now,
            ended_at=self._last_rep_at or now,
            average_form=sum(r.form_score for r in self._current_reps) / len(self._current_reps),
            warnings=warnings,
        )
        self._sets.append(completed)
        self._current_reps = []
        self._current_exercise = None
        self._current_started = None
        return completed

    # -- Live state ---------------------------------------------------------

    @property
    def current_exercise(self) -> Optional[str]:
        return self._current_exercise

    @property
    def current_reps(self) -> int:
        return len(self._current_reps)

    @property
    def sets(self) -> List[WorkoutSet]:
        return list(self._sets)

    @property
    def total_sets(self) -> int:
        return len(self._sets)

    @property
    def total_reps(self) -> int:
        return sum(s.reps for s in self._sets) + len(self._current_reps)

    @property
    def calories(self) -> float:
        return self._calories

    def duration(self, now: Optional[float] = None) -> float:
        now = time.monotonic() if now is None else now
        return max(0.0, now - self.started_at)

    def rest_time(self, now: Optional[float] = None) -> float:
        """Seconds since the last rep — the live rest timer."""
        if self._last_rep_at is None:
            return 0.0
        now = time.monotonic() if now is None else now
        return max(0.0, now - self._last_rep_at)

    def reps_per_minute(self, now: Optional[float] = None) -> float:
        duration = self.duration(now)
        if duration <= 0:
            return 0.0
        return self.total_reps / (duration / 60.0)

    def intensity(self, now: Optional[float] = None) -> str:
        exercise = self._current_exercise or (self._sets[-1].exercise if self._sets else None)
        definition = EXERCISES.get(exercise or "")
        met = definition.met if definition else 5.0
        return estimate_intensity(self.reps_per_minute(now), met)

    def work_ratio(self, now: Optional[float] = None) -> float:
        """Fraction of session time actually spent moving, 0..1."""
        duration = self.duration(now)
        return clamp(self._active_seconds / duration, 0.0, 1.0) if duration > 0 else 0.0

    def volume_by_muscle(self) -> Dict[str, int]:
        """Total reps grouped by muscle group."""
        totals: Dict[str, int] = {}
        for workout_set in self._sets:
            definition = EXERCISES.get(workout_set.exercise)
            group = definition.muscle_group if definition else "general"
            totals[group] = totals.get(group, 0) + workout_set.reps
        return totals

    # -- Reporting ----------------------------------------------------------

    def hud_lines(self, now: Optional[float] = None) -> List[str]:
        """Compact live readout for the HUD."""
        lines = [
            f"Reps: {self.total_reps}  Series: {self.total_sets}",
            f"Kcal: {self._calories:.1f}  Intensidad: {self.intensity(now)}",
        ]
        if self._current_exercise:
            definition = EXERCISES.get(self._current_exercise)
            label = definition.label if definition else self._current_exercise
            lines.insert(0, f"{label}: {len(self._current_reps)}")
        rest = self.rest_time(now)
        if rest >= 3.0 and not self._current_reps:
            lines.append(f"Descanso: {rest:.0f}s")
        return lines

    def summary(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Full session summary, ready to be exported."""
        return {
            "duration_seconds": round(self.duration(now), 1),
            "total_reps": self.total_reps,
            "total_sets": self.total_sets,
            "calories": round(self._calories, 1),
            "reps_per_minute": round(self.reps_per_minute(now), 1),
            "intensity": self.intensity(now),
            "work_ratio": round(self.work_ratio(now), 3),
            "volume_by_muscle": self.volume_by_muscle(),
            "sets": [s.to_dict() for s in self._sets],
        }

    def reset(self, now: Optional[float] = None) -> None:
        self._sets.clear()
        self._current_reps.clear()
        self._current_exercise = None
        self._current_started = None
        self._last_rep_at = None
        self._calories = 0.0
        self._active_seconds = 0.0
        self.started_at = now if now is not None else time.monotonic()


# ---------------------------------------------------------------------------
# Intensity over time
# ---------------------------------------------------------------------------

class IntensityTracker:
    """
    Tracks movement intensity in a rolling window.

    Feeds the effort meter and lets cardio modes show a zone without any
    wearable hardware — it is derived purely from how much the body moves.
    """

    ZONES = (
        (0.15, "reposo"),
        (0.35, "ligera"),
        (0.60, "moderada"),
        (0.85, "vigorosa"),
        (1.01, "maxima"),
    )

    def __init__(self, window_seconds: float = 10.0):
        self.window_seconds = max(1.0, window_seconds)
        self._samples: List[tuple] = []   # (timestamp, motion_energy)
        self._peak = 1e-6
        self._zone_seconds: Dict[str, float] = {}
        self._last_update: Optional[float] = None

    def update(self, motion_energy: float, now: float) -> float:
        """
        Add a motion-energy sample and return the normalized intensity, 0..1.

        The scale is self-calibrating: the running peak defines "maximum",
        so it adapts to whatever range the current camera and framing produce.
        """
        self._samples.append((now, max(0.0, motion_energy)))
        cutoff = now - self.window_seconds
        self._samples = [s for s in self._samples if s[0] >= cutoff]

        self._peak = max(self._peak * 0.999, motion_energy)  # slow decay
        average = sum(s[1] for s in self._samples) / len(self._samples)
        intensity = clamp(average / self._peak, 0.0, 1.0) if self._peak > 0 else 0.0

        if self._last_update is not None:
            elapsed = max(0.0, now - self._last_update)
            zone = self.zone(intensity)
            self._zone_seconds[zone] = self._zone_seconds.get(zone, 0.0) + elapsed
        self._last_update = now
        return intensity

    def zone(self, intensity: float) -> str:
        """Name the effort zone for a 0..1 intensity."""
        for threshold, name in self.ZONES:
            if intensity < threshold:
                return name
        return self.ZONES[-1][1]

    def zone_breakdown(self) -> Dict[str, float]:
        """Seconds spent in each zone."""
        return {k: round(v, 1) for k, v in self._zone_seconds.items()}

    def reset(self) -> None:
        self._samples.clear()
        self._zone_seconds.clear()
        self._peak = 1e-6
        self._last_update = None
