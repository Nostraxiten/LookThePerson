"""
Fitness modes for LookThePerson.

Training-oriented behaviours built on ``analytics``: rep counting, workout
sessions, posture coaching, balance work, stretching and cardio.

These modes are where the joint-angle machinery pays off — everything shown is
computed from the live skeleton, nothing is faked.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2

from analytics.fitness import IntensityTracker, WorkoutSession
from analytics.motion import BalanceAnalyzer, StillnessDetector, symmetry_report
from analytics.posture import PostureMonitor
from analytics.reps import MultiExerciseCounter, RepCounter, exercise_names
from core.events import Events
from core.geometry import clamp, safe_int
from core.state import AppState, FrameContext
from fx.overlays import ParticleSystem, draw_progress_ring
from modes.base import Mode, ModeCategory

__all__ = [
    "RepCounterMode",
    "WorkoutMode",
    "PostureCoachMode",
    "BalanceMode",
    "StretchMode",
    "CardioMode",
    "fitness_modes",
]


class RepCounterMode(Mode):
    """
    Counts repetitions of one selected exercise.

    Cycle exercises with ``e``; the counter shows live depth as a progress
    ring and celebrates each completed rep.
    """

    key = "reps"
    label = "Contador de reps"
    description = "Cuenta repeticiones del ejercicio seleccionado"
    category = ModeCategory.FITNESS
    requires = ("pose",)
    toggles = {
        "skeleton": True, "segmentation": False, "face_mesh": False,
        "face_detect": False, "object_detect": False,
    }
    keys = {"e": "Cambiar ejercicio", "z": "Reiniciar cuenta", "x": "Cerrar serie"}

    def __init__(self, exercise: str = "squat") -> None:
        super().__init__()
        self.counter = RepCounter(exercise)
        self._exercises = exercise_names()
        self._particles = ParticleSystem(max_particles=140)
        self._last_event_time = 0.0

    # -- Lifecycle ----------------------------------------------------------

    def reset(self, state: AppState) -> None:
        self.counter.reset()
        self._particles.clear()

    def on_key(self, key: int, state: AppState) -> bool:
        if key == ord("e"):
            index = self._exercises.index(self.counter.exercise)
            nxt = self._exercises[(index + 1) % len(self._exercises)]
            self.counter.set_exercise(nxt)
            state.notify(f"Ejercicio: {self.counter.label}")
            return True
        if key == ord("z"):
            self.counter.reset()
            state.notify("Contador reiniciado")
            return True
        if key == ord("x"):
            sets = self.counter.complete_set()
            state.notify(f"Serie {sets} cerrada")
            return True
        return False

    # -- Per-frame ----------------------------------------------------------

    def process(self, ctx: FrameContext, state: AppState) -> None:
        self._particles.update(ctx.delta)
        if not ctx.angles:
            return

        event = self.counter.update(ctx.angles, ctx.now)
        if event is None:
            return

        state.increment("reps")
        state.set_gesture(f"REP {event.index}", ctx.now)
        state.bus.emit(
            Events.REP_COMPLETED,
            exercise=event.exercise, index=event.index,
            form=event.form_score, duration=event.duration,
        )
        self._last_event_time = ctx.now

        color = state.theme.good if event.is_clean else state.theme.warn
        if ctx.body_center:
            self._particles.burst(
                ctx.body_center[0] / ctx.width,
                ctx.body_center[1] / ctx.height,
                color,
            )

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        theme = state.theme
        self._particles.draw(ctx.frame)

        # Progress ring, bottom-right.
        center = (ctx.width - 90, ctx.height - 110)
        progress = self.counter.progress
        color = theme.good if self.counter.state == "down" else theme.accent
        draw_progress_ring(
            ctx.frame, center, 46, progress, color=color,
            background=(50, 50, 50), thickness=8, label=str(self.counter.count),
        )
        cv2.putText(
            ctx.frame, self.counter.label,
            (center[0] - 70, center[1] + 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, theme.text, 1, cv2.LINE_AA,
        )

        # Flash the most recent rep's warnings for a moment.
        history = self.counter.history
        if history and ctx.now - self._last_event_time < 2.0 and history[-1].warnings:
            for offset, warning in enumerate(history[-1].warnings[:2]):
                cv2.putText(
                    ctx.frame, warning, (20, ctx.height - 160 - offset * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, theme.warn, 2, cv2.LINE_AA,
                )

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        stats = self.counter.stats()
        lines = [
            f"{stats['label']}: {stats['current_set']} reps",
            f"Series: {stats['sets']}  Total: {stats['total_reps']}",
        ]
        if stats["total_reps"]:
            lines.append(
                f"Forma media: {stats['average_form']:.0f}%  "
                f"Ritmo: {stats['average_tempo']:.1f}s"
            )
        return lines

    def status_text(self, ctx: FrameContext, state: AppState) -> Optional[str]:
        if not ctx.has_pose:
            return "COLOCATE FRENTE A LA CAMARA"
        return f"{self.counter.label.upper()}: {self.counter.count}"


class WorkoutMode(Mode):
    """
    Full training session: detects several exercises at once, tracks sets,
    rest time and estimated calories.

    Unlike :class:`RepCounterMode` you never tell it what you are doing — every
    counter runs in parallel and the one that matches your movement advances.
    """

    key = "workout"
    label = "Entrenamiento"
    description = "Sesion completa: series, descanso y calorias"
    category = ModeCategory.FITNESS
    requires = ("pose",)
    toggles = {
        "skeleton": True, "segmentation": False, "face_mesh": False,
        "face_detect": False, "object_detect": False,
    }
    keys = {"x": "Cerrar serie", "z": "Reiniciar sesion"}

    def __init__(self) -> None:
        super().__init__()
        self.counters = MultiExerciseCounter(
            ["squat", "pushup", "bicep_curl", "jumping_jack", "shoulder_press", "sit_up"]
        )
        self.session: Optional[WorkoutSession] = None
        self.intensity = IntensityTracker()
        self._particles = ParticleSystem(max_particles=160)
        self._intensity_value = 0.0

    def on_enter(self, state: AppState) -> None:
        super().on_enter(state)
        if self.session is None:
            self.session = WorkoutSession(
                weight_kg=state.config.analytics.user_weight_kg,
            )

    def reset(self, state: AppState) -> None:
        self.counters.reset()
        self.intensity.reset()
        if self.session:
            self.session.reset()

    def on_key(self, key: int, state: AppState) -> bool:
        if key == ord("x") and self.session:
            closed = self.session.close_set()
            state.notify(
                f"Serie cerrada: {closed.reps} reps" if closed else "Sin reps que cerrar"
            )
            return True
        if key == ord("z"):
            self.reset(state)
            state.notify("Sesion reiniciada")
            return True
        return False

    def process(self, ctx: FrameContext, state: AppState) -> None:
        self._particles.update(ctx.delta)
        if self.session is None:
            self.session = WorkoutSession(weight_kg=state.config.analytics.user_weight_kg)

        self._intensity_value = self.intensity.update(ctx.motion.get("energy", 0.0), ctx.now)

        closed = self.session.tick(ctx.now)
        if closed:
            state.bus.emit(Events.SET_COMPLETED, exercise=closed.exercise, reps=closed.reps)
            state.notify(f"Serie: {closed.reps}x {closed.exercise}", "info")

        if not ctx.angles:
            return

        for event in self.counters.update(ctx.angles, ctx.now):
            self.session.record_rep(event, ctx.now)
            state.increment("reps")
            state.set_gesture(f"{event.exercise.upper()} {event.index}", ctx.now)
            state.bus.emit(
                Events.REP_COMPLETED,
                exercise=event.exercise, index=event.index, form=event.form_score,
            )
            if ctx.body_center:
                self._particles.burst(
                    ctx.body_center[0] / ctx.width,
                    ctx.body_center[1] / ctx.height,
                    state.theme.good,
                )

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        self._particles.draw(ctx.frame)
        theme = state.theme

        # Effort meter down the right edge.
        bar_x = ctx.width - 34
        bar_top = 110
        bar_height = 190
        cv2.rectangle(ctx.frame, (bar_x, bar_top), (bar_x + 16, bar_top + bar_height),
                      (45, 45, 45), -1)
        filled = int(bar_height * clamp(self._intensity_value, 0.0, 1.0))
        if filled > 0:
            color = theme.good if self._intensity_value < 0.6 else theme.warn
            cv2.rectangle(
                ctx.frame,
                (bar_x, bar_top + bar_height - filled),
                (bar_x + 16, bar_top + bar_height),
                color, -1,
            )
        cv2.rectangle(ctx.frame, (bar_x, bar_top), (bar_x + 16, bar_top + bar_height),
                      theme.text_dim, 1)
        cv2.putText(ctx.frame, "ESF", (bar_x - 6, bar_top - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, theme.text_dim, 1, cv2.LINE_AA)

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        if self.session is None:
            return []
        lines = self.session.hud_lines(ctx.now)
        lines.append(f"Zona: {self.intensity.zone(self._intensity_value)}")
        leaders = self.counters.leaderboard()[:3]
        if leaders:
            lines.append(
                "Top: " + ", ".join(f"{name} x{count}" for name, count in leaders)
            )
        return lines

    def on_exit(self, state: AppState) -> None:
        if self.session:
            self.session.close_set()


class PostureCoachMode(Mode):
    """
    Live posture feedback with alerts.

    Aimed at desk use: it grades your posture continuously and warns when you
    have been slouching for a while, naming the specific problem and the fix.
    """

    key = "posture"
    label = "Coach de postura"
    description = "Puntua la postura y avisa al encorvarte"
    category = ModeCategory.FITNESS
    requires = ("pose",)
    toggles = {
        "skeleton": True, "segmentation": False, "face_mesh": False,
        "face_detect": False, "object_detect": False,
    }
    keys = {"z": "Reiniciar estadisticas"}

    def __init__(self) -> None:
        super().__init__()
        self.monitor = PostureMonitor()
        self._alert_until = 0.0

    def reset(self, state: AppState) -> None:
        self.monitor.reset()

    def on_key(self, key: int, state: AppState) -> bool:
        if key == ord("z"):
            self.monitor.reset()
            state.notify("Estadisticas de postura reiniciadas")
            return True
        return False

    def process(self, ctx: FrameContext, state: AppState) -> None:
        if not ctx.has_pose:
            return
        report = self.monitor.update(ctx.primary_pose, ctx.now)
        ctx.posture = {
            "score": report.score, "grade": report.grade,
            "issues": [i.code for i in report.issues],
        }
        if self.monitor.should_alert(ctx.now):
            self._alert_until = ctx.now + 4.0
            worst = report.worst_issue
            message = worst.advice if worst else "Corrige la postura"
            state.notify(message, "warn")
            state.bus.emit(Events.POSTURE_ALERT, score=report.score, advice=message)

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        report = self.monitor.last_report
        if not report.valid:
            return

        theme = state.theme
        # Score gauge, top-centre.
        center = (ctx.width // 2, 92)
        color = (
            theme.good if report.score >= 80
            else theme.warn if report.score >= 60
            else theme.danger
        )
        draw_progress_ring(
            ctx.frame, center, 40, report.score / 100.0,
            color=color, thickness=7, label=report.grade,
        )

        # Highlight the worst issue's advice while an alert is fresh.
        if ctx.now < self._alert_until:
            worst = report.worst_issue
            if worst:
                text = worst.advice
                size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                x = (ctx.width - size[0]) // 2
                cv2.rectangle(
                    ctx.frame, (x - 12, ctx.height - 130),
                    (x + size[0] + 12, ctx.height - 96), (30, 30, 30), -1,
                )
                cv2.putText(
                    ctx.frame, text, (x, ctx.height - 106),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, theme.warn, 2, cv2.LINE_AA,
                )

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        report = self.monitor.last_report
        if not report.valid:
            return ["Postura: sin datos"]
        lines = [report.summary()]
        lines.append(
            f"Media sesion: {self.monitor.average_score:.0f}  "
            f"Buena: {self.monitor.good_ratio * 100:.0f}%"
        )
        for issue in report.issues[:2]:
            lines.append(f"  - {issue.label}")
        return lines


class BalanceMode(Mode):
    """
    Balance training with a live centre-of-mass marker and stability score.

    Also times how long you hold a stable position, which is the metric that
    matters for single-leg and yoga work.
    """

    key = "balance"
    label = "Equilibrio"
    description = "Centro de masa, estabilidad y tiempo aguantado"
    category = ModeCategory.FITNESS
    requires = ("pose",)
    toggles = {
        "skeleton": True, "segmentation": False, "face_mesh": False,
        "face_detect": False, "object_detect": False,
    }

    STABLE_SCORE = 75.0

    def __init__(self) -> None:
        super().__init__()
        self.analyzer = BalanceAnalyzer()
        self._hold_since: Optional[float] = None
        self._best_hold = 0.0
        self._last_report = None

    def reset(self, state: AppState) -> None:
        self.analyzer.reset()
        self._hold_since = None
        self._best_hold = 0.0

    def process(self, ctx: FrameContext, state: AppState) -> None:
        if not ctx.has_pose:
            self._hold_since = None
            return

        report = self.analyzer.update(ctx.primary_pose)
        self._last_report = report
        if not report.valid:
            return

        if report.stability >= self.STABLE_SCORE:
            self._hold_since = self._hold_since or ctx.now
            self._best_hold = max(self._best_hold, ctx.now - self._hold_since)
        else:
            self._hold_since = None

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        report = self._last_report
        if not report or not report.valid or not report.center_of_mass:
            return

        theme = state.theme
        # The centre of mass is an average over landmarks, so a single NaN
        # landmark makes the whole thing NaN.
        if not ctx.usable(report.center_of_mass):
            return
        cx, cy = ctx.px(report.center_of_mass)
        color = theme.good if report.stability >= self.STABLE_SCORE else theme.warn

        # Centre of mass marker with crosshairs.
        cv2.circle(ctx.frame, (cx, cy), 12, color, 2, cv2.LINE_AA)
        cv2.circle(ctx.frame, (cx, cy), 3, color, -1, cv2.LINE_AA)
        cv2.line(ctx.frame, (cx - 22, cy), (cx - 14, cy), color, 2)
        cv2.line(ctx.frame, (cx + 14, cy), (cx + 22, cy), color, 2)

        # Support base line at the feet.
        if report.base_center is not None:
            bx = safe_int(report.base_center * ctx.width)
            cv2.line(ctx.frame, (bx, cy), (bx, ctx.height - 10), theme.text_dim, 1)
            cv2.line(ctx.frame, (bx - 40, ctx.height - 12), (bx + 40, ctx.height - 12),
                     theme.accent, 2)

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        report = self._last_report
        if not report or not report.valid:
            return ["Equilibrio: sin datos"]
        current_hold = (ctx.now - self._hold_since) if self._hold_since else 0.0
        return [
            f"Estabilidad: {report.stability:.0f} ({report.grade})",
            f"Desviacion: {report.offset:.2f}  Balanceo: {report.sway * 1000:.1f}",
            f"Aguante: {current_hold:.1f}s  Mejor: {self._best_hold:.1f}s",
        ]


class StretchMode(Mode):
    """
    Guided stretching: hold a position still and the timer counts up.

    Reports left/right symmetry so you can tell when one side is tighter than
    the other.
    """

    key = "stretch"
    label = "Estiramientos"
    description = "Cronometra posturas mantenidas y mide simetria"
    category = ModeCategory.FITNESS
    requires = ("pose",)
    toggles = {
        "skeleton": True, "segmentation": False, "face_mesh": False,
        "face_detect": False, "object_detect": False,
    }
    keys = {"z": "Reiniciar"}

    TARGET_SECONDS = 30.0

    def __init__(self) -> None:
        super().__init__()
        self.stillness = StillnessDetector(threshold=0.02, hold_seconds=1.0)
        self._holding = False
        self._completed = 0
        self._symmetry: Dict[str, Any] = {}

    def reset(self, state: AppState) -> None:
        self.stillness.reset()
        self._completed = 0
        self._holding = False

    def on_key(self, key: int, state: AppState) -> bool:
        if key == ord("z"):
            self.reset(state)
            state.notify("Estiramientos reiniciados")
            return True
        return False

    def process(self, ctx: FrameContext, state: AppState) -> None:
        if not ctx.has_pose:
            self.stillness.reset()
            self._holding = False
            return

        was_holding = self._holding
        self._holding = self.stillness.update(ctx.motion.get("energy", 1.0), ctx.now)
        self._symmetry = symmetry_report(ctx.angles)

        held = self.stillness.still_seconds(ctx.now)
        if was_holding and not self._holding and held >= self.TARGET_SECONDS:
            self._completed += 1
            state.notify(f"Estiramiento completado ({self._completed})", "info")

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        theme = state.theme
        held = self.stillness.still_seconds(ctx.now)
        progress = clamp(held / self.TARGET_SECONDS, 0.0, 1.0)
        color = theme.good if progress >= 1.0 else theme.accent
        draw_progress_ring(
            ctx.frame, (ctx.width - 88, 150), 42, progress,
            color=color, thickness=7, label=f"{held:.0f}s",
        )

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        lines = [
            f"Mantenido: {self.stillness.still_seconds(ctx.now):.1f}s / {self.TARGET_SECONDS:.0f}s",
            f"Completados: {self._completed}",
        ]
        if self._symmetry.get("valid"):
            lines.append(f"Simetria: {self._symmetry['score']:.0f}%")
            asymmetric = self._symmetry.get("asymmetric")
            if asymmetric:
                lines.append("Desigual: " + ", ".join(asymmetric[:3]))
        return lines


class CardioMode(Mode):
    """
    Cardio session driven by movement intensity rather than repetitions.

    Tracks time in each effort zone, which is the useful summary for interval
    training and warm-ups.
    """

    key = "cardio"
    label = "Cardio"
    description = "Intensidad, zonas de esfuerzo y calorias"
    category = ModeCategory.FITNESS
    requires = ("pose",)
    toggles = {
        "skeleton": True, "segmentation": True, "face_mesh": False,
        "face_detect": False, "object_detect": False,
    }

    def __init__(self) -> None:
        super().__init__()
        self.intensity = IntensityTracker(window_seconds=8.0)
        self._value = 0.0
        self._calories = 0.0
        self._peak = 0.0

    def reset(self, state: AppState) -> None:
        self.intensity.reset()
        self._calories = 0.0
        self._peak = 0.0

    def process(self, ctx: FrameContext, state: AppState) -> None:
        self._value = self.intensity.update(ctx.motion.get("energy", 0.0), ctx.now)
        self._peak = max(self._peak, self._value)

        # MET scales with intensity: idle ~2, all-out ~10.
        met = 2.0 + self._value * 8.0
        weight = state.config.analytics.user_weight_kg
        self._calories += met * 3.5 * weight / 200.0 * (ctx.delta / 60.0)

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        theme = state.theme
        zone = self.intensity.zone(self._value)
        color = theme.good if self._value < 0.6 else theme.warn if self._value < 0.85 else theme.danger

        # Wide intensity bar across the bottom.
        bar_y = ctx.height - 60
        bar_w = ctx.width - 80
        cv2.rectangle(ctx.frame, (40, bar_y), (40 + bar_w, bar_y + 18), (45, 45, 45), -1)
        cv2.rectangle(
            ctx.frame, (40, bar_y),
            (40 + int(bar_w * clamp(self._value, 0.0, 1.0)), bar_y + 18), color, -1,
        )
        cv2.rectangle(ctx.frame, (40, bar_y), (40 + bar_w, bar_y + 18), theme.text_dim, 1)
        cv2.putText(
            ctx.frame, zone.upper(), (40, bar_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        lines = [
            f"Intensidad: {self._value * 100:.0f}%  Pico: {self._peak * 100:.0f}%",
            f"Kcal: {self._calories:.1f}",
        ]
        zones = self.intensity.zone_breakdown()
        if zones:
            top = sorted(zones.items(), key=lambda item: item[1], reverse=True)[:3]
            lines.append("Zonas: " + ", ".join(f"{k} {v:.0f}s" for k, v in top))
        return lines


def fitness_modes() -> List[Mode]:
    """Every fitness mode, in menu order."""
    return [
        RepCounterMode(), WorkoutMode(), PostureCoachMode(),
        BalanceMode(), StretchMode(), CardioMode(),
    ]
