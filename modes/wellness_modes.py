"""
Wellness modes for LookThePerson.

Screen-time and wellbeing behaviours: drowsiness monitoring, focus tracking,
guided breathing, ergonomic break reminders and a meditation timer.

These lean on the face-mesh metrics (EAR, PERCLOS, head pose, gaze) and on
stillness detection.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import cv2

from analytics.angles import PoseLandmark as L
from analytics.face_metrics import (
    AttentionTracker,
    BlinkDetector,
    DrowsinessMonitor,
    eye_aspect_ratio,
    face_size,
    head_pose,
    mouth_aspect_ratio,
)
from analytics.motion import StillnessDetector
from analytics.posture import PostureMonitor
from core.events import Events
from core.geometry import clamp
from core.state import AppState, FrameContext
from fx.overlays import draw_progress_ring
from modes.base import Mode, ModeCategory

__all__ = [
    "DrowsinessMode",
    "FocusMode",
    "BreathingMode",
    "ErgonomicsMode",
    "MeditationMode",
    "wellness_modes",
]


class DrowsinessMode(Mode):
    """
    Fatigue monitor based on eye closure and yawning.

    Tracks PERCLOS — the share of recent time with the eyes closed — which is
    the standard fatigue measure, and raises an alert when it crosses the
    drowsy threshold.
    """

    key = "drowsiness"
    label = "Somnolencia"
    description = "Vigila fatiga: PERCLOS, parpadeo y bostezos"
    category = ModeCategory.WELLNESS
    requires = ("face_mesh",)
    toggles = {
        "skeleton": False, "segmentation": False, "face_mesh": True,
        "face_detect": False, "object_detect": False,
    }

    def __init__(self) -> None:
        super().__init__()
        self.blinks = BlinkDetector()
        self.monitor = DrowsinessMonitor()
        self._report = None
        self._alert_until = 0.0

    def reset(self, state: AppState) -> None:
        self.blinks.reset()
        self.monitor.reset()

    def process(self, ctx: FrameContext, state: AppState) -> None:
        face = ctx.primary_face
        if face is None:
            return

        left = eye_aspect_ratio(face, "left")
        right = eye_aspect_ratio(face, "right")
        ear = None
        if left is not None and right is not None:
            ear = (left + right) / 2.0
        elif left is not None or right is not None:
            ear = left if left is not None else right

        mar = mouth_aspect_ratio(face)

        if self.blinks.update(ear, ctx.now):
            state.increment("blinks")
            state.bus.emit(Events.BLINK, count=self.blinks.count)

        self._report = self.monitor.update(ear, mar, ctx.now)
        if self._report.alert:
            self._alert_until = ctx.now + 5.0
            state.notify("DESPIERTA — señales de somnolencia", "danger")
            state.bus.emit(Events.DROWSINESS_ALERT, perclos=self._report.perclos)

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        report = self._report
        if report is None:
            return
        theme = state.theme

        color = {
            "alerta": theme.good, "cansancio": theme.warn, "somnolencia": theme.danger,
        }.get(report.level, theme.text)
        draw_progress_ring(
            ctx.frame, (ctx.width - 88, 150), 42, report.severity,
            color=color, thickness=7, label=f"{report.perclos * 100:.0f}%",
        )

        # Full-frame red border while the alert is live.
        if ctx.now < self._alert_until:
            pulse = 0.5 + 0.5 * math.sin(ctx.now * 12.0)
            thickness = int(8 + pulse * 10)
            cv2.rectangle(
                ctx.frame, (0, 0), (ctx.width - 1, ctx.height - 1),
                theme.danger, thickness,
            )
            text = "SOMNOLENCIA DETECTADA"
            size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            cv2.putText(
                ctx.frame, text, ((ctx.width - size[0]) // 2, ctx.height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, theme.danger, 3, cv2.LINE_AA,
            )

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        report = self._report
        if report is None:
            return ["Somnolencia: sin cara detectada"]
        return [
            f"Estado: {report.level.upper()}",
            f"PERCLOS: {report.perclos * 100:.0f}%  EAR: {self.blinks.ear:.2f}",
            f"Parpadeos: {self.blinks.count}  ({self.blinks.blink_rate(ctx.now):.0f}/min)",
            f"Bostezos: {report.yawn_count}",
        ]

    def status_text(self, ctx: FrameContext, state: AppState) -> Optional[str]:
        if self._report is None:
            return None
        return f"VIGILANCIA: {self._report.level.upper()}"


class FocusMode(Mode):
    """
    Attention and screen-time tracker with a Pomodoro-style timer.

    Attention is credited only while you are actually facing the screen, so
    the timer measures focused work rather than elapsed time.
    """

    key = "focus"
    label = "Concentracion"
    description = "Mide atencion real y ciclos de trabajo"
    category = ModeCategory.WELLNESS
    requires = ("face_mesh",)
    toggles = {
        "skeleton": False, "segmentation": False, "face_mesh": True,
        "face_detect": False, "object_detect": False,
    }
    keys = {"z": "Reiniciar ciclo"}

    WORK_SECONDS = 25 * 60.0
    BREAK_SECONDS = 5 * 60.0

    def __init__(self) -> None:
        super().__init__()
        self.tracker = AttentionTracker()
        self._phase = "trabajo"
        self._phase_elapsed = 0.0
        self._cycles = 0

    def reset(self, state: AppState) -> None:
        self.tracker.reset()
        self._phase = "trabajo"
        self._phase_elapsed = 0.0
        self._cycles = 0

    def on_key(self, key: int, state: AppState) -> bool:
        if key == ord("z"):
            self.reset(state)
            state.notify("Ciclo reiniciado")
            return True
        return False

    def process(self, ctx: FrameContext, state: AppState) -> None:
        face = ctx.primary_face
        pose = head_pose(face) if face is not None else None
        gaze = ctx.face_info.get("gaze") if ctx.face_info else None
        attentive = self.tracker.update(pose, gaze, ctx.now)

        # Only attentive time advances the work phase; breaks run on real time.
        if self._phase == "trabajo":
            if attentive:
                self._phase_elapsed += ctx.delta
            if self._phase_elapsed >= self.WORK_SECONDS:
                self._phase = "descanso"
                self._phase_elapsed = 0.0
                self._cycles += 1
                state.notify("Ciclo completado — descansa 5 min", "info")
        else:
            self._phase_elapsed += ctx.delta
            if self._phase_elapsed >= self.BREAK_SECONDS:
                self._phase = "trabajo"
                self._phase_elapsed = 0.0
                state.notify("Descanso terminado — a trabajar", "info")

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        theme = state.theme
        target = self.WORK_SECONDS if self._phase == "trabajo" else self.BREAK_SECONDS
        progress = clamp(self._phase_elapsed / target, 0.0, 1.0)
        remaining = max(0.0, target - self._phase_elapsed)
        minutes, seconds = divmod(int(remaining), 60)

        color = theme.accent if self._phase == "trabajo" else theme.good
        draw_progress_ring(
            ctx.frame, (ctx.width // 2, 96), 44, progress,
            color=color, thickness=7, label=f"{minutes}:{seconds:02d}",
        )

        indicator = theme.good if self.tracker.is_attentive else theme.text_dim
        cv2.circle(ctx.frame, (ctx.width // 2, 152), 6, indicator, -1, cv2.LINE_AA)

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        stats = self.tracker.stats(ctx.now)
        return [
            f"Fase: {self._phase}  Ciclos: {self._cycles}",
            f"Atencion: {stats['attention_ratio'] * 100:.0f}%  "
            f"Distracciones: {stats['distractions']}",
            f"Racha actual: {self.tracker.current_focus_streak(ctx.now):.0f}s  "
            f"Mejor: {stats['longest_focus']:.0f}s",
        ]


class BreathingMode(Mode):
    """
    Guided breathing with an animated pacer.

    Follows the 4-7-8 pattern by default: inhale 4s, hold 7s, exhale 8s. The
    ring expands and contracts to pace you, and the shoulder-line movement is
    used to estimate your actual breathing rate.
    """

    key = "breathing"
    label = "Respiracion"
    description = "Guia de respiracion 4-7-8 con estimacion de ritmo"
    category = ModeCategory.WELLNESS
    requires = ("pose",)
    toggles = {
        "skeleton": False, "segmentation": False, "face_mesh": False,
        "face_detect": False, "object_detect": False, "telemetry": True,
    }
    keys = {"b": "Cambiar patron"}

    PATTERNS = {
        "4-7-8": (4.0, 7.0, 8.0, 0.0),
        "caja": (4.0, 4.0, 4.0, 4.0),
        "calma": (5.0, 0.0, 5.0, 0.0),
    }

    def __init__(self) -> None:
        super().__init__()
        self._pattern_name = "4-7-8"
        self._phase_index = 0
        self._phase_start = 0.0
        self._cycles = 0
        self._shoulder_history: List[float] = []
        self._rate = 0.0

    @property
    def _phases(self):
        inhale, hold, exhale, hold_out = self.PATTERNS[self._pattern_name]
        return [
            ("Inspira", inhale), ("Manten", hold),
            ("Espira", exhale), ("Manten", hold_out),
        ]

    def on_enter(self, state: AppState) -> None:
        super().on_enter(state)
        self._phase_start = state.uptime
        self._phase_index = 0

    def on_key(self, key: int, state: AppState) -> bool:
        if key == ord("b"):
            names = list(self.PATTERNS)
            index = names.index(self._pattern_name)
            self._pattern_name = names[(index + 1) % len(names)]
            self._phase_index = 0
            state.notify(f"Patron: {self._pattern_name}")
            return True
        return False

    def process(self, ctx: FrameContext, state: AppState) -> None:
        # Advance the pacer, skipping zero-length phases.
        phases = self._phases
        for _ in range(len(phases)):
            _name, duration = phases[self._phase_index]
            if duration > 0 and ctx.now - self._phase_start < duration:
                break
            self._phase_start = ctx.now
            self._phase_index = (self._phase_index + 1) % len(phases)
            if self._phase_index == 0:
                self._cycles += 1

        # Estimate real breathing rate from vertical shoulder movement.
        if ctx.has_pose:
            landmarks = ctx.primary_pose
            shoulder_y = (landmarks[L.LEFT_SHOULDER].y + landmarks[L.RIGHT_SHOULDER].y) / 2.0
            self._shoulder_history.append(shoulder_y)
            if len(self._shoulder_history) > 300:
                self._shoulder_history.pop(0)
            self._rate = self._estimate_rate(ctx)

    def _estimate_rate(self, ctx: FrameContext) -> float:
        """Breaths per minute from zero-crossings of the shoulder signal."""
        samples = self._shoulder_history
        if len(samples) < 60:
            return 0.0
        mean = sum(samples) / len(samples)
        crossings = 0
        above = samples[0] > mean
        for value in samples[1:]:
            now_above = value > mean
            if now_above != above:
                crossings += 1
                above = now_above
        fps = 1.0 / ctx.delta if ctx.delta > 1e-6 else 30.0
        seconds = len(samples) / fps
        return (crossings / 2.0) / (seconds / 60.0) if seconds > 0 else 0.0

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        theme = state.theme
        name, duration = self._phases[self._phase_index]
        elapsed = ctx.now - self._phase_start
        progress = clamp(elapsed / duration, 0.0, 1.0) if duration > 0 else 0.0

        # Ring size follows the phase: grows on inhale, shrinks on exhale.
        base_radius = 60
        span = 70
        if name == "Inspira":
            radius = int(base_radius + span * progress)
        elif name == "Espira":
            radius = int(base_radius + span * (1.0 - progress))
        else:
            radius = base_radius + span if self._phase_index == 1 else base_radius

        center = (ctx.width // 2, ctx.height // 2)
        cv2.circle(ctx.frame, center, radius, theme.accent, 3, cv2.LINE_AA)
        cv2.circle(ctx.frame, center, radius + 14, tuple(int(c * 0.3) for c in theme.accent),
                   2, cv2.LINE_AA)

        size, _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        cv2.putText(
            ctx.frame, name,
            (center[0] - size[0] // 2, center[1] + size[1] // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, theme.text, 3, cv2.LINE_AA,
        )
        countdown = f"{max(0.0, duration - elapsed):.0f}"
        cv2.putText(
            ctx.frame, countdown, (center[0] - 10, center[1] + radius + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, theme.text_dim, 2, cv2.LINE_AA,
        )

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        lines = [f"Patron: {self._pattern_name}  Ciclos: {self._cycles}"]
        if self._rate > 0:
            lines.append(f"Ritmo estimado: {self._rate:.1f} resp/min")
        return lines


class ErgonomicsMode(Mode):
    """
    Desk ergonomics watchdog.

    Combines posture scoring with a screen-distance estimate and a break timer,
    then nags you about whichever is worst.
    """

    key = "ergonomics"
    label = "Ergonomia"
    description = "Postura, distancia a pantalla y pausas"
    category = ModeCategory.WELLNESS
    requires = ("pose", "face_mesh")
    toggles = {
        "skeleton": True, "segmentation": False, "face_mesh": True,
        "face_detect": False, "object_detect": False,
    }

    BREAK_INTERVAL = 20 * 60.0     # the 20-20-20 rule
    TOO_CLOSE = 0.42               # face height as a share of frame height

    def __init__(self) -> None:
        super().__init__()
        self.posture = PostureMonitor(alert_seconds=8.0, alert_cooldown=60.0)
        self._last_break = 0.0
        self._face_ratio = 0.0
        self._close_since: Optional[float] = None

    def on_enter(self, state: AppState) -> None:
        super().on_enter(state)
        self._last_break = state.uptime

    def process(self, ctx: FrameContext, state: AppState) -> None:
        if ctx.has_pose:
            self.posture.update(ctx.primary_pose, ctx.now)
            if self.posture.should_alert(ctx.now):
                worst = self.posture.last_report.worst_issue
                state.notify(worst.advice if worst else "Corrige la postura", "warn")

        face = ctx.primary_face
        if face is not None:
            self._face_ratio = face_size(face)
            if self._face_ratio > self.TOO_CLOSE:
                self._close_since = self._close_since or ctx.now
                if ctx.now - self._close_since > 10.0:
                    state.notify("Estas muy cerca de la pantalla", "warn")
                    self._close_since = ctx.now + 30.0   # re-arm well ahead
            else:
                self._close_since = None

        if ctx.now - self._last_break >= self.BREAK_INTERVAL:
            self._last_break = ctx.now
            state.notify("Pausa: mira 20s a 6 metros", "info")

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        report = self.posture.last_report
        next_break = max(0.0, self.BREAK_INTERVAL - (ctx.now - self._last_break))
        minutes, seconds = divmod(int(next_break), 60)
        lines = [f"Proxima pausa: {minutes}:{seconds:02d}"]
        if report.valid:
            lines.append(f"Postura: {report.score:.0f} ({report.grade})")
        if self._face_ratio:
            distance = "muy cerca" if self._face_ratio > self.TOO_CLOSE else "correcta"
            lines.append(f"Distancia: {distance}")
        return lines


class MeditationMode(Mode):
    """
    Stillness-based meditation timer.

    The timer only advances while you stay still, so fidgeting genuinely
    costs you progress.
    """

    key = "meditation"
    label = "Meditacion"
    description = "Temporizador que solo avanza en quietud"
    category = ModeCategory.WELLNESS
    requires = ("pose",)
    toggles = {
        "skeleton": False, "segmentation": False, "face_mesh": False,
        "face_detect": False, "object_detect": False, "telemetry": False,
    }
    keys = {"z": "Reiniciar", "t": "Cambiar duracion"}

    DURATIONS = (60.0, 180.0, 300.0, 600.0)

    def __init__(self) -> None:
        super().__init__()
        self.stillness = StillnessDetector(threshold=0.015, hold_seconds=0.8)
        self._duration_index = 1
        self._accumulated = 0.0
        self._completed = 0

    @property
    def target(self) -> float:
        return self.DURATIONS[self._duration_index]

    def reset(self, state: AppState) -> None:
        self._accumulated = 0.0
        self.stillness.reset()

    def on_key(self, key: int, state: AppState) -> bool:
        if key == ord("z"):
            self.reset(state)
            state.notify("Sesion reiniciada")
            return True
        if key == ord("t"):
            self._duration_index = (self._duration_index + 1) % len(self.DURATIONS)
            self._accumulated = 0.0
            state.notify(f"Duracion: {self.target / 60:.0f} min")
            return True
        return False

    def process(self, ctx: FrameContext, state: AppState) -> None:
        still = self.stillness.update(ctx.motion.get("energy", 1.0), ctx.now)
        if still:
            self._accumulated += ctx.delta
            if self._accumulated >= self.target:
                self._completed += 1
                self._accumulated = 0.0
                state.notify("Sesion completada", "info")

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        theme = state.theme
        progress = clamp(self._accumulated / self.target, 0.0, 1.0)
        remaining = max(0.0, self.target - self._accumulated)
        minutes, seconds = divmod(int(remaining), 60)

        still = self.stillness.still_seconds(ctx.now) > 0.8
        color = theme.good if still else theme.text_dim

        # Soft breathing halo behind the timer.
        center = (ctx.width // 2, ctx.height // 2)
        pulse = int(6 * (0.5 + 0.5 * math.sin(ctx.now * 0.8)))
        cv2.circle(ctx.frame, center, 108 + pulse,
                   tuple(int(c * 0.25) for c in color), 2, cv2.LINE_AA)
        draw_progress_ring(
            ctx.frame, center, 92, progress, color=color,
            background=(40, 40, 40), thickness=6,
            label=f"{minutes}:{seconds:02d}",
        )
        if not still:
            text = "Quedate quieto"
            size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(
                ctx.frame, text, ((ctx.width - size[0]) // 2, center[1] + 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, theme.text_dim, 2, cv2.LINE_AA,
            )

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        return [
            f"Objetivo: {self.target / 60:.0f} min  Completadas: {self._completed}",
            f"Quietud maxima: {self.stillness.longest_stillness:.0f}s",
        ]


def wellness_modes() -> List[Mode]:
    """Every wellness mode, in menu order."""
    return [
        DrowsinessMode(), FocusMode(), BreathingMode(),
        ErgonomicsMode(), MeditationMode(),
    ]
