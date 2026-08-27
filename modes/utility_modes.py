"""
Utility modes for LookThePerson.

Practical, non-decorative behaviours: privacy masking, presence detection,
camera calibration, a photo booth, time-lapse capture, body measurement, a
virtual green screen, and the debug and benchmark tools used to tune the app
itself.

The surveillance mode lives in :mod:`modes.security` — it grew its own sensor
pipeline, identity tracker and operator display, which is more than belongs in
a shared module. It is re-exported here so it keeps its place in this category.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from analytics.angles import PoseLandmark as L
from analytics.angles import limb_lengths
from core.events import Events
from core.filters import Cooldown, Debouncer
from core.geometry import distance
from core.state import AppState, FrameContext
from fx import background as bg
from fx.overlays import draw_landmark_ids
from modes.base import Mode, ModeCategory
from modes.security import SecurityMode

__all__ = [
    "PrivacyMode",
    "SecurityMode",
    "PresenceMode",
    "GreenScreenMode",
    "PhotoBoothMode",
    "TimelapseMode",
    "MeasureMode",
    "CalibrationMode",
    "DebugMode",
    "BenchmarkMode",
    "utility_modes",
]


class PrivacyMode(Mode):
    """
    Hides identity while keeping the tracking useful.

    Faces are blurred or pixelated by default; ``p`` cycles to full-body
    masking, which replaces the person with a solid silhouette.
    """

    key = "privacy"
    label = "Privacidad"
    description = "Difumina caras o oculta el cuerpo entero"
    category = ModeCategory.UTILITY
    requires = ("pose", "face_detect")
    toggles = {
        "skeleton": False, "segmentation": True, "face_mesh": False,
        "face_detect": True, "object_detect": False, "bounding_boxes": False,
    }
    keys = {"p": "Cambiar nivel de privacidad"}

    LEVELS = ("cara_blur", "cara_pixel", "cuerpo", "silueta")

    def __init__(self) -> None:
        super().__init__()
        self._level = 0

    def on_key(self, key: int, state: AppState) -> bool:
        if key == ord("p"):
            self._level = (self._level + 1) % len(self.LEVELS)
            state.notify(f"Privacidad: {self.LEVELS[self._level]}")
            return True
        return False

    def process(self, ctx: FrameContext, state: AppState) -> None:
        level = self.LEVELS[self._level]

        if level in ("cara_blur", "cara_pixel"):
            for box in self._face_boxes(ctx):
                if level == "cara_blur":
                    bg.privacy_blur_region(ctx.frame, box, strength=45)
                else:
                    bg.pixelate_region(ctx.frame, box, block=16)
            return

        mask = self._mask(ctx)
        if mask is None:
            return
        if level == "cuerpo":
            prepared = bg.prepare_mask(mask, ctx.frame.shape, feather=15)
            blurred = cv2.GaussianBlur(ctx.frame, (61, 61), 0)
            alpha = prepared[:, :, None]
            ctx.frame = (blurred * alpha + ctx.frame * (1.0 - alpha)).astype(np.uint8)
        else:
            ctx.frame = bg.silhouette(ctx.frame, mask, (40, 40, 40), (200, 200, 200))

    @staticmethod
    def _mask(ctx: FrameContext):
        result = ctx.pose_result
        masks = getattr(result, "segmentation_masks", None) if result else None
        return masks[0] if masks else None

    @staticmethod
    def _face_boxes(ctx: FrameContext) -> List[Tuple[int, int, int, int]]:
        """Face rectangles from the detector, or estimated from the pose head."""
        boxes: List[Tuple[int, int, int, int]] = []
        result = ctx.face_detect_result
        if result and getattr(result, "detections", None):
            for detection in result.detections:
                box = detection.bounding_box
                boxes.append((
                    int(box.origin_x), int(box.origin_y),
                    int(box.width), int(box.height),
                ))
            return boxes

        # Fall back to the pose skeleton so privacy still works with the face
        # detector switched off.
        for landmarks in ctx.pose_landmarks:
            nose = landmarks[L.NOSE]
            ears = distance(landmarks[L.LEFT_EAR], landmarks[L.RIGHT_EAR])
            size = max(int(ears * ctx.width * 2.2), 60)
            cx = int(nose.x * ctx.width)
            cy = int(nose.y * ctx.height)
            boxes.append((cx - size // 2, cy - size // 2, size, size))
        return boxes

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        return [f"Nivel: {self.LEVELS[self._level]}"]


class PresenceMode(Mode):
    """
    Tracks whether someone is at the desk and for how long.

    Produces a simple attendance log: arrival, departures and total time
    present, which is the raw material for a work-session report.
    """

    key = "presence"
    label = "Presencia"
    description = "Registra presencia, ausencias y tiempo total"
    category = ModeCategory.UTILITY
    requires = ("pose", "face_detect")
    toggles = {
        "skeleton": False, "segmentation": False, "face_mesh": False,
        "face_detect": True, "object_detect": False, "bounding_boxes": True,
    }

    def __init__(self) -> None:
        super().__init__()
        self._present = Debouncer(rise_seconds=1.0, fall_seconds=3.0)
        self._present_seconds = 0.0
        self._absent_seconds = 0.0
        self._absences = 0
        self._since: Optional[float] = None

    def process(self, ctx: FrameContext, state: AppState) -> None:
        detected = ctx.has_pose or bool(ctx.face_landmarks)
        was_present = self._present.state
        present = self._present.update(detected, ctx.now)

        if present:
            self._present_seconds += ctx.delta
        else:
            self._absent_seconds += ctx.delta

        if was_present != present:
            self._since = ctx.now
            if not present:
                self._absences += 1
            state.bus.emit(Events.PRESENCE_CHANGED, present=present)
            state.notify("Presente" if present else "Ausente")

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        total = self._present_seconds + self._absent_seconds
        ratio = self._present_seconds / total * 100 if total > 0 else 0.0
        current = ctx.now - self._since if self._since else 0.0
        return [
            f"Estado: {'presente' if self._present.state else 'ausente'} ({current:.0f}s)",
            f"Presente: {self._present_seconds / 60:.1f} min ({ratio:.0f}%)",
            f"Ausencias: {self._absences}",
        ]


class GreenScreenMode(Mode):
    """
    Virtual background without a physical green screen.

    Cycles between a flat chroma plate, a blurred room, a solid colour and any
    image dropped into ``backgrounds/``.
    """

    key = "greenscreen"
    label = "Fondo virtual"
    description = "Sustituye o difumina el fondo"
    category = ModeCategory.UTILITY
    requires = ("pose",)
    toggles = {
        "skeleton": False, "segmentation": True, "face_mesh": False,
        "face_detect": False, "object_detect": False,
    }
    keys = {"b": "Cambiar fondo"}

    BACKGROUNDS = ("chroma", "blur", "negro", "gradiente", "imagen")

    def __init__(self) -> None:
        super().__init__()
        self._index = 0
        self._image: Optional[np.ndarray] = None
        self._image_loaded = False

    def on_key(self, key: int, state: AppState) -> bool:
        if key == ord("b"):
            self._index = (self._index + 1) % len(self.BACKGROUNDS)
            state.notify(f"Fondo: {self.BACKGROUNDS[self._index]}")
            return True
        return False

    def _load_image(self, state: AppState) -> Optional[np.ndarray]:
        """First image found in ``backgrounds/``, loaded once and cached."""
        if self._image_loaded:
            return self._image
        self._image_loaded = True

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        folder = os.path.join(root, "backgrounds")
        if not os.path.isdir(folder):
            return None
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image = cv2.imread(os.path.join(folder, name))
                if image is not None:
                    self._image = image
                    state.notify(f"Fondo cargado: {name}")
                    return image
        return None

    @staticmethod
    def _gradient(shape: Tuple[int, int]) -> np.ndarray:
        """Vertical two-tone gradient plate."""
        height, width = shape
        ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
        top = np.array([90.0, 30.0, 60.0], dtype=np.float32)
        bottom = np.array([220.0, 130.0, 40.0], dtype=np.float32)
        plate = top + (bottom - top) * ramp
        return np.repeat(plate, width, axis=1).astype(np.uint8)

    def process(self, ctx: FrameContext, state: AppState) -> None:
        mask = self._mask(ctx)
        if mask is None:
            return

        kind = self.BACKGROUNDS[self._index]
        if kind == "chroma":
            ctx.frame = bg.background_color(ctx.frame, mask, (0, 177, 64))
        elif kind == "blur":
            ctx.frame = bg.blur_background(ctx.frame, mask, strength=45)
        elif kind == "negro":
            ctx.frame = bg.cutout(ctx.frame, mask)
        elif kind == "gradiente":
            plate = self._gradient((ctx.height, ctx.width))
            ctx.frame = bg.replace_background(ctx.frame, mask, plate)
        else:
            image = self._load_image(state)
            if image is not None:
                ctx.frame = bg.replace_background(ctx.frame, mask, image)
            else:
                ctx.frame = bg.blur_background(ctx.frame, mask, strength=45)

    @staticmethod
    def _mask(ctx: FrameContext):
        result = ctx.pose_result
        masks = getattr(result, "segmentation_masks", None) if result else None
        return masks[0] if masks else None

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        return [f"Fondo: {self.BACKGROUNDS[self._index]}"]


class PhotoBoothMode(Mode):
    """
    Gesture-triggered photo booth.

    Raise both hands to start a countdown, then it takes a burst of photos —
    no keyboard needed, which is the whole point when you are posing.
    """

    key = "photobooth"
    label = "Fotomaton"
    description = "Cuenta atras por gesto y rafaga de fotos"
    category = ModeCategory.UTILITY
    requires = ("pose",)
    toggles = {
        "skeleton": False, "segmentation": False, "face_mesh": False,
        "face_detect": False, "object_detect": False, "telemetry": False,
    }

    COUNTDOWN = 3.0
    BURST = 3
    BURST_INTERVAL = 0.8

    def __init__(self) -> None:
        super().__init__()
        self._countdown_from: Optional[float] = None
        self._burst_remaining = 0
        self._next_shot = 0.0
        self._taken = 0
        self._flash_until = 0.0
        self._trigger_cooldown = Cooldown(6.0)

    def process(self, ctx: FrameContext, state: AppState) -> None:
        # Start the countdown on "both hands raised".
        if (
            self._countdown_from is None
            and self._burst_remaining == 0
            and ctx.gesture("both_hands_raised")
            and self._trigger_cooldown.trigger(ctx.now)
        ):
            self._countdown_from = ctx.now
            state.notify("Sonrie!")

        if self._countdown_from is not None:
            if ctx.now - self._countdown_from >= self.COUNTDOWN:
                self._countdown_from = None
                self._burst_remaining = self.BURST
                self._next_shot = ctx.now
            return

        if self._burst_remaining > 0 and ctx.now >= self._next_shot:
            self._burst_remaining -= 1
            self._next_shot = ctx.now + self.BURST_INTERVAL
            self._taken += 1
            self._flash_until = ctx.now + 0.18
            # The pipeline owns the recorder; ask it for a shot.
            state.bus.emit(Events.ACTION_TRIGGERED, name="screenshot")

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        theme = state.theme

        if self._countdown_from is not None:
            remaining = self.COUNTDOWN - (ctx.now - self._countdown_from)
            text = str(max(1, int(remaining) + 1))
            size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 6.0, 12)
            cv2.putText(
                ctx.frame, text,
                ((ctx.width - size[0]) // 2, (ctx.height + size[1]) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 6.0, theme.accent, 12, cv2.LINE_AA,
            )

        if ctx.now < self._flash_until:
            ctx.frame[:] = cv2.addWeighted(
                ctx.frame, 0.3, np.full_like(ctx.frame, 255), 0.7, 0,
            )

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        return [
            f"Fotos: {self._taken}",
            "Levanta ambas manos para disparar",
        ]


class TimelapseMode(Mode):
    """
    Captures a frame at a fixed interval to build a time lapse.

    Useful for recording a whole work session or a long stretch as a short
    clip; the interval is adjustable live.
    """

    key = "timelapse"
    label = "Timelapse"
    description = "Captura periodica para acelerado"
    category = ModeCategory.UTILITY
    requires = ("pose",)
    toggles = {
        "skeleton": False, "segmentation": False, "face_mesh": False,
        "face_detect": False, "object_detect": False,
    }
    keys = {"+": "Intervalo mayor", "-": "Intervalo menor", "space": "Pausar"}

    INTERVALS = (1.0, 2.0, 5.0, 10.0, 30.0, 60.0)

    def __init__(self) -> None:
        super().__init__()
        self._interval_index = 2
        self._last_capture = 0.0
        self._captured = 0
        self._paused = False

    @property
    def interval(self) -> float:
        return self.INTERVALS[self._interval_index]

    def on_key(self, key: int, state: AppState) -> bool:
        if key == ord("+"):
            self._interval_index = min(len(self.INTERVALS) - 1, self._interval_index + 1)
            state.notify(f"Intervalo: {self.interval:.0f}s")
            return True
        if key == ord("-"):
            self._interval_index = max(0, self._interval_index - 1)
            state.notify(f"Intervalo: {self.interval:.0f}s")
            return True
        if key == 32:  # space
            self._paused = not self._paused
            state.notify("Timelapse en pausa" if self._paused else "Timelapse activo")
            return True
        return False

    def process(self, ctx: FrameContext, state: AppState) -> None:
        if self._paused:
            return
        if ctx.now - self._last_capture >= self.interval:
            self._last_capture = ctx.now
            self._captured += 1
            state.bus.emit(Events.ACTION_TRIGGERED, name="screenshot")

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        theme = state.theme
        remaining = max(0.0, self.interval - (ctx.now - self._last_capture))
        color = theme.text_dim if self._paused else theme.accent
        cv2.putText(
            ctx.frame, f"{remaining:.1f}s", (ctx.width - 110, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
        )

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        state_text = "PAUSA" if self._paused else "activo"
        duration = self._captured / 24.0
        return [
            f"Capturas: {self._captured} ({state_text})",
            f"Intervalo: {self.interval:.0f}s  Video: {duration:.1f}s a 24fps",
        ]


class MeasureMode(Mode):
    """
    Estimates body proportions from the skeleton.

    Calibrate with your real height (``+``/``-``) and it converts every limb
    length into centimetres. Accuracy depends on standing square to the
    camera; treat the numbers as approximate.
    """

    key = "measure"
    label = "Medidas"
    description = "Estima proporciones corporales en cm"
    category = ModeCategory.UTILITY
    requires = ("pose",)
    toggles = {
        "skeleton": True, "segmentation": False, "face_mesh": False,
        "face_detect": False, "object_detect": False,
    }
    keys = {"+": "Altura +1cm", "-": "Altura -1cm"}

    def __init__(self) -> None:
        super().__init__()
        self._height_cm = 170.0
        self._measurements: Dict[str, float] = {}

    def on_enter(self, state: AppState) -> None:
        super().on_enter(state)
        self._height_cm = state.config.analytics.user_height_cm

    def on_key(self, key: int, state: AppState) -> bool:
        if key == ord("+"):
            self._height_cm += 1.0
            return True
        if key == ord("-"):
            self._height_cm = max(80.0, self._height_cm - 1.0)
            return True
        return False

    def process(self, ctx: FrameContext, state: AppState) -> None:
        if not ctx.has_pose:
            return

        landmarks = ctx.primary_pose
        # Every index below is a fixed BlazePose position, so a short list
        # would raise IndexError rather than simply measure less.
        if len(landmarks) <= L.RIGHT_ANKLE:
            return

        lengths = limb_lengths(landmarks)
        if not lengths:
            return

        # On-screen height: nose to the lower of the two ankles.
        nose = landmarks[L.NOSE]
        if not ctx.usable(nose):
            return
        ankles = [
            landmarks[i] for i in (L.LEFT_ANKLE, L.RIGHT_ANKLE)
            if getattr(landmarks[i], "visibility", 1.0) >= 0.4
            and ctx.usable(landmarks[i])
        ]
        if not ankles:
            return
        span = max(a.y for a in ankles) - nose.y
        if not (span >= 0.15):   # NaN-safe: also rejects an unusable span
            return

        # The nose sits a little below the crown, so the visible span is ~93%
        # of true height.
        scale = (self._height_cm * 0.93) / span
        self._measurements = {name: value * scale for name, value in lengths.items()}

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        lines = [f"Altura calibrada: {self._height_cm:.0f} cm"]
        if not self._measurements:
            return lines + ["Ponte de cuerpo entero"]
        for name in ("shoulder_width", "left_upper_arm", "left_forearm",
                     "left_thigh", "left_shin", "torso"):
            value = self._measurements.get(name)
            if value:
                lines.append(f"  {name}: {value:.0f} cm")
        return lines


class CalibrationMode(Mode):
    """
    Camera and tracking diagnostics.

    Shows framing guides, per-landmark visibility and whether the whole body
    fits in frame — what you need when setting the camera up for the first
    time.
    """

    key = "calibration"
    label = "Calibracion"
    description = "Guias de encuadre y calidad de seguimiento"
    category = ModeCategory.UTILITY
    requires = ("pose",)
    toggles = {
        "skeleton": True, "segmentation": False, "face_mesh": False,
        "face_detect": False, "object_detect": False, "grid": True,
    }

    def __init__(self) -> None:
        super().__init__()
        self._visibility = 0.0
        self._in_frame = False
        self._brightness = 0.0

    def process(self, ctx: FrameContext, state: AppState) -> None:
        # Sample brightness cheaply from a downscaled copy.
        small = cv2.resize(ctx.frame, (64, 36), interpolation=cv2.INTER_AREA)
        self._brightness = float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())

        if not ctx.has_pose:
            self._visibility = 0.0
            self._in_frame = False
            return

        landmarks = ctx.primary_pose
        visibilities = [getattr(lm, "visibility", 1.0) for lm in landmarks]
        self._visibility = sum(visibilities) / len(visibilities) if visibilities else 0.0

        # Framing needs the nose and both ankles; without them there is nothing
        # to judge, and indexing a truncated list would raise.
        if len(landmarks) <= L.RIGHT_ANKLE:
            self._in_frame = False
            return
        corners = (landmarks[L.NOSE], landmarks[L.LEFT_ANKLE], landmarks[L.RIGHT_ANKLE])
        self._in_frame = all(
            ctx.usable(lm) and 0.02 < lm.x < 0.98 and 0.02 < lm.y < 0.98
            for lm in corners
        )

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        theme = state.theme

        # Safe-framing rectangle.
        margin_x, margin_y = int(ctx.width * 0.08), int(ctx.height * 0.06)
        color = theme.good if self._in_frame else theme.warn
        cv2.rectangle(
            ctx.frame, (margin_x, margin_y),
            (ctx.width - margin_x, ctx.height - margin_y), color, 2,
        )

        # Per-landmark visibility dots.
        if ctx.has_pose:
            for landmark in ctx.primary_pose:
                if not ctx.usable(landmark):
                    continue
                visibility = getattr(landmark, "visibility", 1.0)
                x, y = ctx.px(landmark)
                dot = theme.good if visibility > 0.7 else theme.warn if visibility > 0.4 else theme.danger
                cv2.circle(ctx.frame, (x, y), 3, dot, -1, cv2.LINE_AA)

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        if self._brightness < 45:
            light = "muy oscura"
        elif self._brightness > 200:
            light = "sobreexpuesta"
        else:
            light = "correcta"
        return [
            f"Visibilidad media: {self._visibility * 100:.0f}%",
            f"Cuerpo completo: {'si' if self._in_frame else 'NO — alejate'}",
            f"Iluminacion: {light} ({self._brightness:.0f})",
        ]


class DebugMode(Mode):
    """
    Developer view: landmark indices, per-stage timings and event counters.

    This is the mode to open when a gesture is not firing and you need to see
    what the pipeline actually thinks is happening.
    """

    key = "debug"
    label = "Debug"
    description = "Indices de landmarks, tiempos y eventos"
    category = ModeCategory.UTILITY
    requires = ("pose", "hands")
    toggles = {
        "skeleton": True, "segmentation": False, "face_mesh": False,
        "face_detect": False, "object_detect": False,
        "telemetry": True, "landmark_ids": True,
    }
    keys = {"i": "Mostrar/ocultar indices"}

    def __init__(self) -> None:
        super().__init__()
        self._show_ids = True

    def on_key(self, key: int, state: AppState) -> bool:
        if key == ord("i"):
            self._show_ids = not self._show_ids
            return True
        return False

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        if not self._show_ids or not ctx.has_pose:
            return
        # landmark_points keeps the indices aligned with the skeleton table and
        # reports a non-finite landmark as simply not visible.
        points = ctx.landmark_points(ctx.primary_pose, visibility=0.4)
        draw_landmark_ids(ctx.frame, points, state.theme.text_dim)

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        lines = state.profiler.report_lines(count=5)
        lines.append(f"Frame #{ctx.frame_index}  dt={ctx.delta * 1000:.1f}ms")

        active = [name for name, value in ctx.body_gestures.items() if value]
        lines.append("Gestos: " + (", ".join(active) if active else "-"))

        counts = state.bus.counts()
        top = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:3]
        if top:
            lines.append("Eventos: " + ", ".join(f"{k}={v}" for k, v in top))
        return lines


class BenchmarkMode(Mode):
    """
    Measures how each model affects the frame rate.

    Cycles through model combinations for a few seconds each and reports the
    FPS of every configuration — the fastest way to find out what your machine
    can actually run.
    """

    key = "benchmark"
    label = "Benchmark"
    description = "Mide el coste en FPS de cada modelo"
    category = ModeCategory.UTILITY
    requires = ("pose", "hands", "face_mesh", "face_detect", "object")
    toggles = {"skeleton": True, "telemetry": True}
    keys = {"space": "Iniciar/parar"}

    STAGE_SECONDS = 5.0
    STAGES = (
        ("solo pose", {"segmentation": False, "face_mesh": False,
                       "face_detect": False, "object_detect": False}),
        ("pose+segm", {"segmentation": True, "face_mesh": False,
                       "face_detect": False, "object_detect": False}),
        ("+face mesh", {"segmentation": True, "face_mesh": True,
                        "face_detect": False, "object_detect": False}),
        ("+face det", {"segmentation": True, "face_mesh": True,
                       "face_detect": True, "object_detect": False}),
        ("todo", {"segmentation": True, "face_mesh": True,
                  "face_detect": True, "object_detect": True}),
    )

    def __init__(self) -> None:
        super().__init__()
        self._running = False
        self._stage = 0
        self._stage_started = 0.0
        self._samples: List[float] = []
        self._results: List[Tuple[str, float]] = []

    def on_key(self, key: int, state: AppState) -> bool:
        if key == 32:  # space
            self._running = not self._running
            if self._running:
                self._stage = 0
                self._results.clear()
                self._samples.clear()
                self._stage_started = state.uptime
                state.apply_toggles(self.STAGES[0][1])
                state.notify("Benchmark iniciado")
            else:
                state.notify("Benchmark detenido")
            return True
        return False

    def process(self, ctx: FrameContext, state: AppState) -> None:
        if not self._running:
            return

        # Skip the first second of each stage so model warm-up is excluded.
        elapsed = state.uptime - self._stage_started
        if elapsed > 1.0:
            self._samples.append(state.fps.instant_fps)

        if elapsed >= self.STAGE_SECONDS:
            name = self.STAGES[self._stage][0]
            average = sum(self._samples) / len(self._samples) if self._samples else 0.0
            self._results.append((name, average))
            self._samples.clear()

            self._stage += 1
            if self._stage >= len(self.STAGES):
                self._running = False
                state.notify("Benchmark completado")
                self._print_results()
                return
            self._stage_started = state.uptime
            state.apply_toggles(self.STAGES[self._stage][1])

    def _print_results(self) -> None:
        print("\n[benchmark] Resultados:", flush=True)
        for name, fps in self._results:
            print(f"  {name:<14} {fps:6.1f} FPS", flush=True)

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        if self._running:
            name = self.STAGES[self._stage][0]
            elapsed = state.uptime - self._stage_started
            lines = [f"Probando: {name} ({elapsed:.0f}/{self.STAGE_SECONDS:.0f}s)"]
        else:
            lines = ["Pulsa ESPACIO para iniciar"]
        lines.extend(f"  {name}: {fps:.1f} FPS" for name, fps in self._results)
        return lines


def utility_modes() -> List[Mode]:
    """Every utility mode, in menu order."""
    return [
        PrivacyMode(), SecurityMode(), PresenceMode(), GreenScreenMode(),
        PhotoBoothMode(), TimelapseMode(), MeasureMode(), CalibrationMode(),
        DebugMode(), BenchmarkMode(),
    ]
