"""
Creative and visual modes for LookThePerson.

These transform how the frame looks rather than what is measured: image
filters, silhouettes, holograms, motion trails, heat maps and particles.

Most are built on :class:`FilterMode`, which wires a named filter from
``fx.filters`` into the mode lifecycle; the rest need the segmentation mask or
their own per-frame state.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from analytics.angles import PoseLandmark as L
from analytics.motion import TrajectoryRecorder
from core.geometry import clamp
from core.state import AppState, FrameContext
from fx import background as bg
from fx.filters import apply_filter, filter_names
from fx.overlays import (
    MotionHeatmap,
    ParticleSystem,
    draw_scan_line,
    draw_trail,
)
from modes.base import Mode, ModeCategory

__all__ = [
    "FilterMode",
    "NightVisionMode",
    "ThermalMode",
    "MatrixMode",
    "AsciiMode",
    "CartoonMode",
    "SketchMode",
    "GlitchMode",
    "SilhouetteMode",
    "HologramMode",
    "SpotlightMode",
    "GhostMode",
    "TrailsMode",
    "HeatmapMode",
    "ParticleMode",
    "XRayMode",
    "creative_modes",
]


# ---------------------------------------------------------------------------
# Filter-backed modes
# ---------------------------------------------------------------------------

class FilterMode(Mode):
    """
    Base for modes whose effect is a whole-frame filter.

    Subclasses set :attr:`filter_name`; ``f`` cycles through every registered
    filter so any of them is reachable without its own mode.
    """

    category = ModeCategory.CREATIVE
    requires = ("pose",)
    filter_name = "none"
    filter_kwargs: Dict[str, Any] = {}
    toggles = {
        "skeleton": False, "segmentation": False, "face_mesh": False,
        "face_detect": False, "object_detect": False,
    }
    keys = {"f": "Siguiente filtro", "k": "Mostrar/ocultar esqueleto"}

    def __init__(self) -> None:
        super().__init__()
        self._active_filter = self.filter_name
        self._names = filter_names()

    def on_enter(self, state: AppState) -> None:
        super().on_enter(state)
        self._active_filter = self.filter_name

    def on_key(self, key: int, state: AppState) -> bool:
        if key == ord("f"):
            index = self._names.index(self._active_filter) if self._active_filter in self._names else -1
            self._active_filter = self._names[(index + 1) % len(self._names)]
            state.notify(f"Filtro: {self._active_filter}")
            return True
        if key == ord("k"):
            state.toggle("skeleton")
            return True
        return False

    def process(self, ctx: FrameContext, state: AppState) -> None:
        # Filters replace the working image, so they run before overlays draw.
        ctx.frame = apply_filter(ctx.frame, self._active_filter, **self.filter_kwargs)

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        return [f"Filtro: {self._active_filter}"]


class NightVisionMode(FilterMode):
    """Green image-intensifier look with a sweeping scan line."""

    key = "night_vision"
    label = "Vision nocturna"
    description = "Intensificador verde con linea de barrido"
    filter_name = "night_vision"

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        position = (ctx.now * 0.25) % 1.0
        draw_scan_line(ctx.frame, position, color=(80, 255, 120), thickness=1)


class ThermalMode(FilterMode):
    """False-colour heat map of the scene."""

    key = "thermal"
    label = "Termico"
    description = "Falso color termico por luminancia"
    filter_name = "thermal"

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        return ["Termico (simulado por luminancia)"]


class MatrixMode(Mode):
    """
    Falling green code with the person cut out of it.

    The rain is generated once as a column model and advanced per frame, so it
    costs almost nothing regardless of resolution.
    """

    key = "matrix"
    label = "Matrix"
    description = "Lluvia de codigo verde sobre la silueta"
    category = ModeCategory.CREATIVE
    requires = ("pose",)
    toggles = {
        "skeleton": False, "segmentation": True, "face_mesh": False,
        "face_detect": False, "object_detect": False,
    }

    CHARS = "01アイウエオカキクケコサシスセソABCDEF<>#$%"

    def __init__(self) -> None:
        super().__init__()
        self._columns: List[Tuple[float, float, int]] = []   # (y, speed, length)
        self._width = 0
        self._cell = 18
        self._rng = np.random.default_rng(1234)

    def on_enter(self, state: AppState) -> None:
        super().on_enter(state)
        self._columns = []
        state.set_theme("matrix")

    def _ensure_columns(self, width: int) -> None:
        count = max(1, width // self._cell)
        if len(self._columns) == count and self._width == width:
            return
        self._width = width
        self._columns = [
            (
                float(self._rng.uniform(-40, 0)),
                float(self._rng.uniform(6.0, 22.0)),
                int(self._rng.integers(6, 22)),
            )
            for _ in range(count)
        ]

    def process(self, ctx: FrameContext, state: AppState) -> None:
        self._ensure_columns(ctx.width)
        rows = max(1, ctx.height // self._cell)
        advanced = []
        for y, speed, length in self._columns:
            y += speed * ctx.delta
            if y - length > rows:
                y = float(self._rng.uniform(-20, 0))
                speed = float(self._rng.uniform(6.0, 22.0))
                length = int(self._rng.integers(6, 22))
            advanced.append((y, speed, length))
        self._columns = advanced

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        rain = np.zeros_like(ctx.frame)
        for column, (head_y, _speed, length) in enumerate(self._columns):
            x = column * self._cell
            for offset in range(length):
                row = int(head_y) - offset
                if row < 0 or row * self._cell > ctx.height:
                    continue
                fade = 1.0 - offset / length
                color = (0, int(90 + 165 * fade), int(40 * fade))
                if offset == 0:
                    color = (200, 255, 200)
                char = self.CHARS[(column * 7 + row) % len(self.CHARS)]
                cv2.putText(
                    rain, char, (x, row * self._cell),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
                )

        mask = self._segmentation_mask(ctx)
        if mask is not None:
            prepared = bg.prepare_mask(mask, ctx.frame.shape, feather=9)
            # Person shows through the rain as a dim ghost.
            person = (ctx.frame * 0.55).astype(np.uint8)
            alpha = prepared[:, :, None]
            ctx.frame = (person * alpha + rain * (1.0 - alpha)).astype(np.uint8)
        else:
            ctx.frame = cv2.addWeighted(ctx.frame, 0.25, rain, 1.0, 0)

    @staticmethod
    def _segmentation_mask(ctx: FrameContext):
        result = ctx.pose_result
        masks = getattr(result, "segmentation_masks", None) if result else None
        return masks[0] if masks else None


class AsciiMode(FilterMode):
    """The whole scene rendered as coloured ASCII characters."""

    key = "ascii"
    label = "ASCII"
    description = "Renderiza la escena con caracteres"
    filter_name = "ascii"
    keys = {"f": "Siguiente filtro", "+": "Celda mas grande", "-": "Celda mas pequena"}

    def __init__(self) -> None:
        super().__init__()
        self._cell = 8

    def on_key(self, key: int, state: AppState) -> bool:
        if key == ord("+"):
            self._cell = min(24, self._cell + 2)
            state.notify(f"Celda ASCII: {self._cell}")
            return True
        if key == ord("-"):
            self._cell = max(4, self._cell - 2)
            state.notify(f"Celda ASCII: {self._cell}")
            return True
        return super().on_key(key, state)

    def process(self, ctx: FrameContext, state: AppState) -> None:
        if self._active_filter == "ascii":
            ctx.frame = apply_filter(ctx.frame, "ascii", cell=self._cell)
        else:
            super().process(ctx, state)


class CartoonMode(FilterMode):
    """Flat colours and ink outlines."""

    key = "cartoon"
    label = "Cartoon"
    description = "Colores planos con contornos"
    filter_name = "cartoon"


class SketchMode(FilterMode):
    """Pencil drawing."""

    key = "sketch"
    label = "Boceto"
    description = "Dibujo a lapiz"
    filter_name = "sketch"


class GlitchMode(FilterMode):
    """
    Digital corruption that intensifies with how much you move.

    Standing still leaves the image nearly clean; sudden movement tears it
    apart, which makes the effect feel reactive rather than random.
    """

    key = "glitch"
    label = "Glitch"
    description = "Corrupcion digital reactiva al movimiento"
    filter_name = "glitch"

    def process(self, ctx: FrameContext, state: AppState) -> None:
        energy = ctx.motion.get("energy", 0.0)
        intensity = clamp(energy * 6.0, 0.05, 1.0)
        ctx.frame = apply_filter(ctx.frame, "glitch", intensity=intensity)
        self._intensity = intensity

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        return [f"Glitch: {getattr(self, '_intensity', 0.0) * 100:.0f}%"]


# ---------------------------------------------------------------------------
# Segmentation-backed modes
# ---------------------------------------------------------------------------

class _MaskMode(Mode):
    """Shared plumbing for modes that need the person segmentation mask."""

    category = ModeCategory.CREATIVE
    requires = ("pose",)
    toggles = {
        "skeleton": False, "segmentation": True, "face_mesh": False,
        "face_detect": False, "object_detect": False,
    }

    @staticmethod
    def mask_for(ctx: FrameContext, index: int = 0):
        """First available segmentation mask, or None."""
        result = ctx.pose_result
        masks = getattr(result, "segmentation_masks", None) if result else None
        if masks and index < len(masks):
            return masks[index]
        return None


class SilhouetteMode(_MaskMode):
    """
    The person as a flat shape.

    Cycle the palette with ``c``; several presets read very differently on
    camera, from stark black-and-white to neon on deep blue.
    """

    key = "silhouette"
    label = "Silueta"
    description = "Figura plana sobre fondo plano"
    keys = {"c": "Cambiar paleta"}

    PALETTES = (
        ((255, 255, 255), (0, 0, 0)),
        ((0, 0, 0), (255, 255, 255)),
        ((0, 255, 255), (40, 10, 40)),
        ((255, 120, 0), (10, 10, 40)),
        ((120, 255, 120), (5, 20, 5)),
    )

    def __init__(self) -> None:
        super().__init__()
        self._palette = 0

    def on_key(self, key: int, state: AppState) -> bool:
        if key == ord("c"):
            self._palette = (self._palette + 1) % len(self.PALETTES)
            state.notify(f"Paleta {self._palette + 1}")
            return True
        return False

    def process(self, ctx: FrameContext, state: AppState) -> None:
        mask = self.mask_for(ctx)
        if mask is None:
            return
        person, backdrop = self.PALETTES[self._palette]
        ctx.frame = bg.silhouette(ctx.frame, mask, person, backdrop)

    def status_text(self, ctx: FrameContext, state: AppState) -> Optional[str]:
        return None if ctx.has_pose else "SIN SILUETA — ACERCATE"


class HologramMode(_MaskMode):
    """Sci-fi projection: tinted, scanlined subject on a dark field."""

    key = "hologram"
    label = "Holograma"
    description = "Proyeccion holografica con lineas de barrido"

    def process(self, ctx: FrameContext, state: AppState) -> None:
        mask = self.mask_for(ctx)
        if mask is not None:
            ctx.frame = bg.hologram(ctx.frame, mask, color=state.theme.accent)

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        draw_scan_line(ctx.frame, (ctx.now * 0.4) % 1.0, color=state.theme.accent)


class SpotlightMode(_MaskMode):
    """Follow-spot lighting: the subject is lit, everything else falls away."""

    key = "spotlight"
    label = "Foco"
    description = "Ilumina solo a la persona"
    keys = {"+": "Mas oscuro", "-": "Menos oscuro"}

    def __init__(self) -> None:
        super().__init__()
        self._dim = 0.72

    def on_key(self, key: int, state: AppState) -> bool:
        if key == ord("+"):
            self._dim = clamp(self._dim + 0.08, 0.0, 0.95)
            return True
        if key == ord("-"):
            self._dim = clamp(self._dim - 0.08, 0.0, 0.95)
            return True
        return False

    def process(self, ctx: FrameContext, state: AppState) -> None:
        mask = self.mask_for(ctx)
        if mask is not None:
            ctx.frame = bg.spotlight(ctx.frame, mask, dim=self._dim)


class GhostMode(_MaskMode):
    """Motion echo: previous silhouettes fade out behind you."""

    key = "ghost"
    label = "Fantasma"
    description = "Estelas de siluetas anteriores"
    keys = {"+": "Estela mas larga", "-": "Estela mas corta"}

    def __init__(self) -> None:
        super().__init__()
        self._accumulator = None
        self._decay = 0.88

    def on_enter(self, state: AppState) -> None:
        super().on_enter(state)
        self._accumulator = None

    def on_key(self, key: int, state: AppState) -> bool:
        if key == ord("+"):
            self._decay = clamp(self._decay + 0.02, 0.5, 0.98)
            state.notify(f"Estela: {self._decay:.2f}")
            return True
        if key == ord("-"):
            self._decay = clamp(self._decay - 0.02, 0.5, 0.98)
            state.notify(f"Estela: {self._decay:.2f}")
            return True
        return False

    def process(self, ctx: FrameContext, state: AppState) -> None:
        mask = self.mask_for(ctx)
        if mask is None:
            return
        ctx.frame, self._accumulator = bg.ghost_trail(
            ctx.frame, mask, self._accumulator, decay=self._decay,
        )


class XRayMode(_MaskMode):
    """
    Skeleton over a darkened, high-contrast body — a medical-imaging look.
    """

    key = "xray"
    label = "Rayos X"
    description = "Esqueleto sobre cuerpo en alto contraste"
    toggles = {
        "skeleton": True, "segmentation": True, "face_mesh": False,
        "face_detect": False, "object_detect": False,
    }

    def on_enter(self, state: AppState) -> None:
        super().on_enter(state)
        state.set_theme("medical")

    def process(self, ctx: FrameContext, state: AppState) -> None:
        mask = self.mask_for(ctx)
        gray = cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        radiograph = cv2.cvtColor(255 - equalized, cv2.COLOR_GRAY2BGR)

        if mask is None:
            ctx.frame = (radiograph * 0.4).astype(np.uint8)
            return

        prepared = bg.prepare_mask(mask, ctx.frame.shape, feather=11)
        dark = np.zeros_like(ctx.frame)
        alpha = prepared[:, :, None]
        ctx.frame = (radiograph * alpha + dark * (1.0 - alpha)).astype(np.uint8)


# ---------------------------------------------------------------------------
# Overlay-driven modes
# ---------------------------------------------------------------------------

class TrailsMode(Mode):
    """
    Light trails following hands, feet and head.

    Useful for movement analysis as well as visuals — the trail length tells
    you how far a limb has travelled.
    """

    key = "trails"
    label = "Estelas"
    description = "Rastros luminosos de manos, pies y cabeza"
    category = ModeCategory.CREATIVE
    requires = ("pose",)
    toggles = {
        "skeleton": False, "segmentation": False, "face_mesh": False,
        "face_detect": False, "object_detect": False,
    }
    keys = {"c": "Limpiar estelas", "k": "Esqueleto on/off"}

    TRACKED = (L.LEFT_WRIST, L.RIGHT_WRIST, L.LEFT_ANKLE, L.RIGHT_ANKLE, L.NOSE)

    def __init__(self) -> None:
        super().__init__()
        self.recorder = TrajectoryRecorder(self.TRACKED, max_points=48)

    def on_key(self, key: int, state: AppState) -> bool:
        if key == ord("c"):
            self.recorder.clear()
            return True
        if key == ord("k"):
            state.toggle("skeleton")
            return True
        return False

    def process(self, ctx: FrameContext, state: AppState) -> None:
        if ctx.has_pose:
            self.recorder.update(ctx.primary_pose, ctx.now)
        self.recorder.prune(ctx.now, max_age=2.5)

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        theme = state.theme
        for index, landmark_index in enumerate(self.TRACKED):
            path = self.recorder.path(landmark_index)
            draw_trail(
                ctx.frame, path, theme.category_color(index),
                ctx.width, ctx.height, max_thickness=7,
            )

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        left = self.recorder.path_length(L.LEFT_WRIST)
        right = self.recorder.path_length(L.RIGHT_WRIST)
        return [f"Recorrido manos: izq {left:.2f}  der {right:.2f}"]


class HeatmapMode(Mode):
    """
    Accumulates where movement happens and paints it as a heat map.

    Over a few minutes this shows the area a person actually occupies — handy
    for setting up a camera or reviewing how much you moved.
    """

    key = "heatmap"
    label = "Mapa de calor"
    description = "Acumula donde ocurre el movimiento"
    category = ModeCategory.CREATIVE
    requires = ("pose",)
    toggles = {
        "skeleton": True, "segmentation": False, "face_mesh": False,
        "face_detect": False, "object_detect": False,
    }
    keys = {"c": "Limpiar mapa"}

    def __init__(self) -> None:
        super().__init__()
        self.heatmap = MotionHeatmap(decay=0.995)

    def on_key(self, key: int, state: AppState) -> bool:
        if key == ord("c"):
            self.heatmap.clear()
            state.notify("Mapa de calor limpiado")
            return True
        return False

    def process(self, ctx: FrameContext, state: AppState) -> None:
        self.heatmap.decay_step()
        if ctx.has_pose:
            for landmarks in ctx.pose_landmarks:
                self.heatmap.add_landmarks(landmarks, weight=0.6)

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        ctx.frame = self.heatmap.render(ctx.frame, alpha=0.5)

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        hotspot = self.heatmap.hotspot()
        lines = [f"Cobertura: {self.heatmap.coverage() * 100:.1f}%"]
        if hotspot:
            lines.append(f"Punto caliente: {hotspot[0]:.2f}, {hotspot[1]:.2f}")
        return lines


class ParticleMode(Mode):
    """
    Particles that stream from your hands and react to how fast they move.

    Fast movement emits more, so it rewards big gestures — good for demos and
    for kids.
    """

    key = "particles"
    label = "Particulas"
    description = "Particulas emitidas desde las manos"
    category = ModeCategory.CREATIVE
    requires = ("pose",)
    toggles = {
        "skeleton": False, "segmentation": False, "face_mesh": False,
        "face_detect": False, "object_detect": False,
    }
    keys = {"c": "Limpiar particulas"}

    def __init__(self) -> None:
        super().__init__()
        self.particles = ParticleSystem(max_particles=400, gravity=0.18)

    def on_key(self, key: int, state: AppState) -> bool:
        if key == ord("c"):
            self.particles.clear()
            return True
        return False

    def process(self, ctx: FrameContext, state: AppState) -> None:
        self.particles.update(ctx.delta)
        if not ctx.has_pose:
            return

        landmarks = ctx.primary_pose
        theme = state.theme
        for index, (landmark_index, speed_key) in enumerate((
            (L.LEFT_WRIST, "left_hand_speed"),
            (L.RIGHT_WRIST, "right_hand_speed"),
        )):
            landmark = landmarks[landmark_index]
            if getattr(landmark, "visibility", 1.0) < 0.4:
                continue
            speed = ctx.motion.get(speed_key, 0.0)
            count = int(clamp(speed * 30.0, 0.0, 12.0))
            if count:
                self.particles.emit(
                    landmark.x, landmark.y, count=count,
                    color=theme.category_color(index),
                    speed=0.2 + speed * 0.4, life=0.9,
                )

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        self.particles.draw(ctx.frame)

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        return [f"Particulas: {self.particles.count}"]


def creative_modes() -> List[Mode]:
    """Every creative mode, in menu order."""
    return [
        NightVisionMode(), ThermalMode(), MatrixMode(), AsciiMode(),
        CartoonMode(), SketchMode(), GlitchMode(), SilhouetteMode(),
        HologramMode(), SpotlightMode(), GhostMode(), XRayMode(),
        TrailsMode(), HeatmapMode(), ParticleMode(),
    ]
