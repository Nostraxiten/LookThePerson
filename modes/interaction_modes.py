"""
Interaction modes for LookThePerson.

Modes that turn the body into an input device: drawing in the air, moving the
mouse with a finger, playing a virtual instrument, driving media playback and
navigating slides.

Anything that reaches outside the app goes through the platform bridge and is
gated by the gesture permissions in the config, so these modes cannot take over
the machine unless they were allowed to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from analytics.angles import PoseLandmark as L
from core.events import Events
from core.filters import Cooldown, Debouncer
from core.geometry import clamp, distance, to_pixels
from core.state import AppState, FrameContext
from modes.base import Mode, ModeCategory

__all__ = [
    "AirDrawMode",
    "AirMouseMode",
    "VirtualPianoMode",
    "MediaControlMode",
    "PresentationMode",
    "SignCountMode",
    "interaction_modes",
]


# Hand landmark indices used across this module.
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
INDEX_PIP = 6


def _pinching(hand, threshold: float = 0.055) -> bool:
    """Whether thumb and index tips are touching — the universal 'click'."""
    return distance(hand[THUMB_TIP], hand[INDEX_TIP]) < threshold


def _index_extended(hand) -> bool:
    """Whether the index finger is pointing."""
    return hand[INDEX_TIP].y < hand[INDEX_PIP].y - 0.02


# ---------------------------------------------------------------------------
# Air drawing
# ---------------------------------------------------------------------------

@dataclass
class Stroke:
    """One continuous drawn line."""

    color: Tuple[int, int, int]
    thickness: int
    points: List[Tuple[float, float]] = field(default_factory=list)


class AirDrawMode(Mode):
    """
    Draw in the air with your index finger.

    Pinch thumb and index to lift the pen. Strokes are stored in normalized
    coordinates, so they stay put if the window is resized.
    """

    key = "airdraw"
    label = "Dibujo aereo"
    description = "Dibuja con el dedo indice; pellizca para levantar el lapiz"
    category = ModeCategory.INTERACTION
    requires = ("hands",)
    toggles = {
        "skeleton": False, "segmentation": False, "face_mesh": False,
        "face_detect": False, "object_detect": False,
    }
    keys = {
        "c": "Borrar todo", "u": "Deshacer", "v": "Cambiar color",
        "+": "Trazo mas grueso", "-": "Trazo mas fino",
    }

    COLORS = (
        (0, 220, 255), (80, 255, 80), (255, 120, 60),
        (255, 80, 220), (255, 255, 255), (60, 60, 255),
    )

    def __init__(self) -> None:
        super().__init__()
        self._strokes: List[Stroke] = []
        self._active: Optional[Stroke] = None
        self._color_index = 0
        self._thickness = 5

    def reset(self, state: AppState) -> None:
        self._strokes.clear()
        self._active = None

    def on_key(self, key: int, state: AppState) -> bool:
        if key == ord("c"):
            self._strokes.clear()
            self._active = None
            state.notify("Lienzo borrado")
            return True
        if key == ord("u"):
            if self._strokes:
                self._strokes.pop()
                state.notify("Deshecho")
            return True
        if key == ord("v"):
            self._color_index = (self._color_index + 1) % len(self.COLORS)
            state.notify(f"Color {self._color_index + 1}")
            return True
        if key == ord("+"):
            self._thickness = min(24, self._thickness + 2)
            return True
        if key == ord("-"):
            self._thickness = max(1, self._thickness - 2)
            return True
        return False

    def process(self, ctx: FrameContext, state: AppState) -> None:
        if not ctx.has_hands:
            self._active = None
            return

        hand = ctx.hand_landmarks[0]
        tip = hand[INDEX_TIP]
        drawing = _index_extended(hand) and not _pinching(hand)

        if drawing:
            if self._active is None:
                self._active = Stroke(self.COLORS[self._color_index], self._thickness)
                self._strokes.append(self._active)
            self._active.points.append((tip.x, tip.y))
            # Cap memory on very long sessions.
            if len(self._active.points) > 2000:
                self._active.points.pop(0)
        else:
            self._active = None

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        for stroke in self._strokes:
            if len(stroke.points) < 2:
                continue
            points = np.array(
                [[int(x * ctx.width), int(y * ctx.height)] for x, y in stroke.points],
                dtype=np.int32,
            )
            cv2.polylines(
                ctx.frame, [points], False, stroke.color,
                stroke.thickness, cv2.LINE_AA,
            )

        # Pen indicator at the fingertip.
        if ctx.has_hands:
            hand = ctx.hand_landmarks[0]
            x, y = to_pixels(hand[INDEX_TIP], ctx.width, ctx.height)
            color = self.COLORS[self._color_index]
            active = self._active is not None
            cv2.circle(ctx.frame, (x, y), self._thickness + 4, color, 2 if not active else -1, cv2.LINE_AA)

        # Colour palette strip.
        for index, color in enumerate(self.COLORS):
            x0 = 20 + index * 34
            y0 = ctx.height - 46
            cv2.rectangle(ctx.frame, (x0, y0), (x0 + 28, y0 + 28), color, -1)
            if index == self._color_index:
                cv2.rectangle(ctx.frame, (x0 - 2, y0 - 2), (x0 + 30, y0 + 30),
                              state.theme.text, 2)

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        points = sum(len(s.points) for s in self._strokes)
        return [f"Trazos: {len(self._strokes)}  Puntos: {points}  Grosor: {self._thickness}"]


# ---------------------------------------------------------------------------
# Air mouse
# ---------------------------------------------------------------------------

class AirMouseMode(Mode):
    """
    Move the system cursor with your index finger; pinch to click.

    Requires ``gestures.allow_mouse_control`` in the config — without it the
    mode still shows the virtual cursor but does not touch the real one, which
    makes it safe to demo.
    """

    key = "airmouse"
    label = "Raton aereo"
    description = "Controla el cursor con el dedo (requiere permiso)"
    category = ModeCategory.INTERACTION
    requires = ("hands",)
    toggles = {
        "skeleton": False, "segmentation": False, "face_mesh": False,
        "face_detect": False, "object_detect": False,
    }
    keys = {"m": "Activar/desactivar control real"}

    # The active region is a centred sub-rectangle, so you never have to reach
    # the very edge of frame to hit the edge of the screen.
    MARGIN = 0.18

    def __init__(self) -> None:
        super().__init__()
        self._enabled = False
        self._click = Debouncer(rise_seconds=0.08, fall_seconds=0.12)
        self._click_cooldown = Cooldown(0.4)
        self._clicks = 0
        self._cursor: Optional[Tuple[float, float]] = None

    def on_enter(self, state: AppState) -> None:
        super().on_enter(state)
        self._enabled = state.config.gestures.allow_mouse_control
        if not self._enabled:
            state.notify("Control real desactivado (permiso off)", "warn")

    def on_key(self, key: int, state: AppState) -> bool:
        if key == ord("m"):
            if not state.config.gestures.allow_mouse_control:
                state.notify("Habilita gestures.allow_mouse_control primero", "warn")
                return True
            self._enabled = not self._enabled
            state.notify(f"Control real: {'ON' if self._enabled else 'OFF'}")
            return True
        return False

    def process(self, ctx: FrameContext, state: AppState) -> None:
        if not ctx.has_hands:
            self._cursor = None
            return

        hand = ctx.hand_landmarks[0]
        tip = hand[INDEX_TIP]

        # Map the active region to the full 0..1 cursor space.
        span = 1.0 - 2.0 * self.MARGIN
        nx = clamp((tip.x - self.MARGIN) / span, 0.0, 1.0)
        ny = clamp((tip.y - self.MARGIN) / span, 0.0, 1.0)
        self._cursor = (nx, ny)

        clicked = self._click.update(_pinching(hand), ctx.now)
        if clicked and self._click_cooldown.trigger(ctx.now):
            self._clicks += 1
            state.bus.emit(Events.ACTION_TRIGGERED, name="mouse_click")
            if self._enabled:
                self._move_and_click(state, nx, ny)

    def _move_and_click(self, state: AppState, nx: float, ny: float) -> None:
        """Drive the real cursor through the platform bridge, if supported."""
        bridge = state.get_note("platform_bridge")
        if bridge is None or not hasattr(bridge, "move_mouse"):
            return
        try:
            _mx, _my, width, height = bridge.get_monitor_geometry()
            bridge.move_mouse(int(nx * width), int(ny * height))
            if hasattr(bridge, "click_mouse"):
                bridge.click_mouse()
        except Exception as exc:
            state.notify(f"Raton no disponible: {exc}", "warn")
            self._enabled = False

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        theme = state.theme

        # Active region outline.
        x0 = int(self.MARGIN * ctx.width)
        y0 = int(self.MARGIN * ctx.height)
        x1 = int((1.0 - self.MARGIN) * ctx.width)
        y1 = int((1.0 - self.MARGIN) * ctx.height)
        cv2.rectangle(ctx.frame, (x0, y0), (x1, y1), theme.text_dim, 1)

        if self._cursor is None:
            return
        cx = int(self._cursor[0] * ctx.width)
        cy = int(self._cursor[1] * ctx.height)
        color = theme.good if self._enabled else theme.warn
        cv2.line(ctx.frame, (cx - 14, cy), (cx + 14, cy), color, 2)
        cv2.line(ctx.frame, (cx, cy - 14), (cx, cy + 14), color, 2)
        cv2.circle(ctx.frame, (cx, cy), 8, color, 1, cv2.LINE_AA)

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        status = "ACTIVO" if self._enabled else "solo visual"
        return [f"Raton: {status}  Clicks: {self._clicks}"]


# ---------------------------------------------------------------------------
# Virtual instrument
# ---------------------------------------------------------------------------

class VirtualPianoMode(Mode):
    """
    On-screen keyboard played by touching keys with your fingertips.

    Notes are emitted as events, so anything subscribed (a synth, a logger)
    can react without this mode knowing about audio.
    """

    key = "piano"
    label = "Piano virtual"
    description = "Toca teclas en pantalla con los dedos"
    category = ModeCategory.INTERACTION
    requires = ("hands",)
    toggles = {
        "skeleton": False, "segmentation": False, "face_mesh": False,
        "face_detect": False, "object_detect": False,
    }

    NOTES = ("DO", "RE", "MI", "FA", "SOL", "LA", "SI", "DO'")
    FINGERTIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)

    def __init__(self) -> None:
        super().__init__()
        self._active: Dict[int, float] = {}      # key index -> last trigger time
        self._cooldowns = {index: Cooldown(0.28) for index in range(len(self.NOTES))}
        self._played = 0

    def _key_box(self, index: int, width: int, height: int) -> Tuple[int, int, int, int]:
        """Pixel rectangle of one piano key."""
        key_width = width // len(self.NOTES)
        x0 = index * key_width
        return x0, int(height * 0.62), x0 + key_width - 4, height - 20

    def process(self, ctx: FrameContext, state: AppState) -> None:
        if not ctx.has_hands:
            return

        for hand in ctx.hand_landmarks:
            for tip_index in self.FINGERTIPS:
                tip = hand[tip_index]
                px, py = to_pixels(tip, ctx.width, ctx.height)
                for index in range(len(self.NOTES)):
                    x0, y0, x1, y1 = self._key_box(index, ctx.width, ctx.height)
                    if x0 <= px <= x1 and y0 <= py <= y1:
                        if self._cooldowns[index].trigger(ctx.now):
                            self._active[index] = ctx.now
                            self._played += 1
                            state.bus.emit(
                                Events.ACTION_TRIGGERED,
                                name="note", note=self.NOTES[index], index=index,
                            )
                            state.set_gesture(f"NOTA {self.NOTES[index]}", ctx.now)

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        theme = state.theme
        for index, note in enumerate(self.NOTES):
            x0, y0, x1, y1 = self._key_box(index, ctx.width, ctx.height)
            recent = ctx.now - self._active.get(index, -10.0)
            if recent < 0.25:
                color = theme.category_color(index)
                cv2.rectangle(ctx.frame, (x0, y0), (x1, y1), color, -1)
            else:
                overlay = ctx.frame.copy()
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (240, 240, 240), -1)
                cv2.addWeighted(overlay, 0.35, ctx.frame, 0.65, 0, ctx.frame)
            cv2.rectangle(ctx.frame, (x0, y0), (x1, y1), (30, 30, 30), 2)
            cv2.putText(
                ctx.frame, note, (x0 + 12, y1 - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA,
            )

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        return [f"Notas tocadas: {self._played}"]


# ---------------------------------------------------------------------------
# System control
# ---------------------------------------------------------------------------

class MediaControlMode(Mode):
    """
    Control media playback with hand gestures.

    Open palm pauses, fist plays, swipes skip tracks. Requires
    ``gestures.allow_media_keys``; without it gestures are recognised and
    displayed but no keys are sent.
    """

    key = "media"
    label = "Control multimedia"
    description = "Play/pausa y pistas con gestos de mano"
    category = ModeCategory.INTERACTION
    requires = ("hands",)
    toggles = {
        "skeleton": False, "segmentation": False, "face_mesh": False,
        "face_detect": False, "object_detect": False,
    }

    SWIPE_DISTANCE = 0.22
    SWIPE_SECONDS = 0.6

    def __init__(self) -> None:
        super().__init__()
        self._cooldown = Cooldown(1.2)
        self._history: List[Tuple[float, float]] = []      # (time, x)
        self._last_action = ""
        self._actions = 0
        self._allowed = False

    def on_enter(self, state: AppState) -> None:
        super().on_enter(state)
        self._allowed = state.config.gestures.allow_media_keys
        if not self._allowed:
            state.notify("Teclas multimedia desactivadas (permiso off)", "warn")

    def process(self, ctx: FrameContext, state: AppState) -> None:
        if not ctx.has_hands:
            self._history.clear()
            return

        hand = ctx.hand_landmarks[0]
        wrist = hand[WRIST]
        self._history.append((ctx.now, wrist.x))
        self._history = [h for h in self._history if ctx.now - h[0] <= self.SWIPE_SECONDS]

        action = self._detect_swipe()
        if action is None:
            gestures = ctx.hand_info.get("gestures", [])
            gesture = gestures[0] if gestures else None
            if gesture == "open_palm":
                action = "pausa"
            elif gesture == "fist":
                action = "play"

        if action and self._cooldown.trigger(ctx.now):
            self._last_action = action
            self._actions += 1
            self._history.clear()
            state.set_gesture(action.upper(), ctx.now)
            state.bus.emit(Events.ACTION_TRIGGERED, name=f"media_{action}")
            if self._allowed:
                self._send(state, action)

    def _detect_swipe(self) -> Optional[str]:
        """Left/right swipe from the wrist's horizontal travel."""
        if len(self._history) < 4:
            return None
        travel = self._history[-1][1] - self._history[0][1]
        if travel > self.SWIPE_DISTANCE:
            return "siguiente"
        if travel < -self.SWIPE_DISTANCE:
            return "anterior"
        return None

    def _send(self, state: AppState, action: str) -> None:
        bridge = state.get_note("platform_bridge")
        if bridge is None or not hasattr(bridge, "send_media_key"):
            return
        mapping = {
            "play": "play_pause", "pausa": "play_pause",
            "siguiente": "next_track", "anterior": "prev_track",
        }
        try:
            bridge.send_media_key(mapping.get(action, "play_pause"))
        except Exception as exc:
            state.notify(f"Tecla multimedia fallo: {exc}", "warn")

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        mode = "activo" if self._allowed else "solo deteccion"
        return [
            f"Multimedia ({mode})  Acciones: {self._actions}",
            f"Ultima: {self._last_action or '-'}",
            "Palma=pausa  Puño=play  Deslizar=pista",
        ]


class PresentationMode(Mode):
    """
    Navigate slides by pointing left or right, plus a laser-pointer dot.

    Built for standing at a distance from the machine, so it uses whole-arm
    pointing rather than fine finger gestures.
    """

    key = "presentation"
    label = "Presentacion"
    description = "Pasa diapositivas señalando y apunta con laser"
    category = ModeCategory.INTERACTION
    requires = ("pose", "hands")
    toggles = {
        "skeleton": False, "segmentation": False, "face_mesh": False,
        "face_detect": False, "object_detect": False, "telemetry": False,
    }

    POINT_THRESHOLD = 0.28     # wrist distance beyond the shoulder, normalized

    def __init__(self) -> None:
        super().__init__()
        self._cooldown = Cooldown(1.5)
        self._slide = 1
        self._laser: Optional[Tuple[int, int]] = None

    def process(self, ctx: FrameContext, state: AppState) -> None:
        self._laser = None

        if ctx.has_hands:
            hand = ctx.hand_landmarks[0]
            if _index_extended(hand):
                self._laser = to_pixels(hand[INDEX_TIP], ctx.width, ctx.height)

        if not ctx.has_pose:
            return

        landmarks = ctx.primary_pose
        left_shoulder = landmarks[L.LEFT_SHOULDER]
        right_shoulder = landmarks[L.RIGHT_SHOULDER]
        left_wrist = landmarks[L.LEFT_WRIST]
        right_wrist = landmarks[L.RIGHT_WRIST]

        direction = None
        if left_wrist.x - left_shoulder.x > self.POINT_THRESHOLD:
            direction = "siguiente"
        elif right_shoulder.x - right_wrist.x > self.POINT_THRESHOLD:
            direction = "anterior"

        if direction and self._cooldown.trigger(ctx.now):
            self._slide += 1 if direction == "siguiente" else -1
            self._slide = max(1, self._slide)
            state.set_gesture(direction.upper(), ctx.now)
            state.bus.emit(Events.ACTION_TRIGGERED, name=f"slide_{direction}")

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        if self._laser:
            cv2.circle(ctx.frame, self._laser, 10, (60, 60, 255), -1, cv2.LINE_AA)
            cv2.circle(ctx.frame, self._laser, 16, (120, 120, 255), 2, cv2.LINE_AA)

        text = f"Diapositiva {self._slide}"
        cv2.putText(
            ctx.frame, text, (ctx.width - 220, ctx.height - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, state.theme.text, 2, cv2.LINE_AA,
        )

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        return [f"Diapositiva: {self._slide}", "Brazo derecho=siguiente, izquierdo=anterior"]


class SignCountMode(Mode):
    """
    Reads numbers from your fingers and builds a running expression.

    A simple, dependency-free calculator you operate entirely with your hands:
    hold up a number, pinch to accept it.
    """

    key = "signs"
    label = "Numeros con manos"
    description = "Cuenta dedos y compone operaciones"
    category = ModeCategory.INTERACTION
    requires = ("hands",)
    toggles = {
        "skeleton": False, "segmentation": False, "face_mesh": False,
        "face_detect": False, "object_detect": False,
    }
    keys = {"c": "Limpiar expresion"}

    def __init__(self) -> None:
        super().__init__()
        self._expression: List[str] = []
        self._stable = Debouncer(rise_seconds=0.45, fall_seconds=0.2)
        self._cooldown = Cooldown(1.0)
        self._pending: Optional[int] = None

    def on_key(self, key: int, state: AppState) -> bool:
        if key == ord("c"):
            self._expression.clear()
            state.notify("Expresion limpiada")
            return True
        return False

    def process(self, ctx: FrameContext, state: AppState) -> None:
        counts = ctx.hand_info.get("finger_counts", [])
        if not counts:
            self._pending = None
            return

        total = sum(counts)
        self._pending = total

        if ctx.has_hands and _pinching(ctx.hand_landmarks[0]):
            if self._cooldown.trigger(ctx.now):
                self._expression.append(str(total))
                if len(self._expression) > 12:
                    self._expression.pop(0)
                state.set_gesture(f"NUM {total}", ctx.now)

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        theme = state.theme
        if self._pending is not None:
            text = str(self._pending)
            size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 6)
            cv2.putText(
                ctx.frame, text,
                ((ctx.width - size[0]) // 2, ctx.height // 2 + size[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 3.0, theme.accent, 6, cv2.LINE_AA,
            )

        if self._expression:
            joined = " ".join(self._expression)
            cv2.putText(
                ctx.frame, joined, (24, ctx.height - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, theme.text, 2, cv2.LINE_AA,
            )

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        return [
            f"Detectado: {self._pending if self._pending is not None else '-'}",
            "Pellizca para confirmar",
        ]


def interaction_modes() -> List[Mode]:
    """Every interaction mode, in menu order."""
    return [
        AirDrawMode(), AirMouseMode(), VirtualPianoMode(),
        MediaControlMode(), PresentationMode(), SignCountMode(),
    ]
