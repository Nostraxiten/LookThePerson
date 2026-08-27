"""
Detection modes for LookThePerson.

These configure *what* is tracked and shown rather than transforming the image.
They are the modes the original four numeric presets tried to be, now with
their own per-mode behaviour, HUD and key bindings.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2

from analytics.angles import PoseLandmark as L
from analytics.angles import body_orientation, describe_angle
from core.geometry import to_pixels
from core.state import AppState, FrameContext
from modes.base import Mode, ModeCategory

__all__ = [
    "FullMode",
    "PoseMode",
    "HandsMode",
    "FaceMode",
    "ObjectsMode",
    "MinimalMode",
    "CrowdMode",
    "detection_modes",
]


class FullMode(Mode):
    """Everything on at once — the showcase configuration."""

    key = "full"
    label = "Completo"
    description = "Todos los modelos activos a la vez"
    category = ModeCategory.DETECTION
    requires = ("pose", "hands", "face_mesh", "face_detect", "object")
    toggles = {
        "skeleton": True, "segmentation": True, "face_mesh": True,
        "face_detect": True, "object_detect": True, "bounding_boxes": True,
    }

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        parts = []
        if ctx.person_count:
            parts.append(f"Personas: {ctx.person_count}")
        if ctx.hand_count:
            parts.append(f"Manos: {ctx.hand_count}")
        if ctx.face_landmarks:
            parts.append(f"Caras: {len(ctx.face_landmarks)}")
        return [" | ".join(parts)] if parts else []


class PoseMode(Mode):
    """
    Body tracking only, with live joint angles.

    The lightest full-body configuration: no face or object models run, so it
    holds a high frame rate even on modest hardware.
    """

    key = "pose"
    label = "Solo Pose"
    description = "Esqueleto corporal con angulos articulares"
    category = ModeCategory.DETECTION
    requires = ("pose",)
    toggles = {
        "skeleton": True, "segmentation": True, "face_mesh": False,
        "face_detect": False, "object_detect": False, "bounding_boxes": True,
    }
    keys = {"a": "Mostrar/ocultar angulos"}

    def __init__(self) -> None:
        super().__init__()
        self.show_angles = True

    def on_key(self, key: int, state: AppState) -> bool:
        if key == ord("a"):
            self.show_angles = not self.show_angles
            state.notify(f"Angulos: {'ON' if self.show_angles else 'OFF'}")
            return True
        return False

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        if not self.show_angles or not ctx.has_pose:
            return

        landmarks = ctx.primary_pose
        theme = state.theme
        joints = {
            "left_elbow": L.LEFT_ELBOW, "right_elbow": L.RIGHT_ELBOW,
            "left_knee": L.LEFT_KNEE, "right_knee": L.RIGHT_KNEE,
            "left_shoulder": L.LEFT_SHOULDER, "right_shoulder": L.RIGHT_SHOULDER,
        }
        for name, index in joints.items():
            angle = ctx.angles.get(name)
            if angle is None:
                continue
            x, y = to_pixels(landmarks[index], ctx.width, ctx.height)
            cv2.putText(
                ctx.frame, f"{angle:.0f}", (x + 10, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, theme.accent, 1, cv2.LINE_AA,
            )

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        if not ctx.has_pose:
            return []
        lines = [f"Orientacion: {body_orientation(ctx.primary_pose)}"]
        trunk = ctx.angles.get("trunk")
        if trunk is not None:
            lines.append(f"Inclinacion torso: {trunk:.0f}°")
        return lines


class HandsMode(Mode):
    """
    Hand tracking with finger counting and per-hand gesture names.

    Pose stays on at low cost so body gestures keep working, but nothing is
    drawn for it — the frame stays clean around the hands.
    """

    key = "hands"
    label = "Solo Manos"
    description = "Seguimiento de manos, dedos y gestos"
    category = ModeCategory.DETECTION
    requires = ("hands",)
    toggles = {
        "skeleton": False, "segmentation": False, "face_mesh": False,
        "face_detect": False, "object_detect": False, "bounding_boxes": False,
    }

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        if not ctx.has_hands:
            return

        theme = state.theme
        counts = ctx.hand_info.get("finger_counts", [])
        gestures = ctx.hand_info.get("gestures", [])

        for index, landmarks in enumerate(ctx.hand_landmarks):
            wrist = landmarks[0]
            x, y = to_pixels(wrist, ctx.width, ctx.height)

            label_parts = []
            if index < len(ctx.handedness):
                label_parts.append(ctx.handedness[index])
            if index < len(counts):
                label_parts.append(f"{counts[index]} dedos")
            if index < len(gestures) and gestures[index]:
                label_parts.append(str(gestures[index]).upper())

            for offset, text in enumerate(label_parts):
                cv2.putText(
                    ctx.frame, text, (x - 30, y + 30 + offset * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, theme.hand, 2, cv2.LINE_AA,
                )

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        counts = ctx.hand_info.get("finger_counts", [])
        if not counts:
            return ["Sin manos detectadas"]
        return [f"Dedos: {' + '.join(str(c) for c in counts)} = {sum(counts)}"]


class FaceMode(Mode):
    """Facial mesh, expressions and gaze."""

    key = "face"
    label = "Solo Cara"
    description = "Malla facial, expresiones y mirada"
    category = ModeCategory.DETECTION
    requires = ("face_mesh", "face_detect")
    toggles = {
        "skeleton": False, "segmentation": False, "face_mesh": True,
        "face_detect": True, "object_detect": False, "bounding_boxes": True,
    }

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        if not ctx.face_info:
            return ["Sin cara detectada"]

        lines: List[str] = []
        expressions = ctx.face_info.get("raw", {})
        active = [name for name, value in expressions.items() if value]
        if active:
            lines.append("Expr: " + ", ".join(active[:4]))

        gaze = ctx.face_info.get("gaze")
        if gaze:
            horizontal = "izq" if gaze[0] < -0.25 else "der" if gaze[0] > 0.25 else "centro"
            vertical = "arriba" if gaze[1] < -0.25 else "abajo" if gaze[1] > 0.25 else "centro"
            lines.append(f"Mirada: {horizontal}/{vertical}")

        pose = ctx.face_info.get("head_pose")
        if pose:
            lines.append(
                f"Cabeza: yaw {pose['yaw']:.0f}° pitch {pose['pitch']:.0f}° roll {pose['roll']:.0f}°"
            )
        return lines


class ObjectsMode(Mode):
    """
    Object detection with a running tally of what has been seen.

    The tally persists while the mode stays active, which turns it into a
    simple inventory of everything that passed the camera.
    """

    key = "objects"
    label = "Objetos"
    description = "Deteccion de objetos COCO con inventario"
    category = ModeCategory.DETECTION
    requires = ("object",)
    toggles = {
        "skeleton": False, "segmentation": False, "face_mesh": False,
        "face_detect": False, "object_detect": True, "bounding_boxes": True,
    }
    keys = {"z": "Reiniciar inventario"}

    def __init__(self) -> None:
        super().__init__()
        self._seen: Dict[str, int] = {}

    def reset(self, state: AppState) -> None:
        self._seen.clear()

    def on_key(self, key: int, state: AppState) -> bool:
        if key == ord("z"):
            self._seen.clear()
            state.notify("Inventario reiniciado")
            return True
        return False

    def process(self, ctx: FrameContext, state: AppState) -> None:
        result = ctx.object_result
        if not result or not getattr(result, "detections", None):
            return
        for detection in result.detections:
            if not detection.categories:
                continue
            name = detection.categories[0].category_name or "desconocido"
            self._seen[name] = self._seen.get(name, 0) + 1

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        if not self._seen:
            return ["Sin objetos detectados"]
        top = sorted(self._seen.items(), key=lambda item: item[1], reverse=True)[:5]
        return ["Inventario:"] + [f"  {name}: {count}" for name, count in top]


class MinimalMode(Mode):
    """
    Clean video with almost no overlay.

    Useful as a plain mirror, and for recording footage where the HUD would be
    in the way.
    """

    key = "minimal"
    label = "Minimo"
    description = "Video limpio, sin overlays"
    category = ModeCategory.DETECTION
    requires = ("pose",)
    toggles = {
        "skeleton": False, "segmentation": False, "face_mesh": False,
        "face_detect": False, "object_detect": False, "bounding_boxes": False,
        "telemetry": False, "grid": False,
    }

    def status_text(self, ctx: FrameContext, state: AppState) -> Optional[str]:
        return ""


class CrowdMode(Mode):
    """
    Multi-person tracking: counts people and colours each one differently.

    Also reports the maximum simultaneous count seen, which makes it usable as
    a simple occupancy monitor.
    """

    key = "crowd"
    label = "Multitud"
    description = "Cuenta y distingue varias personas"
    category = ModeCategory.DETECTION
    requires = ("pose", "face_detect")
    toggles = {
        "skeleton": True, "segmentation": False, "face_mesh": False,
        "face_detect": True, "object_detect": False, "bounding_boxes": True,
    }

    def __init__(self) -> None:
        super().__init__()
        self._peak = 0
        self._total_samples = 0
        self._sum = 0

    def reset(self, state: AppState) -> None:
        self._peak = 0
        self._total_samples = 0
        self._sum = 0

    def process(self, ctx: FrameContext, state: AppState) -> None:
        count = ctx.person_count
        self._peak = max(self._peak, count)
        self._total_samples += 1
        self._sum += count

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        theme = state.theme
        for index, landmarks in enumerate(ctx.pose_landmarks):
            color = theme.category_color(index)
            head = landmarks[L.NOSE]
            x, y = to_pixels(head, ctx.width, ctx.height)
            cv2.putText(
                ctx.frame, f"P{index + 1}", (x - 14, y - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
            )

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        average = self._sum / self._total_samples if self._total_samples else 0.0
        return [
            f"Personas ahora: {ctx.person_count}",
            f"Maximo: {self._peak}  Media: {average:.1f}",
        ]


def detection_modes() -> List[Mode]:
    """Every detection mode, in menu order."""
    return [
        FullMode(), PoseMode(), HandsMode(), FaceMode(),
        ObjectsMode(), MinimalMode(), CrowdMode(),
    ]
