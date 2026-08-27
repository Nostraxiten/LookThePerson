"""
Professional surveillance mode for LookThePerson.

``security`` turns the app into a single-channel CCTV station. It is
deliberately the odd one out among the modes: every decorative overlay is
switched off, the shared HUD is suppressed, and the mode draws its own operator
display instead — because a surveillance picture is evidence, and anything
painted over it that is not information is in the way.

What it puts on screen:

* **Sensor control** — day / IR / intensifier / thermal / shadow-lift looks with
  metered auto switching and auto-gain (see :mod:`fx.nightvision`).
* **Subject identification** — persistent ``SUJ-nn`` labels that survive a
  subject leaving and re-entering frame, with build, stature estimate, dwell
  time and a per-subject confidence (see :mod:`analytics.identity`).
* **Detection zone** — a named region; an identified subject entering it while
  the system is armed raises an intrusion event.
* **Event log** — timestamped, on screen and exportable to JSON.
* **DVR overlay** — channel label, burnt-in timestamp, arm state, recording
  indicator, sensor readout and signal bars.

Scope, stated plainly: identification here means *telling apart the few people
currently in view* from their skeleton proportions. It carries no identity
between sessions, stores no face data, and cannot recognise a stranger. The
stature figure is a proportional estimate, not a measurement.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from analytics.identity import (
    Detection,
    PersonTracker,
    TrackedPerson,
    build_signature,
    pose_box,
    stature_span,
)
from core.events import Events
from core.filters import Cooldown, Debouncer
from core.geometry import clamp, safe_int
from core.state import AppState, FrameContext
from fx.nightvision import NightVisionProcessor
from modes.base import Mode, ModeCategory

__all__ = ["SecurityMode", "SecurityEvent", "ZONES", "Zone"]


# ---------------------------------------------------------------------------
# Zones
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Zone:
    """A named detection region, in normalized frame coordinates."""

    name: str
    box: Optional[Tuple[float, float, float, float]]   # None = whole frame

    def contains(self, point: Tuple[float, float]) -> bool:
        """Whether a normalized point falls inside the zone."""
        if self.box is None:
            return True
        x0, y0, x1, y1 = self.box
        return x0 <= point[0] <= x1 and y0 <= point[1] <= y1


#: Selectable zones, in cycling order. ``UMBRAL`` is the bottom strip — a
#: doorway seen from a camera mounted above it, which is the common case.
ZONES: Tuple[Zone, ...] = (
    Zone("PERIMETRO", None),
    Zone("CENTRO", (0.30, 0.20, 0.70, 0.90)),
    Zone("IZQUIERDA", (0.00, 0.00, 0.42, 1.00)),
    Zone("DERECHA", (0.58, 0.00, 1.00, 1.00)),
    Zone("UMBRAL", (0.15, 0.62, 0.85, 1.00)),
)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@dataclass
class SecurityEvent:
    """One entry in the security log."""

    stamp: str            # wall-clock HH:MM:SS
    uptime: float         # seconds since the app started
    kind: str             # ALTA | BAJA | REGRESO | INTRUSION | MULTIPLE
    subject: str = ""
    detail: str = ""

    def line(self) -> str:
        """Single-line form for the on-screen log."""
        parts = [self.stamp, self.kind]
        if self.subject:
            parts.append(self.subject)
        if self.detail:
            parts.append(self.detail)
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time": self.stamp,
            "uptime": round(self.uptime, 2),
            "kind": self.kind,
            "subject": self.subject,
            "detail": self.detail,
        }


#: Colour role per event kind, resolved against the active theme.
_EVENT_LEVEL = {
    "ALTA": "good",
    "REGRESO": "warn",
    "BAJA": "dim",
    "MULTIPLE": "warn",
    "INTRUSION": "danger",
}


class SecurityMode(Mode):
    """
    Surveillance station: night vision, subject identification, zone alarms.

    Everything is driven from the keyboard so the mode can be operated without
    the mode picker open — see :attr:`keys`.
    """

    key = "security"
    label = "Seguridad"
    description = "Camara de vigilancia con vision nocturna e identificacion de personas"
    category = ModeCategory.UTILITY
    requires = ("pose", "face_detect")

    #: Everything decorative is off. The mode paints its own operator display,
    #: including its own status line, so the shared HUD is suppressed too.
    toggles = {
        "skeleton": False, "segmentation": False, "face_mesh": False,
        "face_detect": True, "object_detect": False, "bounding_boxes": False,
        "grid": False, "night_mode": False, "trails": False, "heatmap": False,
        "landmark_ids": False, "fps_graph": False, "debug": False,
        "help": False, "telemetry": False,
    }

    #: The mode paints a complete DVR display, so the shared HUD must not also
    #: put a status line and an FPS counter in the corners it uses.
    owns_overlay = True

    keys = {
        "a": "Armar/desarmar",
        "n": "Modo de vision nocturna",
        "j": "Ganancia automatica",
        "+": "Ganancia +",
        "-": "Ganancia -",
        "x": "Zona de deteccion",
        "c": "Canal de camara",
        "i": "Identificacion de sujetos",
        "l": "Registro de eventos",
        "o": "Detalle del OSD",
        "e": "Exportar registro",
        "z": "Borrar registro",
    }

    ARM_DELAY = 10.0
    MAX_EVENTS = 250
    CHANNELS = ("CAM-01", "CAM-02", "ENTRADA", "PASILLO", "EXTERIOR")

    def __init__(self) -> None:
        super().__init__()
        self.nightvision = NightVisionProcessor(mode="auto")
        self.tracker = PersonTracker()

        self._armed = False
        self._arm_at: Optional[float] = None
        self._events: List[SecurityEvent] = []
        self._known: Dict[int, str] = {}          # pid -> label, for ALTA/BAJA
        self._intruded: set = set()               # pids already reported this visit

        self._zone_index = 0
        self._channel_index = 0
        self._show_ids = True
        self._show_log = True
        self._osd_detail = 1                      # 0 minimal, 1 normal, 2 full

        self._presence = Debouncer(rise_seconds=0.5, fall_seconds=2.5)
        self._alert_cooldown = Cooldown(6.0)
        self._multi_cooldown = Cooldown(15.0)
        self._peak_subjects = 0
        self._last_export: Optional[str] = None
        self._last_status = ""

    # -- Lifecycle ----------------------------------------------------------

    @property
    def zone(self) -> Zone:
        return ZONES[self._zone_index % len(ZONES)]

    @property
    def channel(self) -> str:
        return self.CHANNELS[self._channel_index % len(self.CHANNELS)]

    @property
    def armed(self) -> bool:
        return self._armed

    def on_enter(self, state: AppState) -> None:
        super().on_enter(state)
        self._armed = False
        self._arm_at = None
        self.nightvision.reset()

    def reset(self, state: AppState) -> None:
        """Clear the log and every identity, keeping the sensor settings."""
        self.tracker.reset()
        self._events.clear()
        self._known.clear()
        self._intruded.clear()
        self._peak_subjects = 0
        self._armed = False
        self._arm_at = None
        state.notify("Sistema reiniciado", "warn")

    # -- Keys ---------------------------------------------------------------

    def on_key(self, key: int, state: AppState) -> bool:
        if key == ord("a"):
            self._toggle_arm(state)
            return True
        if key == ord("n"):
            mode = self.nightvision.cycle()
            state.notify(f"Vision: {mode}")
            return True
        if key == ord("j"):
            auto = self.nightvision.toggle_auto_gain()
            state.notify(f"Ganancia automatica: {'si' if auto else 'no'}")
            return True
        if key == ord("+"):
            state.notify(f"Ganancia: {self.nightvision.adjust_gain(0.15):.2f}x")
            return True
        if key == ord("-"):
            state.notify(f"Ganancia: {self.nightvision.adjust_gain(-0.15):.2f}x")
            return True
        if key == ord("x"):
            self._zone_index = (self._zone_index + 1) % len(ZONES)
            self._intruded.clear()
            state.notify(f"Zona: {self.zone.name}")
            return True
        if key == ord("c"):
            self._channel_index = (self._channel_index + 1) % len(self.CHANNELS)
            state.notify(f"Canal: {self.channel}")
            return True
        if key == ord("i"):
            self._show_ids = not self._show_ids
            state.notify(f"Identificacion: {'si' if self._show_ids else 'no'}")
            return True
        if key == ord("l"):
            self._show_log = not self._show_log
            return True
        if key == ord("o"):
            self._osd_detail = (self._osd_detail + 1) % 3
            return True
        if key == ord("e"):
            self._export(state)
            return True
        if key == ord("z"):
            self._events.clear()
            self._intruded.clear()
            state.notify("Registro borrado")
            return True
        return False

    def _toggle_arm(self, state: AppState) -> None:
        if self._armed or self._arm_at is not None:
            self._armed = False
            self._arm_at = None
            state.notify("Sistema desarmado")
            return
        self._arm_at = state.uptime + self.ARM_DELAY
        state.notify(f"Armando en {self.ARM_DELAY:.0f}s", "warn")

    # -- Processing ---------------------------------------------------------

    def process(self, ctx: FrameContext, state: AppState) -> None:
        # 1. Sensor stage. Runs before anything else so identification and the
        #    operator both see the same picture.
        ctx.frame = self.nightvision.process(ctx.frame)

        # 2. Arming countdown.
        if self._arm_at is not None and state.uptime >= self._arm_at:
            self._armed = True
            self._arm_at = None
            self._intruded.clear()
            state.notify("SISTEMA ARMADO", "warn")
            self._log("MULTIPLE" if ctx.person_count > 1 else "ALTA",
                      state, detail="sistema armado")

        # 3. Identification.
        subjects = self.tracker.update(self._detections(ctx), ctx.now)
        ctx.extras["security_subjects"] = subjects
        self._peak_subjects = max(self._peak_subjects, len(subjects))

        # 4. Events derived from the identity set. Presence is debounced so a
        #    single-frame false detection cannot raise an alarm.
        self._track_arrivals(subjects, state)
        present = self._presence.update(bool(subjects), ctx.now)
        if self._armed and present:
            self._check_zone(subjects, state, ctx.now)
            if len(subjects) > 1 and self._multi_cooldown.trigger(ctx.now):
                self._log("MULTIPLE", state, detail=f"{len(subjects)} sujetos")

    def _detections(self, ctx: FrameContext) -> List[Detection]:
        """Turn this frame's poses into tracker detections."""
        detections: List[Detection] = []
        has_face = bool(ctx.face_landmarks) or self._face_count(ctx) > 0
        for landmarks in ctx.pose_landmarks:
            box = pose_box(landmarks)
            if box is None:
                continue
            detections.append(Detection(
                box=box,
                signature=build_signature(landmarks),
                span=stature_span(landmarks),
                has_face=has_face,
            ))
        return detections

    @staticmethod
    def _face_count(ctx: FrameContext) -> int:
        result = ctx.face_detect_result
        detections = getattr(result, "detections", None) if result else None
        return len(detections) if detections else 0

    def _track_arrivals(self, subjects: Sequence[TrackedPerson], state: AppState) -> None:
        """Emit ALTA / REGRESO / BAJA as the identity set changes."""
        current = {s.pid: s for s in subjects}

        for pid, subject in current.items():
            if pid in self._known:
                continue
            self._known[pid] = subject.label
            if subject.reappearances:
                self._log("REGRESO", state, subject=subject.label,
                          detail=f"visita {subject.reappearances + 1}")
            else:
                self._log("ALTA", state, subject=subject.label,
                          detail=f"perfil {subject.code}")

        for pid in [p for p in self._known if p not in current]:
            self._log("BAJA", state, subject=self._known.pop(pid))
            self._intruded.discard(pid)

    def _check_zone(
        self,
        subjects: Sequence[TrackedPerson],
        state: AppState,
        now: float,
    ) -> None:
        """
        Raise an intrusion the first time a subject enters the armed zone.

        *now* is the frame clock, the same one the presence debouncer and the
        multi-subject cooldown run on. Every rate limit in the mode has to read
        one clock or their windows drift apart under a replayed source.
        """
        zone = self.zone
        for subject in subjects:
            inside = zone.contains(subject.centroid)
            if not inside:
                self._intruded.discard(subject.pid)
                continue
            if subject.pid in self._intruded:
                continue
            if not self._alert_cooldown.trigger(now):
                continue
            self._intruded.add(subject.pid)
            self._log("INTRUSION", state, subject=subject.label,
                      detail=f"zona {zone.name}")
            state.bus.emit(
                Events.MOTION_ALERT,
                people=len(subjects), zone=zone.name, subject=subject.label,
            )

    def _log(self, kind: str, state: AppState, subject: str = "", detail: str = "") -> None:
        """Append an event, notify the operator and mirror it onto the bus."""
        event = SecurityEvent(
            stamp=time.strftime("%H:%M:%S"),
            uptime=state.uptime,
            kind=kind, subject=subject, detail=detail,
        )
        self._events.append(event)
        if len(self._events) > self.MAX_EVENTS:
            del self._events[:len(self._events) - self.MAX_EVENTS]

        level = _EVENT_LEVEL.get(kind, "info")
        if kind in ("INTRUSION", "MULTIPLE"):
            state.notify(event.line(), "danger" if kind == "INTRUSION" else "warn")
        elif kind == "ALTA":
            state.notify(f"Sujeto {subject} identificado", "good")
        state.increment(f"security_{kind.lower()}")
        if level == "danger":
            state.increment("security_alerts")

    # -- Export -------------------------------------------------------------

    def _export(self, state: AppState) -> Optional[str]:
        """Write the event log to JSON next to the session exports."""
        if not self._events:
            state.notify("Registro vacio, nada que exportar", "warn")
            return None

        base = state.config.analytics.export_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sessions"
        )
        path = os.path.join(base, f"security_{time.strftime('%Y%m%d_%H%M%S')}.json")
        document = {
            "channel": self.channel,
            "zone": self.zone.name,
            "sensor": self.nightvision.status(),
            "subjects_identified": self.tracker.total_identified,
            "peak_subjects": self._peak_subjects,
            "events": [event.to_dict() for event in self._events],
        }
        try:
            os.makedirs(base, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(document, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
        except OSError as exc:
            state.notify("No pude exportar el registro", "danger")
            print(f"[security] Error exportando: {exc}", flush=True)
            return None

        self._last_export = path
        state.notify(f"Registro exportado ({len(self._events)} eventos)", "good")
        print(f"[security] Registro exportado en {path}", flush=True)
        return path

    # -- Drawing ------------------------------------------------------------

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        theme = state.theme
        if self._show_ids:
            self._draw_subjects(ctx, state)
        if self._osd_detail > 0:
            self._draw_zone(ctx, theme)
        self._draw_osd(ctx, state)
        if self._show_log and self._osd_detail > 0:
            self._draw_log(ctx, theme)

    def _draw_subjects(self, ctx: FrameContext, state: AppState) -> None:
        """Reticle and identity card for each tracked subject."""
        theme = state.theme
        zone = self.zone
        reference = state.config.analytics.user_height_cm

        for subject in ctx.extras.get("security_subjects", []):
            x0 = safe_int(subject.box[0] * ctx.width)
            y0 = safe_int(subject.box[1] * ctx.height)
            x1 = safe_int(subject.box[2] * ctx.width)
            y1 = safe_int(subject.box[3] * ctx.height)
            if x1 <= x0 or y1 <= y0:
                continue

            inside = zone.contains(subject.centroid)
            color = theme.danger if (self._armed and inside) else theme.accent
            _corner_reticle(ctx.frame, (x0, y0, x1, y1), color)

            if self._osd_detail == 0:
                continue

            lines = [f"{subject.label}  {subject.code}"]
            if self._osd_detail >= 1:
                dwell = subject.dwell(ctx.now)
                lines.append(f"{dwell:.0f}s  conf {subject.confidence():.0%}")
            if self._osd_detail >= 2:
                height_cm = subject.estimated_height_cm(reference)
                stature = f"{height_cm:.0f}cm est." if height_cm else "altura n/d"
                lines.append(f"{stature}  {subject.build}")
                lines.append(f"cara {'si' if subject.has_face else 'no'}"
                             f"  visitas {subject.reappearances + 1}")

            _label_block(ctx.frame, (x0, y0), lines, color, theme.text)

    def _draw_zone(self, ctx: FrameContext, theme: Any) -> None:
        """Outline the active detection zone."""
        if self.zone.box is None:
            return
        x0, y0, x1, y1 = self.zone.box
        px0, py0 = safe_int(x0 * ctx.width), safe_int(y0 * ctx.height)
        px1, py1 = safe_int(x1 * ctx.width), safe_int(y1 * ctx.height)
        color = theme.danger if self._armed else theme.text_dim

        # Dashed rather than solid, so the zone never reads as a detection.
        _dashed_rect(ctx.frame, (px0, py0, px1, py1), color)
        cv2.putText(ctx.frame, f"ZONA {self.zone.name}", (px0 + 6, py0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    def _draw_osd(self, ctx: FrameContext, state: AppState) -> None:
        """The DVR overlay: channel, sensor, arm state, timestamp."""
        theme = state.theme
        frame = ctx.frame
        width, height = ctx.width, ctx.height
        sensor = self.nightvision

        # -- Top strip: channel and sensor readout.
        cv2.putText(frame, self.channel, (14, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.62, theme.text, 1, cv2.LINE_AA)

        readout = f"{sensor.effective_mode.upper()}"
        if sensor.mode == "auto":
            readout += " (AUTO)"
        if self._osd_detail >= 1:
            readout += f"  {sensor.luminance:.0f}lx  G{sensor.applied_gain:.2f}x"
            if sensor.auto_gain:
                readout += " AGC"
        cv2.putText(frame, readout, (14, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    theme.warn if sensor.is_night else theme.good, 1, cv2.LINE_AA)

        # -- Top right: arm state and recording.
        if self._armed:
            _blinking_dot(frame, (width - 24, 26), ctx.now, theme.danger)
            _right_text(frame, "ARMADO", (width - 40, 32), theme.danger, 0.52, 2)
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1), theme.danger, 3)
        elif self._arm_at is not None:
            remaining = max(0.0, self._arm_at - state.uptime)
            _right_text(frame, f"ARMANDO {remaining:.0f}s", (width - 16, 32),
                        theme.warn, 0.5, 2)
        else:
            _right_text(frame, "DESARMADO", (width - 16, 32), theme.text_dim, 0.46, 1)

        if state.recording:
            _blinking_dot(frame, (width - 24, 54), ctx.now, theme.danger, radius=5)
            _right_text(frame, "REC", (width - 40, 58), theme.danger, 0.44, 1)

        # -- Subject counter.
        subjects = ctx.extras.get("security_subjects", [])
        if self._osd_detail >= 1:
            counter = f"SUJETOS {len(subjects)}"
            if self.tracker.total_identified:
                counter += f" / {self.tracker.total_identified}"
            _right_text(frame, counter, (width - 16, 80),
                        theme.accent if subjects else theme.text_dim, 0.44, 1)
            if self._osd_detail >= 2:
                _signal_bars(frame, (width - 76, 96), state.fps.fps,
                             state.config.camera.fps, theme)

        # -- Timestamp burn-in, bottom left, as recorded footage carries.
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, stamp, (14, height - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)

        # -- Bottom right: key reminder, replacing the suppressed HUD bar.
        if self._osd_detail >= 1:
            hint = "A=armar N=vision X=zona I=ids L=log E=exportar"
            (hint_w, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.putText(frame, hint, (max(8, width - hint_w - 14), height - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, theme.text_dim, 1, cv2.LINE_AA)

    def _draw_log(self, ctx: FrameContext, theme: Any) -> None:
        """Recent events, newest at the bottom, in the lower-left corner."""
        if not self._events:
            return
        rows = self._events[-6:]
        y = ctx.height - 38
        for event in reversed(rows):
            role = _EVENT_LEVEL.get(event.kind, "info")
            color = {
                "danger": theme.danger, "warn": theme.warn,
                "good": theme.good, "dim": theme.text_dim,
            }.get(role, theme.text)
            cv2.putText(ctx.frame, event.line(), (14, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            y -= 17
            if y < 90:
                break

    # -- HUD / status -------------------------------------------------------

    def status_text(self, ctx: FrameContext, state: AppState) -> Optional[str]:
        """
        The mode owns its status line.

        Telemetry is off in this mode, so this text only surfaces if the
        operator turns the HUD back on with ``T``.
        """
        subjects = ctx.extras.get("security_subjects", [])
        if self._armed and subjects:
            zone = self.zone
            if any(zone.contains(s.centroid) for s in subjects):
                self._last_status = f"INTRUSION — ZONA {zone.name}"
                return self._last_status
        if subjects:
            self._last_status = f"{len(subjects)} SUJETO(S) EN CAMPO"
        elif self._armed:
            self._last_status = "VIGILANDO"
        else:
            self._last_status = "EN ESPERA"
        return self._last_status

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        sensor = self.nightvision
        subjects = ctx.extras.get("security_subjects", [])
        lines = [
            f"Canal: {self.channel}  Zona: {self.zone.name}",
            f"Estado: {'ARMADO' if self._armed else 'desarmado'}"
            + (f" (en {max(0.0, self._arm_at - state.uptime):.0f}s)"
               if self._arm_at is not None else ""),
            f"Sensor: {sensor.effective_mode}  {sensor.luminance:.0f}lx"
            f"  G{sensor.applied_gain:.2f}x",
            f"Sujetos: {len(subjects)} en campo, "
            f"{self.tracker.total_identified} identificados, pico {self._peak_subjects}",
            f"Eventos: {len(self._events)}",
        ]
        for subject in subjects[:3]:
            lines.append(
                f"  {subject.label} {subject.code}  {subject.dwell(ctx.now):.0f}s"
                f"  conf {subject.confidence():.0%}"
            )
        if self._last_export:
            lines.append(f"Ultimo export: {os.path.basename(self._last_export)}")
        return lines


# ---------------------------------------------------------------------------
# Drawing primitives specific to the surveillance overlay
# ---------------------------------------------------------------------------

def _corner_reticle(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Targeting brackets around a subject, leaving the subject itself clear."""
    x0, y0, x1, y1 = box
    length = max(6, min(26, (x1 - x0) // 4, (y1 - y0) // 4))
    for (px, py), (dx, dy) in (
        ((x0, y0), (1, 1)), ((x1, y0), (-1, 1)),
        ((x0, y1), (1, -1)), ((x1, y1), (-1, -1)),
    ):
        cv2.line(frame, (px, py), (px + dx * length, py), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (px, py), (px, py + dy * length), color, thickness, cv2.LINE_AA)

    # Centre tick, so the operator can see what the tracker considers centre.
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    cv2.line(frame, (cx - 5, cy), (cx + 5, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - 5), (cx, cy + 5), color, 1, cv2.LINE_AA)


def _label_block(
    frame: np.ndarray,
    origin: Tuple[int, int],
    lines: Sequence[str],
    accent: Tuple[int, int, int],
    text: Tuple[int, int, int],
) -> None:
    """Identity card anchored to a subject's box, flipped to stay on screen."""
    if not lines:
        return
    height, width = frame.shape[:2]
    scale, pad, line_h = 0.4, 5, 15

    widest = max(
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        for line in lines
    )
    box_w = widest + pad * 2
    box_h = line_h * len(lines) + pad * 2

    x, y = origin
    # Prefer above the box; drop below when there is no room up top.
    top = y - box_h - 4
    if top < 0:
        top = min(y + 4, max(0, height - box_h))
    left = int(clamp(x, 0, max(0, width - box_w)))
    top = int(clamp(top, 0, max(0, height - box_h)))

    region = frame[top:top + box_h, left:left + box_w]
    if region.size:
        # Darken rather than fill, so the picture underneath stays visible.
        cv2.addWeighted(region, 0.25, np.zeros_like(region), 0.75, 0, region)
    cv2.line(frame, (left, top), (left, top + box_h), accent, 2, cv2.LINE_AA)

    for index, line in enumerate(lines):
        colour = accent if index == 0 else text
        cv2.putText(
            frame, line, (left + pad, top + pad + line_h * (index + 1) - 4),
            cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA,
        )


def _dashed_rect(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    dash: int = 12,
    thickness: int = 1,
) -> None:
    """Rectangle drawn as dashes, so it never reads as a detection box."""
    x0, y0, x1, y1 = box
    for x in range(x0, x1, dash * 2):
        end = min(x + dash, x1)
        cv2.line(frame, (x, y0), (end, y0), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y1), (end, y1), color, thickness, cv2.LINE_AA)
    for y in range(y0, y1, dash * 2):
        end = min(y + dash, y1)
        cv2.line(frame, (x0, y), (x0, end), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y), (x1, end), color, thickness, cv2.LINE_AA)


def _blinking_dot(
    frame: np.ndarray,
    center: Tuple[int, int],
    now: float,
    color: Tuple[int, int, int],
    radius: int = 6,
) -> None:
    """Recording-style indicator that blinks once a second."""
    if int(now * 2) % 2 == 0:
        cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)


def _right_text(
    frame: np.ndarray,
    text: str,
    anchor: Tuple[int, int],
    color: Tuple[int, int, int],
    scale: float,
    thickness: int,
) -> None:
    """Draw text right-aligned to *anchor*."""
    (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(4, anchor[0] - text_w)
    cv2.putText(frame, text, (x, anchor[1]), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _signal_bars(
    frame: np.ndarray,
    origin: Tuple[int, int],
    fps: float,
    target_fps: float,
    theme: Any,
    bars: int = 5,
) -> None:
    """Signal-strength bars driven by how close the frame rate is to target."""
    quality = clamp(fps / max(1.0, target_fps * 0.9), 0.0, 1.0)
    lit = int(round(quality * bars))
    x, y = origin
    for index in range(bars):
        bar_h = 4 + index * 3
        colour = (
            theme.good if index < lit and quality > 0.66
            else theme.warn if index < lit
            else theme.text_dim
        )
        filled = -1 if index < lit else 1
        cv2.rectangle(
            frame, (x + index * 9, y - bar_h), (x + index * 9 + 6, y),
            colour, filled,
        )
