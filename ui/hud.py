"""
Heads-up display for LookThePerson.

The HUD is assembled from independent regions, each of which can be shown or
hidden on its own:

* top-left — status line, mode badge, uptime
* top-right — FPS, active models, recording indicator
* bottom-left — detection counts, gesture, mode-supplied lines
* right panel — help / key bindings
* centre — mode picker overlay
* bottom-right — toast notifications
* optional — FPS sparkline and profiler breakdown

The old ``draw_hud_panel`` entry point is kept as a thin wrapper so existing
code and scripts keep working.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.state import AppState, FrameContext
from core.theme import Theme
from ui.widgets import (
    ToastManager,
    draw_badge,
    draw_bar,
    draw_key_hint,
    draw_panel,
    draw_sparkline,
    draw_text,
    text_size,
    truncate,
)

__all__ = ["HUD", "draw_hud_panel", "draw_center_point"]


class HUD:
    """
    Renders every overlay region.

    Holds no application state of its own beyond the toast queue — everything
    it displays is read from :class:`~core.state.AppState` and the current
    frame context, so what you see is always what the app actually thinks.
    """

    def __init__(self, toasts: Optional[ToastManager] = None):
        self.toasts = toasts or ToastManager()
        self.show_picker = False
        self._picker_index = 0

    # -- Top-level ----------------------------------------------------------

    def draw(
        self,
        ctx: FrameContext,
        state: AppState,
        mode_lines: Optional[Sequence[str]] = None,
        key_bindings: Optional[Sequence[Tuple[str, str, Optional[bool]]]] = None,
        mode_keys: Optional[Sequence[Tuple[str, str]]] = None,
        overlay_owner: bool = False,
    ) -> None:
        """
        Draw the whole HUD for one frame.

        *overlay_owner* says the active mode paints its own complete display.
        The HUD then stays out of the corners entirely — toasts, the picker and
        the help panel still work, because those are things the operator asked
        for rather than chrome the mode did not.
        """
        theme = state.theme
        frame = ctx.frame

        if state.is_active("telemetry"):
            self._draw_status(frame, ctx, state, theme)
            self._draw_top_right(frame, ctx, state, theme)
            self._draw_bottom_left(frame, ctx, state, theme, mode_lines or [])
            self._draw_hint_bar(frame, theme)
        elif not overlay_owner:
            self._draw_minimal(frame, ctx, state, theme)

        if state.is_active("fps_graph"):
            self._draw_fps_graph(frame, state, theme)

        if state.is_active("debug"):
            self._draw_profiler(frame, state, theme)

        if state.is_active("help") and (key_bindings or mode_keys):
            self._draw_help(frame, state, theme, key_bindings or [], mode_keys or [])

        if self.show_picker:
            self._draw_picker(frame, state, theme)

        if state.config.display.show_toasts:
            self.toasts.draw(frame, theme, ctx.now)

    # -- Regions ------------------------------------------------------------

    def _draw_status(
        self, frame: np.ndarray, ctx: FrameContext, state: AppState, theme: Theme
    ) -> None:
        """Top-left: mode badge, status text and uptime."""
        badge_width = draw_badge(frame, state.mode_name.upper(), (12, 26), theme.accent)

        status = state.status_text
        if status:
            color = theme.warn if any(
                word in status for word in ("TOCANDO", "ALERTA", "DETECTADO")
            ) else theme.text
            draw_text(frame, status, (12 + badge_width + 10, 26), color, 0.62, 2)

        draw_text(frame, state.uptime_text(), (12, 48), theme.text_dim, 0.4, 1)

    def _draw_top_right(
        self, frame: np.ndarray, ctx: FrameContext, state: AppState, theme: Theme
    ) -> None:
        """Top-right: FPS, recording state and the models currently running."""
        width = frame.shape[1]
        fps = state.fps.fps

        # Colour the FPS by health so a slow machine is obvious at a glance.
        target = max(1.0, state.config.camera.fps * 0.8)
        fps_color = (
            theme.good if fps >= target
            else theme.warn if fps >= target * 0.6
            else theme.danger
        )
        fps_text = f"{fps:.0f} FPS"
        fps_w, _ = text_size(fps_text, 0.55, 2)
        draw_text(frame, fps_text, (width - fps_w - 12, 26), fps_color, 0.55, 2)

        y = 48
        if state.recording:
            # Blinking dot, as a recorder should have.
            if int(ctx.now * 2) % 2 == 0:
                cv2.circle(frame, (width - 20, y - 4), 6, theme.danger, -1, cv2.LINE_AA)
            draw_text(frame, "REC", (width - 62, y), theme.danger, 0.45, 1)
            y += 20

        for name in self._active_models(state):
            label = f"[{name}]"
            label_w, _ = text_size(label, 0.38, 1)
            draw_text(frame, label, (width - label_w - 12, y), theme.good, 0.38, 1)
            y += 16

    @staticmethod
    def _active_models(state: AppState) -> List[str]:
        """Names of the models currently switched on."""
        mapping = (
            ("segmentation", "SEG"), ("face_mesh", "MESH"),
            ("face_detect", "FACE"), ("object_detect", "OBJ"),
        )
        active = ["POSE"]
        active.extend(label for toggle, label in mapping if state.is_active(toggle))
        return active

    def _draw_bottom_left(
        self,
        frame: np.ndarray,
        ctx: FrameContext,
        state: AppState,
        theme: Theme,
        mode_lines: Sequence[str],
    ) -> None:
        """Bottom-left: counts, last gesture and the active mode's own lines."""
        height = frame.shape[0]
        y = height - 34

        # Mode-supplied lines sit above everything else, drawn bottom-up.
        for line in reversed(list(mode_lines)[:8]):
            if not line:
                continue
            draw_text(frame, truncate(line, 52), (12, y), theme.text, 0.44, 1)
            y -= 19

        gesture = state.last_gesture
        if gesture and state.gesture_age(ctx.now) < 3.0:
            draw_text(frame, f"Gesto: {gesture}", (12, y), theme.accent, 0.46, 1)
            y -= 20

        counts = []
        if ctx.person_count:
            counts.append(f"Personas {ctx.person_count}")
        if ctx.hand_count:
            counts.append(f"Manos {ctx.hand_count}")
        if ctx.face_landmarks:
            counts.append(f"Caras {len(ctx.face_landmarks)}")
        if counts:
            draw_text(frame, " | ".join(counts), (12, y), theme.text_dim, 0.42, 1)

    def _draw_hint_bar(self, frame: np.ndarray, theme: Theme) -> None:
        """Bottom-centre: the handful of keys worth always advertising."""
        height, width = frame.shape[:2]
        hint = "H=ayuda  TAB=modos  [ ]=cambiar  S=foto  R=grabar  Q=salir"
        hint_w, _ = text_size(hint, 0.38, 1)
        draw_text(frame, hint, ((width - hint_w) // 2, height - 12), theme.text_dim, 0.38, 1)

    def _draw_minimal(
        self, frame: np.ndarray, ctx: FrameContext, state: AppState, theme: Theme
    ) -> None:
        """The stripped-down HUD used when telemetry is off."""
        width = frame.shape[1]
        if state.status_text:
            draw_text(frame, state.status_text, (12, 28), theme.text, 0.6, 2)
        fps_text = f"{state.fps.fps:.0f} FPS"
        fps_w, _ = text_size(fps_text, 0.5, 1)
        draw_text(frame, fps_text, (width - fps_w - 12, 28), theme.text_dim, 0.5, 1)

    # -- Optional panels ----------------------------------------------------

    def _draw_fps_graph(self, frame: np.ndarray, state: AppState, theme: Theme) -> None:
        """Frame-rate history with the target as a reference line."""
        history = state.fps.fps_history()
        if len(history) < 2:
            return

        width = frame.shape[1]
        panel_w, panel_h = 190, 56
        x = width - panel_w - 12
        y = 150

        draw_panel(frame, (x, y), (panel_w, panel_h), theme, alpha=0.6, border=False)
        draw_sparkline(
            frame, history[-90:], (x + 8, y + 8), (panel_w - 16, panel_h - 24),
            theme.accent, baseline=state.config.camera.fps,
            baseline_color=tuple(int(c * 0.5) for c in theme.text_dim),
        )
        draw_text(
            frame,
            f"min {state.fps.one_percent_low:.0f}  jitter {state.fps.jitter_ms:.1f}ms",
            (x + 8, y + panel_h - 6), theme.text_dim, 0.34, 1,
        )

    def _draw_profiler(self, frame: np.ndarray, state: AppState, theme: Theme) -> None:
        """Per-stage timing breakdown, for the debug mode."""
        breakdown = state.profiler.breakdown()[:6]
        if not breakdown:
            return

        panel_w = 220
        panel_h = 40 + len(breakdown) * 18
        x, y = 12, 70
        origin_x, origin_y = draw_panel(
            frame, (x, y), (panel_w, panel_h), theme, alpha=0.7, title="RENDIMIENTO",
        )

        for name, ms, share in breakdown:
            draw_text(frame, truncate(name, 12), (origin_x, origin_y), theme.text_dim, 0.36, 1)
            draw_text(frame, f"{ms:5.1f}ms", (origin_x + 92, origin_y), theme.text, 0.36, 1)
            draw_bar(
                frame, (origin_x + 148, origin_y - 8), (52, 8),
                share / 100.0, theme.accent,
            )
            origin_y += 18

    def _draw_help(
        self,
        frame: np.ndarray,
        state: AppState,
        theme: Theme,
        bindings: Sequence[Tuple[str, str, Optional[bool]]],
        mode_keys: Sequence[Tuple[str, str]] = (),
    ) -> None:
        """
        Right-hand panel listing the key bindings.

        The active mode's own keys come first, under their own heading: they
        used to be visible only through ``--list-modes`` in a terminal, which is
        no use to someone already running the app — and a mode like ``security``
        keeps most of its controls there.
        """
        height, width = frame.shape[:2]
        panel_w = 330
        mode_rows = list(mode_keys)[:14]
        global_rows = list(bindings)[:22]

        header = 20 if mode_rows else 0
        wanted = 60 + header + (len(mode_rows) + len(global_rows)) * 20
        panel_h = min(height - 100, wanted)
        x = width - panel_w - 12
        y = 70

        origin_x, origin_y = draw_panel(
            frame, (x, y), (panel_w, panel_h), theme, alpha=0.78, title="CONTROLES",
        )
        limit = y + panel_h - 12

        if mode_rows:
            draw_text(frame, f"MODO — {state.mode_name}", (origin_x, origin_y),
                      theme.accent, 0.38, 1)
            origin_y += 20
            for key, description in mode_rows:
                if origin_y > limit:
                    return
                label = "SPACE" if key == "space" else key
                draw_key_hint(frame, label, description, (origin_x, origin_y), theme)
                origin_y += 20
            origin_y += 4

        for key, description, active in global_rows:
            if origin_y > limit:
                break
            draw_key_hint(frame, key, description, (origin_x, origin_y), theme, active)
            origin_y += 20

    def _draw_picker(self, frame: np.ndarray, state: AppState, theme: Theme) -> None:
        """Centre overlay listing modes by category."""
        height, width = frame.shape[:2]
        panel_w, panel_h = 460, min(height - 80, 420)
        x = (width - panel_w) // 2
        y = (height - panel_h) // 2

        origin_x, origin_y = draw_panel(
            frame, (x, y), (panel_w, panel_h), theme, alpha=0.86,
            title="MODOS — TAB cierra, [ ] cambia",
        )

        modes = state.get_note("mode_catalog", {})
        if not modes:
            draw_text(frame, "Sin modos registrados", (origin_x, origin_y), theme.text_dim, 0.45, 1)
            return

        column_x = origin_x
        row_y = origin_y
        for category, keys in modes.items():
            if row_y > y + panel_h - 30:
                column_x += 230
                row_y = origin_y
                if column_x > x + panel_w - 120:
                    break
            draw_text(frame, category.upper(), (column_x, row_y), theme.accent, 0.42, 1)
            row_y += 18
            for key in keys:
                current = key == state.mode_name
                color = theme.good if current else theme.text_dim
                marker = "> " if current else "  "
                draw_text(frame, f"{marker}{truncate(key, 20)}", (column_x, row_y), color, 0.4, 1)
                row_y += 16
            row_y += 6

    # -- Convenience --------------------------------------------------------

    def toggle_picker(self) -> bool:
        self.show_picker = not self.show_picker
        return self.show_picker

    def notify(self, message: str, level: str = "info") -> None:
        self.toasts.push(message, level)


# ---------------------------------------------------------------------------
# Standalone helpers (kept for compatibility with the original API)
# ---------------------------------------------------------------------------

def draw_center_point(
    frame: np.ndarray,
    center: Optional[Tuple[int, int]],
    color: Tuple[int, int, int] = (255, 0, 255),
) -> None:
    """Draw the body-centre marker with its coordinates."""
    if center is None:
        return
    cv2.circle(frame, center, 9, color, -1, cv2.LINE_AA)
    cv2.circle(frame, center, 13, (20, 20, 20), 2, cv2.LINE_AA)
    draw_text(frame, f"{center[0]}, {center[1]}",
              (center[0] + 14, center[1] - 14), color, 0.5, 2)


def draw_hud_panel(
    frame: np.ndarray,
    key_handler: Any,
    stats: Dict[str, Any],
    show_help: bool = True,
) -> None:
    """
    Draw a simple HUD from a plain stats dict.

    This is the pre-refactor entry point, kept working for scripts that drive
    the drawing code directly. New code should use :class:`HUD`, which reads
    from the application state instead of a hand-built dict.
    """
    from core.theme import get_theme

    theme = get_theme(stats.get("theme", "cyber"))
    height, width = frame.shape[:2]

    status = stats.get("status_text", "")
    if status:
        draw_text(frame, status, (12, 30), theme.text, 0.75, 2)

    fps_text = f"{stats.get('fps', 0):.0f} FPS"
    fps_w, _ = text_size(fps_text, 0.55, 1)
    draw_text(frame, fps_text, (max(12, width - fps_w - 12), 30), theme.text_dim, 0.55, 1)

    if stats.get("recording"):
        cv2.circle(frame, (width - fps_w - 40, 25), 8, theme.danger, -1)
        draw_text(frame, "REC", (width - fps_w - 80, 30), theme.danger, 0.5, 1)

    y = height - 14
    counts = [
        f"{label}: {stats[key]}"
        for key, label in (
            ("pose_count", "Poses"), ("hand_count", "Manos"),
            ("face_count", "Caras"), ("object_count", "Objetos"),
        )
        if stats.get(key)
    ]
    if counts:
        draw_text(frame, " | ".join(counts), (12, y - 22), theme.text_dim, 0.45, 1)

    gesture = stats.get("last_gesture")
    if gesture:
        draw_text(frame, f"Gesto: {gesture}", (12, y - 44), theme.accent, 0.45, 1)

    center = stats.get("body_center")
    if center:
        draw_center_point(frame, center)

    if show_help and key_handler is not None and hasattr(key_handler, "get_status_lines"):
        rows = [
            (label, description, active)
            for label, _name, active, description in key_handler.get_status_lines()
        ]
        panel_w = 320
        panel_h = min(height - 90, 60 + len(rows) * 20)
        x = width - panel_w - 12
        origin_x, origin_y = draw_panel(
            frame, (x, 70), (panel_w, panel_h), theme, title="CONTROLES",
        )
        for label, description, active in rows[:20]:
            draw_key_hint(frame, label, description, (origin_x, origin_y), theme, active)
            origin_y += 20
