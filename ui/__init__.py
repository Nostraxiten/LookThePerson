"""
User interface layer for LookThePerson.

* ``ui.widgets`` — reusable primitives (panels, bars, sparklines, toasts).
* ``ui.hud`` — the composed heads-up display.
* ``ui.renderer`` — frame fitting, guides and detection drawing.
"""

from ui.hud import HUD, draw_center_point, draw_hud_panel
from ui.renderer import (
    apply_night_mode,
    draw_bounding_boxes,
    draw_grid,
    draw_head_circle,
    fit_frame_to_screen,
    letterbox,
)
from ui.widgets import ToastManager, draw_bar, draw_panel, draw_sparkline, draw_text

__all__ = [
    "HUD",
    "draw_hud_panel",
    "draw_center_point",
    "fit_frame_to_screen",
    "letterbox",
    "draw_grid",
    "apply_night_mode",
    "draw_head_circle",
    "draw_bounding_boxes",
    "ToastManager",
    "draw_panel",
    "draw_text",
    "draw_bar",
    "draw_sparkline",
]
