"""
HUD (Heads-Up Display) overlay for LookThePerson.
Draws real-time status information directly onto the camera frame.
"""

import cv2


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

HUD_BG = (20, 20, 20)
HUD_TEXT = (220, 220, 220)
HUD_ACCENT = (0, 220, 255)
HUD_GREEN = (80, 255, 80)
HUD_RED = (80, 80, 255)
HUD_DIM = (120, 120, 120)
HUD_WARN = (0, 180, 255)
HUD_RECORDING = (0, 0, 255)


def draw_hud_panel(frame, key_handler, stats, show_help=True):
    """
    Draw a semi-transparent HUD panel on the frame.

    Args:
        frame: BGR numpy array to draw on
        key_handler: KeyHandler instance for action status
        stats: dict with optional keys:
            fps, status_text, pose_count, hand_count, face_count,
            object_count, recording, last_gesture, body_center,
            active_models, expressions
        show_help: whether to show the full help panel
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ---- Top-left: Status + FPS ----
    status = stats.get("status_text", "")
    fps = stats.get("fps", 0)

    if status:
        color = HUD_WARN if "TOCANDO" in status else HUD_TEXT
        cv2.putText(frame, status, (12, 30), font, 0.75, color, 2)

    fps_text = f"{fps:.0f} FPS"
    fps_size, _ = cv2.getTextSize(fps_text, font, 0.55, 1)
    cv2.putText(frame, fps_text, (max(12, w - fps_size[0] - 12), 30), font, 0.55, HUD_DIM, 1)

    # ---- Recording indicator ----
    if stats.get("recording"):
        rec_y = 30
        cv2.circle(frame, (w - fps_size[0] - 40, rec_y - 5), 8, HUD_RECORDING, -1)
        cv2.putText(frame, "REC", (w - fps_size[0] - 80, rec_y), font, 0.5, HUD_RECORDING, 1)

    # ---- Top-right: Active models ----
    active_models = stats.get("active_models", [])
    if active_models:
        model_y = 55
        for model_name in active_models:
            label = f"[{model_name}]"
            size, _ = cv2.getTextSize(label, font, 0.4, 1)
            cv2.putText(frame, label, (w - size[0] - 12, model_y), font, 0.4, HUD_GREEN, 1)
            model_y += 18

    # ---- Bottom-left: Detection counts ----
    y_bottom = h - 14
    counts = []
    for key, label in [("pose_count", "Poses"), ("hand_count", "Manos"),
                        ("face_count", "Caras"), ("object_count", "Objetos")]:
        val = stats.get(key)
        if val is not None and val > 0:
            counts.append(f"{label}: {val}")

    if counts:
        count_text = " | ".join(counts)
        cv2.putText(frame, count_text, (12, y_bottom - 22), font, 0.45, HUD_DIM, 1)

    # Body center
    center = stats.get("body_center")
    if center:
        cv2.putText(
            frame,
            f"Centro: {center[0]}, {center[1]}",
            (12, y_bottom - 44),
            font, 0.4, HUD_DIM, 1,
        )

    # Last gesture
    gesture = stats.get("last_gesture", "")
    if gesture:
        cv2.putText(frame, f"Gesto: {gesture}", (12, y_bottom - 66), font, 0.45, HUD_ACCENT, 1)

    # Expressions
    expressions = stats.get("expressions", {})
    active_exprs = [name for name, active in expressions.items() if active]
    if active_exprs:
        expr_text = " | ".join(active_exprs[:4])
        cv2.putText(frame, f"Expr: {expr_text}", (12, y_bottom - 88), font, 0.4, HUD_ACCENT, 1)

    # ---- Bottom center: Quick help ----
    help_text = "Q/Esc=Salir | H=Ayuda | S=Screenshot | R=Grabar"
    help_size, _ = cv2.getTextSize(help_text, font, 0.4, 1)
    cv2.putText(frame, help_text, ((w - help_size[0]) // 2, y_bottom), font, 0.4, HUD_DIM, 1)

    # ---- Help panel (full key list) ----
    if show_help:
        _draw_help_panel(frame, key_handler)


def _draw_help_panel(frame, key_handler):
    """Draw a semi-transparent help panel with all key bindings."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    actions = key_handler.get_status_lines()
    if not actions:
        return

    panel_w = 320
    line_h = 22
    padding = 12
    panel_h = padding * 2 + len(actions) * line_h + 30

    # Position: right side
    px = w - panel_w - 12
    py = 70

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), HUD_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), HUD_ACCENT, 1)

    # Title
    cv2.putText(frame, "CONTROLES", (px + padding, py + 22), font, 0.55, HUD_ACCENT, 1)

    # Key list
    y = py + 44
    for key_label, name, active, desc in actions:
        # Key badge
        cv2.putText(frame, f"[{key_label}]", (px + padding, y), font, 0.4, HUD_ACCENT, 1)

        # Status indicator
        if active is not None:
            color = HUD_GREEN if active else HUD_RED
            indicator = "ON " if active else "OFF"
            cv2.putText(frame, indicator, (px + padding + 40, y), font, 0.35, color, 1)

        # Description
        cv2.putText(frame, desc[:28], (px + padding + 78, y), font, 0.35, HUD_TEXT, 1)
        y += line_h


def draw_center_point(frame, center):
    """Draw the body center point marker."""
    if center is None:
        return
    cv2.circle(frame, center, 9, (255, 0, 255), -1)
    cv2.circle(frame, center, 13, (20, 20, 20), 2)
    cv2.putText(
        frame,
        f"{center[0]}, {center[1]}",
        (center[0] + 14, center[1] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 255),
        2,
    )
