"""
Body gesture detection for LookThePerson.
Detects: clap, arms open/closed, both hands raised, head touch, T-pose, squat.
"""

# ---------------------------------------------------------------------------
# Visibility helpers
# ---------------------------------------------------------------------------

GESTURE_VISIBILITY = 0.2
CLAP_DISTANCE = 0.11
ARMS_OPEN_MIN_SHOULDER_RATIO = 1.75
ARMS_OPEN_OUTSIDE_SHOULDER_MARGIN = 0.25
ARMS_OPEN_MAX_VERTICAL_DIFF = 0.38
ARMS_CLOSED_MAX_WRIST_DISTANCE = 0.24
ARMS_CLOSED_TORSO_MARGIN = 0.08
MIN_VISIBILITY = 0.35


def _visible(landmark, threshold=GESTURE_VISIBILITY):
    return getattr(landmark, "visibility", 1.0) >= threshold


def _visible_strict(landmark):
    return _visible(landmark, MIN_VISIBILITY)


# ---------------------------------------------------------------------------
# Head circle helper (shared by multiple gestures)
# ---------------------------------------------------------------------------

def head_circle(landmarks, width=1, height=1):
    """
    Compute bounding circle around head landmarks.
    Returns (px, py, pr, norm_x, norm_y, norm_radius) or None.
    """
    head_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    pts = []
    for i in head_ids:
        if _visible(landmarks[i]):
            pts.append((landmarks[i].x, landmarks[i].y))

    if len(pts) < 2:
        return None

    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    radius = max(
        ((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5
        for p in pts
    )

    if _visible(landmarks[11]) and _visible(landmarks[12]):
        sw = abs(landmarks[11].x - landmarks[12].x)
        radius = max(radius, sw * 0.28)

    radius = min(max(radius * 1.35, 0.045), 0.16)
    cy -= radius * 0.15

    return (
        int(cx * width),
        int(cy * height),
        int(radius * max(width, height)),
        cx, cy, radius,
    )


# ---------------------------------------------------------------------------
# Gesture detectors
# ---------------------------------------------------------------------------

def wrists_are_clapping(landmarks):
    """Detect if wrists are close enough for a clap."""
    lw, rw = landmarks[15], landmarks[16]
    if not _visible_strict(lw) or not _visible_strict(rw):
        return False
    dx = lw.x - rw.x
    dy = lw.y - rw.y
    return (dx * dx + dy * dy) ** 0.5 <= CLAP_DISTANCE


def arms_are_open(landmarks):
    """Detect T-pose / arms spread wide."""
    ls, rs = landmarks[11], landmarks[12]
    lw, rw = landmarks[15], landmarks[16]
    if not all(_visible(lm) for lm in (ls, rs, lw, rw)):
        return False

    wrist_min_x = min(lw.x, rw.x)
    wrist_max_x = max(lw.x, rw.x)
    shoulder_min_x = min(ls.x, rs.x)
    shoulder_max_x = max(ls.x, rs.x)
    shoulder_dist = max(shoulder_max_x - shoulder_min_x, 0.01)
    wrist_dist = wrist_max_x - wrist_min_x
    wrist_v_diff = abs(lw.y - rw.y)
    shoulder_cy = (ls.y + rs.y) / 2
    margin = shoulder_dist * ARMS_OPEN_OUTSIDE_SHOULDER_MARGIN

    wrists_outside = (
        wrist_min_x <= shoulder_min_x - margin
        and wrist_max_x >= shoulder_max_x + margin
    )
    wrists_near_shoulders = (
        abs(lw.y - shoulder_cy) <= ARMS_OPEN_MAX_VERTICAL_DIFF
        and abs(rw.y - shoulder_cy) <= ARMS_OPEN_MAX_VERTICAL_DIFF
    )

    return (
        wrists_outside
        and wrist_dist / shoulder_dist >= ARMS_OPEN_MIN_SHOULDER_RATIO
        and wrist_v_diff <= ARMS_OPEN_MAX_VERTICAL_DIFF
        and wrists_near_shoulders
    )


def arms_are_closed(landmarks):
    """Detect arms crossed / closed in front of torso."""
    ls, rs = landmarks[11], landmarks[12]
    lw, rw = landmarks[15], landmarks[16]
    if not all(_visible_strict(lm) for lm in (ls, rs, lw, rw)):
        return False

    dx = lw.x - rw.x
    dy = lw.y - rw.y
    dist = (dx * dx + dy * dy) ** 0.5

    s_min = min(ls.x, rs.x)
    s_max = max(ls.x, rs.x)
    sw = max(s_max - s_min, 0.01)
    inner_min = s_min + sw * ARMS_CLOSED_TORSO_MARGIN
    inner_max = s_max - sw * ARMS_CLOSED_TORSO_MARGIN

    wrists_inside = (
        inner_min <= lw.x <= inner_max
        and inner_min <= rw.x <= inner_max
    )

    return dist <= ARMS_CLOSED_MAX_WRIST_DISTANCE and wrists_inside


def both_hands_raised(landmarks):
    """Detect both hands raised above the head."""
    required = (landmarks[13], landmarks[14], landmarks[15], landmarks[16])
    if not all(_visible(lm) for lm in required):
        return False

    circle = head_circle(landmarks)
    if circle is None:
        return False

    _, _, _, _cx, cy, radius = circle
    lw, rw = landmarks[15], landmarks[16]
    le, re = landmarks[13], landmarks[14]

    return (
        lw.y < le.y
        and rw.y < re.y
        and lw.y <= cy - radius * 0.15
        and rw.y <= cy - radius * 0.15
    )


def hand_touches_top_of_head(landmarks):
    """Detect if either wrist is touching the top part of the head circle."""
    circle = head_circle(landmarks)
    if circle is None:
        return False

    _, _, _, cx, cy, radius = circle
    for wrist_id in (15, 16):
        wrist = landmarks[wrist_id]
        if not _visible(wrist):
            continue
        dx = wrist.x - cx
        dy = wrist.y - cy
        inside = dx * dx + dy * dy <= radius * radius
        in_top = wrist.y <= cy + radius * 0.15
        if inside and in_top:
            return True
    return False


def is_t_pose(landmarks):
    """
    Detect a strict T-pose: arms extended horizontally,
    wrists at shoulder height, elbows straight.
    """
    ls, rs = landmarks[11], landmarks[12]
    le, re = landmarks[13], landmarks[14]
    lw, rw = landmarks[15], landmarks[16]
    if not all(_visible(lm) for lm in (ls, rs, le, re, lw, rw)):
        return False

    shoulder_cy = (ls.y + rs.y) / 2

    # Wrists, elbows, and shoulders all near the same height
    max_diff = 0.12
    all_level = (
        abs(lw.y - shoulder_cy) < max_diff
        and abs(rw.y - shoulder_cy) < max_diff
        and abs(le.y - shoulder_cy) < max_diff
        and abs(re.y - shoulder_cy) < max_diff
    )

    # Arms extended wide
    return all_level and arms_are_open(landmarks)


def is_squatting(landmarks):
    """
    Detect a squat: hips (23, 24) are below or near knee level (25, 26).
    """
    lh, rh = landmarks[23], landmarks[24]
    lk, rk = landmarks[25], landmarks[26]
    if not all(_visible(lm) for lm in (lh, rh, lk, rk)):
        return False

    hip_y = (lh.y + rh.y) / 2
    knee_y = (lk.y + rk.y) / 2

    # In normalized coords, y increases downward
    return hip_y >= knee_y - 0.05


def detect_all_body_gestures(landmarks):
    """
    Run all body gesture detectors.
    Returns dict of gesture_name -> bool.
    """
    return {
        "clap": wrists_are_clapping(landmarks),
        "arms_open": arms_are_open(landmarks),
        "arms_closed": arms_are_closed(landmarks),
        "both_hands_raised": both_hands_raised(landmarks),
        "head_touch": hand_touches_top_of_head(landmarks),
        "t_pose": is_t_pose(landmarks),
        "squat": is_squatting(landmarks),
    }
