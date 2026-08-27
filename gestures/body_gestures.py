"""
Body gesture detection for LookThePerson.
Detects: clap, arms open/closed, both hands raised, head touch, T-pose, squat.
"""

from core.geometry import is_finite_point

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
    if landmarks is None or len(landmarks) <= max(head_ids):
        return None

    pts = []
    for i in head_ids:
        landmark = landmarks[i]
        # A NaN landmark would poison the centroid and every value derived
        # from it, ending in `int(nan)` at the bottom of this function.
        if _visible(landmark) and is_finite_point(landmark):
            pts.append((landmark.x, landmark.y))

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


# ---------------------------------------------------------------------------
# Extended gesture set
# ---------------------------------------------------------------------------

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
NOSE = 0


def _torso_height(landmarks):
    """Shoulder-to-hip distance, the scale for body-relative thresholds."""
    shoulder_y = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2
    hip_y = (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2
    return max(abs(hip_y - shoulder_y), 1e-6)


def one_hand_raised(landmarks):
    """
    Detect a single raised hand.

    Returns ``"left"``, ``"right"`` or ``None``. Deliberately excludes the
    both-hands case, which has its own detector and its own meaning.
    """
    if not all(_visible(landmarks[i]) for i in (LEFT_WRIST, RIGHT_WRIST, NOSE)):
        return None
    head_y = landmarks[NOSE].y
    left_up = landmarks[LEFT_WRIST].y < head_y
    right_up = landmarks[RIGHT_WRIST].y < head_y
    if left_up and not right_up:
        return "left"
    if right_up and not left_up:
        return "right"
    return None


def is_leaning(landmarks, threshold=0.18):
    """
    Detect a sideways lean.

    Returns ``"left"``, ``"right"`` or ``None``, based on how far the shoulder
    midpoint has moved horizontally away from the hip midpoint.
    """
    required = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
    if not all(_visible(landmarks[i]) for i in required):
        return None

    shoulder_x = (landmarks[LEFT_SHOULDER].x + landmarks[RIGHT_SHOULDER].x) / 2
    hip_x = (landmarks[LEFT_HIP].x + landmarks[RIGHT_HIP].x) / 2
    offset = (shoulder_x - hip_x) / _torso_height(landmarks)

    if offset > threshold:
        return "right"
    if offset < -threshold:
        return "left"
    return None


def arms_crossed(landmarks):
    """
    Detect arms folded across the chest.

    Distinguished from :func:`arms_are_closed` by requiring the wrists to
    actually swap sides, which is what makes a fold a fold.
    """
    required = (LEFT_WRIST, RIGHT_WRIST, LEFT_SHOULDER, RIGHT_SHOULDER)
    if not all(_visible_strict(landmarks[i]) for i in required):
        return False

    left_wrist, right_wrist = landmarks[LEFT_WRIST], landmarks[RIGHT_WRIST]
    # In a mirrored image the left wrist normally sits at a lower x than the
    # right; crossing inverts that.
    crossed = left_wrist.x > right_wrist.x
    chest_y = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2
    torso = _torso_height(landmarks)
    at_chest = (
        abs(left_wrist.y - chest_y) < torso * 0.9
        and abs(right_wrist.y - chest_y) < torso * 0.9
    )
    return crossed and at_chest


def hands_on_hips(landmarks):
    """
    Detect hands resting on the hips, elbows flared out.

    Wrist position alone is not enough — arms hanging at the sides also put
    the wrists near hip height. The distinguishing feature is that the elbows
    swing *outside* the wrists, which only happens when the hands come in to
    the waist.
    """
    required = (LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP, LEFT_ELBOW, RIGHT_ELBOW)
    if not all(_visible(landmarks[i]) for i in required):
        return False

    torso = _torso_height(landmarks)
    tolerance = torso * 0.35

    def side_ok(wrist_id, hip_id, elbow_id, outward):
        wrist, hip, elbow = landmarks[wrist_id], landmarks[hip_id], landmarks[elbow_id]
        near_hip = (
            abs(wrist.y - hip.y) < tolerance
            and abs(wrist.x - hip.x) < tolerance
        )
        # ``outward`` is -1 for the viewer-left side, +1 for the right.
        elbow_flared = (elbow.x - wrist.x) * outward > torso * 0.12
        return near_hip and elbow_flared

    return (
        side_ok(LEFT_WRIST, LEFT_HIP, LEFT_ELBOW, -1)
        and side_ok(RIGHT_WRIST, RIGHT_HIP, RIGHT_ELBOW, 1)
    )


def is_pointing(landmarks, threshold=0.55):
    """
    Detect an arm extended out to the side, pointing.

    Returns ``"left"``, ``"right"`` or ``None``. The threshold is a fraction
    of torso height, so it holds at any distance from the camera.
    """
    torso = _torso_height(landmarks)
    # ``outward`` is the sign the wrist must move in to be reaching away from
    # the body; an arm crossing inward is not pointing, however far it travels.
    for side, shoulder, wrist, outward in (
        ("left", LEFT_SHOULDER, LEFT_WRIST, -1),
        ("right", RIGHT_SHOULDER, RIGHT_WRIST, 1),
    ):
        if not (_visible(landmarks[shoulder]) and _visible(landmarks[wrist])):
            continue
        reach = (landmarks[wrist].x - landmarks[shoulder].x) * outward / torso
        level = abs(landmarks[wrist].y - landmarks[shoulder].y) / torso < 0.6
        if reach > threshold and level:
            return side
    return None


def is_jumping(landmarks, baseline_hip_y, threshold=0.06):
    """
    Detect a jump by comparing hip height against a running baseline.

    The caller owns the baseline (typically a slow-moving average of hip
    height), which is what makes this robust to the person simply standing
    somewhere else in frame.
    """
    if not all(_visible(landmarks[i]) for i in (LEFT_HIP, RIGHT_HIP)):
        return False
    hip_y = (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2
    return (baseline_hip_y - hip_y) > threshold


def is_sitting(landmarks):
    """
    Detect a seated posture.

    Sitting compresses the hip-to-knee distance relative to the torso and
    brings the knees up toward hip height.
    """
    required = (LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE)
    if not all(_visible(landmarks[i]) for i in required):
        return False

    hip_y = (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2
    knee_y = (landmarks[LEFT_KNEE].y + landmarks[RIGHT_KNEE].y) / 2
    # Standing, the thigh projects to roughly a torso length; sitting collapses
    # it toward zero, so half a torso is a safe dividing line.
    return (knee_y - hip_y) < _torso_height(landmarks) * 0.5


def is_lying_down(landmarks):
    """Detect a horizontal body: the torso is wider than it is tall."""
    required = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
    if not all(_visible(landmarks[i]) for i in required):
        return False

    shoulder_x = (landmarks[LEFT_SHOULDER].x + landmarks[RIGHT_SHOULDER].x) / 2
    shoulder_y = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2
    hip_x = (landmarks[LEFT_HIP].x + landmarks[RIGHT_HIP].x) / 2
    hip_y = (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2
    return abs(shoulder_x - hip_x) > abs(shoulder_y - hip_y) * 1.4


def is_kicking(landmarks, threshold=0.55):
    """
    Detect a raised leg.

    Returns ``"left"``, ``"right"`` or ``None``.
    """
    torso = _torso_height(landmarks)
    for side, hip, ankle in (
        ("left", LEFT_HIP, LEFT_ANKLE), ("right", RIGHT_HIP, RIGHT_ANKLE),
    ):
        if not (_visible(landmarks[hip]) and _visible(landmarks[ankle])):
            continue
        if (landmarks[hip].y - landmarks[ankle].y) > -torso * threshold:
            return side
    return None


def hand_to_face(landmarks):
    """Detect a hand near the face — covering the mouth, scratching, thinking."""
    circle = head_circle(landmarks)
    if circle is None:
        return False
    _px, _py, _pr, cx, cy, radius = circle
    for wrist_id in (LEFT_WRIST, RIGHT_WRIST):
        wrist = landmarks[wrist_id]
        if not _visible(wrist):
            continue
        if ((wrist.x - cx) ** 2 + (wrist.y - cy) ** 2) <= (radius * 1.5) ** 2:
            return True
    return False


def detect_extended_body_gestures(landmarks, baseline_hip_y=None):
    """
    Run the extended detectors.

    Kept separate from :func:`detect_all_body_gestures` so the original
    gesture set — and anything bound to it — behaves exactly as before.
    """
    return {
        "one_hand_raised": one_hand_raised(landmarks),
        "leaning": is_leaning(landmarks),
        "arms_crossed": arms_crossed(landmarks),
        "hands_on_hips": hands_on_hips(landmarks),
        "pointing": is_pointing(landmarks),
        "jumping": (
            is_jumping(landmarks, baseline_hip_y) if baseline_hip_y is not None else False
        ),
        "sitting": is_sitting(landmarks),
        "lying_down": is_lying_down(landmarks),
        "kicking": is_kicking(landmarks),
        "hand_to_face": hand_to_face(landmarks),
    }


#: Every detector below indexes the full BlazePose topology by position.
POSE_LANDMARK_COUNT = 33

#: Gestures that report a side (``"left"`` / ``"right"``) when active, so their
#: inactive value is ``None`` rather than ``False``.
SIDED_GESTURES = ("one_hand_raised", "leaning", "pointing")

#: Every name :func:`detect_every_gesture` can return, in detector order.
#:
#: The two detector dicts below are built from this tuple rather than beside
#: it. A hand-maintained copy drifted once already — it grew four gestures that
#: no detector produced and lost three that did, so a caller reading
#: ``gestures["kicking"]`` hit a ``KeyError`` on exactly the malformed frame the
#: fallback existed to survive.
GESTURE_NAMES = (
    # detect_all_body_gestures
    "clap", "arms_open", "arms_closed", "both_hands_raised",
    "head_touch", "t_pose", "squat",
    # detect_extended_body_gestures
    "one_hand_raised", "leaning", "arms_crossed", "hands_on_hips",
    "pointing", "jumping", "sitting", "lying_down", "kicking", "hand_to_face",
)


def _gesture_defaults():
    """The full gesture dict with everything inactive."""
    return {
        name: (None if name in SIDED_GESTURES else False)
        for name in GESTURE_NAMES
    }


def detect_every_gesture(landmarks, baseline_hip_y=None):
    """
    Both gesture sets merged into a single dict.

    A landmark list shorter than the full BlazePose topology short-circuits to
    an all-inactive result: every detector below indexes fixed positions, so a
    truncated list would raise ``IndexError`` deep inside one of them and take
    the frame loop down.
    """
    if landmarks is None or len(landmarks) < POSE_LANDMARK_COUNT:
        return _gesture_defaults()

    combined = detect_all_body_gestures(landmarks)
    combined.update(detect_extended_body_gestures(landmarks, baseline_hip_y))
    return combined
