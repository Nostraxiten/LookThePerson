"""
Face gesture detection for LookThePerson.
Wraps face_mesh expression detection with temporal stability.
"""

import time

from models.face_mesh import detect_expressions, estimate_gaze_direction


class FaceGestureDetector:
    """
    Adds temporal stability to face expression detection.
    An expression must be sustained for a minimum duration before firing.
    """

    STABLE_SECONDS = 0.3  # how long expression must hold to count
    COOLDOWN_SECONDS = 1.0  # minimum time between repeated triggers

    def __init__(self):
        self._state = {}  # expression_name -> first_seen_time
        self._last_trigger = {}  # expression_name -> last trigger time
        self._last_expressions = {}

    def update(self, face_landmarks):
        """
        Analyze face landmarks and return stable expressions.

        Returns:
            dict with:
                raw: dict of all detected expressions (bool)
                stable: dict of expressions that have been stable long enough
                gaze: (horizontal, vertical) tuple
                triggered: list of expression names that just triggered
        """
        now = time.monotonic()
        raw = detect_expressions(face_landmarks)
        gaze = estimate_gaze_direction(face_landmarks)
        triggered = []

        for expr_name, is_active in raw.items():
            if is_active:
                if expr_name not in self._state:
                    self._state[expr_name] = now
                elif now - self._state[expr_name] >= self.STABLE_SECONDS:
                    last = self._last_trigger.get(expr_name, 0.0)
                    if now - last >= self.COOLDOWN_SECONDS:
                        triggered.append(expr_name)
                        self._last_trigger[expr_name] = now
            else:
                self._state.pop(expr_name, None)

        stable = {}
        for expr_name, first_seen in self._state.items():
            stable[expr_name] = now - first_seen >= self.STABLE_SECONDS

        self._last_expressions = raw

        return {
            "raw": raw,
            "stable": stable,
            "gaze": gaze,
            "triggered": triggered,
        }

    def reset(self):
        self._state.clear()
        self._last_trigger.clear()
        self._last_expressions.clear()
