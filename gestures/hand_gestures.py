"""
Hand gesture recognition for LookThePerson.

Classifies the 21-point hand skeleton into named gestures, and adds the
temporal layer that turns per-frame guesses into stable, debounced signals.

Recognition is geometric rather than learned: each gesture is a set of
conditions on which fingers are extended and where the tips sit relative to
each other. That keeps it dependency-free, explainable and easy to tune.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.filters import Cooldown, Debouncer
from core.geometry import angle_between, distance, vector

# Re-exported for backwards compatibility with the original module layout.
from models.hands import (  # noqa: F401
    count_extended_fingers,
    detect_hand_gesture,
    hand_calculator_gesture,
)

__all__ = [
    "HandLandmark",
    "GESTURE_LABELS",
    "detect_all_hand_gestures",
    "classify_hand",
    "extended_fingers",
    "is_pinching",
    "pinch_distance",
    "hand_openness",
    "hand_orientation",
    "hand_center",
    "HandGestureTracker",
]


class HandLandmark:
    """Named indices into the 21-point MediaPipe hand skeleton."""

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    COUNT = 21

    TIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
    PIPS = (THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP)


H = HandLandmark

GESTURE_LABELS: Dict[str, str] = {
    "fist": "Puño",
    "open_palm": "Palma abierta",
    "thumbs_up": "Pulgar arriba",
    "thumbs_down": "Pulgar abajo",
    "peace": "Paz / victoria",
    "rock": "Cuernos",
    "ok": "OK",
    "point": "Señalando",
    "call_me": "Llámame",
    "gun": "Pistola",
    "spock": "Spock",
    "pinch": "Pellizco",
    "three": "Tres",
    "four": "Cuatro",
}

PINCH_THRESHOLD = 0.055


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _hand_scale(hand: Sequence[Any]) -> float:
    """
    Reference length for normalising distances: wrist to middle knuckle.

    Using this instead of raw coordinates makes every threshold independent of
    how close the hand is to the camera.
    """
    return max(distance(hand[H.WRIST], hand[H.MIDDLE_MCP]), 1e-6)


def extended_fingers(hand: Sequence[Any]) -> List[bool]:
    """
    Which of the five fingers are extended, thumb first.

    The four long fingers are judged by tip-above-PIP; the thumb, which folds
    sideways rather than down, by how far its tip is from the wrist.
    """
    if len(hand) < H.COUNT:
        return [False] * 5

    scale = _hand_scale(hand)
    thumb = distance(hand[H.THUMB_TIP], hand[H.WRIST]) > (
        distance(hand[H.THUMB_IP], hand[H.WRIST]) + 0.28 * scale
    )

    states = [thumb]
    for tip, pip in ((H.INDEX_TIP, H.INDEX_PIP), (H.MIDDLE_TIP, H.MIDDLE_PIP),
                     (H.RING_TIP, H.RING_PIP), (H.PINKY_TIP, H.PINKY_PIP)):
        states.append(hand[tip].y < hand[pip].y - 0.05 * scale)
    return states


def pinch_distance(hand: Sequence[Any]) -> float:
    """Thumb-to-index distance, normalised by hand size."""
    if len(hand) < H.COUNT:
        return 1.0
    return distance(hand[H.THUMB_TIP], hand[H.INDEX_TIP]) / _hand_scale(hand)


def is_pinching(hand: Sequence[Any], threshold: float = 0.42) -> bool:
    """Whether thumb and index are touching — the universal 'click'."""
    return pinch_distance(hand) < threshold


def hand_openness(hand: Sequence[Any]) -> float:
    """
    How open the hand is, 0 (fist) to 1 (spread palm).

    Continuous rather than binary, which lets modes drive things like brush
    size or volume from it.
    """
    if len(hand) < H.COUNT:
        return 0.0
    scale = _hand_scale(hand)
    spread = sum(distance(hand[tip], hand[H.WRIST]) for tip in H.TIPS) / (5.0 * scale)
    # Empirically a closed fist sits near 1.1 and a spread palm near 2.1.
    return max(0.0, min(1.0, (spread - 1.1) / 1.0))


def hand_center(hand: Sequence[Any]) -> Tuple[float, float]:
    """Palm centre, averaged over the wrist and the four knuckles."""
    indices = (H.WRIST, H.INDEX_MCP, H.MIDDLE_MCP, H.RING_MCP, H.PINKY_MCP)
    return (
        sum(hand[i].x for i in indices) / len(indices),
        sum(hand[i].y for i in indices) / len(indices),
    )


def hand_orientation(hand: Sequence[Any]) -> str:
    """
    Which way the hand points: ``arriba``, ``abajo``, ``izquierda``, ``derecha``.
    """
    if len(hand) < H.COUNT:
        return "desconocida"
    dx, dy = vector(hand[H.WRIST], hand[H.MIDDLE_MCP])
    if abs(dy) > abs(dx):
        return "arriba" if dy < 0 else "abajo"
    return "derecha" if dx > 0 else "izquierda"


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_hand(hand: Sequence[Any]) -> Optional[str]:
    """
    Name the gesture a single hand is making.

    Checks run from most specific to least so that, for example, an "OK" sign
    is not reported as "three fingers".
    """
    if len(hand) < H.COUNT:
        return None

    thumb, index, middle, ring, pinky = extended_fingers(hand)
    count = sum((thumb, index, middle, ring, pinky))
    pinching = is_pinching(hand)

    # OK: thumb and index form a ring while the rest stay extended.
    if pinching and middle and ring:
        return "ok"

    # Pinch: only the thumb and index are involved.
    if pinching and not middle and not ring and not pinky:
        return "pinch"

    if count == 0:
        return "fist"

    if count == 5:
        # Spock: the middle and ring fingers separate into a V.
        scale = _hand_scale(hand)
        gap_middle = distance(hand[H.MIDDLE_TIP], hand[H.RING_TIP]) / scale
        gap_index = distance(hand[H.INDEX_TIP], hand[H.MIDDLE_TIP]) / scale
        if gap_middle > gap_index * 1.9:
            return "spock"
        return "open_palm"

    if count == 1:
        if thumb:
            # Up or down depends on where the tip sits relative to the wrist.
            return "thumbs_up" if hand[H.THUMB_TIP].y < hand[H.WRIST].y else "thumbs_down"
        if index:
            return "point"
        if pinky:
            return "call_me" if thumb else "pinky"

    if count == 2:
        if index and middle:
            return "peace"
        if index and pinky:
            return "rock"
        if thumb and index:
            return "gun"
        if thumb and pinky:
            return "call_me"

    if count == 3:
        if index and middle and ring:
            return "three"
        if thumb and index and pinky:
            return "rock"

    if count == 4 and not thumb:
        return "four"

    return f"{count}_fingers"


def detect_all_hand_gestures(hand_landmarks_list: Sequence[Any]) -> Dict[str, Any]:
    """
    Analyse every detected hand.

    Returns:
        dict with ``finger_counts``, ``gestures``, ``openness``,
        ``orientations``, ``pinching``, ``both_hands_open``,
        ``calculator_input`` and ``centers``.
    """
    if not hand_landmarks_list:
        return {
            "finger_counts": [], "gestures": [], "openness": [],
            "orientations": [], "pinching": [], "centers": [],
            "both_hands_open": False, "calculator_input": None,
        }

    finger_counts = [sum(extended_fingers(hand)) for hand in hand_landmarks_list]
    gestures = [classify_hand(hand) for hand in hand_landmarks_list]
    openness = [hand_openness(hand) for hand in hand_landmarks_list]
    orientations = [hand_orientation(hand) for hand in hand_landmarks_list]
    pinching = [is_pinching(hand) for hand in hand_landmarks_list]
    centers = [hand_center(hand) for hand in hand_landmarks_list]

    both_open = len(finger_counts) >= 2 and all(count == 5 for count in finger_counts[:2])

    calculator_input = None
    if not both_open:
        calculator_input = hand_calculator_gesture(hand_landmarks_list[0])

    return {
        "finger_counts": finger_counts,
        "gestures": gestures,
        "openness": openness,
        "orientations": orientations,
        "pinching": pinching,
        "centers": centers,
        "both_hands_open": both_open,
        "calculator_input": calculator_input,
    }


# ---------------------------------------------------------------------------
# Temporal stability
# ---------------------------------------------------------------------------

class HandGestureTracker:
    """
    Turns per-frame classifications into stable, debounced gesture events.

    A gesture must hold for *stable_seconds* before it fires, and cannot fire
    again within *cooldown_seconds* — without both, a single flickering frame
    would trigger whatever action is bound to it.
    """

    def __init__(self, stable_seconds: float = 0.3, cooldown_seconds: float = 0.9):
        self.stable_seconds = stable_seconds
        self.cooldown_seconds = cooldown_seconds
        self._debouncers: Dict[str, Debouncer] = {}
        self._cooldowns: Dict[str, Cooldown] = {}
        self._current: Optional[str] = None
        self._history: List[Tuple[float, str]] = []

    def update(self, hand_info: Dict[str, Any], now: float) -> List[str]:
        """
        Feed one frame of hand analysis; returns gestures that just fired.
        """
        gestures = [g for g in hand_info.get("gestures", []) if g]
        active = set(gestures)
        fired: List[str] = []

        # Every gesture seen so far gets a debouncer, driven each frame.
        for name in active | set(self._debouncers):
            debouncer = self._debouncers.get(name)
            if debouncer is None:
                debouncer = Debouncer(self.stable_seconds, 0.15)
                self._debouncers[name] = debouncer
                self._cooldowns[name] = Cooldown(self.cooldown_seconds)

            was_stable = debouncer.state
            is_stable = debouncer.update(name in active, now)
            if is_stable and not was_stable and self._cooldowns[name].trigger(now):
                fired.append(name)
                self._history.append((now, name))

        self._current = gestures[0] if gestures else None
        if len(self._history) > 100:
            self._history = self._history[-100:]
        return fired

    @property
    def current(self) -> Optional[str]:
        """The gesture on the primary hand right now, stable or not."""
        return self._current

    def is_stable(self, name: str) -> bool:
        debouncer = self._debouncers.get(name)
        return bool(debouncer and debouncer.state)

    def history(self, limit: int = 10) -> List[Tuple[float, str]]:
        return self._history[-limit:]

    def counts(self) -> Dict[str, int]:
        """How many times each gesture has fired."""
        totals: Dict[str, int] = {}
        for _time, name in self._history:
            totals[name] = totals.get(name, 0) + 1
        return totals

    def reset(self) -> None:
        self._debouncers.clear()
        self._cooldowns.clear()
        self._history.clear()
        self._current = None
