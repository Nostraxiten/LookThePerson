"""
Hand gesture detection for LookThePerson.
Re-exports from models.hands for convenience and adds composite detectors.
"""

from models.hands import (
    count_extended_fingers,
    detect_hand_gesture,
    hand_calculator_gesture,
)


def detect_all_hand_gestures(hand_landmarks_list):
    """
    Analyze all detected hands and return a summary dict.

    Args:
        hand_landmarks_list: list of hand landmarks from HandLandmarkerResult.

    Returns:
        dict with keys:
            finger_counts: list[int] — finger count per hand
            gestures: list[str] — detected gesture per hand
            both_hands_open: bool — both hands showing 5 fingers
            calculator_input: str or None — calculator gesture from first hand
    """
    if not hand_landmarks_list:
        return {
            "finger_counts": [],
            "gestures": [],
            "both_hands_open": False,
            "calculator_input": None,
        }

    finger_counts = [count_extended_fingers(h) for h in hand_landmarks_list]
    gestures = [detect_hand_gesture(h) for h in hand_landmarks_list]

    both_open = (
        len(finger_counts) >= 2
        and all(c == 5 for c in finger_counts[:2])
    )

    calc_input = None
    if not both_open and hand_landmarks_list:
        calc_input = hand_calculator_gesture(hand_landmarks_list[0])

    return {
        "finger_counts": finger_counts,
        "gestures": gestures,
        "both_hands_open": both_open,
        "calculator_input": calc_input,
    }
