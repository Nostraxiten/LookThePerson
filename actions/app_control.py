"""
Application control actions for LookThePerson.
Calculator and YouTube gesture-triggered actions using the platform bridge.
"""

import time
import webbrowser

CALCULATOR_TITLES = ("calculadora", "calculator", "gnome-calculator", "kcalc", "galculator")
CALCULATOR_WINDOW_WIDTH = 420
CALCULATOR_WINDOW_HEIGHT = 620

YOUTUBE_URL = "https://www.youtube.com"

# Timing constants
ARMS_OPEN_SECONDS = 0.45
ARMS_CLOSED_SECONDS = 0.45
YOUTUBE_OPEN_SECONDS = 0.45
YOUTUBE_OPEN_COOLDOWN_SECONDS = 2.0
CALCULATOR_INPUT_STABLE_SECONDS = 0.45
CALCULATOR_INPUT_COOLDOWN_SECONDS = 0.8
CALCULATOR_CLEAR_STABLE_SECONDS = 0.35
CALCULATOR_CLEAR_COOLDOWN_SECONDS = 1.0


class AppController:
    """Manages calculator and YouTube interactions via body/hand gestures."""

    def __init__(self, platform_bridge, disabled=False):
        self.bridge = platform_bridge
        self.disabled = disabled
        self.locked = disabled

        # Calculator state
        self.calculator_process = None
        self.calculator_is_open = False
        self.calculator_move_attempts = 0

        # Timing
        self.arms_open_since = None
        self.arms_closed_since = None
        self.both_hands_up_since = None
        self.last_youtube_time = 0.0

        # Hand gesture input
        self.current_hand_gesture = None
        self.hand_gesture_since = 0.0
        self.last_sent_gesture = None
        self.last_sent_gesture_time = 0.0
        self.both_hands_open_since = None
        self.last_clear_time = 0.0

    def toggle_lock(self):
        """Toggle calculator control lock. Returns new lock state."""
        if self.disabled:
            return True
        self.locked = not self.locked
        return self.locked

    def update_body_gestures(self, body_gestures, now=None):
        """
        Process body gestures dict and trigger app actions.
        Returns status text describing the current gesture state.
        """
        now = now or time.monotonic()
        status = ""

        if self.locked:
            return "CALC BLOQUEADA" if not self.disabled else "CALC DESACTIVADA"

        arms_open = body_gestures.get("arms_open", False)
        arms_closed = body_gestures.get("arms_closed", False)
        both_up = body_gestures.get("both_hands_raised", False)

        # Arms open -> open calculator
        if arms_open:
            self.arms_open_since = self.arms_open_since or now
            self.arms_closed_since = None
            status = "BRAZOS ABIERTOS..."
        elif arms_closed:
            self.arms_closed_since = self.arms_closed_since or now
            self.arms_open_since = None
            if self.calculator_is_open:
                status = "CERRANDO..."
        else:
            self.arms_open_since = None
            self.arms_closed_since = None

        # Both hands up -> YouTube
        if both_up:
            self.both_hands_up_since = self.both_hands_up_since or now
            status = "YOUTUBE..."
        else:
            self.both_hands_up_since = None

        if (
            self.both_hands_up_since
            and now - self.both_hands_up_since >= YOUTUBE_OPEN_SECONDS
            and now - self.last_youtube_time >= YOUTUBE_OPEN_COOLDOWN_SECONDS
        ):
            self._open_youtube()
            self.last_youtube_time = now
            status = "YOUTUBE ABIERTA"
            self.both_hands_up_since = None

        # Open calculator
        if (
            self.arms_open_since
            and not self.calculator_is_open
            and now - self.arms_open_since >= ARMS_OPEN_SECONDS
        ):
            self.calculator_process = self.bridge.open_calculator()
            if self.calculator_process is not None:
                self.calculator_is_open = True
                self.calculator_move_attempts = 20
                status = "CALCULADORA ABIERTA"

        # Close calculator
        if (
            self.arms_closed_since
            and self.calculator_is_open
            and now - self.arms_closed_since >= ARMS_CLOSED_SECONDS
        ):
            self.bridge.close_calculator(self.calculator_process)
            self.calculator_process = None
            self.calculator_is_open = False
            self.calculator_move_attempts = 0
            status = "CALCULADORA CERRADA"

        return status

    def update_hand_input(self, hand_info, now=None):
        """
        Process hand gesture info for calculator input.
        Returns status text or empty string.
        """
        now = now or time.monotonic()
        status = ""

        if self.locked:
            return status

        calc_input = hand_info.get("calculator_input")
        both_open = hand_info.get("both_hands_open", False)

        if both_open:
            self.both_hands_open_since = self.both_hands_open_since or now
        else:
            self.both_hands_open_since = None

        if calc_input != self.current_hand_gesture:
            self.current_hand_gesture = calc_input
            self.hand_gesture_since = now

        stable = (
            self.current_hand_gesture
            and now - self.hand_gesture_since >= CALCULATOR_INPUT_STABLE_SECONDS
        )

        # Check calculator availability
        calc_available = (
            self.calculator_is_open
            or bool(self.bridge.find_window_by_title(CALCULATOR_TITLES))
        )

        if self.calculator_is_open and not calc_available:
            self.calculator_is_open = False
            self.calculator_process = None
            self.calculator_move_attempts = 0

        # Clear
        clear_stable = (
            self.both_hands_open_since
            and now - self.both_hands_open_since >= CALCULATOR_CLEAR_STABLE_SECONDS
            and now - self.last_clear_time >= CALCULATOR_CLEAR_COOLDOWN_SECONDS
        )

        if calc_available and clear_stable:
            hwnd = self.bridge.find_window_by_title(CALCULATOR_TITLES)
            if hwnd:
                self.bridge.send_key_to_window(hwnd, "\x1b")  # Escape
                self.last_clear_time = now
                self.both_hands_open_since = None
                self.current_hand_gesture = None
                self.last_sent_gesture = None
                status = "CALC: BORRAR"

        # Send input
        can_send = (
            stable
            and calc_available
            and not clear_stable
            and (
                self.current_hand_gesture != self.last_sent_gesture
                or now - self.last_sent_gesture_time >= CALCULATOR_INPUT_COOLDOWN_SECONDS
            )
        )

        if can_send:
            hwnd = self.bridge.find_window_by_title(CALCULATOR_TITLES)
            if hwnd and self.bridge.send_key_to_window(hwnd, self.current_hand_gesture):
                self.last_sent_gesture = self.current_hand_gesture
                self.last_sent_gesture_time = now
                status = f"CALC: {self.current_hand_gesture}"

        # Move calculator window
        if self.calculator_move_attempts > 0:
            moved_pid = (
                self.calculator_process
                and self.bridge.move_windows_by_pid(
                    self.calculator_process.pid,
                    CALCULATOR_WINDOW_WIDTH,
                    CALCULATOR_WINDOW_HEIGHT,
                )
            )
            moved_title = self.bridge.move_windows_by_title(
                CALCULATOR_TITLES,
                CALCULATOR_WINDOW_WIDTH,
                CALCULATOR_WINDOW_HEIGHT,
            )
            if moved_pid or moved_title:
                self.calculator_move_attempts = 0
            else:
                self.calculator_move_attempts -= 1

        return status

    def _open_youtube(self):
        try:
            webbrowser.open_new_tab(YOUTUBE_URL)
            print("gesture: youtube abierta", flush=True)
        except Exception as exc:
            print(f"gesture: no pude abrir YouTube: {exc}", flush=True)
