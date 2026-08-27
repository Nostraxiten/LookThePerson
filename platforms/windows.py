"""
Windows platform bridge for LookThePerson.
Uses ctypes.windll for native window management, calc.exe, and CAP_DSHOW.
"""

import ctypes
import subprocess
import unicodedata
from ctypes import wintypes

import cv2

from platforms import PlatformBridge


# ---------------------------------------------------------------------------
# Win32 helpers
# ---------------------------------------------------------------------------

class _Rect(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class _MonitorInfo(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_ulong),
        ("rcMonitor", _Rect),
        ("rcWork", _Rect),
        ("dwFlags", ctypes.c_ulong),
    ]


_ENUM_PROC = ctypes.WINFUNCTYPE(
    wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC,
    ctypes.POINTER(_Rect), wintypes.LPARAM,
)

_ENUM_WINDOWS_PROC = ctypes.WINFUNCTYPE(
    wintypes.BOOL, wintypes.HWND, wintypes.LPARAM,
)

_WM_CLOSE = 0x0010
_KEYEVENTF_KEYUP = 0x0002

# Mouse event flags for user32.mouse_event.
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
_MOUSEEVENTF_RIGHTDOWN = 0x0008
_MOUSEEVENTF_RIGHTUP = 0x0010
_MOUSEEVENTF_MIDDLEDOWN = 0x0020
_MOUSEEVENTF_MIDDLEUP = 0x0040
_MOUSEEVENTF_WHEEL = 0x0800

# Virtual key codes for the media keys.
_MEDIA_KEYS = {
    "volume_down": 0xAE,
    "volume_up": 0xAF,
    "mute": 0xAD,
    "next_track": 0xB0,
    "prev_track": 0xB1,
    "play_pause": 0xB3,
}

CALCULATOR_TITLES = ("calculadora", "calculator")
CALCULATOR_WINDOW_WIDTH = 420
CALCULATOR_WINDOW_HEIGHT = 620


def _normalize_text(text):
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return " ".join(text.split())


def _collect_monitors():
    monitors = []

    def callback(monitor, _dc, _rect, _data):
        info = _MonitorInfo()
        info.cbSize = ctypes.sizeof(_MonitorInfo)
        ctypes.windll.user32.GetMonitorInfoW(monitor, ctypes.byref(info))
        r = info.rcMonitor
        monitors.append((r.left, r.top, r.right - r.left, r.bottom - r.top))
        return True

    try:
        proc = _ENUM_PROC(callback)
        ctypes.windll.user32.EnumDisplayMonitors(0, 0, proc, 0)
    except Exception:
        monitors.clear()

    return monitors


# ---------------------------------------------------------------------------
# WindowsBridge
# ---------------------------------------------------------------------------

class WindowsBridge(PlatformBridge):

    # -- Monitor helpers ----------------------------------------------------

    def get_monitor_geometry(self):
        monitors = _collect_monitors()
        if not monitors:
            return 0, 0, 1280, 720
        monitors.sort(key=lambda m: (m[0], m[1]))
        return monitors[1] if len(monitors) > 1 else monitors[0]

    def get_upper_monitor_geometry(self):
        monitors = _collect_monitors()
        if not monitors:
            return 0, 0, 1280, 720
        monitors.sort(key=lambda m: (m[1], m[0]))
        return monitors[0]

    # -- Camera -------------------------------------------------------------

    def get_camera_backend(self):
        return cv2.CAP_DSHOW

    # -- Calculator ----------------------------------------------------------

    def open_calculator(self):
        try:
            return subprocess.Popen(["calc.exe"])
        except OSError as exc:
            print(f"platform/win: no pude abrir la calculadora: {exc}", flush=True)
            return None

    def close_calculator(self, process_handle):
        closed = self.close_windows_by_title(CALCULATOR_TITLES)
        if not closed and process_handle and process_handle.poll() is None:
            process_handle.terminate()

    # -- Window management ---------------------------------------------------

    def find_window_by_title(self, title_parts):
        title_parts = [_normalize_text(p) for p in title_parts if p]
        found = [None]

        def callback(hwnd, _data):
            if not ctypes.windll.user32.IsWindowVisible(hwnd):
                return True
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
            title = _normalize_text(buf.value)
            if title and any(p in title for p in title_parts):
                found[0] = hwnd
                return False
            return True

        ctypes.windll.user32.EnumWindows(_ENUM_WINDOWS_PROC(callback), 0)
        return found[0]

    def send_key_to_window(self, window_handle, key):
        if not window_handle:
            return False
        vk = 0x6B if key == "+" else ord(key)
        ctypes.windll.user32.SetForegroundWindow(window_handle)
        ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
        ctypes.windll.user32.keybd_event(vk, 0, _KEYEVENTF_KEYUP, 0)
        return True

    def move_window(self, window_handle, x, y, width, height):
        ctypes.windll.user32.ShowWindow(window_handle, 1)
        ctypes.windll.user32.MoveWindow(window_handle, x, y, width, height, True)
        ctypes.windll.user32.ShowWindow(window_handle, 1)

    def close_window(self, window_handle):
        if window_handle:
            ctypes.windll.user32.PostMessageW(window_handle, _WM_CLOSE, 0, 0)

    def move_windows_by_title(self, title_parts, width=None, height=None):
        title_parts = [_normalize_text(p) for p in title_parts if p]
        mx, my, mw, mh = self.get_upper_monitor_geometry()
        w = width or mw
        h = height or mh
        wx = mx + max(0, (mw - w) // 2)
        wy = my + max(0, (mh - h) // 2)
        moved = [False]

        def callback(hwnd, _data):
            if not ctypes.windll.user32.IsWindowVisible(hwnd):
                return True
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
            title = _normalize_text(buf.value)
            if title and any(p in title for p in title_parts):
                self.move_window(hwnd, wx, wy, w, h)
                moved[0] = True
            return True

        ctypes.windll.user32.EnumWindows(_ENUM_WINDOWS_PROC(callback), 0)
        return moved[0]

    def close_windows_by_title(self, title_parts):
        title_parts = [_normalize_text(p) for p in title_parts if p]
        closed = [False]

        def callback(hwnd, _data):
            if not ctypes.windll.user32.IsWindowVisible(hwnd):
                return True
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
            title = _normalize_text(buf.value)
            if title and any(p in title for p in title_parts):
                ctypes.windll.user32.PostMessageW(hwnd, _WM_CLOSE, 0, 0)
                closed[0] = True
            return True

        ctypes.windll.user32.EnumWindows(_ENUM_WINDOWS_PROC(callback), 0)
        return closed[0]

    def move_windows_by_pid(self, pid, width=None, height=None):
        mx, my, mw, mh = self.get_upper_monitor_geometry()
        w = width or CALCULATOR_WINDOW_WIDTH
        h = height or CALCULATOR_WINDOW_HEIGHT
        wx = mx + max(0, (mw - w) // 2)
        wy = my + max(0, (mh - h) // 2)
        moved = [False]

        def callback(hwnd, _data):
            if not ctypes.windll.user32.IsWindowVisible(hwnd):
                return True
            window_pid = wintypes.DWORD()
            ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(window_pid))
            if window_pid.value == pid:
                self.move_window(hwnd, wx, wy, w, h)
                moved[0] = True
            return True

        ctypes.windll.user32.EnumWindows(_ENUM_WINDOWS_PROC(callback), 0)
        return moved[0]


    # -- Optional capabilities ----------------------------------------------

    def capabilities(self):
        """user32 provides the mouse and media keys natively."""
        return ("mouse", "media_keys")

    def move_mouse(self, x, y):
        try:
            ctypes.windll.user32.SetCursorPos(int(x), int(y))
            return True
        except Exception:
            return False

    def click_mouse(self, button="left"):
        events = {
            "left": (_MOUSEEVENTF_LEFTDOWN, _MOUSEEVENTF_LEFTUP),
            "right": (_MOUSEEVENTF_RIGHTDOWN, _MOUSEEVENTF_RIGHTUP),
            "middle": (_MOUSEEVENTF_MIDDLEDOWN, _MOUSEEVENTF_MIDDLEUP),
        }.get(button)
        if not events:
            return False
        try:
            down, up = events
            ctypes.windll.user32.mouse_event(down, 0, 0, 0, 0)
            ctypes.windll.user32.mouse_event(up, 0, 0, 0, 0)
            return True
        except Exception:
            return False

    def scroll_mouse(self, amount):
        try:
            # One notch is 120 units; positive scrolls up.
            ctypes.windll.user32.mouse_event(
                _MOUSEEVENTF_WHEEL, 0, 0, int(amount) * 120, 0,
            )
            return True
        except Exception:
            return False

    def send_media_key(self, key):
        vk = _MEDIA_KEYS.get(key)
        if vk is None:
            return False
        try:
            ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
            ctypes.windll.user32.keybd_event(vk, 0, _KEYEVENTF_KEYUP, 0)
            return True
        except Exception:
            return False
