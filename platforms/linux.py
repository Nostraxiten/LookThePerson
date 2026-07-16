"""
Linux platform bridge for LookThePerson.
Uses subprocess + xdotool/wmctrl for window management, V4L2 for camera.
"""

import os
import shutil
import subprocess

import cv2

from platforms import PlatformBridge


CALCULATOR_COMMANDS = [
    ["gnome-calculator"],
    ["kcalc"],
    ["galculator"],
    ["xcalc"],
]

CALCULATOR_TITLES = ("calculator", "calculadora", "gnome-calculator", "kcalc", "galculator", "xcalc")
CALCULATOR_WINDOW_WIDTH = 420
CALCULATOR_WINDOW_HEIGHT = 620


def _has_command(cmd):
    return shutil.which(cmd) is not None


def _run_silent(args, timeout=3):
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


def _xdotool_search(title_part):
    """Return list of window IDs matching title."""
    output = _run_silent(["xdotool", "search", "--name", title_part])
    if not output:
        return []
    return [wid.strip() for wid in output.splitlines() if wid.strip()]


def _get_screen_resolution():
    """Get primary screen resolution via xrandr."""
    output = _run_silent(["xrandr", "--current"])
    for line in output.splitlines():
        if " connected primary" in line or (" connected" in line and "primary" not in output):
            parts = line.split()
            for part in parts:
                if "x" in part and "+" in part:
                    res = part.split("+")[0]
                    w, h = res.split("x")
                    return int(w), int(h)
    return 1920, 1080


class LinuxBridge(PlatformBridge):

    def __init__(self):
        self._has_xdotool = _has_command("xdotool")
        self._has_wmctrl = _has_command("wmctrl")
        self._has_xrandr = _has_command("xrandr")
        if not self._has_xdotool:
            print(
                "platform/linux: xdotool no encontrado. "
                "Instala con: sudo apt install xdotool",
                flush=True,
            )

    # -- Monitor helpers ----------------------------------------------------

    def get_monitor_geometry(self):
        w, h = _get_screen_resolution()
        return 0, 0, w, h

    def get_upper_monitor_geometry(self):
        return self.get_monitor_geometry()

    # -- Camera -------------------------------------------------------------

    def get_camera_backend(self):
        return cv2.CAP_V4L2

    # -- Calculator ----------------------------------------------------------

    def open_calculator(self):
        for cmd in CALCULATOR_COMMANDS:
            if _has_command(cmd[0]):
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    print(f"platform/linux: calculadora abierta ({cmd[0]})", flush=True)
                    return proc
                except OSError:
                    continue
        print("platform/linux: no se encontro ninguna calculadora instalada", flush=True)
        return None

    def close_calculator(self, process_handle):
        closed = self.close_windows_by_title(CALCULATOR_TITLES)
        if not closed and process_handle and process_handle.poll() is None:
            process_handle.terminate()

    # -- Window management ---------------------------------------------------

    def find_window_by_title(self, title_parts):
        if not self._has_xdotool:
            return None
        for part in title_parts:
            if not part:
                continue
            wids = _xdotool_search(part.lower())
            if wids:
                return wids[0]
        return None

    def send_key_to_window(self, window_handle, key):
        if not self._has_xdotool or not window_handle:
            return False
        xdo_key = key
        if key == "+":
            xdo_key = "plus"
        elif key == "-":
            xdo_key = "minus"
        elif key == "*":
            xdo_key = "asterisk"
        elif key == "/":
            xdo_key = "slash"
        elif key == "=":
            xdo_key = "equal"
        try:
            subprocess.run(
                ["xdotool", "windowactivate", "--sync", str(window_handle)],
                timeout=2,
                capture_output=True,
            )
            subprocess.run(
                ["xdotool", "key", "--window", str(window_handle), xdo_key],
                timeout=2,
                capture_output=True,
            )
            return True
        except (subprocess.TimeoutExpired, OSError):
            return False

    def move_window(self, window_handle, x, y, width, height):
        if not self._has_xdotool or not window_handle:
            return
        try:
            subprocess.run(
                [
                    "xdotool", "windowsize", str(window_handle),
                    str(width), str(height),
                ],
                timeout=2,
                capture_output=True,
            )
            subprocess.run(
                [
                    "xdotool", "windowmove", str(window_handle),
                    str(x), str(y),
                ],
                timeout=2,
                capture_output=True,
            )
        except (subprocess.TimeoutExpired, OSError):
            pass

    def close_window(self, window_handle):
        if not self._has_xdotool or not window_handle:
            return
        try:
            subprocess.run(
                ["xdotool", "windowclose", str(window_handle)],
                timeout=2,
                capture_output=True,
            )
        except (subprocess.TimeoutExpired, OSError):
            pass

    def move_windows_by_title(self, title_parts, width=None, height=None):
        moved = False
        mx, my, mw, mh = self.get_upper_monitor_geometry()
        w = width or mw
        h = height or mh
        wx = mx + max(0, (mw - w) // 2)
        wy = my + max(0, (mh - h) // 2)

        for part in title_parts:
            if not part:
                continue
            for wid in _xdotool_search(part.lower()):
                self.move_window(wid, wx, wy, w, h)
                moved = True
        return moved

    def close_windows_by_title(self, title_parts):
        closed = False
        for part in title_parts:
            if not part:
                continue
            for wid in _xdotool_search(part.lower()):
                self.close_window(wid)
                closed = True
        return closed

    def move_windows_by_pid(self, pid, width=None, height=None):
        if not self._has_xdotool:
            return False
        output = _run_silent(["xdotool", "search", "--pid", str(pid)])
        if not output:
            return False
        mx, my, mw, mh = self.get_upper_monitor_geometry()
        w = width or CALCULATOR_WINDOW_WIDTH
        h = height or CALCULATOR_WINDOW_HEIGHT
        wx = mx + max(0, (mw - w) // 2)
        wy = my + max(0, (mh - h) // 2)
        moved = False
        for wid in output.splitlines():
            wid = wid.strip()
            if wid:
                self.move_window(wid, wx, wy, w, h)
                moved = True
        return moved
