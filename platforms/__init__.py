"""
Platform abstraction layer for LookThePerson.
Detects OS and provides the correct platform bridge implementation.
"""

import abc
import platform


class PlatformBridge(abc.ABC):
    """Abstract base class for OS-specific operations."""

    @abc.abstractmethod
    def get_monitor_geometry(self):
        """Return (x, y, width, height) of the preferred monitor."""

    @abc.abstractmethod
    def get_upper_monitor_geometry(self):
        """Return (x, y, width, height) of the topmost monitor."""

    @abc.abstractmethod
    def get_camera_backend(self):
        """Return the OpenCV camera backend constant for this OS."""

    @abc.abstractmethod
    def open_calculator(self):
        """Open the system calculator. Returns a process handle or None."""

    @abc.abstractmethod
    def close_calculator(self, process_handle):
        """Close the calculator opened by open_calculator."""

    @abc.abstractmethod
    def find_window_by_title(self, title_parts):
        """Find a window whose title contains any of the given parts. Returns handle or None."""

    @abc.abstractmethod
    def send_key_to_window(self, window_handle, key):
        """Send a keypress to the specified window. Returns True on success."""

    @abc.abstractmethod
    def move_window(self, window_handle, x, y, width, height):
        """Move and resize a window."""

    @abc.abstractmethod
    def close_window(self, window_handle):
        """Close a specific window by handle."""

    @abc.abstractmethod
    def move_windows_by_title(self, title_parts, width=None, height=None):
        """Find windows matching title parts and move them. Returns True if any were moved."""

    @abc.abstractmethod
    def close_windows_by_title(self, title_parts):
        """Close all windows matching title parts. Returns True if any were closed."""

    @abc.abstractmethod
    def move_windows_by_pid(self, pid, width=None, height=None):
        """Move windows belonging to a process. Returns True if any were moved."""


def get_platform():
    """Factory: returns the correct PlatformBridge for the current OS."""
    system = platform.system().lower()

    if system == "windows":
        from platforms.windows import WindowsBridge
        return WindowsBridge()
    elif system == "linux":
        from platforms.linux import LinuxBridge
        return LinuxBridge()
    else:
        raise RuntimeError(
            f"Unsupported operating system: {platform.system()}. "
            f"LookThePerson supports Windows and Linux."
        )
