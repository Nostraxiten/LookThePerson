"""
Camera and video input for LookThePerson.

:class:`CameraSource` wraps ``cv2.VideoCapture`` with the things a long-running
app actually needs:

* automatic reconnection when a USB camera drops out
* video files and image folders as drop-in replacements for a live camera
* device enumeration and resolution probing
* frame-rate limiting and a dropped-frame counter

The rest of the app never touches ``VideoCapture`` directly, so adding a new
input type means changing only this file.
"""

from __future__ import annotations

import glob
import os
import time
from typing import Any, Iterator, List, Optional, Tuple

import cv2
import numpy as np

__all__ = ["CameraSource", "list_cameras", "probe_resolutions", "COMMON_RESOLUTIONS"]

COMMON_RESOLUTIONS: Tuple[Tuple[int, int], ...] = (
    (640, 360), (640, 480), (800, 600), (1280, 720),
    (1600, 900), (1920, 1080), (2560, 1440),
)

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------

def list_cameras(maximum: int = 8, backend: Optional[int] = None) -> List[int]:
    """
    Indices of cameras that can actually be opened.

    Probing is genuinely slow on some systems (each failed open can take a
    second), so keep *maximum* small unless the user asked for a full scan.
    """
    found: List[int] = []
    for index in range(maximum):
        capture = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
        try:
            if capture.isOpened():
                ok, _frame = capture.read()
                if ok:
                    found.append(index)
        finally:
            capture.release()
    return found


def probe_resolutions(
    index: int = 0,
    backend: Optional[int] = None,
    candidates: Tuple[Tuple[int, int], ...] = COMMON_RESOLUTIONS,
) -> List[Tuple[int, int]]:
    """
    Which of *candidates* the camera accepts.

    Cameras silently substitute the nearest supported mode, so each request is
    verified by reading back what the device actually set.
    """
    capture = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
    supported: List[Tuple[int, int]] = []
    try:
        if not capture.isOpened():
            return supported
        for width, height in candidates:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            actual = (
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            if actual == (width, height) and actual not in supported:
                supported.append(actual)
    finally:
        capture.release()
    return supported


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------

class CameraSource:
    """
    A resilient frame source.

    Args:
        index: camera device index (ignored when *source* is set).
        width / height / fps: requested capture format.
        backend: OpenCV backend constant, from the platform bridge.
        mirror: horizontally flip frames, which is what people expect from a
            front-facing camera.
        source: path to a video file or a folder of images to read instead of
            a camera.
        loop: restart a file source when it reaches the end.
        auto_reconnect: try to reopen the device after read failures.
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        backend: Optional[int] = None,
        mirror: bool = True,
        buffer_size: int = 1,
        source: Optional[str] = None,
        loop: bool = True,
        auto_reconnect: bool = True,
        reconnect_delay: float = 1.5,
        max_reconnect_attempts: int = 5,
    ):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.backend = backend
        self.mirror = mirror
        self.buffer_size = buffer_size
        self.source = source
        self.loop = loop
        self.auto_reconnect = auto_reconnect
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts

        self._capture: Optional[cv2.VideoCapture] = None
        self._images: List[str] = []
        self._image_index = 0
        self._is_images = False
        self._is_file = False

        self._frames_read = 0
        self._frames_failed = 0
        self._reconnects = 0
        self._last_error = ""
        self._opened_at = 0.0

    # -- Lifecycle ----------------------------------------------------------

    def open(self) -> bool:
        """Open the source. Returns False when it cannot be opened."""
        if self.source:
            return self._open_source()
        return self._open_camera()

    def _open_source(self) -> bool:
        path = os.path.abspath(os.path.expanduser(self.source or ""))

        if os.path.isdir(path):
            patterns = [os.path.join(path, f"*{ext}") for ext in _IMAGE_EXTENSIONS]
            files: List[str] = []
            for pattern in patterns:
                files.extend(glob.glob(pattern))
            self._images = sorted(files)
            self._is_images = bool(self._images)
            if not self._is_images:
                self._last_error = f"Sin imagenes en {path}"
                return False
            self._image_index = 0
            self._opened_at = time.monotonic()
            print(f"[camera] Carpeta de imagenes: {len(self._images)} archivos", flush=True)
            return True

        if not os.path.exists(path):
            self._last_error = f"No existe: {path}"
            return False

        self._capture = cv2.VideoCapture(path)
        if not self._capture.isOpened():
            self._last_error = f"No pude abrir {path}"
            return False
        self._is_file = True
        self._opened_at = time.monotonic()
        print(f"[camera] Origen de video: {os.path.basename(path)}", flush=True)
        return True

    def _open_camera(self) -> bool:
        self._capture = (
            cv2.VideoCapture(self.index, self.backend) if self.backend is not None
            else cv2.VideoCapture(self.index)
        )
        if not self._capture.isOpened():
            self._last_error = f"No pude abrir la camara {self.index}"
            return False

        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture.set(cv2.CAP_PROP_FPS, self.fps)

        # Record what we actually got, which is often not what we asked for.
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
        self._opened_at = time.monotonic()
        print(
            f"[camera] Camara {self.index} abierta a {self.width}x{self.height}",
            flush=True,
        )
        return True

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._images.clear()

    def __enter__(self) -> "CameraSource":
        self.open()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # -- Reading ------------------------------------------------------------

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame.

        Returns ``(ok, frame)``. On failure with reconnection enabled it
        transparently reopens the device and retries.
        """
        if self._is_images:
            return self._read_image()

        if self._capture is None:
            return False, None

        ok, frame = self._capture.read()
        if ok and frame is not None:
            self._frames_read += 1
            return True, self._postprocess(frame)

        self._frames_failed += 1

        # A file that ran out is not an error when looping.
        if self._is_file and self.loop:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
            if ok and frame is not None:
                self._frames_read += 1
                return True, self._postprocess(frame)

        if self.auto_reconnect and not self._is_file:
            return self._reconnect()
        return False, None

    def _read_image(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._images:
            return False, None
        if self._image_index >= len(self._images):
            if not self.loop:
                return False, None
            self._image_index = 0

        path = self._images[self._image_index]
        self._image_index += 1
        frame = cv2.imread(path)
        if frame is None:
            self._frames_failed += 1
            return self._read_image() if self._image_index < len(self._images) else (False, None)

        self._frames_read += 1
        return True, self._postprocess(frame)

    def _postprocess(self, frame: np.ndarray) -> np.ndarray:
        """Mirror and, for file sources, normalise to the requested size."""
        if self.mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def _reconnect(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reopen the device, retrying with a short delay."""
        for attempt in range(1, self.max_reconnect_attempts + 1):
            print(
                f"[camera] Reconectando ({attempt}/{self.max_reconnect_attempts})...",
                flush=True,
            )
            self.close()
            time.sleep(self.reconnect_delay)
            if self.open():
                self._reconnects += 1
                ok, frame = (self._capture.read() if self._capture else (False, None))
                if ok and frame is not None:
                    self._frames_read += 1
                    return True, self._postprocess(frame)
        self._last_error = "Reconexion fallida"
        print("[camera] No pude reconectar con la camara", flush=True)
        return False, None

    def frames(self) -> Iterator[np.ndarray]:
        """Iterate frames until the source stops delivering them."""
        while True:
            ok, frame = self.read()
            if not ok or frame is None:
                return
            yield frame

    # -- Controls -----------------------------------------------------------

    def set_resolution(self, width: int, height: int) -> bool:
        """Change capture resolution live. Returns True if the device agreed."""
        if self._capture is None or self._is_images:
            return False
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        actual_w = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width, self.height = actual_w, actual_h
        return (actual_w, actual_h) == (width, height)

    def set_property(self, prop: int, value: float) -> bool:
        """Set any ``cv2.CAP_PROP_*`` value (exposure, gain, focus...)."""
        if self._capture is None:
            return False
        return bool(self._capture.set(prop, value))

    def get_property(self, prop: int) -> float:
        return self._capture.get(prop) if self._capture is not None else 0.0

    def switch_camera(self, index: int) -> bool:
        """Close the current device and open a different one."""
        self.close()
        self.index = index
        self.source = None
        self._is_file = False
        self._is_images = False
        return self.open()

    def toggle_mirror(self) -> bool:
        self.mirror = not self.mirror
        return self.mirror

    # -- Status -------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        if self._is_images:
            return bool(self._images)
        return self._capture is not None and self._capture.isOpened()

    @property
    def resolution(self) -> Tuple[int, int]:
        return self.width, self.height

    @property
    def frames_read(self) -> int:
        return self._frames_read

    @property
    def last_error(self) -> str:
        return self._last_error

    def stats(self) -> dict:
        """Diagnostics for the debug HUD."""
        return {
            "source": self.source or f"camara {self.index}",
            "resolution": f"{self.width}x{self.height}",
            "frames_read": self._frames_read,
            "frames_failed": self._frames_failed,
            "reconnects": self._reconnects,
            "uptime": round(time.monotonic() - self._opened_at, 1) if self._opened_at else 0.0,
            "last_error": self._last_error,
        }
