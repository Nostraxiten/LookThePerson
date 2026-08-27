"""
Input/output helpers for LookThePerson.

Kept in ``io_utils`` rather than ``io`` so it never shadows the standard
library module of that name.

* ``io_utils.camera`` — resilient camera/video/image-folder input.
* ``io_utils.capture`` — screenshots, video, animated GIFs and montages.
"""

from io_utils.camera import CameraSource, list_cameras, probe_resolutions
from io_utils.capture import MediaRecorder, build_montage, write_gif

__all__ = [
    "CameraSource",
    "list_cameras",
    "probe_resolutions",
    "MediaRecorder",
    "write_gif",
    "build_montage",
]
