"""
Screenshot and video recording actions for LookThePerson.
"""

import os
import time

import cv2


class Recorder:
    """Handles screenshot capture and video recording."""

    def __init__(self, output_dir=None):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._screenshot_dir = os.path.join(output_dir or base, "screenshots")
        self._recording_dir = os.path.join(output_dir or base, "recordings")
        self._video_writer = None
        self._recording = False
        self._recording_path = None
        self._recording_fps = 20.0

    @property
    def is_recording(self):
        return self._recording

    def take_screenshot(self, frame):
        """Save current frame as PNG. Returns the file path."""
        os.makedirs(self._screenshot_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self._screenshot_dir, f"capture_{timestamp}.png")
        cv2.imwrite(path, frame)
        print(f"[recorder] Screenshot guardado: {path}", flush=True)
        return path

    def toggle_recording(self, frame):
        """
        Toggle video recording on/off.
        When starting, initializes the writer with the frame dimensions.
        When stopping, releases the writer.
        Returns (is_recording, file_path).
        """
        if self._recording:
            return self.stop_recording()
        else:
            return self.start_recording(frame)

    def start_recording(self, frame):
        """Start recording video."""
        os.makedirs(self._recording_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._recording_path = os.path.join(
            self._recording_dir, f"recording_{timestamp}.avi",
        )
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self._video_writer = cv2.VideoWriter(
            self._recording_path, fourcc, self._recording_fps, (w, h),
        )
        self._recording = True
        print(f"[recorder] Grabacion iniciada: {self._recording_path}", flush=True)
        return True, self._recording_path

    def stop_recording(self):
        """Stop recording and finalize the file."""
        if self._video_writer:
            self._video_writer.release()
            self._video_writer = None
        self._recording = False
        path = self._recording_path
        self._recording_path = None
        print(f"[recorder] Grabacion finalizada: {path}", flush=True)
        return False, path

    def write_frame(self, frame):
        """Write a frame to the video file if recording is active."""
        if self._recording and self._video_writer:
            self._video_writer.write(frame)

    def cleanup(self):
        """Release resources on shutdown."""
        if self._recording:
            self.stop_recording()
