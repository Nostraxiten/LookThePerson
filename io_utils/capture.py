"""
Screenshot, video and animated-GIF capture for LookThePerson.

Extends the original recorder with burst mode, contact-sheet montages,
animated GIFs (written with OpenCV/numpy only — no extra dependencies), a
capture history and automatic disk-space awareness.
"""

from __future__ import annotations

import os
import shutil
import struct
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

__all__ = ["MediaRecorder", "write_gif", "build_montage"]


class MediaRecorder:
    """
    Owns everything written to disk from the live frame stream.

    All paths are timestamped, and the recorder keeps a history so the HUD can
    show what was captured and the session report can list it.
    """

    def __init__(
        self,
        output_dir: str = "",
        video_fps: float = 20.0,
        codec: str = "XVID",
        extension: str = "avi",
        screenshot_format: str = "png",
    ):
        base = output_dir or os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        self.screenshot_dir = os.path.join(base, "screenshots")
        self.recording_dir = os.path.join(base, "recordings")
        self.gif_dir = os.path.join(base, "gifs")

        self.video_fps = video_fps
        self.codec = codec
        self.extension = extension
        self.screenshot_format = screenshot_format.lstrip(".")

        self._writer: Optional[cv2.VideoWriter] = None
        self._recording = False
        self._recording_path: Optional[str] = None
        self._recording_started = 0.0
        self._recorded_frames = 0

        self._shot_sequence = 0
        self._burst_remaining = 0
        self._burst_next = 0.0
        self._burst_interval = 0.4

        self._gif_frames: List[np.ndarray] = []
        self._gif_recording = False
        self._gif_max_frames = 60
        self._gif_stride = 3
        self._gif_counter = 0

        self._history: List[Dict[str, Any]] = []

    # -- Screenshots --------------------------------------------------------

    def screenshot(self, frame: np.ndarray, prefix: str = "capture") -> Optional[str]:
        """Save a single frame. Returns the path, or None on failure."""
        try:
            os.makedirs(self.screenshot_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            # A per-session sequence number guarantees uniqueness: burst shots
            # can land inside the same second, and even the same millisecond.
            self._shot_sequence += 1
            path = os.path.join(
                self.screenshot_dir,
                f"{prefix}_{stamp}_{self._shot_sequence:04d}.{self.screenshot_format}",
            )
            if not cv2.imwrite(path, frame):
                print("[capture] No pude escribir la captura", flush=True)
                return None
            self._remember("screenshot", path)
            print(f"[capture] Captura guardada: {path}", flush=True)
            return path
        except OSError as exc:
            print(f"[capture] Error al guardar captura: {exc}", flush=True)
            return None

    def start_burst(self, count: int = 5, interval: float = 0.4, now: float = 0.0) -> None:
        """Queue a burst of *count* screenshots spaced *interval* apart."""
        self._burst_remaining = max(1, count)
        self._burst_interval = max(0.05, interval)
        self._burst_next = now

    def update_burst(self, frame: np.ndarray, now: float) -> Optional[str]:
        """Call each frame; saves the next burst shot when it is due."""
        if self._burst_remaining <= 0 or now < self._burst_next:
            return None
        self._burst_remaining -= 1
        self._burst_next = now + self._burst_interval
        return self.screenshot(frame, prefix="burst")

    @property
    def burst_active(self) -> bool:
        return self._burst_remaining > 0

    # -- Video --------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return self._recording

    def toggle_recording(self, frame: np.ndarray) -> Tuple[bool, Optional[str]]:
        """Start or stop recording. Returns ``(is_recording, path)``."""
        return self.stop_recording() if self._recording else self.start_recording(frame)

    def start_recording(self, frame: np.ndarray) -> Tuple[bool, Optional[str]]:
        """Begin writing video sized to *frame*."""
        try:
            os.makedirs(self.recording_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.recording_dir, f"recording_{stamp}.{self.extension}")

            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            writer = cv2.VideoWriter(path, fourcc, self.video_fps, (width, height))
            if not writer.isOpened():
                print(
                    f"[capture] El codec {self.codec} no esta disponible; "
                    "prueba con MJPG en la configuracion",
                    flush=True,
                )
                return False, None

            self._writer = writer
            self._recording = True
            self._recording_path = path
            self._recording_started = time.monotonic()
            self._recorded_frames = 0
            print(f"[capture] Grabacion iniciada: {path}", flush=True)
            return True, path
        except OSError as exc:
            print(f"[capture] No pude iniciar la grabacion: {exc}", flush=True)
            return False, None

    def stop_recording(self) -> Tuple[bool, Optional[str]]:
        """Finalise the video file."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

        path = self._recording_path
        duration = time.monotonic() - self._recording_started if self._recording else 0.0
        self._recording = False
        self._recording_path = None

        if path:
            self._remember("video", path, duration=round(duration, 1),
                           frames=self._recorded_frames)
            print(
                f"[capture] Grabacion finalizada: {path} "
                f"({duration:.1f}s, {self._recorded_frames} frames)",
                flush=True,
            )
        return False, path

    def write_frame(self, frame: np.ndarray) -> None:
        """Append a frame to the active recording."""
        if self._recording and self._writer is not None:
            self._writer.write(frame)
            self._recorded_frames += 1

    def recording_duration(self) -> float:
        return time.monotonic() - self._recording_started if self._recording else 0.0

    # -- GIF ----------------------------------------------------------------

    def start_gif(self, max_frames: int = 60, stride: int = 3) -> None:
        """
        Begin collecting frames for an animated GIF.

        *stride* keeps every Nth frame, which is how a 2-second clip becomes a
        reasonably sized loop.
        """
        self._gif_frames = []
        self._gif_recording = True
        self._gif_max_frames = max(2, max_frames)
        self._gif_stride = max(1, stride)
        self._gif_counter = 0
        print("[capture] Captura de GIF iniciada", flush=True)

    def update_gif(self, frame: np.ndarray) -> Optional[str]:
        """
        Collect a frame; writes the GIF once enough have been gathered.

        Returns the path when the GIF is finished, else None.
        """
        if not self._gif_recording:
            return None

        self._gif_counter += 1
        if self._gif_counter % self._gif_stride == 0:
            # Half size keeps GIFs manageable without looking broken.
            height, width = frame.shape[:2]
            self._gif_frames.append(
                cv2.resize(frame, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
            )

        if len(self._gif_frames) >= self._gif_max_frames:
            return self.finish_gif()
        return None

    def finish_gif(self) -> Optional[str]:
        """Write the collected frames out as an animated GIF."""
        self._gif_recording = False
        if len(self._gif_frames) < 2:
            self._gif_frames = []
            return None

        try:
            os.makedirs(self.gif_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.gif_dir, f"clip_{stamp}.gif")
            delay_cs = max(2, int(100 / max(1.0, self.video_fps / self._gif_stride)))
            write_gif(path, self._gif_frames, delay_cs=delay_cs)
            self._remember("gif", path, frames=len(self._gif_frames))
            print(f"[capture] GIF guardado: {path}", flush=True)
            return path
        except (OSError, ValueError) as exc:
            print(f"[capture] No pude escribir el GIF: {exc}", flush=True)
            return None
        finally:
            self._gif_frames = []

    @property
    def gif_recording(self) -> bool:
        return self._gif_recording

    @property
    def gif_progress(self) -> float:
        """How full the GIF buffer is, 0..1."""
        if not self._gif_recording:
            return 0.0
        return len(self._gif_frames) / self._gif_max_frames

    # -- Montage ------------------------------------------------------------

    def montage(self, columns: int = 3, limit: int = 9) -> Optional[str]:
        """Build a contact sheet from the most recent screenshots."""
        shots = [item["path"] for item in self._history if item["kind"] == "screenshot"]
        shots = shots[-limit:]
        if len(shots) < 2:
            return None

        images = [cv2.imread(path) for path in shots]
        images = [image for image in images if image is not None]
        if len(images) < 2:
            return None

        sheet = build_montage(images, columns=columns)
        os.makedirs(self.screenshot_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.screenshot_dir, f"montage_{stamp}.{self.screenshot_format}")
        cv2.imwrite(path, sheet)
        self._remember("montage", path, count=len(images))
        print(f"[capture] Montaje guardado: {path}", flush=True)
        return path

    # -- Housekeeping -------------------------------------------------------

    def _remember(self, kind: str, path: str, **extra: Any) -> None:
        self._history.append({
            "kind": kind, "path": path,
            "time": time.strftime("%H:%M:%S"), **extra,
        })
        if len(self._history) > 200:
            self._history.pop(0)

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def counts(self) -> Dict[str, int]:
        """How many of each kind of file were produced this session."""
        totals: Dict[str, int] = {}
        for item in self._history:
            totals[item["kind"]] = totals.get(item["kind"], 0) + 1
        return totals

    def free_space_mb(self) -> float:
        """Free space on the output volume, in megabytes."""
        try:
            target = self.screenshot_dir if os.path.isdir(self.screenshot_dir) else "."
            return shutil.disk_usage(target).free / (1024 * 1024)
        except OSError:
            return 0.0

    def cleanup(self) -> None:
        """Flush anything still open — call at shutdown."""
        if self._recording:
            self.stop_recording()
        if self._gif_recording:
            self.finish_gif()


# ---------------------------------------------------------------------------
# GIF writer
# ---------------------------------------------------------------------------

def write_gif(
    path: str,
    frames: Sequence[np.ndarray],
    delay_cs: int = 8,
    loop: int = 0,
) -> str:
    """
    Write BGR frames as an animated GIF.

    Implemented directly against the GIF89a spec so the project keeps its
    three-dependency footprint. Colours are quantised to a shared 216-entry
    web-safe palette, which avoids a per-frame palette pass and is more than
    good enough for a webcam clip.

    Args:
        path: destination file.
        frames: BGR images, all the same size.
        delay_cs: delay between frames in hundredths of a second.
        loop: 0 loops forever, otherwise the repeat count.
    """
    if len(frames) < 1:
        raise ValueError("Se necesita al menos un frame")

    height, width = frames[0].shape[:2]

    # 6x6x6 web-safe palette, padded to 256 entries.
    levels = np.array([0, 51, 102, 153, 204, 255], dtype=np.uint8)
    palette = np.zeros((256, 3), dtype=np.uint8)
    index = 0
    for r in levels:
        for g in levels:
            for b in levels:
                palette[index] = (r, g, b)
                index += 1

    def quantise(frame: np.ndarray) -> np.ndarray:
        """Map a BGR frame onto palette indices."""
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.int16)
        # Nearest of the six levels per channel: round(value / 51).
        quantised = np.clip((rgb + 25) // 51, 0, 5).astype(np.int32)
        return (quantised[:, :, 0] * 36 + quantised[:, :, 1] * 6 + quantised[:, :, 2]).astype(np.uint8)

    with open(path, "wb") as handle:
        handle.write(b"GIF89a")
        handle.write(struct.pack("<HHBBB", width, height, 0xF7, 0, 0))
        handle.write(palette.tobytes())

        # Netscape looping extension.
        handle.write(b"\x21\xFF\x0BNETSCAPE2.0\x03\x01")
        handle.write(struct.pack("<H", loop))
        handle.write(b"\x00")

        for frame in frames:
            indices = quantise(frame)
            handle.write(b"\x21\xF9\x04\x00")
            handle.write(struct.pack("<H", max(2, delay_cs)))
            handle.write(b"\x00\x00")
            handle.write(b"\x2C")
            handle.write(struct.pack("<HHHHB", 0, 0, width, height, 0))
            handle.write(_gif_lzw_encode(indices.tobytes()))

    return path


def _gif_lzw_encode(data: bytes, min_code_size: int = 8) -> bytes:
    """
    LZW-compress pixel indices into GIF image sub-blocks.

    Follows the GIF variable-width LZW scheme: codes grow from
    ``min_code_size + 1`` bits, the dictionary resets when it reaches 4096
    entries, and output is emitted in sub-blocks of at most 255 bytes.
    """
    clear_code = 1 << min_code_size
    end_code = clear_code + 1

    dictionary: Dict[bytes, int] = {bytes([i]): i for i in range(clear_code)}
    next_code = end_code + 1
    code_size = min_code_size + 1

    out = bytearray()
    bit_buffer = 0
    bit_count = 0

    def emit(code: int) -> None:
        nonlocal bit_buffer, bit_count
        bit_buffer |= code << bit_count
        bit_count += code_size
        while bit_count >= 8:
            out.append(bit_buffer & 0xFF)
            bit_buffer >>= 8
            bit_count -= 8

    emit(clear_code)
    current = b""
    for byte in data:
        candidate = current + bytes([byte])
        if candidate in dictionary:
            current = candidate
            continue
        emit(dictionary[current])
        if next_code < 4096:
            dictionary[candidate] = next_code
            next_code += 1
            if next_code > (1 << code_size) and code_size < 12:
                code_size += 1
        else:
            emit(clear_code)
            dictionary = {bytes([i]): i for i in range(clear_code)}
            next_code = end_code + 1
            code_size = min_code_size + 1
        current = bytes([byte])

    if current:
        emit(dictionary[current])
    emit(end_code)

    if bit_count > 0:
        out.append(bit_buffer & 0xFF)

    # Split into sub-blocks, each prefixed with its length.
    blocks = bytearray([min_code_size])
    for start in range(0, len(out), 255):
        chunk = out[start:start + 255]
        blocks.append(len(chunk))
        blocks.extend(chunk)
    blocks.append(0)
    return bytes(blocks)


# ---------------------------------------------------------------------------
# Montage
# ---------------------------------------------------------------------------

def build_montage(
    images: Sequence[np.ndarray],
    columns: int = 3,
    cell_width: int = 480,
    padding: int = 6,
    background: Tuple[int, int, int] = (18, 18, 18),
) -> np.ndarray:
    """Arrange images into a padded grid contact sheet."""
    if not images:
        raise ValueError("Se necesita al menos una imagen")

    columns = max(1, columns)
    rows = (len(images) + columns - 1) // columns

    first_h, first_w = images[0].shape[:2]
    cell_height = int(cell_width * first_h / first_w)

    sheet_w = columns * cell_width + (columns + 1) * padding
    sheet_h = rows * cell_height + (rows + 1) * padding
    sheet = np.full((sheet_h, sheet_w, 3), background, dtype=np.uint8)

    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        resized = cv2.resize(image, (cell_width, cell_height), interpolation=cv2.INTER_AREA)
        y = padding + row * (cell_height + padding)
        x = padding + column * (cell_width + padding)
        sheet[y:y + cell_height, x:x + cell_width] = resized

    return sheet
