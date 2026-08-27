"""
Segmentation-driven background effects for LookThePerson.

These take the person mask produced by the pose model and use it to separate
subject from background: blur it, replace it, cut the person out, or hide the
person entirely for privacy.

All functions accept the raw MediaPipe mask (a float image in 0..1) or a plain
numpy array, and handle resizing to the frame themselves.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

__all__ = [
    "prepare_mask",
    "blur_background",
    "replace_background",
    "silhouette",
    "cutout",
    "spotlight",
    "background_color",
    "outline_person",
    "hologram",
    "ghost_trail",
    "privacy_blur_region",
    "pixelate_region",
]


# ---------------------------------------------------------------------------
# Mask preparation
# ---------------------------------------------------------------------------

def prepare_mask(
    mask,
    shape: Tuple[int, int],
    threshold: float = 0.4,
    feather: int = 9,
    soft: bool = True,
) -> np.ndarray:
    """
    Normalise a segmentation mask to a float32 array matching *shape*.

    Args:
        mask: MediaPipe mask image or numpy array.
        shape: ``(height, width)`` of the target frame.
        threshold: values above this count as person.
        feather: blur radius applied to the edge; 0 disables feathering.
        soft: when True the mask keeps soft edges (better compositing);
            when False it is a hard 0/1 mask (faster, crisper).

    Returns:
        Float32 array in 0..1 with the same height and width as the frame.
    """
    data = mask.numpy_view() if hasattr(mask, "numpy_view") else np.asarray(mask)
    if data.ndim == 3:
        data = data[:, :, 0]
    data = data.astype(np.float32)

    height, width = shape[:2]
    if data.shape[:2] != (height, width):
        data = cv2.resize(data, (width, height), interpolation=cv2.INTER_LINEAR)

    if data.max() > 1.5:      # some sources give 0..255
        data = data / 255.0

    if soft:
        binary = np.clip((data - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)
    else:
        binary = (data > threshold).astype(np.float32)

    if feather > 0:
        k = max(3, int(feather) | 1)
        binary = cv2.GaussianBlur(binary, (k, k), 0)
    return binary


def _composite(foreground: np.ndarray, background: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Alpha-blend two frames using a single-channel float mask."""
    alpha = mask[:, :, None]
    return (foreground * alpha + background * (1.0 - alpha)).astype(np.uint8)


# ---------------------------------------------------------------------------
# Effects
# ---------------------------------------------------------------------------

def blur_background(frame: np.ndarray, mask, strength: int = 25) -> np.ndarray:
    """Keep the person sharp and blur everything behind them."""
    prepared = prepare_mask(mask, frame.shape)
    k = max(3, int(strength) | 1)
    blurred = cv2.GaussianBlur(frame, (k, k), 0)
    return _composite(frame, blurred, prepared)


def replace_background(
    frame: np.ndarray,
    mask,
    background: np.ndarray,
) -> np.ndarray:
    """
    Composite the person onto *background*.

    The background is resized to the frame automatically, so any image works
    as a virtual backdrop.
    """
    prepared = prepare_mask(mask, frame.shape)
    height, width = frame.shape[:2]
    if background.shape[:2] != (height, width):
        background = cv2.resize(background, (width, height), interpolation=cv2.INTER_LINEAR)
    if background.ndim == 2:
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    return _composite(frame, background, prepared)


def background_color(
    frame: np.ndarray,
    mask,
    color: Tuple[int, int, int] = (0, 177, 64),
) -> np.ndarray:
    """Flat colour background — a green screen without the screen."""
    plate = np.full_like(frame, color, dtype=np.uint8)
    return replace_background(frame, mask, plate)


def silhouette(
    frame: np.ndarray,
    mask,
    person_color: Tuple[int, int, int] = (255, 255, 255),
    background: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Render the person as a flat shape against a flat background."""
    prepared = prepare_mask(mask, frame.shape, soft=False, feather=5)
    person_plate = np.full_like(frame, person_color, dtype=np.uint8)
    background_plate = np.full_like(frame, background, dtype=np.uint8)
    return _composite(person_plate, background_plate, prepared)


def cutout(frame: np.ndarray, mask, invert: bool = False) -> np.ndarray:
    """
    Keep only the person (or, with *invert*, only the background).

    Removed pixels become black.
    """
    prepared = prepare_mask(mask, frame.shape)
    if invert:
        prepared = 1.0 - prepared
    return (frame * prepared[:, :, None]).astype(np.uint8)


def spotlight(
    frame: np.ndarray,
    mask,
    dim: float = 0.72,
    glow: bool = True,
) -> np.ndarray:
    """
    Darken everything except the person, as if lit by a follow spot.

    With *glow* the edge of the subject gets a soft halo, which reads much
    more like real light than a hard cut.
    """
    prepared = prepare_mask(mask, frame.shape, feather=21)
    darkened = (frame * (1.0 - dim)).astype(np.uint8)
    out = _composite(frame, darkened, prepared)

    if glow:
        edge = cv2.GaussianBlur(prepared, (0, 0), 12) - prepared
        edge = np.clip(edge, 0.0, 1.0)
        halo = (np.ones_like(frame) * 255 * edge[:, :, None] * 0.35).astype(np.uint8)
        out = cv2.add(out, halo)
    return out


def outline_person(
    frame: np.ndarray,
    mask,
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a contour around the person's silhouette."""
    prepared = prepare_mask(mask, frame.shape, soft=False, feather=0)
    binary = (prepared * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = frame.copy()
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def hologram(
    frame: np.ndarray,
    mask,
    color: Tuple[int, int, int] = (255, 180, 60),
    line_spacing: int = 4,
) -> np.ndarray:
    """
    Sci-fi projection look: tinted, scanlined subject on a dark field.
    """
    prepared = prepare_mask(mask, frame.shape, feather=7)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tint = np.zeros_like(frame, dtype=np.float32)
    normalized = (gray.astype(np.float32) / 255.0)[:, :, None]
    tint[:] = np.array(color, dtype=np.float32)
    subject = (tint * normalized).astype(np.uint8)

    spacing = max(2, line_spacing)
    subject[::spacing, :, :] = (subject[::spacing, :, :] * 0.35).astype(np.uint8)

    dark = (frame * 0.12).astype(np.uint8)
    out = _composite(subject, dark, prepared)
    return cv2.addWeighted(out, 1.0, cv2.GaussianBlur(out, (0, 0), 9), 0.45, 0)


def ghost_trail(
    frame: np.ndarray,
    mask,
    accumulator: Optional[np.ndarray],
    decay: float = 0.88,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Motion echo: past silhouettes fade out behind the live subject.

    Returns ``(rendered_frame, new_accumulator)`` — the caller keeps the
    accumulator between frames.
    """
    prepared = prepare_mask(mask, frame.shape, feather=11)
    person = (frame * prepared[:, :, None]).astype(np.float32)

    if accumulator is None or accumulator.shape != frame.shape:
        accumulator = np.zeros_like(frame, dtype=np.float32)

    accumulator = accumulator * decay
    accumulator = np.maximum(accumulator, person)

    out = cv2.addWeighted(frame, 1.0, accumulator.astype(np.uint8), 0.65, 0)
    return out, accumulator


# ---------------------------------------------------------------------------
# Region privacy helpers
# ---------------------------------------------------------------------------

def privacy_blur_region(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    strength: int = 35,
    padding: int = 8,
) -> np.ndarray:
    """
    Blur a rectangular region in place — used to anonymise faces.

    *box* is ``(x, y, width, height)`` in pixels. Out-of-frame boxes are
    clipped, so detector output can be passed straight in.
    """
    x, y, w, h = box
    height, width = frame.shape[:2]
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(width, x + w + padding)
    y1 = min(height, y + h + padding)
    if x1 <= x0 or y1 <= y0:
        return frame

    region = frame[y0:y1, x0:x1]
    k = max(3, int(strength) | 1)
    frame[y0:y1, x0:x1] = cv2.GaussianBlur(region, (k, k), 0)
    return frame


def pixelate_region(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    block: int = 12,
    padding: int = 6,
) -> np.ndarray:
    """Pixelate a rectangular region in place."""
    x, y, w, h = box
    height, width = frame.shape[:2]
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(width, x + w + padding)
    y1 = min(height, y + h + padding)
    if x1 <= x0 or y1 <= y0:
        return frame

    region = frame[y0:y1, x0:x1]
    rh, rw = region.shape[:2]
    block = max(2, block)
    small = cv2.resize(
        region, (max(1, rw // block), max(1, rh // block)), interpolation=cv2.INTER_LINEAR
    )
    frame[y0:y1, x0:x1] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    return frame
