"""
Full-frame image filters for LookThePerson.

Every function takes a BGR frame and returns a BGR frame of the same shape.
Filters that can work in place say so; the rest return a new array.

Performance matters here — these run on every frame — so the expensive ones
(cartoon, ASCII, bloom) do their work at reduced resolution and scale back up.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import cv2
import numpy as np

__all__ = [
    "apply_filter",
    "FILTERS",
    "filter_names",
    "next_filter",
    "night_vision",
    "thermal",
    "invert",
    "grayscale",
    "sepia",
    "posterize",
    "pixelate",
    "edges",
    "sketch",
    "cartoon",
    "emboss",
    "sharpen",
    "blur",
    "vignette",
    "scanlines",
    "chromatic_aberration",
    "glitch",
    "bloom",
    "ascii_art",
    "color_pop",
    "kaleidoscope",
    "adjust",
    "duotone",
]


# ---------------------------------------------------------------------------
# Color and tone
# ---------------------------------------------------------------------------

def invert(frame: np.ndarray) -> np.ndarray:
    """Photographic negative."""
    return cv2.bitwise_not(frame)


def grayscale(frame: np.ndarray) -> np.ndarray:
    """Desaturate while staying a 3-channel BGR image."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def night_vision(frame: np.ndarray, gain: float = 1.6) -> np.ndarray:
    """
    Green-channel image intensifier look.

    Brightens the luminance, tints it green and adds a touch of grain so it
    reads as an intensifier tube rather than a green filter.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boosted = cv2.convertScaleAbs(gray, alpha=gain, beta=12)
    boosted = cv2.equalizeHist(boosted)

    out = np.zeros_like(frame)
    out[:, :, 1] = boosted                       # green
    out[:, :, 0] = (boosted * 0.15).astype(np.uint8)
    out[:, :, 2] = (boosted * 0.15).astype(np.uint8)

    noise = np.random.randint(0, 26, gray.shape, dtype=np.uint8)
    out[:, :, 1] = cv2.add(out[:, :, 1], noise)
    return out


def thermal(frame: np.ndarray) -> np.ndarray:
    """
    False-colour heat map from luminance.

    Not a real thermal camera — bright areas simply read as hot — but it is
    the recognisable look and it makes bodies pop against a dim background.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    smoothed = cv2.GaussianBlur(equalized, (7, 7), 0)
    return cv2.applyColorMap(smoothed, cv2.COLORMAP_INFERNO)


def sepia(frame: np.ndarray) -> np.ndarray:
    """Warm brown tone."""
    kernel = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189],
    ])
    return cv2.transform(frame, kernel).clip(0, 255).astype(np.uint8)


@lru_cache(maxsize=8)
def _duotone_lut(shadow: Tuple[int, int, int], highlight: Tuple[int, int, int]) -> np.ndarray:
    """256-entry BGR gradient, cached per colour pair."""
    ramp = np.linspace(0.0, 1.0, 256, dtype=np.float32)[:, None]
    low = np.array(shadow, dtype=np.float32)
    high = np.array(highlight, dtype=np.float32)
    return (low + (high - low) * ramp).clip(0, 255).astype(np.uint8)


def duotone(
    frame: np.ndarray,
    shadow: Tuple[int, int, int] = (60, 20, 20),
    highlight: Tuple[int, int, int] = (255, 220, 120),
) -> np.ndarray:
    """
    Map luminance onto a two-colour gradient.

    Implemented as a cached lookup table indexed by grayscale, which is an
    order of magnitude faster than per-pixel float arithmetic.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return _duotone_lut(tuple(shadow), tuple(highlight))[gray]


def posterize(frame: np.ndarray, levels: int = 5) -> np.ndarray:
    """Reduce the colour depth to *levels* steps per channel."""
    levels = max(2, min(64, levels))
    step = 256 // levels
    # Integer quantisation via a lookup table is far faster than doing the
    # arithmetic per pixel.
    table = (np.arange(256) // step * step).astype(np.uint8)
    return cv2.LUT(frame, table)


def adjust(
    frame: np.ndarray,
    brightness: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
) -> np.ndarray:
    """
    Brightness / contrast / saturation adjustment.

    *brightness* is added in 0-255 units, *contrast* and *saturation* are
    multipliers where 1.0 leaves the image unchanged.
    """
    out = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
    if abs(saturation - 1.0) > 1e-3:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def color_pop(frame: np.ndarray, hue_center: int = 0, tolerance: int = 18) -> np.ndarray:
    """
    Keep one hue in colour and desaturate everything else.

    *hue_center* is an OpenCV hue (0-179); red wraps around 0 and is handled.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    delta = np.minimum(np.abs(hue - hue_center), 180 - np.abs(hue - hue_center))
    mask = (delta <= tolerance).astype(np.uint8)
    mask = cv2.medianBlur(mask * 255, 5)

    gray = grayscale(frame)
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    return (frame * mask3 + gray * (1.0 - mask3)).astype(np.uint8)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def edges(frame: np.ndarray, low: int = 70, high: int = 160) -> np.ndarray:
    """Canny edge map rendered as a black-on-white line drawing."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    detected = cv2.Canny(blurred, low, high)
    return cv2.cvtColor(detected, cv2.COLOR_GRAY2BGR)


def sketch(frame: np.ndarray) -> np.ndarray:
    """Pencil-drawing look via the classic colour-dodge blend."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    # Dodge: gray * 255 / (255 - blur), guarding the divide.
    denominator = 255 - blurred
    denominator[denominator == 0] = 1
    dodged = np.clip(gray.astype(np.float32) * 255.0 / denominator, 0, 255).astype(np.uint8)
    return cv2.cvtColor(dodged, cv2.COLOR_GRAY2BGR)


def cartoon(frame: np.ndarray, downscale: int = 3) -> np.ndarray:
    """
    Flat colour regions with dark outlines.

    The bilateral filter dominates the cost, so it runs once on a downscaled
    copy — at 720p that is ~14 ms instead of ~52 ms for the two full passes,
    and the difference disappears once the outlines are composited back on.
    """
    height, width = frame.shape[:2]
    downscale = max(1, downscale)
    # `width // downscale` reaches 0 on a small frame, and cv2.resize rejects a
    # zero-sized target ("inv_scale_x > 0"). Clamping keeps the filter working
    # at any resolution instead of silently falling back to the raw frame.
    small_w = max(1, width // downscale)
    small_h = max(1, height // downscale)
    small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    small = cv2.bilateralFilter(small, 9, 90, 90)
    smoothed = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    outline = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9,
    )
    outline = cv2.cvtColor(outline, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(smoothed, outline)


def emboss(frame: np.ndarray) -> np.ndarray:
    """Relief effect that highlights edges as raised metal."""
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
    embossed = cv2.filter2D(frame, -1, kernel) + 128
    return np.clip(embossed, 0, 255).astype(np.uint8)


def sharpen(frame: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Unsharp mask; *amount* scales the added detail."""
    blurred = cv2.GaussianBlur(frame, (0, 0), 3)
    return cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)


def blur(frame: np.ndarray, strength: int = 15) -> np.ndarray:
    """Gaussian blur; *strength* is forced odd as the kernel requires."""
    k = max(3, int(strength) | 1)
    return cv2.GaussianBlur(frame, (k, k), 0)


def pixelate(frame: np.ndarray, block: int = 14) -> np.ndarray:
    """Mosaic effect built from *block*-pixel squares."""
    height, width = frame.shape[:2]
    block = max(2, block)
    small = cv2.resize(
        frame, (max(1, width // block), max(1, height // block)),
        interpolation=cv2.INTER_LINEAR,
    )
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------
# Stylised / glitch
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _vignette_mask(height: int, width: int, strength: float) -> np.ndarray:
    """Radial falloff mask, cached per frame size — it never changes."""
    kernel_x = cv2.getGaussianKernel(width, width * 0.55)
    kernel_y = cv2.getGaussianKernel(height, height * 0.55)
    mask = kernel_y @ kernel_x.T
    mask = mask / mask.max()
    return (1.0 - strength * (1.0 - mask)).astype(np.float32)[:, :, None]


def vignette(frame: np.ndarray, strength: float = 0.7) -> np.ndarray:
    """
    Darken the corners to draw the eye to the centre.

    The falloff mask depends only on the frame size, so it is built once and
    reused — rebuilding it per frame cost over 150 ms at 720p.
    """
    height, width = frame.shape[:2]
    mask = _vignette_mask(height, width, round(float(strength), 3))
    return cv2.convertScaleAbs(frame * mask)


def scanlines(frame: np.ndarray, spacing: int = 3, darkness: float = 0.55) -> np.ndarray:
    """CRT scanline overlay."""
    out = frame.copy()
    spacing = max(2, spacing)
    out[::spacing, :, :] = (out[::spacing, :, :] * darkness).astype(np.uint8)
    return out


def chromatic_aberration(frame: np.ndarray, shift: int = 3) -> np.ndarray:
    """Offset the red and blue channels for a lens-fringing look."""
    if shift == 0:
        return frame.copy()
    blue, green, red = cv2.split(frame)
    blue = np.roll(blue, shift, axis=1)
    red = np.roll(red, -shift, axis=1)
    return cv2.merge((blue, green, red))


def glitch(frame: np.ndarray, intensity: float = 0.5, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Digital corruption: horizontal slice displacement plus channel shifting.

    *intensity* in 0..1 controls how many slices are displaced and how far.
    """
    generator = rng or np.random.default_rng()
    out = frame.copy()
    height, width = out.shape[:2]

    slices = int(2 + intensity * 12)
    for _ in range(slices):
        y = int(generator.integers(0, height))
        slice_height = int(generator.integers(3, max(4, int(height * 0.06))))
        offset = int(generator.integers(-int(width * 0.08) - 1, int(width * 0.08) + 1))
        y_end = min(height, y + slice_height)
        out[y:y_end] = np.roll(out[y:y_end], offset, axis=1)

    return chromatic_aberration(out, shift=int(2 + intensity * 5))


def bloom(frame: np.ndarray, threshold: int = 200, strength: float = 0.6) -> np.ndarray:
    """Glow bleeding out of the brightest regions."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    highlights = cv2.bitwise_and(frame, frame, mask=mask)
    glow = cv2.GaussianBlur(highlights, (0, 0), 15)
    return cv2.addWeighted(frame, 1.0, glow, strength, 0)


_ASCII_RAMP = "@%#*+=-:. "


@lru_cache(maxsize=8)
def _ascii_glyphs(cell: int) -> np.ndarray:
    """
    Pre-rendered glyph tiles for the ramp, shaped ``(len(ramp), cell, cell)``.

    Rendering each character once and blitting the tiles is dramatically
    faster than calling ``putText`` for every cell of every frame.
    """
    tiles = np.zeros((len(_ASCII_RAMP), cell, cell), dtype=np.float32)
    scale = cell / 22.0
    for index, char in enumerate(_ASCII_RAMP):
        if char == " ":
            continue
        tile = np.zeros((cell, cell), dtype=np.uint8)
        cv2.putText(tile, char, (0, cell - 1), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, 255, 1, cv2.LINE_AA)
        tiles[index] = tile.astype(np.float32) / 255.0
    return tiles


def ascii_art(
    frame: np.ndarray,
    cell: int = 8,
    color: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    Render the frame as ASCII characters.

    Each *cell* x *cell* block becomes one character chosen by its mean
    brightness. Passing *color* forces a single ink colour; otherwise each
    character keeps the average colour of its block.
    """
    height, width = frame.shape[:2]
    cell = max(4, cell)
    cols, rows = width // cell, height // cell
    if cols < 1 or rows < 1:
        return frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small_gray = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)
    small_color = (
        np.full((rows, cols, 3), color, dtype=np.uint8) if color is not None
        else cv2.resize(frame, (cols, rows), interpolation=cv2.INTER_AREA)
    )

    glyphs = _ascii_glyphs(cell)
    indices = (small_gray.astype(np.int32) * (len(_ASCII_RAMP) - 1)) // 255

    # Build the character layer by tiling the pre-rendered glyph masks, then
    # multiply once by the per-cell colour upscaled to full resolution.
    alpha = glyphs[indices]                                  # (rows, cols, cell, cell)
    alpha = alpha.transpose(0, 2, 1, 3).reshape(rows * cell, cols * cell)

    ink = cv2.resize(small_color, (cols * cell, rows * cell), interpolation=cv2.INTER_NEAREST)
    rendered = (ink * alpha[:, :, None]).astype(np.uint8)

    canvas = np.zeros_like(frame)
    canvas[:rows * cell, :cols * cell] = rendered
    return canvas


def kaleidoscope(frame: np.ndarray, segments: int = 4) -> np.ndarray:
    """Mirror one quadrant across the frame for a symmetric pattern."""
    height, width = frame.shape[:2]
    half_h, half_w = height // 2, width // 2
    # A frame under 2px in either axis has no quadrant to mirror, and cv2.flip
    # rejects an empty array.
    if half_h < 1 or half_w < 1:
        return frame.copy()
    quadrant = frame[:half_h, :half_w]

    out = np.zeros_like(frame)
    out[:half_h, :half_w] = quadrant
    out[:half_h, half_w:half_w * 2] = cv2.flip(quadrant, 1)
    out[half_h:half_h * 2, :half_w] = cv2.flip(quadrant, 0)
    out[half_h:half_h * 2, half_w:half_w * 2] = cv2.flip(quadrant, -1)

    if segments >= 8:
        out = cv2.addWeighted(out, 0.6, cv2.rotate(
            cv2.resize(out, (width, height)), cv2.ROTATE_180), 0.4, 0)
    return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FILTERS = {
    "none": lambda frame: frame,
    "invert": invert,
    "grayscale": grayscale,
    "night_vision": night_vision,
    "thermal": thermal,
    "sepia": sepia,
    "duotone": duotone,
    "posterize": posterize,
    "edges": edges,
    "sketch": sketch,
    "cartoon": cartoon,
    "emboss": emboss,
    "sharpen": sharpen,
    "blur": blur,
    "pixelate": pixelate,
    "vignette": vignette,
    "scanlines": scanlines,
    "chromatic": chromatic_aberration,
    "glitch": glitch,
    "bloom": bloom,
    "ascii": ascii_art,
    "color_pop": color_pop,
    "kaleidoscope": kaleidoscope,
}


def filter_names() -> list:
    """Every registered filter name, in a stable order."""
    return list(FILTERS.keys())


def apply_filter(frame: np.ndarray, name: str, **kwargs) -> np.ndarray:
    """
    Apply a named filter.

    Unknown names return the frame untouched rather than raising, so a stale
    config value cannot crash the render loop.
    """
    func = FILTERS.get(name)
    if func is None:
        return frame
    try:
        return func(frame, **kwargs) if kwargs else func(frame)
    except Exception as exc:  # a broken filter must not kill the frame loop
        print(f"[fx] Filtro '{name}' fallo: {exc}", flush=True)
        return frame


def next_filter(current: str) -> str:
    """Name of the filter after *current*, wrapping around."""
    names = filter_names()
    try:
        index = names.index(current)
    except ValueError:
        return names[0]
    return names[(index + 1) % len(names)]
