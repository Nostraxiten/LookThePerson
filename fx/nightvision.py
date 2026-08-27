"""
Night vision pipeline for LookThePerson.

A surveillance camera does not simply darken its picture after sunset — it
changes sensor mode. This module reproduces the four looks a real installation
switches between, plus the metering that decides when to switch:

* ``dia`` — daylight, colour, mild contrast lift only.
* ``ir`` — infrared cut filter removed: monochrome, local-contrast boosted, with
  the radial falloff of an on-board IR illuminator and sensor grain.
* ``intensificador`` — image-intensifier tube: the green phosphor look, heavy
  gain, bloom on highlights, scan structure.
* ``termico`` — false-colour thermal palette, for picking a body out of a scene
  with no usable light at all.
* ``realce`` — keeps colour but lifts the shadows, which is what you want when
  there is *some* light and you care about the colour of a jacket.

``auto`` meters the scene and moves between ``dia`` and ``ir`` on a hysteresis
band, so a camera pointed at a doorway at dusk does not oscillate.

Only ``termico`` is false colour; none of these modes recover detail the sensor
never captured. They make an underexposed frame legible, which is a different
claim from seeing in the dark.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import cv2
import numpy as np

from core.filters import ExponentialFilter, Hysteresis

__all__ = [
    "NIGHT_MODES",
    "NightVisionProcessor",
    "scene_luminance",
    "ir_illuminated",
    "intensifier",
    "thermal_view",
    "shadow_lift",
    "next_night_mode",
]

#: Selectable modes, in cycling order. ``auto`` leads because it is the one an
#: unattended camera should be left in.
NIGHT_MODES: Tuple[str, ...] = ("auto", "dia", "ir", "intensificador", "termico", "realce")

#: Scene luminance (0-255) below which ``auto`` calls it night. The two values
#: form a hysteresis band; a single threshold makes the mode flicker at dusk.
NIGHT_ENTER_LUX = 62.0
NIGHT_EXIT_LUX = 88.0


def next_night_mode(current: str) -> str:
    """Name of the mode after *current*, wrapping around."""
    try:
        index = NIGHT_MODES.index(current)
    except ValueError:
        return NIGHT_MODES[0]
    return NIGHT_MODES[(index + 1) % len(NIGHT_MODES)]


# ---------------------------------------------------------------------------
# Metering
# ---------------------------------------------------------------------------

def scene_luminance(frame: np.ndarray) -> float:
    """
    Mean luminance of the frame, 0-255.

    Measured on a heavily downscaled copy: the average of a 64x36 thumbnail is
    within a fraction of a level of the full-resolution mean and costs
    essentially nothing per frame.
    """
    if frame is None or frame.size == 0:
        return 0.0
    height, width = frame.shape[:2]
    small = cv2.resize(
        frame, (min(64, max(1, width)), min(36, max(1, height))),
        interpolation=cv2.INTER_AREA,
    )
    if small.ndim == 3:
        small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(small.mean())


# ---------------------------------------------------------------------------
# Cached masks and tables
# ---------------------------------------------------------------------------

@lru_cache(maxsize=6)
def _illuminator_mask(height: int, width: int, reach: float) -> np.ndarray:
    """
    Radial falloff of an on-camera IR illuminator, cached per frame size.

    The LEDs sit next to the lens, so the centre of the shot is lit and the
    corners fall away — the single most recognisable trait of IR CCTV footage,
    and the reason a subject at the edge of frame looks underexposed.
    """
    ys = np.linspace(-1.0, 1.0, max(1, height), dtype=np.float32)[:, None]
    xs = np.linspace(-1.0, 1.0, max(1, width), dtype=np.float32)[None, :]
    radius = np.sqrt(xs * xs + ys * ys) / 1.4142
    falloff = np.clip(1.0 - (radius / max(0.05, reach)) ** 2.1, 0.06, 1.0)
    return falloff.astype(np.float32)[:, :, None]


@lru_cache(maxsize=4)
def _noise_bank(height: int, width: int, seed: int) -> np.ndarray:
    """
    A fixed bank of sensor noise, cached and rolled rather than regenerated.

    Drawing fresh Gaussian noise every frame at 720p costs more than the rest
    of the night-vision pipeline put together; rolling one buffer by a random
    offset is visually indistinguishable.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, (max(1, height), max(1, width))).astype(np.float32)


@lru_cache(maxsize=8)
def _gamma_table(gamma: float) -> np.ndarray:
    """256-entry gamma LUT — far cheaper than per-pixel float powers."""
    safe = max(0.05, gamma)
    ramp = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    return np.clip((ramp ** (1.0 / safe)) * 255.0, 0, 255).astype(np.uint8)


def _clahe(gray: np.ndarray, clip: float = 2.6, grid: int = 8) -> np.ndarray:
    """
    Local contrast equalisation.

    Global ``equalizeHist`` blows out a frame with one bright lamp in it, which
    is the usual night scene; CLAHE lifts each region against its own
    neighbourhood, so a face in shadow becomes readable without the lamp
    swallowing the rest of the shot.
    """
    height, width = gray.shape[:2]
    tiles = max(1, min(grid, height, width))
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(gray)


def _add_grain(gray: np.ndarray, amount: float, offset: int) -> np.ndarray:
    """Apply cached sensor grain, rolled by *offset* so it animates."""
    if amount <= 0.0:
        return gray
    height, width = gray.shape[:2]
    bank = _noise_bank(height, width, 0x5EED)
    grain = np.roll(bank, offset % max(1, height), axis=0)
    out = gray.astype(np.float32) + grain * amount
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Looks
# ---------------------------------------------------------------------------

def ir_illuminated(
    frame: np.ndarray,
    gain: float = 1.0,
    reach: float = 0.95,
    grain: float = 5.0,
    offset: int = 0,
) -> np.ndarray:
    """
    Monochrome IR look with illuminator falloff.

    *gain* multiplies the exposure after equalisation, *reach* is how far the
    illuminator throws (1.0 lights the corners, 0.6 is a tight hotspot).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    lifted = _clahe(gray, clip=2.8)
    if abs(gain - 1.0) > 1e-3:
        lifted = cv2.convertScaleAbs(lifted, alpha=gain, beta=0)
    lifted = cv2.LUT(lifted, _gamma_table(1.25))
    lifted = _add_grain(lifted, grain, offset)

    out = cv2.cvtColor(lifted, cv2.COLOR_GRAY2BGR).astype(np.float32)
    out *= _illuminator_mask(frame.shape[0], frame.shape[1], round(reach, 2))
    # IR sensors read very slightly cool rather than perfectly neutral.
    out[:, :, 0] *= 1.06
    out[:, :, 2] *= 0.97
    return np.clip(out, 0, 255).astype(np.uint8)


def intensifier(
    frame: np.ndarray,
    gain: float = 1.7,
    grain: float = 9.0,
    offset: int = 0,
    scanlines: bool = True,
) -> np.ndarray:
    """Green image-intensifier tube, with bloom on the brightest regions."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    lifted = _clahe(gray, clip=3.4)
    lifted = cv2.convertScaleAbs(lifted, alpha=gain, beta=10)
    lifted = _add_grain(lifted, grain, offset)

    out = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.float32)
    out[:, :, 1] = lifted                       # green phosphor
    out[:, :, 0] = lifted * 0.16
    out[:, :, 2] = lifted * 0.13

    # Highlights bloom in a real tube; that halo is most of the look.
    _, hot = cv2.threshold(lifted, 205, 255, cv2.THRESH_BINARY)
    if hot.any():
        halo = cv2.GaussianBlur(hot.astype(np.float32), (0, 0), 9)
        out[:, :, 1] = np.minimum(out[:, :, 1] + halo * 0.55, 255.0)

    out *= _illuminator_mask(frame.shape[0], frame.shape[1], 1.05)
    if scanlines and frame.shape[0] >= 4:
        out[::3, :, :] *= 0.72
    return np.clip(out, 0, 255).astype(np.uint8)


def thermal_view(frame: np.ndarray, palette: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """
    False-colour heat palette driven by luminance.

    This is not a thermal sensor: it maps brightness, not temperature. It earns
    its place because a body lit by an IR illuminator is usually the brightest
    thing in the shot, so the palette does separate people from the background
    — but a warm radiator in a dark room will not show up at all.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    lifted = _clahe(gray, clip=3.0)
    smoothed = cv2.GaussianBlur(lifted, (0, 0), 1.6)
    return cv2.applyColorMap(smoothed, palette)


def shadow_lift(frame: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Lift the shadows while keeping colour.

    Works on the L channel in LAB so the hues survive — the point of this mode
    is being able to say what colour a coat was.
    """
    if frame.ndim != 3:
        return frame.copy()
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    lifted = _clahe(lightness, clip=1.6 + 1.8 * max(0.0, strength))
    lifted = cv2.LUT(lifted, _gamma_table(1.0 + 0.35 * max(0.0, strength)))
    merged = cv2.merge((lifted, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class NightVisionProcessor:
    """
    Stateful night-vision stage for the frame loop.

    Owns the metering, the day/night hysteresis and the animation offset that
    keeps the sensor grain moving, so a mode only has to call
    :meth:`process` once per frame.
    """

    def __init__(self, mode: str = "auto", gain: float = 1.0, auto_gain: bool = True):
        self.mode = mode if mode in NIGHT_MODES else "auto"
        self.gain = gain
        self.auto_gain = auto_gain

        self._lux = ExponentialFilter(alpha=0.12, initial=128.0)
        self._night = Hysteresis(low=NIGHT_ENTER_LUX, high=NIGHT_EXIT_LUX, initial=False)
        self._effective = "dia"
        self._offset = 0
        self._applied_gain = 1.0

    # -- Introspection ------------------------------------------------------

    @property
    def luminance(self) -> float:
        """Smoothed scene luminance, 0-255."""
        return float(self._lux.value or 0.0)

    @property
    def is_night(self) -> bool:
        """Whether metering currently calls the scene dark."""
        return not self._night.state

    @property
    def effective_mode(self) -> str:
        """The mode actually applied last frame — resolves ``auto``."""
        return self._effective

    @property
    def applied_gain(self) -> float:
        """Exposure multiplier used last frame, after auto-gain."""
        return self._applied_gain

    def status(self) -> Dict[str, object]:
        """Everything the OSD needs about the sensor, in one call."""
        return {
            "mode": self.mode,
            "effective": self._effective,
            "luminance": round(self.luminance, 1),
            "night": self.is_night,
            "gain": round(self._applied_gain, 2),
            "auto_gain": self.auto_gain,
        }

    # -- Control ------------------------------------------------------------

    def cycle(self) -> str:
        """Advance to the next mode and return its name."""
        self.mode = next_night_mode(self.mode)
        return self.mode

    def set_mode(self, mode: str) -> str:
        if mode in NIGHT_MODES:
            self.mode = mode
        return self.mode

    def adjust_gain(self, delta: float) -> float:
        """Nudge manual gain; also switches auto-gain off."""
        self.auto_gain = False
        self.gain = float(min(4.0, max(0.25, self.gain + delta)))
        return self.gain

    def toggle_auto_gain(self) -> bool:
        self.auto_gain = not self.auto_gain
        return self.auto_gain

    # -- Per-frame ----------------------------------------------------------

    def resolve(self, frame: np.ndarray) -> str:
        """
        Meter the frame and decide which look applies, without processing it.

        Split out from :meth:`process` so a caller can meter a frame it has
        decided not to transform and still get a truthful OSD reading.
        """
        self._lux.update(scene_luminance(frame))
        self._night.update(self.luminance)

        if self.mode == "auto":
            self._effective = "ir" if self.is_night else "dia"
        else:
            self._effective = self.mode
        return self._effective

    def _gain_for(self, look: str) -> float:
        """Exposure multiplier, from metering when auto-gain is on."""
        if not self.auto_gain:
            return self.gain
        # Aim for a mid-grey average; clamp hard so a nearly black frame does
        # not amplify pure noise to full scale.
        target = 118.0
        measured = max(6.0, self.luminance)
        base = 1.0 if look == "dia" else 1.25
        return float(min(3.2, max(0.6, base * target / measured)))

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Meter and transform one frame, returning the processed image.

        The input is never modified in place; every look allocates its own
        output, because the mode still needs the original for its own metering
        on the next frame.
        """
        if frame is None or frame.size == 0:
            return frame

        look = self.resolve(frame)
        self._offset = (self._offset + 7) % 4096
        gain = self._gain_for(look)
        self._applied_gain = gain

        if look == "ir":
            return ir_illuminated(frame, gain=gain, offset=self._offset)
        if look == "intensificador":
            return intensifier(frame, gain=max(1.0, gain), offset=self._offset)
        if look == "termico":
            return thermal_view(frame)
        if look == "realce":
            return shadow_lift(frame, strength=min(1.6, gain))

        # Daylight: a gentle lift only, so the picture stays honest.
        if abs(gain - 1.0) < 0.06:
            return frame.copy()
        return cv2.convertScaleAbs(frame, alpha=min(1.6, gain), beta=0)

    def reset(self) -> None:
        self._lux.reset()
        self._lux.update(128.0)
        self._night.reset(False)
        self._effective = "dia"
        self._offset = 0
        self._applied_gain = 1.0
