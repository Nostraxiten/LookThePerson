"""
Color themes for LookThePerson.

Every drawing routine pulls its colors from the active :class:`Theme` rather
than hard-coding BGR tuples, so a single keypress restyles the whole app.

Colors are OpenCV-native BGR triples.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple

__all__ = ["Theme", "THEMES", "get_theme", "theme_names", "next_theme", "palette_for"]

Color = Tuple[int, int, int]


@dataclass(frozen=True)
class Theme:
    """A complete visual palette."""

    name: str
    skeleton: Color
    joints: Color
    hand: Color
    face_mesh: Color
    face_contour: Color
    iris: Color
    box: Color
    accent: Color
    text: Color
    text_dim: Color
    background: Color
    good: Color
    warn: Color
    danger: Color
    grid: Color
    trail: Color
    # Cycled through for per-category or per-person coloring.
    categorical: Tuple[Color, ...] = field(default=())

    def category_color(self, index: int) -> Color:
        """Stable color for a category or person index."""
        if not self.categorical:
            return self.accent
        return self.categorical[index % len(self.categorical)]

    def color_for_name(self, name: str) -> Color:
        """Deterministic color derived from a label, stable across runs."""
        if not self.categorical:
            return self.accent
        digest = 0
        for char in name:
            digest = (digest * 131 + ord(char)) & 0xFFFFFFFF
        return self.categorical[digest % len(self.categorical)]

    def status_color(self, level: str) -> Color:
        """Map ``"good"`` / ``"warn"`` / ``"danger"`` to a color."""
        return {"good": self.good, "warn": self.warn, "danger": self.danger}.get(level, self.text)

    def variant(self, **overrides) -> "Theme":
        """Copy of this theme with individual colors replaced."""
        return replace(self, **overrides)


_DEFAULT_CATEGORICAL: Tuple[Color, ...] = (
    (255, 128, 64), (80, 220, 120), (80, 120, 255), (0, 210, 240),
    (220, 120, 255), (120, 255, 220), (255, 200, 80), (180, 180, 255),
    (100, 255, 100), (255, 100, 180), (60, 200, 255), (200, 255, 60),
)


THEMES: Dict[str, Theme] = {
    "cyber": Theme(
        name="cyber",
        skeleton=(255, 200, 0), joints=(0, 255, 120), hand=(255, 180, 0),
        face_mesh=(90, 140, 90), face_contour=(0, 255, 200), iris=(255, 100, 255),
        box=(0, 220, 255), accent=(0, 220, 255), text=(230, 230, 230),
        text_dim=(130, 130, 130), background=(18, 18, 22),
        good=(90, 255, 90), warn=(0, 190, 255), danger=(70, 70, 255),
        grid=(60, 60, 60), trail=(255, 160, 40),
        categorical=_DEFAULT_CATEGORICAL,
    ),
    "matrix": Theme(
        name="matrix",
        skeleton=(0, 255, 60), joints=(120, 255, 160), hand=(0, 220, 90),
        face_mesh=(0, 140, 40), face_contour=(0, 255, 90), iris=(160, 255, 190),
        box=(0, 255, 80), accent=(0, 255, 80), text=(150, 255, 170),
        text_dim=(0, 130, 45), background=(4, 12, 4),
        good=(0, 255, 90), warn=(0, 230, 200), danger=(60, 90, 255),
        grid=(0, 70, 25), trail=(0, 200, 70),
        categorical=((0, 255, 70), (0, 200, 90), (60, 255, 140), (0, 150, 60)),
    ),
    "sunset": Theme(
        name="sunset",
        skeleton=(80, 120, 255), joints=(120, 200, 255), hand=(60, 160, 255),
        face_mesh=(110, 130, 200), face_contour=(120, 200, 255), iris=(200, 140, 255),
        box=(80, 170, 255), accent=(90, 160, 255), text=(240, 235, 230),
        text_dim=(150, 140, 140), background=(30, 20, 35),
        good=(120, 220, 160), warn=(60, 190, 255), danger=(80, 80, 240),
        grid=(70, 55, 70), trail=(90, 140, 255),
        categorical=((80, 120, 255), (120, 190, 255), (170, 140, 255),
                     (90, 220, 240), (110, 110, 250), (200, 170, 255)),
    ),
    "mono": Theme(
        name="mono",
        skeleton=(235, 235, 235), joints=(255, 255, 255), hand=(200, 200, 200),
        face_mesh=(120, 120, 120), face_contour=(220, 220, 220), iris=(255, 255, 255),
        box=(210, 210, 210), accent=(255, 255, 255), text=(245, 245, 245),
        text_dim=(140, 140, 140), background=(15, 15, 15),
        good=(220, 220, 220), warn=(180, 180, 180), danger=(255, 255, 255),
        grid=(60, 60, 60), trail=(200, 200, 200),
        categorical=((240, 240, 240), (190, 190, 190), (140, 140, 140), (95, 95, 95)),
    ),
    "medical": Theme(
        name="medical",
        skeleton=(255, 240, 220), joints=(255, 190, 60), hand=(255, 220, 150),
        face_mesh=(190, 170, 140), face_contour=(255, 210, 120), iris=(255, 255, 255),
        box=(255, 200, 90), accent=(255, 200, 90), text=(250, 250, 250),
        text_dim=(160, 160, 170), background=(25, 28, 38),
        good=(140, 230, 140), warn=(90, 200, 255), danger=(80, 90, 255),
        grid=(55, 60, 75), trail=(255, 210, 140),
        categorical=((255, 210, 120), (150, 230, 190), (200, 190, 255),
                     (120, 210, 255), (255, 170, 170)),
    ),
    "neon": Theme(
        name="neon",
        skeleton=(255, 0, 200), joints=(0, 255, 255), hand=(255, 60, 220),
        face_mesh=(160, 40, 160), face_contour=(255, 0, 220), iris=(0, 255, 255),
        box=(255, 0, 200), accent=(0, 255, 255), text=(255, 240, 255),
        text_dim=(170, 110, 170), background=(20, 4, 24),
        good=(120, 255, 190), warn=(0, 220, 255), danger=(120, 0, 255),
        grid=(70, 20, 70), trail=(255, 0, 220),
        categorical=((255, 0, 200), (0, 255, 255), (255, 120, 0),
                     (140, 0, 255), (0, 255, 140), (255, 255, 0)),
    ),
    "arctic": Theme(
        name="arctic",
        skeleton=(255, 220, 150), joints=(255, 255, 220), hand=(255, 200, 140),
        face_mesh=(190, 170, 130), face_contour=(255, 230, 180), iris=(255, 255, 255),
        box=(255, 225, 170), accent=(255, 210, 130), text=(255, 250, 245),
        text_dim=(180, 170, 160), background=(45, 35, 25),
        good=(180, 240, 200), warn=(120, 210, 255), danger=(120, 120, 255),
        grid=(80, 70, 60), trail=(255, 220, 170),
        categorical=((255, 220, 150), (230, 200, 255), (200, 240, 255),
                     (180, 255, 220), (255, 190, 190)),
    ),
}


def theme_names() -> List[str]:
    """Every available theme name, in a stable order."""
    return list(THEMES.keys())


def get_theme(name: str) -> Theme:
    """Look up a theme, falling back to ``cyber`` for unknown names."""
    return THEMES.get(name, THEMES["cyber"])


def next_theme(current: str) -> str:
    """Name of the theme after *current*, wrapping around."""
    names = theme_names()
    try:
        index = names.index(current)
    except ValueError:
        return names[0]
    return names[(index + 1) % len(names)]


def palette_for(count: int, theme_name: str = "cyber") -> List[Color]:
    """*count* visually distinct colors drawn from a theme's categorical set."""
    theme = get_theme(theme_name)
    return [theme.category_color(i) for i in range(max(0, count))]
