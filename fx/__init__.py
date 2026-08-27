"""
Visual effects for LookThePerson.

Three layers, kept separate because they compose differently:

* ``fx.filters`` — whole-frame image transformations (thermal, cartoon, ASCII).
* ``fx.background`` — segmentation-driven subject/background separation.
* ``fx.overlays`` — things drawn on top (skeleton styles, trails, particles,
  heat maps, instruments).

These modules require OpenCV and numpy; the pure-logic packages (``core`` and
``analytics``) deliberately do not.
"""

from fx.background import (
    blur_background,
    background_color,
    cutout,
    hologram,
    outline_person,
    prepare_mask,
    privacy_blur_region,
    replace_background,
    silhouette,
    spotlight,
)
from fx.filters import FILTERS, apply_filter, filter_names, next_filter
from fx.overlays import (
    MotionHeatmap,
    ParticleSystem,
    draw_progress_ring,
    draw_radar,
    draw_skeleton_styled,
    draw_trail,
    SKELETON_STYLES,
)

__all__ = [
    "apply_filter",
    "filter_names",
    "next_filter",
    "FILTERS",
    "prepare_mask",
    "blur_background",
    "replace_background",
    "background_color",
    "silhouette",
    "cutout",
    "spotlight",
    "outline_person",
    "hologram",
    "privacy_blur_region",
    "MotionHeatmap",
    "ParticleSystem",
    "draw_trail",
    "draw_radar",
    "draw_progress_ring",
    "draw_skeleton_styled",
    "SKELETON_STYLES",
]
