"""
Configuration system for LookThePerson.

Settings resolve in a fixed precedence order — later sources win:

1. Defaults baked into :class:`Config`.
2. ``~/.looktheperson/config.json`` (user config).
3. ``./looktheperson.json`` in the working directory (project config).
4. A named profile inside either file.
5. Explicit CLI flags.

Everything is plain JSON and plain dataclasses, so nothing here needs numpy,
OpenCV or a camera to be exercised.
"""

from __future__ import annotations

import json
import os
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional

__all__ = [
    "CameraConfig",
    "DetectionConfig",
    "DisplayConfig",
    "GestureConfig",
    "RecordingConfig",
    "AnalyticsConfig",
    "Config",
    "USER_CONFIG_PATH",
    "PROJECT_CONFIG_NAME",
    "load_config",
    "save_config",
    "list_profiles",
    "deep_merge",
]

USER_CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".looktheperson")
USER_CONFIG_PATH = os.path.join(USER_CONFIG_DIR, "config.json")
PROJECT_CONFIG_NAME = "looktheperson.json"


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    """Capture device and frame acquisition settings."""

    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True
    buffer_size: int = 1
    source: Optional[str] = None  # video file or image folder instead of a camera
    loop_source: bool = True
    auto_reconnect: bool = True
    reconnect_delay: float = 1.5
    max_reconnect_attempts: int = 5


@dataclass
class DetectionConfig:
    """Which models run and how confident they must be."""

    max_poses: int = 4
    max_hands: int = 2
    max_faces: int = 2
    pose_confidence: float = 0.35
    hand_confidence: float = 0.45
    face_confidence: float = 0.40
    object_confidence: float = 0.35
    max_objects: int = 10
    segmentation: bool = True
    # Run heavy models every Nth frame; 1 means every frame.
    object_stride: int = 2
    face_mesh_stride: int = 1
    smoothing: bool = True
    smoothing_min_cutoff: float = 1.0
    smoothing_beta: float = 0.007


@dataclass
class DisplayConfig:
    """Window, HUD and theming."""

    fullscreen: bool = True
    headless: bool = False        # run without a window (benchmarks, CI, servers)
    max_frames: int = 0           # stop after N frames; 0 runs until quit
    theme: str = "cyber"
    show_hud: bool = True
    show_help: bool = False
    show_grid: bool = False
    show_fps_graph: bool = False
    show_toasts: bool = True
    hud_scale: float = 1.0
    skeleton_style: str = "classic"
    skeleton_thickness: int = 3
    landmark_radius: int = 5
    show_landmark_ids: bool = False
    target_window_title: str = "LookThePerson - @nostraxiten"


@dataclass
class GestureConfig:
    """Gesture sensitivity and system-control permissions."""

    enabled: bool = True
    stable_seconds: float = 0.3
    cooldown_seconds: float = 1.0
    clap_distance: float = 0.11
    allow_app_control: bool = True
    allow_calculator: bool = True
    allow_browser: bool = True
    allow_media_keys: bool = False
    allow_mouse_control: bool = False
    # gesture name -> action name, applied on top of the built-in bindings
    bindings: Dict[str, str] = field(default_factory=dict)


@dataclass
class RecordingConfig:
    """Screenshot and video capture behaviour."""

    output_dir: str = ""
    video_fps: float = 20.0
    video_codec: str = "XVID"
    video_extension: str = "avi"
    screenshot_format: str = "png"
    burst_count: int = 5
    burst_interval: float = 0.4
    include_hud: bool = True
    auto_stop_seconds: float = 0.0  # 0 disables the limit


@dataclass
class AnalyticsConfig:
    """Metrics, session logging and export."""

    enabled: bool = True
    track_posture: bool = True
    track_reps: bool = True
    track_blinks: bool = True
    track_heatmap: bool = False
    session_log: bool = False
    export_format: str = "json"
    export_dir: str = ""
    user_height_cm: float = 170.0
    user_weight_kg: float = 70.0


@dataclass
class Config:
    """Root configuration object."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    mode: str = "full"
    profile: str = "default"
    verbose: bool = False

    # -- Serialization ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict view suitable for ``json.dump``."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Build a Config from a (possibly partial) dict, ignoring unknowns."""
        return _build_dataclass(cls, data or {})

    def merged_with(self, overrides: Dict[str, Any]) -> "Config":
        """Return a new Config with *overrides* applied on top."""
        return Config.from_dict(deep_merge(self.to_dict(), overrides))

    def get(self, dotted_path: str, default: Any = None) -> Any:
        """Read a nested value, e.g. ``cfg.get("camera.width")``."""
        node: Any = self
        for part in dotted_path.split("."):
            if is_dataclass(node) and hasattr(node, part):
                node = getattr(node, part)
            elif isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def set(self, dotted_path: str, value: Any) -> bool:
        """Write a nested value. Returns False when the path does not exist."""
        parts = dotted_path.split(".")
        node: Any = self
        for part in parts[:-1]:
            if is_dataclass(node) and hasattr(node, part):
                node = getattr(node, part)
            elif isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return False
        leaf = parts[-1]
        if is_dataclass(node) and hasattr(node, leaf):
            setattr(node, leaf, value)
            return True
        if isinstance(node, dict):
            node[leaf] = value
            return True
        return False

    def describe(self) -> List[str]:
        """Flat ``"section.key = value"`` listing, handy for the debug HUD."""
        lines: List[str] = []

        def walk(prefix: str, data: Dict[str, Any]) -> None:
            for key in sorted(data):
                value = data[key]
                path = f"{prefix}{key}"
                if isinstance(value, dict):
                    walk(f"{path}.", value)
                else:
                    lines.append(f"{path} = {value}")

        walk("", self.to_dict())
        return lines


# ---------------------------------------------------------------------------
# Dataclass reconstruction
# ---------------------------------------------------------------------------

def _build_dataclass(cls, data: Dict[str, Any]):
    """
    Instantiate dataclass *cls* from *data*, recursing into nested dataclasses.

    Unknown keys are ignored rather than raising, so a config written by a
    newer version of the app still loads in an older one.
    """
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        # ``from __future__ import annotations`` turns ``f.type`` into a string,
        # so nested sections are identified through the field's default instead.
        default = _default_for(f)
        if is_dataclass(default) and isinstance(value, dict):
            kwargs[f.name] = _build_dataclass(type(default), value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def _default_for(f):
    """Best-effort default value for a dataclass field, or None."""
    if f.default_factory is not MISSING:  # type: ignore[misc]
        try:
            return f.default_factory()
        except TypeError:
            return None
    return None if f.default is MISSING else f.default


# ---------------------------------------------------------------------------
# Merging and IO
# ---------------------------------------------------------------------------

def deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge *overrides* into *base*, returning a new dict.

    Nested dicts merge key by key; every other type is replaced outright.
    ``None`` values in *overrides* are skipped so unset CLI flags do not wipe
    configured values.
    """
    out = dict(base)
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_json(path: str) -> Dict[str, Any]:
    """Load a JSON file, returning ``{}`` when missing or malformed."""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[config] No pude leer {path}: {exc}", flush=True)
        return {}


def _extract_profile(data: Dict[str, Any], profile: str) -> Dict[str, Any]:
    """
    Split a config document into base settings plus the selected profile.

    Profiles live under a top-level ``"profiles"`` key:

    ```json
    {"camera": {"fps": 30}, "profiles": {"gym": {"mode": "workout"}}}
    ```
    """
    profiles = data.get("profiles")
    base = {k: v for k, v in data.items() if k != "profiles"}
    if isinstance(profiles, dict) and profile and profile in profiles:
        overlay = profiles[profile]
        if isinstance(overlay, dict):
            return deep_merge(base, overlay)
    return base


def list_profiles(paths: Optional[List[str]] = None) -> List[str]:
    """Names of every profile defined across the config files."""
    names: List[str] = []
    for path in paths or [USER_CONFIG_PATH, os.path.join(os.getcwd(), PROJECT_CONFIG_NAME)]:
        profiles = _read_json(path).get("profiles")
        if isinstance(profiles, dict):
            names.extend(k for k in profiles if k not in names)
    return names


def load_config(
    cli_overrides: Optional[Dict[str, Any]] = None,
    profile: str = "default",
    extra_path: Optional[str] = None,
) -> Config:
    """
    Build the effective configuration from every source.

    Args:
        cli_overrides: nested dict of explicit command-line values.
        profile: profile name to overlay from the config files.
        extra_path: optional additional config file, applied after the
            standard locations and before the CLI overrides.
    """
    merged: Dict[str, Any] = Config().to_dict()

    for path in (USER_CONFIG_PATH, os.path.join(os.getcwd(), PROJECT_CONFIG_NAME), extra_path):
        if not path:
            continue
        merged = deep_merge(merged, _extract_profile(_read_json(path), profile))

    merged = deep_merge(merged, cli_overrides or {})
    merged["profile"] = profile
    return Config.from_dict(merged)


def save_config(config: Config, path: str = USER_CONFIG_PATH, profile: Optional[str] = None) -> str:
    """
    Persist *config* to disk, preserving any profiles already stored there.

    When *profile* is given the settings are written into that profile instead
    of the document root. Returns the path written.
    """
    existing = _read_json(path)
    payload = config.to_dict()

    if profile:
        profiles = existing.get("profiles")
        if not isinstance(profiles, dict):
            profiles = {}
        profiles[profile] = payload
        existing["profiles"] = profiles
        document = existing
    else:
        profiles = existing.get("profiles")
        document = payload
        if isinstance(profiles, dict):
            document["profiles"] = profiles

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path
