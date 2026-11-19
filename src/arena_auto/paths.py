"""Path helpers that work in both development and PyInstaller builds."""

from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    """Return True when running from a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False))


def get_app_root() -> Path:
    """Writable application root (exe folder when frozen, project root otherwise)."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    # src/arena_auto/paths.py -> parents[0]=arena_auto, [1]=src, [2]=project root
    return Path(__file__).resolve().parents[2]


def get_resource_path(name: str) -> Path:
    """Resolve a bundled read-only resource path."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidate = Path(meipass) / name
        if candidate.exists():
            return candidate
    return get_app_root() / name


def get_writable_path(name: str) -> Path:
    """Resolve (and create) a writable directory under the app root."""
    path = get_app_root() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_config_path() -> Path:
    """External YAML config path next to the executable / project root."""
    path = get_app_root() / "config" / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_calibrated_config_path() -> Path:
    """Secondary config written by auto-calibration (never clobbers config.yaml)."""
    path = get_app_root() / "config" / "config.calibrated.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_generated_template_dir() -> Path:
    """Writable directory for templates produced by auto-calibration."""
    return get_writable_path("resources/templates/generated")


def get_template_dir() -> Path:
    """User-writable button template directory."""
    return get_writable_path("resources/templates")


def get_log_dir() -> Path:
    return get_writable_path("logs")


def get_debug_dir() -> Path:
    return get_writable_path("debug")


def get_data_dir() -> Path:
    return get_writable_path("data")
