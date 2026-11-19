"""Save categorized debug screenshots and failed OCR ROIs."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from arena_auto.models.config import DebugSettings
from arena_auto.paths import get_debug_dir

logger = logging.getLogger(__name__)

CATEGORIES = (
    "arena",
    "prepare",
    "battle",
    "result",
    "purchase",
    "failed",
    "recovery",
    "debug",
    "calibration",
)


class DebugRecorder:
    """Writes timestamped PNGs under debug/<category>/."""

    def __init__(self, settings: Optional[DebugSettings] = None, base_dir: Optional[Path] = None):
        self.settings = settings or DebugSettings()
        self.base_dir = Path(base_dir) if base_dir else self._resolve_base_dir()
        self._ensure_dirs()

    def _resolve_base_dir(self) -> Path:
        configured = (self.settings.directory or "debug").strip()
        path = Path(configured)
        if not path.is_absolute():
            # keep debug dir next to app root
            from arena_auto.paths import get_app_root

            path = get_app_root() / path
        return path

    def _ensure_dirs(self) -> None:
        if not self.settings.enabled:
            return
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            for name in CATEGORIES:
                (self.base_dir / name).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("无法创建调试目录 %s: %s", self.base_dir, exc)

    @property
    def enabled(self) -> bool:
        return bool(self.settings.enabled and self.settings.save_screenshots)

    @property
    def failed_enabled(self) -> bool:
        return bool(
            self.settings.enabled and self.settings.save_failed_recognition
        )

    def _timestamp(self) -> str:
        return time.strftime("%Y%m%d_%H%M%S")

    def save(
        self,
        image: Optional[np.ndarray],
        category: str = "debug",
        suffix: str = "",
        tag: Optional[str] = None,
    ) -> Optional[Path]:
        """Save a full screenshot into category folder. Returns path or None."""
        if image is None or not self.enabled:
            return None
        if category not in CATEGORIES:
            category = "debug"
        name = self._timestamp()
        if tag:
            name = f"{name}_{tag}"
        if suffix:
            name = f"{name}_{suffix}"
        path = self.base_dir / category / f"{name}.png"
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            (self.base_dir / category).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), image)
            logger.debug("调试截图: %s", path)
            return path
        except Exception as exc:  # noqa: BLE001
            logger.warning("保存调试截图失败: %s", exc)
            return None

    def save_failed(
        self,
        image: Optional[np.ndarray],
        name: str = "failed",
        roi: Optional[np.ndarray] = None,
    ) -> Optional[Path]:
        """Save failure evidence (full frame and optional ROI)."""
        if image is None or not self.failed_enabled:
            return None
        path = self.save(image, category="failed", tag=name)
        if roi is not None:
            roi_path = self.base_dir / "failed" / f"{self._timestamp()}_{name}_roi.png"
            try:
                cv2.imwrite(str(roi_path), roi)
            except Exception as exc:  # noqa: BLE001
                logger.warning("保存 ROI 失败: %s", exc)
        return path

    def save_roi(
        self,
        roi: Optional[np.ndarray],
        name: str,
        category: str = "failed",
    ) -> Optional[Path]:
        """Save a cropped ROI (e.g. power_1.png)."""
        if roi is None or not self.settings.enabled:
            return None
        if category not in CATEGORIES:
            category = "failed"
        path = self.base_dir / category / f"{self._timestamp()}_{name}.png"
        try:
            (self.base_dir / category).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), roi)
            return path
        except Exception as exc:  # noqa: BLE001
            logger.warning("保存 ROI 失败: %s", exc)
            return None

    def open_directory(self) -> str:
        return str(self.base_dir)
