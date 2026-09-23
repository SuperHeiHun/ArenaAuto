"""OpenCV template matching with optional multi-scale search."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from arena_auto.exceptions import TemplateMissingError
from arena_auto.models.config import TemplateEntry
from arena_auto.models.recognition import DetectionResult

logger = logging.getLogger(__name__)

DEFAULT_SCALES: Sequence[float] = (0.8, 0.9, 1.0, 1.1, 1.2)


class TemplateRecognizer:
    """Load named templates and match them against screenshots."""

    def __init__(
        self,
        templates: Optional[Dict[str, TemplateEntry]] = None,
        base_dir: Optional[Path] = None,
        multi_scale: bool = True,
        scales: Sequence[float] = DEFAULT_SCALES,
    ) -> None:
        self._entries: Dict[str, TemplateEntry] = dict(templates or {})
        self._cache: Dict[str, np.ndarray] = {}
        self._base_dir = Path(base_dir) if base_dir else None
        self.multi_scale = multi_scale
        self.scales = tuple(scales)
        # Same-frame gray reuse: ScreenDetector.find ×N on one BGR frame.
        self._gray_src_id: Optional[int] = None
        self._gray_src: Optional[np.ndarray] = None
        self._gray_cache: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ config
    def register(self, name: str, entry: TemplateEntry) -> None:
        self._entries[name] = entry
        self._cache.pop(name, None)

    def set_entries(self, templates: Dict[str, TemplateEntry]) -> None:
        self._entries = dict(templates)
        self._cache.clear()

    def entry(self, name: str) -> TemplateEntry:
        if name not in self._entries:
            raise TemplateMissingError(f"未配置模板: {name}")
        return self._entries[name]

    def resolve_path(self, name: str) -> Path:
        entry = self.entry(name)
        path = Path(entry.path)
        if not path.is_absolute():
            if self._base_dir is None:
                from arena_auto.paths import get_app_root

                root = get_app_root()
            else:
                root = self._base_dir
            path = root / path
        return path

    def is_loaded(self, name: str) -> bool:
        return name in self._cache

    def missing_templates(self) -> List[str]:
        missing = []
        for name in self._entries:
            try:
                if not self.resolve_path(name).is_file():
                    missing.append(name)
            except TemplateMissingError:
                missing.append(name)
        return sorted(set(missing))

    # ------------------------------------------------------------------- load
    def load(self, name: str, force: bool = False) -> Optional[np.ndarray]:
        if not force and name in self._cache:
            return self._cache[name]
        # Remember missing/invalid paths so detect() does not re-stat every frame.
        miss_key = f"__miss__{name}"
        if not force and self._cache.get(miss_key):
            return None
        path = self.resolve_path(name)
        if not path.is_file():
            logger.warning("模板不存在: %s (%s)", name, path)
            self._cache[miss_key] = True
            return None
        # np.fromfile+imdecode: cv2.imread fails on non-ASCII Windows paths
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception as exc:  # noqa: BLE001
            logger.warning("模板读取失败: %s (%s)", path, exc)
            self._cache[miss_key] = True
            return None
        if image is None:
            logger.warning("模板读取失败: %s", path)
            self._cache[miss_key] = True
            return None
        # all-black / empty templates produce bogus conf=1.0 matches at (0,0)
        if float(image.max()) <= 0:
            logger.warning("模板无效(全黑): %s", path)
            self._cache[miss_key] = True
            return None
        self._cache[name] = image
        return image

    def preload_all(self) -> List[str]:
        """Load every configured template; return list of missing names."""
        missing: List[str] = []
        for name in self._entries:
            if self.load(name) is None:
                missing.append(name)
        return sorted(set(missing))

    # ------------------------------------------------------------------ match
    def _to_gray(self, screenshot: np.ndarray) -> np.ndarray:
        if screenshot.ndim == 2:
            return screenshot
        src_id = id(screenshot)
        if self._gray_src is screenshot and self._gray_src_id == src_id:
            cached = self._gray_cache
            if cached is not None and cached.shape[:2] == screenshot.shape[:2]:
                return cached
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        self._gray_src = screenshot
        self._gray_src_id = src_id
        self._gray_cache = gray
        return gray

    def find(
        self,
        screenshot: np.ndarray,
        template_name: str,
        threshold: Optional[float] = None,
    ) -> DetectionResult:
        """Find template center; returns DetectionResult with confidence."""
        if screenshot is None or screenshot.size == 0:
            return DetectionResult(found=False, template_name=template_name)

        entry = None
        try:
            entry = self.entry(template_name)
        except TemplateMissingError as exc:
            logger.warning("%s", exc)
            return DetectionResult(found=False, template_name=template_name)

        thr = float(entry.threshold if threshold is None else threshold)
        template = self.load(template_name)
        if template is None:
            return DetectionResult(found=False, template_name=template_name)

        best_conf = -1.0
        best_loc: Optional[tuple[int, int]] = None
        best_size: tuple[int, int] = (template.shape[1], template.shape[0])

        # 1.0 first: same-resolution hits can skip the rest of the pyramid.
        candidate_scales: List[float] = [1.0]
        if self.multi_scale:
            candidate_scales = list(
                dict.fromkeys([1.0] + [s for s in self.scales if s != 1.0])
            )

        screen_gray = self._to_gray(screenshot)
        template_gray = template if template.ndim == 2 else cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        for scale in candidate_scales:
            tw = int(round(template_gray.shape[1] * scale))
            th = int(round(template_gray.shape[0] * scale))
            if tw < 5 or th < 5 or tw > screen_gray.shape[1] or th > screen_gray.shape[0]:
                continue
            if scale == 1.0:
                scaled = template_gray
            else:
                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
                scaled = cv2.resize(template_gray, (tw, th), interpolation=interp)
            try:
                match = cv2.matchTemplate(screen_gray, scaled, cv2.TM_CCOEFF_NORMED)
            except cv2.error as exc:
                logger.debug("matchTemplate failed scale=%s: %s", scale, exc)
                continue
            _, max_val, _, max_loc = cv2.minMaxLoc(match)
            if float(max_val) > best_conf:
                best_conf = float(max_val)
                best_loc = (int(max_loc[0]), int(max_loc[1]))
                best_size = (tw, th)
            # Near-perfect hit: other scales cannot meaningfully improve.
            if best_conf >= 0.98 and best_conf >= thr:
                break

        if best_loc is None or best_conf < thr:
            logger.debug(
                "模板匹配失败 %s conf=%.3f threshold=%.3f",
                template_name,
                best_conf,
                thr,
            )
            return DetectionResult(
                found=False,
                confidence=max(best_conf, 0.0),
                template_name=template_name,
            )

        x, y = best_loc
        w, h = best_size
        return DetectionResult(
            found=True,
            confidence=best_conf,
            bbox=(x, y, w, h),
            template_name=template_name,
        )

    def click_point(self, result: DetectionResult) -> tuple[int, int]:
        return result.center
