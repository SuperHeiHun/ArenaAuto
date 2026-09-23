"""Crop discovered buttons into generated template PNGs (never overwrites user templates)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from arena_auto.calibration.coordinate_mapper import CoordinateMapper
from arena_auto.calibration.models import CalibrationRegion
from arena_auto.models.config import AppConfig
from arena_auto.paths import get_generated_template_dir

logger = logging.getLogger(__name__)


class TemplateGenerator:
    """Saves padded button crops under resources/templates/generated/."""

    def __init__(
        self,
        config: AppConfig,
        mapper: CoordinateMapper,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.mapper = mapper
        self.output_dir = Path(output_dir) if output_dir else get_generated_template_dir()

    def generate(
        self, screenshot: np.ndarray, name: str, region: CalibrationRegion
    ) -> Optional[Path]:
        if screenshot is None or screenshot.size == 0:
            return None
        h_img, w_img = screenshot.shape[:2]
        pad = self.config.calibration.template_padding
        pad_x, pad_y = self.mapper.pad_to_actual(pad.x, pad.y)
        x0 = max(0, region.x - pad_x)
        y0 = max(0, region.y - pad_y)
        x1 = min(w_img, region.x + region.width + pad_x)
        y1 = min(h_img, region.y + region.height + pad_y)
        if x1 - x0 < 4 or y1 - y0 < 4:
            logger.warning("[CALIBRATION] 按钮区域过小，跳过模板: %s", name)
            return None
        crop = screenshot[y0:y1, x0:x1].copy()
        # 全黑/近似纯色裁剪会导致运行时模板在 (0,0) 误匹配 conf=1.0
        if crop.size == 0 or float(crop.max()) <= 0 or float(crop.std()) < 1.0:
            logger.warning(
                "[CALIBRATION] 按钮裁剪无效(全黑/纯色)，跳过模板: %s", name
            )
            return None
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f"{name}.png"
            ok, buf = cv2.imencode(".png", crop)
            if not ok:
                logger.warning("[CALIBRATION] 模板编码失败: %s", path)
                return None
            buf.tofile(str(path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[CALIBRATION] 模板保存异常 %s: %s", name, exc)
            return None
        logger.info("[CALIBRATION] 生成模板 %s -> %s", name, path)
        return path

    def generate_all(
        self, screenshot: np.ndarray, regions: Dict[str, CalibrationRegion]
    ) -> Dict[str, str]:
        generated: Dict[str, str] = {}
        if not self.config.calibration.generate_templates:
            return generated
        for name, region in regions.items():
            path = self.generate(screenshot, name, region)
            if path is not None:
                try:
                    rel = path.relative_to(self.output_dir.parent.parent.parent)
                    generated[name] = str(rel).replace("\\", "/")
                except ValueError:
                    generated[name] = str(path)
        if generated:
            logger.info(
                "[CALIBRATION] 自动生成模板: %s", ", ".join(sorted(generated))
            )
        return generated
