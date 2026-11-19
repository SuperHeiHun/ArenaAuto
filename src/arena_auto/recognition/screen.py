"""Screen geometry helpers and reference-resolution scaling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from arena_auto.models.recognition import Point, Region


def crop_region(image: np.ndarray, region: Region) -> np.ndarray:
    """Crop ROI given as (x, y, width, height) with bounds clamping."""
    if image is None:
        raise ValueError("image is None")
    height, width = image.shape[:2]
    x, y, w, h = region
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(width, int(x + w))
    y1 = min(height, int(y + h))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"ROI 越界或无效: {region} image={width}x{height}")
    return image[y0:y1, x0:x1].copy()


@dataclass
class CoordinateScaler:
    """Scale configured reference coordinates to actual device pixels."""

    reference_width: int = 1920
    reference_height: int = 1080
    actual_width: int = 0
    actual_height: int = 0

    def update_actual(self, width: int, height: int) -> None:
        self.actual_width = int(width)
        self.actual_height = int(height)

    @property
    def scale_x(self) -> float:
        if self.actual_width and self.reference_width:
            return self.actual_width / float(self.reference_width)
        return 1.0

    @property
    def scale_y(self) -> float:
        if self.actual_height and self.reference_height:
            return self.actual_height / float(self.reference_height)
        return 1.0

    def scale_point(self, x: int, y: int) -> Point:
        return (int(round(x * self.scale_x)), int(round(y * self.scale_y)))

    def scale_region(self, region: Region) -> Region:
        x, y, w, h = region
        sx, sy = self.scale_x, self.scale_y
        return (
            int(round(x * sx)),
            int(round(y * sy)),
            max(1, int(round(w * sx))),
            max(1, int(round(h * sy))),
        )

    def maybe_update_from_image(self, image: Optional[np.ndarray]) -> None:
        if image is None:
            return
        height, width = image.shape[:2]
        if (width, height) != (self.actual_width, self.actual_height):
            self.update_actual(width, height)
