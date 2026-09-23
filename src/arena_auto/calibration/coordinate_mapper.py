"""Actual-screenshot <-> reference-resolution coordinate mapping."""

from __future__ import annotations

from typing import Tuple

from arena_auto.calibration.models import CalibrationPoint, CalibrationRegion


class CoordinateMapper:
    """Bidirectional mapper between device pixels and reference coordinates."""

    def __init__(
        self,
        reference_width: int = 1920,
        reference_height: int = 1080,
        actual_width: int = 0,
        actual_height: int = 0,
    ) -> None:
        self.reference_width = max(1, int(reference_width))
        self.reference_height = max(1, int(reference_height))
        self.actual_width = int(actual_width)
        self.actual_height = int(actual_height)

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

    def pad_to_actual(self, pad_x: int, pad_y: int) -> Tuple[int, int]:
        """Convert a reference-resolution padding value into actual pixels."""
        return (
            max(0, int(round(pad_x * self.scale_x))),
            max(0, int(round(pad_y * self.scale_y))),
        )

    def actual_point_to_reference(self, point: CalibrationPoint) -> Tuple[int, int]:
        if not self.actual_width or not self.actual_height:
            return point.as_tuple()
        return (
            int(round(point.x * self.reference_width / self.actual_width)),
            int(round(point.y * self.reference_height / self.actual_height)),
        )

    def actual_region_to_reference(self, region: CalibrationRegion) -> Tuple[int, int, int, int]:
        if not self.actual_width or not self.actual_height:
            return region.as_tuple()
        rx = region.x * self.reference_width / self.actual_width
        ry = region.y * self.reference_height / self.actual_height
        rw = region.width * self.reference_width / self.actual_width
        rh = region.height * self.reference_height / self.actual_height
        return (
            int(round(rx)),
            int(round(ry)),
            max(1, int(round(rw))),
            max(1, int(round(rh))),
        )

    def reference_point_to_actual(self, x: int, y: int) -> CalibrationPoint:
        return CalibrationPoint(
            x=int(round(x * self.scale_x)),
            y=int(round(y * self.scale_y)),
        )

    def reference_region_to_actual(
        self, region: Tuple[int, int, int, int]
    ) -> CalibrationRegion:
        x, y, w, h = region
        return CalibrationRegion(
            x=int(round(x * self.scale_x)),
            y=int(round(y * self.scale_y)),
            width=max(1, int(round(w * self.scale_x))),
            height=max(1, int(round(h * self.scale_y))),
        )
