"""ROI generation helpers: power regions and challenge-count region."""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

from arena_auto.calibration.coordinate_mapper import CoordinateMapper
from arena_auto.calibration.models import (
    SOURCE_OCR,
    CalibrationDetection,
    CalibrationRegion,
)
from arena_auto.models.config import AppConfig
from arena_auto.models.recognition import OCRBox
from arena_auto.recognition.ocr import parse_challenge_count

logger = logging.getLogger(__name__)

_CHALLENGE_KEYWORDS = ("可挑战", "挑战次数", "次数")


class RoiDetector:
    """Expands OCR bounding boxes into calibrated ROIs."""

    def __init__(self, config: AppConfig, mapper: CoordinateMapper) -> None:
        self.config = config
        self.mapper = mapper

    def power_region_from_bbox(
        self, bbox: Tuple[int, int, int, int]
    ) -> CalibrationRegion:
        region = CalibrationRegion(
            x=int(bbox[0]),
            y=int(bbox[1]),
            width=max(1, int(bbox[2])),
            height=max(1, int(bbox[3])),
        )
        pad = self.config.calibration.power_region_padding
        px, py = self.mapper.pad_to_actual(pad.x, pad.y)
        return region.expanded(
            px, py, bounds=(self.mapper.actual_width, self.mapper.actual_height)
        )

    def detect_challenge(
        self,
        boxes: Sequence[OCRBox],
        slot_boxes: Optional[Sequence[CalibrationRegion]] = None,
    ) -> Optional[CalibrationDetection]:
        """Find `current / maximum` region, preferring boxes near keywords."""
        exclude = list(slot_boxes or [])
        candidates: List[Tuple[OCRBox, float]] = []
        for box in boxes:
            parsed = parse_challenge_count(box.text)
            if parsed is None:
                continue
            if self._inside_slots(box.bbox, exclude):
                continue
            bonus = 0.15 if self._near_keyword(box, boxes) else 0.0
            candidates.append((box, min(1.0, box.confidence + bonus)))

        if not candidates:
            logger.info("[CALIBRATION] 未找到挑战次数")
            return None

        box, confidence = max(candidates, key=lambda item: item[1])
        pad = self.config.calibration.power_region_padding
        px, py = self.mapper.pad_to_actual(pad.x, pad.y)
        region = CalibrationRegion(
            x=int(box.bbox[0]),
            y=int(box.bbox[1]),
            width=max(1, int(box.bbox[2])),
            height=max(1, int(box.bbox[3])),
        ).expanded(
            px, py, bounds=(self.mapper.actual_width, self.mapper.actual_height)
        )
        parsed = parse_challenge_count(box.text)
        detail = f"{parsed[0]}/{parsed[1]}" if parsed else box.text
        logger.info("[CALIBRATION] 找到挑战次数 %s conf=%.2f", detail, confidence)
        return CalibrationDetection(
            name="challenge_count",
            region=region,
            click_point=region.center,
            confidence=confidence,
            source=SOURCE_OCR,
            detail=detail,
        )

    @staticmethod
    def _inside_slots(
        bbox: Tuple[int, int, int, int], slots: Sequence[CalibrationRegion]
    ) -> bool:
        cx = bbox[0] + bbox[2] // 2
        cy = bbox[1] + bbox[3] // 2
        return any(slot.contains(cx, cy) for slot in slots)

    @staticmethod
    def _near_keyword(box: OCRBox, boxes: Sequence[OCRBox]) -> bool:
        bx, by, bw, bh = box.bbox
        bcx, bcy = bx + bw // 2, by + bh // 2
        reach = max(bw, bh) * 3 + 40
        for other in boxes:
            if other is box:
                continue
            ox, oy, ow, oh = other.bbox
            ocx, ocy = ox + ow // 2, oy + oh // 2
            if abs(ocx - bcx) > reach or abs(ocy - bcy) > reach:
                continue
            if any(kw in other.text for kw in _CHALLENGE_KEYWORDS):
                return True
        return False
