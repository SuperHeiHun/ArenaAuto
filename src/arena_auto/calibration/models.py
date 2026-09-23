"""Data models for the auto-calibration pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

SOURCE_OCR = "ocr"
SOURCE_TEMPLATE = "template"
SOURCE_STRUCTURE = "structure"
SOURCE_MANUAL = "manual"
SOURCE_DEFAULT = "default"
SOURCE_CALIBRATED = "calibrated"

VALID_SOURCES = (
    SOURCE_OCR,
    SOURCE_TEMPLATE,
    SOURCE_STRUCTURE,
    SOURCE_MANUAL,
    SOURCE_DEFAULT,
    SOURCE_CALIBRATED,
)


@dataclass
class CalibrationPoint:
    x: int
    y: int

    def as_tuple(self) -> Tuple[int, int]:
        return (int(self.x), int(self.y))

    @classmethod
    def from_tuple(cls, value: Tuple[int, int]) -> "CalibrationPoint":
        return cls(x=int(value[0]), y=int(value[1]))


@dataclass
class CalibrationRegion:
    x: int
    y: int
    width: int
    height: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (int(self.x), int(self.y), int(self.width), int(self.height))

    @classmethod
    def from_tuple(cls, value: Tuple[int, int, int, int]) -> "CalibrationRegion":
        return cls(
            x=int(value[0]),
            y=int(value[1]),
            width=int(value[2]),
            height=int(value[3]),
        )

    @property
    def center(self) -> CalibrationPoint:
        return CalibrationPoint(
            x=int(self.x + self.width // 2),
            y=int(self.y + self.height // 2),
        )

    def expanded(self, pad_x: int, pad_y: int, bounds: Optional[Tuple[int, int]] = None) -> "CalibrationRegion":
        x = self.x - int(pad_x)
        y = self.y - int(pad_y)
        w = self.width + 2 * int(pad_x)
        h = self.height + 2 * int(pad_y)
        if bounds is not None:
            screen_w, screen_h = bounds
            x = max(0, x)
            y = max(0, y)
            w = max(1, min(w, screen_w - x))
            h = max(1, min(h, screen_h - y))
        else:
            x = max(0, x)
            y = max(0, y)
            w = max(1, w)
            h = max(1, h)
        return CalibrationRegion(x=x, y=y, width=w, height=h)

    def contains(self, px: int, py: int) -> bool:
        return (
            self.x <= px <= self.x + self.width
            and self.y <= py <= self.y + self.height
        )


@dataclass
class CalibrationDetection:
    name: str
    region: CalibrationRegion
    click_point: CalibrationPoint
    confidence: float
    source: str = SOURCE_OCR
    value: Optional[int] = None
    detail: str = ""

    @property
    def is_low_confidence(self) -> bool:
        return self.confidence < 0.75

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "region": list(self.region.as_tuple()),
            "click": list(self.click_point.as_tuple()),
            "confidence": round(float(self.confidence), 4),
            "source": self.source,
            "value": self.value,
            "detail": self.detail,
        }


@dataclass
class CalibrationBundle:
    """Everything discovered from one arena screenshot (actual-pixel coords)."""

    screen_width: int = 0
    screen_height: int = 0
    reference_width: int = 1920
    reference_height: int = 1080
    arena_detected: bool = False
    arena_confidence: float = 0.0
    screen_kind: str = "unknown"
    ocr_box_count: int = 0
    opponents: List[CalibrationDetection] = field(default_factory=list)
    challenge: Optional[CalibrationDetection] = None
    buttons: Dict[str, CalibrationDetection] = field(default_factory=dict)
    generated_templates: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def opponents_found(self) -> int:
        return sum(
            1
            for d in self.opponents
            if d.source in (SOURCE_OCR, SOURCE_STRUCTURE) and d.region.width > 0
        )

    @property
    def low_confidence_items(self) -> List[CalibrationDetection]:
        items = [d for d in self.opponents if d.region.width > 0 and d.confidence < 0.75]
        if self.challenge is not None and self.challenge.confidence < 0.75:
            items.append(self.challenge)
        items.extend(
            d for d in self.buttons.values() if d.region.width > 0 and d.confidence < 0.75
        )
        return items
