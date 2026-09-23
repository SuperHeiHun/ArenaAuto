"""Recognition-related data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

Point = Tuple[int, int]
Region = Tuple[int, int, int, int]  # x, y, width, height


@dataclass(frozen=True)
class OCRBox:
    """Single OCR text detection."""

    text: str
    confidence: float
    bbox: Region  # x, y, w, h


@dataclass(frozen=True)
class DetectionResult:
    """Template / detector match result."""

    found: bool
    confidence: float = 0.0
    bbox: Region = (0, 0, 0, 0)
    template_name: str = ""

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def width(self) -> int:
        return self.bbox[2]

    @property
    def height(self) -> int:
        return self.bbox[3]

    @property
    def center(self) -> Point:
        return (self.bbox[0] + self.bbox[2] // 2, self.bbox[1] + self.bbox[3] // 2)


@dataclass
class Opponent:
    """One arena opponent slot."""

    id: int
    power: Optional[int] = None
    power_region: Region = (0, 0, 100, 50)
    click_position: Point = (0, 0)


@dataclass(frozen=True)
class ChallengeCount:
    """Remaining / maximum challenge count."""

    current: int
    maximum: int

    def __str__(self) -> str:
        return f"{self.current}/{self.maximum}"


@dataclass
class OpponentScan:
    """Result of scanning all opponent power ROIs."""

    opponents: list[Opponent] = field(default_factory=list)
    all_valid: bool = False
