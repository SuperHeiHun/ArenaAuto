"""Discover named buttons on the current screen via OCR text search."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

from arena_auto.calibration.coordinate_mapper import CoordinateMapper
from arena_auto.calibration.models import (
    SOURCE_OCR,
    CalibrationDetection,
    CalibrationRegion,
)
from arena_auto.models.config import AppConfig
from arena_auto.models.recognition import OCRBox

logger = logging.getLogger(__name__)

# Guided page-by-page detection: each screen only carries its own buttons.
BUTTON_PAGE_ORDER = ("arena", "prepare", "purchase", "victory", "defeat")

BUTTON_PAGES: Dict[str, Dict[str, object]] = {
    "arena": {
        "title": "① 竞技场选对手页",
        "targets": ("buy_challenge",),
        "hint": (
            "手机停留在竞技场（请选择对手）页面，点「检测当前页面」："
            "购买挑战次数。"
        ),
    },
    "prepare": {
        "title": "② 去获胜页",
        "targets": ("go_win",),
        "hint": (
            "选中对手后进入带「去获胜」的准备页"
            "（testimg/去获胜.jpg），再检测：去获胜。"
        ),
    },
    "purchase": {
        "title": "③ 购买挑战弹窗",
        "targets": ("buy_challenge", "confirm_buy"),
        "hint": (
            "挑战次数为 0 时点对手会弹出购买弹窗，弹窗出现后再检测："
            "购买 / 确认。"
        ),
    },
    "victory": {
        "title": "④ 胜利结算页",
        "targets": ("victory", "exit"),
        "hint": "打完一局进入胜利结算画面后再检测：胜利 / 退出。",
    },
    "defeat": {
        "title": "⑤ 失败结算页",
        "targets": ("defeat", "exit"),
        "hint": "进入失败结算画面后检测：失败 / 退出（可选，可跳过）。",
    },
}

# Unique display order: (button name, home page)
BUTTON_DISPLAY = (
    ("go_win", "prepare"),
    ("exit", "victory"),
    ("buy_challenge", "arena"),
    ("confirm_buy", "purchase"),
    ("victory", "victory"),
    ("defeat", "defeat"),
)


def page_targets(page: str) -> Tuple[str, ...]:
    spec = BUTTON_PAGES.get(page) or BUTTON_PAGES["arena"]
    return tuple(spec["targets"])  # type: ignore[return-value]


def pages_for_button(name: str) -> Tuple[str, ...]:
    return tuple(
        key for key in BUTTON_PAGE_ORDER if name in page_targets(key)
    )


def button_specs(config: AppConfig) -> Dict[str, str]:
    """Ordered mapping of button name -> OCR keyword."""
    profile = config.profile
    return {
        "go_win": profile.go_win_text,
        "exit": profile.exit_text,
        "buy_challenge": profile.buy_text,
        "confirm_buy": profile.confirm_text,
        "victory": profile.victory_text,
        "defeat": profile.defeat_text,
    }


class ButtonDetector:
    """Finds button text on screen and grows it into a clickable region."""

    def __init__(self, config: AppConfig, mapper: CoordinateMapper) -> None:
        self.config = config
        self.mapper = mapper

    def detect(
        self,
        boxes: Sequence[OCRBox],
        names: Optional[Sequence[str]] = None,
    ) -> Dict[str, CalibrationDetection]:
        """Detect buttons; `names` limits detection to a page's targets."""
        wanted = set(names) if names is not None else None
        results: Dict[str, CalibrationDetection] = {}
        pad = self.config.calibration.template_padding
        pad_x, pad_y = self.mapper.pad_to_actual(pad.x, pad.y)
        bounds = (self.mapper.actual_width, self.mapper.actual_height)

        for name, keyword in button_specs(self.config).items():
            if wanted is not None and name not in wanted:
                continue
            if not keyword:
                continue
            box = self._best_box(keyword, boxes)
            if box is None:
                logger.info("[CALIBRATION] 未找到「%s」", keyword)
                continue
            region = CalibrationRegion(
                x=int(box.bbox[0]),
                y=int(box.bbox[1]),
                width=max(1, int(box.bbox[2])),
                height=max(1, int(box.bbox[3])),
            ).expanded(pad_x, pad_y, bounds=bounds)
            confidence = max(0.0, min(1.0, box.confidence + 0.05))
            results[name] = CalibrationDetection(
                name=name,
                region=region,
                click_point=region.center,
                confidence=confidence,
                source=SOURCE_OCR,
                detail=keyword,
            )
            logger.info(
                "[CALIBRATION] 找到「%s」 conf=%.2f region=%s",
                keyword,
                confidence,
                list(region.as_tuple()),
            )
        return results

    @staticmethod
    def _matches(keyword: str, text: str) -> bool:
        text = text.strip()
        if not text:
            return False
        if keyword in text:
            if len(keyword) <= 2:
                stripped = text.strip("！!。.·… 　")
                return stripped == keyword
            return True
        if len(keyword) >= 2 and text in keyword and len(text) >= 2:
            return True
        return False

    @classmethod
    def _best_box(
        cls, keyword: str, boxes: Sequence[OCRBox]
    ) -> Optional[OCRBox]:
        candidates: List[OCRBox] = []
        for box in boxes:
            if cls._matches(keyword, box.text):
                candidates.append(box)
        if not candidates:
            return None
        return max(candidates, key=lambda b: (b.confidence, b.bbox[2] * b.bbox[3]))
