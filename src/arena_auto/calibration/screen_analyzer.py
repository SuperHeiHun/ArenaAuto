"""Full-screen OCR analysis and arena page assessment for calibration."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from arena_auto.models.config import AppConfig
from arena_auto.models.recognition import OCRBox
from arena_auto.models.state import ScreenKind
from arena_auto.recognition.detector import ScreenDetector
from arena_auto.recognition.ocr import OCRProvider, preprocess_for_ocr

logger = logging.getLogger(__name__)


class _CachedOCR:
    """OCR stand-in that replays a previous full-frame recognition result."""

    def __init__(self, boxes: List[OCRBox]) -> None:
        self.boxes = list(boxes)

    def recognize(self, image: np.ndarray) -> List[OCRBox]:
        return list(self.boxes)


class ScreenAnalyzer:
    """Runs one full-frame OCR pass and assesses the arena page."""

    def __init__(self, ocr: OCRProvider, config: AppConfig) -> None:
        self.ocr = ocr
        self.config = config

    def recognize(self, screenshot: np.ndarray) -> List[OCRBox]:
        if screenshot is None or screenshot.size == 0:
            return []
        boxes: List[OCRBox] = []
        try:
            boxes = self.ocr.recognize(screenshot)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[CALIBRATION] 全屏 OCR 失败: %s", exc)
            boxes = []
        if not boxes:
            try:
                processed = preprocess_for_ocr(screenshot, self.config.ocr)
                boxes = self.ocr.recognize(processed)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[CALIBRATION] 预处理 OCR 失败: %s", exc)
                boxes = []
        logger.info("[CALIBRATION] OCR detected %s text regions", len(boxes))
        return boxes

    def detect_screen(
        self, screenshot: np.ndarray, boxes: List[OCRBox]
    ) -> Tuple[ScreenKind, bool, float]:
        """Classify the screen using cached OCR boxes (no second OCR pass)."""
        detector = ScreenDetector(
            _CachedOCR(boxes),
            self.config,
            templates=None,
        )
        kind = detector.detect(screenshot)
        confidence = self._arena_confidence(boxes) if kind == ScreenKind.ARENA else 0.0
        return kind, kind == ScreenKind.ARENA, confidence

    def _arena_confidence(self, boxes: List[OCRBox]) -> float:
        blob = " ".join(b.text for b in boxes)
        profile = self.config.profile
        groups = (
            profile.arena_title_keywords,
            profile.arena_select_keywords,
            profile.challenge_keywords,
        )
        total = sum(len([k for k in g if k]) for g in groups)
        if total == 0:
            return 0.0
        hits = 0
        for group in groups:
            for keyword in group:
                if keyword and keyword in blob:
                    hits += 1
        return min(1.0, hits / float(max(1, total)) + 0.25 * min(hits, 2))

    def draw_ocr_overlay(
        self, screenshot: np.ndarray, boxes: List[OCRBox]
    ) -> np.ndarray:
        overlay = screenshot.copy()
        for box in boxes:
            x, y, w, h = box.bbox
            color = (0, 200, 255) if box.confidence >= 0.5 else (0, 0, 220)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)
            label = f"{box.text[:16]} {box.confidence:.2f}"
            cv2.putText(
                overlay,
                label,
                (x, max(12, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
        return overlay
