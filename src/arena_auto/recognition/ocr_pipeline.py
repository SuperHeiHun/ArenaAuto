"""Shared full-frame OCR input/output contract for calibration and targeting."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List

import numpy as np

from arena_auto.exceptions import StoppedError
from arena_auto.models.config import OcrSettings
from arena_auto.models.recognition import OCRBox
from arena_auto.recognition.ocr import OCRProvider, preprocess_for_ocr

logger = logging.getLogger(__name__)


@dataclass
class FrameOCRResult:
    boxes: List[OCRBox]
    input_mode: str = "原始截图"


def recognize_frame(provider: OCRProvider, screenshot: np.ndarray,
                    settings: OcrSettings) -> FrameOCRResult:
    """Original image first, preprocessed retry only when empty or failed.

    Both callers receive OCRBox(text, confidence, bbox) in ORIGINAL screenshot
    pixels. Preprocessing may enlarge the image; its boxes must be mapped back
    before any card detection, crop, overlay, or click calculation.
    """
    if screenshot is None or screenshot.size == 0:
        return FrameOCRResult([])
    try:
        boxes = list(provider.recognize(screenshot))
    except StoppedError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("[OCR] 原始截图识别失败: %s", exc)
        boxes = []
    if boxes:
        return FrameOCRResult(boxes)
    try:
        processed = preprocess_for_ocr(screenshot, settings)
        boxes = list(provider.recognize(processed))
        original_h, original_w = screenshot.shape[:2]
        processed_h, processed_w = processed.shape[:2]
        scale_x = original_w / processed_w
        scale_y = original_h / processed_h
        mapped = []
        for box in boxes:
            x, y, w, h = box.bbox
            if w <= 0 or h <= 0:
                continue
            left = max(0, math.floor(x * scale_x))
            top = max(0, math.floor(y * scale_y))
            right = min(original_w, math.ceil((x + w) * scale_x))
            bottom = min(original_h, math.ceil((y + h) * scale_y))
            if right > left and bottom > top:
                mapped.append(OCRBox(box.text, box.confidence,
                                     (left, top, right - left, bottom - top)))
        return FrameOCRResult(mapped, "预处理截图")
    except StoppedError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("[OCR] 预处理截图识别失败: %s", exc)
        return FrameOCRResult([], "预处理截图")
