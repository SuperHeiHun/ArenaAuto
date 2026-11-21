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


class FrameOCRCache:
    """Share OCR only for the exact same immutable frame, never stable ROI reads.

    Retains one frame. A fresh screenshot, config reload or device action clears
    the evidence; pixel similarity is deliberately not used as a cache key.
    """

    def __init__(self, provider: OCRProvider) -> None:
        self.provider = provider
        self.clear()

    def clear(self) -> None:
        self._image = None
        self._raw = None
        self._result = None

    def peek(self, image: np.ndarray):
        """Read already computed same-frame evidence without starting OCR."""
        if image is not self._image:
            return None
        if self._result is not None:
            return list(self._result.boxes)
        return list(self._raw) if self._raw is not None else None

    def raw(self, image: np.ndarray) -> List[OCRBox]:
        if image is not self._image:
            self.clear()
            self._image = image
        if self._raw is None:
            self._raw = list(self.provider.recognize(image))
        return list(self._raw)

    def recognize(self, image: np.ndarray, settings: OcrSettings) -> FrameOCRResult:
        try:
            boxes = self.raw(image)
        except StoppedError:
            raise
        except Exception as exc:
            logger.warning("[OCR] 原始截图识别失败: %s", exc)
            self._image, self._raw = image, []
            boxes = []
        if boxes:
            return FrameOCRResult(boxes)
        if self._result is None:
            # Retry preprocessing through the existing shared pipeline. The raw
            # call is already cached; only the processed retry performs OCR.
            cache = self

            class CachedRaw:
                def recognize(self, frame):
                    return cache.raw(frame)

            self._result = recognize_frame(CachedRaw(), image, settings)
            # The retry changes the raw cache's frame. Restore the original
            # frame and empty raw result so future consumers share the retry.
            self._image, self._raw = image, boxes
        return FrameOCRResult(list(self._result.boxes), self._result.input_mode)


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
