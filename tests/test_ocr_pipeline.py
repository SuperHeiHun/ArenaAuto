"""Calibration and automatic targeting share one OCR and pixel-coordinate contract."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from arena_auto.calibration.screen_analyzer import ScreenAnalyzer
from arena_auto.exceptions import OCRProviderError, StoppedError
from arena_auto.models.config import AppConfig, OcrSettings
from arena_auto.models.recognition import OCRBox
from arena_auto.recognition.auto_targeting import AutoTargeting
from arena_auto.recognition.ocr_engine import OcrEngine
from arena_auto.recognition.ocr_pipeline import recognize_frame


def test_both_entrypoints_use_same_raw_format_and_provider():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    boxes = [OCRBox("战斗力：67,499", 0.9, (120, 20, 70, 25))]
    provider = MagicMock()
    provider.recognize.return_value = boxes
    config = AppConfig()
    calibration = ScreenAnalyzer(provider, config)
    targeting = AutoTargeting(provider, config)
    assert calibration.recognize(image) == targeting._boxes(image) == boxes
    assert calibration.input_mode == targeting.input_mode == "原始截图"
    assert provider.recognize.call_count == 2  # One raw pass per caller; no extra preprocessing.


@pytest.mark.parametrize("scale", [1, 2, 3, 5])
def test_both_entrypoints_map_preprocessed_boxes_to_actual_screenshot(scale):
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    original = OCRBox("战斗力：67,499", 0.9, (120, 20, 70, 25))
    enlarged = OCRBox(original.text, original.confidence, tuple(v * scale for v in original.bbox))
    provider = MagicMock()
    provider.recognize.side_effect = [[], [enlarged], [], [enlarged]]
    config = AppConfig()
    config.ocr.scale = scale
    calibration = ScreenAnalyzer(provider, config)
    targeting = AutoTargeting(provider, config)
    assert calibration.recognize(image) == targeting._boxes(image) == [original]
    assert calibration.input_mode == targeting.input_mode == "预处理截图"
    assert provider.recognize.call_args_list[1].args[0].shape == (100 * scale, 200 * scale)
    assert provider.recognize.call_args_list[3].args[0].shape == (100 * scale, 200 * scale)


def test_failed_raw_pass_retries_same_as_calibration():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    box = OCRBox("请选择对手", 0.9, (60, 30, 120, 60))
    provider = MagicMock()
    provider.recognize.side_effect = [OCRProviderError("raw failed"), [box]]
    result = recognize_frame(provider, image, OcrSettings(scale=3))
    assert result.boxes == [OCRBox("请选择对手", 0.9, (20, 10, 40, 20))]
    assert result.input_mode == "预处理截图"


@pytest.mark.parametrize("stage", [1, 2])
def test_stop_is_not_swallowed_or_retried(stage):
    provider = MagicMock()
    provider.recognize.side_effect = [StoppedError("stop")] if stage == 1 else [[], StoppedError("stop")]
    with pytest.raises(StoppedError):
        recognize_frame(provider, np.zeros((100, 200, 3), dtype=np.uint8), OcrSettings())
    assert provider.recognize.call_count == stage


def test_failed_both_passes_return_empty_not_coordinates_or_zero():
    provider = MagicMock()
    provider.recognize.side_effect = [OCRProviderError("raw failed"), OCRProviderError("processed failed")]
    result = recognize_frame(provider, np.zeros((100, 200, 3), dtype=np.uint8), OcrSettings())
    assert result.boxes == []
    assert result.input_mode == "预处理截图"


def test_empty_input_never_starts_ocr():
    provider = MagicMock()
    assert recognize_frame(provider, None, OcrSettings()).boxes == []
    assert recognize_frame(provider, np.empty((0, 0, 3), dtype=np.uint8), OcrSettings()).boxes == []
    provider.recognize.assert_not_called()


def test_scaled_boxes_clamp_to_screenshot_and_discard_invalid_regions():
    provider = MagicMock()
    provider.recognize.side_effect = [[], [
        OCRBox("边缘", 0.9, (-6, -3, 612, 306)),
        OCRBox("屏幕外", 0.9, (700, 500, 30, 30)),
        OCRBox("零宽", 0.9, (10, 10, 0, 30)),
    ]]
    result = recognize_frame(provider, np.zeros((100, 200, 3), dtype=np.uint8), OcrSettings(scale=3))
    assert result.boxes == [OCRBox("边缘", 0.9, (0, 0, 200, 100))]


def test_power_roi_retry_uses_original_pixels_not_repeated_thresholded_input():
    config = AppConfig()
    engine = OcrEngine(provider=MagicMock(), settings=config.ocr, use_shared=False)
    engine.read_roi = MagicMock(side_effect=[
        ([OCRBox("67,499", 0.1, (0, 0, 60, 20))], "67,499"),
        ([OCRBox("战斗力：67,499", 0.9, (0, 0, 60, 20))], "战斗力：67,499"),
    ])
    targeting = AutoTargeting(engine, config)
    assert targeting._read_power(np.zeros((100, 200, 3), dtype=np.uint8), (120, 20, 70, 25)) == 67499
    first, second = engine.read_roi.call_args_list
    assert first.kwargs["preprocess"] is True
    assert second.kwargs["preprocess"] is False
    for call in (first, second):
        assert call.kwargs["cache_key"] is None
        assert call.kwargs["force"] is True
        assert call.kwargs["allowlist"] == config.ocr.power_allowlist
