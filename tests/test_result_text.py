"""Victory and defeat use the same spacing, geometry and template rules."""

import time
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from arena_auto.automation.controller import AutomationController
from arena_auto.calibration.button_detector import ButtonDetector
from arena_auto.calibration.coordinate_mapper import CoordinateMapper
from arena_auto.calibration.models import CalibrationRegion
from arena_auto.calibration.template_generator import TemplateGenerator
from arena_auto.config.manager import ConfigManager
from arena_auto.models.config import AppConfig, ButtonGeometry, TemplateEntry
from arena_auto.models.recognition import OCRBox, DetectionResult
from arena_auto.models.state import AutomationState, ScreenKind
from arena_auto.recognition.detector import ButtonFinder, ScreenDetector
from arena_auto.recognition.result_text import find_result_text_boxes
from arena_auto.recognition.template import TemplateRecognizer


def title():
    return OCRBox("胜利者的竞技场", 0.94, (409, 463, 192, 36))


def victory():
    return OCRBox("胜  利", 0.86, (380, 498, 248, 104))


def test_spaced_victory_uses_banner_not_title():
    assert find_result_text_boxes([title(), victory()], "胜利") == [victory()]
    assert not find_result_text_boxes([title()], "胜利")


@pytest.mark.parametrize("keyword", ["胜利", "失败"])
@pytest.mark.parametrize("separator", [" ", "  ", "\t", "\u3000", ""])
def test_normalized_result_label(keyword, separator):
    text = separator.join(keyword) + "!!"
    box = OCRBox(text, 0.9, (10, 10, 200, 100))
    assert find_result_text_boxes([box], keyword) == [box]


@pytest.mark.parametrize("keyword", ["胜利", "失败"])
def test_split_glyph_boxes_join_by_geometry(keyword):
    boxes = [OCRBox(keyword[0], 0.9, (380, 498, 80, 100)), OCRBox(keyword[1], 0.8, (540, 500, 80, 100))]
    matched = find_result_text_boxes(boxes, keyword)
    assert len(matched) == 1
    assert matched[0].bbox == (380, 498, 240, 102)
    assert matched[0].confidence == 0.8
    assert matched[0].text == keyword


def test_split_glyphs_on_different_lines_or_far_apart_are_not_joined():
    first = OCRBox("胜", 0.9, (10, 10, 50, 50))
    assert not find_result_text_boxes([first, OCRBox("利", 0.9, (500, 10, 50, 50))], "胜利")
    assert not find_result_text_boxes([first, OCRBox("利", 0.9, (70, 100, 50, 50))], "胜利")


def test_split_title_not_mistaken_for_result():
    boxes = [OCRBox("胜", 0.9, (10, 10, 20, 20)), OCRBox("利", 0.9, (35, 10, 20, 20)),
             OCRBox("者的竞技场", 0.9, (60, 10, 110, 20))]
    assert not find_result_text_boxes(boxes, "胜利")


def test_no_win_from_title_and_defeat_is_not_ambiguous():
    config = AppConfig()
    detector = ScreenDetector(MagicMock(), config)
    image = np.zeros((720, 1600, 3), dtype=np.uint8)
    assert detector.detect_outcome(image, [title()]) is None
    assert detector.detect_outcome(image, [title(), OCRBox("失  败", 0.9, (380, 498, 248, 104))]) == "LOSE"
    assert detector.detect_outcome(image, [title(), victory()]) == "WIN"
    assert detector._classify_from_boxes(image, [title(), victory()]) == ScreenKind.RESULT


def test_conflicting_or_low_confidence_result_is_unknown():
    detector = ScreenDetector(MagicMock(), AppConfig())
    image = np.zeros((720, 1600, 3), dtype=np.uint8)
    assert detector.detect_outcome(image, [victory(), OCRBox("失败", 0.9, (700, 500, 200, 100))]) is None
    assert detector.detect_outcome(image, [title(), OCRBox("胜利", 0.1, (380, 498, 248, 104))]) is None


@pytest.mark.parametrize("region_hint", [False, True])
@pytest.mark.parametrize("name,keyword", [("victory", "胜利"), ("defeat", "失败")])
def test_debug_button_finder_matches_spaced_banner(region_hint, name, keyword):
    config = AppConfig()
    banner = OCRBox("  ".join(keyword), 0.86, victory().bbox)
    if region_hint:
        config.screen.reference_width = 1600
        config.screen.reference_height = 720
        config.buttons[name] = ButtonGeometry(region=(0, 0, 1600, 720))
    templates = MagicMock()
    templates.find.return_value = DetectionResult(found=False)
    ocr = MagicMock()
    ocr.recognize.return_value = [title(), banner]
    finder = ButtonFinder(templates, ocr, config)
    result = finder.find(np.zeros((720, 1600, 3), dtype=np.uint8), name, keyword)
    assert result.found and result.bbox == banner.bbox
    ocr.recognize.return_value = [title()]
    assert not finder.find(np.zeros((720, 1600, 3), dtype=np.uint8), name, keyword).found


@pytest.mark.parametrize("name,keyword", [("victory", "胜利"), ("defeat", "失败")])
def test_calibration_detects_result_for_template_generation(name, keyword):
    config = AppConfig()
    mapper = CoordinateMapper(1600, 720, 1600, 720)
    detector = ButtonDetector(config, mapper)
    banner = OCRBox("  ".join(keyword), 0.86, victory().bbox)
    found = detector.detect([title(), banner], names=[name])
    assert name in found
    assert found[name].region.y >= 470
    assert name not in detector.detect([title()], names=[name])


def test_custom_result_alternatives_and_multiword_text():
    assert find_result_text_boxes([OCRBox("You  Win!", 0.9, (10, 10, 100, 30))], "胜利,You Win")
    assert not find_result_text_boxes([OCRBox("Victory Arena", 0.9, (10, 10, 100, 30))], "Victory")
    assert not find_result_text_boxes([victory()], ",，")


@pytest.mark.parametrize("name,outcome", [("victory", "WIN"), ("defeat", "LOSE")])
def test_generated_banner_template_handles_normal_battle_loop_without_ocr(tmp_path, name, outcome):
    image = np.zeros((300, 500, 3), dtype=np.uint8)
    cv2.rectangle(image, (60, 160), (230, 250), (160, 30, 80), -1)
    cv2.putText(image, outcome, (85, 225), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (230, 210, 255), 3)
    manager = ConfigManager(tmp_path / "config.yaml")
    manager.config.debug.enabled = False
    mapper = CoordinateMapper(500, 300, 500, 300)
    path = TemplateGenerator(manager.config, mapper, tmp_path / "generated").generate(
        image, name, CalibrationRegion(60, 160, 170, 90))
    manager.config.templates[name] = TemplateEntry(str(path), 0.8)
    templates = TemplateRecognizer(manager.config.templates, base_dir=tmp_path)
    adb = MagicMock()
    adb.screenshot.return_value = image
    ocr = MagicMock()
    ctrl = AutomationController(manager, adb=adb, ocr=ocr, templates=templates)
    ctrl._wait = MagicMock(return_value=False)
    ctrl.ctx.battle_started_at = time.monotonic()
    ctrl.sm.transition(AutomationState.WAIT_RESULT)
    ctrl._state_wait_result()
    assert ctrl.sm.state == AutomationState.RESULT
    assert ctrl.ctx.outcome == outcome
    ctrl._state_result()
    assert ctrl.sm.state == AutomationState.EXIT_RESULT
    assert ctrl.statistics.victories == (1 if outcome == "WIN" else 0)
    assert ctrl.statistics.defeats == (1 if outcome == "LOSE" else 0)
    ocr.recognize.assert_not_called()
    adb.tap.assert_not_called()


def test_failure_page_with_split_letters_is_result_and_not_win():
    config = AppConfig()
    ocr = MagicMock()
    ocr.recognize.return_value = [title(), OCRBox("失", 0.9, (380, 498, 80, 100)),
                                OCRBox("败", 0.8, (540, 500, 80, 100)),
                                OCRBox("退 出", 0.9, (1082, 648, 52, 30))]
    detector = ScreenDetector(ocr, config)
    kind, outcome = detector.detect_screen_and_outcome(np.zeros((720, 1600, 3), dtype=np.uint8))
    assert kind == ScreenKind.RESULT
    assert outcome == "LOSE"
    ocr.recognize.assert_called_once()
