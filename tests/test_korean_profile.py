"""Default Korean UI keywords and label parsing; no external model required."""

from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from arena_auto.calibration.button_detector import ButtonDetector
from arena_auto.calibration.coordinate_mapper import CoordinateMapper
from arena_auto.calibration.opponent_detector import OpponentDetector
from arena_auto.config.manager import ConfigManager
from arena_auto.models.config import AppConfig, ArenaProfile, split_ocr_text
from arena_auto.models.recognition import OCRBox, DetectionResult
from arena_auto.models.state import ScreenKind
from arena_auto.recognition.auto_targeting import AutoTargeting
from arena_auto.recognition.detector import ScreenDetector, ButtonFinder
from arena_auto.recognition.ocr import parse_power


def box(text, x=100, y=100, w=200, h=30, confidence=0.95):
    return OCRBox(text, confidence, (x, y, w, h))


@pytest.mark.parametrize("text,expected", [
    ("승자의 줘팸터", ScreenKind.ARENA),
    ("대전상대를선택하여주세요.", ScreenKind.ARENA),
    ("도전 가능 횟수", ScreenKind.ARENA),
    ("이기러 간다", ScreenKind.PREPARE),
    ("이기러간다", ScreenKind.PREPARE),
    ("도전 횟수 구매", ScreenKind.PURCHASE),
    ("구매", ScreenKind.PURCHASE),
    ("나가기", ScreenKind.RESULT),
])
def test_korean_pages_match_default_profile_without_manual_keywords(text, expected):
    detector = ScreenDetector(MagicMock(), AppConfig())
    assert detector._classify_from_boxes(np.zeros((720, 1600, 3), dtype=np.uint8), [box(text)]) == expected


@pytest.mark.parametrize("text,outcome", [("VICTORY", "WIN"), ("DEFEAT", "LOSE"), ("V I C T O R Y", "WIN")])
def test_korean_server_english_result_banner(text, outcome):
    detector = ScreenDetector(MagicMock(), AppConfig())
    boxes = [box("승자의 줘팸터"), box(text, y=500), box("나가기", y=650)]
    assert detector.detect_outcome(np.zeros((720, 1600, 3), dtype=np.uint8), boxes=boxes) == outcome


def test_view_opponent_arrangement_is_not_purchase_confirmation():
    config = AppConfig()
    detector = ScreenDetector(MagicMock(), config)
    image = np.zeros((720, 1600, 3), dtype=np.uint8)
    assert detector._classify_from_boxes(image, [box("상대 배치 확인")]) != ScreenKind.PURCHASE
    assert "confirm_buy" not in ButtonDetector(config, CoordinateMapper(1600, 720, 1600, 720)).detect(
        [box("상대 배치 확인")], names=["confirm_buy"])


def test_calibration_purchase_uses_button_not_dialog_title():
    config = AppConfig()
    detector = ButtonDetector(config, CoordinateMapper(1600, 720, 1600, 720))
    found = detector.detect([box("도전 횟수 구매", 740, 160, 180, 35, 0.99),
                             box("구매", 920, 500, 60, 35, 0.9)], names=["buy_challenge"])
    assert found["buy_challenge"].click_point.y >= 500


@pytest.mark.parametrize("name,text", [("go_win", "이기러 간다"), ("exit", "나가기"),
                                       ("buy_challenge", "구매"), ("confirm_buy", "확인")])
def test_korean_buttons_work_with_default_alternatives(name, text):
    config = AppConfig()
    detector = ButtonDetector(config, CoordinateMapper(1600, 720, 1600, 720))
    ocr = MagicMock()
    ocr.recognize.return_value = [box(text)]
    templates = MagicMock()
    templates.find.return_value = DetectionResult(found=False)
    field = {"go_win": "go_win_text", "exit": "exit_text", "buy_challenge": "buy_text", "confirm_buy": "confirm_text"}[name]
    result = ButtonFinder(templates, ocr, config).find(np.zeros((720, 1600, 3), dtype=np.uint8), name, getattr(config.profile, field))
    assert result.found
    assert name in detector.detect([box(text)], names=[name])


@pytest.mark.parametrize("text", ["전투력 : 86,050", "전투력：86,050", "전투력 86050"])
def test_korean_power_label_parses_without_being_treated_as_noise(text):
    assert parse_power(text) == 86050
    assert parse_power("방어덱 전투력 768382") is None


def test_auto_initialization_accepts_korean_headers_and_excludes_own_defense():
    config = AppConfig()
    image = np.zeros((600, 1000, 3), dtype=np.uint8)
    boxes = [box("대전 상대를 선택하여 주세요.", 340, 60, 270, 25)]
    for index, power in enumerate([86050, 184390, 42666, 11396]):
        top = 100 + index * 145
        cv2.rectangle(image, (330, top), (930, top + 130), (220, 220, 220), -1)
        boxes += [box(f"상대{index}", 340, top + 10, 100, 20),
                  box(f"순위{2866 + index}", 620, top + 10, 80, 25),
                  box(f"전투력：{power}", 710, top + 10, 160, 25)]
    boxes.append(box("방어덱 전투력：768,382", 30, 450, 250, 30))
    targeting = AutoTargeting(MagicMock(), config)
    cards = targeting.detect_cards(image, boxes)
    assert [c.opponent.power for c in cards] == [86050, 184390, 42666]
    mapper = CoordinateMapper(config.screen.reference_width, config.screen.reference_height, 1000, 600)
    candidates = OpponentDetector(config, mapper).extract_candidates(boxes)
    assert 768382 not in [c.value for c in candidates]


def test_challenge_count_accepts_korean_anchor_with_spacing_variation():
    ocr = MagicMock()
    ocr.recognize.return_value = [box("도전가능횟수", 100, 350, 160, 30), box("5 / 5", 300, 350, 70, 30)]
    assert str(AutoTargeting(ocr, AppConfig()).read_challenge(np.zeros((720, 1600, 3), dtype=np.uint8))) == "5/5"


def test_new_defaults_roundtrip_but_explicit_custom_and_disabled_text_survive(tmp_path):
    manager = ConfigManager(tmp_path / "config.yaml", AppConfig())
    manager.save()
    restored = manager.load().profile
    assert "이기러 간다" in split_ocr_text(restored.go_win_text)
    assert "VICTORY" in split_ocr_text(restored.victory_text)
    assert "去获胜" in split_ocr_text(restored.go_win_text)
    custom = ConfigManager.from_dict({"profile": {"go_win_text": "开始挑战", "exit_text": "", "arena_title_keywords": []}})
    assert custom.profile.go_win_text == "开始挑战"
    assert custom.profile.exit_text == ""
    assert custom.profile.arena_title_keywords == []
    assert ArenaProfile().confirm_text == restored.confirm_text
