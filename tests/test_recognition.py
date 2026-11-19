"""Recognition tests: template matching and screen detection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from arena_auto.config.manager import ConfigManager
from arena_auto.models.config import TemplateEntry
from arena_auto.models.state import ScreenKind
from arena_auto.recognition.detector import ScreenDetector
from arena_auto.recognition.ocr import OCRBox, NullOCRProvider, preprocess_for_ocr
from arena_auto.recognition.template import TemplateRecognizer


class FakeOCR(NullOCRProvider):
    def __init__(self, texts):
        super().__init__()
        self.texts = list(texts)
        self.calls = 0

    def recognize(self, image):
        self.calls += 1
        out = []
        for i, t in enumerate(self.texts):
            out.append(OCRBox(text=t, confidence=0.95, bbox=(0, i * 10, 50, 10)))
        return out


def test_template_matching_hit(tmp_path: Path) -> None:
    screen = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.rectangle(screen, (50, 60), (110, 90), (0, 200, 255), -1)
    cv2.putText(screen, "GO", (55, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    tpl_dir = tmp_path / "templates"
    tpl_dir.mkdir()
    template = screen[60:90, 50:110].copy()
    cv2.imwrite(str(tpl_dir / "go_win.png"), template)

    recognizer = TemplateRecognizer(
        templates={"go_win": TemplateEntry(path=str(tpl_dir / "go_win.png"), threshold=0.8)},
        base_dir=tmp_path,
        multi_scale=False,
    )
    result = recognizer.find(screen, "go_win")
    assert result.found
    assert result.confidence >= 0.8
    cx, cy = result.center
    assert 50 <= cx <= 110
    assert 60 <= cy <= 90


def test_template_matching_miss() -> None:
    screen = np.zeros((200, 300, 3), dtype=np.uint8)
    tmp = TemplateRecognizer(templates={}, base_dir=Path("."))
    tmp.register("exit", TemplateEntry(path="__missing__.png", threshold=0.9))
    result = tmp.find(screen, "exit")
    assert not result.found


def test_preprocess_for_ocr_shape() -> None:
    from arena_auto.models.config import OcrSettings

    roi = np.random.randint(0, 255, (40, 120, 3), dtype=np.uint8)
    out = preprocess_for_ocr(roi, OcrSettings(scale=3, threshold=True))
    assert out.ndim == 2
    assert out.shape[0] >= 40 * 3 - 2


def test_screen_detector_arena() -> None:
    from arena_auto.config.manager import default_config_dict

    config = ConfigManager.from_dict(default_config_dict())
    ocr = FakeOCR(["胜利者的竞技场", "请选择对手", "可挑战次数 4 / 5"])
    detector = ScreenDetector(ocr, config, templates=None)
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert detector.detect(image) == ScreenKind.ARENA
    assert detector.detect_arena_screen(image)


def test_screen_detector_purchase_dialog_beats_arena() -> None:
    """购买弹窗盖在竞技场上：标题「购买挑战次数」必须判为 PURCHASE。"""
    from arena_auto.config.manager import default_config_dict

    config = ConfigManager.from_dict(default_config_dict())
    ocr = FakeOCR(
        [
            "胜利者的竞技场",
            "请选择对手",
            "可挑战次数 0 / 5",
            "购买挑战次数",
            "取消",
            "购买",
        ]
    )
    detector = ScreenDetector(ocr, config, templates=None)
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert detector.detect(image) == ScreenKind.PURCHASE


def test_screen_detector_prepare_deck_not_arena() -> None:
    """设置卡组页标题含竞技场名，但只有「去获胜」时应判 PREPARE。"""
    from arena_auto.config.manager import default_config_dict

    config = ConfigManager.from_dict(default_config_dict())
    ocr = FakeOCR(["设置胜利者的竞技场攻击卡组", "去获胜"])
    detector = ScreenDetector(ocr, config, templates=None)
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert detector.detect(image) == ScreenKind.PREPARE


def test_screen_detector_result_win() -> None:
    from arena_auto.config.manager import default_config_dict

    config = ConfigManager.from_dict(default_config_dict())
    ocr = FakeOCR(["胜利", "退出"])
    detector = ScreenDetector(ocr, config, templates=None)
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert detector.detect(image) == ScreenKind.RESULT
    assert detector.detect_outcome(image) == "WIN"


def test_screen_detector_result_lose() -> None:
    from arena_auto.config.manager import default_config_dict

    config = ConfigManager.from_dict(default_config_dict())
    ocr = FakeOCR(["失败", "退出"])
    detector = ScreenDetector(ocr, config, templates=None)
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert detector.detect_outcome(image) == "LOSE"


def test_outcome_unknown_not_assumed_defeat() -> None:
    from arena_auto.config.manager import default_config_dict

    config = ConfigManager.from_dict(default_config_dict())
    ocr = FakeOCR(["战斗中..."])
    detector = ScreenDetector(ocr, config, templates=None)
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert detector.detect_outcome(image) is None


def test_buy_ocr_prefers_button_over_title() -> None:
    """全图 OCR 同时命中标题与「购买」按钮时点按钮，不点标题。"""
    from arena_auto.config.manager import default_config_dict
    from arena_auto.recognition.detector import ButtonFinder

    class MultiBoxOCR(NullOCRProvider):
        def recognize(self, image):
            return [
                OCRBox(text="购买挑战次数", confidence=0.99, bbox=(727, 163, 168, 36)),
                OCRBox(text="购买", confidence=0.97, bbox=(906, 504, 58, 30)),
                OCRBox(text="取消", confidence=0.99, bbox=(656, 502, 56, 32)),
            ]

    config = ConfigManager.from_dict(default_config_dict())
    # 不配置模板/区域提示，强制走全图 OCR
    config.templates.clear()
    finder = ButtonFinder(TemplateRecognizer(templates={}), MultiBoxOCR(), config, None)
    result = finder.find(np.zeros((720, 1600, 3), dtype=np.uint8), "buy_challenge", "购买挑战次数")
    assert result.found
    assert result.bbox == (906, 504, 58, 30)


def test_confirm_ocr_ignores_longer_phrase() -> None:
    """「确认对手布置」不应被当成确认购买按钮。"""
    from arena_auto.config.manager import default_config_dict
    from arena_auto.recognition.detector import ButtonFinder

    class PhraseOCR(NullOCRProvider):
        def recognize(self, image):
            return [
                OCRBox(text="确认对手布置", confidence=0.95, bbox=(1008, 562, 102, 24)),
            ]

    config = ConfigManager.from_dict(default_config_dict())
    config.templates.clear()
    finder = ButtonFinder(TemplateRecognizer(templates={}), PhraseOCR(), config, None)
    result = finder.find(np.zeros((720, 1600, 3), dtype=np.uint8), "confirm_buy", "确认")
    assert not result.found


def test_confirm_ocr_prefers_exact_over_phrase() -> None:
    from arena_auto.config.manager import default_config_dict
    from arena_auto.recognition.detector import ButtonFinder

    class BothOCR(NullOCRProvider):
        def recognize(self, image):
            return [
                OCRBox(text="确认对手布置", confidence=0.99, bbox=(1008, 562, 102, 24)),
                OCRBox(text="确认", confidence=0.90, bbox=(900, 500, 50, 30)),
            ]

    config = ConfigManager.from_dict(default_config_dict())
    config.templates.clear()
    finder = ButtonFinder(TemplateRecognizer(templates={}), BothOCR(), config, None)
    result = finder.find(np.zeros((720, 1600, 3), dtype=np.uint8), "confirm_buy", "确认")
    assert result.found
    assert result.bbox == (900, 500, 50, 30)


def test_template_rejects_all_black(tmp_path: Path) -> None:
    black = np.zeros((40, 80, 3), dtype=np.uint8)
    path = tmp_path / "black.png"
    cv2.imwrite(str(path), black)
    recon = TemplateRecognizer(
        templates={"go_win": TemplateEntry(path=str(path), threshold=0.5)},
        base_dir=tmp_path,
        multi_scale=False,
    )
    assert recon.load("go_win") is None
    result = recon.find(np.zeros((200, 300, 3), dtype=np.uint8), "go_win")
    assert not result.found
