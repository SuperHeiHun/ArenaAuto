"""Custom OCR configuration, GUI editing and recognition integration."""

from dataclasses import asdict
from unittest.mock import MagicMock

import numpy as np
import pytest

from arena_auto.calibration.button_detector import ButtonDetector
from arena_auto.calibration.coordinate_mapper import CoordinateMapper
from arena_auto.config.manager import ConfigManager, parse_arena_profile
from arena_auto.exceptions import ConfigError
from arena_auto.models.config import AppConfig, ArenaProfile, ButtonGeometry, split_ocr_text
from arena_auto.models.recognition import DetectionResult, OCRBox
from arena_auto.models.state import ScreenKind
from arena_auto.recognition.detector import ButtonFinder, ScreenDetector
from arena_auto.recognition.ocr import NullOCRProvider


def test_profile_roundtrip_and_empty_keywords(tmp_path):
    cfg = ConfigManager.from_dict({"profile": {
        "arena_title_keywords": "  自定义竞技场  \n备用标题\n自定义竞技场\n",
        "arena_select_keywords": [], "challenge_keywords": [],
        "go_win_text": "  开始挑战  ", "exit_text": "",
    }})
    assert cfg.profile.arena_title_keywords == ["自定义竞技场", "备用标题"]
    assert cfg.profile.arena_select_keywords == []
    assert cfg.profile.go_win_text == "开始挑战"
    manager = ConfigManager(tmp_path / "config.yaml", cfg)
    manager.save()
    assert asdict(manager.load().profile) == asdict(cfg.profile)


@pytest.mark.parametrize("raw", [
    None, [], {"arena_title_keywords": [1]}, {"go_win_text": None},
    {"min_keyword_matches": 0}, {"min_keyword_matches": "two"},
])
def test_invalid_profile_rejected(raw):
    with pytest.raises(ConfigError):
        parse_arena_profile(raw)


class TextOCR(NullOCRProvider):
    def __init__(self, text):
        self.text = text

    def recognize(self, image):
        return [OCRBox(text=self.text, confidence=0.95, bbox=(20, 20, 100, 30))]


@pytest.mark.parametrize("text,kind,outcome", [
    ("自定义竞技场", ScreenKind.ARENA, None),
    ("开始挑战", ScreenKind.PREPARE, None),
    ("补充门票", ScreenKind.PURCHASE, None),
    ("获胜 返回", ScreenKind.RESULT, "WIN"),
    ("落败 返回", ScreenKind.RESULT, "LOSE"),
])
def test_custom_page_and_outcome(text, kind, outcome):
    cfg = AppConfig(profile=ArenaProfile(
        arena_title_keywords=["其他竞技场,自定义竞技场"], go_win_text="去获胜,开始挑战",
        buy_text="购买挑战次数，补充门票", victory_text="胜利,获胜", defeat_text="失败，落败", exit_text="退出,返回",
    ))
    detector = ScreenDetector(TextOCR(text), cfg)
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    assert detector.detect_screen_and_outcome(image) == (kind, outcome)


@pytest.mark.parametrize("name,field,method", [
    ("go_win", "go_win_text", "find_go_win"),
    ("exit", "exit_text", "find_exit"),
    ("buy_challenge", "buy_text", "find_buy"),
    ("confirm_buy", "confirm_text", "find_confirm"),
])
def test_custom_button_runtime_and_calibration(name, field, method):
    cfg = AppConfig()
    setattr(cfg.profile, field, "其他按钮,自定义按钮")
    templates = MagicMock()
    templates.find.return_value = DetectionResult(found=False)
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    ocr = TextOCR("自定义按钮")
    finder = ButtonFinder(templates, ocr, cfg)
    assert getattr(finder, method)(image).found
    mapper = CoordinateMapper(300, 200, 300, 200)
    assert name in ButtonDetector(cfg, mapper).detect(ocr.recognize(image), names=[name])


def test_empty_button_text_disables_ocr():
    cfg = AppConfig()
    cfg.profile.go_win_text = ""
    ocr = MagicMock()
    templates = MagicMock()
    templates.find.return_value = DetectionResult(found=False)
    assert not ButtonFinder(templates, ocr, cfg).find_go_win(np.zeros((100, 100, 3))).found
    ocr.recognize.assert_not_called()


@pytest.fixture
def qt_app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_dialog_save_cancel_reset(qt_app):
    from arena_auto.gui.ocr_text_dialog import OcrTextDialog
    original = ArenaProfile(go_win_text="旧按钮")
    dialog = OcrTextDialog(original)
    dialog.text_edits["go_win_text"].setText(" 新按钮 ")
    dialog.keyword_edits["arena_title_keywords"].setPlainText("标题一\n标题二\n标题一")
    dialog.reject()
    assert original.go_win_text == "旧按钮"
    assert dialog.profile is original
    dialog.accept()
    assert dialog.profile.go_win_text == "新按钮"
    assert dialog.profile.arena_title_keywords == ["标题一", "标题二"]
    assert original.go_win_text == "旧按钮"
    dialog.load_profile(ArenaProfile())
    assert dialog.edited_profile() == ArenaProfile()


@pytest.mark.parametrize("name,expected", [("buy", "buy_challenge"), ("confirm", "confirm_buy")])
def test_debug_button_aliases(qt_app, name, expected):
    from arena_auto.gui.workers import DebugTestWorker
    adb = MagicMock()
    adb.screenshot.return_value = np.zeros((100, 200, 3), dtype=np.uint8)
    ctrl = MagicMock()
    ctrl.config = AppConfig()
    ctrl.buttons.find.return_value = DetectionResult(found=False)
    worker = DebugTestWorker(adb, ctrl, name)
    worker.run()
    assert ctrl.buttons.find.call_args.args[1] == expected


@pytest.mark.parametrize("save_fails", [False, True])
def test_main_window_applies_only_after_successful_save(qt_app, monkeypatch, tmp_path, save_fails):
    from PySide6.QtWidgets import QDialog
    from arena_auto.gui import main_window

    monkeypatch.setattr(main_window.MainWindow, "refresh_devices", lambda self: None)
    monkeypatch.setattr(main_window.MainWindow, "_attach_log_handler", lambda self: None)
    manager = ConfigManager(tmp_path / "config.yaml")
    original = manager.config
    window = main_window.MainWindow(manager)
    window._controller = MagicMock()
    dialog = MagicMock()
    dialog.exec.return_value = QDialog.Accepted
    dialog.profile = ArenaProfile(go_win_text="开始挑战")
    monkeypatch.setattr(main_window, "OcrTextDialog", lambda *args: dialog)
    error = MagicMock()
    monkeypatch.setattr(main_window.QMessageBox, "critical", error)
    if save_fails:
        monkeypatch.setattr(ConfigManager, "save", MagicMock(side_effect=OSError("write failed")))
    try:
        window.open_ocr_text_settings()
        assert original.profile.go_win_text == ArenaProfile().go_win_text
        if save_fails:
            assert window.config is original
            assert manager.config is original
            window._controller.reload_config.assert_not_called()
            error.assert_called_once()
        else:
            assert window.config is manager.config
            assert manager.config.profile.go_win_text == "开始挑战"
            assert ConfigManager.load_or_create(manager.path).config.profile.go_win_text == "开始挑战"
            window._controller.reload_config.assert_called_once()
    finally:
        window._controller = None
        window.close()


def test_comma_normalization_and_roundtrip(tmp_path):
    assert split_ocr_text(" , 开始，挑战\n开始,, ") == ["开始", "挑战"]
    cfg = ConfigManager.from_dict({"profile": {
        "display_name": "游戏,配置",
        "arena_title_keywords": [" 标题一,标题二 ", "标题二，标题三"],
        "go_win_text": " 去获胜，开始挑战,,去获胜 ",
    }})
    assert cfg.profile.display_name == "游戏,配置"
    assert cfg.profile.arena_title_keywords == ["标题一", "标题二", "标题三"]
    assert cfg.profile.go_win_text == "去获胜,开始挑战"
    manager = ConfigManager(tmp_path / "config.yaml", cfg)
    manager.save()
    assert manager.load().profile == cfg.profile


def test_dialog_accepts_commas(qt_app):
    from arena_auto.gui.ocr_text_dialog import OcrTextDialog
    dialog = OcrTextDialog(ArenaProfile())
    dialog.text_edits["go_win_text"].setText("去获胜, 开始挑战，去获胜")
    dialog.keyword_edits["arena_title_keywords"].setPlainText("竞技场,对战大厅\n竞技场，比赛大厅")
    dialog.accept()
    assert dialog.profile.go_win_text == "去获胜,开始挑战"
    assert dialog.profile.arena_title_keywords == ["竞技场", "对战大厅", "比赛大厅"]


@pytest.mark.parametrize("region", [False, True])
def test_multiple_button_texts_use_one_ocr_pass(region):
    cfg = AppConfig()
    cfg.profile.confirm_text = "确认,确定"
    if region:
        cfg.screen.reference_width = 300
        cfg.screen.reference_height = 200
        cfg.buttons["confirm_buy"] = ButtonGeometry(region=(0, 0, 200, 100))
    ocr = MagicMock()
    ocr.recognize.return_value = [OCRBox(text="确定", confidence=0.95, bbox=(20, 20, 100, 30))]
    templates = MagicMock()
    templates.find.return_value = DetectionResult(found=False)
    finder = ButtonFinder(templates, ocr, cfg)
    result = finder.find_confirm(np.zeros((200, 300, 3), dtype=np.uint8))
    assert result.found
    ocr.recognize.assert_called_once()


def test_multiple_short_keywords_do_not_match_long_phrase():
    cfg = AppConfig()
    cfg.profile.confirm_text = "确认,确定"
    templates = MagicMock()
    templates.find.return_value = DetectionResult(found=False)
    finder = ButtonFinder(templates, TextOCR("确认对手布置"), cfg)
    assert not finder.find_confirm(np.zeros((200, 300, 3), dtype=np.uint8)).found
    assert not ButtonDetector._matches("确认,确定", "确认对手布置")


def test_comma_only_disables_text_and_alternatives_are_not_extra_signals():
    cfg = ConfigManager.from_dict({"profile": {"go_win_text": ",， ,"}})
    assert cfg.profile.go_win_text == ""
    assert ScreenDetector._count_hits("胜利 获胜", ["胜利,获胜"]) == 1
    assert not ButtonDetector._matches(",，", "任何文字")
