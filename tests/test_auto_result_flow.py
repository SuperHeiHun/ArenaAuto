"""Auto mode recovers from absent result templates without busy-loop OCR."""

import time
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from arena_auto.automation.controller import AutomationController
from arena_auto.config.manager import ConfigManager
from arena_auto.exceptions import ConfigError, StoppedError
from arena_auto.models.config import AppConfig
from arena_auto.models.recognition import OCRBox, DetectionResult
from arena_auto.models.state import AutomationState
from arena_auto.recognition.template import TemplateRecognizer


def scene():
    image = np.zeros((300, 500, 3), dtype=np.uint8)
    cv2.rectangle(image, (50, 100), (240, 180), (160, 60, 80), -1)
    cv2.putText(image, "WIN", (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (230, 210, 255), 3)
    cv2.rectangle(image, (310, 230), (440, 280), (130, 220, 255), -1)
    cv2.putText(image, "EXIT", (320, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    return image, [OCRBox("胜  利", 0.9, (60, 100, 170, 80)),
                   OCRBox("退出", 0.9, (320, 235, 110, 40))]


def controller(tmp_path, automatic=True, boxes=None):
    manager = ConfigManager(tmp_path / "config.yaml")
    manager.config.automation.auto_targeting = automatic
    manager.config.debug.enabled = False
    image, default_boxes = scene()
    adb = MagicMock()
    adb.screenshot.side_effect = lambda: image.copy()
    ocr = MagicMock()
    ocr.recognize.return_value = default_boxes if boxes is None else boxes
    templates = TemplateRecognizer(manager.config.templates, base_dir=tmp_path)
    ctrl = AutomationController(manager, adb=adb, ocr=ocr, templates=templates)
    ctrl._wait = MagicMock(return_value=False)
    ctrl.ctx.battle_started_at = time.monotonic()
    ctrl.ctx.wait_result_template_fails = 2
    ctrl.sm.transition(AutomationState.WAIT_RESULT)
    return ctrl, image, ocr


@pytest.mark.parametrize("outcome", ["WIN", "LOSE"])
def test_missing_all_templates_auto_mode_recognizes_and_caches_result(tmp_path, outcome):
    _, boxes = scene()
    if outcome == "LOSE":
        boxes[0] = OCRBox("失  败", 0.9, boxes[0].bbox)
    ctrl, image, ocr = controller(tmp_path, boxes=boxes)
    assert ctrl.config.debug.allow_ocr_fallback is False
    ctrl._state_wait_result()
    assert ctrl.sm.state == AutomationState.RESULT
    assert ctrl.ctx.outcome == outcome
    name = "victory" if outcome == "WIN" else "defeat"
    assert ctrl.templates.is_loaded(name)
    assert ctrl.templates.is_loaded("exit")
    ctrl._state_result()
    assert ctrl.sm.state == AutomationState.EXIT_RESULT
    assert ctrl.statistics.victories == (outcome == "WIN")
    assert ctrl.statistics.defeats == (outcome == "LOSE")
    assert ctrl.buttons.find_exit(image, full_ocr=False).found
    ocr.recognize.assert_called_once()  # Later steps use live in-memory templates.
    ctrl.adb.tap.assert_not_called()
    assert not (tmp_path / "config.yaml").exists()


def test_manual_mode_still_does_not_run_ocr_without_opt_in(tmp_path):
    ctrl, _, ocr = controller(tmp_path, automatic=False)
    ctrl.ctx.wait_result_template_fails = 99
    ctrl._state_wait_result()
    assert ctrl.sm.state == AutomationState.WAIT_RESULT
    ocr.recognize.assert_not_called()


def test_ocr_cooldown_starts_after_probe_and_unknown_never_becomes_win(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("arena_auto.automation.controller.time.monotonic", lambda: clock[0])
    ctrl, _, ocr = controller(tmp_path, boxes=[])
    ctrl._state_wait_result()
    assert ocr.recognize.call_count == 1
    assert ctrl.ctx.outcome is None
    assert ctrl.sm.state == AutomationState.WAIT_RESULT
    clock[0] = 114.0
    ctrl._state_wait_result()
    assert ocr.recognize.call_count == 1
    clock[0] = 116.0
    ctrl._state_wait_result()
    assert ocr.recognize.call_count == 2
    assert not ctrl.templates.is_loaded("victory")
    ctrl.adb.tap.assert_not_called()


def test_ocr_does_not_start_before_minimum_template_misses(tmp_path):
    ctrl, _, ocr = controller(tmp_path)
    ctrl.ctx.wait_result_template_fails = 0
    ctrl._state_wait_result()
    assert ctrl.sm.state == AutomationState.WAIT_RESULT
    ocr.recognize.assert_not_called()


def test_exit_template_without_outcome_can_confirm_win_by_ocr_in_auto_mode(tmp_path):
    ctrl, image, ocr = controller(tmp_path)
    assert ctrl.templates.register_image("exit", image[235:275, 320:430])
    ctrl._state_wait_result()
    assert ctrl.sm.state == AutomationState.RESULT
    assert ctrl.ctx.outcome is None
    ocr.recognize.assert_not_called()
    ctrl._state_result()
    assert ctrl.statistics.victories == 1
    assert ctrl.sm.state == AutomationState.EXIT_RESULT
    ocr.recognize.assert_called_once()


def test_exit_evidence_without_outcome_does_not_invent_statistics(tmp_path):
    _, boxes = scene()
    ctrl, _, _ = controller(tmp_path, boxes=[boxes[1]])
    ctrl._state_wait_result()
    assert ctrl.sm.state == AutomationState.RESULT
    ctrl.ctx.result_ocr_last_at = 0  # Simulate cooldown expiry before another fresh-frame probe.
    ctrl._state_result()
    assert ctrl.sm.state == AutomationState.EXIT_RESULT
    assert ctrl.statistics.victories == ctrl.statistics.defeats == 0


@pytest.mark.parametrize("automatic,expected", [(False, AutomationState.RECOVERY), (True, AutomationState.STOPPED)])
def test_result_wait_does_not_reset_timeout_forever(tmp_path, automatic, expected):
    ctrl, _, _ = controller(tmp_path, automatic=automatic, boxes=[])
    ctrl.sm.transition(AutomationState.RESULT)
    ctrl.config.timeouts.result = 2
    ctrl.sm._entered_at = time.monotonic() - 1
    ctrl._state_result()
    assert ctrl.sm.state == AutomationState.RESULT
    assert ctrl.sm.time_in_state() >= 1
    ctrl.sm._entered_at = time.monotonic() - 3
    ctrl._state_result()
    assert ctrl.sm.state == expected
    ctrl.adb.tap.assert_not_called()


def test_stop_during_result_probe_prevents_template_learning_and_transition(tmp_path):
    ctrl, _, ocr = controller(tmp_path)
    def recognize(_image):
        ctrl.stop_event.set()
        return scene()[1]
    ocr.recognize.side_effect = recognize
    with pytest.raises(StoppedError):
        ctrl._state_wait_result()
    assert not ctrl.templates.is_loaded("victory")
    ctrl.adb.tap.assert_not_called()


def test_auto_exit_allows_live_ocr_without_configured_region(tmp_path):
    ctrl, _, _ = controller(tmp_path)
    ctrl.buttons.find_exit = MagicMock(return_value=DetectionResult(found=False))
    ctrl.sm.transition(AutomationState.EXIT_RESULT)
    ctrl._state_exit_result()
    assert ctrl.buttons.find_exit.call_args.kwargs["full_ocr"] is True


def test_runtime_template_reload_and_manual_override(tmp_path):
    config = AppConfig()
    templates = TemplateRecognizer(config.templates, base_dir=tmp_path)
    image, _ = scene()
    assert not templates.register_image("exit", np.zeros((20, 20, 3), dtype=np.uint8))
    assert templates.register_image("exit", image[235:275, 320:430])
    assert templates.find(image, "exit").found
    templates.set_entries(config.templates)
    assert templates.find(image, "exit").found
    config.templates["exit"].path = "manual/new_exit.png"  # In-place edits must invalidate live cache too.
    templates.set_entries(config.templates)
    assert not templates.is_loaded("exit")
    assert not templates.find(image, "exit").found


def test_result_ocr_interval_validation():
    assert ConfigManager.from_dict({}).battle.result_ocr_interval == 15
    with pytest.raises(ConfigError):
        ConfigManager.from_dict({"battle": {"result_ocr_interval": 0}})


@pytest.fixture
def qt_app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def window(qt_app, monkeypatch, tmp_path):
    from arena_auto.gui.main_window import MainWindow
    monkeypatch.setattr(MainWindow, "refresh_devices", lambda self: None)
    monkeypatch.setattr(MainWindow, "_attach_log_handler", lambda self: None)
    result = MainWindow(ConfigManager(tmp_path / "config.yaml"))
    result._ensure_controller = MagicMock()
    return result


def test_gui_blocks_all_debug_tests_while_running(qt_app, monkeypatch, tmp_path):
    from arena_auto.gui.main_window import QMessageBox
    ui = window(qt_app, monkeypatch, tmp_path)
    monkeypatch.setattr(ui, "_automation_running", lambda: True)
    warning = MagicMock()
    monkeypatch.setattr(QMessageBox, "warning", warning)
    try:
        ui.run_debug_test("powers")
        ui._ensure_controller.assert_not_called()
        warning.assert_called_once()
    finally:
        ui.close()


def test_gui_cannot_start_while_debug_worker_is_running(qt_app, monkeypatch, tmp_path):
    from arena_auto.gui.main_window import QMessageBox
    ui = window(qt_app, monkeypatch, tmp_path)
    ui._test_worker = MagicMock()
    ui._test_worker.isRunning.return_value = True
    monkeypatch.setattr(QMessageBox, "warning", MagicMock())
    try:
        ui.start_automation()
        ui._ensure_controller.assert_not_called()
    finally:
        ui.close()


def test_running_preview_does_not_mutate_form_config(qt_app, monkeypatch, tmp_path):
    from arena_auto.gui import main_window
    ui = window(qt_app, monkeypatch, tmp_path)
    monkeypatch.setattr(ui, "_automation_running", lambda: True)
    ui._apply_form_to_config = MagicMock()
    worker = MagicMock()
    monkeypatch.setattr(main_window, "ScreenshotWorker", MagicMock(return_value=worker))
    try:
        ui.refresh_screenshot()
        ui._apply_form_to_config.assert_not_called()
        worker.start.assert_called_once()
        worker.isRunning.return_value = True
        ui.refresh_screenshot()
        worker.start.assert_called_once()
    finally:
        ui.close()
