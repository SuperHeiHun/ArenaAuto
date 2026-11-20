"""Regression: repeated recognition failure must not recover to the same page forever."""

import copy
from unittest.mock import MagicMock

import numpy as np
import pytest

from arena_auto.automation.controller import AutomationController, ControllerCallbacks
from arena_auto.config.manager import ConfigManager
from arena_auto.debug.recorder import DebugRecorder
from arena_auto.exceptions import OCRFailedError, StoppedError
from arena_auto.models.recognition import ChallengeCount, DetectionResult, OCRBox, Opponent
from arena_auto.models.state import AutomationState, ScreenKind
from arena_auto.recognition.detector import PowerReader
from arena_auto.recognition.ocr import parse_power


def controller(tmp_path, *, automatic=False):
    manager = ConfigManager(tmp_path / "config.yaml")
    manager.config.automation.auto_targeting = automatic
    manager.config.automation.power_threshold = 700000
    manager.config.automation.dry_run = True
    manager.config.debug.enabled = False
    adb = MagicMock()
    adb.screenshot.side_effect = lambda: np.zeros((720, 1600, 3), dtype=np.uint8)
    templates = MagicMock()
    templates.preload_all.return_value = []
    templates.find.return_value = DetectionResult(found=False)
    ctrl = AutomationController(manager, adb=adb, ocr=MagicMock(), templates=templates,
                                debug=DebugRecorder(manager.config.debug, base_dir=tmp_path / "debug"))
    ctrl._wait = MagicMock(return_value=False)
    ctrl._cumulative_path = tmp_path / "statistics.json"
    ctrl._historical = {}
    return ctrl


@pytest.mark.parametrize("raw", ["전투럭:462,892", "전투럭：462,892", "전투럭 462892", "전투력:462,892"])
def test_observed_korean_label_typo_does_not_change_digits(raw):
    assert parse_power(raw) == 462892


@pytest.mark.parametrize("raw", [
    "방어덱 전투럭:462,892", "순위:462892", "전투갛:462892", "전투럭:462A892",
    "전력:102,063", "random text 462892", "전투럭:0", "전투럭:",
])
def test_unknown_labels_defense_rank_and_invalid_digits_still_fail(raw):
    assert parse_power(raw) is None


def test_repeated_same_arena_failure_exhausts_budget_without_clicks(tmp_path):
    ctrl = controller(tmp_path)
    ctrl.power_reader.read_until_at_or_below = MagicMock(side_effect=OCRFailedError("unknown power label"))
    ctrl.detector.detect = MagicMock(return_value=ScreenKind.ARENA)
    ctrl.challenge_reader.read = MagicMock(return_value=ChallengeCount(0, 5))
    for attempt in range(1, 4):
        ctrl.sm.transition(AutomationState.SELECT_OPPONENT)
        ctrl._state_select_opponent()
        assert ctrl.sm.state == AutomationState.RECOVERY
        ctrl._state_recovery()
        assert ctrl.sm.state == AutomationState.ARENA
        assert ctrl.ctx.recovery_attempts == attempt  # Not reset by recognizing ARENA.
        ctrl._state_arena()
        assert ctrl.sm.state == AutomationState.SELECT_OPPONENT
    screenshots_before = ctrl.adb.screenshot.call_count
    ctrl._state_select_opponent()
    ctrl._state_recovery()
    assert ctrl.sm.state == AutomationState.ERROR
    assert ctrl.ctx.recovery_attempts == ctrl.config.recovery.max_attempts == 3
    assert ctrl.statistics.recoveries == 3
    assert ctrl.adb.screenshot.call_count == screenshots_before
    ctrl.adb.tap.assert_not_called()
    ctrl.adb.back.assert_not_called()


@pytest.mark.parametrize("kind", [ScreenKind.ARENA, ScreenKind.PREPARE, ScreenKind.RESULT, ScreenKind.PURCHASE])
def test_page_recognition_alone_is_not_progress(kind, tmp_path):
    ctrl = controller(tmp_path)
    ctrl.ctx.recovery_attempts = 1
    ctrl.detector.detect = MagicMock(return_value=kind)
    ctrl.sm.transition(AutomationState.RECOVERY)
    ctrl._state_recovery()
    assert ctrl.ctx.recovery_attempts == 2


def test_verified_purchase_restoration_resets_failure_budget(tmp_path):
    ctrl = controller(tmp_path)
    ctrl.ctx.recovery_attempts = 2
    ctrl.ctx.screenshot = np.zeros((720, 1600, 3), dtype=np.uint8)
    ctrl.detector.detect = MagicMock(return_value=ScreenKind.ARENA)
    ctrl._finish_purchase_success(ChallengeCount(1, 5))
    assert ctrl.ctx.recovery_attempts == 0
    assert ctrl.ctx.challenge.current == 1


def test_verified_round_exit_resets_budget_but_missing_exit_does_not(tmp_path):
    ctrl = controller(tmp_path)
    ctrl.ctx.recovery_attempts = 2
    ctrl.sm.transition(AutomationState.EXIT_RESULT)
    ctrl.buttons.find_exit = MagicMock(return_value=DetectionResult(found=False))
    ctrl._state_exit_result()
    assert ctrl.ctx.recovery_attempts == 2
    ctrl.buttons.find_exit.return_value = DetectionResult(found=True, confidence=0.95, bbox=(1000, 630, 160, 60))
    ctrl.challenge_reader.read = MagicMock(return_value=ChallengeCount(1, 5))
    ctrl._state_exit_result()
    assert ctrl.ctx.recovery_attempts == 0
    assert ctrl.sm.state == AutomationState.CHECK_CHALLENGE_COUNT


def test_active_run_keeps_a_config_snapshot_and_logs_mode(tmp_path):
    ctrl = controller(tmp_path, automatic=True)
    messages = []
    ctrl.callbacks = ControllerCallbacks(on_log=lambda level, text: messages.append(text))
    def change_external_config_during_run():
        ctrl.config_manager.config.automation.auto_targeting = False
        ctrl.config_manager.config.automation.power_threshold = 1
        ctrl.reload_config()  # Must not switch modes or clear settings while active.
        assert ctrl.config.automation.auto_targeting is True
        assert ctrl.config.automation.power_threshold == 700000
        assert ctrl.detector.config is ctrl.config
        assert ctrl.auto_targeting.config is ctrl.config
        ctrl.stop_event.set()
    ctrl._step = change_external_config_during_run
    ctrl.run()
    assert ctrl.sm.state == AutomationState.STOPPED
    assert any("本轮模式：自动初始化" in text for text in messages)
    assert not ctrl.config_manager.config.automation.auto_targeting
    ctrl.reload_config()
    assert not ctrl.config.automation.auto_targeting  # New values apply after stopping.


def test_early_stop_request_is_not_cleared_when_worker_run_starts(tmp_path):
    ctrl = controller(tmp_path)
    ctrl.request_stop()
    ctrl.run()
    ctrl.adb.ensure_connected.assert_not_called()
    ctrl.adb.screenshot.assert_not_called()
    assert ctrl.sm.state == AutomationState.STOPPED
    assert ctrl.statistics.recognition_failures == 0


def test_stop_is_not_counted_as_ocr_failure_or_recovery(tmp_path):
    ctrl = controller(tmp_path)
    ctrl.power_reader.read_until_at_or_below = MagicMock(side_effect=OCRFailedError("停止请求"))
    ctrl.stop_event.set()
    with pytest.raises(StoppedError):
        ctrl._state_select_opponent()
    assert ctrl.statistics.recognition_failures == 0
    assert ctrl.sm.state != AutomationState.RECOVERY
    with pytest.raises(StoppedError):
        ctrl._state_recovery()
    assert ctrl.ctx.recovery_attempts == 0


def test_manual_click_coordinates_are_scaled_but_live_auto_coordinates_are_not(tmp_path):
    ctrl = controller(tmp_path)
    ctrl.scaler.update_actual(1600, 720)
    selected = Opponent(1, 255276, click_position=(1361, 318))
    ctrl.power_reader.read_until_at_or_below = MagicMock(return_value=([selected], selected))
    ctrl._safe_tap = MagicMock()
    ctrl._state_select_opponent()
    assert ctrl._safe_tap.call_args.args[:2] == (1134, 212)
    assert selected.click_position == (1361, 318)  # Saved/configured point is not rewritten.
    ctrl.config.automation.auto_targeting = True
    selected = Opponent(1, 255276, click_position=(1209, 159))
    ctrl.auto_targeting.select = MagicMock(return_value=([selected], selected))
    ctrl.auto_targeting.verify_target = MagicMock(return_value=True)
    ctrl._state_select_opponent()
    assert ctrl._safe_tap.call_args.args[:2] == (1209, 159)


def test_power_raw_retry_bypasses_preprocessing_and_failed_cache(tmp_path):
    ctrl = controller(tmp_path)
    engine = MagicMock()
    engine.read_roi.side_effect = [([], ""), ([OCRBox("전투력:462,892", 0.95, (0, 0, 180, 25))], "전투력:462,892")]
    reader = PowerReader(MagicMock(), ctrl.config)
    reader.engine = engine
    value, _ = reader._read_once_debug(np.zeros((720, 1600, 3), dtype=np.uint8), (1300, 450, 250, 60), 2,
                                     allow_raw_fallback=True)
    assert value == 462892
    assert engine.read_roi.call_args.kwargs["preprocess"] is False
    assert engine.read_roi.call_args.kwargs["force"] is True
    assert engine.read_roi.call_args.kwargs["cache_key"] is None


def test_korean_power_reader_retries_raw_roi_without_top_padding(tmp_path):
    ctrl = controller(tmp_path)
    ctrl.config.ocr.lang = "korean"
    engine = MagicMock()
    label_error = ([OCRBox("전력:100.761", 0.95, (0, 0, 180, 25))], "전력:100.761")
    engine.read_roi.side_effect = [
        label_error,  # Configured/preprocessed crop.
        label_error,  # Existing raw-image retry.
        ([OCRBox("전투력:100,761", 0.96, (0, 0, 180, 25))], "전투력:100,761"),
    ]
    reader = PowerReader(MagicMock(), ctrl.config)
    reader.engine = engine
    image = np.zeros((720, 1600, 3), dtype=np.uint8)

    value, raw = reader._read_once_debug(
        image, (1537, 204, 252, 60), 1, allow_raw_fallback=True
    )

    assert value == 100761
    assert raw == "전투력:100,761"
    assert parse_power("전력:100,761") is None  # No extra label alias.
    assert engine.read_roi.call_count == 3
    _, framed_region = engine.read_roi.call_args.args[:2]
    assert framed_region == (1281, 144, 210, 32)
    assert engine.read_roi.call_args.kwargs == {
        "allowlist": ctrl.config.ocr.power_allowlist,
        "preprocess": False,
        "cache_key": None,
        "force": True,
    }


def test_form_selection_is_bound_to_manager_before_controller_creation(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from arena_auto.gui.main_window import MainWindow
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(MainWindow, "refresh_devices", lambda self: None)
    monkeypatch.setattr(MainWindow, "_attach_log_handler", lambda self: None)
    manager = ConfigManager(tmp_path / "config.yaml")
    ui = MainWindow(manager)
    try:
        ui.config = copy.deepcopy(manager.config)
        ui.auto_target_check.setChecked(True)
        assert ui._apply_form_to_config()
        assert manager.config is ui.config
        assert manager.config.automation.auto_targeting
        monkeypatch.setattr(ui, "_automation_running", lambda: True)
        ui.auto_target_check.setChecked(False)
        assert not ui._apply_form_to_config()
        assert manager.config.automation.auto_targeting
    finally:
        ui.close()
        app.processEvents()
