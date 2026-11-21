"""Capped refresh clicks must never follow OCR failure or trigger purchase confirmation."""

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from arena_auto.automation.controller import AutomationController
from arena_auto.config.manager import ConfigManager
from arena_auto.exceptions import AdbError, ConfigError, OCRFailedError, StoppedError
from arena_auto.models.config import AppConfig, ButtonGeometry, OpponentConfig
from arena_auto.models.recognition import DetectionResult, Opponent
from arena_auto.models.state import AutomationState, ScreenKind
from arena_auto.recognition.auto_targeting import OpponentCard
from arena_auto.recognition.detector import ButtonFinder
from arena_auto.recognition.template import TemplateRecognizer


def controller(tmp_path, automatic=True):
    manager = ConfigManager(tmp_path / "config.yaml")
    cfg = manager.config
    cfg.automation.auto_targeting = automatic
    cfg.automation.click_delay = 0
    cfg.screen.reference_width, cfg.screen.reference_height = 800, 360
    cfg.debug.enabled = False
    cfg.opponents = [OpponentConfig(id=i, power_region=(600, 80 + (i - 1) * 70, 80, 20), click=(400, 100 + (i - 1) * 70)) for i in range(1, 4)]
    ctrl = AutomationController(manager, adb=MagicMock(), ocr=MagicMock(), templates=MagicMock())
    ctrl._wait = MagicMock(return_value=False)
    image = np.full((360, 800, 3), 100, dtype=np.uint8)
    ctrl.ctx.screenshot = image
    ctrl.adb.screenshot.side_effect = lambda: image.copy()
    ctrl.detector.detect = MagicMock(return_value=ScreenKind.ARENA)
    ctrl.buttons.find_refresh = MagicMock(return_value=DetectionResult(True, .99, (680, 20, 35, 28)))
    opponents = [Opponent(c.id, 3000000 + c.id, c.power_region, c.click) for c in cfg.opponents]
    ctrl.auto_targeting.last_image = image
    ctrl.auto_targeting.last_cards = [OpponentCard(o, (360, 70 + (o.id - 1) * 70, 360, 60), f"player{o.id}", "test", True) for o in opponents]
    ctrl.auto_targeting.select = MagicMock(return_value=(opponents, None))
    ctrl.auto_targeting.verify_target = MagicMock(return_value=True)
    ctrl.power_reader.read_until_at_or_below = MagicMock(return_value=(opponents, None))
    return ctrl, opponents, image


@pytest.mark.parametrize("automatic", [True, False])
def test_all_high_refreshes_once_then_waits_for_new_list(tmp_path, automatic):
    ctrl, _, _ = controller(tmp_path, automatic)
    ctrl._state_select_opponent()
    assert ctrl.ctx.consecutive_refreshes == 1
    assert ctrl.sm.state == AutomationState.WAIT_REFRESH
    ctrl.adb.tap.assert_called_once_with(697, 34, delay=0)
    assert ctrl.adb.screenshot.call_count == 2  # Discovery and fresh pre-click check.


def test_third_click_stops_immediately_without_last_batch_check(tmp_path):
    ctrl, opponents, _ = controller(tmp_path)
    for index in range(3):
        ctrl._refresh_opponents(opponents)
        assert ctrl.ctx.consecutive_refreshes == index + 1
    assert ctrl.adb.tap.call_count == 3
    assert ctrl.adb.screenshot.call_count == 6
    assert ctrl.sm.state == AutomationState.STOPPED
    ctrl._refresh_opponents(opponents)
    assert ctrl.adb.tap.call_count == 3


def test_configured_limit_one_stops_after_first_click(tmp_path):
    ctrl, opponents, _ = controller(tmp_path)
    ctrl.config.automation.max_consecutive_refreshes = 1
    ctrl._refresh_opponents(opponents)
    assert ctrl.sm.state == AutomationState.STOPPED and ctrl.ctx.consecutive_refreshes == 1


def test_disabled_refresh_retains_wait_without_click(tmp_path):
    ctrl, _, _ = controller(tmp_path)
    ctrl.config.automation.max_consecutive_refreshes = 0
    ctrl._state_select_opponent()
    ctrl.adb.tap.assert_not_called()
    ctrl.buttons.find_refresh.assert_not_called()
    assert ctrl.sm.state == AutomationState.ARENA


@pytest.mark.parametrize("power", [None, 2000000, 100])
def test_unknown_or_eligible_opponent_never_refreshes(tmp_path, power):
    ctrl, opponents, _ = controller(tmp_path)
    opponents[0].power = power
    ctrl._refresh_opponents(opponents)
    ctrl.adb.tap.assert_not_called()
    assert ctrl.ctx.consecutive_refreshes == 0


@pytest.mark.parametrize("kind", ["empty", "gap", "unverified", "missing_card", "missing_manual"])
def test_incomplete_list_does_not_refresh(tmp_path, kind):
    ctrl, opponents, _ = controller(tmp_path, automatic=kind != "missing_manual")
    if kind == "empty":
        opponents = []
    elif kind == "gap":
        opponents = opponents[1:]
    elif kind == "unverified":
        ctrl.auto_targeting.last_cards[0].power_verified = False
    elif kind == "missing_card":
        ctrl.auto_targeting.last_cards.pop()
    else:
        opponents = opponents[:2]
    ctrl._refresh_opponents(opponents)
    ctrl.adb.tap.assert_not_called()
    assert ctrl.sm.state == AutomationState.STOPPED


def test_ocr_failure_never_increments_or_refreshes(tmp_path):
    ctrl, _, _ = controller(tmp_path)
    ctrl.ctx.consecutive_refreshes = 1
    ctrl.auto_targeting.select.side_effect = OCRFailedError("unknown power")
    ctrl._state_select_opponent()
    ctrl.adb.tap.assert_not_called()
    assert ctrl.ctx.consecutive_refreshes == 1


def test_found_eligible_resets_counter_and_selects_opponent(tmp_path):
    ctrl, opponents, _ = controller(tmp_path)
    ctrl.ctx.consecutive_refreshes = 2
    opponents[0].power = 2000000  # Equality remains eligible.
    ctrl.auto_targeting.select.return_value = ([opponents[0]], opponents[0])
    ctrl._state_select_opponent()
    assert ctrl.ctx.consecutive_refreshes == 0
    ctrl.buttons.find_refresh.assert_not_called()
    assert ctrl.sm.state == AutomationState.WAIT_PREPARE


@pytest.mark.parametrize("mode", ["page", "missing_button", "moved_button", "number_changed", "old_number_changed", "card_changed", "missing_frame"])
def test_preclick_changes_cancel_refresh(tmp_path, mode):
    ctrl, opponents, image = controller(tmp_path)
    if mode == "page":
        ctrl.detector.detect.return_value = ScreenKind.PURCHASE
    elif mode == "missing_button":
        ctrl.buttons.find_refresh.return_value = DetectionResult(False)
    elif mode == "moved_button":
        ctrl.buttons.find_refresh.side_effect = [DetectionResult(True, .99, (680, 20, 35, 28)), DetectionResult(True, .99, (690, 20, 35, 28))]
    elif mode in ("number_changed", "old_number_changed"):
        changed = image.copy()
        changed[80:100, 600:620] = 220
        ctrl.adb.screenshot.side_effect = [image.copy(), changed] if mode == "number_changed" else [changed, changed.copy()]
    elif mode == "card_changed":
        ctrl.auto_targeting.verify_target.return_value = False
    else:
        ctrl.auto_targeting.last_image = None
    ctrl._refresh_opponents(opponents)
    ctrl.adb.tap.assert_not_called()
    assert ctrl.ctx.consecutive_refreshes == 0 and ctrl.sm.state == AutomationState.STOPPED


def test_stop_before_refresh_tap_does_not_consume_budget(tmp_path):
    ctrl, opponents, image = controller(tmp_path)

    def shot():
        ctrl.stop_event.set()
        return image.copy()

    ctrl.adb.screenshot.side_effect = shot
    with pytest.raises(StoppedError):
        ctrl._refresh_opponents(opponents)
    ctrl.adb.tap.assert_not_called()
    assert ctrl.ctx.consecutive_refreshes == 0


def test_uncertain_tap_stops_without_recovery_retry_or_counter_increment(tmp_path):
    ctrl, opponents, _ = controller(tmp_path)
    ctrl.adb.tap.side_effect = AdbError("timeout may already have clicked")
    ctrl._refresh_opponents(opponents)
    ctrl.adb.tap.assert_called_once()
    assert ctrl.ctx.consecutive_refreshes == 0 and ctrl.sm.state == AutomationState.STOPPED


def test_dry_run_logs_only_and_stops_without_consuming_budget(tmp_path):
    ctrl, opponents, _ = controller(tmp_path)
    ctrl.config.automation.dry_run = True
    ctrl._refresh_opponents(opponents)
    ctrl.adb.tap.assert_not_called()
    assert ctrl.ctx.consecutive_refreshes == 0 and ctrl.sm.state == AutomationState.STOPPED


def test_wait_ignores_avatar_animation_and_never_reclicks(tmp_path):
    ctrl, opponents, image = controller(tmp_path)
    ctrl._refresh_opponents(opponents)
    animated = image.copy()
    animated[100:130, 400:440] = 230
    ctrl.adb.screenshot.side_effect = lambda: animated.copy()
    ctrl._state_wait_refresh()
    assert ctrl.sm.state == AutomationState.WAIT_REFRESH
    assert ctrl.adb.tap.call_count == 1


def test_wait_changed_number_region_returns_to_arena_for_fresh_reading(tmp_path):
    ctrl, opponents, image = controller(tmp_path)
    ctrl._refresh_opponents(opponents)
    changed = image.copy()
    changed[80:100, 600:620] = 220
    ctrl.adb.screenshot.side_effect = lambda: changed.copy()
    ctrl._state_wait_refresh()
    assert ctrl.sm.state == AutomationState.ARENA
    assert ctrl.ctx.consecutive_refreshes == 1
    assert ctrl.ctx.refresh_frame is None
    assert ctrl.adb.tap.call_count == 1


@pytest.mark.parametrize("mode", ["purchase", "timeout", "resolution"])
def test_wait_failure_stops_without_purchasing_or_reclicking(tmp_path, mode):
    ctrl, opponents, _ = controller(tmp_path)
    ctrl.config.purchase.enabled = ctrl.config.purchase.auto_confirm = True
    ctrl._refresh_opponents(opponents)
    if mode == "purchase":
        ctrl.detector.detect.return_value = ScreenKind.PURCHASE
    elif mode == "timeout":
        ctrl.sm._entered_at -= 100
    else:
        ctrl.adb.screenshot.side_effect = lambda: np.zeros((180, 400, 3), dtype=np.uint8)
    ctrl._state_wait_refresh()
    assert ctrl.sm.state == AutomationState.STOPPED
    assert ctrl.adb.tap.call_count == 1 and ctrl.statistics.session_purchase_count == 0


@pytest.mark.parametrize("failure", [AdbError("offline"), OCRFailedError("OCR failed"), RuntimeError("provider failed")])
def test_wait_errors_cannot_escape_into_purchase_recovery(tmp_path, failure):
    ctrl, opponents, _ = controller(tmp_path)
    ctrl.config.purchase.enabled = ctrl.config.purchase.auto_confirm = True
    ctrl._refresh_opponents(opponents)
    ctrl.detector.detect.side_effect = failure
    ctrl._state_wait_refresh()
    assert ctrl.sm.state == AutomationState.STOPPED
    assert ctrl.adb.tap.call_count == 1


def test_dimmed_arena_background_is_not_accepted_as_a_new_list(tmp_path):
    ctrl, opponents, image = controller(tmp_path)
    ctrl.config.purchase.enabled = ctrl.config.purchase.auto_confirm = True
    ctrl._refresh_opponents(opponents)
    ctrl.adb.screenshot.side_effect = lambda: (image * .5).astype(np.uint8)
    ctrl._state_wait_refresh()
    assert ctrl.sm.state == AutomationState.WAIT_REFRESH
    assert ctrl.adb.tap.call_count == 1


@pytest.mark.parametrize("value", [-1, True, False, 1.5, "3.0", "bad", None, 1000])
def test_invalid_refresh_limits_rejected(value):
    with pytest.raises(ConfigError, match="max_consecutive_refreshes"):
        ConfigManager.from_dict({"automation": {"max_consecutive_refreshes": value}})


def test_default_and_refresh_limit_round_trip(tmp_path):
    cfg = ConfigManager.from_dict({})
    assert cfg.automation.max_consecutive_refreshes == 3
    manager = ConfigManager(tmp_path / "config.yaml")
    cfg.automation.max_consecutive_refreshes = 7
    manager.save(cfg)
    assert manager.load().automation.max_consecutive_refreshes == 7


def test_refresh_finder_never_uses_countdown_ocr_and_preserves_manual_roi():
    cfg = AppConfig()
    cfg.screen.reference_width, cfg.screen.reference_height = 800, 360
    templates, provider = MagicMock(), MagicMock()
    templates.find.return_value = DetectionResult(True, .99, (680, 20, 35, 28))
    finder = ButtonFinder(templates, provider, cfg)
    image = np.zeros((360, 800, 3), dtype=np.uint8)
    assert finder.find_refresh(image).found
    provider.recognize.assert_not_called()
    cfg.buttons["refresh_opponents"] = ButtonGeometry(region=(0, 0, 100, 100))
    assert not finder.find_refresh(image).found
    templates.find.return_value = DetectionResult(True, .90, (680, 20, 35, 28))
    assert not finder.find_refresh(image).found


def test_bundled_refresh_template_matches_icon_not_back_or_home():
    cfg = AppConfig()
    recognizer = TemplateRecognizer(cfg.templates, base_dir=Path(__file__).resolve().parents[1])
    image = np.full((360, 800, 3), 60, dtype=np.uint8)
    icon = recognizer.load("refresh_opponents")
    assert icon is not None
    image[20:20 + icon.shape[0], 650:650 + icon.shape[1]] = icon
    finder = ButtonFinder(recognizer, MagicMock(), cfg)
    assert finder.find_refresh(image).found
    blank = np.full_like(image, 60)
    assert not finder.find_refresh(blank).found


def test_packaged_template_fallback_preserves_external_override(tmp_path, monkeypatch):
    from arena_auto import paths
    external, internal = tmp_path / "external", tmp_path / "internal"
    monkeypatch.setattr(paths, "get_app_root", lambda: external)
    monkeypatch.setattr(paths.sys, "_MEIPASS", str(internal), raising=False)
    bundled = internal / "resources/templates/refresh_opponents.png"
    bundled.parent.mkdir(parents=True)
    bundled.write_bytes(b"bundled")
    cfg = AppConfig()
    template = TemplateRecognizer(cfg.templates)
    manager = ConfigManager(tmp_path / "config.yaml")
    assert template.resolve_path("refresh_opponents") == bundled
    assert manager.get_template_path("refresh_opponents") == bundled
    override = external / "resources/templates/refresh_opponents.png"
    override.parent.mkdir(parents=True)
    override.write_bytes(b"user override")
    assert template.resolve_path("refresh_opponents") == override
    assert manager.get_template_path("refresh_opponents") == override


def test_refresh_limit_gui_load_save_and_running_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from arena_auto.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(MainWindow, "refresh_devices", lambda self: None)
    monkeypatch.setattr(MainWindow, "_attach_log_handler", lambda self: None)
    manager = ConfigManager.load_or_create(tmp_path / "config.yaml")
    window = MainWindow(manager)
    try:
        assert window.refresh_limit_spin.value() == 3
        window.refresh_limit_spin.setValue(7)
        assert window._apply_form_to_config()
        manager.save()
        assert manager.load().automation.max_consecutive_refreshes == 7
        monkeypatch.setattr(window, "_automation_running", lambda: True)
        window.refresh_limit_spin.setValue(9)
        assert not window._apply_form_to_config()
        assert window.config.automation.max_consecutive_refreshes == 7
        window.refresh_limit_spin.setEnabled(False)
        window._on_worker_finished()
        assert window.refresh_limit_spin.isEnabled()
    finally:
        window.close()
        app.processEvents()
