"""Exit saves preserve user work; typed logs are filtered separately from severity."""

import logging
from unittest.mock import MagicMock

import pytest
import yaml

from arena_auto.config.manager import ConfigManager
from arena_auto.exceptions import ConfigError
from arena_auto.logging.categories import ADB_TIMING, OCR_TIMING, FLOW_TIMING, GENERAL, CategoryFormatter
from arena_auto.logging.logger import CallbackHandler, LOG_FORMAT


@pytest.fixture
def ui(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QMessageBox
    from arena_auto.gui.main_window import MainWindow
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QApplication, "beep", MagicMock())
    monkeypatch.setattr(QMessageBox, "warning", MagicMock())
    monkeypatch.setattr(QMessageBox, "critical", MagicMock())
    monkeypatch.setattr(QMessageBox, "question", MagicMock(return_value=QMessageBox.Yes))
    monkeypatch.setattr(MainWindow, "refresh_devices", lambda self: None)
    monkeypatch.setattr(MainWindow, "_attach_log_handler", lambda self: None)
    manager = ConfigManager.load_or_create(tmp_path / "config.yaml")
    manager.config.adb.serial = "kept-fake-device"
    manager.save()
    window = MainWindow(manager)
    window._sound_player = MagicMock()
    yield window
    window._automation_thread = None
    window._device_worker = window._shot_worker = window._test_worker = window._calibration_worker = None
    window.close()
    app.processEvents()


def close(ui):
    event = MagicMock()
    ui.closeEvent(event)
    return event


def test_close_saves_unsaved_form_sound_refresh_and_log_preferences(ui):
    ui.threshold_edit.setText("3456789")
    ui.refresh_limit_spin.setValue(7)
    ui.completion_sound_check.setChecked(False)
    ui.sound_path_edit.setText("我的提示.mp3")
    ui.log_level_combo.setCurrentIndex(ui.log_level_combo.findData(logging.INFO))
    ui.log_type_combo.setCurrentIndex(ui.log_type_combo.findData(ADB_TIMING))
    event = close(ui)
    event.accept.assert_called_once()
    cfg = ConfigManager(ui.config_manager.path).load()
    assert cfg.automation.power_threshold == 3456789
    assert cfg.automation.max_consecutive_refreshes == 7
    assert not cfg.completion_sound.enabled and cfg.completion_sound.path == "我的提示.mp3"
    assert cfg.logging.gui_level == "INFO" and cfg.logging.gui_category == ADB_TIMING
    assert cfg.adb.serial == "kept-fake-device"  # No device list must not erase the serial.
    ui._sound_player.play.assert_not_called()


def test_close_without_edits_does_not_round_or_rewrite_config(ui):
    ui.config.battle.poll_interval = 5.5
    ui.config_manager.save(ui.config)
    ui._load_config_into_form()
    before = ui.config_manager.path.read_bytes()
    close(ui).accept.assert_called_once()
    assert ui.config_manager.path.read_bytes() == before
    assert ConfigManager(ui.config_manager.path).load().battle.poll_interval == 5.5


def test_close_preserves_external_geometry_templates_and_unknown_fields(ui):
    path = ui.config_manager.path
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["opponents"][0]["power_region"] = [100, 100, 120, 25]
    raw["opponents"][0]["click"] = [200, 100]
    raw["templates"]["user_template"] = {"path": "private-template.png", "threshold": .9}
    raw["user_extra"] = {"keep": "unchanged"}
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    ui.threshold_edit.setText("3000000")
    close(ui).accept.assert_called_once()
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved["automation"]["power_threshold"] == 3000000
    assert saved["opponents"][0] == raw["opponents"][0]
    assert saved["templates"]["user_template"] == raw["templates"]["user_template"]
    assert saved["user_extra"] == raw["user_extra"]


def test_exit_save_does_not_downgrade_external_version(ui):
    path = ui.config_manager.path
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["version"] = 3
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    ui.threshold_edit.setText("3000000")
    close(ui).accept.assert_called_once()
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["version"] == 3


def test_conflicting_external_setting_cancels_exit_without_overwriting(ui):
    from PySide6.QtWidgets import QMessageBox
    path = ui.config_manager.path
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["automation"]["power_threshold"] = 4000000
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    before = path.read_bytes()
    ui.threshold_edit.setText("3000000")
    event = close(ui)
    event.ignore.assert_called_once()
    event.accept.assert_not_called()
    QMessageBox.critical.assert_called_once()
    assert path.read_bytes() == before and not ui._closing


def test_known_internal_save_updates_exit_merge_baseline(ui):
    ui.threshold_edit.setText("3000000")
    assert ui._apply_form_to_config()
    ui.config_manager.save()  # Simulates a successful in-app save while the form stays loaded.
    ui.threshold_edit.setText("4000000")
    close(ui).accept.assert_called_once()
    assert ConfigManager(ui.config_manager.path).load().automation.power_threshold == 4000000


def test_invalid_form_cancels_exit_and_keeps_config(ui):
    before = ui.config_manager.path.read_bytes()
    ui.threshold_edit.setText("invalid-number")
    event = close(ui)
    event.ignore.assert_called_once()
    assert ui.config_manager.path.read_bytes() == before
    assert not ui._closing


def test_atomic_replace_failure_keeps_old_file_and_removes_temporary(ui, monkeypatch):
    from arena_auto.config import manager
    before = ui.config_manager.path.read_bytes()
    ui.threshold_edit.setText("3000000")
    monkeypatch.setattr(manager.os, "replace", MagicMock(side_effect=PermissionError("locked")))
    event = close(ui)
    event.ignore.assert_called_once()
    assert ui.config_manager.path.read_bytes() == before
    assert not list(ui.config_manager.path.parent.glob(".arena-config-*.tmp"))


def test_file_changing_during_save_is_not_overwritten(ui, monkeypatch):
    from arena_auto.config import manager
    path = ui.config_manager.path
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["battle"]["poll_interval"] = 9
    external_text = yaml.safe_dump(raw)
    monkeypatch.setattr(manager.os, "fsync", lambda _: path.write_text(external_text, encoding="utf-8"))
    ui.threshold_edit.setText("3000000")
    close(ui).ignore.assert_called_once()
    assert path.read_text(encoding="utf-8") == external_text
    assert not list(path.parent.glob(".arena-config-*.tmp"))


def test_running_thread_must_stop_before_exit_save(ui, monkeypatch):
    order = []
    thread = MagicMock()
    thread.isRunning.return_value = True

    def waited(_):
        order.append("wait")
        thread.isRunning.return_value = False
        return True

    thread.wait.side_effect = waited
    ui._automation_thread = thread
    ui._automation_worker = MagicMock()
    original = ui.config_manager.save_ui_settings

    def saved(*args):
        order.append("save")
        return original(*args)

    monkeypatch.setattr(ui.config_manager, "save_ui_settings", saved)
    ui.threshold_edit.setText("3000000")
    event = close(ui)
    assert order == ["wait", "save"]
    ui._automation_worker.request_stop.assert_called_once()
    event.accept.assert_called_once()
    ui._sound_player.play.assert_not_called()


def test_stop_timeout_cancels_exit_without_saving(ui):
    before = ui.config_manager.path.read_bytes()
    ui._automation_thread = MagicMock()
    ui._automation_thread.isRunning.return_value = True
    ui._automation_thread.wait.return_value = False
    ui._automation_worker = MagicMock()
    ui._completion_pending = True
    ui.threshold_edit.setText("3000000")
    close(ui).ignore.assert_called_once()
    assert ui.config_manager.path.read_bytes() == before
    assert ui._completion_pending and not ui._closing


def test_busy_debug_worker_prevents_exit_save(ui):
    before = ui.config_manager.path.read_bytes()
    ui._shot_worker = MagicMock()
    ui._shot_worker.isRunning.return_value = True
    ui._shot_worker.wait.return_value = False
    ui.threshold_edit.setText("3000000")
    close(ui).ignore.assert_called_once()
    assert ui.config_manager.path.read_bytes() == before


def test_log_preferences_restore_when_form_is_loaded(ui):
    ui.config.logging.gui_level = "WARNING"
    ui.config.logging.gui_category = OCR_TIMING
    ui._load_config_into_form()
    assert ui.log_level_combo.currentData() == logging.WARNING
    assert ui.log_type_combo.currentData() == OCR_TIMING


def test_existing_empty_yaml_can_save_a_changed_setting(tmp_path):
    manager = ConfigManager(tmp_path / "config.yaml")
    manager.path.write_text("", encoding="utf-8")
    baseline = manager.load()
    import copy
    desired = copy.deepcopy(baseline)
    desired.automation.power_threshold = 3000000
    saved = manager.save_ui_settings(desired, baseline, [("automation", "power_threshold")])
    assert saved.automation.power_threshold == 3000000


def test_exit_merge_refuses_non_ui_fields(ui):
    with pytest.raises(ConfigError, match="界面设置"):
        ui.config_manager.save_ui_settings(ui.config, ui.config, [("adb", "executable")])


def test_types_and_severity_filter_independently_with_history(ui):
    ui._clear_log()
    ui._append_log("INFO", "business-info", GENERAL)
    ui._append_log("INFO", "adb-time", ADB_TIMING)
    ui._append_log("INFO", "ocr-time", OCR_TIMING)
    ui._append_log("DEBUG", "flow-time", FLOW_TIMING)
    ui._append_log("ERROR", "business-error", GENERAL)
    ui.log_type_combo.setCurrentIndex(ui.log_type_combo.findData(ADB_TIMING))
    text = ui.log_view.toPlainText()
    assert "[ADB耗时]" in text and "adb-time" in text
    assert "business-info" not in text and "ocr-time" not in text
    ui.log_level_combo.setCurrentIndex(ui.log_level_combo.findData(logging.ERROR))
    assert not ui.log_view.toPlainText()
    ui.log_type_combo.setCurrentIndex(ui.log_type_combo.findData("all"))
    assert "business-error" in ui.log_view.toPlainText() and "adb-time" not in ui.log_view.toPlainText()
    ui.log_level_combo.setCurrentIndex(ui.log_level_combo.findData(logging.DEBUG))
    assert all(name in ui.log_view.toPlainText() for name in ("business-info", "adb-time", "ocr-time", "flow-time"))


def test_general_filter_does_not_guess_category_from_adb_words(ui):
    ui._clear_log()
    ui._append_log("INFO", "ordinary message mentions ADB截图耗时", GENERAL)
    ui._append_log("INFO", "real adb event", ADB_TIMING)
    ui.log_type_combo.setCurrentIndex(ui.log_type_combo.findData(GENERAL))
    assert "ordinary message" in ui.log_view.toPlainText() and "real adb event" not in ui.log_view.toPlainText()
    ui._clear_log()
    ui.log_type_combo.setCurrentIndex(ui.log_type_combo.findData("all"))
    assert not ui.log_view.toPlainText()


def record(category=GENERAL, level=logging.INFO):
    result = logging.LogRecord("arena_auto.test", level, __file__, 1, "event", (), None)
    result.log_category = category
    return result


def test_callback_handler_preserves_legacy_api_and_exposes_structured_type():
    legacy = MagicMock()
    CallbackHandler(legacy).emit(record(ADB_TIMING))
    legacy.assert_called_once_with("INFO", "[INFO] [ADB耗时] event")
    categorized = MagicMock()
    handler = CallbackHandler(legacy, categorized_callback=categorized)
    handler.emit(record(OCR_TIMING, logging.WARNING))
    categorized.assert_called_once_with("WARNING", "[WARNING] [OCR耗时] event", OCR_TIMING)
    assert legacy.call_count == 1


@pytest.mark.parametrize("category", [None, "unknown", [], {}])
def test_unknown_record_type_is_safe_general(category):
    callback = MagicMock()
    CallbackHandler(MagicMock(), categorized_callback=callback).emit(record(category))
    callback.assert_called_once_with("INFO", "[INFO] event", GENERAL)


def test_gui_filter_does_not_filter_file_output(ui, tmp_path):
    ui._clear_log()
    ui.log_type_combo.setCurrentIndex(ui.log_type_combo.findData(GENERAL))
    path = tmp_path / "types.log"
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(CategoryFormatter(LOG_FORMAT))
    callback = CallbackHandler(MagicMock(), categorized_callback=ui._append_log)
    for category in (GENERAL, ADB_TIMING, OCR_TIMING, FLOW_TIMING):
        item = record(category)
        callback.emit(item)
        file_handler.emit(item)
    file_handler.close()
    text = path.read_text(encoding="utf-8")
    assert "[ADB耗时]" in text and "[OCR耗时]" in text and "[流程/识别耗时]" in text
    assert "[ADB耗时]" not in ui.log_view.toPlainText()


def test_actual_fake_adb_timing_records_are_typed(caplog, monkeypatch):
    from test_latency_diagnostics import fake_adb
    adb, _, _ = fake_adb(monkeypatch)
    with caplog.at_level(logging.INFO):
        adb.screenshot()
        adb.tap(10, 20)
    timings = [r for r in caplog.records if "ADB截图耗时" in r.getMessage() or "ADB点击耗时" in r.getMessage()]
    assert len(timings) == 2 and all(r.log_category == ADB_TIMING for r in timings)


def test_controller_screenshot_workflow_and_ocr_perf_have_types(tmp_path, caplog):
    from test_opponent_refresh import controller
    from arena_auto.recognition import ocr_engine
    ctrl, _, _ = controller(tmp_path)
    previous = ocr_engine.perf_enabled()
    try:
        ocr_engine.set_performance_logging(True)
        with caplog.at_level(logging.DEBUG):
            ctrl._shot()
            ocr_engine.perf_log("OCR test", 12, category=OCR_TIMING)
        assert any("截图流程耗时" in r.getMessage() and r.log_category == ADB_TIMING for r in caplog.records)
        assert any("OCR test" in r.getMessage() and r.log_category == OCR_TIMING for r in caplog.records)
    finally:
        ocr_engine.set_performance_logging(previous)


def test_worker_forwards_category_without_changing_legacy_logs(ui):
    from arena_auto.gui.workers import AutomationWorker
    ctrl = MagicMock()
    worker = AutomationWorker(ctrl)
    worker.categorized_log_message.connect(ui._append_log)
    ui._clear_log()
    ctrl.callbacks.on_categorized_log("INFO", "categorized event", FLOW_TIMING)
    assert ui._log_entries[-1][3] == FLOW_TIMING
    assert "[流程/识别耗时]" in ui.log_view.toPlainText()


@pytest.mark.parametrize("value", [{"gui_level": 3}, {"gui_level": "TRACE"}, {"gui_category": "bad"}])
def test_invalid_gui_log_preferences_rejected(value):
    with pytest.raises(ConfigError):
        ConfigManager.from_dict({"logging": value})
