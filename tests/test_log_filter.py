"""GUI log-page level filter: display only, bounded history, clear behavior."""

import logging

import pytest


@pytest.fixture
def ui(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from arena_auto.config.manager import ConfigManager
    from arena_auto.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(MainWindow, "refresh_devices", lambda self: None)
    monkeypatch.setattr(MainWindow, "_attach_log_handler", lambda self: None)
    window = MainWindow(ConfigManager.load_or_create(tmp_path / "config.yaml"))
    yield window
    window.close()
    app.processEvents()


def select_level(ui, level):
    index = ui.log_level_combo.findData(level)
    assert index >= 0
    ui.log_level_combo.setCurrentIndex(index)


def test_log_filter_applies_to_new_and_buffered_lines(ui):
    ui._clear_log()
    for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
        ui._append_log(level, f"event-{level}")
    assert "event-DEBUG" in ui.log_view.toPlainText()

    select_level(ui, logging.WARNING)
    text = ui.log_view.toPlainText()
    assert "event-WARNING" in text and "event-ERROR" in text
    assert "event-DEBUG" not in text and "event-INFO" not in text

    select_level(ui, logging.DEBUG)
    text = ui.log_view.toPlainText()
    for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
        assert f"event-{level}" in text


def test_clear_log_discards_filter_history(ui):
    ui._clear_log()
    ui._append_log("ERROR", "before-clear")
    ui._clear_log()
    select_level(ui, logging.DEBUG)
    assert "before-clear" not in ui.log_view.toPlainText()
    assert not ui._log_entries


def test_unknown_level_is_treated_as_info(ui):
    ui._clear_log()
    select_level(ui, logging.WARNING)
    ui._append_log("nonsense", "unknown-level-message")
    assert "unknown-level-message" not in ui.log_view.toPlainText()

    select_level(ui, logging.INFO)
    assert "unknown-level-message" in ui.log_view.toPlainText()
