"""Wireless main-window flow; no actual ADB/device connections."""

from unittest.mock import MagicMock

import pytest

from arena_auto.adb.controller import AdbController
from arena_auto.config.manager import ConfigManager
from arena_auto.exceptions import AdbError
from arena_auto.models.config import AdbConfig


@pytest.fixture
def qt_app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.mark.parametrize("task,code", [("connect", ""), ("connect", "123456"), ("pair", "123456")])
def test_worker_pair_and_connect(qt_app, task, code):
    from arena_auto.gui.wireless_dialog import WirelessConnectionWorker
    config = AdbConfig(serial="USB")
    worker = WirelessConnectionWorker(config, task, "192.168.1.2:39867", "192.168.1.2:37123", code)
    assert worker.adb.config is not config
    worker.adb = MagicMock()
    worker.adb.connect_wireless.return_value = "192.168.1.2:39867"
    results = []
    worker.result_ready.connect(lambda *args: results.append(args))
    worker.run()
    assert worker.code == ""
    assert config.serial == "USB"
    if code:
        worker.adb.pair_wireless.assert_called_once_with("192.168.1.2:37123", code)
    else:
        worker.adb.pair_wireless.assert_not_called()
    if task == "connect":
        worker.adb.connect_wireless.assert_called_once_with("192.168.1.2:39867", verify=True)
        assert results == [("connect", "192.168.1.2:39867")]
    else:
        worker.adb.connect_wireless.assert_not_called()
        assert results == [("pair", "192.168.1.2:37123")]


def test_pair_failure_stops_connect(qt_app):
    from arena_auto.gui.wireless_dialog import WirelessConnectionWorker
    worker = WirelessConnectionWorker(AdbConfig(), "connect", "host:5555", "host:37000", "123456")
    worker.adb = MagicMock()
    worker.adb.pair_wireless.side_effect = AdbError("配对失败")
    errors, results = [], []
    worker.failed.connect(errors.append)
    worker.result_ready.connect(lambda *args: results.append(args))
    worker.run()
    assert errors == ["配对失败"]
    assert not results
    worker.adb.connect_wireless.assert_not_called()


@pytest.mark.parametrize("task,address,pair,code", [
    ("connect", "", "", ""),
    ("connect", "host:99999", "", ""),
    ("pair", "", "host", "123456"),
    ("pair", "", "host:37000", "123"),
])
def test_dialog_invalid_inputs(qt_app, monkeypatch, task, address, pair, code):
    from arena_auto.gui import wireless_dialog
    dialog = wireless_dialog.WirelessConnectionDialog(AdbConfig())
    dialog.endpoint_edit.setText(address)
    dialog.pair_endpoint_edit.setText(pair)
    dialog.code_edit.setText(code)
    warning = MagicMock()
    monkeypatch.setattr(wireless_dialog.QMessageBox, "warning", warning)
    dialog.start_task(task)
    warning.assert_called_once()
    assert dialog._worker is None


def test_dialog_lifecycle(qt_app):
    from PySide6.QtWidgets import QDialog
    from arena_auto.gui.wireless_dialog import WirelessConnectionDialog
    dialog = WirelessConnectionDialog(AdbConfig(serial="host:39867"))
    assert dialog.endpoint_edit.text() == "host:39867"
    dialog._worker = MagicMock()
    worker = dialog._worker
    dialog.code_edit.setText("123456")
    dialog._on_result("pair", "host:37123")
    assert not dialog.code_edit.text()
    assert not dialog.connected_serial
    dialog._on_finished()
    worker.deleteLater.assert_called_once()
    assert dialog.result() != QDialog.Accepted
    dialog._worker = MagicMock()
    dialog._on_result("connect", "host:39867")
    dialog._on_finished()
    assert dialog.result() == QDialog.Accepted


@pytest.mark.parametrize("save_fails", [False, True])
def test_main_selects_and_saves_wireless(qt_app, monkeypatch, tmp_path, save_fails):
    from PySide6.QtWidgets import QDialog
    from arena_auto.gui import main_window
    monkeypatch.setattr(main_window.MainWindow, "refresh_devices", lambda self: None)
    monkeypatch.setattr(main_window.MainWindow, "_attach_log_handler", lambda self: None)
    manager = ConfigManager(tmp_path / "config.yaml")
    window = main_window.MainWindow(manager)
    window.device_combo.addItem("未检测到 Android 设备")
    dialog = MagicMock()
    dialog.exec.return_value = QDialog.Accepted
    dialog.connected_serial = "192.168.1.2:39867"
    monkeypatch.setattr(main_window, "WirelessConnectionDialog", lambda *args: dialog)
    warning = MagicMock()
    monkeypatch.setattr(main_window.QMessageBox, "warning", warning)
    if save_fails:
        monkeypatch.setattr(ConfigManager, "save", MagicMock(side_effect=OSError("write failed")))
    try:
        window.open_wireless_connection()
        if save_fails:
            assert manager.config.adb.serial == ""
            assert window.device_combo.currentText() == "未检测到 Android 设备"
            warning.assert_called_once()
        else:
            assert window.device_combo.currentText() == dialog.connected_serial
            assert manager.config.adb.serial == dialog.connected_serial
            assert window.adb.serial == dialog.connected_serial
            assert ConfigManager.load_or_create(manager.path).config.adb.serial == dialog.connected_serial
    finally:
        window.close()


def test_adb_pair_uses_digits_and_rejects_failed_paired_message(monkeypatch):
    adb = AdbController()
    monkeypatch.setattr(adb, "_base_cmd", lambda: ["adb"])
    run = MagicMock(return_value="Successfully paired to host:37000")
    monkeypatch.setattr(adb, "_run", run)
    assert adb.pair_wireless("host:37000", "123-456") == "host:37000"
    assert run.call_args.args[0] == ["adb", "pair", "host:37000", "123456"]
    run.return_value = "failed to pair: not paired"
    with pytest.raises(AdbError):
        adb.pair_wireless("host:37000", "123456")


def test_connect_failure_does_not_emit_success(qt_app):
    from arena_auto.gui.wireless_dialog import WirelessConnectionWorker
    worker = WirelessConnectionWorker(AdbConfig(serial="USB"), "connect", "host:5555")
    worker.adb = MagicMock()
    worker.adb.connect_wireless.side_effect = AdbError("设备未就绪")
    errors, results = [], []
    worker.failed.connect(errors.append)
    worker.result_ready.connect(lambda *args: results.append(args))
    worker.run()
    assert errors == ["设备未就绪"]
    assert not results


def test_pair_timeout_does_not_log_code(monkeypatch, caplog):
    import logging
    import subprocess
    adb = AdbController()
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 15)
    monkeypatch.setattr(subprocess, "run", timeout)
    with caplog.at_level(logging.DEBUG, logger="arena_auto.adb.controller"):
        with pytest.raises(AdbError) as error:
            adb._run(["adb", "pair", "host:37000", "123456"])
    assert "123456" not in caplog.text
    assert "123456" not in str(error.value)


@pytest.mark.parametrize("running", [False, True])
def test_cancel_or_running_preserves_device(qt_app, monkeypatch, tmp_path, running):
    from PySide6.QtWidgets import QDialog
    from arena_auto.gui import main_window
    monkeypatch.setattr(main_window.MainWindow, "refresh_devices", lambda self: None)
    monkeypatch.setattr(main_window.MainWindow, "_attach_log_handler", lambda self: None)
    manager = ConfigManager(tmp_path / "config.yaml")
    manager.config.adb.serial = "USB"
    window = main_window.MainWindow(manager)
    window.device_combo.addItem("USB")
    factory = MagicMock()
    factory.return_value.exec.return_value = QDialog.Rejected
    monkeypatch.setattr(main_window, "WirelessConnectionDialog", factory)
    monkeypatch.setattr(main_window.QMessageBox, "warning", MagicMock())
    monkeypatch.setattr(window, "_automation_running", lambda: running)
    try:
        window.open_wireless_connection()
        assert window.device_combo.currentText() == "USB"
        assert manager.config.adb.serial == "USB"
        assert not manager.path.exists()
        assert factory.call_count == (0 if running else 1)
    finally:
        window.close()
