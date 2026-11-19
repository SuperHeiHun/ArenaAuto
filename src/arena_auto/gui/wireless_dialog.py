"""Main-window wireless debugging connection and Android 11+ pairing."""

from __future__ import annotations

import copy

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QVBoxLayout,
)

from arena_auto.adb.controller import AdbController
from arena_auto.adb.wireless import is_wireless_serial, parse_endpoint, parse_pair_code
from arena_auto.exceptions import AdbError
from arena_auto.models.config import AdbConfig


class WirelessConnectionWorker(QThread):
    result_ready = Signal(str, str)
    failed = Signal(str)

    def __init__(self, config: AdbConfig, task: str, endpoint: str,
                 pair_endpoint: str = "", code: str = "", parent=None) -> None:
        super().__init__(parent)
        # Failed attempts must not replace the active controller's target.
        self.adb = AdbController(copy.deepcopy(config))
        self.task = task
        self.endpoint = endpoint
        self.pair_endpoint = pair_endpoint
        self.code = code

    def run(self) -> None:
        try:
            if self.task == "pair" or self.code:
                self.adb.pair_wireless(self.pair_endpoint, self.code)
            if self.task == "pair":
                self.result_ready.emit("pair", self.pair_endpoint)
            else:
                serial = self.adb.connect_wireless(self.endpoint, verify=True)
                self.result_ready.emit("connect", serial)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
        finally:
            self.code = ""


class WirelessConnectionDialog(QDialog):
    def __init__(self, config: AdbConfig, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("无线调试设备连接")
        self.resize(540, 340)
        self.config = config
        self.connected_serial = ""
        self._worker = None
        layout = QVBoxLayout(self)
        hint = QLabel(
            "手机与电脑需在同一局域网。Android 11+：开发者选项 → 无线调试。\n"
            "首次连接请打开「使用配对码配对设备」，填写配对地址和 6 位配对码。"
            "连接地址使用无线调试主页的 IP:端口，通常与配对端口不同。\n"
            "已配对设备只需连接地址；传统 ADB TCP 模式可使用 IP:5555。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        form = QFormLayout()
        self.endpoint_edit = QLineEdit()
        self.endpoint_edit.setPlaceholderText("例如 192.168.1.23:39867")
        if is_wireless_serial(config.serial):
            self.endpoint_edit.setText(config.serial)
        form.addRow("连接地址 IP:端口", self.endpoint_edit)
        self.pair_endpoint_edit = QLineEdit()
        self.pair_endpoint_edit.setPlaceholderText("例如 192.168.1.23:37123（首次配对填写）")
        form.addRow("配对地址 IP:端口", self.pair_endpoint_edit)
        self.code_edit = QLineEdit()
        self.code_edit.setEchoMode(QLineEdit.Password)
        self.code_edit.setPlaceholderText("6 位配对码；已配对可留空")
        form.addRow("配对码", self.code_edit)
        layout.addLayout(form)
        self.status_label = QLabel("连接成功后会自动选择设备并保存到配置。配对码不会保存。")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        row = QHBoxLayout()
        self.pair_btn = QPushButton("仅配对")
        self.connect_btn = QPushButton("连接设备")
        self.close_btn = QPushButton("关闭")
        self.pair_btn.clicked.connect(lambda: self.start_task("pair"))
        self.connect_btn.clicked.connect(lambda: self.start_task("connect"))
        self.close_btn.clicked.connect(self.reject)
        for button in (self.pair_btn, self.connect_btn, self.close_btn):
            row.addWidget(button)
        layout.addLayout(row)

    def _busy(self) -> bool:
        return self._worker is not None

    def start_task(self, task: str) -> None:
        if self._busy():
            return
        try:
            endpoint = parse_endpoint(self.endpoint_edit.text()).serial() if task == "connect" else ""
            code = self.code_edit.text().strip()
            pair_endpoint = ""
            if task == "pair" or code:
                code = parse_pair_code(code)
                raw_pair = self.pair_endpoint_edit.text().strip()
                if ":" not in raw_pair:
                    raise AdbError("请填写手机配对窗口显示的完整配对地址 IP:端口")
                pair_endpoint = parse_endpoint(raw_pair).serial()
        except AdbError as exc:
            QMessageBox.warning(self, "无线调试连接", str(exc))
            return
        self._worker = WirelessConnectionWorker(self.config, task, endpoint, pair_endpoint, code, self)
        self._worker.result_ready.connect(self._on_result)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_finished)
        self._set_enabled(False)
        self.status_label.setText("配对中..." if task == "pair" else "连接中（如填写配对码，将先配对）...")
        self._worker.start()

    def _set_enabled(self, enabled: bool) -> None:
        for widget in (self.endpoint_edit, self.pair_endpoint_edit, self.code_edit,
                       self.pair_btn, self.connect_btn, self.close_btn):
            widget.setEnabled(enabled)

    def _on_result(self, task: str, serial: str) -> None:
        self.code_edit.clear()
        if task == "pair":
            self.status_label.setText(f"配对成功: {serial}。请填写无线调试主页的连接地址，再点连接设备。")
        else:
            self.connected_serial = serial
            self.status_label.setText(f"已连接: {serial}")

    def _on_failed(self, message: str) -> None:
        self.status_label.setText(f"操作失败: {message}\n请检查地址、端口、配对码，以及手机与电脑网络。")

    def _on_finished(self) -> None:
        self._worker.deleteLater()
        self._worker = None
        self._set_enabled(True)
        if self.connected_serial:
            self.accept()

    def reject(self) -> None:
        if not self._busy():
            super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._busy():
            event.ignore()
        else:
            super().closeEvent(event)
