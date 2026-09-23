"""PySide6 main window for ArenaAuto."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QThread, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from arena_auto.adb.controller import AdbController
from arena_auto.automation.controller import AutomationController
from arena_auto.calibration.wizard import CalibrationWizard, CalibrationWorker
from arena_auto.config.manager import ConfigManager
from arena_auto.debug.recorder import DebugRecorder
from arena_auto.gui.widgets import ImageView, TemplateCaptureDialog
from arena_auto.gui.workers import (
    AutomationWorker,
    DebugTestWorker,
    DeviceListWorker,
    ScreenshotWorker,
)
from arena_auto.logging.logger import CallbackHandler
from arena_auto.models.config import AppConfig
from arena_auto.models.recognition import ChallengeCount, Opponent
from arena_auto.models.state import AutomationState, ScreenKind, SessionStatistics
from arena_auto.paths import get_template_dir
from arena_auto.recognition.ocr import create_ocr_provider
from arena_auto.recognition.template import TemplateRecognizer

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window - all heavy work runs on worker threads."""

    def __init__(self, config_manager: ConfigManager, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ArenaAuto — 胜利者的竞技场自动化")
        self.resize(1180, 820)

        self.config_manager = config_manager
        self.config: AppConfig = config_manager.config
        self.adb = AdbController(self.config.adb)
        self.templates = TemplateRecognizer(self.config.templates)
        self.debug_recorder = DebugRecorder(self.config.debug)

        self._automation_thread: Optional[QThread] = None
        self._automation_worker: Optional[AutomationWorker] = None
        self._device_worker: Optional[DeviceListWorker] = None
        self._shot_worker: Optional[ScreenshotWorker] = None
        self._test_worker: Optional[DebugTestWorker] = None
        self._calibration_worker: Optional[CalibrationWorker] = None
        self._controller: Optional[AutomationController] = None
        self._preview_timer: Optional[QTimer] = None

        self._build_ui()
        self._attach_log_handler()
        self._load_config_into_form()
        self.refresh_devices()

        missing = self.config_manager.missing_templates()
        if missing:
            self._append_log(
                "WARNING",
                f"缺少模板文件: {', '.join(missing)}。请进入调试工具创建模板",
            )
            self.statusBar().showMessage("缺少模板文件，请打开调试工具创建", 8000)

    # --------------------------------------------------------------------- ui
    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, stretch=1)

        # ----- left column: controls + status
        left = QWidget()
        left_layout = QVBoxLayout(left)

        # device group
        device_box = QGroupBox("ADB设备")
        device_form = QHBoxLayout(device_box)
        self.device_combo = QComboBox()
        self.device_combo.setEditable(False)
        device_form.addWidget(self.device_combo, stretch=1)
        self.refresh_btn = QPushButton("刷新设备")
        self.refresh_btn.clicked.connect(self.refresh_devices)
        device_form.addWidget(self.refresh_btn)
        left_layout.addWidget(device_box)

        # parameters
        param_box = QGroupBox("参数")
        param_form = QFormLayout(param_box)
        self.threshold_edit = QLineEdit()
        self.threshold_edit.setPlaceholderText("例如 2000000")
        param_form.addRow("战力阈值", self.threshold_edit)

        self.poll_spin = QSpinBox()
        self.poll_spin.setRange(1, 120)
        self.poll_spin.setSuffix(" 秒")
        param_form.addRow("检测间隔", self.poll_spin)

        self.dry_run_check = QCheckBox("测试模式（不真实点击）")
        param_form.addRow(self.dry_run_check)

        self.purchase_check = QCheckBox("自动购买挑战次数")
        self.auto_confirm_check = QCheckBox("自动确认购买")
        self.debug_check = QCheckBox("调试截图")
        param_form.addRow(self.purchase_check)
        param_form.addRow(self.auto_confirm_check)
        param_form.addRow(self.debug_check)

        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["easyocr", "paddle", "tesseract"])
        param_form.addRow("OCR 引擎", self.provider_combo)

        self.ocr_scale_spin = QSpinBox()
        self.ocr_scale_spin.setRange(1, 6)
        param_form.addRow("OCR 放大", self.ocr_scale_spin)

        left_layout.addWidget(param_box)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("开始运行")
        self.start_btn.clicked.connect(self.start_automation)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_automation)
        self.save_cfg_btn = QPushButton("保存配置")
        self.save_cfg_btn.clicked.connect(self.save_config)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(self.save_cfg_btn)
        left_layout.addLayout(btn_row)

        cal_row = QHBoxLayout()
        self.auto_cfg_btn = QPushButton("自动配置")
        self.auto_cfg_btn.clicked.connect(lambda: self.open_calibration(False))
        self.recalib_btn = QPushButton("重新标定")
        self.recalib_btn.clicked.connect(lambda: self.open_calibration(True))
        self.test_cfg_btn = QPushButton("测试配置")
        self.test_cfg_btn.clicked.connect(self.test_current_config)
        self.manual_cfg_btn = QPushButton("手动配置")
        self.manual_cfg_btn.clicked.connect(self.open_manual_config)
        cal_row.addWidget(self.auto_cfg_btn)
        cal_row.addWidget(self.recalib_btn)
        cal_row.addWidget(self.test_cfg_btn)
        cal_row.addWidget(self.manual_cfg_btn)
        left_layout.addLayout(cal_row)

        # status group
        status_box = QGroupBox("当前状态")
        status_layout = QVBoxLayout(status_box)
        self.state_label = QLabel("IDLE")
        self.state_label.setStyleSheet("font-size:18px;font-weight:bold;color:#2a6;")
        self.screen_label = QLabel("页面: -")
        self.target_label = QLabel("当前目标: -")
        self.challenge_label = QLabel("挑战次数: -")
        self.ocr_status_label = QLabel("OCR: -")
        status_layout.addWidget(self.state_label)
        status_layout.addWidget(self.screen_label)
        status_layout.addWidget(self.target_label)
        status_layout.addWidget(self.challenge_label)
        status_layout.addWidget(self.ocr_status_label)
        left_layout.addWidget(status_box)

        # powers
        power_box = QGroupBox("对手战力")
        power_layout = QVBoxLayout(power_box)
        self.power_labels: List[QLabel] = []
        for i in range(1, 6):
            lab = QLabel(f"#{i}  -")
            self.power_labels.append(lab)
            power_layout.addWidget(lab)
        left_layout.addWidget(power_box)

        # stats
        stats_box = QGroupBox("运行统计")
        stats_layout = QVBoxLayout(stats_box)
        self.stats_label = QLabel(
            "胜利：0\n失败：0\n购买：0\n识别失败：0\n恢复：0\n挑战：0"
        )
        stats_layout.addWidget(self.stats_label)
        left_layout.addWidget(stats_box)

        left_layout.addStretch(1)
        splitter.addWidget(left)

        # ----- right column: tabs
        right = QTabWidget()

        # runtime tab
        runtime = QWidget()
        rt_layout = QVBoxLayout(runtime)
        tool_row = QHBoxLayout()
        self.shot_btn = QPushButton("刷新截图")
        self.shot_btn.clicked.connect(self.refresh_screenshot)
        self.preview_check = QCheckBox("1 FPS 预览")
        self.preview_check.toggled.connect(self.toggle_preview)
        self.open_debug_btn = QPushButton("打开调试目录")
        self.open_debug_btn.clicked.connect(self.open_debug_dir)
        tool_row.addWidget(self.shot_btn)
        tool_row.addWidget(self.preview_check)
        tool_row.addWidget(self.open_debug_btn)
        tool_row.addStretch(1)
        rt_layout.addLayout(tool_row)
        self.runtime_image = ImageView()
        rt_layout.addWidget(self.runtime_image, stretch=1)
        right.addTab(runtime, "实时截图")

        # debug tools
        debug = QWidget()
        dbg_layout = QVBoxLayout(debug)
        dbg_hint = QLabel(
            "开发/调试：识别测试会在截图上绘制 ROI / 模板框 / confidence。"
            "坐标与 power_region 请在 config.yaml 中按实际设备修改。"
        )
        dbg_hint.setWordWrap(True)
        dbg_layout.addWidget(dbg_hint)

        test_row1 = QHBoxLayout()
        for label, name in [
            ("截图", "screenshot"),
            ("测试OCR", "ocr"),
            ("测试竞技场检测", "arena"),
            ("测试战力ROI", "powers"),
            ("测试挑战次数", "challenge"),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _=False, n=name: self.run_debug_test(n))
            test_row1.addWidget(btn)
        dbg_layout.addLayout(test_row1)

        test_row2 = QHBoxLayout()
        for label, name in [
            ("测试去获胜", "go_win"),
            ("测试退出", "exit"),
            ("测试购买按钮", "buy"),
            ("测试确认", "confirm"),
            ("测试胜利", "victory"),
            ("测试失败", "defeat"),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _=False, n=name: self.run_debug_test(n))
            test_row2.addWidget(btn)
        dbg_layout.addLayout(test_row2)

        test_row3 = QHBoxLayout()
        self.capture_tpl_btn = QPushButton("从当前截图创建模板")
        self.capture_tpl_btn.clicked.connect(self.capture_template)
        test_row3.addWidget(self.capture_tpl_btn)
        test_row3.addStretch(1)
        dbg_layout.addLayout(test_row3)

        self.debug_image = ImageView()
        dbg_layout.addWidget(self.debug_image, stretch=1)
        self.debug_payload_label = QLabel("")
        self.debug_payload_label.setWordWrap(True)
        dbg_layout.addWidget(self.debug_payload_label)
        right.addTab(debug, "开发/调试")

        # log tab
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        log_layout.addWidget(self.log_view)
        log_actions = QHBoxLayout()
        clear_log_btn = QPushButton("清空日志")
        clear_log_btn.clicked.connect(self.log_view.clear)
        log_actions.addWidget(clear_log_btn)
        log_actions.addStretch(1)
        log_layout.addLayout(log_actions)
        right.addTab(log_tab, "日志")

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("就绪")

    # ---------------------------------------------------------------- logging
    def _attach_log_handler(self) -> None:
        root = logging.getLogger("arena_auto")

        def _emit(level: str, message: str) -> None:
            # marshal to GUI thread
            QTimer.singleShot(0, lambda: self._append_log(level, message))

        handler = CallbackHandler(_emit, level=logging.DEBUG)
        handler.setLevel(logging.DEBUG)
        root.addHandler(handler)
        self._log_handler = handler

    def _append_log(self, level: str, message: str) -> None:
        # HH:MM:SS prefix on every GUI log line (CallbackHandler no longer adds asctime)
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{ts}] {message}")
        self.log_view.verticalScrollBar().setValue(
            self.log_view.verticalScrollBar().maximum()
        )
        if "OCR Ready" in message or "OCR 初始化中" in message:
            self.ocr_status_label.setText(
                message if len(message) < 60 else "OCR: Ready"
            )
            if "OCR Ready" in message:
                self.statusBar().showMessage("OCR Ready", 4000)

    # ----------------------------------------------------------------- config
    def _load_config_into_form(self) -> None:
        cfg = self.config
        self.threshold_edit.setText(str(cfg.automation.power_threshold))
        self.poll_spin.setValue(int(cfg.battle.poll_interval))
        self.dry_run_check.setChecked(cfg.automation.dry_run)
        self.purchase_check.setChecked(cfg.purchase.enabled)
        self.auto_confirm_check.setChecked(cfg.purchase.auto_confirm)
        self.debug_check.setChecked(cfg.debug.enabled)
        idx = self.provider_combo.findText(cfg.ocr.provider)
        if idx >= 0:
            self.provider_combo.setCurrentIndex(idx)
        self.ocr_scale_spin.setValue(cfg.ocr.scale)

    def _apply_form_to_config(self) -> bool:
        cfg = self.config
        text = self.threshold_edit.text().strip().replace(",", "")
        try:
            threshold = int(text)
            if threshold < 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "配置错误", "power_threshold 必须是数字")
            return False
        cfg.automation.power_threshold = threshold
        cfg.battle.poll_interval = float(self.poll_spin.value())
        cfg.automation.poll_interval = float(self.poll_spin.value())
        cfg.automation.dry_run = self.dry_run_check.isChecked()
        cfg.purchase.enabled = self.purchase_check.isChecked()
        cfg.purchase.auto_confirm = self.auto_confirm_check.isChecked()
        cfg.debug.enabled = self.debug_check.isChecked()
        cfg.ocr.provider = self.provider_combo.currentText()
        cfg.ocr.scale = int(self.ocr_scale_spin.value())

        serial = self.device_combo.currentText().strip()
        cfg.adb.serial = serial
        self.adb.update_config(cfg.adb)
        return True

    def save_config(self) -> None:
        if not self._apply_form_to_config():
            return
        try:
            self.config_manager.save(self.config)
            self.debug_recorder = DebugRecorder(self.config.debug)
            if self._controller is not None:
                self._controller.reload_config()
            self.statusBar().showMessage(f"配置已保存: {self.config_manager.path}", 4000)
            self._append_log("INFO", f"配置已保存: {self.config_manager.path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "配置", f"保存失败:\n{exc}")

    # ------------------------------------------------------------- calibration
    def _automation_running(self) -> bool:
        return self._automation_thread is not None and self._automation_thread.isRunning()

    def open_calibration(self, recalibrate: bool = False) -> None:
        if self._automation_running():
            QMessageBox.warning(self, "自动配置", "自动化运行中，请先停止")
            return
        if not self._apply_form_to_config():
            return
        ocr = self._controller.ocr if self._controller is not None else None
        wizard = CalibrationWizard(
            config_manager=self.config_manager,
            adb=self.adb,
            ocr_provider=ocr,
            templates=self.templates,
            debug=self.debug_recorder,
            recalibrate=recalibrate,
            parent=self,
        )
        wizard.exec()
        if wizard.saved_path is not None:
            self.config = self.config_manager.config
            self._load_config_into_form()
            self.templates.set_entries(self.config.templates)
            self.debug_recorder = DebugRecorder(self.config.debug)
            if self._controller is not None:
                self._controller.reload_config()
            self._append_log("INFO", f"自动配置已保存: {wizard.saved_path}")
            self.statusBar().showMessage(f"自动配置已保存: {wizard.saved_path}", 6000)

    def test_current_config(self) -> None:
        if self._automation_running():
            QMessageBox.warning(self, "测试配置", "自动化运行中，请先停止")
            return
        if not self._apply_form_to_config():
            return
        controller = self._ensure_controller()
        if controller is None:
            return
        ocr = controller.ocr
        self.test_cfg_btn.setEnabled(False)
        self.statusBar().showMessage("配置测试中（不会点击）...")
        from arena_auto.calibration.wizard import _OcrHolder

        worker = CalibrationWorker(
            task="verify",
            adb=self.adb,
            config_manager=self.config_manager,
            ocr_holder=_OcrHolder(self.config, ocr),
            templates=self.templates,
            bundle=None,
        )
        self._calibration_worker = worker
        worker.result_ready.connect(self._on_config_test_result)
        worker.failed.connect(self._on_config_test_failed)
        worker.finished.connect(self._on_config_test_finished)
        worker.start()

    def _on_config_test_result(self, task: str, payload: dict) -> None:
        if task != "verify":
            return
        summary = payload.get("summary") or {}
        lines = ["配置测试结果", ""]
        for check in summary.get("checks") or []:
            mark = "✓" if check.get("ok") else "✗"
            lines.append(f"{mark} {check.get('name')}: {check.get('detail')}")
        lines.append("")
        lines.append(
            f"通过 {summary.get('passed', 0)}/{summary.get('total', 0)}"
        )
        if not summary.get("all_ok"):
            lines.append("⚠ 配置可能存在问题")
        text = "\n".join(lines)
        self._append_log("INFO", text.replace("\n", " | "))
        QMessageBox.information(self, "测试当前配置", text)

    def _on_config_test_failed(self, task: str, message: str) -> None:
        self.statusBar().showMessage(f"测试失败: {message}", 6000)
        QMessageBox.warning(self, "测试当前配置", f"{task} 失败:\n{message}")

    def _on_config_test_finished(self) -> None:
        self.test_cfg_btn.setEnabled(True)
        self.statusBar().showMessage("就绪", 2000)
        if self._calibration_worker is not None:
            self._calibration_worker.deleteLater()
            self._calibration_worker = None

    def open_manual_config(self) -> None:
        path = self.config_manager.path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.is_file():
                self.config_manager.save(self.config)
            if sys.platform == "win32":
                os.startfile(str(path))  # noqa: S606
            else:
                subprocess.Popen(["xdg-open", str(path)])
            self._append_log("INFO", f"打开手动配置: {path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.information(
                self,
                "手动配置",
                f"无法打开编辑器: {exc}\n\n请手动编辑:\n{path}",
            )

    # ----------------------------------------------------------------- devices
    def refresh_devices(self) -> None:
        self.device_combo.clear()
        self._device_worker = DeviceListWorker(self.adb, self)
        self._device_worker.devices_found.connect(self._on_devices)
        self._device_worker.error_occurred.connect(
            lambda msg: self.statusBar().showMessage(f"ADB 错误: {msg}", 6000)
        )
        self._device_worker.start()

    def _on_devices(self, devices: List[str]) -> None:
        self.device_combo.clear()
        if not devices:
            self.device_combo.addItem("未检测到 Android 设备")
            self.statusBar().showMessage("未检测到 Android 设备", 6000)
            self._append_log("WARNING", "未检测到 Android 设备")
            return
        for serial in devices:
            self.device_combo.addItem(serial)
        if len(devices) == 1:
            self.statusBar().showMessage(f"自动选择设备: {devices[0]}", 4000)
        # keep configured serial if present
        configured = self.config.adb.serial
        if configured:
            idx = self.device_combo.findText(configured)
            if idx >= 0:
                self.device_combo.setCurrentIndex(idx)

    # ------------------------------------------------------------- automation
    def _ensure_controller(self) -> Optional[AutomationController]:
        if not self._apply_form_to_config():
            return None
        try:
            if self._controller is None:
                # Lazy OcrEngine: Reader loads on the automation worker thread.
                ocr = create_ocr_provider(self.config.ocr)
                self.templates.set_entries(self.config.templates)
                self.debug_recorder.settings = self.config.debug
                self._controller = AutomationController(
                    config_manager=self.config_manager,
                    adb=self.adb,
                    ocr=ocr,
                    templates=self.templates,
                    debug=self.debug_recorder,
                )
            else:
                self._controller.reload_config()
            return self._controller
        except Exception as exc:  # noqa: BLE001
            logger.exception("初始化控制器失败")
            QMessageBox.critical(self, "错误", f"初始化失败:\n{exc}")
            return None

    def start_automation(self) -> None:
        if self._automation_thread is not None and self._automation_thread.isRunning():
            return
        controller = self._ensure_controller()
        if controller is None:
            return

        missing = self.config_manager.missing_templates()
        if missing:
            answer = QMessageBox.question(
                self,
                "缺少模板",
                f"缺少模板文件: {', '.join(missing)}\n\n"
                "部分按钮可能只能依赖 OCR fallback。仍要启动吗？",
            )
            if answer != QMessageBox.Yes:
                return

        controller.reset_stop()
        self._automation_worker = AutomationWorker(controller)
        self._automation_thread = QThread(self)
        self._automation_worker.moveToThread(self._automation_thread)
        self._automation_thread.started.connect(self._automation_worker.run)

        self._automation_worker.state_changed.connect(self._on_state)
        self._automation_worker.screen_detected.connect(self._on_screen)
        self._automation_worker.log_message.connect(self._append_log)
        self._automation_worker.powers_detected.connect(self._on_powers)
        self._automation_worker.target_changed.connect(self._on_target)
        self._automation_worker.challenge_changed.connect(self._on_challenge)
        self._automation_worker.statistics_changed.connect(self._on_stats)
        self._automation_worker.screenshot_updated.connect(self._on_live_screenshot)
        self._automation_worker.error_occurred.connect(self._on_error)
        self._automation_worker.finished.connect(self._on_worker_finished)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.save_cfg_btn.setEnabled(False)
        # OCR loads on the automation worker thread — keep the GUI free.
        self.statusBar().showMessage("OCR 初始化中...")
        self.ocr_status_label.setText("OCR: 初始化中...")
        self._append_log("INFO", "开始运行（OCR 于后台线程初始化）")
        self._automation_thread.start()

    def stop_automation(self) -> None:
        if self._automation_worker is not None:
            self._automation_worker.request_stop()
        self.stop_btn.setEnabled(False)
        self.statusBar().showMessage("正在停止...")
        self._append_log("INFO", "请求停止")

    def _on_worker_finished(self) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_cfg_btn.setEnabled(True)
        self.statusBar().showMessage("已停止")
        if self._automation_thread is not None:
            self._automation_thread.quit()
            self._automation_thread.wait(3000)
            self._automation_thread = None
        self._automation_worker = None

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._automation_thread is not None and self._automation_thread.isRunning():
            answer = QMessageBox.question(
                self,
                "退出",
                "自动化仍在运行，确定停止并退出吗？",
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.stop_automation()
            self._automation_thread.wait(5000)
        try:
            if self._controller is not None:
                self._controller._save_statistics()
        except Exception:  # noqa: BLE001
            pass
        event.accept()

    # ---------------------------------------------------------------- signals
    def _on_state(self, state: AutomationState) -> None:
        self.state_label.setText(state.name)

    def _on_screen(self, kind: ScreenKind) -> None:
        self.screen_label.setText(f"页面: {kind.value.upper()}")

    def _on_powers(self, powers: List[Opponent]) -> None:
        for i, lab in enumerate(self.power_labels):
            if i < len(powers):
                p = powers[i].power
                text = f"#{powers[i].id}  {p:,}" if p is not None else f"#{powers[i].id}  OCR_FAILED"
                lab.setText(text)
                if p is None:
                    lab.setStyleSheet("color:#c33;")
                else:
                    lab.setStyleSheet("")
            else:
                lab.setText(f"#{i + 1}  -")

    def _on_target(self, target_id: int) -> None:
        self.target_label.setText(f"当前目标: #{target_id}")

    def _on_challenge(self, count: ChallengeCount) -> None:
        self.challenge_label.setText(f"挑战次数: {count}")

    def _on_stats(self, stats: SessionStatistics) -> None:
        self.stats_label.setText(
            f"胜利：{stats.victories}\n"
            f"失败：{stats.defeats}\n"
            f"购买：{stats.purchases}\n"
            f"识别失败：{stats.recognition_failures}\n"
            f"恢复：{stats.recoveries}\n"
            f"挑战：{stats.total_battles}"
        )

    def _on_live_screenshot(self, image) -> None:
        self.runtime_image.set_image(image)

    def _on_error(self, message: str) -> None:
        self.statusBar().showMessage(f"错误: {message}", 8000)
        self._append_log("ERROR", message)

    # -------------------------------------------------------------- screenshot
    def refresh_screenshot(self) -> None:
        if not self._apply_form_to_config():
            return
        self._shot_worker = ScreenshotWorker(self.adb, self)
        self._shot_worker.captured.connect(
            lambda img: (self.runtime_image.set_image(img), self.statusBar().showMessage("截图完成", 2000))
        )
        self._shot_worker.failed.connect(
            lambda msg: self.statusBar().showMessage(f"截图失败: {msg}", 6000)
        )
        self._shot_worker.start()

    def toggle_preview(self, checked: bool) -> None:
        if checked:
            self._preview_timer = QTimer(self)
            self._preview_timer.setInterval(1000)
            self._preview_timer.timeout.connect(self.refresh_screenshot)
            self._preview_timer.start()
        elif self._preview_timer is not None:
            self._preview_timer.stop()
            self._preview_timer = None

    def open_debug_dir(self) -> None:
        path = Path(self.debug_recorder.open_directory())
        path.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(str(path))  # noqa: S606
        else:
            subprocess.Popen(["xdg-open", str(path)])

    # ------------------------------------------------------------ debug tools
    def run_debug_test(self, test_name: str) -> None:
        if not self._apply_form_to_config():
            return
        controller = self._ensure_controller()
        if controller is None:
            return
        self.debug_payload_label.setText(f"运行 {test_name} ...")
        self._test_worker = DebugTestWorker(self.adb, controller, test_name, self)
        self._test_worker.result_ready.connect(self._on_test_result)
        self._test_worker.start()

    def _on_test_result(self, name: str, image, payload: dict) -> None:
        if image is not None:
            self.debug_image.set_image(image)
        if payload.get("error"):
            self.debug_payload_label.setText(f"{name}: 错误 - {payload['error']}")
            return
        if name == "ocr":
            texts = payload.get("texts") or []
            self.debug_payload_label.setText("OCR: " + (" | ".join(texts) if texts else "(空)"))
        elif name == "arena":
            self.debug_payload_label.setText(f"页面识别: {payload.get('screen')}")
        elif name == "powers":
            lines = []
            for item in payload.get("powers") or []:
                p = item.get("power")
                raw = item.get("raw") or ""
                region = item.get("region")
                if p is not None:
                    lines.append(f"#{item['id']}={p}")
                else:
                    lines.append(
                        f"#{item['id']}=FAIL raw={raw!r} roi={region}"
                    )
            self.debug_payload_label.setText("战力: " + "; ".join(lines))
        elif name == "challenge":
            c = payload.get("count")
            raw = payload.get("raw") or ""
            self.debug_payload_label.setText(
                f"挑战次数: {c[0]}/{c[1]}" if c else f"挑战次数: OCR_FAILED raw={raw!r}"
            )
        else:
            found = payload.get("found")
            conf = payload.get("confidence")
            conf_text = f"{conf:.3f}" if isinstance(conf, (int, float)) else "n/a"
            if found:
                self.debug_payload_label.setText(
                    f"{name}: 找到 conf={conf_text} bbox={payload.get('bbox')}"
                )
            else:
                self.debug_payload_label.setText(f"{name}: 未找到 (conf={conf_text})")

    def capture_template(self) -> None:
        if not self._apply_form_to_config():
            return
        try:
            image = self.adb.screenshot()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "模板", f"截图失败:\n{exc}")
            return
        suggested = "go_win"
        dlg = TemplateCaptureDialog(image, get_template_dir(), suggested, self)
        dlg.exec()
        if dlg.saved_path is not None:
            name = dlg.saved_path.stem
            # auto-register into config if missing
            from arena_auto.models.config import TemplateEntry

            if name not in self.config.templates:
                rel = f"resources/templates/{name}.png"
                self.config.templates[name] = TemplateEntry(path=rel, threshold=0.82)
                try:
                    self.config_manager.save(self.config)
                    self.templates.set_entries(self.config.templates)
                    self._append_log("INFO", f"模板已加入配置: {name}")
                except Exception as exc:  # noqa: BLE001
                    self._append_log("WARNING", f"模板已保存但写入配置失败: {exc}")
