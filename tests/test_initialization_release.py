"""Formal automatic initialization: first-run defaults, wording and release CI."""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from arena_auto.config.manager import ConfigManager
from arena_auto.models.config import AppConfig

ROOT = Path(__file__).resolve().parents[1]


def test_no_config_file_enables_initialization_and_persists_choice(tmp_path):
    manager = ConfigManager.load_or_create(tmp_path / "config.yaml")
    assert manager.config.automation.auto_targeting
    assert yaml.safe_load(manager.path.read_text(encoding="utf-8"))["automation"]["auto_targeting"] is True
    manager.config.automation.auto_targeting = False
    manager.save()
    assert not ConfigManager.load_or_create(manager.path).config.automation.auto_targeting


def test_no_file_load_path_also_enables_initialization(tmp_path):
    assert ConfigManager(tmp_path / "new.yaml").load().automation.auto_targeting


def test_old_yaml_without_geometry_enables_initialization():
    assert ConfigManager.from_dict({}).automation.auto_targeting
    assert ConfigManager.from_dict({"automation": {"power_threshold": 100000}}).automation.auto_targeting


def test_old_calibrated_yaml_keeps_manual_mode():
    config = ConfigManager.from_dict({"opponents": [
        {"id": 1, "power_region": [1300, 150, 200, 40], "click": [1200, 170]},
    ]})
    assert not config.automation.auto_targeting


@pytest.mark.parametrize("enabled", [False, True])
def test_explicit_mode_is_never_overwritten(enabled):
    config = ConfigManager.from_dict({"automation": {"auto_targeting": enabled}})
    assert config.automation.auto_targeting is enabled


def test_first_run_prefers_bundled_backend_without_importing_model(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name in ("rapidocr", "onnxruntime") else None)
    cfg = ConfigManager.initial_config()
    assert cfg.ocr.provider == "rapidocr" and cfg.ocr.device == "auto"
    assert cfg.adb.serial == ""


def test_source_install_without_optional_backend_keeps_existing_default(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    cfg = ConfigManager.initial_config()
    assert cfg.ocr.provider == AppConfig().ocr.provider
    assert cfg.automation.auto_targeting


@pytest.fixture
def ui(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from arena_auto.gui.main_window import MainWindow
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(MainWindow, "refresh_devices", lambda self: None)
    monkeypatch.setattr(MainWindow, "_attach_log_handler", lambda self: None)
    window = MainWindow(ConfigManager.load_or_create(tmp_path / "config.yaml"))
    yield window
    window.close()
    app.processEvents()


def test_formal_labels_and_new_user_default(ui):
    assert ui.auto_target_check.text() == "自动初始化"
    assert ui.auto_target_check.isChecked()
    assert ui.auto_cfg_btn.text() == "半自动配置"
    assert "试验" not in ui.auto_target_check.text()
    assert "无需先标定" in ui.log_view.toPlainText()


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("purchase", [False, True])
def test_start_message_explains_initialization_not_template_error(ui, monkeypatch, dry_run, purchase):
    from arena_auto.gui.main_window import QMessageBox
    ui.config.automation.dry_run = dry_run
    ui.config.purchase.enabled = purchase
    prompt = MagicMock(return_value=QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, "question", prompt)
    assert ui._confirm_automation_start()
    assert prompt.call_args.args[1] == "开始自动初始化"
    text = prompt.call_args.args[2]
    assert "无需先手动标定" in text and "未找到模板时会使用 OCR" in text
    assert "安全停止" in text
    assert ("只记录操作" in text) is dry_run
    assert ("自动购买：已开启" in text) is purchase
    assert prompt.call_args.args[-1] == QMessageBox.No


def test_cancel_initialization_does_not_start_worker_or_click(ui, monkeypatch):
    from arena_auto.gui.main_window import QMessageBox
    controller = MagicMock()
    ui._ensure_controller = MagicMock(return_value=controller)
    monkeypatch.setattr(QMessageBox, "question", MagicMock(return_value=QMessageBox.No))
    ui.start_automation()
    controller.reset_stop.assert_not_called()
    assert ui._automation_thread is None


def test_existing_manual_mode_retains_template_warning(ui, monkeypatch):
    from arena_auto.gui.main_window import QMessageBox
    ui.config.automation.auto_targeting = False
    prompt = MagicMock(return_value=QMessageBox.No)
    monkeypatch.setattr(QMessageBox, "question", prompt)
    assert not ui._confirm_automation_start()
    assert prompt.call_args.args[1] == "缺少模板"
    assert "半自动配置" in prompt.call_args.args[2]


def test_half_automatic_wizard_title(ui, monkeypatch):
    from arena_auto.calibration.wizard import CalibrationWizard
    monkeypatch.setattr(CalibrationWizard, "_goto", lambda *args: None)
    wizard = CalibrationWizard(ui.config_manager, MagicMock(), parent=ui)
    assert wizard.windowTitle() == "ArenaAuto - 半自动配置"
    wizard.close()


def test_release_ci_installs_gpu_runs_tests_and_checks_packaged_inference():
    workflow = yaml.load((ROOT / ".github/workflows/build-release.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert workflow["on"]["push"]["branches"] == ["main"]
    build = workflow["jobs"]["build"]
    commands = "\n".join(step.get("run", "") for step in build["steps"])
    assert "requirements-build.txt" in commands
    assert "pytest tests" in commands
    assert "--self-test" in commands
    assert "Start-Process" in commands and "model_count" in commands
    assert workflow["permissions"]["contents"] == "read"
    release = workflow["jobs"]["release"]
    assert release["needs"] == "build"
    assert release["if"] == "github.ref_type == 'tag'"
    assert release["permissions"]["contents"] == "write"


def test_spec_does_not_bundle_private_config_or_generated_templates():
    spec = (ROOT / "ArenaAuto.spec").read_text(encoding="utf-8")
    assert '("config", "config")' not in spec
    assert '"generated" not in relative.parts' in spec
    assert '"rapidocr", "onnxruntime"' in spec


@pytest.mark.parametrize("provider,device", [
    ("rapidocr", "auto"), ("rapidocr", "directml"), ("rapidocr", "cpu"),
    ("easyocr", "cuda"), ("easyocr", "cpu"), ("paddle", "cpu"), ("tesseract", "cpu"),
])
def test_config_persists_all_ocr_engine_choices(tmp_path, provider, device):
    path = tmp_path / "config.yaml"
    config = ConfigManager.from_dict({"ocr": {"provider": provider, "device": device,
                                             "gpu_device_id": 1, "gpu_fallback_to_cpu": False}})
    ConfigManager(path, config).save()
    restored = ConfigManager.load_or_create(path).config
    assert restored.ocr.provider == provider
    assert restored.ocr.device == device
    assert restored.ocr.gpu_device_id == 1
    assert restored.ocr.gpu_fallback_to_cpu is False


def test_gui_saves_engine_choice_in_config_and_restores_it(ui):
    ui.provider_combo.setCurrentText("rapidocr")
    ui.ocr_device_combo.setCurrentText("directml")
    ui.gpu_device_spin.setValue(1)
    ui.gpu_fallback_check.setChecked(False)
    ui.save_config()
    loaded = ConfigManager.load_or_create(ui.config_manager.path).config
    assert loaded.ocr.provider == "rapidocr"
    assert loaded.ocr.device == "directml"
    assert loaded.ocr.gpu_device_id == 1
    ui.config = loaded
    ui._load_config_into_form()
    assert ui.provider_combo.currentText() == "rapidocr"
    assert ui.ocr_device_combo.currentText() == "directml"


def test_portable_config_example_can_be_loaded_without_device_info():
    raw = yaml.safe_load((ROOT / "config/config.example.yaml").read_text(encoding="utf-8"))
    config = ConfigManager.from_dict(raw)
    assert config.ocr.provider == "rapidocr"
    assert config.automation.auto_targeting
    assert config.adb.serial == ""


def test_gpu_launcher_respects_config_instead_of_overriding_saved_engine(monkeypatch):
    import runpy
    from arena_auto import app

    start = MagicMock(return_value=0)
    monkeypatch.setattr(app, "main", start)
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(ROOT / "scripts/run_gpu.py"), run_name="__main__")
    assert result.value.code == 0
    start.assert_called_once_with()
