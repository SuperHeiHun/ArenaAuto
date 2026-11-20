"""Application and packaged executable icon wiring."""

from pathlib import Path

from arena_auto.paths import get_resource_path


def test_application_window_uses_bundled_icon(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from arena_auto.config.manager import ConfigManager
    from arena_auto.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(MainWindow, "refresh_devices", lambda self: None)
    monkeypatch.setattr(MainWindow, "_attach_log_handler", lambda self: None)
    window = MainWindow(ConfigManager.load_or_create(tmp_path / "config.yaml"))
    try:
        assert not window.windowIcon().isNull()
    finally:
        window.close()
        app.processEvents()


def test_icon_assets_are_present_and_spec_uses_windows_icon():
    png_path = get_resource_path("resources/app_icon.png")
    ico_path = get_resource_path("resources/app_icon.ico")
    assert png_path.is_file() and png_path.stat().st_size > 0
    assert ico_path.is_file() and ico_path.stat().st_size > 0
    assert ico_path.read_bytes().startswith(b"\x00\x00\x01\x00")

    spec_path = Path(__file__).resolve().parents[1] / "ArenaAuto.spec"
    spec = spec_path.read_text(encoding="utf-8")
    assert 'icon=str(Path(project_root) / "resources" / "app_icon.ico")' in spec
