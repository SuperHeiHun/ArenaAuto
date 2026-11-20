"""Application and packaged executable icon wiring."""

import struct
import sys
from pathlib import Path

import pytest

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


def test_icon_small_sizes_use_native_windows_bitmaps():
    icon = get_resource_path("resources/app_icon.ico").read_bytes()
    reserved, kind, count = struct.unpack_from("<3H", icon)
    assert (reserved, kind, count) == (0, 1, 7)
    sizes = []
    for index in range(count):
        width, height, _, _, planes, depth, length, offset = struct.unpack_from(
            "<4B2H2I", icon, 6 + index * 16
        )
        size = width or 256
        assert (height or 256, planes, depth) == (size, 1, 32)
        frame = icon[offset:offset + length]
        assert len(frame) == length
        if size == 256:
            assert frame.startswith(b"\x89PNG\r\n\x1a\n")
        else:
            assert struct.unpack_from("<IiiHH", frame) == (40, size, size * 2, 1, 32)
            mask_stride = ((size + 31) // 32) * 4
            assert length == 40 + size * size * 4 + mask_stride * size
        sizes.append(size)
    assert sizes == [16, 24, 32, 48, 64, 128, 256]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows icon loading")
def test_windows_can_load_icon_at_shell_sizes():
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.LoadImageW.argtypes = (
        wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT, ctypes.c_int, ctypes.c_int, wintypes.UINT
    )
    user32.LoadImageW.restype = wintypes.HANDLE
    user32.DestroyIcon.argtypes = (wintypes.HICON,)
    icon_path = str(get_resource_path("resources/app_icon.ico"))
    image_icon, load_from_file = 1, 0x10
    for size in (16, 24, 32, 48, 64, 128, 256):
        handle = user32.LoadImageW(None, icon_path, image_icon, size, size, load_from_file)
        assert handle, f"Cannot load icon at {size}px: {ctypes.get_last_error()}"
        user32.DestroyIcon(handle)
