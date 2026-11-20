"""No real ADB/device operations: inspect launches and probe local processes."""

import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from arena_auto import process_utils
from arena_auto.adb.controller import AdbController
from arena_auto.exceptions import AdbError, OCRProviderError
from arena_auto.process_utils import no_console_window_options
from arena_auto.recognition import ocr


def test_non_windows_process_options_are_empty(monkeypatch):
    monkeypatch.setattr(process_utils, "sys", SimpleNamespace(platform="linux"))
    assert no_console_window_options() == {}


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process flags")
def test_windows_process_options_do_not_allocate_console():
    assert no_console_window_options() == {"creationflags": subprocess.CREATE_NO_WINDOW}


@pytest.mark.parametrize("binary", [False, True])
def test_adb_runner_uses_no_console_flags(monkeypatch, binary):
    flags = {"creationflags": 0x08000000}
    monkeypatch.setattr("arena_auto.adb.controller.no_console_window_options", lambda: flags)
    run = MagicMock(return_value=subprocess.CompletedProcess(["fake-adb"], 0, b"result", b""))
    monkeypatch.setattr(subprocess, "run", run)
    adb = AdbController()
    assert adb._run(["fake-adb", "devices"], timeout=2, binary=binary) == (b"result" if binary else "result")
    run.assert_called_once_with(
        ["fake-adb", "devices"], capture_output=True, timeout=2, check=False, **flags
    )


def test_hidden_adb_keeps_error_behavior(monkeypatch):
    run = MagicMock(return_value=subprocess.CompletedProcess(["fake-adb"], 1, b"", b"command failed"))
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(AdbError, match="command failed"):
        AdbController()._run(["fake-adb", "version"])
    assert AdbController()._run(["fake-adb", "version"], check=False) == ""


@pytest.fixture
def fake_tesseract(monkeypatch):
    launches = SimpleNamespace(
        check_output=MagicMock(return_value=b"5.0.0"),
        run=MagicMock(),
        Popen=MagicMock(),
        PIPE=subprocess.PIPE,
        TimeoutExpired=subprocess.TimeoutExpired,
    )
    implementation = SimpleNamespace(subprocess=launches)

    def recognize(image, **kwargs):
        implementation.subprocess.Popen(["fake-tesseract", "image", "stdout"], stdin=subprocess.PIPE)
        return {"text": ["12345"], "conf": ["90"], "left": [0], "top": [0], "width": [50], "height": [20]}

    module = SimpleNamespace(
        pytesseract=implementation,
        Output=SimpleNamespace(DICT="dict"),
        get_tesseract_version=lambda: implementation.subprocess.check_output(["fake-tesseract", "--version"]),
        image_to_data=recognize,
    )
    monkeypatch.setitem(sys.modules, "pytesseract", module)
    monkeypatch.setattr(ocr, "no_console_window_options", lambda: {"creationflags": 0x08000000})
    return implementation, launches


def test_tesseract_version_and_recognition_are_hidden(fake_tesseract):
    implementation, launches = fake_tesseract
    original_popen = subprocess.Popen
    provider = ocr.TesseractOCRProvider()
    launches.check_output.assert_called_once_with(["fake-tesseract", "--version"], creationflags=0x08000000)
    boxes = provider.recognize(np.zeros((30, 60, 3), dtype=np.uint8))
    assert boxes[0].text == "12345"
    launches.Popen.assert_called_once_with(
        ["fake-tesseract", "image", "stdout"], stdin=subprocess.PIPE, creationflags=0x08000000
    )
    assert implementation.subprocess.TimeoutExpired is subprocess.TimeoutExpired
    assert subprocess.Popen is original_popen


def test_tesseract_preserves_other_flags_and_wraps_only_once(fake_tesseract):
    implementation, launches = fake_tesseract
    ocr.TesseractOCRProvider()
    proxy = implementation.subprocess
    ocr.TesseractOCRProvider()
    assert implementation.subprocess is proxy
    startupinfo = object()
    proxy.run(["fake-tesseract", "--list-langs"], creationflags=0x00000200, startupinfo=startupinfo)
    launches.run.assert_called_once_with(
        ["fake-tesseract", "--list-langs"], creationflags=0x08000200, startupinfo=startupinfo
    )


def test_tesseract_is_not_wrapped_on_other_platforms(fake_tesseract, monkeypatch):
    implementation, launches = fake_tesseract
    monkeypatch.setattr(ocr, "no_console_window_options", lambda: {})
    ocr.TesseractOCRProvider()
    assert implementation.subprocess is launches
    launches.check_output.assert_called_once_with(["fake-tesseract", "--version"])


def test_tesseract_probe_failure_still_raises_provider_error(fake_tesseract):
    _, launches = fake_tesseract
    launches.check_output.side_effect = FileNotFoundError("missing engine")
    with pytest.raises(OCRProviderError, match="Tesseract"):
        ocr.TesseractOCRProvider()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows console handle probe")
def test_real_local_child_has_no_console_window():
    # A local console-mode Python is sufficient to test the same creation flag
    # used for adb.exe/Tesseract; this neither discovers nor contacts devices.
    code = (
        "import ctypes,json; k=ctypes.WinDLL('kernel32'); "
        "k.GetConsoleWindow.restype=ctypes.c_void_p; "
        "print(json.dumps({'console_window': bool(k.GetConsoleWindow())}))"
    )
    stdout = AdbController()._run([sys.executable, "-c", code], timeout=15)
    assert json.loads(stdout) == {"console_window": False}
