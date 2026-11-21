"""Release smoke checks import audio APIs without constructing devices or playing."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from arena_auto import __version__
from arena_auto.models.recognition import OCRBox
from arena_auto.runtime_check import main


@pytest.fixture
def provider(monkeypatch, tmp_path):
    from arena_auto.recognition import rapidocr_provider
    paths = [tmp_path / "det.onnx", tmp_path / "rec.onnx"]
    for path in paths:
        path.write_bytes(b"mock model")
    result = SimpleNamespace(_model_paths=paths, name="mock", device_label="CPU",
                             session_providers=[["CPUExecutionProvider"]],
                             recognize=MagicMock(return_value=[OCRBox("12345", .99, (1, 1, 40, 20))]))
    monkeypatch.setattr(rapidocr_provider, "RapidOCRProvider", lambda settings: result)
    return result


def audio_api(monkeypatch):
    output = MagicMock()
    player = SimpleNamespace(errorOccurred=object())
    monkeypatch.setitem(sys.modules, "PySide6.QtMultimedia", SimpleNamespace(QAudioOutput=output, QMediaPlayer=player))
    return output


def test_desktop_check_verifies_icon_without_audio_or_adb(provider, monkeypatch, tmp_path):
    output = audio_api(monkeypatch)
    report = tmp_path / "report.json"
    assert main(["--self-test", "--check-desktop", "--report", str(report)]) == 0
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["version"] == __version__ and data["device_actions"] == 0
    assert data["desktop_checked"] and data["audio_api_available"] and data["refresh_template_available"]
    output.assert_not_called()


def test_ocr_only_check_does_not_require_multimedia(provider, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "PySide6.QtMultimedia", None)
    report = tmp_path / "report.json"
    assert main(["--self-test", "--report", str(report)]) == 0
    assert not json.loads(report.read_text(encoding="utf-8")).get("desktop_checked", False)


def test_desktop_check_fails_for_missing_audio(provider, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "PySide6.QtMultimedia", None)
    report = tmp_path / "report.json"
    assert main(["--self-test", "--check-desktop", "--report", str(report)]) == 1
    assert not json.loads(report.read_text(encoding="utf-8"))["ok"]


@pytest.mark.parametrize("exists", [True, False])
def test_desktop_check_fails_for_invalid_or_missing_icon(provider, monkeypatch, tmp_path, exists):
    from arena_auto import paths
    audio_api(monkeypatch)
    icon = tmp_path / "bad.png"
    if exists:
        icon.write_bytes(b"not an image")
    monkeypatch.setattr(paths, "get_resource_path", lambda _: icon)
    report = tmp_path / "report.json"
    assert main(["--self-test", "--check-desktop", "--report", str(report)]) == 1
    assert not json.loads(report.read_text(encoding="utf-8"))["refresh_template_available"]
