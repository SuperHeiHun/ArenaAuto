"""Language selection, model-cache integrity and offline compatibility tests."""

import hashlib
from io import BytesIO
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock
from zipfile import ZipFile
import sys

import numpy as np
import pytest

from arena_auto.config.manager import ConfigManager
from arena_auto.exceptions import OCRProviderError
from arena_auto.models.config import OcrSettings, normalize_ocr_language
from arena_auto.recognition import model_store
from arena_auto.recognition.ocr import EasyOCRProvider
from arena_auto.recognition.ocr_engine import OcrEngine


@pytest.mark.parametrize("alias,expected", [
    ("ch", "ch"), ("zh", "ch"), ("zh-cn", "ch"), ("ch_sim", "ch"),
    ("ja", "japan"), ("jp", "japan"), ("japanese", "japan"),
    ("ko", "korean"), ("kr", "korean"), (" KOREAN ", "korean"),
])
def test_config_language_aliases_and_roundtrip(alias, expected, tmp_path):
    assert normalize_ocr_language(alias) == expected
    config = ConfigManager.from_dict({"ocr": {"lang": alias}})
    assert config.ocr.lang == expected
    manager = ConfigManager(tmp_path / "config.yaml", config)
    manager.save()
    assert manager.load().ocr.lang == expected


def test_language_participates_in_facade_identity():
    settings = OcrSettings(lang="ch")
    original = OcrEngine.shared_provider_key(settings)
    settings.lang = "korean"
    assert original != OcrEngine.shared_provider_key(settings)
    settings.lang = "ko"
    assert OcrEngine.shared_provider_key(settings) == OcrEngine.shared_provider_key(OcrSettings(lang="korean"))


def test_download_is_verified_and_cached_without_network_on_second_call(monkeypatch, tmp_path):
    payload = b"verified ONNX content"
    checksum = hashlib.sha256(payload).hexdigest()
    network = MagicMock(side_effect=lambda *args, **kwargs: BytesIO(payload))
    monkeypatch.setattr(model_store, "urlopen", network)
    path = tmp_path / "models" / "ko.onnx"
    assert model_store.download_model(path, "https://official.example/ko.onnx", checksum) == path
    assert path.read_bytes() == payload
    assert model_store.download_model(path, "https://official.example/ko.onnx", checksum) == path
    network.assert_called_once()
    assert list(path.parent.glob("*.part")) == []


def test_failed_checksum_does_not_overwrite_existing_model(monkeypatch, tmp_path):
    path = tmp_path / "existing.onnx"
    path.write_bytes(b"old cached model")
    monkeypatch.setattr(model_store, "urlopen", lambda *args, **kwargs: BytesIO(b"truncated bad download"))
    with pytest.raises(OCRProviderError, match="校验失败"):
        model_store.download_model(path, "https://official.example/ko.onnx", hashlib.sha256(b"expected").hexdigest())
    assert path.read_bytes() == b"old cached model"
    assert len(list(tmp_path.iterdir())) == 1


def test_only_expected_zip_member_is_written_and_weights_md5_is_checked(monkeypatch, tmp_path):
    payload = b"official Japanese weights"
    archive_bytes = BytesIO()
    with ZipFile(archive_bytes, "w") as archive:
        archive.writestr("japanese_g2.pth", payload)
        archive.writestr("../not-extracted.txt", b"must never be extracted")
    network = MagicMock(return_value=BytesIO(archive_bytes.getvalue()))
    monkeypatch.setattr(model_store, "urlopen", network)
    path = tmp_path / "models" / "japanese_g2.pth"
    model_store.download_model(path, "https://official.example/japanese.zip", hashlib.md5(payload).hexdigest(),
                               algorithm="md5", archive_member="japanese_g2.pth")
    assert path.read_bytes() == payload
    assert not (tmp_path / "not-extracted.txt").exists()
    assert list(path.parent.iterdir()) == [path]


def test_download_failure_cleans_partial_files(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise TimeoutError("network timeout")
    monkeypatch.setattr(model_store, "urlopen", fail)
    with pytest.raises(OCRProviderError, match="下载失败"):
        model_store.download_model(tmp_path / "ko.onnx", "https://official.example/ko.onnx", "f" * 64)
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def easy_runtime(monkeypatch, tmp_path):
    module = ModuleType("easyocr")
    created = []
    def reader(languages, **options):
        obj = SimpleNamespace(device="cpu", languages=tuple(languages), options=options,
                              readtext=MagicMock(return_value=[([[0, 0], [30, 0], [30, 10], [0, 10]], "나가기", 0.9)]))
        created.append(obj)
        return obj
    module.Reader = reader
    monkeypatch.setitem(sys.modules, "easyocr", module)
    for name, value in (("_shared_reader", None), ("_shared_key", None),
                        ("_shared_readers", {}), ("_language_download_status", {})):
        monkeypatch.setattr(EasyOCRProvider, name, value)
    prepare = MagicMock(side_effect=lambda language: None if language == "ch" else tmp_path)
    monkeypatch.setattr(EasyOCRProvider, "_ensure_easyocr_language_model", prepare)
    return created, prepare, tmp_path


def test_easyocr_language_switch_keeps_chinese_japanese_korean_separate(easy_runtime):
    created, prepare, root = easy_runtime
    chinese = EasyOCRProvider(OcrSettings(lang="ch", device="cpu"))
    korean = EasyOCRProvider(OcrSettings(lang="ko", device="cpu"))
    japanese = EasyOCRProvider(OcrSettings(lang="ja", device="cpu"))
    restored = EasyOCRProvider(OcrSettings(lang="korean", device="cpu"))
    assert [obj.languages for obj in created] == [("ch_sim", "en"), ("ko", "en"), ("ja", "en")]
    assert restored._reader is korean._reader
    assert chinese._reader is not japanese._reader
    assert korean._reader.options["model_storage_directory"] == str(root)
    assert "model_storage_directory" not in chinese._reader.options
    assert prepare.call_count == 3
    result = korean.recognize(np.zeros((20, 40, 3), dtype=np.uint8))
    assert result[0].text == "나가기" and result[0].bbox == (0, 0, 30, 10)


def test_easyocr_actual_device_is_not_mislabelled_as_cuda(easy_runtime):
    provider = EasyOCRProvider(OcrSettings(lang="korean", device="cuda"))
    assert provider.device_label == "CPU"  # Test Reader explicitly reports CPU fallback.


def test_easyocr_gpu_fallback_can_be_disabled(easy_runtime):
    with pytest.raises(OCRProviderError, match="CPU 回退已关闭"):
        EasyOCRProvider(OcrSettings(lang="korean", device="cuda", gpu_fallback_to_cpu=False))


@pytest.mark.parametrize("language", ["japan", "korean"])
def test_easyocr_rejects_directml_instead_of_fake_gpu_support(easy_runtime, language):
    with pytest.raises(OCRProviderError, match="DirectML"):
        EasyOCRProvider(OcrSettings(lang=language, device="directml"))
    assert easy_runtime[0] == []


def test_unknown_language_fails_before_model_download(easy_runtime):
    with pytest.raises(OCRProviderError, match="语言配置"):
        EasyOCRProvider(OcrSettings(lang="nonexistent", device="cpu"))
    easy_runtime[1].assert_not_called()


def test_gui_language_selection_saves_and_restores_config(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from arena_auto.gui.main_window import MainWindow
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(MainWindow, "refresh_devices", lambda self: None)
    monkeypatch.setattr(MainWindow, "_attach_log_handler", lambda self: None)
    manager = ConfigManager(tmp_path / "config.yaml")
    ui = MainWindow(manager)
    try:
        ui.provider_combo.setCurrentText("easyocr")
        ui.ocr_device_combo.setCurrentText("cpu")
        ui.ocr_lang_combo.setCurrentIndex(ui.ocr_lang_combo.findData("korean"))
        ui.save_config()
        assert manager.load().ocr.lang == "korean"
        ui.config = manager.config
        ui._load_config_into_form()
        assert ui.ocr_lang_combo.currentData() == "korean"
        ui.ocr_lang_combo.setCurrentIndex(ui.ocr_lang_combo.findData("japan"))
        ui.save_config()
        assert manager.load().ocr.lang == "japan"
    finally:
        ui.close()
        app.processEvents()
