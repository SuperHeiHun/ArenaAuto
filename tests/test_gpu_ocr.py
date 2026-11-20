"""GPU backend contract/fallback tests without requiring installed GPU packages."""

from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
from enum import Enum
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from arena_auto.automation.controller import AutomationController
from arena_auto.config.manager import ConfigManager
from arena_auto.exceptions import ConfigError, OCRProviderError
from arena_auto.models.config import AppConfig, OcrSettings
from arena_auto.models.recognition import OCRBox
from arena_auto.recognition.auto_targeting import AutoTargeting
from arena_auto.recognition.ocr import create_base_ocr_provider, create_ocr_provider
from arena_auto.recognition.ocr_engine import OcrEngine
from arena_auto.recognition.rapidocr_provider import RapidOCRProvider


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    """Stand in for the pinned RapidOCR API and ONNX Runtime sessions."""
    root = tmp_path / "rapidocr"
    (root / "models").mkdir(parents=True)
    for name in ("PP-OCRv6_det_small.onnx", "PP-OCRv6_rec_small.onnx"):
        (root / "models" / name).write_bytes(b"test-model")
    state = SimpleNamespace(root=root, sessions=[], engines=[], fail_at=None, cpu_only_at=None,
                            available=["DmlExecutionProvider", "CPUExecutionProvider"])

    class Session:
        def __init__(self, path=None, sess_options=None, providers=None):
            if path is not None:
                state.sessions.append(self)
                if state.fail_at == len(state.sessions):
                    raise RuntimeError("GPU initialization failed")
            self.options = sess_options
            self.requested = providers
            self.providers = (["CPUExecutionProvider"] if path is None or state.cpu_only_at == len(state.sessions)
                              else ["DmlExecutionProvider", "CPUExecutionProvider"])
            self.fallback_disabled = False

        def get_providers(self):
            return self.providers

        def disable_fallback(self):
            self.fallback_disabled = True

    class Engine:
        def __init__(self, params):
            state.engines.append(self)
            self.params = params
            class Version(Enum):
                V4 = "PP-OCRv4"
                V5 = "PP-OCRv5"
                V6 = "PP-OCRv6"
            class Language(Enum):
                CH = "ch"
                JAPAN = "japan"
                KOREAN = "korean"
            class Model(Enum):
                MOBILE = "mobile"
                SMALL = "small"
            self.cfg = SimpleNamespace(
                Det=SimpleNamespace(ocr_version=Version.V6, lang_type=Language.CH, model_type=Model.SMALL),
                Rec=SimpleNamespace(ocr_version=Version.V6, lang_type=Language.CH, model_type=Model.SMALL),
            )
            self.det = SimpleNamespace(session=SimpleNamespace(session=Session()))
            self.rec = SimpleNamespace(session=SimpleNamespace(session=Session()))
            self.calls = []
            self.error_count = 0
            self.result = SimpleNamespace(
                boxes=np.array([[[-2, -3], [30, -3], [30, 12], [-2, 12]]]),
                txts=("67,499",), scores=(0.95,))

        def _load_det_model(self):
            return self.det

        def _load_rec_model(self):
            return self.rec

        def __call__(self, image, **kwargs):
            self.calls.append((image, kwargs))
            if self.error_count:
                self.error_count -= 1
                raise RuntimeError("GPU inference failed")
            return self.result

    rapid = ModuleType("rapidocr")
    rapid.__file__ = str(root / "__init__.py")
    rapid.RapidOCR = Engine
    utils = ModuleType("rapidocr.utils")
    typings = ModuleType("rapidocr.utils.typings")
    # Explicit test enums are attached after module creation by the provider's
    # own enum import path; every member must still derive from Enum.
    class OCRVersion(Enum):
        PPOCRV4 = "PP-OCRv4"
        PPOCRV5 = "PP-OCRv5"
        PPOCRV6 = "PP-OCRv6"
    class LangDet(Enum):
        CH = "ch"
        MULTI = "multi"
    class LangRec(Enum):
        CH = "ch"
        JAPAN = "japan"
        KOREAN = "korean"
    class ModelType(Enum):
        MOBILE = "mobile"
        SMALL = "small"
    typings.OCRVersion, typings.LangDet = OCRVersion, LangDet
    typings.LangRec, typings.ModelType = LangRec, ModelType
    utils.typings = typings
    rapid.utils = utils
    ort = ModuleType("onnxruntime")
    ort.get_available_providers = lambda: list(state.available)
    ort.SessionOptions = SimpleNamespace
    ort.ExecutionMode = SimpleNamespace(ORT_SEQUENTIAL="sequential")
    ort.InferenceSession = Session
    monkeypatch.setitem(sys.modules, "rapidocr", rapid)
    monkeypatch.setitem(sys.modules, "rapidocr.utils", utils)
    monkeypatch.setitem(sys.modules, "rapidocr.utils.typings", typings)
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setattr("arena_auto.recognition.rapidocr_provider.platform.system", lambda: "Windows")
    monkeypatch.setattr("arena_auto.recognition.rapidocr_provider.get_resource_path", lambda name: root.parent / "bundle" / "resources" / "models" / "rapidocr")
    monkeypatch.setattr("arena_auto.recognition.rapidocr_provider.get_writable_path", lambda name: root.parent / "writable" / "resources" / "models" / "rapidocr")
    return state


def settings(**kwargs):
    return OcrSettings(provider="rapidocr", device="directml", **kwargs)


def test_both_gpu_sessions_use_correct_options_and_device(runtime):
    provider = RapidOCRProvider(settings(gpu_device_id=2))
    assert provider.device_label == "DirectML GPU 2"
    assert len(runtime.sessions) == 2
    for session in runtime.sessions:
        assert session.options.enable_mem_pattern is False
        assert session.options.execution_mode == "sequential"
        assert session.requested[0] == ("DmlExecutionProvider", {"device_id": "2"})
        assert session.fallback_disabled
    assert all(v[0] == "DmlExecutionProvider" for v in provider.session_providers.values())


@pytest.mark.parametrize("fallback", [False, True])
def test_absent_directml_is_not_reported_as_gpu(runtime, fallback):
    runtime.available = ["CPUExecutionProvider"]
    if not fallback:
        with pytest.raises(OCRProviderError, match="没有可用 DirectML"):
            RapidOCRProvider(settings(gpu_fallback_to_cpu=False))
    else:
        provider = RapidOCRProvider(settings(gpu_fallback_to_cpu=True))
        assert provider.device_label == "CPU"
        assert not runtime.sessions


@pytest.mark.parametrize("fallback", [False, True])
def test_second_gpu_session_failure_does_not_publish_first(runtime, fallback):
    runtime.fail_at = 2
    if fallback:
        provider = RapidOCRProvider(settings())
        assert provider.device_label == "CPU"
        assert all(v == ["CPUExecutionProvider"] for v in provider.session_providers.values())
    else:
        with pytest.raises(OCRProviderError, match="初始化失败"):
            RapidOCRProvider(settings(gpu_fallback_to_cpu=False))


def test_session_silently_falling_back_is_detected(runtime):
    runtime.cpu_only_at = 1
    provider = RapidOCRProvider(settings())
    assert provider.device_label == "CPU"
    assert not provider._gpu_active


def test_cpu_selection_does_not_attempt_gpu(runtime):
    provider = RapidOCRProvider(OcrSettings(provider="rapidocr", device="cpu"))
    assert provider.device_label == "CPU"
    assert not runtime.sessions


@pytest.mark.parametrize("language,expected", [("japan", "japan"), ("korean", "korean")])
def test_rapidocr_language_routes_recognizer_model(runtime, monkeypatch, language, expected):
    import hashlib
    from arena_auto.recognition import rapidocr_provider
    files = (("det.onnx", "deterministic detector"), ("rec.onnx", f"{language} recognizer"))
    bundled = runtime.root.parent / "bundle" / "resources" / "models" / "rapidocr"
    bundled.mkdir(parents=True)
    specs = []
    for name, content in files:
        path = bundled / name
        path.write_text(content, encoding="utf-8")
        specs.append((name, f"https://models.example/{name}", hashlib.sha256(path.read_bytes()).hexdigest()))
    if language == "korean":
        monkeypatch.setattr(rapidocr_provider, "KOREAN_MODELS", tuple(specs))
    config = OcrSettings(provider="rapidocr", lang=language, device="directml")
    provider = RapidOCRProvider(config)
    engine = runtime.engines[-1]
    assert engine.cfg.Rec.ocr_version.value == ("PP-OCRv5" if language == "korean" else "PP-OCRv6")
    assert engine.cfg.Rec.lang_type.value == expected
    assert engine.cfg.Det.lang_type.value == ("ch" if language == "korean" else "multi")
    assert provider.device_label == "DirectML GPU 0"


def test_runtime_gpu_failure_retries_once_on_cpu(runtime):
    provider = RapidOCRProvider(settings())
    provider._engine.error_count = 1
    result = provider.recognize(np.zeros((20, 40, 3), dtype=np.uint8))
    assert result[0].text == "67,499"
    assert len(provider._engine.calls) == 2
    assert provider.device_label == "CPU"
    assert all(v == ["CPUExecutionProvider"] for v in provider.session_providers.values())


def test_runtime_strict_gpu_failure_does_not_use_cpu(runtime):
    provider = RapidOCRProvider(settings(gpu_fallback_to_cpu=False))
    provider._engine.error_count = 1
    with pytest.raises(OCRProviderError):
        provider.recognize(np.zeros((20, 40, 3), dtype=np.uint8))
    assert len(provider._engine.calls) == 1


def test_output_coordinates_clamped_and_grayscale_supported(runtime):
    provider = RapidOCRProvider(settings())
    boxes = provider.recognize(np.zeros((20, 40), dtype=np.uint8))
    assert boxes == [OCRBox("67,499", 0.95, (0, 0, 30, 12))]
    assert provider._engine.calls[-1][0].shape == (20, 40, 3)
    assert provider.recognize(None) == []
    provider._engine.result = SimpleNamespace(boxes=None, txts=None, scores=None)
    assert provider.recognize(np.zeros((20, 40, 3), dtype=np.uint8)) == []


def test_single_line_roi_skips_detector_and_restores_full_image_detection(runtime):
    provider = RapidOCRProvider(settings())
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    line = provider.recognize_line(image)
    assert line[0].bbox == (0, 0, 40, 20)
    assert provider._engine.calls[-1][1] == {"use_det": False, "use_cls": False}
    provider.recognize(image)
    assert provider._engine.calls[-1][1] == {"use_det": True, "use_cls": False}


def test_rapidocr_roi_does_not_strip_non_numeric_text_into_fake_power(runtime):
    provider = RapidOCRProvider(settings())
    provider._engine.result.txts = ("金币67499",)
    engine = OcrEngine(provider=provider, settings=settings(), use_shared=False)
    targeting = AutoTargeting(engine, AppConfig())
    assert targeting._read_power(np.zeros((50, 100, 3), dtype=np.uint8), (0, 0, 80, 20)) is None


def test_actual_device_label_updates_in_facade_after_fallback(runtime):
    provider = RapidOCRProvider(settings())
    engine = OcrEngine(provider=provider, settings=settings(), use_shared=False)
    assert engine.get_stats()["device"] == "DirectML GPU 0"
    provider._engine.error_count = 1
    engine.recognize(np.zeros((20, 40, 3), dtype=np.uint8))
    assert engine.get_stats()["device"] == "CPU"


def test_easyocr_creates_and_reuses_separate_language_readers(monkeypatch, tmp_path):
    import types
    from arena_auto.recognition.ocr import EasyOCRProvider
    fake = types.ModuleType("easyocr")
    calls = []
    fake.Reader = lambda langs, **kwargs: calls.append((tuple(langs), kwargs)) or object()
    monkeypatch.setitem(sys.modules, "easyocr", fake)
    monkeypatch.setattr(EasyOCRProvider, "_ensure_easyocr_language_model", classmethod(lambda cls, language: tmp_path))
    EasyOCRProvider.reset_shared_reader()
    korean = EasyOCRProvider(OcrSettings(lang="korean", device="cpu"))
    korean_again = EasyOCRProvider(OcrSettings(lang="korean", device="cpu"))
    japanese = EasyOCRProvider(OcrSettings(lang="japan", device="cpu"))
    assert korean._reader is korean_again._reader
    assert japanese._reader is not korean._reader
    assert [c[0] for c in calls] == [("ko", "en"), ("ja", "en")]
    assert all(c[1]["model_storage_directory"] == str(tmp_path) for c in calls)
    EasyOCRProvider.reset_shared_reader()


def test_gpu_policy_and_device_id_participate_in_shared_identity():
    base = settings()
    assert OcrEngine.shared_provider_key(base) != OcrEngine.shared_provider_key(replace(base, gpu_device_id=1))
    assert OcrEngine.shared_provider_key(base) != OcrEngine.shared_provider_key(replace(base, gpu_fallback_to_cpu=False))


def test_optional_backend_initializes_lazily(runtime):
    engine = create_ocr_provider(settings(), allow_fallback=False, allow_null=False)
    assert not runtime.engines
    assert not engine.ready
    engine.ensure_ready()
    assert engine.ready
    assert engine.provider.name == "rapidocr"


def test_explicit_rapidocr_cpu_fallback_uses_cpu_settings(monkeypatch):
    from arena_auto.recognition import ocr
    failed = MagicMock(side_effect=OCRProviderError("not installed"))
    cpu = MagicMock()
    monkeypatch.setitem(ocr._PROVIDER_FACTORIES, "rapidocr", failed)
    monkeypatch.setitem(ocr._PROVIDER_FACTORIES, "easyocr", cpu)
    assert create_base_ocr_provider(settings(), allow_null=False) is cpu.return_value
    assert cpu.call_args.args[0].device == "cpu"
    cpu.reset_mock()
    with pytest.raises(OCRProviderError):
        create_base_ocr_provider(settings(gpu_fallback_to_cpu=False), allow_null=False)
    cpu.assert_not_called()


def test_provider_change_rebuilds_lazy_facade_without_model_load(tmp_path, monkeypatch):
    manager = ConfigManager(tmp_path / "config.yaml")
    ctrl = AutomationController(manager, adb=MagicMock(), ocr=MagicMock(), templates=MagicMock())
    old = ctrl.ocr
    manager.config.ocr.provider = "rapidocr"
    manager.config.ocr.device = "directml"
    ctrl.reload_config()
    assert ctrl.ocr is not old
    assert isinstance(ctrl.ocr, OcrEngine) and not ctrl.ocr.ready
    assert ctrl.detector.ocr is ctrl.ocr
    assert ctrl.buttons.ocr is ctrl.ocr
    assert ctrl.auto_targeting.ocr is ctrl.ocr
    same = ctrl.ocr
    ctrl.reload_config()
    assert ctrl.ocr is same


def test_gpu_config_roundtrip_and_validation(tmp_path):
    manager = ConfigManager(tmp_path / "config.yaml")
    manager.config.ocr = settings(gpu_device_id=1, gpu_fallback_to_cpu=False)
    manager.save()
    assert manager.load().ocr == manager.config.ocr
    assert manager.config.ocr.gpu_device_id == 1
    assert manager.config.ocr.gpu_fallback_to_cpu is False
    with pytest.raises(ConfigError):
        ConfigManager.from_dict({"ocr": {"gpu_device_id": -1}})


def count_scene(slash=True):
    image = np.zeros((100, 400, 3), dtype=np.uint8)
    cv2.putText(image, "4", (208, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, "5", (267, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.line(image, (249 if slash else 244, 44), (241 if slash else 244, 64), (255, 255, 255), 3)
    return image, [OCRBox("可挑战次数", 0.95, (80, 35, 120, 32)), OCRBox("415", 0.9, (200, 35, 90, 35))]


@pytest.mark.parametrize("slash", [False, True])
def test_challenge_separator_requires_slanted_visual_evidence(slash):
    image, boxes = count_scene(slash)
    provider = MagicMock()
    provider.prefers_single_line = True
    provider.recognize.return_value = boxes
    provider.recognize_line.side_effect = [[OCRBox("4", 0.95, (0, 0, 20, 20))], [OCRBox("5", 0.95, (0, 0, 20, 20))]]
    result = AutoTargeting(provider, AppConfig()).read_challenge(image)
    if slash:
        assert str(result) == "4/5"
    else:
        assert result is None
        provider.recognize_line.assert_not_called()


@pytest.fixture
def qt_app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_gui_gpu_options_and_invalid_pair(qt_app, monkeypatch, tmp_path):
    from arena_auto.gui.main_window import MainWindow, QMessageBox
    monkeypatch.setattr(MainWindow, "refresh_devices", lambda self: None)
    monkeypatch.setattr(MainWindow, "_attach_log_handler", lambda self: None)
    warning = MagicMock()
    monkeypatch.setattr(QMessageBox, "warning", warning)
    manager = ConfigManager(tmp_path / "config.yaml")
    ui = MainWindow(manager)
    try:
        ui.ocr_device_combo.setCurrentText("directml")
        assert not ui._apply_form_to_config()
        assert manager.config.ocr.device == "auto"
        ui.provider_combo.setCurrentText("rapidocr")
        ui.gpu_device_spin.setValue(1)
        ui.gpu_fallback_check.setChecked(False)
        assert ui._apply_form_to_config()
        assert manager.config.ocr == replace(manager.config.ocr, provider="rapidocr", device="directml",
                                             gpu_device_id=1, gpu_fallback_to_cpu=False)
        manager.save()
        ui._load_config_into_form()
        assert ui.ocr_device_combo.currentText() == "directml"
    finally:
        ui.close()
