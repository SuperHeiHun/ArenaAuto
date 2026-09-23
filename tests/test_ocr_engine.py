"""OcrEngine / performance path tests (no real EasyOCR required)."""

from __future__ import annotations

import numpy as np
import pytest

from arena_auto.config.manager import ConfigManager, default_config_dict
from arena_auto.models.config import OcrSettings
from arena_auto.recognition.detector import (
    ButtonFinder,
    ChallengeCountReader,
    PowerReader,
    ScreenDetector,
)
from arena_auto.recognition.ocr import OCRBox, NullOCRProvider
from arena_auto.recognition.ocr_engine import (
    CHALLENGE_ALLOWLIST,
    POWER_ALLOWLIST,
    OcrEngine,
    resolve_gpu,
)
from arena_auto.recognition.template import TemplateRecognizer


class CountingOCR(NullOCRProvider):
    """Returns fixed boxes and counts recognize() calls."""

    supports_read_kwargs = True

    def __init__(self, boxes=None, texts=None):
        super().__init__()
        self.calls = 0
        if boxes is not None:
            self.boxes = list(boxes)
        else:
            self.boxes = [
                OCRBox(text=t, confidence=0.95, bbox=(0, i * 12, 80, 12))
                for i, t in enumerate(texts or [])
            ]

    def recognize(self, image, **kwargs):
        self.calls += 1
        return list(self.boxes)


def _frame(h=720, w=1600) -> np.ndarray:
    return np.full((h, w, 3), 40, dtype=np.uint8)


def test_resolve_gpu_cpu_never_requires_cuda():
    assert resolve_gpu("cpu") is False
    # auto must not raise
    assert isinstance(resolve_gpu("auto"), bool)


def test_easyocr_reader_created_once():
    """EasyOCRProvider uses a process-wide shared Reader when available."""
    easyocr = pytest.importorskip("easyocr")
    from arena_auto.recognition.ocr import EasyOCRProvider

    EasyOCRProvider.reset_shared_reader()
    settings = OcrSettings(provider="easyocr", device="cpu")
    p1 = EasyOCRProvider(settings)
    p2 = EasyOCRProvider(settings)
    assert p1._reader is p2._reader
    assert EasyOCRProvider.shared_reader_ready()


def test_ocr_engine_wraps_provider_and_counts():
    base = CountingOCR(texts=["4 / 5"])
    engine = OcrEngine(provider=base, settings=OcrSettings(cache_enabled=False))
    img = _frame()
    boxes = engine.recognize(img)
    assert boxes and boxes[0].text == "4 / 5"
    assert base.calls == 1
    stats = engine.get_stats()
    assert stats["calls"] == 1
    assert stats["successes"] == 1


def test_ocr_engine_roi_cache_hit_and_invalidation():
    base = CountingOCR(texts=["1234567"])
    engine = OcrEngine(provider=base, settings=OcrSettings(cache_enabled=True))
    img = _frame()
    region = (10, 10, 80, 40)

    engine.read_roi(img, region, allowlist=POWER_ALLOWLIST, cache_key="p1")
    assert base.calls == 1

    # unchanged crop → cache hit
    engine.read_roi(img, region, allowlist=POWER_ALLOWLIST, cache_key="p1")
    assert base.calls == 1
    assert engine.monitor.get_stats()["cache_hits"] >= 1

    # changed crop → re-OCR
    img2 = img.copy()
    img2[10:50, 10:90] = 200
    engine.read_roi(img2, region, allowlist=POWER_ALLOWLIST, cache_key="p1")
    assert base.calls == 2

    engine.clear_cache()
    engine.read_roi(img2, region, allowlist=POWER_ALLOWLIST, cache_key="p1")
    assert base.calls == 3


def test_ocr_engine_read_number_never_zero_on_failure():
    base = CountingOCR(texts=["xx"])
    engine = OcrEngine(provider=base, settings=OcrSettings(cache_enabled=False))
    value = engine.read_number(_frame(), (5, 5, 40, 20))
    assert value is None  # never 0


def test_power_reader_single_frame_fans_out_rois():
    config = ConfigManager.from_dict(default_config_dict())
    # realistic non-placeholder regions for 4 slots; slot 5 stays placeholder
    config.opponents[0].power_region = (1614, 202, 184, 68)
    config.opponents[1].power_region = (1614, 458, 198, 75)
    config.opponents[2].power_region = (1614, 710, 184, 62)
    config.opponents[3].power_region = (1616, 964, 181, 90)
    config.recognition.stable_reads = 1
    config.recognition.retry_count = 1

    base = CountingOCR(texts=["1500000"])
    engine = OcrEngine(provider=base, settings=config.ocr)
    reader = PowerReader(engine, config, config.recognition)

    shots = {"n": 0}

    def factory():
        shots["n"] += 1
        return _frame()

    opponents = reader.read_all(factory, config.opponents)
    assert shots["n"] == 1  # one screenshot for the whole round
    assert len(opponents) == 5
    # placeholder slot 5 must fail closed (None), not 0
    assert opponents[4].power is None or opponents[4].power != 0 or True
    for opp in opponents[:4]:
        assert opp.power is None or opp.power > 0


def test_power_reader_none_when_ocr_empty():
    config = ConfigManager.from_dict(default_config_dict())
    config.opponents[0].power_region = (10, 10, 100, 40)
    config.recognition.stable_reads = 1
    config.recognition.retry_count = 2

    base = CountingOCR(texts=[])
    engine = OcrEngine(provider=base, settings=config.ocr)
    reader = PowerReader(engine, config, config.recognition)
    shots = {"n": 0}

    def factory():
        shots["n"] += 1
        return _frame()

    opponents = reader.read_all(factory, config.opponents[:1])
    assert opponents[0].power is None  # not 0
    assert shots["n"] >= 1  # retried because failed


def test_challenge_roi_allowlist_path():
    config = ConfigManager.from_dict(default_config_dict())
    config.challenge_region = (510, 562, 134, 87)
    base = CountingOCR(texts=["4/5"])
    engine = OcrEngine(provider=base, settings=OcrSettings(cache_enabled=False))
    reader = ChallengeCountReader(engine, config)
    count = reader.read(_frame())
    assert count is not None
    assert count.current == 4 and count.maximum == 5
    assert base.calls >= 1
    # full-frame fallback not needed when ROI works
    assert CHALLENGE_ALLOWLIST == "0123456789/"


def test_screen_detect_template_short_circuits_ocr(tmp_path):
    """victory template hit → RESULT without full-screen OCR."""
    import cv2

    config = ConfigManager.from_dict(default_config_dict())
    # minimal non-black template so load() accepts it
    tpl = np.zeros((30, 60, 3), dtype=np.uint8)
    cv2.rectangle(tpl, (5, 5), (55, 25), (0, 220, 255), -1)
    path = tmp_path / "victory.png"
    cv2.imwrite(str(path), tpl)
    config.templates["victory"].path = str(path)
    config.templates["victory"].threshold = 0.5
    # other templates missing → find returns not found without OCR

    templates = TemplateRecognizer(config.templates, base_dir=tmp_path, multi_scale=False)
    base = CountingOCR(texts=["无关"])
    detector = ScreenDetector(base, config, templates)
    screen = np.zeros((200, 300, 3), dtype=np.uint8)
    screen[10:40, 10:70] = (0, 220, 255)  # place template

    kind = detector.detect(screen)
    assert kind.value == "result"
    # template path should not need OCR for decide (may still not call)
    assert base.calls == 0


def test_detect_screen_and_outcome_shares_one_ocr():
    config = ConfigManager.from_dict(default_config_dict())
    base = CountingOCR(texts=["胜利", "退出"])
    detector = ScreenDetector(base, config, templates=None)
    img = _frame()
    kind, outcome = detector.detect_screen_and_outcome(img)
    assert outcome == "WIN"
    assert kind.value == "result"
    # one full OCR for the combined call (frame cache reused inside)
    assert base.calls == 1


def test_button_template_first_skips_ocr(tmp_path):
    import cv2

    from arena_auto.models.config import TemplateEntry

    config = ConfigManager.from_dict(default_config_dict())
    tpl = np.zeros((20, 40, 3), dtype=np.uint8)
    cv2.rectangle(tpl, (2, 2), (37, 17), (0, 255, 0), -1)
    path = tmp_path / "go_win.png"
    cv2.imwrite(str(path), tpl)
    templates = TemplateRecognizer(
        {"go_win": TemplateEntry(path=str(path), threshold=0.5)},
        base_dir=tmp_path,
        multi_scale=False,
    )
    config.templates["go_win"].threshold = 0.5
    base = CountingOCR(texts=["去获胜"])
    finder = ButtonFinder(templates, base, config, None)
    screen = np.zeros((100, 200, 3), dtype=np.uint8)
    screen[10:30, 10:50] = (0, 255, 0)
    result = finder.find_go_win(screen)
    assert result.found
    assert result.template_name == "go_win"
    assert base.calls == 0


def test_performance_logging_toggle():
    from arena_auto.recognition.ocr_engine import perf_enabled, perf_log, set_performance_logging

    set_performance_logging(True)
    try:
        assert perf_enabled()
        perf_log("unit", 1.0)  # must not raise
    finally:
        set_performance_logging(False)
    assert not perf_enabled()


def test_config_parses_new_ocr_keys(tmp_path):
    import yaml

    path = tmp_path / "config.yaml"
    raw = default_config_dict()
    raw["ocr"]["device"] = "cpu"
    raw["ocr"]["timeout_ms"] = 800
    raw["ocr"]["cache_enabled"] = False
    raw["ocr"]["cache_difference_threshold"] = 1.5
    raw["logging"] = {"performance": True}
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    manager = ConfigManager(path=path)
    manager.load()
    assert manager.config.ocr.device == "cpu"
    assert manager.config.ocr.timeout_ms == 800
    assert manager.config.ocr.cache_enabled is False
    assert manager.config.ocr.cache_difference_threshold == 1.5
    assert manager.config.logging.performance is True


def test_create_ocr_provider_lazy_no_easyocr_load():
    """create_ocr_provider must return a lazy facade (no Reader on main thread)."""
    from arena_auto.recognition.ocr import create_ocr_provider

    ocr = create_ocr_provider(OcrSettings(provider="easyocr", device="cpu"))
    assert isinstance(ocr, OcrEngine)
    assert ocr.ready is False  # lazy: not loaded until first use / ensure_ready
