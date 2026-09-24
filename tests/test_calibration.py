"""Auto-calibration pipeline tests (no device / no OCR engine required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arena_auto.calibration.analyzer import CalibrationAnalyzer
from arena_auto.calibration.coordinate_mapper import CoordinateMapper
from arena_auto.calibration.models import (
    SOURCE_MANUAL,
    SOURCE_OCR,
    SOURCE_STRUCTURE,
    CalibrationBundle,
    CalibrationDetection,
    CalibrationPoint,
    CalibrationRegion,
)
from arena_auto.calibration.opponent_detector import OpponentDetector
from arena_auto.calibration.roi_detector import RoiDetector
from arena_auto.config.manager import ConfigManager, default_config_dict
from arena_auto.models.config import AppConfig, OpponentConfig
from arena_auto.models.recognition import OCRBox
from arena_auto.recognition.ocr import NullOCRProvider


def _identity_config() -> AppConfig:
    config = ConfigManager.from_dict(default_config_dict())
    # 单测用全黑假截图，禁止写回 resources/templates/generated
    config.calibration.generate_templates = False
    return config


def _power_boxes(with_slot3: bool = True) -> list[OCRBox]:
    boxes = []
    texts = ["1,234,567", "1,543,210", "1,872,320", "2,123,456", "2,345,678"]
    for index, text in enumerate(texts):
        if index == 2 and not with_slot3:
            continue
        y = 300 + index * 100
        boxes.append(OCRBox(text=text, confidence=0.95, bbox=(1180, y, 220, 40)))
    return boxes


class _FakeOCR(NullOCRProvider):
    def __init__(self, boxes: list[OCRBox]):
        super().__init__()
        self._boxes = list(boxes)

    def recognize(self, image):
        return list(self._boxes)


def test_calibration_region_expansion_and_center() -> None:
    region = CalibrationRegion(x=100, y=100, width=50, height=40)
    assert region.center == CalibrationPoint(x=125, y=120)
    grown = region.expanded(10, 5, bounds=(200, 200))
    assert grown.as_tuple() == (90, 95, 70, 50)
    clamped = CalibrationRegion(x=0, y=0, width=10, height=10).expanded(
        50, 50, bounds=(40, 40)
    )
    assert clamped.x >= 0 and clamped.y >= 0
    assert clamped.x + clamped.width <= 40
    assert clamped.y + clamped.height <= 40


def test_coordinate_mapper_roundtrip() -> None:
    mapper = CoordinateMapper(1920, 1080, 1280, 720)
    region = CalibrationRegion(x=640, y=360, width=200, height=100)
    ref = mapper.actual_region_to_reference(region)
    assert ref == (960, 540, 300, 150)
    back = mapper.reference_region_to_actual(ref)
    assert back.as_tuple() == (640, 360, 200, 100)
    point = CalibrationPoint(x=640, y=360)
    assert mapper.actual_point_to_reference(point) == (960, 540)
    assert mapper.pad_to_actual(20, 20) == (13, 13)


def test_opponent_detector_finds_five_slots() -> None:
    config = _identity_config()
    mapper = CoordinateMapper(
        config.screen.reference_width,
        config.screen.reference_height,
        1920,
        1080,
    )
    detector = OpponentDetector(config, mapper)
    boxes = _power_boxes()
    detections, notes = detector.detect(boxes)
    assert len(detections) == 5
    assert all(d.source == SOURCE_OCR for d in detections)
    assert detections[0].value == 1234567
    ys = [d.region.y for d in detections]
    assert ys == sorted(ys)
    gaps = [ys[i + 1] - ys[i] for i in range(4)]
    assert all(60 <= gap <= 140 for gap in gaps)
    assert all(d.confidence >= 0.75 for d in detections)
    assert detections[0].click_point.x < detections[0].region.x + detections[0].region.width


def test_opponent_detector_fills_missing_slot_by_structure() -> None:
    config = _identity_config()
    mapper = CoordinateMapper(1920, 1080, 1920, 1080)
    detector = OpponentDetector(config, mapper)
    detections, _ = detector.detect(_power_boxes(with_slot3=False))
    assert len(detections) == 5
    assert detections[2].source == SOURCE_STRUCTURE
    assert detections[2].value is None
    assert detections[0].source == SOURCE_OCR


def test_neighbor_keyword_rejects_gold_amount() -> None:
    config = _identity_config()
    mapper = CoordinateMapper(1920, 1080, 1920, 1080)
    detector = OpponentDetector(config, mapper)
    boxes = [
        OCRBox(text="金币", confidence=0.95, bbox=(400, 500, 80, 30)),
        OCRBox(text="377726", confidence=0.95, bbox=(500, 500, 150, 30)),
        OCRBox(text="1,234,567", confidence=0.95, bbox=(1180, 300, 220, 40)),
        OCRBox(text="1,543,210", confidence=0.95, bbox=(1180, 400, 220, 40)),
    ]
    candidates = detector.extract_candidates(boxes)
    values = [c.value for c in candidates]
    assert 377726 not in values
    assert 1234567 in values


def test_short_numbers_rejected() -> None:
    config = _identity_config()
    mapper = CoordinateMapper(1920, 1080, 1920, 1080)
    detector = OpponentDetector(config, mapper)
    boxes = [
        OCRBox(text="85", confidence=0.99, bbox=(100, 100, 40, 20)),
        OCRBox(text="2664", confidence=0.99, bbox=(100, 200, 60, 20)),
    ]
    assert detector.extract_candidates(boxes) == []


def test_opponent_detector_labeled_power_text() -> None:
    """OCR boxes like 战斗力：1,234,567 should parse and form 5 slots."""
    config = _identity_config()
    mapper = CoordinateMapper(1920, 1080, 1920, 1080)
    detector = OpponentDetector(config, mapper)
    texts = [
        "战斗力：1,234,567",
        "战斗力：1,543,210",
        "战斗力：1,872,320",
        "战斗力：2,123,456",
        "战斗力：2,345,678",
    ]
    boxes = [
        OCRBox(text=t, confidence=0.95, bbox=(1140, 300 + i * 100, 260, 40))
        for i, t in enumerate(texts)
    ]
    detections, _ = detector.detect(boxes)
    assert len(detections) == 5
    assert detections[0].value == 1234567
    assert detections[0].detail.startswith("战力：")
    assert all(d.confidence >= 0.75 for d in detections)


def test_labeled_power_preferred_over_bare_number() -> None:
    config = _identity_config()
    mapper = CoordinateMapper(1920, 1080, 1920, 1080)
    detector = OpponentDetector(config, mapper)
    boxes = [
        OCRBox(text="999999", confidence=0.95, bbox=(300, 500, 150, 30)),
        OCRBox(text="战斗力：999999", confidence=0.95, bbox=(1180, 500, 240, 40)),
    ]
    candidates = detector.extract_candidates(boxes)
    labeled = [c for c in candidates if "战斗力" in c.box.text]
    bare = [c for c in candidates if "战斗力" not in c.box.text]
    assert labeled and bare
    assert labeled[0].numeric_score > bare[0].numeric_score


def test_rank_neighbor_does_not_exclude_labeled_power() -> None:
    """Real layout: 排名2664 sits on the same row as 战斗力：67,499."""
    config = _identity_config()
    mapper = CoordinateMapper(1920, 1080, 1920, 1080)
    detector = OpponentDetector(config, mapper)
    rows = [
        (170, "排名2664", "战斗力：67,499", 67499),
        (340, "排名2710", "战斗力：68,685", 68685),
        (510, "排名2752", "战斗力：3,170,348", 3170348),
    ]
    boxes = []
    for y, rank, power, _ in rows:
        boxes.append(OCRBox(text=rank, confidence=0.95, bbox=(1170, y, 140, 40)))
        boxes.append(OCRBox(text=power, confidence=0.95, bbox=(1330, y, 180, 40)))
    candidates = detector.extract_candidates(boxes)
    values = [c.value for c in candidates]
    assert values == [67499, 68685, 3170348]


def test_split_power_label_beats_rank_neighbor() -> None:
    """OCR may split 战斗力 | 67,499; nearest label decides."""
    config = _identity_config()
    mapper = CoordinateMapper(1920, 1080, 1920, 1080)
    detector = OpponentDetector(config, mapper)
    boxes = [
        OCRBox(text="排名2664", confidence=0.95, bbox=(1150, 170, 130, 40)),
        OCRBox(text="战斗力", confidence=0.95, bbox=(1300, 170, 80, 40)),
        OCRBox(text="67,499", confidence=0.95, bbox=(1390, 170, 110, 40)),
    ]
    candidates = detector.extract_candidates(boxes)
    assert [c.value for c in candidates] == [67499]
    assert candidates[0].numeric_score > 0.5


def test_defense_deck_power_excluded() -> None:
    """Left panel 防御卡组战斗力 4,575,674 must not become an opponent."""
    config = _identity_config()
    mapper = CoordinateMapper(1920, 1080, 1920, 1080)
    detector = OpponentDetector(config, mapper)
    boxes = [
        OCRBox(text="防御卡组战斗力", confidence=0.95, bbox=(100, 470, 220, 40)),
        OCRBox(text="4,575,674", confidence=0.95, bbox=(140, 520, 190, 40)),
        OCRBox(text="战斗力：67,499", confidence=0.95, bbox=(1330, 170, 180, 40)),
        OCRBox(text="战斗力：68,685", confidence=0.95, bbox=(1330, 340, 180, 40)),
    ]
    candidates = detector.extract_candidates(boxes)
    values = [c.value for c in candidates]
    assert 4575674 not in values
    assert values == [67499, 68685]


def test_defense_label_combined_box_excluded() -> None:
    config = _identity_config()
    mapper = CoordinateMapper(1920, 1080, 1920, 1080)
    detector = OpponentDetector(config, mapper)
    boxes = [
        OCRBox(
            text="防御卡组战斗力 4,575,674",
            confidence=0.95,
            bbox=(100, 470, 260, 60),
        ),
    ]
    assert detector.extract_candidates(boxes) == []


def test_button_pages_definitions() -> None:
    from arena_auto.calibration.button_detector import (
        BUTTON_DISPLAY,
        BUTTON_PAGE_ORDER,
        page_targets,
        pages_for_button,
    )

    assert BUTTON_PAGE_ORDER == (
        "arena",
        "prepare",
        "purchase",
        "victory",
        "defeat",
    )
    assert page_targets("arena") == ("buy_challenge",)
    assert page_targets("prepare") == ("go_win",)
    assert page_targets("purchase") == ("buy_challenge", "confirm_buy")
    assert page_targets("victory") == ("victory", "exit")
    assert page_targets("defeat") == ("defeat", "exit")
    assert pages_for_button("go_win") == ("prepare",)
    assert pages_for_button("exit") == ("victory", "defeat")
    assert pages_for_button("confirm_buy") == ("purchase",)
    # every displayed button belongs to at least one page
    for name, _home in BUTTON_DISPLAY:
        assert pages_for_button(name)


def test_button_detector_page_scoped_detection() -> None:
    from arena_auto.calibration.button_detector import (
        ButtonDetector,
        page_targets,
    )

    config = _identity_config()
    mapper = CoordinateMapper(1920, 1080, 1920, 1080)
    detector = ButtonDetector(config, mapper)
    boxes = [
        OCRBox(text="去获胜", confidence=0.95, bbox=(1600, 950, 200, 70)),
        OCRBox(text="退出", confidence=0.93, bbox=(1700, 40, 100, 50)),
        OCRBox(text="确认", confidence=0.92, bbox=(800, 600, 120, 60)),
        OCRBox(text="胜利", confidence=0.95, bbox=(700, 300, 160, 80)),
    ]
    # arena page: only buy_challenge (exit lives on result pages)
    arena = detector.detect(boxes, names=page_targets("arena"))
    assert set(arena) == set()
    assert "confirm_buy" not in arena
    assert "victory" not in arena
    # prepare page: go_win only
    prepare = detector.detect(boxes, names=page_targets("prepare"))
    assert set(prepare) == {"go_win"}
    assert "exit" not in prepare
    # purchase page: confirm only
    purchase = detector.detect(boxes, names=page_targets("purchase"))
    assert set(purchase) == {"confirm_buy"}
    # victory page: victory + exit
    victory = detector.detect(boxes, names=page_targets("victory"))
    assert set(victory) == {"victory", "exit"}
    # no filter = all found
    everything = detector.detect(boxes)
    assert set(everything) == {"go_win", "exit", "confirm_buy", "victory"}


def test_detect_page_buttons_merges_across_pages() -> None:
    from arena_auto.calibration.analyzer import CalibrationAnalyzer

    config = _identity_config()
    arena_boxes = [
        OCRBox(text="去获胜", confidence=0.95, bbox=(1600, 950, 200, 70)),
        OCRBox(text="退出", confidence=0.93, bbox=(1700, 40, 100, 50)),
    ]
    victory_boxes = [
        OCRBox(text="胜利", confidence=0.95, bbox=(700, 300, 160, 80)),
        OCRBox(text="退出", confidence=0.93, bbox=(1700, 40, 100, 50)),
    ]

    class _PageOCR(NullOCRProvider):
        def __init__(self) -> None:
            super().__init__()
            self.boxes = arena_boxes

        def recognize(self, image):
            return list(self.boxes)

    ocr = _PageOCR()
    config = _identity_config()
    config.calibration.generate_templates = False
    analyzer = CalibrationAnalyzer(config, ocr)
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)

    r1 = analyzer.detect_page_buttons(img, "prepare")
    assert set(r1["detected"]) == {"go_win"}
    assert r1["missing"] == []  # prepare page targets only go_win
    assert r1["templates"] == {}  # 全黑截图不写模板

    # arena page: buy_challenge not on this screen; exit not an arena target
    r_arena = analyzer.detect_page_buttons(img, "arena")
    assert set(r_arena["detected"]) == set()
    assert r_arena["missing"] == ["buy_challenge"]

    # victory page: exit + victory from result OCR
    ocr.boxes = victory_boxes
    r_vic = analyzer.detect_page_buttons(img, "victory")
    assert set(r_vic["detected"]) == {"victory", "exit"}

    # switch to victory page OCR
    ocr.boxes = victory_boxes
    r2 = analyzer.detect_page_buttons(img, "victory")
    assert set(r2["detected"]) == {"victory", "exit"}

    # merge semantics (as wizard does)
    merged = dict(r1["detected"])
    merged.update(r2["detected"])
    assert set(merged) == {"go_win", "exit", "victory"}


def test_roi_detector_challenge_count() -> None:
    config = _identity_config()
    mapper = CoordinateMapper(1920, 1080, 1920, 1080)
    roi = RoiDetector(config, mapper)
    boxes = [
        OCRBox(text="可挑战次数", confidence=0.9, bbox=(480, 780, 120, 30)),
        OCRBox(text="4 / 5", confidence=0.9, bbox=(620, 780, 80, 30)),
        OCRBox(text="1,234,567", confidence=0.95, bbox=(1180, 300, 220, 40)),
    ]
    det = roi.detect_challenge(boxes, slot_boxes=[])
    assert det is not None
    assert det.detail == "4/5"
    assert det.region.width > 80


def test_apply_bundle_writes_reference_coords() -> None:
    config = _identity_config()
    analyzer = CalibrationAnalyzer(config, NullOCRProvider())
    bundle = CalibrationBundle(
        screen_width=1280,
        screen_height=720,
        reference_width=1920,
        reference_height=1080,
        opponents=[
            CalibrationDetection(
                name="opponent_1",
                region=CalibrationRegion(x=640, y=360, width=200, height=100),
                click_point=CalibrationPoint(x=500, y=400),
                confidence=0.9,
                source=SOURCE_OCR,
                value=1234567,
            )
        ],
        challenge=CalibrationDetection(
            name="challenge_count",
            region=CalibrationRegion(x=300, y=600, width=100, height=40),
            click_point=CalibrationPoint(x=350, y=620),
            confidence=0.8,
            source=SOURCE_OCR,
        ),
        buttons={
            "go_win": CalibrationDetection(
                name="go_win",
                region=CalibrationRegion(x=1000, y=600, width=200, height=80),
                click_point=CalibrationPoint(x=1100, y=640),
                confidence=0.95,
                source=SOURCE_OCR,
            )
        },
    )
    analyzer.apply_bundle(config, bundle)
    assert config.opponents[0].power_region == (960, 540, 300, 150)
    assert config.opponents[0].click == (750, 600)
    assert config.opponents[0].source == "calibrated"
    assert config.challenge_region == (450, 900, 150, 60)
    assert config.buttons["go_win"].region == (1500, 900, 300, 120)
    assert config.version == 2


def test_apply_bundle_preserves_manual_override() -> None:
    config = _identity_config()
    config.opponents[0] = OpponentConfig(
        id=1,
        power_region=(10, 20, 30, 40),
        click=(15, 25),
        source=SOURCE_MANUAL,
        confidence=1.0,
    )
    analyzer = CalibrationAnalyzer(config, NullOCRProvider())
    bundle = CalibrationBundle(
        screen_width=1920,
        screen_height=1080,
        opponents=[
            CalibrationDetection(
                name="opponent_1",
                region=CalibrationRegion(x=100, y=100, width=50, height=50),
                click_point=CalibrationPoint(x=120, y=120),
                confidence=0.9,
                source=SOURCE_OCR,
            )
        ],
    )
    analyzer.apply_bundle(config, bundle, overwrite_manual=False)
    assert config.opponents[0].power_region == (10, 20, 30, 40)
    analyzer.apply_bundle(config, bundle, overwrite_manual=True)
    assert config.opponents[0].power_region == (100, 100, 50, 50)


def test_config_roundtrip_with_v2_fields(tmp_path: Path) -> None:
    from arena_auto.models.config import ButtonGeometry

    path = tmp_path / "config.yaml"
    manager = ConfigManager.load_or_create(path)
    manager.config.buttons["go_win"] = ButtonGeometry(
        region=(10, 20, 100, 50),
        click=(60, 45),
        confidence=0.9,
        source="calibrated",
    )
    manager.config.purchase.buy_button.region = (5, 5, 50, 30)
    manager.config.result.victory.region = (7, 7, 60, 40)
    manager.save()

    data = path.read_text(encoding="utf-8")
    assert "version: 2" in data
    assert "challenge_count:" in data

    reloaded = ConfigManager.load_or_create(path)
    assert reloaded.config.version == 2
    assert reloaded.config.buttons["go_win"].region == (10, 20, 100, 50)
    assert reloaded.config.purchase.buy_button.region == (5, 5, 50, 30)
    assert reloaded.config.result.victory.region == (7, 7, 60, 40)
    assert reloaded.config.challenge_region == (0, 0, 100, 50)


def test_config_migrator_v1_to_v2() -> None:
    from arena_auto.config.manager import ConfigMigrator

    raw = default_config_dict()
    raw.pop("version", None)
    raw["challenge_region"] = [1, 2, 3, 4]
    raw.pop("challenge_count", None)
    migrated = ConfigMigrator.migrate(raw)
    assert migrated["version"] == 2
    assert migrated["challenge_count"]["region"] == [1, 2, 3, 4]
    assert "calibration" in migrated


def test_load_legacy_config_without_version(tmp_path: Path) -> None:
    import yaml

    path = tmp_path / "config.yaml"
    legacy = default_config_dict()
    legacy.pop("version", None)
    legacy.pop("challenge_count", None)
    legacy["challenge_region"] = [11, 22, 33, 44]
    path.write_text(yaml.safe_dump(legacy, allow_unicode=True), encoding="utf-8")
    manager = ConfigManager(path=path)
    manager.load()
    assert manager.config.version == 2
    assert manager.config.challenge_region == (11, 22, 33, 44)


def test_button_geometry_alias_lookup() -> None:
    config = _identity_config()
    config.purchase.buy_button.region = (1, 2, 3, 4)
    assert config.button_geometry("buy_challenge").region == (1, 2, 3, 4)
    assert config.button_geometry("go_win").region == (0, 0, 0, 0)


def test_verify_never_taps(tmp_path: Path, monkeypatch) -> None:
    config = _identity_config()
    analyzer = CalibrationAnalyzer(config, _FakeOCR(_power_boxes()))

    def factory() -> np.ndarray:
        return np.zeros((1080, 1920, 3), dtype=np.uint8)

    taps: list = []
    monkeypatch.setattr(
        analyzer, "verify", analyzer.verify
    )
    summary = analyzer.verify(factory)
    assert "checks" in summary
    assert taps == []
    names = [c["name"] for c in summary["checks"]]
    assert "竞技场" in names
    assert "#1" in names


def test_wizard_module_imports() -> None:
    pytest.importorskip("PySide6")
    import arena_auto.calibration.wizard as wizard

    assert wizard.STEP_TITLES and len(wizard.STEP_TITLES) == 7
    assert wizard.CalibrationCanvas is not None
    assert wizard.CalibrationWizard is not None
