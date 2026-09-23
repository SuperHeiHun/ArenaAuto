"""Typed configuration models for ArenaAuto."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Region = Tuple[int, int, int, int]  # x, y, width, height
Point = Tuple[int, int]


@dataclass
class AdbConfig:
    executable: str = "adb"
    serial: str = ""
    reconnect_attempts: int = 3
    command_timeout: float = 15.0


@dataclass
class AutomationSettings:
    power_threshold: int = 2_000_000
    click_delay: float = 0.3
    poll_interval: float = 5.0
    dry_run: bool = False


@dataclass
class RecognitionSettings:
    stable_reads: int = 2
    retry_count: int = 3


@dataclass
class OcrCacheSettings:
    enabled: bool = True
    difference_threshold: float = 3.0


@dataclass
class OcrStabilitySettings:
    enabled: bool = True
    max_attempts: int = 3


@dataclass
class OcrSettings:
    provider: str = "easyocr"
    scale: int = 3
    threshold: bool = True
    whitelist: str = "0123456789,"
    lang: str = "ch"
    dpi: int = 300
    device: str = "auto"  # auto | cpu | cuda
    timeout_ms: int = 1500
    cache_enabled: bool = True
    cache_difference_threshold: float = 3.0
    power_allowlist: str = "0123456789,."
    challenge_allowlist: str = "0123456789/"
    cache: OcrCacheSettings = field(default_factory=OcrCacheSettings)
    stability: OcrStabilitySettings = field(default_factory=OcrStabilitySettings)


@dataclass
class LoggingSettings:
    performance: bool = False


@dataclass
class BattleSettings:
    poll_interval: float = 5.0
    max_duration: float = 300.0


@dataclass
class PaddingSettings:
    x: int = 20
    y: int = 20


@dataclass
class CalibrationWeights:
    numeric: float = 0.30
    position: float = 0.25
    spacing: float = 0.20
    structure: float = 0.15
    ocr: float = 0.10


@dataclass
class CalibrationSettings:
    """Tuning knobs for the auto-calibration pipeline (reference-resolution px)."""

    power_region_padding: PaddingSettings = field(
        default_factory=lambda: PaddingSettings(x=20, y=20)
    )
    template_padding: PaddingSettings = field(
        default_factory=lambda: PaddingSettings(x=20, y=15)
    )
    weights: CalibrationWeights = field(default_factory=CalibrationWeights)
    min_power_digits: int = 5
    max_power_digits: int = 10
    min_ocr_confidence: float = 0.30
    x_tolerance: int = 80
    spacing_min: int = 40
    spacing_max: int = 400
    slot_match_tolerance: int = 40
    card_extend_left: int = 400
    high_confidence: float = 0.90
    low_confidence: float = 0.75
    generate_templates: bool = True
    debug_overlays: bool = True


@dataclass
class ButtonGeometry:
    """Calibrated (or manual) region + click point for a named button."""

    region: Region = (0, 0, 0, 0)
    click: Point = (0, 0)
    confidence: float = 0.0
    source: str = "default"  # default | ocr | template | structure | manual | calibrated


@dataclass
class PurchaseSettings:
    enabled: bool = True
    auto_confirm: bool = True
    max_per_session: int = 10
    buy_button: ButtonGeometry = field(default_factory=ButtonGeometry)
    confirm_button: ButtonGeometry = field(default_factory=ButtonGeometry)


@dataclass
class ResultSettings:
    victory: ButtonGeometry = field(default_factory=ButtonGeometry)
    defeat: ButtonGeometry = field(default_factory=ButtonGeometry)
    exit: ButtonGeometry = field(default_factory=ButtonGeometry)


@dataclass
class RecoverySettings:
    allow_back: bool = True
    max_attempts: int = 3


@dataclass
class DebugSettings:
    enabled: bool = True
    save_screenshots: bool = True
    save_failed_recognition: bool = True
    directory: str = "debug"
    # WAIT_RESULT OCR fallback: default OFF (normal battle loop = template only).
    allow_ocr_fallback: bool = False
    # Consecutive template misses before OCR fallback is even considered.
    ocr_fallback_min_failures: int = 3


@dataclass
class TimeoutSettings:
    detect_screen: float = 10.0
    prepare: float = 15.0
    start_battle: float = 10.0
    result: float = 300.0
    purchase: float = 15.0
    recovery: float = 30.0


@dataclass
class ScreenSettings:
    reference_width: int = 1920
    reference_height: int = 1080


@dataclass
class OpponentConfig:
    id: int
    power_region: Region = (0, 0, 100, 50)
    click: Point = (0, 0)
    source: str = "default"  # default | ocr | structure | manual | calibrated
    confidence: float = 0.0


@dataclass
class TemplateEntry:
    path: str = ""
    threshold: float = 0.82


@dataclass
class ArenaProfile:
    """Game-specific UI keywords so modules are not hard-coded to one title."""

    display_name: str = "胜利者的竞技场"
    arena_title_keywords: List[str] = field(
        default_factory=lambda: ["胜利者的竞技场"]
    )
    arena_select_keywords: List[str] = field(
        default_factory=lambda: ["请选择对手"]
    )
    challenge_keywords: List[str] = field(
        default_factory=lambda: ["可挑战次数"]
    )
    go_win_text: str = "去获胜"
    exit_text: str = "退出"
    victory_text: str = "胜利"
    defeat_text: str = "失败"
    buy_text: str = "购买挑战次数"
    confirm_text: str = "确认"
    min_keyword_matches: int = 1


@dataclass
class AppConfig:
    """Root configuration object loaded from config/config.yaml."""

    version: int = 2
    adb: AdbConfig = field(default_factory=AdbConfig)
    automation: AutomationSettings = field(default_factory=AutomationSettings)
    recognition: RecognitionSettings = field(default_factory=RecognitionSettings)
    ocr: OcrSettings = field(default_factory=OcrSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    battle: BattleSettings = field(default_factory=BattleSettings)
    purchase: PurchaseSettings = field(default_factory=PurchaseSettings)
    recovery: RecoverySettings = field(default_factory=RecoverySettings)
    debug: DebugSettings = field(default_factory=DebugSettings)
    timeouts: TimeoutSettings = field(default_factory=TimeoutSettings)
    screen: ScreenSettings = field(default_factory=ScreenSettings)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    opponents: List[OpponentConfig] = field(
        default_factory=lambda: [
            OpponentConfig(id=i, power_region=(0, 0, 100, 50), click=(0, 0))
            for i in range(1, 6)
        ]
    )
    challenge_region: Region = (0, 0, 100, 50)
    buttons: Dict[str, ButtonGeometry] = field(default_factory=dict)
    result: ResultSettings = field(default_factory=ResultSettings)
    templates: Dict[str, TemplateEntry] = field(
        default_factory=lambda: {
            "go_win": TemplateEntry(path="resources/templates/go_win.png", threshold=0.82),
            "exit": TemplateEntry(path="resources/templates/exit.png", threshold=0.82),
            "buy_challenge": TemplateEntry(
                path="resources/templates/buy_challenge.png", threshold=0.82
            ),
            "confirm_buy": TemplateEntry(
                path="resources/templates/confirm_buy.png", threshold=0.82
            ),
            "victory": TemplateEntry(
                path="resources/templates/victory.png", threshold=0.80
            ),
            "defeat": TemplateEntry(
                path="resources/templates/defeat.png", threshold=0.80
            ),
        }
    )
    profile: ArenaProfile = field(default_factory=ArenaProfile)

    def button_geometry(self, name: str) -> ButtonGeometry:
        """Resolve calibrated geometry for a button name (with legacy aliases)."""
        entry = self.buttons.get(name)
        if entry is not None and (entry.region[2] > 0 and entry.region[3] > 0):
            return entry
        if name == "buy_challenge":
            return self.purchase.buy_button
        if name == "confirm_buy":
            return self.purchase.confirm_button
        if name == "victory":
            return self.result.victory
        if name == "defeat":
            return self.result.defeat
        if name == "exit" and (
            self.result.exit.region[2] > 0 and self.result.exit.region[3] > 0
        ):
            return self.result.exit
        if entry is not None:
            return entry
        return ButtonGeometry()
