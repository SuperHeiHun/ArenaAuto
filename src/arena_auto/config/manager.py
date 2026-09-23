"""YAML configuration load / save / validate."""

from __future__ import annotations

import copy
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from arena_auto.exceptions import ConfigError
from arena_auto.models.config import (
    AdbConfig,
    AppConfig,
    ArenaProfile,
    AutomationSettings,
    BattleSettings,
    ButtonGeometry,
    CalibrationSettings,
    CalibrationWeights,
    DebugSettings,
    LoggingSettings,
    OpponentConfig,
    OcrCacheSettings,
    OcrSettings,
    OcrStabilitySettings,
    PaddingSettings,
    PurchaseSettings,
    RecognitionSettings,
    RecoverySettings,
    ResultSettings,
    ScreenSettings,
    TemplateEntry,
    TimeoutSettings,
)
from arena_auto.paths import get_config_path

logger = logging.getLogger(__name__)


def _as_region(value: Any, field_name: str) -> Tuple[int, int, int, int]:
    if value is None:
        return (0, 0, 100, 50)
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ConfigError(f"{field_name} 必须是 [x, y, width, height] 四元组")
    try:
        x, y, w, h = (int(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} 必须全部为整数") from exc
    if w <= 0 or h <= 0:
        raise ConfigError(f"{field_name} 宽高必须大于 0")
    return (x, y, w, h)


def _as_region_opt(value: Any, field_name: str) -> Tuple[int, int, int, int]:
    """Like _as_region but an all-zero region means 'unset' and is allowed."""
    if value is None:
        return (0, 0, 0, 0)
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ConfigError(f"{field_name} 必须是 [x, y, width, height] 四元组")
    try:
        x, y, w, h = (int(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} 必须全部为整数") from exc
    if w < 0 or h < 0:
        raise ConfigError(f"{field_name} 宽高不能为负")
    if (w == 0 or h == 0) and (w != 0 or h != 0):
        raise ConfigError(f"{field_name} 宽高必须同时为 0（未设置）或同时大于 0")
    return (x, y, w, h)


def _as_geometry(value: Any, field_name: str) -> ButtonGeometry:
    if value is None:
        return ButtonGeometry()
    if not isinstance(value, dict):
        raise ConfigError(f"{field_name} 必须是映射")
    region = _as_region_opt(value.get("region"), f"{field_name}.region")
    click = _as_point(value.get("click"), f"{field_name}.click")
    try:
        confidence = float(value.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name}.confidence 必须是数字") from exc
    source = str(value.get("source", "") or "")
    if not source:
        source = "manual" if (region[2] > 0 or click != (0, 0)) else "default"
    return ButtonGeometry(
        region=region, click=click, confidence=confidence, source=source
    )


def _as_point(value: Any, field_name: str) -> Tuple[int, int]:
    if value is None:
        return (0, 0)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"{field_name} 必须是 [x, y]")
    try:
        return (int(value[0]), int(value[1]))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} 必须为整数坐标") from exc


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def default_config_dict() -> Dict[str, Any]:
    """Serializable default configuration."""
    return asdict(AppConfig())


class ConfigMigrator:
    """Upgrade older YAML payloads to the current schema version."""

    CURRENT_VERSION = 2

    @classmethod
    def migrate(cls, raw: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            return raw
        try:
            version = int(raw.get("version", 1) or 1)
        except (TypeError, ValueError):
            version = 1
        if version >= cls.CURRENT_VERSION:
            data = dict(raw)
            data["version"] = cls.CURRENT_VERSION
            return data

        data = dict(raw)
        if version < 2:
            logger.info("[CONFIG] 配置迁移: version %s -> %s", version, cls.CURRENT_VERSION)
            data.setdefault("calibration", {})
            # challenge_region (v1 flat key) keeps working; challenge_count.region
            # is now the preferred alias and is mirrored on save.
            if data.get("challenge_region") and not data.get("challenge_count"):
                data["challenge_count"] = {"region": data["challenge_region"]}
        data["version"] = cls.CURRENT_VERSION
        return data


class ConfigManager:
    """Reads, validates, and persists config/config.yaml."""

    def __init__(self, path: Optional[Path] = None, config: Optional[AppConfig] = None):
        self.path = Path(path) if path else get_config_path()
        self.config: AppConfig = config or AppConfig()
        self.last_error: str = ""

    # ------------------------------------------------------------------ factory
    @classmethod
    def load_or_create(cls, path: Optional[Path] = None) -> "ConfigManager":
        manager = cls(path=path)
        if not manager.path.exists():
            manager.config = AppConfig()
            manager.save()
            logger.info("已创建默认配置: %s", manager.path)
            return manager
        manager.load()
        return manager

    # --------------------------------------------------------------------- io
    def load(self) -> AppConfig:
        if not self.path.exists():
            self.config = AppConfig()
            self.save()
            return self.config
        try:
            raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"YAML 解析失败: {exc}") from exc
        except OSError as exc:
            raise ConfigError(f"无法读取配置: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError("config.yaml 根节点必须是映射(dict)")
        raw = ConfigMigrator.migrate(raw)
        self.config = self.from_dict(raw)
        return self.config

    def save(self, config: Optional[AppConfig] = None) -> None:
        if config is not None:
            self.config = config
        data = asdict(self.config)
        data["version"] = max(int(self.config.version), ConfigMigrator.CURRENT_VERSION)
        data["challenge_count"] = {"region": list(data.get("challenge_region", []))}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
            self.path.write_text(text, encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"无法写入配置: {exc}") from exc

    # ---------------------------------------------------------------- convert
    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> AppConfig:
        if not isinstance(raw, dict):
            raise ConfigError("配置必须是 dict")

        defaults = default_config_dict()
        data = _merge_dict(defaults, raw)

        try:
            power_threshold = int(data["automation"].get("power_threshold", 2_000_000))
        except (TypeError, ValueError) as exc:
            raise ConfigError("power_threshold 必须是数字") from exc
        if power_threshold < 0:
            raise ConfigError("power_threshold 不能为负数")

        adb = AdbConfig(
            executable=str(data["adb"].get("executable", "adb")),
            serial=str(data["adb"].get("serial", "") or ""),
            reconnect_attempts=int(data["adb"].get("reconnect_attempts", 3)),
            command_timeout=float(data["adb"].get("command_timeout", 15.0)),
        )
        automation = AutomationSettings(
            power_threshold=power_threshold,
            click_delay=float(data["automation"].get("click_delay", 0.3)),
            poll_interval=float(data["automation"].get("poll_interval", 5.0)),
            dry_run=bool(data["automation"].get("dry_run", False)),
        )
        recognition = RecognitionSettings(
            stable_reads=max(1, int(data["recognition"].get("stable_reads", 2))),
            retry_count=max(1, int(data["recognition"].get("retry_count", 3))),
        )
        ocr_raw = data.get("ocr") or {}
        if not isinstance(ocr_raw, dict):
            raise ConfigError("ocr 必须是映射")
        cache_raw = ocr_raw.get("cache") or {}
        if not isinstance(cache_raw, dict):
            raise ConfigError("ocr.cache 必须是映射")
        stability_raw = ocr_raw.get("stability") or {}
        if not isinstance(stability_raw, dict):
            raise ConfigError("ocr.stability 必须是映射")
        cache_enabled = bool(
            ocr_raw.get("cache_enabled", cache_raw.get("enabled", True))
        )
        cache_threshold = float(
            ocr_raw.get(
                "cache_difference_threshold",
                cache_raw.get("difference_threshold", 3.0),
            )
        )
        ocr = OcrSettings(
            provider=str(ocr_raw.get("provider", "easyocr")),
            scale=max(1, int(ocr_raw.get("scale", 3))),
            threshold=bool(ocr_raw.get("threshold", True)),
            whitelist=str(ocr_raw.get("whitelist", "0123456789,")),
            lang=str(ocr_raw.get("lang", "ch")),
            dpi=int(ocr_raw.get("dpi", 300)),
            device=str(ocr_raw.get("device", "auto")),
            timeout_ms=max(0, int(ocr_raw.get("timeout_ms", 1500))),
            cache_enabled=cache_enabled,
            cache_difference_threshold=cache_threshold,
            power_allowlist=str(
                ocr_raw.get("power_allowlist", "0123456789,.")
            ),
            challenge_allowlist=str(
                ocr_raw.get("challenge_allowlist", "0123456789/")
            ),
            cache=OcrCacheSettings(
                enabled=cache_enabled,
                difference_threshold=cache_threshold,
            ),
            stability=OcrStabilitySettings(
                enabled=bool(stability_raw.get("enabled", True)),
                max_attempts=max(1, int(stability_raw.get("max_attempts", 3))),
            ),
        )
        log_raw = data.get("logging") or {}
        if not isinstance(log_raw, dict):
            raise ConfigError("logging 必须是映射")
        logging_settings = LoggingSettings(
            performance=bool(log_raw.get("performance", False)),
        )
        battle = BattleSettings(
            poll_interval=float(data["battle"].get("poll_interval", 5.0)),
            max_duration=float(data["battle"].get("max_duration", 300.0)),
        )
        purchase = PurchaseSettings(
            enabled=bool(data["purchase"].get("enabled", True)),
            auto_confirm=bool(data["purchase"].get("auto_confirm", True)),
            max_per_session=int(data["purchase"].get("max_per_session", 10)),
            buy_button=_as_geometry(
                data["purchase"].get("buy_button"), "purchase.buy_button"
            ),
            confirm_button=_as_geometry(
                data["purchase"].get("confirm_button"), "purchase.confirm_button"
            ),
        )
        recovery = RecoverySettings(
            allow_back=bool(data["recovery"].get("allow_back", True)),
            max_attempts=int(data["recovery"].get("max_attempts", 3)),
        )
        debug = DebugSettings(
            enabled=bool(data["debug"].get("enabled", True)),
            save_screenshots=bool(data["debug"].get("save_screenshots", True)),
            save_failed_recognition=bool(
                data["debug"].get("save_failed_recognition", True)
            ),
            directory=str(data["debug"].get("directory", "debug")),
            allow_ocr_fallback=bool(data["debug"].get("allow_ocr_fallback", False)),
            ocr_fallback_min_failures=max(
                1, int(data["debug"].get("ocr_fallback_min_failures", 3))
            ),
        )
        timeouts = TimeoutSettings(**{
            field: float(data["timeouts"].get(field, getattr(TimeoutSettings(), field)))
            for field in TimeoutSettings.__dataclass_fields__
        })
        screen = ScreenSettings(
            reference_width=int(data["screen"].get("reference_width", 1920)),
            reference_height=int(data["screen"].get("reference_height", 1080)),
        )

        try:
            version = int(data.get("version", ConfigMigrator.CURRENT_VERSION))
        except (TypeError, ValueError) as exc:
            raise ConfigError("version 必须是整数") from exc
        if version < 1:
            raise ConfigError("version 必须 >= 1")

        cal_raw = data.get("calibration") or {}
        if not isinstance(cal_raw, dict):
            raise ConfigError("calibration 必须是映射")
        pad_power = cal_raw.get("power_region_padding") or {}
        pad_tpl = cal_raw.get("template_padding") or {}
        weights_raw = cal_raw.get("weights") or {}
        defaults_cal = CalibrationSettings()
        calibration = CalibrationSettings(
            power_region_padding=PaddingSettings(
                x=int(pad_power.get("x", 20)),
                y=int(pad_power.get("y", 20)),
            ),
            template_padding=PaddingSettings(
                x=int(pad_tpl.get("x", 20)),
                y=int(pad_tpl.get("y", 15)),
            ),
            weights=CalibrationWeights(
                numeric=float(weights_raw.get("numeric", 0.30)),
                position=float(weights_raw.get("position", 0.25)),
                spacing=float(weights_raw.get("spacing", 0.20)),
                structure=float(weights_raw.get("structure", 0.15)),
                ocr=float(weights_raw.get("ocr", 0.10)),
            ),
            min_power_digits=int(
                cal_raw.get("min_power_digits", defaults_cal.min_power_digits)
            ),
            max_power_digits=int(
                cal_raw.get("max_power_digits", defaults_cal.max_power_digits)
            ),
            min_ocr_confidence=float(
                cal_raw.get("min_ocr_confidence", defaults_cal.min_ocr_confidence)
            ),
            x_tolerance=int(cal_raw.get("x_tolerance", defaults_cal.x_tolerance)),
            spacing_min=int(cal_raw.get("spacing_min", defaults_cal.spacing_min)),
            spacing_max=int(cal_raw.get("spacing_max", defaults_cal.spacing_max)),
            slot_match_tolerance=int(
                cal_raw.get("slot_match_tolerance", defaults_cal.slot_match_tolerance)
            ),
            card_extend_left=int(
                cal_raw.get("card_extend_left", defaults_cal.card_extend_left)
            ),
            high_confidence=float(
                cal_raw.get("high_confidence", defaults_cal.high_confidence)
            ),
            low_confidence=float(
                cal_raw.get("low_confidence", defaults_cal.low_confidence)
            ),
            generate_templates=bool(
                cal_raw.get("generate_templates", defaults_cal.generate_templates)
            ),
            debug_overlays=bool(
                cal_raw.get("debug_overlays", defaults_cal.debug_overlays)
            ),
        )

        buttons_raw = data.get("buttons") or {}
        if not isinstance(buttons_raw, dict):
            raise ConfigError("buttons 必须是映射")
        buttons: Dict[str, ButtonGeometry] = {}
        for name, entry in buttons_raw.items():
            buttons[str(name)] = _as_geometry(entry, f"buttons.{name}")

        result_raw = data.get("result") or {}
        if not isinstance(result_raw, dict):
            raise ConfigError("result 必须是映射")
        result = ResultSettings(
            victory=_as_geometry(result_raw.get("victory"), "result.victory"),
            defeat=_as_geometry(result_raw.get("defeat"), "result.defeat"),
            exit=_as_geometry(result_raw.get("exit"), "result.exit"),
        )

        opponents_raw = data.get("opponents") or []
        if not isinstance(opponents_raw, list) or not opponents_raw:
            raise ConfigError("opponents 必须是非空列表")
        opponents: List[OpponentConfig] = []
        for index, item in enumerate(opponents_raw, start=1):
            if not isinstance(item, dict):
                raise ConfigError(f"opponents[{index - 1}] 必须是映射")
            oid = int(item.get("id", index))
            power_region = _as_region(
                item.get("power_region"), f"opponents[{oid}].power_region"
            )
            click = _as_point(item.get("click"), f"opponents[{oid}].click")
            source = str(item.get("source", "") or "")
            if not source:
                is_placeholder = power_region == (0, 0, 100, 50) and click == (0, 0)
                source = "default" if is_placeholder else "manual"
            try:
                confidence = float(item.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"opponents[{oid}].confidence 必须是数字"
                ) from exc
            opponents.append(
                OpponentConfig(
                    id=oid,
                    power_region=power_region,
                    click=click,
                    source=source,
                    confidence=confidence,
                )
            )
        if len(opponents) < 5:
            logger.warning("对手配置少于 5 个，将补全占位项")
            existing_ids = {o.id for o in opponents}
            for oid in range(1, 6):
                if oid not in existing_ids:
                    opponents.append(OpponentConfig(id=oid))
        opponents.sort(key=lambda o: o.id)

        challenge_region = _as_region(
            data.get("challenge_count", {}).get("region")
            or data.get("challenge_region"),
            "challenge_count.region",
        )

        templates: Dict[str, TemplateEntry] = {}
        templates_raw = data.get("templates") or {}
        if not isinstance(templates_raw, dict):
            raise ConfigError("templates 必须是映射")
        for name, entry in templates_raw.items():
            if not isinstance(entry, dict):
                raise ConfigError(f"templates.{name} 必须是映射")
            templates[str(name)] = TemplateEntry(
                path=str(entry.get("path", "")),
                threshold=float(entry.get("threshold", 0.82)),
            )
        # ensure required keys exist
        for required in (
            "go_win",
            "exit",
            "buy_challenge",
            "confirm_buy",
            "victory",
            "defeat",
        ):
            templates.setdefault(
                required,
                TemplateEntry(path=f"resources/templates/{required}.png", threshold=0.82),
            )

        profile_raw = data.get("profile") or {}
        if not isinstance(profile_raw, dict):
            raise ConfigError("profile 必须是映射")
        profile = ArenaProfile(
            display_name=str(profile_raw.get("display_name", "胜利者的竞技场")),
            arena_title_keywords=list(
                profile_raw.get("arena_title_keywords") or ["胜利者的竞技场"]
            ),
            arena_select_keywords=list(
                profile_raw.get("arena_select_keywords") or ["请选择对手"]
            ),
            challenge_keywords=list(
                profile_raw.get("challenge_keywords") or ["可挑战次数"]
            ),
            go_win_text=str(profile_raw.get("go_win_text", "去获胜")),
            exit_text=str(profile_raw.get("exit_text", "退出")),
            victory_text=str(profile_raw.get("victory_text", "胜利")),
            defeat_text=str(profile_raw.get("defeat_text", "失败")),
            buy_text=str(profile_raw.get("buy_text", "购买挑战次数")),
            confirm_text=str(profile_raw.get("confirm_text", "确认")),
            min_keyword_matches=int(profile_raw.get("min_keyword_matches", 1)),
        )

        config = AppConfig(
            version=version,
            adb=adb,
            automation=automation,
            recognition=recognition,
            ocr=ocr,
            logging=logging_settings,
            battle=battle,
            purchase=purchase,
            recovery=recovery,
            debug=debug,
            timeouts=timeouts,
            screen=screen,
            calibration=calibration,
            opponents=opponents,
            challenge_region=challenge_region,
            buttons=buttons,
            result=result,
            templates=templates,
            profile=profile,
        )
        cls.validate(config)
        return config

    @staticmethod
    def validate(config: AppConfig) -> None:
        if config.version < 1:
            raise ConfigError("version 必须 >= 1")
        if config.automation.power_threshold < 0:
            raise ConfigError("power_threshold 必须是数字且不能为负")
        if config.battle.poll_interval <= 0:
            raise ConfigError("battle.poll_interval 必须大于 0")
        if config.recognition.stable_reads < 1:
            raise ConfigError("recognition.stable_reads 至少为 1")
        if config.purchase.max_per_session < 0:
            raise ConfigError("purchase.max_per_session 不能为负")
        if config.screen.reference_width <= 0 or config.screen.reference_height <= 0:
            raise ConfigError("screen.reference_width/height 必须大于 0")

    # ---------------------------------------------------------------- helpers
    def missing_templates(self) -> List[str]:
        """Return template names whose PNG files are missing on disk."""
        from arena_auto.paths import get_app_root

        missing: List[str] = []
        for name, entry in self.config.templates.items():
            path = Path(entry.path)
            if not path.is_absolute():
                path = get_app_root() / path
            if not path.is_file():
                missing.append(name)
        return sorted(missing)

    def get_template_path(self, name: str) -> Path:
        entry = self.config.templates.get(name)
        if entry is None:
            raise ConfigError(f"未配置模板: {name}")
        path = Path(entry.path)
        if not path.is_absolute():
            path = get_app_root() / path
        return path
