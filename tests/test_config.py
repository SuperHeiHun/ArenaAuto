"""Config manager tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from arena_auto.config.manager import ConfigManager, default_config_dict
from arena_auto.exceptions import ConfigError


def test_default_config_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    manager = ConfigManager.load_or_create(path)
    assert path.is_file()
    assert manager.config.automation.power_threshold == 2_000_000
    assert len(manager.config.opponents) == 5

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["automation"]["power_threshold"] == 2_000_000


def test_invalid_power_threshold_raises(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"automation": {"power_threshold": "not-a-number"}}),
        encoding="utf-8",
    )
    manager = ConfigManager(path=path)
    with pytest.raises(ConfigError, match="power_threshold"):
        manager.load()


def test_region_parsing(tmp_path: Path) -> None:
    raw = default_config_dict()
    raw["opponents"][0]["power_region"] = [10, 20, 100, 50]
    raw["opponents"][0]["click"] = [15, 25]
    cfg = ConfigManager.from_dict(raw)
    assert cfg.opponents[0].power_region == (10, 20, 100, 50)
    assert cfg.opponents[0].click == (15, 25)


def test_bad_region_raises() -> None:
    raw = default_config_dict()
    raw["opponents"][0]["power_region"] = [1, 2, 3]
    with pytest.raises(ConfigError, match="power_region"):
        ConfigManager.from_dict(raw)


def test_missing_templates_listed(tmp_path: Path) -> None:
    manager = ConfigManager.load_or_create(tmp_path / "config.yaml")
    missing = manager.missing_templates()
    assert "go_win" in missing
