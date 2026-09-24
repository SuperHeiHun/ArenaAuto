"""Wireless ADB endpoint helpers + ConfigManager.save_adb_serial."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from arena_auto.adb.wireless import (
    format_endpoint,
    is_valid_host,
    is_valid_port,
    is_wireless_serial,
    parse_endpoint,
    parse_pair_code,
    pick_wireless_candidates,
)
from arena_auto.config.manager import ConfigManager
from arena_auto.exceptions import AdbError


def test_parse_endpoint_host_only_defaults_port() -> None:
    ep = parse_endpoint("192.168.1.20")
    assert ep.host == "192.168.1.20"
    assert ep.port == 5555
    assert ep.serial() == "192.168.1.20:5555"


def test_parse_endpoint_with_port_and_scheme() -> None:
    ep = parse_endpoint("http://10.0.0.8:5555")
    assert ep.serial() == "10.0.0.8:5555"


def test_parse_endpoint_invalid_ip() -> None:
    with pytest.raises(AdbError):
        parse_endpoint("999.1.1.1")
    with pytest.raises(AdbError):
        parse_endpoint("192.168.1.1:99999")
    with pytest.raises(AdbError):
        parse_endpoint("")


def test_format_endpoint() -> None:
    assert format_endpoint("192.168.0.5", 5555) == "192.168.0.5:5555"
    with pytest.raises(AdbError):
        format_endpoint("", 5555)


def test_is_wireless_serial() -> None:
    assert is_wireless_serial("192.168.1.9:5555")
    assert not is_wireless_serial("AKMP9K2217902880")
    assert not is_wireless_serial("emulator-5554")
    assert not is_wireless_serial("192.168.1.9")


def test_parse_pair_code() -> None:
    assert parse_pair_code("123456") == "123-456"
    assert parse_pair_code("123 456") == "123-456"
    with pytest.raises(AdbError):
        parse_pair_code("12345")


def test_is_valid_host_port() -> None:
    assert is_valid_host("192.168.1.1")
    assert is_valid_host("pixel.local")
    assert not is_valid_host("")
    assert is_valid_port(1) and is_valid_port(65535)
    assert not is_valid_port(0) and not is_valid_port(65536)


def test_pick_wireless_candidates() -> None:
    serials = ["AKMP9K", "192.168.1.3:5555", "emulator-5554", "10.0.0.2:5555"]
    assert pick_wireless_candidates(serials) == [
        "192.168.1.3:5555",
        "10.0.0.2:5555",
    ]


def test_save_adb_serial_writes_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    manager = ConfigManager.load_or_create(path)
    manager.config.adb.serial = "OLD"
    manager.save()

    manager.save_adb_serial("192.168.1.77:5555")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["adb"]["serial"] == "192.168.1.77:5555"
    assert manager.config.adb.serial == "192.168.1.77:5555"
    # reload round-trip
    reloaded = ConfigManager(path=path)
    reloaded.load()
    assert reloaded.config.adb.serial == "192.168.1.77:5555"
