"""Wireless ADB helpers: endpoint parse/validate, connect/pair orchestration."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from arena_auto.exceptions import AdbError

DEFAULT_WIRELESS_PORT = 5555
DEFAULT_PAIR_PORT = 37000  # Android 11+ often uses high ports; caller supplies real one

_IPV4_RE = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)$"
)
_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")


@dataclass(frozen=True)
class WirelessEndpoint:
    host: str
    port: int

    def serial(self) -> str:
        return f"{self.host}:{self.port}"

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.serial()


def is_valid_host(host: str) -> bool:
    host = (host or "").strip()
    if not host:
        return False
    # Looks like IPv4 (4 numeric labels) — must be a real IPv4, not a hostname.
    if host.count(".") == 3 and all(
        part.isdigit() for part in host.split(".")
    ):
        return bool(_IPV4_RE.match(host))
    if _IPV4_RE.match(host):
        return True
    if "." in host:
        # allow simple hostnames / mDNS like pixel.local
        labels = host.split(".")
        return all(_HOSTNAME_RE.match(label) for label in labels if label)
    return bool(_HOSTNAME_RE.match(host))


def is_valid_port(port: int) -> bool:
    try:
        p = int(port)
    except (TypeError, ValueError):
        return False
    return 1 <= p <= 65535


def parse_endpoint(value: str) -> WirelessEndpoint:
    """Parse `host`, `host:port`, or `http://host:port` into WirelessEndpoint."""
    raw = (value or "").strip()
    if not raw:
        raise AdbError("无线 ADB 地址为空")
    # strip scheme if present
    if "://" in raw:
        raw = raw.split("://", 1)[1]
    raw = raw.split("/", 1)[0]
    if ":" in raw:
        host, _, port_s = raw.rpartition(":")
        if not is_valid_port(int(port_s)) if port_s.isdigit() else True:
            if not port_s.isdigit() or not is_valid_port(int(port_s)):
                raise AdbError(f"无效端口: {port_s!r}")
        port = int(port_s)
    else:
        host, port = raw, DEFAULT_WIRELESS_PORT
    host = host.strip()
    if not is_valid_host(host):
        raise AdbError(f"无效主机/IP: {host!r}")
    return WirelessEndpoint(host=host, port=port)


def format_endpoint(host: str, port: int = DEFAULT_WIRELESS_PORT) -> str:
    endpoint = WirelessEndpoint(host=str(host).strip(), port=int(port))
    if not is_valid_host(endpoint.host) or not is_valid_port(endpoint.port):
        raise AdbError(f"无效无线端点: {endpoint.serial()}")
    return endpoint.serial()


def is_wireless_serial(serial: str) -> bool:
    s = (serial or "").strip()
    if not s or s.startswith("emulator-"):
        return False
    if ":" not in s:
        return False
    # USB serials can contain ':' rarely; require host:port with numeric port
    host, _, port = s.rpartition(":")
    return bool(host) and port.isdigit() and is_valid_port(int(port))


def parse_pair_code(code: str) -> str:
    """Normalize pairing code: digits and optional hyphen, 6 digits total."""
    digits = re.sub(r"[^0-9]", "", code or "")
    if len(digits) != 6:
        raise AdbError(f"配对码应为 6 位数字，收到: {code!r}")
    return f"{digits[:3]}-{digits[3:]}"


def classify_device_line(line: str) -> Optional[Tuple[str, str]]:
    """Parse `adb devices` line -> (serial, state) or None."""
    line = (line or "").strip()
    if not line or line.startswith("*"):
        return None
    parts = line.split()
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def pick_wireless_candidates(serials: List[str]) -> List[str]:
    """Filter device list down to wireless (host:port) serials."""
    return [s for s in serials if is_wireless_serial(s)]
