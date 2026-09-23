"""ADB device controller built on the adb CLI."""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from typing import List, Optional

import cv2
import numpy as np

from arena_auto.exceptions import (
    AdbDisconnectedError,
    AdbError,
    AdbNotFoundError,
    ScreenshotError,
)
from arena_auto.models.config import AdbConfig

logger = logging.getLogger(__name__)


class AdbController:
    """Thin, thread-safe wrapper around an adb executable."""

    def __init__(self, config: Optional[AdbConfig] = None) -> None:
        self._config = config or AdbConfig()
        self._serial: str = self._config.serial or ""
        self._lock = threading.RLock()
        self._connected: bool = False

    # ------------------------------------------------------------------ config
    @property
    def config(self) -> AdbConfig:
        return self._config

    def update_config(self, config: AdbConfig) -> None:
        with self._lock:
            self._config = config
            if config.serial:
                self._serial = config.serial

    @property
    def serial(self) -> str:
        return self._serial

    # ------------------------------------------------------------------- paths
    def resolve_executable(self) -> str:
        """Locate adb without assuming PATH entry."""
        configured = (self._config.executable or "adb").strip()
        if not configured:
            raise AdbNotFoundError("adb executable 未配置")

        candidate = configured
        # Expand simple relative paths
        if configured.lower().endswith(".exe") or "\\" in configured or "/" in configured:
            from pathlib import Path

            path = Path(configured).expanduser()
            if path.is_file():
                return str(path.resolve())
            # also try next to app root later via PATH search below

        found = shutil.which(candidate)
        if found:
            return found
        if configured.lower() != "adb.exe":
            found = shutil.which("adb") or shutil.which("adb.exe")
            if found:
                return found
        raise AdbNotFoundError(
            f"找不到 adb：{configured!r}。请安装 platform-tools 或在配置中填写完整路径。"
        )

    # ---------------------------------------------------------------- commands
    def _base_cmd(self) -> List[str]:
        return [self.resolve_executable()]

    def _target_cmd(self) -> List[str]:
        cmd = self._base_cmd()
        if self._serial:
            cmd.extend(["-s", self._serial])
        return cmd

    def shell(
        self,
        command: str,
        timeout: Optional[float] = None,
        check: bool = True,
    ) -> str:
        """Run `adb shell <command>` and return stdout text."""
        return self._run(
            self._target_cmd() + ["shell", command],
            timeout=timeout,
            check=check,
            binary=False,
        )

    def _run(
        self,
        args: List[str],
        timeout: Optional[float] = None,
        check: bool = True,
        binary: bool = False,
    ) -> "str | bytes":
        timeout = timeout if timeout is not None else self._config.command_timeout
        logger.debug("adb run: %s", " ".join(args))
        try:
            completed = subprocess.run(
                args,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise AdbNotFoundError(str(exc)) from exc
        except subprocess.TimeoutExpired as exc:
            raise AdbError(f"ADB 命令超时: {' '.join(args)}") from exc
        except OSError as exc:
            raise AdbError(f"ADB 执行失败: {exc}") from exc

        if completed.returncode != 0:
            err = (completed.stderr or b"").decode("utf-8", errors="replace")
            out = (completed.stdout or b"").decode("utf-8", errors="replace")
            message = err.strip() or out.strip() or f"exit={completed.returncode}"
            if check:
                if "device" in message.lower() and (
                    "not found" in message.lower()
                    or "offline" in message.lower()
                    or "disconnected" in message.lower()
                ):
                    self._connected = False
                    raise AdbDisconnectedError(message)
                raise AdbError(message)
            return "" if not binary else b""

        if binary:
            return completed.stdout or b""
        return (completed.stdout or b"").decode("utf-8", errors="replace")

    # --------------------------------------------------------------- discovery
    def list_devices(self) -> List[str]:
        """Return serials of devices in state `device`."""
        with self._lock:
            try:
                raw = self._run(self._base_cmd() + ["devices"], check=True)
            except AdbNotFoundError:
                raise
            except AdbError as exc:
                logger.warning("adb devices failed: %s", exc)
                return []

            devices: List[str] = []
            for line in raw.splitlines()[1:]:
                line = line.strip()
                if not line or line.startswith("*"):
                    continue
                parts = line.split()
                if len(parts) >= 2 and parts[1] == "device":
                    devices.append(parts[0])
            return devices

    def connect(self, serial: Optional[str] = None) -> bool:
        """Connect to a device (USB / Wi-Fi / emulator)."""
        with self._lock:
            if serial:
                self._serial = serial

            if not self._serial:
                devices = self.list_devices()
                if not devices:
                    self._connected = False
                    raise AdbDisconnectedError("未检测到 Android 设备")
                self._serial = devices[0]
                logger.info("自动选择设备: %s", self._serial)

            # Wi-Fi adb host:port style
            if ":" in self._serial and not self._serial.startswith("emulator-"):
                try:
                    self._run(self._base_cmd() + ["connect", self._serial], check=False)
                except AdbError as exc:
                    logger.debug("adb connect skipped: %s", exc)

            try:
                self.shell("echo ok", timeout=5)
            except AdbError as exc:
                # one retry after devices refresh
                devices = self.list_devices()
                if self._serial not in devices:
                    self._connected = False
                    raise AdbDisconnectedError(
                        f"设备 {self._serial} 不可用: {exc}"
                    ) from exc
                self.shell("echo ok", timeout=5)

            self._connected = True
            logger.info("ADB 已连接: %s", self._serial)
            return True

    def disconnect(self) -> None:
        with self._lock:
            self._connected = False
            logger.info("ADB 断开连接: %s", self._serial or "-")

    def is_connected(self) -> bool:
        if not self._connected or not self._serial:
            return False
        try:
            out = self.shell("echo ok", timeout=5, check=True)
            return "ok" in out
        except AdbError:
            self._connected = False
            return False

    def ensure_connected(self, max_attempts: int = 3) -> bool:
        """Reconnect using configured attempt budget."""
        attempts = max(1, max_attempts or self._config.reconnect_attempts)
        for attempt in range(1, attempts + 1):
            try:
                if self.is_connected():
                    return True
                logger.warning("ADB 重连尝试 %s/%s", attempt, attempts)
                self.connect(self._serial or None)
                if self.is_connected():
                    return True
            except AdbError as exc:
                logger.warning("ADB 重连失败: %s", exc)
                time.sleep(0.5)
        return False

    # ---------------------------------------------------------------- actions
    def screenshot(self) -> np.ndarray:
        """Capture screen as BGR ndarray via `exec-out screencap -p`."""
        with self._lock:
            if not self._serial:
                raise ScreenshotError("未选择设备，无法截图")
            raw = self._run(
                self._target_cmd() + ["exec-out", "screencap", "-p"],
                binary=True,
                check=True,
                timeout=self._config.command_timeout,
            )
            if not raw:
                raise ScreenshotError("截图数据为空")
            # Some Windows adb builds convert LF -> CRLF in binary streams.
            if b"\r\n" in raw and raw.count(b"\r\n") > raw.count(b"\n") // 2:
                fixed = raw.replace(b"\r\n", b"\n")
                if fixed != raw:
                    raw = fixed
            buffer = np.frombuffer(raw, dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if image is None:
                raise ScreenshotError("截图解码失败")
            return image

    def tap(self, x: int, y: int, delay: float = 0.0) -> None:
        """Tap at device coordinates with connection check + optional delay."""
        with self._lock:
            if not self.is_connected():
                raise AdbDisconnectedError("点击前设备未连接")
            x, y = int(x), int(y)
            logger.debug("tap(%s, %s)", x, y)
            try:
                self.shell(f"input tap {x} {y}", timeout=10)
            except AdbError as exc:
                self._connected = False
                raise AdbError(f"点击失败 ({x},{y}): {exc}") from exc
            if delay and delay > 0:
                time.sleep(delay)

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration: int = 300,
    ) -> None:
        with self._lock:
            if not self.is_connected():
                raise AdbDisconnectedError("滑动前设备未连接")
            try:
                self.shell(
                    f"input swipe {int(x1)} {int(y1)} {int(x2)} {int(y2)} {int(duration)}",
                    timeout=10,
                )
            except AdbError as exc:
                self._connected = False
                raise AdbError(f"滑动失败: {exc}") from exc

    def keyevent(self, key: int | str) -> None:
        with self._lock:
            if not self.is_connected():
                raise AdbDisconnectedError("按键前设备未连接")
            self.shell(f"input keyevent {int(key)}", timeout=10)

    def back(self) -> None:
        """Android BACK key (keycode 4)."""
        self.keyevent(4)
