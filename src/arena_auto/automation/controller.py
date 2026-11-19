"""Main automation engine: explicit state machine driving ADB + recognition."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from arena_auto.adb.controller import AdbController
from arena_auto.automation.state_machine import StateMachine
from arena_auto.config.manager import ConfigManager
from arena_auto.debug.recorder import DebugRecorder
from arena_auto.exceptions import (
    AdbDisconnectedError,
    AdbError,
    ArenaAutoError,
    OCRFailedError,
    RecoveryFailedError,
    RecognitionError,
    StoppedError,
)
from arena_auto.models.config import AppConfig
from arena_auto.models.recognition import ChallengeCount, Opponent
from arena_auto.models.state import AutomationState, ScreenKind, SessionStatistics
from arena_auto.paths import get_data_dir
from arena_auto.recognition.detector import (
    ButtonFinder,
    ChallengeCountReader,
    PowerReader,
    ScreenDetector,
)
from arena_auto.recognition.ocr import OCRProvider, create_ocr_provider
from arena_auto.recognition.screen import CoordinateScaler
from arena_auto.recognition.template import TemplateRecognizer

logger = logging.getLogger(__name__)


@dataclass
class ControllerCallbacks:
    """GUI-facing notifications (all optional, must be non-blocking)."""

    on_state: Optional[Callable[[AutomationState], None]] = None
    on_log: Optional[Callable[[str, str], None]] = None
    on_powers: Optional[Callable[[List[Opponent]], None]] = None
    on_target: Optional[Callable[[int], None]] = None
    on_challenge: Optional[Callable[[ChallengeCount], None]] = None
    on_statistics: Optional[Callable[[SessionStatistics], None]] = None
    on_screenshot: Optional[Callable[[np.ndarray], None]] = None
    on_screen: Optional[Callable[[ScreenKind], None]] = None
    on_error: Optional[Callable[[str], None]] = None
    on_finished: Optional[Callable[[], None]] = None


@dataclass
class _RuntimeContext:
    screenshot: Optional[np.ndarray] = None
    screen: ScreenKind = ScreenKind.UNKNOWN
    opponents: List[Opponent] = field(default_factory=list)
    target: Optional[Opponent] = None
    challenge: Optional[ChallengeCount] = None
    outcome: Optional[str] = None
    recovery_attempts: int = 0
    exit_attempts: int = 0
    opponent_wait_started: float = 0.0
    battle_started_at: float = 0.0
    last_log_banner: str = ""
    pending_purchase: bool = False
    # WAIT_RESULT: consecutive template-only misses (for optional OCR fallback)
    wait_result_template_fails: int = 0


class AutomationController:
    """Orchestrates ADB, OCR, OpenCV, and the state machine safely."""

    def __init__(
        self,
        config_manager: ConfigManager,
        adb: Optional[AdbController] = None,
        ocr: Optional[OCRProvider] = None,
        templates: Optional[TemplateRecognizer] = None,
        callbacks: Optional[ControllerCallbacks] = None,
        debug: Optional[DebugRecorder] = None,
    ) -> None:
        self.config_manager = config_manager
        self.config: AppConfig = config_manager.config
        self.adb = adb or AdbController(self.config.adb)
        self.ocr = ocr or create_ocr_provider(self.config.ocr)
        self.templates = templates or TemplateRecognizer(self.config.templates)
        self.debug = debug or DebugRecorder(self.config.debug)
        self.callbacks = callbacks or ControllerCallbacks()

        self.scaler = CoordinateScaler(
            self.config.screen.reference_width,
            self.config.screen.reference_height,
        )
        self.detector = ScreenDetector(self.ocr, self.config, self.templates, self.scaler)
        self.power_reader = PowerReader(
            self.ocr, self.config, self.config.recognition, self.scaler, self.debug
        )
        self.challenge_reader = ChallengeCountReader(
            self.ocr, self.config, self.scaler, self.debug
        )
        self.buttons = ButtonFinder(self.templates, self.ocr, self.config, self.debug)

        self.sm = StateMachine(
            initial=AutomationState.IDLE,
            on_transition=self._handle_state_change,
        )
        self.stop_event = threading.Event()
        self.statistics = SessionStatistics()
        self.ctx = _RuntimeContext()
        self._running = False
        self._cumulative_path = get_data_dir() / "statistics.json"
        self._historical = self._load_historical()

    # ------------------------------------------------------------------ public
    @property
    def is_running(self) -> bool:
        return self._running

    def request_stop(self) -> None:
        logger.info("请求停止自动化")
        self.stop_event.set()
        # Cancel pending OCR immediately so mid-loop calls abort before starting.
        self._install_ocr_cancel()

    def reset_stop(self) -> None:
        self.stop_event.clear()

    def _install_ocr_cancel(self) -> None:
        ocr = getattr(self, "ocr", None)
        if ocr is not None and hasattr(ocr, "set_cancel_check"):
            try:
                ocr.set_cancel_check(self.stop_event.is_set)
            except Exception:  # noqa: BLE001
                pass
        engine = getattr(getattr(self, "detector", None), "engine", None)
        if engine is not None and hasattr(engine, "set_cancel_check"):
            try:
                engine.set_cancel_check(self.stop_event.is_set)
            except Exception:  # noqa: BLE001
                pass
        for obj_name in ("power_reader", "challenge_reader", "buttons"):
            eng = getattr(getattr(self, obj_name, None), "engine", None)
            if eng is not None and hasattr(eng, "set_cancel_check"):
                try:
                    eng.set_cancel_check(self.stop_event.is_set)
                except Exception:  # noqa: BLE001
                    pass

    def reload_config(self) -> None:
        self.config = self.config_manager.config
        self.adb.update_config(self.config.adb)
        self.templates.set_entries(self.config.templates)
        self.scaler.reference_width = self.config.screen.reference_width
        self.scaler.reference_height = self.config.screen.reference_height
        self.detector.config = self.config
        self.power_reader.config = self.config
        self.power_reader.recognition = self.config.recognition
        self.challenge_reader.config = self.config
        self.buttons.config = self.config
        try:
            from arena_auto.recognition.ocr_engine import set_performance_logging

            set_performance_logging(
                bool(getattr(self.config.logging, "performance", False))
            )
        except Exception:  # noqa: BLE001
            pass
        # refresh engines' settings if OcrEngine is in use
        for obj in (
            getattr(self, "ocr", None),
            getattr(self.detector, "engine", None),
            getattr(self.power_reader, "engine", None),
            getattr(self.challenge_reader, "engine", None),
            getattr(self.buttons, "engine", None),
        ):
            if obj is not None and hasattr(obj, "settings"):
                try:
                    obj.settings = self.config.ocr
                    obj.clear_cache()
                except Exception:  # noqa: BLE001
                    pass
        logger.info("配置已重新加载")

    def run(self) -> None:
        """Blocking automation loop - run inside a worker thread only."""
        if self._running:
            logger.warning("自动化已在运行")
            return
        self._running = True
        self.reset_stop()
        self.ctx = _RuntimeContext()
        self.statistics = SessionStatistics()
        self._emit_statistics()

        # OCR engine init happens here (worker thread), never on the GUI thread.
        try:
            from arena_auto.recognition.ocr_engine import set_performance_logging

            set_performance_logging(
                bool(getattr(self.config.logging, "performance", False))
            )
        except Exception:  # noqa: BLE001
            pass
        self._install_ocr_cancel()
        if hasattr(self.ocr, "ensure_ready") and not self.stop_event.is_set():
            self._log("INFO", "OCR 初始化中...")
            try:
                self.ocr.ensure_ready()  # type: ignore[attr-defined]
                name = getattr(self.ocr, "name", "ocr")
                provider = getattr(getattr(self.ocr, "_provider", None), "name", name)
                self._log("INFO", f"OCR Ready ({provider})")
            except Exception as exc:  # noqa: BLE001
                logger.warning("OCR 初始化失败: %s", exc)
                self._log("WARNING", f"OCR 初始化失败: {exc}")

        missing = self.templates.preload_all()
        if missing:
            self._log(
                "WARNING",
                f"缺少模板文件: {', '.join(missing)}。请进入调试工具创建模板",
            )

        try:
            self.sm.transition(AutomationState.CHECK_DEVICE)
            while not self.stop_event.is_set() and self.sm.state not in (
                AutomationState.STOPPED,
                AutomationState.ERROR,
            ):
                try:
                    self._step()
                except StoppedError:
                    break
                except AdbDisconnectedError as exc:
                    self._handle_adb_failure(exc)
                except (RecognitionError, OCRFailedError) as exc:
                    self._handle_recognition_failure(exc)
                except ArenaAutoError as exc:
                    logger.error("自动化错误: %s", exc)
                    self._emit_error(str(exc))
                    self.sm.transition(AutomationState.RECOVERY)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("未预期异常")
                    self._emit_error(f"未预期异常: {exc}")
                    self.sm.transition(AutomationState.RECOVERY)
        finally:
            self._running = False
            self._save_statistics()
            if self.sm.state not in (AutomationState.STOPPED, AutomationState.ERROR):
                self.sm.transition(AutomationState.STOPPED)
            self._log("INFO", "自动化已停止")
            if self.callbacks.on_finished:
                try:
                    self.callbacks.on_finished()
                except Exception:  # noqa: BLE001
                    pass

    # ----------------------------------------------------------------- helpers
    def _handle_state_change(self, old: AutomationState, new: AutomationState) -> None:
        if self.callbacks.on_state:
            try:
                self.callbacks.on_state(new)
            except Exception:  # noqa: BLE001
                pass
        if new != old:
            self._log("INFO", f"状态: {old.name} -> {new.name}")

    def _log(self, level: str, message: str) -> None:
        logger.log(getattr(logging, level.upper(), logging.INFO), message)
        if self.callbacks.on_log:
            try:
                self.callbacks.on_log(level, message)
            except Exception:  # noqa: BLE001
                pass

    def _emit_error(self, message: str) -> None:
        if self.callbacks.on_error:
            try:
                self.callbacks.on_error(message)
            except Exception:  # noqa: BLE001
                pass

    def _emit_statistics(self) -> None:
        if self.callbacks.on_statistics:
            try:
                self.callbacks.on_statistics(self.statistics)
            except Exception:  # noqa: BLE001
                pass

    def _emit_screenshot(self, image: Optional[np.ndarray]) -> None:
        if image is not None and self.callbacks.on_screenshot:
            try:
                self.callbacks.on_screenshot(image)
            except Exception:  # noqa: BLE001
                pass

    def _wait(self, seconds: float) -> bool:
        """Interruptible sleep. Returns True if stop was requested."""
        if seconds <= 0:
            return self.stop_event.is_set()
        return self.stop_event.wait(seconds)

    def _safe_tap(self, x: int, y: int, label: str = "") -> None:
        if self.stop_event.is_set():
            raise StoppedError("用户停止")
        if self.config.automation.dry_run:
            self._log("INFO", f"[DRY RUN] tap({x},{y}) {label}".rstrip())
            return
        delay = self.config.automation.click_delay
        self._log("INFO", f"点击 ({x},{y}) {label}".rstrip())
        self.adb.tap(x, y, delay=delay)

    def _shot(self) -> np.ndarray:
        started = time.perf_counter()
        image = self.adb.screenshot()
        elapsed = (time.perf_counter() - started) * 1000.0
        try:
            from arena_auto.recognition.ocr_engine import perf_log

            perf_log("ADB screenshot", elapsed)
        except Exception:  # noqa: BLE001
            pass
        self.scaler.maybe_update_from_image(image)
        self.ctx.screenshot = image
        self._emit_screenshot(image)
        return image

    def _step(self) -> None:
        if self.stop_event.is_set():
            raise StoppedError("用户停止")
        state = self.sm.state
        handler = {
            AutomationState.CHECK_DEVICE: self._state_check_device,
            AutomationState.DETECT_SCREEN: self._state_detect_screen,
            AutomationState.ARENA: self._state_arena,
            AutomationState.SELECT_OPPONENT: self._state_select_opponent,
            AutomationState.WAIT_PREPARE: self._state_wait_prepare,
            AutomationState.PREPARE: self._state_prepare,
            AutomationState.START_BATTLE: self._state_start_battle,
            AutomationState.WAIT_RESULT: self._state_wait_result,
            AutomationState.RESULT: self._state_result,
            AutomationState.EXIT_RESULT: self._state_exit_result,
            AutomationState.CHECK_CHALLENGE_COUNT: self._state_check_challenge,
            AutomationState.BUY_CHALLENGE: self._state_buy_challenge,
            AutomationState.CONFIRM_PURCHASE: self._state_confirm_purchase,
            AutomationState.RECOVERY: self._state_recovery,
            AutomationState.IDLE: self._state_check_device,
            AutomationState.STOPPED: self._state_stopped,
            AutomationState.ERROR: self._state_error,
        }.get(state)
        if handler is None:
            logger.error("未知状态: %s", state)
            self.sm.transition(AutomationState.RECOVERY)
            return
        handler()

    # ------------------------------------------------------------------ states
    def _state_check_device(self) -> None:
        self._log("INFO", "检查 ADB 设备...")
        attempts = max(1, self.config.adb.reconnect_attempts)
        if not self.adb.ensure_connected(attempts):
            raise AdbDisconnectedError("ADB 设备连接失败")
        self._log("INFO", f"ADB device: {self.adb.serial}")
        missing = self.config_manager.missing_templates()
        if missing:
            self._log("WARNING", f"缺少模板文件: {', '.join(missing)}")
        self.sm.transition(AutomationState.DETECT_SCREEN)

    def _state_detect_screen(self) -> None:
        timeout = self.config.timeouts.detect_screen
        if self.sm.has_timed_out(timeout):
            self._log("ERROR", "页面识别超时")
            self.sm.transition(AutomationState.RECOVERY)
            return
        image = self._shot()
        kind = self.detector.detect(image)
        self.ctx.screen = kind
        if self.callbacks.on_screen:
            try:
                self.callbacks.on_screen(kind)
            except Exception:  # noqa: BLE001
                pass
        self.debug.save(image, category=kind.value if kind != ScreenKind.UNKNOWN else "debug")

        if kind == ScreenKind.ARENA:
            self.sm.transition(AutomationState.ARENA)
        elif kind == ScreenKind.PREPARE:
            self.sm.transition(AutomationState.PREPARE)
        elif kind == ScreenKind.RESULT:
            self.sm.transition(AutomationState.RESULT)
        elif kind == ScreenKind.PURCHASE:
            self.sm.transition(AutomationState.BUY_CHALLENGE)
        elif kind == ScreenKind.UNKNOWN:
            # battle in progress is also unknown chrome - if battle timer active, wait
            if self.ctx.battle_started_at > 0:
                self.sm.transition(AutomationState.WAIT_RESULT)
            else:
                self._log("WARNING", "未知页面，进入恢复流程")
                self.sm.transition(AutomationState.RECOVERY)
        else:
            self.sm.transition(AutomationState.RECOVERY)

    def _state_arena(self) -> None:
        image = self._shot()
        # 1) challenge count
        challenge = self.challenge_reader.read(image)
        if challenge is None:
            self.statistics.recognition_failures += 1
            self._emit_statistics()
            self.sm.transition(AutomationState.RECOVERY)
            return
        self.ctx.challenge = challenge
        if self.callbacks.on_challenge:
            try:
                self.callbacks.on_challenge(challenge)
            except Exception:  # noqa: BLE001
                pass

        if challenge.current <= 0:
            self._log("WARNING", "挑战次数不足")
            if (
                self.config.purchase.enabled
                and self.statistics.session_purchase_count
                < self.config.purchase.max_per_session
            ):
                # 次数为 0 时不直接找购买按钮：先选战力并点击对手，
                # 由游戏弹出购买弹窗后再点「购买」。
                self.ctx.pending_purchase = True
                self._log(
                    "INFO",
                    "挑战次数为 0，先检测战力并点击对手（随后将弹出购买弹窗）",
                )
                self.sm.transition(AutomationState.SELECT_OPPONENT)
            else:
                self._log("ERROR", "挑战次数不足且不可购买，停止自动化")
                self.sm.transition(AutomationState.STOPPED)
            return

        if self.ctx.pending_purchase:
            self.ctx.pending_purchase = False

        # 2) read powers
        self.sm.transition(AutomationState.SELECT_OPPONENT)

    def _state_select_opponent(self) -> None:
        def factory() -> np.ndarray:
            if self.stop_event.is_set():
                raise StoppedError("用户停止")
            return self.shot_safe()

        threshold = self.config.automation.power_threshold
        self._log("INFO", f"战力阈值：{threshold}")

        try:
            # Short-circuit: read #1 first; stop OCR as soon as power <= threshold.
            opponents, selected = self.power_reader.read_until_at_or_below(
                factory,
                threshold,
                self.config.opponents,
                stop_event=self.stop_event,
            )
        except AdbError as exc:
            raise AdbDisconnectedError(str(exc)) from exc
        except OCRFailedError as exc:
            self.statistics.recognition_failures += 1
            self._emit_statistics()
            self._log("ERROR", str(exc))
            if self.debug.enabled and self.ctx.screenshot is not None:
                self.debug.save(self.ctx.screenshot, category="failed", tag="ocr_power")
            self.sm.transition(AutomationState.RECOVERY)
            return

        if self.stop_event.is_set():
            raise StoppedError("用户停止")

        self.ctx.opponents = opponents
        if self.callbacks.on_powers:
            try:
                self.callbacks.on_powers(opponents)
            except Exception:  # noqa: BLE001
                pass

        powers = [opp.power for opp in opponents]

        if selected is None:
            self._log(
                "WARNING",
                f"没有战力 <= 阈值的对手: {powers}，等待刷新",
            )
            self.sm.transition(AutomationState.ARENA)
            self._wait(self.config.automation.poll_interval)
            return

        assert selected.power is not None
        self.ctx.target = selected
        self._log(
            "INFO",
            f"选择目标：#{selected.id} 战力={selected.power}（已读 {len(opponents)} 个）",
        )
        if self.callbacks.on_target:
            try:
                self.callbacks.on_target(selected.id)
            except Exception:  # noqa: BLE001
                pass

        x, y = selected.click_position
        if x == 0 and y == 0:
            self._log("ERROR", f"对手 #{selected.id} 的点击坐标未配置，禁止盲点")
            self.sm.transition(AutomationState.RECOVERY)
            return
        self._safe_tap(x, y, label=f"对手 #{selected.id}")
        self.ctx.opponent_wait_started = time.monotonic()
        if self.ctx.pending_purchase:
            self._log("INFO", "已点击对手，等待购买弹窗")
            self.sm.transition(AutomationState.BUY_CHALLENGE)
            return
        self.sm.transition(AutomationState.WAIT_PREPARE)

    def shot_safe(self) -> np.ndarray:
        return self._shot()

    def _state_wait_prepare(self) -> None:
        timeout = self.config.timeouts.prepare
        if self.sm.has_timed_out(timeout):
            self._log("ERROR", "等待备战页面超时")
            self.sm.transition(AutomationState.RECOVERY)
            return
        if self._wait(min(1.0, max(0.2, self.config.automation.click_delay))):
            raise StoppedError("用户停止")
        image = self._shot()
        kind = self.detector.detect(image)
        if kind == ScreenKind.PREPARE:
            self._log("INFO", "当前页面：PREPARE")
            self.sm.transition(AutomationState.PREPARE)
            return
        if kind == ScreenKind.ARENA:
            # click may have failed
            self._log("WARNING", "仍在竞技场，重新选择对手")
            self.sm.transition(AutomationState.ARENA)
            return
        if kind == ScreenKind.RESULT:
            self.sm.transition(AutomationState.RESULT)
            return

    def _state_prepare(self) -> None:
        timeout = self.config.timeouts.start_battle
        image = self._shot()
        result = self.buttons.find_go_win(image)
        if result.found:
            self._log("INFO", "检测到「去获胜」")
            cx, cy = result.center
            self._safe_tap(cx, cy, label="去获胜")
            self.ctx.battle_started_at = time.monotonic()
            self.sm.transition(AutomationState.WAIT_RESULT)
            return
        if self.sm.has_timed_out(timeout):
            self._log("ERROR", "未找到「去获胜」，超时")
            self.sm.transition(AutomationState.RECOVERY)

    def _state_start_battle(self) -> None:
        # merged into prepare -> wait_result; kept for enum completeness
        self.sm.transition(AutomationState.WAIT_RESULT)

    def _state_wait_result(self) -> None:
        """Normal path: ADB screenshot + OpenCV templates only (OCR = 0).

        EasyOCR / full-screen OCR is forbidden unless debug.allow_ocr_fallback
        is true AND templates failed N consecutive polls AND state looks abnormal.
        """
        if self.stop_event.is_set():
            raise StoppedError("用户停止")
        battle_cfg = self.config.battle
        if self.ctx.battle_started_at <= 0:
            self.ctx.battle_started_at = time.monotonic()
        elapsed = time.monotonic() - self.ctx.battle_started_at
        if elapsed > battle_cfg.max_duration:
            self._log("ERROR", "战斗超时")
            self.sm.transition(AutomationState.RECOVERY)
            return

        interval = max(0.5, battle_cfg.poll_interval)
        self._log("INFO", "战斗进行中")
        if self._wait(interval):
            raise StoppedError("用户停止")
        if self.stop_event.is_set():
            raise StoppedError("用户停止")

        image = self._shot()
        if self.stop_event.is_set():
            raise StoppedError("用户停止")

        # --- Template-only (priority: Victory, Defeat, then exit / go_win) ---
        outcome = self.detector.outcome_from_templates(image)
        if outcome is not None:
            self.ctx.outcome = outcome
            self.ctx.wait_result_template_fails = 0
            self.sm.transition(AutomationState.RESULT)
            return

        if self.templates.find(image, "exit").found:
            # On result chrome; outcome already checked (templates only).
            self.ctx.wait_result_template_fails = 0
            self.sm.transition(AutomationState.RESULT)
            return

        if self.templates.find(image, "go_win").found:
            self.ctx.wait_result_template_fails = 0
            self.sm.transition(AutomationState.PREPARE)
            return

        if self.templates.find(image, "buy_challenge").found:
            self.ctx.wait_result_template_fails = 0
            self.sm.transition(AutomationState.BUY_CHALLENGE)
            return

        # Still in battle (or unknown chrome): no OCR on normal path.
        self.ctx.wait_result_template_fails += 1

        # --- Optional OCR fallback (explicit opt-in only) ---
        debug_cfg = self.config.debug
        if not getattr(debug_cfg, "allow_ocr_fallback", False):
            return
        min_fails = max(1, int(getattr(debug_cfg, "ocr_fallback_min_failures", 3)))
        if self.ctx.wait_result_template_fails < min_fails:
            return
        # "State looks abnormal": battle well past typical duration, or long miss streak.
        abnormal = (
            elapsed > min(30.0, battle_cfg.max_duration * 0.25)
            or self.ctx.wait_result_template_fails >= min_fails * 2
        )
        if not abnormal:
            return

        self._log(
            "WARNING",
            f"模板连续失败 {self.ctx.wait_result_template_fails} 次且状态异常，"
            f"启用 OCR 兜底（allow_ocr_fallback）",
        )
        kind, ocr_outcome = self.detector.detect_screen_and_outcome(
            image, allow_ocr=True
        )
        if kind == ScreenKind.RESULT or ocr_outcome is not None:
            self.ctx.outcome = ocr_outcome
            self.ctx.wait_result_template_fails = 0
            self.sm.transition(AutomationState.RESULT)
            return
        if kind == ScreenKind.ARENA:
            self._log("WARNING", "战斗期间回到竞技场，未检测到结算")
            self.ctx.battle_started_at = 0.0
            self.ctx.wait_result_template_fails = 0
            self.sm.transition(AutomationState.ARENA)
            return
        if kind == ScreenKind.PREPARE:
            self.ctx.wait_result_template_fails = 0
            self.sm.transition(AutomationState.PREPARE)
            return

    def _state_result(self) -> None:
        """Already know RESULT: template outcome + find exit only (no detect(), no full OCR)."""
        if self.stop_event.is_set():
            raise StoppedError("用户停止")
        image = self._shot()
        if self.stop_event.is_set():
            raise StoppedError("用户停止")

        # Template-only outcome (victory / defeat).
        outcome = self.detector.outcome_from_templates(image)
        if outcome is None and self.ctx.outcome is not None:
            outcome = self.ctx.outcome

        exit_visible = self.templates.find(image, "exit").found

        if outcome is None and not exit_visible:
            # Wait for victory/defeat/exit templates — never detector.detect() / full OCR.
            if self.sm.has_timed_out(self.config.timeouts.result):
                self._log(
                    "ERROR",
                    "无法识别结算结果（模板未命中，OCR 兜底未开启）",
                )
                self.statistics.recognition_failures += 1
                self._emit_statistics()
                if getattr(self.config.debug, "allow_ocr_fallback", False):
                    outcome = self.detector.detect_outcome(image)
                    exit_visible = self.buttons.find_exit(image, full_ocr=False).found
                if outcome is None and not exit_visible:
                    self.sm.transition(AutomationState.RECOVERY)
                    return
            else:
                if self._wait(1.0):
                    raise StoppedError("用户停止")
                self.sm.reenter()
                return

        if outcome is None and exit_visible:
            self._log("WARNING", "结算页未识别胜负模板，继续退出")

        self.ctx.outcome = outcome
        if outcome is not None:
            self.statistics.total_battles += 1
            if outcome == "WIN":
                self.statistics.victories += 1
                self._log("INFO", "检测到胜利")
            else:
                self.statistics.defeats += 1
                self._log("INFO", "检测到失败")
            self._emit_statistics()
        self.ctx.battle_started_at = 0.0
        self.ctx.exit_attempts = 0
        self.sm.transition(AutomationState.EXIT_RESULT)

    def _state_exit_result(self) -> None:
        if self.stop_event.is_set():
            raise StoppedError("用户停止")
        max_attempts = 3
        if self.ctx.exit_attempts >= max_attempts:
            self._log("ERROR", "退出结算页失败次数过多")
            self.sm.transition(AutomationState.RECOVERY)
            return

        image = self._shot()
        if self.stop_event.is_set():
            raise StoppedError("用户停止")

        # Only find the exit button (template first; no detector.detect / full OCR).
        exit_result = self.buttons.find_exit(image, full_ocr=False)
        if exit_result.found:
            self._log("INFO", "点击「退出」")
            cx, cy = exit_result.center
            self._safe_tap(cx, cy, label="退出")
            self.ctx.exit_attempts += 1
            # wait for arena: templates only + optional challenge ROI (allowed)
            for _ in range(int(self.config.timeouts.prepare) + 1):
                if self._wait(1.0):
                    raise StoppedError("用户停止")
                if self.stop_event.is_set():
                    raise StoppedError("用户停止")
                frame = self._shot()
                if self.stop_event.is_set():
                    raise StoppedError("用户停止")
                # Still on result chrome?
                if self.templates.find(frame, "exit").found:
                    continue
                if self.templates.find(frame, "go_win").found:
                    self._log("INFO", "返回备战页")
                    self.ctx.outcome = None
                    self.sm.transition(AutomationState.PREPARE)
                    return
                # Arena confirm: challenge ROI OCR is allowed (not full-screen).
                count = self.challenge_reader.read(frame)
                if count is not None:
                    self._log("INFO", "返回竞技场")
                    self.ctx.challenge = count
                    outcome = self.ctx.outcome or "?"
                    self._log("INFO", f"本轮完成：{outcome}")
                    self.ctx.outcome = None
                    self.sm.transition(AutomationState.CHECK_CHALLENGE_COUNT)
                    return
                # Optional full detect only with OCR fallback enabled.
                if getattr(self.config.debug, "allow_ocr_fallback", False):
                    kind = self.detector.detect(frame)
                    if kind == ScreenKind.ARENA:
                        self._log("INFO", "返回竞技场")
                        outcome = self.ctx.outcome or "?"
                        self._log("INFO", f"本轮完成：{outcome}")
                        self.ctx.outcome = None
                        self.sm.transition(AutomationState.CHECK_CHALLENGE_COUNT)
                        return
                    if kind == ScreenKind.RESULT:
                        continue
            self._log("WARNING", "退出后未回到竞技场，再次尝试")
            return

        if self.sm.has_timed_out(self.config.timeouts.prepare):
            self.sm.transition(AutomationState.RECOVERY)
            return
        if self._wait(1.0):
            raise StoppedError("用户停止")

    def _state_check_challenge(self) -> None:
        image = self._shot()
        challenge = self.challenge_reader.read(image)
        if challenge is None:
            self.statistics.recognition_failures += 1
            self._emit_statistics()
            self.sm.transition(AutomationState.RECOVERY)
            return
        self.ctx.challenge = challenge
        if self.callbacks.on_challenge:
            try:
                self.callbacks.on_challenge(challenge)
            except Exception:  # noqa: BLE001
                pass
        self.sm.transition(AutomationState.ARENA)

    def _state_buy_challenge(self) -> None:
        if self.sm.has_timed_out(self.config.timeouts.purchase):
            self._log("ERROR", "购买挑战次数超时")
            self.sm.transition(AutomationState.RECOVERY)
            return
        image = self._shot()
        result = self.buttons.find_buy(image)
        if result.found:
            self._log("INFO", "点击「购买」")
            cx, cy = result.center
            self._safe_tap(cx, cy, label="购买")
            self.sm.transition(AutomationState.CONFIRM_PURCHASE)
            return
        # maybe dialog already open with confirm only
        if self.buttons.find_confirm(image).found:
            self.sm.transition(AutomationState.CONFIRM_PURCHASE)
            return
        # 点击对手后未弹出购买弹窗（例如实际进入了备战）
        if self.ctx.pending_purchase and self.buttons.find_go_win(image).found:
            self._log("WARNING", "未出现购买弹窗，直接进入备战")
            self.ctx.pending_purchase = False
            self.sm.transition(AutomationState.PREPARE)
            return
        if self._wait(1.0):
            raise StoppedError("用户停止")

    def _state_confirm_purchase(self) -> None:
        if self.sm.has_timed_out(self.config.timeouts.purchase):
            self._log("ERROR", "确认购买超时")
            self.sm.transition(AutomationState.RECOVERY)
            return

        image = self._shot()
        result = self.buttons.find_confirm(image)
        if result.found:
            if not self.config.purchase.auto_confirm:
                self._log("WARNING", "自动确认购买已关闭，停止自动化")
                self.sm.transition(AutomationState.STOPPED)
                return
            self._log("INFO", "点击「确认」购买")
            cx, cy = result.center
            self._safe_tap(cx, cy, label="确认购买")
            self._count_purchase()
            self._await_challenge_restore()
            return

        # 无「确认」按钮：点击「购买」可能已直接完成购买
        count = self.challenge_reader.read(image)
        previous = self.ctx.challenge
        if count is not None and count.current > 0 and (
            previous is None
            or previous.current <= 0
            or count.current != previous.current
        ):
            self._count_purchase()
            self._finish_purchase_success(count)
            return

        if self._wait(1.0):
            raise StoppedError("用户停止")

    def _count_purchase(self) -> None:
        self.statistics.session_purchase_count += 1
        self.statistics.purchases += 1
        self._emit_statistics()

    def _finish_purchase_success(self, count: ChallengeCount) -> None:
        self.ctx.challenge = count
        self.ctx.pending_purchase = False
        self._log("INFO", f"购买成功，挑战次数：{count}")
        if self.callbacks.on_challenge:
            try:
                self.callbacks.on_challenge(count)
            except Exception:  # noqa: BLE001
                pass
        # 对手可能已在购买前点击：备战页直接继续，否则回竞技场/重新识别
        frame = self.ctx.screenshot
        kind = self.detector.detect(frame) if frame is not None else ScreenKind.ARENA
        if kind == ScreenKind.PREPARE:
            self.sm.transition(AutomationState.PREPARE)
            return
        if kind == ScreenKind.ARENA:
            self.sm.transition(AutomationState.ARENA)
            return
        self.sm.transition(AutomationState.DETECT_SCREEN)

    def _await_challenge_restore(self) -> None:
        """Poll OCR challenge count after confirm; route by success/failure."""
        previous = self.ctx.challenge
        prev_n = previous.current if previous is not None else 0
        for i in range(8):
            if i > 0 and self._wait(1.0):
                raise StoppedError("用户停止")
            frame = self._shot()
            count = self.challenge_reader.read(frame)
            if count is not None and (count.current > 0 or count.current > prev_n):
                self._finish_purchase_success(count)
                return
            kind = self.detector.detect(frame)
            if kind == ScreenKind.ARENA:
                # 回到竞技场但次数未恢复：停止重复购买
                break
        self._log("ERROR", "购买后挑战次数未恢复，停止重复购买")
        self.ctx.pending_purchase = False
        self.sm.transition(AutomationState.ARENA)

    def _state_recovery(self) -> None:
        self.statistics.recoveries += 1
        self._emit_statistics()
        max_attempts = max(1, self.config.recovery.max_attempts)
        self.ctx.recovery_attempts += 1
        self._log(
            "WARNING",
            f"进入恢复流程 ({self.ctx.recovery_attempts}/{max_attempts})",
        )
        if self.ctx.recovery_attempts > max_attempts:
            self._log("ERROR", "超过最大恢复次数，安全停止")
            self.sm.transition(AutomationState.ERROR)
            return

        try:
            image = self._shot()
        except AdbError as exc:
            raise AdbDisconnectedError(str(exc)) from exc

        self.debug.save(image, category="recovery")
        kind = self.detector.detect(image)

        if kind == ScreenKind.ARENA:
            self.ctx.recovery_attempts = 0
            self.ctx.battle_started_at = 0.0
            self.ctx.pending_purchase = False
            self.sm.transition(AutomationState.ARENA)
            return
        if kind == ScreenKind.PREPARE:
            self.ctx.recovery_attempts = 0
            self.sm.transition(AutomationState.PREPARE)
            return
        if kind == ScreenKind.RESULT:
            self.ctx.recovery_attempts = 0
            self.sm.transition(AutomationState.RESULT)
            return
        if kind == ScreenKind.PURCHASE:
            self.ctx.recovery_attempts = 0
            self.sm.transition(AutomationState.BUY_CHALLENGE)
            return

        # unknown page
        if self.config.recovery.allow_back and not self.config.automation.dry_run:
            self._log("INFO", "未知页面，尝试 BACK")
            try:
                self.adb.back()
            except AdbError as exc:
                logger.warning("BACK 失败: %s", exc)
        elif self.config.recovery.allow_back and self.config.automation.dry_run:
            self._log("INFO", "[DRY RUN] back()")

        if self._wait(1.5):
            raise StoppedError("用户停止")

    def _state_stopped(self) -> None:
        self.stop_event.set()

    def _state_error(self) -> None:
        self._emit_error("自动化进入 ERROR 状态，已停止点击")
        self.stop_event.set()

    # ----------------------------------------------------------------- errors
    def _handle_adb_failure(self, exc: Exception) -> None:
        self._log("ERROR", f"ADB 错误: {exc}")
        self._emit_error(str(exc))
        attempts = max(1, self.config.adb.reconnect_attempts)
        if self.adb.ensure_connected(attempts):
            self._log("INFO", "ADB 已重新连接")
            self.sm.transition(AutomationState.DETECT_SCREEN)
            return
        self._log("ERROR", "ADB 重连失败，停止自动化")
        self.sm.transition(AutomationState.ERROR)

    def _handle_recognition_failure(self, exc: Exception) -> None:
        self.statistics.recognition_failures += 1
        self._emit_statistics()
        self._log("ERROR", f"识别失败: {exc}")
        if self.ctx.screenshot is not None:
            self.debug.save_failed(self.ctx.screenshot, name="recognition")
        self.sm.transition(AutomationState.RECOVERY)

    # ------------------------------------------------------------- statistics
    def _load_historical(self) -> dict:
        try:
            if self._cumulative_path.is_file():
                data = json.loads(self._cumulative_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取历史统计失败: %s", exc)
        return {}

    def _save_statistics(self) -> None:
        try:
            current_session = self.statistics.to_dict()
            # strip session-only fields from cumulative merge
            cumulative = dict(self._historical)
            for key in (
                "total_battles",
                "victories",
                "defeats",
                "purchases",
                "recognition_failures",
                "recoveries",
            ):
                cumulative[key] = int(cumulative.get(key, 0) or 0) + int(
                    current_session.get(key, 0) or 0
                )
            cumulative["last_saved_at"] = time.time()
            cumulative["last_session"] = current_session
            self._cumulative_path.parent.mkdir(parents=True, exist_ok=True)
            self._cumulative_path.write_text(
                json.dumps(cumulative, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._historical = cumulative
        except Exception as exc:  # noqa: BLE001
            logger.warning("保存统计失败: %s", exc)

    @property
    def historical_statistics(self) -> dict:
        return dict(self._historical)
