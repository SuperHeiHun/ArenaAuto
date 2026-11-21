"""Main automation engine: explicit state machine driving ADB + recognition."""

from __future__ import annotations

import json
import copy
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from arena_auto.adb.controller import AdbController
from arena_auto.automation.state_machine import StateMachine
from arena_auto.calibration.coordinate_mapper import CoordinateMapper
from arena_auto.calibration.models import CalibrationPoint, CalibrationRegion
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
from arena_auto.models.config import AppConfig, OpponentConfig
from arena_auto.models.recognition import ChallengeCount, Opponent, OCRBox
from arena_auto.models.state import AutomationState, ScreenKind, SessionStatistics
from arena_auto.paths import get_data_dir
from arena_auto.recognition.detector import (
    ButtonFinder,
    ChallengeCountReader,
    PowerReader,
    ScreenDetector,
)
from arena_auto.recognition.ocr import OCRProvider, create_ocr_provider
from arena_auto.recognition.auto_targeting import AutoTargeting
from arena_auto.recognition.scene_change import SettledSceneChange
from arena_auto.recognition.ocr_engine import OcrEngine
from arena_auto.recognition.screen import CoordinateScaler, crop_region
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
    screenshot_at: float = 0.0
    screen: ScreenKind = ScreenKind.UNKNOWN
    opponents: List[Opponent] = field(default_factory=list)
    target: Optional[Opponent] = None
    challenge: Optional[ChallengeCount] = None
    outcome: Optional[str] = None
    recovery_attempts: int = 0
    exit_attempts: int = 0
    opponent_wait_started: float = 0.0
    opponent_wait_snapshot_saved: bool = False
    battle_started_at: float = 0.0
    last_log_banner: str = ""
    pending_purchase: bool = False
    # WAIT_RESULT: consecutive template-only misses (for optional OCR fallback)
    wait_result_template_fails: int = 0
    auto_selection_attempts: int = 0
    auto_config_saved: bool = False
    result_ocr_last_at: float = 0.0
    layout_frame: Optional[np.ndarray] = None
    layout_frame_at: float = 0.0
    arena_frame: Optional[np.ndarray] = None
    prepare_frame: Optional[np.ndarray] = None
    prepare_frame_at: float = 0.0
    result_scene_change: SettledSceneChange = field(default_factory=SettledSceneChange)


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
        self.auto_targeting = AutoTargeting(self.ocr, self.config)
        self._attach_frame_cache()
        self._ocr_signature = OcrEngine.shared_provider_key(self.config.ocr)

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
    def _attach_frame_cache(self) -> None:
        from arena_auto.recognition.ocr_pipeline import FrameOCRCache

        self.frame_cache = FrameOCRCache(self.ocr)
        # Manual ROI mode retains its original page recognition path.
        self.detector.frame_cache = self.frame_cache if self.config.automation.auto_targeting else None
        self.auto_targeting.frame_cache = self.frame_cache
        self.auto_targeting.debug = self.debug
        self.buttons.frame_cache = self.frame_cache if self.config.automation.auto_targeting else None

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

    def reload_config(self, *, snapshot: bool = False) -> None:
        if self._running:
            logger.warning("自动化运行中，本轮配置已锁定；停止后才能重载")
            return
        self.config = copy.deepcopy(self.config_manager.config) if snapshot else self.config_manager.config
        self.adb.update_config(self.config.adb)
        self.templates.set_entries(self.config.templates)
        self.scaler.reference_width = self.config.screen.reference_width
        self.scaler.reference_height = self.config.screen.reference_height
        signature = OcrEngine.shared_provider_key(self.config.ocr)
        if signature != self._ocr_signature or (snapshot and isinstance(self.ocr, OcrEngine) and not self.ocr.ready):
            # Rebuild lazily: changing dropdowns must not retain an old CPU Reader.
            self.ocr = create_ocr_provider(self.config.ocr)
            self.detector = ScreenDetector(self.ocr, self.config, self.templates, self.scaler)
            self.power_reader = PowerReader(self.ocr, self.config, self.config.recognition, self.scaler, self.debug)
            self.challenge_reader = ChallengeCountReader(self.ocr, self.config, self.scaler, self.debug)
            self.buttons = ButtonFinder(self.templates, self.ocr, self.config, self.debug)
            self.auto_targeting = AutoTargeting(self.ocr, self.config)
            self._ocr_signature = signature
            self._install_ocr_cancel()
        self.detector.config = self.config
        self.power_reader.config = self.config
        self.power_reader.recognition = self.config.recognition
        self.challenge_reader.config = self.config
        self.buttons.config = self.config
        self.auto_targeting.config = self.config
        self._attach_frame_cache()
        self.ctx.layout_frame = None
        self.ctx.arena_frame = None
        self.ctx.prepare_frame = None
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
        self.reload_config(snapshot=True)
        self._running = True
        self.ctx = _RuntimeContext()
        self.statistics = SessionStatistics()
        self._emit_statistics()
        mode = "自动初始化（实时定位）" if self.config.automation.auto_targeting else "固定 ROI（已标定配置）"
        self._log("INFO", f"本轮模式：{mode}；OCR={self.config.ocr.provider}/{self.config.ocr.lang}；"
                  f"战力阈值={self.config.automation.power_threshold}；本轮参数已锁定")

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
                device = getattr(getattr(self.ocr, "_provider", None), "device_label", "CPU")
                self._log("INFO", f"OCR Ready ({provider}) device={device}")
            except Exception as exc:  # noqa: BLE001
                logger.warning("OCR 初始化失败: %s", exc)
                self._log("WARNING", f"OCR 初始化失败: {exc}")

        missing = self.templates.preload_all()
        if missing:
            self._log(
                "WARNING",
                (f"自动初始化：以下可选模板缺失，将使用 OCR: {', '.join(missing)}"
                 if self.config.automation.auto_targeting else
                 f"缺少模板文件: {', '.join(missing)}。可通过半自动配置生成模板"),
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
        started = time.perf_counter()
        stopped = self.stop_event.wait(seconds)
        from arena_auto.recognition.ocr_engine import perf_log

        perf_log("Interruptible wait", (time.perf_counter() - started) * 1000.0)
        return stopped

    def _safe_tap(self, x: int, y: int, label: str = "") -> None:
        if self.stop_event.is_set():
            raise StoppedError("用户停止")
        if self.config.automation.dry_run:
            self._log("INFO", f"[DRY RUN] tap({x},{y}) {label}".rstrip())
            return
        delay = self.config.automation.click_delay
        self.frame_cache.clear()
        self.detector.invalidate_frame_cache()
        self.ctx.layout_frame = None
        self.ctx.arena_frame = None
        self.ctx.prepare_frame = None
        self._log("INFO", f"点击 ({x},{y}) {label}".rstrip())
        self.adb.tap(x, y, delay=delay)

    def _shot(self) -> np.ndarray:
        started = time.perf_counter()
        captured_at = time.monotonic()
        self.frame_cache.clear()
        self.ctx.layout_frame = None
        self.ctx.arena_frame = None
        self.ctx.prepare_frame = None
        image = self.adb.screenshot()
        elapsed = (time.perf_counter() - started) * 1000.0
        try:
            from arena_auto.recognition.ocr_engine import perf_log

            perf_log("ADB screenshot", elapsed)
        except Exception:  # noqa: BLE001
            pass
        processing_started = time.perf_counter()
        self.scaler.maybe_update_from_image(image)
        self.ctx.screenshot = image
        self.ctx.screenshot_at = captured_at
        self._emit_screenshot(image)
        logger.info("截图流程耗时：获取 %.0fms；缩放/状态/界面回调 %.0fms；总计 %.0fms",
                    elapsed, (time.perf_counter() - processing_started) * 1000.0,
                    (time.perf_counter() - started) * 1000.0)
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
        started = time.perf_counter()
        try:
            handler()
        finally:
            from arena_auto.recognition.ocr_engine import perf_log

            perf_log(f"State {state.name}", (time.perf_counter() - started) * 1000.0)

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
            if self.config.automation.auto_targeting:
                self.ctx.arena_frame = image
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
        image = self.ctx.arena_frame
        self.ctx.arena_frame = None
        if image is None or time.monotonic() - self.ctx.screenshot_at > 1.0:
            image = self._shot()
        # 1) challenge count
        challenge = self._read_challenge(image)
        if challenge is None:
            self.statistics.recognition_failures += 1
            self._emit_statistics()
            self.sm.transition(AutomationState.STOPPED if self.config.automation.auto_targeting else AutomationState.RECOVERY)
            return
        self.ctx.challenge = challenge
        if self.config.automation.auto_targeting:
            self.ctx.layout_frame = image
            self.ctx.layout_frame_at = self.ctx.screenshot_at
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

        automatic = self.config.automation.auto_targeting
        if automatic and self.ctx.auto_selection_attempts >= self.config.automation.auto_targeting_max_attempts:
            self._log("ERROR", "自动选点重复点击后仍未进入备战，安全停止")
            self.sm.transition(AutomationState.STOPPED)
            return

        try:
            if automatic:
                self._log("INFO", "自动初始化：共用半自动配置 OCR（原图优先，空结果预处理重试），战力仍多帧复核")
                layout_frame = self.ctx.layout_frame
                self.ctx.layout_frame = None
                # Only discovery may reuse the immediately preceding arena
                # frame. All numeric reads and pre-tap validation stay fresh.
                age = time.monotonic() - self.ctx.layout_frame_at
                layout_hint = None
                if layout_frame is not None and 1.0 < age <= 5.0:
                    layout_hint, layout_frame = layout_frame, None
                    self._log("INFO", f"布局缓存年龄 {age:.2f}s：尝试新截图验证后复用")
                elif age > 5.0:
                    layout_frame = None
                    self._log("INFO", f"布局缓存过期 {age:.2f}s：重新整图定位")
                elif layout_frame is not None:
                    self._log("INFO", f"布局复用同帧 OCR（年龄 {age:.2f}s）")

                def selection_factory():
                    nonlocal layout_frame
                    if self.stop_event.is_set():
                        raise StoppedError("用户停止")
                    if layout_frame is not None:
                        image, layout_frame = layout_frame, None
                        return image
                    return factory()

                if layout_hint is not None:
                    opponents, selected = self.auto_targeting.select(
                        selection_factory, threshold, self.stop_event, layout_hint=layout_hint
                    )
                else:
                    opponents, selected = self.auto_targeting.select(selection_factory, threshold, self.stop_event)
                if self.auto_targeting.last_image is not None:
                    overlay = self.auto_targeting.draw(self.auto_targeting.last_image)
                    self.debug.save(overlay, category="arena", tag="auto_targeting")
                    self._emit_screenshot(overlay)
                    self._log("INFO", f"布局定位输入: {self.auto_targeting.input_mode}")
            else:
                # Short-circuit manual/configured mode preserves the existing path.
                opponents, selected = self.power_reader.read_until_at_or_below(
                    factory,
                    threshold,
                    self.config.opponents,
                    stop_event=self.stop_event,
                )
        except AdbError as exc:
            raise AdbDisconnectedError(str(exc)) from exc
        except OCRFailedError as exc:
            if self.stop_event.is_set():
                raise StoppedError("用户停止") from exc
            self.statistics.recognition_failures += 1
            self._emit_statistics()
            self._log("ERROR", str(exc))
            if self.debug.enabled and self.ctx.screenshot is not None:
                self.debug.save(self.ctx.screenshot, category="failed", tag="ocr_power")
            self.sm.transition(AutomationState.STOPPED if automatic else AutomationState.RECOVERY)
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
        if not automatic:
            # Calibrated/manual YAML stores reference coordinates. Template and
            # live auto-targeting results already use screenshot pixels.
            x, y = self.scaler.scale_point(x, y)
        if x == 0 and y == 0:
            self._log("ERROR", f"对手 #{selected.id} 的点击坐标未配置，禁止盲点")
            self.sm.transition(AutomationState.RECOVERY)
            return
        if automatic and not self.auto_targeting.verify_target(self._shot(), selected):
            self._log("ERROR", "点击前截图与识别时的卡片不一致（可能已刷新），取消点击")
            self.sm.transition(AutomationState.STOPPED)
            return
        if automatic:
            card = next((c for c in self.auto_targeting.last_cards if c.opponent.id == selected.id), None)
            if card is not None:
                self._log("INFO", f"对手点击区域：{card.target_source} card={card.region} point=({x},{y})")
        self._safe_tap(x, y, label=f"对手 #{selected.id}")
        if automatic:
            self.ctx.auto_selection_attempts += 1
        self.ctx.opponent_wait_started = time.monotonic()
        self.ctx.opponent_wait_snapshot_saved = False
        if self.ctx.pending_purchase:
            self._log("INFO", "已点击对手，等待购买弹窗")
            self.sm.transition(AutomationState.BUY_CHALLENGE)
            return
        self.sm.transition(AutomationState.WAIT_PREPARE)

    def shot_safe(self) -> np.ndarray:
        return self._shot()

    def _read_challenge(self, image: np.ndarray) -> Optional[ChallengeCount]:
        if self.config.automation.auto_targeting:
            return self.auto_targeting.read_challenge(image)
        return self.challenge_reader.read(image)

    def _save_auto_targeting_config(self) -> None:
        """Persist verified auto-targeting geometry after reaching PREPARE.

        The run keeps using its frozen snapshot. Only opponent ROI/click fields
        are updated, in reference-resolution coordinates, so the user can switch
        to fixed ROI mode later without repeating the semi-automatic wizard.
        """
        if not self.config.automation.auto_targeting:
            return
        if not self.config.automation.auto_targeting_save_config:
            return
        if self.ctx.auto_config_saved or self.ctx.target is None:
            return
        if self.ctx.auto_selection_attempts <= 0:
            return
        image = self.auto_targeting.last_image
        cards = self.auto_targeting.last_cards
        if image is None or not cards:
            return
        if not any(card.opponent.id == self.ctx.target.id for card in cards):
            return

        height, width = image.shape[:2]
        mapper = CoordinateMapper(
            self.config.screen.reference_width,
            self.config.screen.reference_height,
            width,
            height,
        )
        updates: Dict[int, OpponentConfig] = {}
        for card in cards:
            opponent = card.opponent
            x, y, w, h = opponent.power_region
            px, py = opponent.click_position
            if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
                logger.warning("自动配置跳过越界战力区域: #%s %s", opponent.id, opponent.power_region)
                continue
            if not (0 <= px < width and 0 <= py < height):
                logger.warning("自动配置跳过越界点击坐标: #%s %s", opponent.id, opponent.click_position)
                continue
            updates[opponent.id] = OpponentConfig(
                id=opponent.id,
                power_region=mapper.actual_region_to_reference(
                    CalibrationRegion.from_tuple(opponent.power_region)
                ),
                click=mapper.actual_point_to_reference(
                    CalibrationPoint.from_tuple(opponent.click_position)
                ),
                source="calibrated",
                confidence=1.0 if card.power_verified else 0.5,
            )
        if not updates:
            self._log("WARNING", "自动初始化成功，但没有可保存的有效对手坐标")
            return

        try:
            saved = self.config_manager.save_opponent_geometry(updates)
        except Exception as exc:  # noqa: BLE001
            logger.warning("自动保存对手配置失败: %s", exc)
            self._log("WARNING", f"自动初始化成功，但保存配置失败（本轮继续）: {exc}")
            return
        if not saved:
            return
        self.ctx.auto_config_saved = True
        ids = ", ".join(f"#{opponent_id}" for opponent_id in sorted(saved))
        self._log(
            "INFO",
            f"自动初始化成功：已保存对手配置 {ids} -> {self.config_manager.path}；"
            "关闭自动初始化后可直接使用",
        )

    def _state_wait_prepare(self) -> None:
        timeout = self.config.timeouts.prepare
        if self.sm.has_timed_out(timeout):
            self._log("ERROR", "等待备战页面超时")
            self.sm.transition(AutomationState.STOPPED if self.config.automation.auto_targeting else AutomationState.RECOVERY)
            return
        if self._wait(min(1.0, max(0.2, self.config.automation.click_delay))):
            raise StoppedError("用户停止")
        image = self._shot()
        kind = self.detector.detect(image)
        if self.config.automation.auto_targeting and not self.ctx.opponent_wait_snapshot_saved:
            self.debug.save(image, category="prepare", tag=f"after_opponent_{self.ctx.auto_selection_attempts}_{kind.value}")
            self.ctx.opponent_wait_snapshot_saved = True
        if kind == ScreenKind.PREPARE:
            self._log("INFO", "当前页面：PREPARE")
            self._save_auto_targeting_config()
            self.ctx.auto_selection_attempts = 0
            if self.config.automation.auto_targeting:
                self.ctx.prepare_frame = image
                self.ctx.prepare_frame_at = self.ctx.screenshot_at
            self.sm.transition(AutomationState.PREPARE)
            return
        if kind == ScreenKind.PURCHASE and self.config.automation.auto_targeting:
            self.ctx.pending_purchase = True
            self.sm.transition(AutomationState.BUY_CHALLENGE)
            return
        if kind == ScreenKind.ARENA:
            # A single early frame is not enough to declare the tap ineffective.
            if self.sm.time_in_state() < min(5.0, max(0.5, timeout / 3.0)):
                self._log("INFO", "点击后暂仍在竞技场，等待页面切换，不重复点击")
                return
            if self.config.automation.auto_targeting:
                self.debug.save(image, category="prepare", tag="opponent_click_unchanged")
            self._log("WARNING", "仍在竞技场，重新选择对手")
            self.sm.transition(AutomationState.ARENA)
            return
        if kind == ScreenKind.RESULT:
            self.sm.transition(AutomationState.RESULT)
            return

    def _state_prepare(self) -> None:
        timeout = self.config.timeouts.start_battle
        old_frame = self.ctx.prepare_frame
        self.ctx.prepare_frame = None
        observed_button = None
        prepare_structure = None
        age = time.monotonic() - self.ctx.prepare_frame_at
        if old_frame is None:
            reason = "无备战前帧"
        elif age > 5.0:
            reason = f"备战证据过期（{age:.2f}s）"
        else:
            # Read evidence before _shot invalidates the same-frame OCR cache.
            observed_button, reason = self.buttons.cached_go_win_evidence(old_frame)
            if observed_button is not None:
                from arena_auto.recognition.prepare_guard import PrepareStructure

                started = time.perf_counter()
                prepare_structure = PrepareStructure.discover(
                    old_frame, self.frame_cache.peek(old_frame), observed_button.bbox
                )
                self._log("INFO", f"备战稳定结构定位耗时 {(time.perf_counter() - started) * 1000.0:.0f}ms；"
                          f"定位成功={prepare_structure is not None}")
        image = self._shot()
        from arena_auto.recognition.frame_guard import compare_regions, save_guard_evidence

        if self.stop_event.is_set():
            raise StoppedError("用户停止")
        if observed_button is not None and time.monotonic() - self.ctx.prepare_frame_at > 5.0:
            observed_button = None
            reason = "备战证据在新截图获取后过期"
        report = None
        block_click = False
        if observed_button is not None and observed_button.found:
            started = time.perf_counter()
            if prepare_structure is not None:
                report, block_click = prepare_structure.validate(old_frame, image)
                self._log("INFO", f"备战结构复用校验：面板={prepare_structure.panel}；对手摘要={prepare_structure.summary}")
            else:
                report = compare_regions(old_frame, image,
                                         [(0, 0, image.shape[1], image.shape[0]), observed_button.bbox])
                self._log("INFO", "备战稳定结构未定位：保留整页严格校验")
            self._log("INFO", f"备战画面校验耗时 {(time.perf_counter() - started) * 1000.0:.0f}ms")
            reason = report.describe()
        if report is not None and report.unchanged:
            result = observed_button
            self._log("INFO", "备战按钮复用：新截图验证稳定结构，跳过重复 OCR" if prepare_structure is not None
                      else "备战按钮复用：新截图验证页面未变化，跳过重复 OCR")
        elif block_click:
            self._log("WARNING", f"备战点击阻止：{reason}；停止自动化，需处理遮挡后重新开始")
            save_guard_evidence(self.debug, old_frame, image, report, "prepare_blocked")
            self.sm.transition(AutomationState.STOPPED)
            return
        else:
            self._log("INFO", f"备战按钮复用未命中：{reason}；回退新图检测")
            if report is not None:
                save_guard_evidence(self.debug, old_frame, image, report, "prepare_rejected")
            result = self.buttons.find_go_win(image)
        if result.found:
            self._log("INFO", "检测到「去获胜」")
            self.ctx.auto_selection_attempts = 0
            self.ctx.result_ocr_last_at = 0.0
            self.ctx.result_scene_change = SettledSceneChange()
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
        """Template first; auto mode has a rate-limited OCR safety fallback."""
        if self.stop_event.is_set():
            raise StoppedError("用户停止")
        battle_cfg = self.config.battle
        if self.ctx.battle_started_at <= 0:
            self.ctx.battle_started_at = time.monotonic()
        elapsed = time.monotonic() - self.ctx.battle_started_at
        if elapsed > battle_cfg.max_duration:
            self._log("ERROR", "战斗超时")
            self.sm.transition(AutomationState.STOPPED if self.config.automation.auto_targeting else AutomationState.RECOVERY)
            return

        interval = max(0.5, battle_cfg.poll_interval)
        self._log("INFO", "等待战斗结束（正在检查结算画面）")
        if self._wait(interval):
            raise StoppedError("用户停止")
        if self.stop_event.is_set():
            raise StoppedError("用户停止")

        image = self._shot()
        if self.stop_event.is_set():
            raise StoppedError("用户停止")

        # --- Template-only (priority: Victory, Defeat, then exit / go_win) ---
        scene_changed = (self.ctx.result_scene_change.update(image)
                         if self.config.automation.auto_targeting and battle_cfg.result_scene_probe else False)
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

        # Auto mode must not require pre-calibrated result templates. Manual
        # mode keeps the prior explicit debug opt-in and abnormal-state gate.
        debug_cfg = self.config.debug
        automatic = self.config.automation.auto_targeting
        if not automatic and not getattr(debug_cfg, "allow_ocr_fallback", False):
            return
        min_fails = max(1, int(getattr(debug_cfg, "ocr_fallback_min_failures", 3)))
        if self.ctx.wait_result_template_fails < min_fails and not scene_changed:
            return
        # "State looks abnormal": battle well past typical duration, or long miss streak.
        abnormal = (
            elapsed > min(30.0, battle_cfg.max_duration * 0.25)
            or self.ctx.wait_result_template_fails >= min_fails * 2
        )
        if not automatic and not abnormal:
            return

        probe = self._probe_result_ocr(image, scene_changed=scene_changed)
        if probe is None:
            return
        kind, ocr_outcome = probe
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

    def _probe_result_ocr(self, image: np.ndarray, *, scene_changed: bool = False) -> Optional[Tuple[ScreenKind, Optional[str]]]:
        """At most one OCR probe per cooldown; reuses boxes for live templates."""
        now = time.monotonic()
        # A one-shot settled transition can shorten a long cooldown, but never
        # starts probes less than five seconds apart or guesses a result.
        cooldown = self.config.battle.result_ocr_interval
        if scene_changed and self.config.automation.auto_targeting:
            cooldown = 5.0
        if self.ctx.result_ocr_last_at > 0 and now - self.ctx.result_ocr_last_at < cooldown:
            return None
        if self.stop_event.is_set():
            raise StoppedError("用户停止")
        self._log("WARNING", "结算模板未命中，执行限频 OCR 兜底（识别失败不会猜测胜负）")
        kind, outcome = self.detector.detect_screen_and_outcome(image, allow_ocr=True)
        self.ctx.result_ocr_last_at = time.monotonic()  # Cooldown starts after slow OCR finishes.
        if self.stop_event.is_set():
            raise StoppedError("用户停止")
        if self.config.automation.auto_targeting and kind == ScreenKind.RESULT:
            evidence = self.detector.result_text_evidence(image)
            if isinstance(evidence, dict) and "exit" in evidence:
                result_names = [name for name in ("victory", "defeat") if name in evidence]
                # Learn only when both the distinct outcome and exit are observed.
                expected = "victory" if outcome == "WIN" else "defeat" if outcome == "LOSE" else ""
                if result_names == [expected]:
                    for name in (expected, "exit"):
                        box = evidence[name]
                        if not isinstance(box, OCRBox):
                            continue
                        try:
                            crop = crop_region(image, box.bbox)
                            if self.templates.register_image(name, crop):
                                self._log("INFO", f"已缓存当前结算 {name} 模板（仅内存，不改文件或坐标）")
                        except ValueError:
                            pass
        self.debug.save(image, category="battle", tag="result_ocr_probe")
        return kind, outcome

    def _state_result(self) -> None:
        """Known RESULT: template first; auto mode can confirm an unknown outcome."""
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

        if outcome is None and self.config.automation.auto_targeting:
            probe = self._probe_result_ocr(image)
            if probe is not None:
                kind, detected_outcome = probe
                if kind == ScreenKind.RESULT:
                    outcome = detected_outcome
                    exit_visible = exit_visible or self.templates.find(image, "exit").found
                    evidence = self.detector.result_text_evidence(image)
                    if isinstance(evidence, dict) and isinstance(evidence.get("exit"), OCRBox):
                        exit_visible = True

        if outcome is None and not exit_visible:
            # Wait for victory/defeat/exit templates — never detector.detect() / full OCR.
            if self.sm.has_timed_out(self.config.timeouts.result):
                self._log(
                    "ERROR",
                    "结算识别超时：没有可靠的胜负或退出证据",
                )
                self.statistics.recognition_failures += 1
                self._emit_statistics()
                if getattr(self.config.debug, "allow_ocr_fallback", False):
                    outcome = self.detector.detect_outcome(image)
                    exit_visible = self.buttons.find_exit(image, full_ocr=False).found
                if outcome is None and not exit_visible:
                    self.sm.transition(AutomationState.STOPPED if self.config.automation.auto_targeting else AutomationState.RECOVERY)
                    return
            else:
                if self._wait(1.0):
                    raise StoppedError("用户停止")
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
        exit_result = self.buttons.find_exit(
            image, full_ocr=self.config.automation.auto_targeting or self.config.debug.allow_ocr_fallback
        )
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
                count = self._read_challenge(frame)
                if count is not None:
                    self._log("INFO", "返回竞技场")
                    self.ctx.challenge = count
                    outcome = self.ctx.outcome or "?"
                    self._log("INFO", f"本轮完成：{outcome}")
                    self.ctx.outcome = None
                    self._mark_progress("已确认退出结算并返回竞技场")
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
                        self._mark_progress("已确认退出结算并返回竞技场")
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
        challenge = self._read_challenge(image)
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
        if self.config.automation.auto_targeting and (
            not self.config.purchase.enabled
            or self.statistics.session_purchase_count >= self.config.purchase.max_per_session
        ):
            self._log("ERROR", "购买未授权或已达上限，自动选点模式安全停止")
            self.sm.transition(AutomationState.STOPPED)
            return
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
            self.ctx.auto_selection_attempts = 0
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
        count = self._read_challenge(image)
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
        self._mark_progress("购买后已确认挑战次数恢复")
        self._save_auto_targeting_config()
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
            count = self._read_challenge(frame)
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

    def _mark_progress(self, reason: str) -> None:
        """Only verified round completion/restored attempts reset failure budget."""
        if self.ctx.recovery_attempts:
            self._log("INFO", f"{reason}，重置连续恢复计数")
        self.ctx.recovery_attempts = 0

    def _state_recovery(self) -> None:
        if self.stop_event.is_set():
            raise StoppedError("用户停止")
        max_attempts = max(1, self.config.recovery.max_attempts)
        if self.ctx.recovery_attempts >= max_attempts:
            self._log("ERROR", f"连续恢复已达 {max_attempts} 次且未确认有效进展，安全停止")
            self.sm.transition(AutomationState.ERROR)
            return
        self.ctx.recovery_attempts += 1
        self.statistics.recoveries += 1
        self._emit_statistics()
        self._log(
            "WARNING",
            f"进入恢复流程 ({self.ctx.recovery_attempts}/{max_attempts})",
        )
        try:
            image = self._shot()
        except AdbError as exc:
            raise AdbDisconnectedError(str(exc)) from exc

        self.debug.save(image, category="recovery")
        kind = self.detector.detect(image)
        if self.stop_event.is_set():
            raise StoppedError("用户停止")

        if kind == ScreenKind.ARENA:
            self.ctx.battle_started_at = 0.0
            self.ctx.pending_purchase = False
            self.sm.transition(AutomationState.ARENA)
            return
        if kind == ScreenKind.PREPARE:
            self.sm.transition(AutomationState.PREPARE)
            return
        if kind == ScreenKind.RESULT:
            self.sm.transition(AutomationState.RESULT)
            return
        if kind == ScreenKind.PURCHASE:
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
        if self.stop_event.is_set():
            self.sm.transition(AutomationState.STOPPED)
            return
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
