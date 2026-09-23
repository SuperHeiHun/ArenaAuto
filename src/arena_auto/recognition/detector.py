"""Page detection, power OCR, challenge count OCR, and button finding.

Performance notes (runtime path):
- ScreenDetector: template-first for result/prepare/purchase; one shared OCR
  pass per screenshot when keywords are required; detect+outcome reuse boxes.
- PowerReader: one screenshot fans out to all 5 ROIs; digit allowlist + cache;
  force re-read only for stability confirmation; never returns power=0.
- ChallengeCountReader: ROI + digit/slash allowlist; full-frame OCR only as
  fallback and without ×3 upscale of the whole frame.
- ButtonFinder: template matching first, then region OCR, then full-frame OCR.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from arena_auto.debug.recorder import DebugRecorder
from arena_auto.exceptions import OCRFailedError, RecognitionError, TemplateMissingError
from arena_auto.models.config import AppConfig, OpponentConfig, RecognitionSettings
from arena_auto.models.recognition import (
    ChallengeCount,
    DetectionResult,
    OCRBox,
    Opponent,
    Region,
)
from arena_auto.models.state import ScreenKind
from arena_auto.recognition.ocr import (
    OCRProvider,
    find_text_boxes,
    parse_challenge_count,
    parse_power,
    preprocess_for_ocr,
)
from arena_auto.recognition.ocr_engine import (
    CHALLENGE_ALLOWLIST,
    POWER_ALLOWLIST,
    OcrEngine,
    perf_log,
)
from arena_auto.recognition.screen import CoordinateScaler, crop_region
from arena_auto.recognition.template import TemplateRecognizer

logger = logging.getLogger(__name__)


def _join_texts(boxes: Sequence[OCRBox]) -> str:
    return " ".join(box.text for box in boxes)


def _as_engine(ocr: Optional[OCRProvider], config: Optional[AppConfig] = None) -> Optional[OcrEngine]:
    if ocr is None:
        return None
    if isinstance(ocr, OcrEngine):
        return ocr
    settings = config.ocr if config is not None else None
    try:
        return OcrEngine(provider=ocr, settings=settings, lazy=False, use_shared=False)
    except Exception:  # noqa: BLE001
        return None


class ScreenDetector:
    """Multi-signal page classifier (not a single keyword)."""

    def __init__(
        self,
        ocr: OCRProvider,
        config: AppConfig,
        templates: Optional[TemplateRecognizer] = None,
        scaler: Optional[CoordinateScaler] = None,
    ) -> None:
        self.ocr = ocr
        self.config = config
        self.templates = templates
        self.scaler = scaler or CoordinateScaler(
            config.screen.reference_width, config.screen.reference_height
        )
        self.engine = _as_engine(ocr, config)
        # Same-frame OCR reuse: detect() + detect_outcome() on one screenshot.
        self._frame_boxes_ref: Optional[np.ndarray] = None
        self._frame_boxes: List[OCRBox] = []
        self._last_template_kind: Optional[ScreenKind] = None

    def invalidate_frame_cache(self) -> None:
        self._frame_boxes_ref = None
        self._frame_boxes = []

    def _recognize_all(self, screenshot: np.ndarray) -> List[OCRBox]:
        if screenshot is None:
            return []
        # Reuse OCR when the exact same frame object is classified twice.
        if self._frame_boxes_ref is screenshot:
            return self._frame_boxes
        started = time.perf_counter()
        try:
            if self.engine is not None:
                boxes = self.engine.recognize(screenshot)
            else:
                boxes = self.ocr.recognize(screenshot)
        except Exception as exc:  # noqa: BLE001
            logger.warning("页面 OCR 失败: %s", exc)
            boxes = []
        elapsed = (time.perf_counter() - started) * 1000.0
        perf_log("Page OCR", elapsed)
        self._frame_boxes_ref = screenshot
        self._frame_boxes = list(boxes)
        return self._frame_boxes

    @staticmethod
    def _count_hits(texts: str, keywords: Sequence[str]) -> int:
        return sum(1 for kw in keywords if kw and kw in texts)

    def _template_kind(self, screenshot: np.ndarray) -> Optional[ScreenKind]:
        """Cheap OpenCV template pass before any full-screen OCR."""
        if self.templates is None:
            return None
        try:
            if self.templates.find(screenshot, "buy_challenge").found:
                return ScreenKind.PURCHASE
            if self.templates.find(screenshot, "victory").found or self.templates.find(
                screenshot, "defeat"
            ).found:
                return ScreenKind.RESULT
            if self.templates.find(screenshot, "go_win").found:
                return ScreenKind.PREPARE
            if self.templates.find(screenshot, "exit").found:
                return ScreenKind.RESULT
        except Exception as exc:  # noqa: BLE001
            logger.debug("页面模板检测异常: %s", exc)
        return None

    def _classify_from_boxes(
        self, screenshot: np.ndarray, boxes: List[OCRBox]
    ) -> ScreenKind:
        text_blob = _join_texts(boxes)
        profile = self.config.profile
        min_hits = max(1, profile.min_keyword_matches)

        scores: Dict[ScreenKind, int] = {
            ScreenKind.ARENA: self._count_hits(text_blob, profile.arena_title_keywords)
            + self._count_hits(text_blob, profile.arena_select_keywords)
            + self._count_hits(text_blob, profile.challenge_keywords),
            ScreenKind.PREPARE: self._count_hits(text_blob, [profile.go_win_text]),
            ScreenKind.PURCHASE: self._count_hits(
                text_blob, [profile.buy_text, profile.confirm_text]
            ),
            ScreenKind.RESULT: self._count_hits(
                text_blob,
                [profile.victory_text, profile.defeat_text, profile.exit_text],
            ),
        }

        result_hit = scores[ScreenKind.RESULT]
        title_hits = self._count_hits(text_blob, profile.arena_title_keywords)
        go_win_hit = scores[ScreenKind.PREPARE]
        buy_hit = self._count_hits(text_blob, [profile.buy_text])
        if buy_hit >= 1:
            kind = ScreenKind.PURCHASE
        elif title_hits >= 1 and scores[ScreenKind.ARENA] >= 2:
            kind = ScreenKind.ARENA
        elif result_hit >= 2:
            kind = ScreenKind.RESULT
        elif go_win_hit >= 1 and scores[ScreenKind.ARENA] < 2:
            kind = ScreenKind.PREPARE
        elif scores[ScreenKind.ARENA] >= min_hits:
            kind = ScreenKind.ARENA
        elif go_win_hit >= 1:
            kind = ScreenKind.PREPARE
        elif scores[ScreenKind.PURCHASE] >= 1:
            kind = ScreenKind.PURCHASE
        elif result_hit >= 1 and profile.exit_text in text_blob:
            kind = ScreenKind.RESULT
        else:
            kind = ScreenKind.UNKNOWN

        if kind == ScreenKind.UNKNOWN and self.templates is not None:
            if self.templates.find(screenshot, "go_win").found:
                kind = ScreenKind.PREPARE
            elif self.templates.find(screenshot, "victory").found or self.templates.find(
                screenshot, "defeat"
            ).found:
                kind = ScreenKind.RESULT
            elif self.templates.find(screenshot, "buy_challenge").found:
                kind = ScreenKind.PURCHASE

        logger.debug("页面识别结果: %s (scores=%s)", kind.value, scores)
        return kind

    def detect(self, screenshot: np.ndarray) -> ScreenKind:
        """Classify current screen into ScreenKind (template first, then OCR)."""
        if screenshot is None:
            return ScreenKind.UNKNOWN

        started = time.perf_counter()
        # 1) Template matching for unambiguous pages (no OCR).
        tpl_kind = self._template_kind(screenshot)
        if tpl_kind is not None:
            elapsed = (time.perf_counter() - started) * 1000.0
            perf_log(f"Screen detect (template {tpl_kind.value})", elapsed)
            logger.debug("页面识别结果: %s (template)", tpl_kind.value)
            return tpl_kind

        # 2) Full OCR only when templates cannot decide (arena / battle / chrome).
        boxes = self._recognize_all(screenshot)
        kind = self._classify_from_boxes(screenshot, boxes)
        elapsed = (time.perf_counter() - started) * 1000.0
        perf_log(f"Screen detect ({kind.value})", elapsed)
        return kind

    def detect_arena_screen(self, screenshot: np.ndarray) -> bool:
        return self.detect(screenshot) == ScreenKind.ARENA

    def detect_prepare_screen(self, screenshot: np.ndarray) -> bool:
        return self.detect(screenshot) == ScreenKind.PREPARE

    def detect_battle_screen(self, screenshot: np.ndarray) -> bool:
        # battle = not any known chrome
        return self.detect(screenshot) == ScreenKind.UNKNOWN

    def detect_result_screen(self, screenshot: np.ndarray) -> bool:
        return self.detect(screenshot) == ScreenKind.RESULT

    def detect_purchase_dialog(self, screenshot: np.ndarray) -> bool:
        return self.detect(screenshot) == ScreenKind.PURCHASE

    def _outcome_from_templates(self, screenshot: np.ndarray) -> Optional[str]:
        if self.templates is None:
            return None
        victory_tpl = self.templates.find(screenshot, "victory").found
        defeat_tpl = self.templates.find(screenshot, "defeat").found
        if victory_tpl and not defeat_tpl:
            return "WIN"
        if defeat_tpl and not victory_tpl:
            return "LOSE"
        if victory_tpl and defeat_tpl:
            return None
        return None

    def outcome_from_templates(self, screenshot: np.ndarray) -> Optional[str]:
        """Public template-only outcome (no OCR). WIN | LOSE | None."""
        return self._outcome_from_templates(screenshot)

    def has_result_template(self, screenshot: np.ndarray) -> bool:
        """True when victory/defeat/exit template hits (OpenCV only, no OCR)."""
        if self.templates is None or screenshot is None:
            return False
        try:
            if self.templates.find(screenshot, "victory").found:
                return True
            if self.templates.find(screenshot, "defeat").found:
                return True
            if self.templates.find(screenshot, "exit").found:
                return True
        except Exception:  # noqa: BLE001
            return False
        return False

    def detect_outcome(
        self, screenshot: np.ndarray, boxes: Optional[List[OCRBox]] = None
    ) -> Optional[str]:
        """Return 'WIN', 'LOSE', or None on result screen.

        Template victory/defeat win without OCR. Pass `boxes` to reuse a
        prior full-frame OCR on the same screenshot. `allow_ocr=False` keeps
        template-only (normal RESULT path).
        """
        if screenshot is None:
            return None
        started = time.perf_counter()

        tpl = self._outcome_from_templates(screenshot)
        if tpl is not None:
            elapsed = (time.perf_counter() - started) * 1000.0
            perf_log(f"Outcome (template {tpl})", elapsed)
            return tpl

        if boxes is None:
            boxes = self._recognize_all(screenshot)
        text_blob = _join_texts(boxes)
        profile = self.config.profile

        victory_tpl = False
        defeat_tpl = False
        if self.templates is not None:
            victory_tpl = self.templates.find(screenshot, "victory").found
            defeat_tpl = self.templates.find(screenshot, "defeat").found

        has_victory = victory_tpl or (
            profile.victory_text and profile.victory_text in text_blob
        )
        has_defeat = defeat_tpl or (
            profile.defeat_text and profile.defeat_text in text_blob
        )

        outcome: Optional[str] = None
        if has_victory and not has_defeat:
            outcome = "WIN"
        elif has_defeat and not has_victory:
            outcome = "LOSE"
        elif has_victory and has_defeat:
            if victory_tpl and not defeat_tpl:
                outcome = "WIN"
            elif defeat_tpl and not victory_tpl:
                outcome = "LOSE"
            else:
                outcome = None
        elapsed = (time.perf_counter() - started) * 1000.0
        perf_log(f"Outcome ({outcome})", elapsed)
        return outcome

    def detect_screen_and_outcome(
        self, screenshot: np.ndarray, *, allow_ocr: bool = True
    ) -> Tuple[ScreenKind, Optional[str]]:
        """One screenshot → page kind + battle outcome.

        allow_ocr=False: OpenCV templates only (WAIT_RESULT normal path).
        allow_ocr=True: may fall through to full-screen OCR when templates cannot decide.
        """
        if screenshot is None:
            return ScreenKind.UNKNOWN, None
        started = time.perf_counter()

        tpl_outcome = self._outcome_from_templates(screenshot)
        tpl_kind = self._template_kind(screenshot)
        if tpl_outcome is not None:
            elapsed = (time.perf_counter() - started) * 1000.0
            perf_log("detect+outcome (template result)", elapsed)
            return ScreenKind.RESULT, tpl_outcome
        if tpl_kind is not None and tpl_kind != ScreenKind.RESULT:
            # Non-result chrome decided by template alone (purchase/prepare).
            elapsed = (time.perf_counter() - started) * 1000.0
            perf_log(f"detect+outcome (template {tpl_kind.value})", elapsed)
            return tpl_kind, None
        if tpl_kind == ScreenKind.RESULT:
            elapsed = (time.perf_counter() - started) * 1000.0
            perf_log("detect+outcome (template result, no outcome)", elapsed)
            return ScreenKind.RESULT, None

        if not allow_ocr:
            elapsed = (time.perf_counter() - started) * 1000.0
            perf_log("detect+outcome (template-only miss)", elapsed)
            return ScreenKind.UNKNOWN, None

        boxes = self._recognize_all(screenshot)
        kind = self._classify_from_boxes(screenshot, boxes)
        outcome = self.detect_outcome(screenshot, boxes=boxes)
        if kind != ScreenKind.RESULT and outcome is not None:
            kind = ScreenKind.RESULT
        elapsed = (time.perf_counter() - started) * 1000.0
        perf_log(f"detect+outcome ({kind.value}/{outcome})", elapsed)
        return kind, outcome


def is_placeholder_power_region(region: Region) -> bool:
    """Default un-calibrated ROI — must not be OCR'd (wastes time, fails)."""
    return tuple(region) == (0, 0, 100, 50)


class PowerReader:
    """Stable-read power OCR over configured opponent ROIs.

    Runtime strategy: short-circuit read_until_at_or_below stops at the first
    power <= threshold (normal path). read_all fans out remaining slots only
    when a full list is required. power=None means OCR_FAILED (never 0).
    """

    def __init__(
        self,
        ocr: OCRProvider,
        config: AppConfig,
        recognition: Optional[RecognitionSettings] = None,
        scaler: Optional[CoordinateScaler] = None,
        debug: Optional[DebugRecorder] = None,
    ) -> None:
        self.ocr = ocr
        self.config = config
        self.recognition = recognition or config.recognition
        self.scaler = scaler or CoordinateScaler(
            config.screen.reference_width, config.screen.reference_height
        )
        self.debug = debug
        self.engine = _as_engine(ocr, config)
        self._allowlist = str(
            getattr(config.ocr, "power_allowlist", POWER_ALLOWLIST) or POWER_ALLOWLIST
        )

    def _read_once(
        self, screenshot: np.ndarray, region: Region, slot: int, *, force: bool = False
    ) -> Optional[int]:
        value, _raw = self._read_once_debug(
            screenshot, region, slot, force=force, allow_raw_fallback=True
        )
        return value

    def _read_once_debug(
        self,
        screenshot: np.ndarray,
        region: Region,
        slot: int,
        *,
        force: bool = False,
        allow_raw_fallback: bool = False,
    ) -> tuple[Optional[int], str]:
        """Crop one power ROI and OCR with digit allowlist + optional cache."""
        self.scaler.maybe_update_from_image(screenshot)
        scaled = self.scaler.scale_region(region)
        cache_key = None if force else f"power:{slot}:{scaled}"

        if self.engine is not None:
            try:
                boxes, raw_text = self.engine.read_roi(
                    screenshot,
                    scaled,
                    allowlist=self._allowlist,
                    preprocess=True,
                    cache_key=cache_key,
                    force=force,
                )
            except ValueError as exc:
                logger.warning("战力 ROI 无效 #%s: %s", slot, exc)
                return None, f"ROI_INVALID:{exc}"
            except Exception as exc:  # noqa: BLE001
                logger.warning("战力 OCR 异常 #%s: %s", slot, exc)
                return None, f"OCR_EXCEPTION:{exc}"

            value: Optional[int] = None
            for text in [b.text for b in boxes] + [raw_text]:
                if not text or not str(text).strip():
                    continue
                value = parse_power(str(text))
                if value is not None:
                    break
            logger.debug("#%s OCR raw=%r parsed=%s", slot, raw_text, value)
            if value is None and allow_raw_fallback:
                try:
                    boxes2, raw2 = self.engine.read_roi(
                        screenshot,
                        scaled,
                        allowlist=None,
                        preprocess=True,
                        cache_key=None,
                        force=True,
                    )
                except Exception:  # noqa: BLE001
                    boxes2, raw2 = [], ""
                for text in [b.text for b in boxes2] + [raw2]:
                    if not text or not str(text).strip():
                        continue
                    value = parse_power(str(text))
                    if value is not None:
                        raw_text = raw2 or raw_text
                        break
            if value is None and (raw_text or "").strip():
                logger.warning(
                    "#%s 战力解析失败 raw=%r roi=%s", slot, raw_text, region
                )
            return value, raw_text

        # Fallback path without OcrEngine (injected providers / tests).
        try:
            roi = crop_region(screenshot, scaled)
        except ValueError as exc:
            logger.warning("战力 ROI 无效 #%s: %s", slot, exc)
            return None, f"ROI_INVALID:{exc}"

        processed = preprocess_for_ocr(roi, self.config.ocr)
        try:
            boxes = self.ocr.recognize(processed)
        except Exception as exc:  # noqa: BLE001
            logger.warning("战力 OCR 异常 #%s: %s", slot, exc)
            return None, f"OCR_EXCEPTION:{exc}"

        raw_text = _join_texts(boxes)
        value = parse_power(raw_text)
        logger.debug("#%s OCR raw=%r parsed=%s", slot, raw_text, value)
        if value is None and raw_text.strip() and allow_raw_fallback:
            try:
                boxes2 = self.ocr.recognize(roi)
            except Exception:  # noqa: BLE001
                boxes2 = []
            raw2 = _join_texts(boxes2)
            for text in [b.text for b in boxes2] + [raw2]:
                value = parse_power(text)
                if value is not None:
                    raw_text = raw2 or raw_text
                    break
        if value is None and raw_text.strip():
            logger.warning("#%s 战力解析失败 raw=%r roi=%s", slot, raw_text, region)
        return value, raw_text

    def read_power(self, screenshot: np.ndarray, opponent: OpponentConfig) -> Optional[int]:
        """Read one opponent power with stable confirmation."""
        stable = max(1, self.recognition.stable_reads)
        retries = max(1, self.recognition.retry_count)

        last_value: Optional[int] = None
        for attempt in range(1, retries + 1):
            readings: List[Optional[int]] = []
            for read_index in range(stable):
                value = self._read_once(
                    screenshot,
                    opponent.power_region,
                    opponent.id,
                    force=read_index > 0,
                )
                readings.append(value)
                if self.debug is not None and read_index == 0:
                    try:
                        scaled = self.scaler.scale_region(opponent.power_region)
                        roi = crop_region(screenshot, scaled)
                        if value is None:
                            self.debug.save_roi(roi, f"power_{opponent.id}", category="failed")
                        else:
                            self.debug.save_roi(roi, f"power_{opponent.id}", category="debug")
                    except Exception:  # noqa: BLE001
                        pass

            if all(r is not None for r in readings) and len(set(readings)) == 1:
                return readings[0]

            last_value = next((r for r in readings if r is not None), None)
            logger.warning(
                "战力读取不稳定 #%s attempt=%s readings=%s",
                opponent.id,
                attempt,
                readings,
            )
            if stable == 1:
                if last_value is not None and attempt == 1 and retries == 1:
                    return last_value

        if last_value is not None and self.recognition.stable_reads == 1:
            return last_value
        return None

    def read_until_at_or_below(
        self,
        screenshot_factory,
        threshold: int,
        opponents: Optional[List[OpponentConfig]] = None,
        *,
        stop_event=None,
    ) -> Tuple[List[Opponent], Optional[Opponent]]:
        """Short-circuit: read slots 1..N in order; stop at first power <= threshold.

        - Reads only until a match (or OCR failure / placeholder / list end).
        - power=None raises OCRFailedError (never treated as 0, never skipped).
        - Placeholder ROI [0,0,100,50] fails immediately without OCR.
        - stop_event (threading.Event): abort before each new slot OCR.

        Returns (opponents_read_so_far, selected_or_None).
        """
        configs = opponents or self.config.opponents
        stable = max(1, self.recognition.stable_reads)
        results: List[Opponent] = []
        selected: Optional[Opponent] = None
        frame: Optional[np.ndarray] = None

        for cfg in configs:
            if stop_event is not None and stop_event.is_set():
                raise OCRFailedError("停止请求，中断战力识别")

            if is_placeholder_power_region(cfg.power_region):
                if results:
                    # Already read real slots; uncalibrated tail must not OCR_FAILED.
                    logger.warning(
                        "#%s power_region 未标定 [0,0,100,50]，短路停止（请自动标定）",
                        cfg.id,
                    )
                    break
                logger.error(
                    "#%s power_region 仍为默认占位 [0,0,100,50]，请先自动标定",
                    cfg.id,
                )
                raise OCRFailedError(
                    f"#{cfg.id} power_region 未标定 [0,0,100,50]，请先运行自动标定"
                )

            if frame is None:
                frame = screenshot_factory()
                if stop_event is not None and stop_event.is_set():
                    raise OCRFailedError("停止请求，中断战力识别")

            value, raw = self._read_once_debug(
                frame,
                cfg.power_region,
                cfg.id,
                force=False,
                allow_raw_fallback=True,
            )
            # Stability confirmation on the same frame (force bypasses cache).
            if value is not None and stable > 1 and self.engine is not None:
                if stop_event is not None and stop_event.is_set():
                    raise OCRFailedError("停止请求，中断战力识别")
                value2, _raw2 = self._read_once_debug(
                    frame,
                    cfg.power_region,
                    cfg.id,
                    force=True,
                    allow_raw_fallback=False,
                )
                if value2 != value:
                    value = None

            if self.debug is not None:
                try:
                    scaled = self.scaler.scale_region(cfg.power_region)
                    roi = crop_region(frame, scaled)
                    category = "debug" if value is not None else "failed"
                    self.debug.save_roi(roi, f"power_{cfg.id}", category=category)
                except Exception:  # noqa: BLE001
                    pass

            opp = Opponent(
                id=cfg.id,
                power=value,
                power_region=cfg.power_region,
                click_position=cfg.click,
            )
            opp.raw_ocr = raw  # type: ignore[attr-defined]
            results.append(opp)

            if value is None:
                logger.warning(
                    "#%s 战力 OCR_FAILED raw=%r roi=%s",
                    cfg.id,
                    raw,
                    list(cfg.power_region),
                )
                raise OCRFailedError(f"对手 #{cfg.id} 战力 OCR_FAILED，禁止继续选择")

            logger.info("#%s 战力：%s", cfg.id, value)
            if value <= threshold:
                selected = opp
                logger.info(
                    "短路命中 #%s (%s <= %s)，跳过后续对手",
                    cfg.id,
                    value,
                    threshold,
                )
                break

        return results, selected

    def read_all(
        self,
        screenshot_factory,
        opponents: Optional[List[OpponentConfig]] = None,
    ) -> List[Opponent]:
        """Read all 1..5 powers with minimal screenshots (full list).

        Prefer read_until_at_or_below for the normal selection path.
        """
        configs = opponents or self.config.opponents
        results: List[Opponent] = []
        placeholder_warned = False
        retries = max(1, self.recognition.retry_count)
        stable = max(1, self.recognition.stable_reads)
        allowlist = self._allowlist

        is_placeholder: Dict[int, bool] = {}
        for cfg in configs:
            placeholder = tuple(cfg.power_region) == (0, 0, 100, 50)
            is_placeholder[cfg.id] = placeholder
            if placeholder and not placeholder_warned:
                logger.warning(
                    "战力 power_region 仍为默认占位 [0,0,100,50]，"
                    "请在 config.yaml 中按截图填写真实 ROI"
                )
                placeholder_warned = True

        values: Dict[int, Optional[int]] = {cfg.id: None for cfg in configs}
        raws: Dict[int, str] = {cfg.id: "" for cfg in configs}
        pending = [cfg for cfg in configs]

        for attempt in range(retries):
            if not pending:
                break
            started = time.perf_counter()
            frame = screenshot_factory()
            if attempt == 0:
                # Round 0: every pending slot on this single frame.
                round_slots = list(pending)
            else:
                round_slots = list(pending)

            for cfg in round_slots:
                value, raw = self._read_once_debug(
                    frame,
                    cfg.power_region,
                    cfg.id,
                    force=attempt > 0,
                    allow_raw_fallback=True,
                )
                # Stability confirmation on the same frame (force bypasses cache).
                if (
                    value is not None
                    and stable > 1
                    and attempt == 0
                    and self.engine is not None
                ):
                    value2, _raw2 = self._read_once_debug(
                        frame,
                        cfg.power_region,
                        cfg.id,
                        force=True,
                        allow_raw_fallback=False,
                    )
                    if value2 != value:
                        values[cfg.id] = None
                        raws[cfg.id] = raw
                        continue
                values[cfg.id] = value
                raws[cfg.id] = raw

                if self.debug is not None:
                    try:
                        scaled = self.scaler.scale_region(cfg.power_region)
                        roi = crop_region(frame, scaled)
                        category = "debug" if value is not None else "failed"
                        self.debug.save_roi(roi, f"power_{cfg.id}", category=category)
                    except Exception:  # noqa: BLE001
                        pass

            pending = [cfg for cfg in configs if values[cfg.id] is None]
            elapsed = (time.perf_counter() - started) * 1000.0
            perf_log(
                f"Power ROI round attempt={attempt + 1} pending={len(pending)}",
                elapsed,
            )
            if pending and attempt + 1 < retries:
                # brief pause then re-capture only if something failed
                continue

        for cfg in configs:
            power = values.get(cfg.id)
            raw_text = raws.get(cfg.id, "")
            if power is None:
                logger.warning(
                    "#%s 战力 OCR_FAILED raw=%r roi=%s",
                    cfg.id,
                    raw_text,
                    list(cfg.power_region),
                )
            else:
                logger.info("#%s 战力：%s", cfg.id, power)
            results.append(
                Opponent(
                    id=cfg.id,
                    power=power,
                    power_region=cfg.power_region,
                    click_position=cfg.click,
                )
            )
            results[-1].raw_ocr = raw_text  # type: ignore[attr-defined]
        return results


class ChallengeCountReader:
    """OCR `可挑战次数 current / maximum` from calibrated ROI (not full screen)."""

    def __init__(
        self,
        ocr: OCRProvider,
        config: AppConfig,
        scaler: Optional[CoordinateScaler] = None,
        debug: Optional[DebugRecorder] = None,
    ) -> None:
        self.ocr = ocr
        self.config = config
        self.scaler = scaler or CoordinateScaler(
            config.screen.reference_width, config.screen.reference_height
        )
        self.debug = debug
        self.last_raw: str = ""
        self.engine = _as_engine(ocr, config)
        self._allowlist = str(
            getattr(config.ocr, "challenge_allowlist", CHALLENGE_ALLOWLIST)
            or CHALLENGE_ALLOWLIST
        )

    def read(self, screenshot: np.ndarray) -> Optional[ChallengeCount]:
        self.scaler.maybe_update_from_image(screenshot)
        region = self.config.challenge_region
        try:
            scaled = self.scaler.scale_region(region)
        except ValueError as exc:
            logger.warning("挑战次数 ROI 无效: %s", exc)
            return None

        if self.debug is not None:
            try:
                roi = crop_region(screenshot, scaled)
                self.debug.save_roi(roi, "challenge_count", category="debug")
            except Exception:  # noqa: BLE001
                pass

        started = time.perf_counter()
        boxes: List[OCRBox] = []
        raw_text = ""
        if self.engine is not None:
            try:
                boxes, raw_text = self.engine.read_roi(
                    screenshot,
                    scaled,
                    allowlist=self._allowlist,
                    preprocess=True,
                    cache_key=f"challenge:{scaled}",
                    force=False,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("挑战次数 OCR 异常: %s", exc)
                self.last_raw = f"OCR_EXCEPTION:{exc}"
                return None
        else:
            try:
                roi = crop_region(screenshot, scaled)
                processed = preprocess_for_ocr(roi, self.config.ocr)
                boxes = self.ocr.recognize(processed)
                raw_text = _join_texts(boxes)
            except Exception as exc:  # noqa: BLE001
                logger.warning("挑战次数 OCR 异常: %s", exc)
                self.last_raw = f"OCR_EXCEPTION:{exc}"
                return None

        candidates = [b.text for b in boxes if "/" in b.text]
        if not candidates and raw_text.strip():
            candidates = [raw_text]
        if not candidates:
            candidates = [_join_texts(boxes)]
        for text in candidates:
            self.last_raw = text
            parsed = parse_challenge_count(text)
            if parsed is not None:
                count = ChallengeCount(current=parsed[0], maximum=parsed[1])
                elapsed = (time.perf_counter() - started) * 1000.0
                perf_log("Challenge OCR (roi)", elapsed)
                logger.info("挑战次数：%s", count)
                return count

        # Fallback: full-frame OCR WITHOUT ×3 upscale (huge win on 1080p).
        try:
            if self.engine is not None:
                full_boxes = self.engine.recognize(screenshot)
            else:
                full_boxes = self.ocr.recognize(screenshot)
        except Exception:  # noqa: BLE001
            full_boxes = []
        for box in full_boxes:
            self.last_raw = box.text
            parsed = parse_challenge_count(box.text)
            if parsed is not None:
                count = ChallengeCount(current=parsed[0], maximum=parsed[1])
                elapsed = (time.perf_counter() - started) * 1000.0
                perf_log("Challenge OCR (full fallback)", elapsed)
                logger.info("挑战次数(全图)：%s", count)
                return count

        elapsed = (time.perf_counter() - started) * 1000.0
        perf_log("Challenge OCR (failed)", elapsed)
        logger.warning("挑战次数 OCR_FAILED roi_raw=%r", self.last_raw)
        if self.debug is not None:
            try:
                roi = crop_region(screenshot, scaled)
                self.debug.save_failed(screenshot, name="challenge_count", roi=roi)
            except Exception:  # noqa: BLE001
                pass
        return None


class ButtonFinder:
    """Template matching first, calibrated-region OCR, then full-frame OCR."""

    def __init__(
        self,
        templates: TemplateRecognizer,
        ocr: OCRProvider,
        config: AppConfig,
        debug: Optional[DebugRecorder] = None,
    ) -> None:
        self.templates = templates
        self.ocr = ocr
        self.config = config
        self.debug = debug
        self.engine = _as_engine(ocr, config)

    def _region_hint(self, name: str) -> Optional[Region]:
        geo = self.config.button_geometry(name)
        if geo.region[2] > 0 and geo.region[3] > 0:
            return geo.region
        return None

    def _ocr_keyword(self, image: np.ndarray, keyword: str) -> List[OCRBox]:
        if self.engine is not None and self.engine is not self.ocr:
            try:
                return self.engine.recognize(image)
            except Exception:  # noqa: BLE001
                return find_text_boxes(self.ocr, image, keyword)
        return find_text_boxes(self.ocr, image, keyword)

    def find(
        self,
        screenshot: np.ndarray,
        template_name: str,
        fallback_text: str = "",
        *,
        full_ocr: bool = True,
    ) -> DetectionResult:
        entry = self.config.templates.get(template_name)
        threshold = entry.threshold if entry else 0.82

        started = time.perf_counter()
        result = self.templates.find(screenshot, template_name, threshold=threshold)
        if result.found:
            elapsed = (time.perf_counter() - started) * 1000.0
            perf_log(f"Button {template_name} (template)", elapsed)
            logger.debug(
                "模板命中 %s conf=%.3f center=%s",
                template_name,
                result.confidence,
                result.center,
            )
            return result

        if fallback_text:
            hint = self._region_hint(template_name)
            if hint is not None:
                try:
                    scaler = CoordinateScaler(
                        self.config.screen.reference_width,
                        self.config.screen.reference_height,
                    )
                    scaler.maybe_update_from_image(screenshot)
                    scaled = scaler.scale_region(hint)
                    crop = crop_region(screenshot, scaled)
                    ocr_started = time.perf_counter()
                    region_boxes = find_text_boxes(self.ocr, crop, fallback_text)
                    if not region_boxes and self.engine is not None:
                        boxes_e, _ = self.engine.read_roi(
                            crop,
                            (0, 0, crop.shape[1], crop.shape[0]),
                            allowlist=None,
                            preprocess=False,
                            cache_key=f"btn:{template_name}:region",
                            force=False,
                        )
                        keyword = fallback_text.strip()
                        region_boxes = [
                            b
                            for b in boxes_e
                            if b.confidence >= 0.3
                            and (keyword in b.text or b.text in keyword)
                        ]
                    perf_log(
                        f"Button {template_name} (region ocr)",
                        (time.perf_counter() - ocr_started) * 1000.0,
                    )
                    if region_boxes:
                        box = max(region_boxes, key=lambda b: b.confidence)
                        ox, oy = scaled[0], scaled[1]
                        bx, by, bw, bh = box.bbox
                        logger.debug(
                            "区域 OCR 命中 %s -> %s", template_name, box.text
                        )
                        return DetectionResult(
                            found=True,
                            confidence=box.confidence,
                            bbox=(ox + bx, oy + by, bw, bh),
                            template_name=f"{template_name}#region",
                        )
                except ValueError as exc:
                    logger.debug("按钮区域提示无效 %s: %s", template_name, exc)

            if not full_ocr:
                if self.debug is not None:
                    self.debug.save_failed(screenshot, name=f"miss_{template_name}")
                elapsed = (time.perf_counter() - started) * 1000.0
                perf_log(f"Button {template_name} (template+region miss)", elapsed)
                logger.debug("按钮未找到: %s", template_name)
                return DetectionResult(
                    found=False, confidence=0.0, template_name=template_name
                )

            full_started = time.perf_counter()
            boxes = find_text_boxes(self.ocr, screenshot, fallback_text)
            perf_log(
                f"Button {template_name} (full ocr)",
                (time.perf_counter() - full_started) * 1000.0,
            )
            if boxes:
                keyword = fallback_text.strip()
                # 短关键字（确认/购买等）不要匹配更长句子（确认对手布置）
                if len(keyword) <= 2:
                    tight = [
                        b
                        for b in boxes
                        if b.text.strip() == keyword
                        or len(b.text.strip()) <= len(keyword) + 1
                    ]
                    boxes = tight
                if boxes:
                    exact = [b for b in boxes if b.text.strip() == keyword]
                    # 标题=完整关键字、按钮=更短子串：优先更短标签，再靠下(按钮在标题下)，再置信度
                    if exact and len(exact) == len(boxes):
                        box = max(exact, key=lambda b: b.confidence)
                    else:
                        box = max(
                            boxes,
                            key=lambda b: (
                                -len(b.text.strip()),
                                b.bbox[1],
                                b.confidence,
                            ),
                        )
                    logger.debug("OCR fallback 命中 %s -> %s", template_name, box.text)
                    return DetectionResult(
                        found=True,
                        confidence=box.confidence,
                        bbox=box.bbox,
                        template_name=f"{template_name}#ocr",
                    )

        if self.debug is not None:
            self.debug.save_failed(screenshot, name=f"miss_{template_name}")

        elapsed = (time.perf_counter() - started) * 1000.0
        perf_log(f"Button {template_name} (miss)", elapsed)
        logger.debug("按钮未找到: %s", template_name)
        return DetectionResult(found=False, confidence=0.0, template_name=template_name)

    def find_go_win(self, screenshot: np.ndarray) -> DetectionResult:
        return self.find(screenshot, "go_win", self.config.profile.go_win_text)

    def find_exit(self, screenshot: np.ndarray, *, full_ocr: bool = True) -> DetectionResult:
        """Find 退出. full_ocr=False: template + region OCR only (RESULT path)."""
        return self.find(
            screenshot, "exit", self.config.profile.exit_text, full_ocr=full_ocr
        )

    def find_buy(self, screenshot: np.ndarray) -> DetectionResult:
        return self.find(screenshot, "buy_challenge", self.config.profile.buy_text)

    def find_confirm(self, screenshot: np.ndarray) -> DetectionResult:
        return self.find(screenshot, "confirm_buy", self.config.profile.confirm_text)
